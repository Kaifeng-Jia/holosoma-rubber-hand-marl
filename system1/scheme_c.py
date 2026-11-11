"""Scheme C (MARL + frozen SONIC): wrap a holosoma paired env so a MAPPO policy drives SONIC.

Per T03 (2nd ed.): the policy outputs a [E, A, 29]; dq = paper-D action scale * a is added to
every frame of SONIC's reference window; SONIC (frozen, no grad) returns q*; the PD gets q*
verbatim. Everything else is D's code path:
* the env receives a_eff = (q* - default) / scale_D, so the action-rate penalty is computed on
  the physical target in D's action units, with D's reset semantics;
* the last-action observations (actor 158-D and critic) read the policy's own a via
  ``action_manager.policy_action`` (zeroed on reset like D's action buffer);
* SONIC history is reset per environment on episode reset and when the Plan5 command wraps
  the reference (robots are re-written to a new frame without an env reset).

Kinds (T03 C v2, 2026-11-11; ``kind`` defaults to "v1" = the behaviour above, unchanged):
* "in"  (C-in): a is clipped to [-1, 1] in the env, dq = scale_D * clip(a) (so |dq_i| <= scale_i); the joint
  reference SONIC receives (reference + dq on every window frame) is clamped to the robot's joint limits
  (robot_config dof_pos_lower/upper_limit_list, the URDF hard limits); the last-action observation is clip(a);
  the learner keeps the unclipped a (its log-probability is of the unclipped sample).
* "out" (C-out): SONIC receives the unmodified reference (as scheme A); q* = SONIC's output + scale_D * a_c with
  a_c = a clipped exactly as D's action term clips it (robot_config.control clip_actions / action_clip_value);
  the last-action observation is the policy's own a (as D); SONIC's own last-action input stays its own output.
"""
from __future__ import annotations

import os

import torch

from system1 import sonic_controller as sc
from system1.scheme_a import SonicDualDriver


KINDS = ("v1", "in", "out")
C_IN_PPO_OVERRIDES = dict(init_noise_std=0.3, entropy_coef=0.0)  # T03 C v2: C-in (former C2b); C-out keeps D's values


def zero_init_output_layer(actor):
    """Zero the last Linear layer (weights and bias) so a = 0 at iteration 0 (C == scheme A)."""
    last = [m for m in actor.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    return last


class SonicResidualEnv:
    # "step" stays on the wrapper: scripts patch env.step with observers (telemetry, raw-reward
    # diagnostics); those must wrap scheme C's step, which then calls the real env step.
    _OWN = ("env", "driver", "E", "A", "scale_d", "command", "_prev_t", "_dones", "sonic_resets", "step", "_diag",
            "kind", "ref_limits_h", "out_clip", "train_diag")

    def __setattr__(self, name, value):  # assignments like env.episode_length_buf = ... reach the real env
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __init__(self, env, device, tf32=False, infer_block=sc.INFER_BLOCK, kind="v1", train_diag=False):
        if kind not in KINDS:
            raise ValueError(f"scheme C kind {kind!r} not in {KINDS}")
        self.kind = kind
        self.env = env
        self.driver = SonicDualDriver(env, device, tf32=tf32)
        self.driver.ctl.infer_block = infer_block
        self.E, self.A = env.num_envs, env.num_agents
        self.scale_d = env.action_scales.reshape(-1)  # frozen paper-D scales (F2)
        self.command = env.command_manager.get_state("paired_motion_command")
        am = env.action_manager
        am.policy_action = torch.zeros_like(am.action)
        self._prev_t = None
        self._dones = torch.zeros(self.E, device=device)
        self.sonic_resets = 0
        rc = env.robot_config
        if list(rc.dof_names) != list(env.dof_names):
            raise AssertionError("robot_config and env DOF order differ")
        self.ref_limits_h = None
        if kind == "in":
            self.ref_limits_h = (torch.tensor(rc.dof_pos_lower_limit_list, dtype=torch.float32, device=device),
                                 torch.tensor(rc.dof_pos_upper_limit_list, dtype=torch.float32, device=device))
        self.out_clip = float(rc.control.action_clip_value) if (kind == "out" and rc.control.clip_actions) else None
        self._diag = None
        if os.environ.get("SYSTEM1_C_DIAG"):  # diagnostics switch (off: behaviour unchanged)
            from system1.c_diag import CDiag
            self._diag = CDiag(self, os.environ["SYSTEM1_C_DIAG"])
        self.train_diag = None
        if train_diag:  # T03 C v2 training-log diagnostics switch (off: behaviour unchanged)
            from system1.c_train_diag import CTrainDiag
            self.train_diag = CTrainDiag(self)

    def __getattr__(self, name):  # everything else is the wrapped env
        return getattr(self.env, name)

    def reset_all(self):
        obs = self.env.reset_all()
        self.driver.reset()
        self.env.action_manager.policy_action.zero_()
        self._prev_t = self.command.time_steps.clone()
        return obs

    def _sync_reference_jumps(self):
        t = self.command.time_steps
        if self._prev_t is not None:
            jumped = (t != self._prev_t + 1).nonzero(as_tuple=False).flatten()
            if jumped.numel():
                self.driver.reset(jumped)
                self.sonic_resets += int(jumped.numel())

    def step(self, actor_state):
        a = actor_state["actions"].reshape(self.E, self.A, -1)
        if self.kind == "v1":
            a_used = a
            dq = a * self.scale_d
            kw = dict(delta_q_h=dq)
        elif self.kind == "in":
            a_used = a.clamp(-1.0, 1.0)
            dq = a_used * self.scale_d
            kw = dict(delta_q_h=dq, ref_limits_h=self.ref_limits_h)
        else:  # "out"
            a_used = a if self.out_clip is None else a.clamp(-self.out_clip, self.out_clip)
            dq = a_used * self.scale_d  # correction on SONIC's output joint target
            kw = dict(target_residual_h=dq)
        if self._diag is not None:
            self._diag.pre(a, dq)
        if self.train_diag is not None:
            self.train_diag.step(a, a_used, dq)
        a_eff, _ = self.driver.actions(self.command, dones=self._dones, **kw)
        if self._diag is not None:
            self._diag.mid()
        if self.train_diag is not None:
            self.train_diag.mid(dq)
        self.env.action_manager.policy_action[:] = (a_used if self.kind == "in" else a).reshape(self.E, -1)
        self._prev_t = self.command.time_steps.clone()
        out = self.env.step({**actor_state, "actions": a_eff})
        dones = out[2].reshape(self.E)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            self.driver.reset(done_ids)
        self._sync_reference_jumps()  # reference frame jumped without an env reset (Plan5 wrap)
        self._prev_t = self.command.time_steps.clone()
        self._dones = dones.float()
        if self._diag is not None:
            self._diag.post(dones)
        if self.train_diag is not None:
            self.train_diag.post(dones, out[3])
        return out


def scheme_c_run_config(env, config, tf32: bool, kind: str = "v1", ppo_config=None, train_diag: bool = False):
    """Fields that differ from D because actions go through SONIC (recorded in run_config). kind="v1" returns exactly
    the v1 dict; C-in / C-out add their own fields (T03 C v2)."""
    import hashlib

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    term = env.action_manager.get_term("joint_control")
    out = dict(
        scheme="C",
        sonic_checkpoint=str(sc.DEFAULT_CHECKPOINT),
        sonic_checkpoint_sha256=sha(sc.DEFAULT_CHECKPOINT),
        sonic_spec=str(sc.DEFAULT_SPEC),
        sonic_spec_sha256=sha(sc.DEFAULT_SPEC),
        sonic_tf32=bool(tf32),
        sonic_infer_block=None if tf32 else sc.INFER_BLOCK,
        p_gains=term.p_gains.tolist(),
        d_gains=term.d_gains.tolist(),
        frozen_action_scales=list(config.robot.control.frozen_action_scales),
        action_scales=term.action_scales.tolist(),
        residual="dq = frozen paper-D action scale * a, added to every frame of SONIC's 10-frame window",
        env_action="a_eff = (q* - default) / scale_D (action-rate penalty on the physical target, D units)",
        last_action_observation="policy's own a via action_manager.policy_action",
        pd_target="q* verbatim (joint_target_override)",
        output_layer_init="zeros (weights and bias)",
        sonic_history_reset="per env on episode reset and on Plan5 reference wrap",
    )
    if kind == "v1":
        return out
    rc = env.robot_config
    out.update(kind=kind, train_diag=bool(train_diag),
               init_noise_std=None if ppo_config is None else float(ppo_config.init_noise_std),
               entropy_coef=None if ppo_config is None else float(ppo_config.entropy_coef),
               d_action_clip=dict(clip_actions=bool(rc.control.clip_actions), action_clip_value=float(rc.control.action_clip_value)),
               joint_names=list(env.dof_names),
               frozen_action_scales_deg=[float(v) * 57.29577951308232 for v in config.robot.control.frozen_action_scales])
    if kind == "in":
        out.update(residual="C-in: a clipped to [-1, 1] in the env; dq = frozen paper-D scale * clip(a) added to every frame "
                            "of SONIC's 10-frame joint reference window, then clamped to the joint limits",
                   joint_limits_lower=list(rc.dof_pos_lower_limit_list), joint_limits_upper=list(rc.dof_pos_upper_limit_list),
                   last_action_observation="clip(a, -1, 1) via action_manager.policy_action",
                   log_prob="of the unclipped sampled a (learner unchanged)")
    else:
        out.update(residual="C-out: SONIC gets the unmodified reference; q* = SONIC output + frozen paper-D scale * a_c, "
                            "a_c = a clipped as D's action term (clip_actions / action_clip_value)",
                   last_action_observation="policy's own a via action_manager.policy_action (as D)",
                   sonic_last_action="SONIC's own previous output, without the correction",
                   action_rate_penalty="on q* in D units: a_eff = (q* - default) / scale_D")
    return out
