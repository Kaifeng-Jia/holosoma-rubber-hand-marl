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
"""
from __future__ import annotations

import torch

from system1 import sonic_controller as sc
from system1.scheme_a import SonicDualDriver


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
    _OWN = ("env", "driver", "E", "A", "scale_d", "command", "_prev_t", "_dones", "sonic_resets", "step")

    def __setattr__(self, name, value):  # assignments like env.episode_length_buf = ... reach the real env
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __init__(self, env, device, tf32=False, infer_block=sc.INFER_BLOCK):
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
        dq = a * self.scale_d
        a_eff, _ = self.driver.actions(self.command, dones=self._dones, delta_q_h=dq)
        self.env.action_manager.policy_action[:] = a.reshape(self.E, -1)
        self._prev_t = self.command.time_steps.clone()
        out = self.env.step({**actor_state, "actions": a_eff})
        dones = out[2].reshape(self.E)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            self.driver.reset(done_ids)
        self._sync_reference_jumps()  # reference frame jumped without an env reset (Plan5 wrap)
        self._prev_t = self.command.time_steps.clone()
        self._dones = dones.float()
        return out


def scheme_c_run_config(env, config, tf32: bool):
    """Fields that differ from D because actions go through SONIC (recorded in run_config)."""
    import hashlib

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    term = env.action_manager.get_term("joint_control")
    return dict(
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
