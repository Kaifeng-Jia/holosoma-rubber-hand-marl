"""Scheme C v2 training diagnostics written into metrics.jsonl (T03 C v2, "训练中要记录的诊断"). Off unless the trainer
passes --c-train-diag; with it off nothing here runs and training is unchanged.

Accumulated over every env step of every rollout; the trainer calls ``end_iteration()`` after each rollout and, every
``every`` iterations, ``summary(actor)`` which returns one dict for that window (iterations k-every+1 .. k):
  * correction magnitude |c| per robot-joint-step, c = dq on SONIC's input (C-in, after clipping a) or scale * a on
    SONIC's output (C-out, after D's action clip): window mean of the per-iteration mean, window mean of the
    per-iteration 95th percentile, window max; per-joint mean / p95 of the last iteration of the window;
  * fraction of action dims with |a| > 1 (a = the sampled action the learner stores) and the fraction actually clipped
    by this kind's clip (C-in: |a| > 1; C-out: |a| > D's action_clip_value);
  * action std (mean and per joint) read from the actor at summary time;
  * SONIC fall rule per robot-step, pre-reset (same rule and frame convention as system1/c_diag.py), by criterion;
  * finished episodes split as the learner does (timeout = env extras time_outs, else terminated), terminated ones by
    the joint_bad_tracking sub-terms of the same step: object position / object orientation / robot torso (several can
    be true);
  * checks: C-in max |dq_i| / scale_i (must be <= 1) and max violation of the joint limits by the reference SONIC
    received; C-out max |q* - (SONIC output + c)|.
"""
from __future__ import annotations

import os

import torch

from system1 import sonic_controller as sc


class CTrainDiag:
    def __init__(self, wrap, every=None):
        # SYSTEM1_C_TRAIN_DIAG_EVERY: smoke tests only (production launches do not set it: every 100 iterations)
        every = every or int(os.environ.get("SYSTEM1_C_TRAIN_DIAG_EVERY", "100"))
        self.w, self.d, self.every = wrap, wrap.driver, every
        self.dev = self.d.device
        self.names = list(self.d.sim.dof_names)
        self._reset_window()
        self._steps = []
        self._last_iter_corr = None
        self.checks = dict(c_in_max_dq_over_scale=0.0, c_in_max_ref_limit_violation_rad=0.0, c_out_max_qstar_minus_sonic_plus_c=0.0)
        self._jp = None
        orig_act = self.d.ctl.act

        def act(*args, **kw):
            self._jp = args[4]  # joint reference window SONIC receives (SONIC order)
            return orig_act(*args, **kw)
        self.d.ctl.act = act
        env = wrap.env
        orig_reward = env._compute_reward

        def compute_reward():
            out = orig_reward()
            self._fall()
            return out
        env._compute_reward = compute_reward
        self.term = env.termination_manager._term_instances["joint_bad_tracking"]

    def _reset_window(self):
        self.win = dict(n_iter=0, corr_mean=[], corr_p95=[], corr_max=0.0, a_gt1=0, clipped=0, a_n=0,
                        robot_steps=0, fall=0, pelvis_height=0, orientation=0, wrist_ankle_height=0,
                        done=0, timeout=0, terminated=0, object_position=0, object_orientation=0, robot_torso=0)

    @torch.no_grad()
    def step(self, a, a_used, corr):
        """a: sampled action [E, A, 29]; a_used: after this kind's clip; corr: correction c [E, A, 29]."""
        self._steps.append(corr.abs().reshape(-1, 29).to(torch.float16))
        w = self.win
        w["a_gt1"] += int((a.abs() > 1).sum().item())
        w["clipped"] += int((a_used != a).sum().item())
        w["a_n"] += a.numel()
        k = self.w.kind
        if k == "in":
            r = (corr.abs() / self.w.scale_d).amax().item()
            self.checks["c_in_max_dq_over_scale"] = max(self.checks["c_in_max_dq_over_scale"], float(r))

    @torch.no_grad()
    def mid(self, corr):
        k = self.w.kind
        if k == "in" and self._jp is not None:
            lo, hi = (self.d.ctl.to_sonic(x) for x in self.w.ref_limits_h)
            v = torch.maximum(self._jp - hi, lo - self._jp).clamp(min=0).amax().item()
            self.checks["c_in_max_ref_limit_violation_rad"] = max(self.checks["c_in_max_ref_limit_violation_rad"], float(v))
        if k == "out":
            v = (self.d.term.joint_target_override - (self.d.last_sonic_target_h + corr)).abs().amax().item()
            self.checks["c_out_max_qstar_minus_sonic_plus_c"] = max(self.checks["c_out_max_qstar_minus_sonic_plus_c"], float(v))

    @torch.no_grad()
    def post(self, dones, extras):
        """Same split as the learner (timeout = extras time_outs; terminated = done and not timeout). The termination
        manager's own flags are cleared by the in-step reset, so they are not used; the joint_bad_tracking term keeps
        this step's sub-term flags in last_diagnostics."""
        done = dones.reshape(-1).bool()
        n = int(done.sum().item())
        if not n:
            return
        w = self.win
        to = extras["time_outs"].reshape(-1).bool()
        term = done & ~to
        diag = self.term.last_diagnostics
        w["done"] += n
        w["timeout"] += int((done & to).sum().item())
        w["terminated"] += int(term.sum().item())
        w["object_position"] += int((term & diag["bad_object_position"]).sum().item())
        w["object_orientation"] += int((term & diag["bad_object_orientation"]).sum().item())
        w["robot_torso"] += int((term & diag["bad_robot"]).sum().item())

    @torch.no_grad()
    def _fall(self):
        d, cmd = self.d, self.w.command
        ref = cmd.reference
        f = torch.clamp(cmd.time_steps.long() + 1, max=ref.num_frames - 1)
        s = ref.sample(f)
        origin = d.sim.scene.env_origins[:, None, None, :]
        r = s["agent_body_pos_w"][:, :, d.cmd_idx] + origin
        body = d.sim.agent_rigid_body_pos[:, :, d.cmd_idx]
        rq_ref = sc.xyzw_to_wxyz(s["agent_body_quat_w"][:, :, d.pelvis])
        rq = sc.xyzw_to_wxyz(d.sim.agent_rigid_body_rot[:, :, d.pelvis])
        dq = sc.quat_mul(sc.quat_conj(rq_ref), rq)
        ang = 2 * torch.atan2(torch.linalg.norm(dq[..., 1:], dim=-1), dq[..., 0].abs())
        pel = (body[:, :, d.pel_in_cmd, 2] - r[:, :, d.pel_in_cmd, 2]).abs() > 0.25
        ori = ang.square() > 1.0
        ee = (body[:, :, d.ee_in_cmd, 2] - r[:, :, d.ee_in_cmd, 2]).abs().amax(dim=-1) > 0.25
        fall = pel | ori | ee
        w = self.win
        w["robot_steps"] += int(fall.numel())
        for k, v in (("fall", fall), ("pelvis_height", pel), ("orientation", ori), ("wrist_ankle_height", ee)):
            w[k] += int(v.sum().item())

    @torch.no_grad()
    def end_iteration(self):
        if not self._steps:
            return
        x = torch.cat(self._steps).float()  # [steps * E * A, 29]
        self._steps = []
        flat = x.reshape(-1)
        k = max(1, int(round(0.95 * flat.numel())))
        w = self.win
        w["n_iter"] += 1
        w["corr_mean"].append(float(flat.mean().item()))
        w["corr_p95"].append(float(flat.kthvalue(k).values.item()))
        w["corr_max"] = max(w["corr_max"], float(flat.max().item()))
        self._last_iter_corr = x

    @torch.no_grad()
    def summary(self, actor):
        w = self.win
        std = actor.std.detach().float()
        x = self._last_iter_corr
        per_joint = {}
        if x is not None:
            kq = max(1, int(round(0.95 * x.shape[0])))
            m, p = x.mean(0), x.kthvalue(kq, dim=0).values
            per_joint = {n: dict(mean=float(m[i]), p95=float(p[i])) for i, n in enumerate(self.names)}
        rs = max(w["robot_steps"], 1)
        out = dict(
            kind=self.w.kind, window_iterations=w["n_iter"],
            correction_abs_mean_rad=sum(w["corr_mean"]) / max(len(w["corr_mean"]), 1),
            correction_abs_p95_rad=sum(w["corr_p95"]) / max(len(w["corr_p95"]), 1),
            correction_abs_max_rad=w["corr_max"],
            correction_last_iteration_per_joint=per_joint,
            frac_abs_a_gt_1=w["a_gt1"] / max(w["a_n"], 1), frac_clipped=w["clipped"] / max(w["a_n"], 1),
            action_std_mean=float(std.mean()), action_std_per_joint=[float(v) for v in std.reshape(-1)],
            sonic_fall_rate_per_robot_step=w["fall"] / rs,
            sonic_fall_rate_by_criterion=dict(pelvis_height=w["pelvis_height"] / rs, orientation=w["orientation"] / rs,
                                              wrist_ankle_height=w["wrist_ankle_height"] / rs),
            finished_episodes=w["done"], finished_timeout=w["timeout"], finished_terminated=w["terminated"],
            terminated_object_position=w["object_position"], terminated_object_orientation=w["object_orientation"],
            terminated_robot_torso=w["robot_torso"], checks=dict(self.checks))
        self._reset_window()
        return out
