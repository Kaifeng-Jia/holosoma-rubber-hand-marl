"""Scheme C training-rollout diagnostics (Kevin 2026-11-11: why C does not learn). Off unless SYSTEM1_C_DIAG=<out.npz>.

Installed by SonicResidualEnv when the variable is set; records, for every env step of the rollout:
  * |a| and |dq| = |scale_D * a| per env / robot / joint (a = the sampled action the learner stores, unchanged);
  * reference check: the joint window SONIC actually receives (captured at ctl.act) minus
    (reference window at the same time_steps + to_sonic(dq) on every frame), max abs;
  * PD check: max |PD target - q*| (driver.check_pd_target);
  * SONIC fall rule per robot, pre-reset (env._compute_reward hook, like the evaluator), against each env's own
    reference frame min(time_steps + 1, N - 1) (the evaluator's frame convention), split by criterion;
  * per finished episode: whether the SONIC fall rule fired at any step of it.
T03 C v2 kinds (wrap.kind): C-in expects the window clamped to the joint limits after adding dq and records
max |dq_i| / scale_i and the max joint-limit violation of the window SONIC receives; C-out expects the unmodified window
(dq goes on SONIC's output) and records max |q* - (SONIC output + dq)|.
"""
from __future__ import annotations

import atexit
import json
from pathlib import Path

import numpy as np
import torch

from system1 import sonic_controller as sc


class CDiag:
    def __init__(self, wrap, out):
        self.w, self.d = wrap, wrap.driver
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        E, A = wrap.E, wrap.A
        self.dq_abs, self.a_abs, self.t_hist = [], [], []
        self.jp_check, self.pd_check = 0.0, 0.0
        self.kind = getattr(wrap, "kind", "v1")
        self.c_in_dq_over_scale, self.c_in_limit_violation, self.c_out_qstar = 0.0, 0.0, 0.0
        self._dq = None
        self.counts = dict(robot_steps=0, fall=0, pelvis_height=0, orientation=0, wrist_ankle_height=0)
        self.ep_fell = torch.zeros(E, A, dtype=torch.bool, device=self.d.device)
        self.ep_any, self.ep_len = [], []
        self.ep_steps = torch.zeros(E, dtype=torch.long, device=self.d.device)
        self._jp = None
        self._expected = None
        x = torch.arange(29, dtype=torch.float32, device=self.d.device)[None]
        s = self.d.ctl.to_sonic(x)
        self.to_sonic_is_permutation = bool(torch.equal(torch.sort(s, dim=-1).values, x) and torch.equal(self.d.ctl.to_holosoma(s), x))
        orig_act = self.d.ctl.act

        def act(*args, **kw):
            self._jp = args[4].detach().clone()
            return orig_act(*args, **kw)
        self.d.ctl.act = act
        env = wrap.env
        orig_reward = env._compute_reward

        def compute_reward():
            out = orig_reward()
            self._fall()
            return out
        env._compute_reward = compute_reward
        atexit.register(self.save)

    @torch.no_grad()
    def pre(self, a, dq):
        jp_ref, _, _, _ = self.d.reference_window(self.w.command)
        if self.kind == "out":
            self._expected = jp_ref
        else:
            self._expected = jp_ref + self.d.ctl.to_sonic(self.d._flat(dq))[:, None, :]
        if self.kind == "in":
            lo, hi = (self.d.ctl.to_sonic(x) for x in self.w.ref_limits_h)
            self._expected = torch.maximum(torch.minimum(self._expected, hi), lo)
            self.c_in_dq_over_scale = max(self.c_in_dq_over_scale, float((dq.abs() / self.w.scale_d).amax().item()))
        self._dq = dq
        self.dq_abs.append(dq.abs().to(torch.float16).cpu().numpy())
        self.a_abs.append(a.abs().to(torch.float16).cpu().numpy())
        self.t_hist.append(self.w.command.time_steps.detach().cpu().numpy().astype(np.int32))

    @torch.no_grad()
    def mid(self):
        self.jp_check = max(self.jp_check, float((self._jp - self._expected).abs().max().item()))
        if self.kind == "in":
            lo, hi = (self.d.ctl.to_sonic(x) for x in self.w.ref_limits_h)
            v = torch.maximum(self._jp - hi, lo - self._jp).clamp(min=0).amax().item()
            self.c_in_limit_violation = max(self.c_in_limit_violation, float(v))
        if self.kind == "out":
            v = (self.d.term.joint_target_override - (self.d.last_sonic_target_h + self._dq)).abs().amax().item()
            self.c_out_qstar = max(self.c_out_qstar, float(v))

    @torch.no_grad()
    def post(self, dones):
        pd = self.d.check_pd_target()
        if pd is not None:
            self.pd_check = max(self.pd_check, pd)
        self.ep_steps += 1
        done = dones.reshape(-1).bool()
        if done.any():
            self.ep_any.append(self.ep_fell[done].cpu().numpy())
            self.ep_len.append(self.ep_steps[done].cpu().numpy())
            self.ep_fell[done] = False
            self.ep_steps[done] = 0
        if len(self.dq_abs) % 24 == 0:
            self.save()

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
        self.counts["robot_steps"] += int(fall.numel())
        for k, v in (("fall", fall), ("pelvis_height", pel), ("orientation", ori), ("wrist_ankle_height", ee)):
            self.counts[k] += int(v.sum().item())
        self.ep_fell |= fall

    def save(self):
        if not self.dq_abs:
            return
        ep_any = np.concatenate(self.ep_any) if self.ep_any else np.zeros((0, self.w.A), bool)
        ep_len = np.concatenate(self.ep_len) if self.ep_len else np.zeros(0, np.int64)
        np.savez_compressed(self.out, dq_abs=np.stack(self.dq_abs), a_abs=np.stack(self.a_abs), time_steps=np.stack(self.t_hist),
                            scale_d=self.w.scale_d.detach().cpu().numpy(), episode_fell_any=ep_any, episode_length=ep_len,
                            joint_names=np.array(list(self.d.sim.dof_names)))
        self.out.with_suffix(".json").write_text(json.dumps(dict(
            steps=len(self.dq_abs), num_envs=self.w.E, num_agents=self.w.A, reference_check_max_abs=self.jp_check,
            pd_target_minus_qstar_max_abs=self.pd_check, to_sonic_is_permutation=self.to_sonic_is_permutation,
            sonic_fall_counts=self.counts, finished_episodes=int(len(ep_len)), kind=self.kind,
            c_in_max_dq_over_scale=self.c_in_dq_over_scale if self.kind == "in" else None,
            c_in_max_reference_joint_limit_violation_rad=self.c_in_limit_violation if self.kind == "in" else None,
            c_out_max_qstar_minus_sonic_output_plus_dq=self.c_out_qstar if self.kind == "out" else None), indent=1) + "\n")
