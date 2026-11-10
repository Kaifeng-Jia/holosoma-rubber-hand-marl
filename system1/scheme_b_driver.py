"""Scheme B (T02 v2) for both robots of the holosoma paired env: SONIC planner (lower body / root path, position
feedback through the waypoints) + teleop encoder (3-point targets; B1 adds object feedback on the two wrists).

Variants (preregistered, all reported): B-open (official planner behaviour: no position feedback, wrist targets =
reference; +/-2 only), B0 (wrist targets = reference), B1-0 / B1-2 / B1-4 (wrist targets follow the
actual object pose; pushed inward by delta = 0 / 2 / 4 cm along the precomputed contact normal at reference contact
frames). Everything else - PD, action scales, robot, object, SONIC - is as scheme A (subclass of SonicDualDriver).
Planner rules: system1/scheme_b.py. Preregistration: results_system1/T02/preregistration.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from system1 import scheme_b as sb
from system1 import sonic_controller as sc
from system1.scheme_a import SonicDualDriver

ROOT = Path(__file__).resolve().parents[1]
CONTACTS = ROOT / "logs_system1/T02/contacts_v2"   # contact definition v2 (Kevin 2026-11-10)
DELTA = {"B0": None, "B1-0": 0.0, "B1-2": 0.02, "B1-4": 0.04, "B-open": None}
FEEDBACK = {"B0": True, "B1-0": True, "B1-2": True, "B1-4": True, "B-open": False}  # B-open: official behaviour


def _xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def _rot(q_wxyz):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(q_wxyz)[..., [1, 2, 3, 0]]).as_matrix()


class SonicPlannerDualDriver(SonicDualDriver):
    def __init__(self, env, device, variant, case, tf32=False):
        super().__init__(env, device, tf32)
        if variant not in DELTA:
            raise ValueError(variant)
        self.variant, self.delta, self.feedback = variant, DELTA[variant], FEEDBACK[variant]
        sim = env.simulator
        bl = list(sim._body_list)
        command = env.command_manager.get_state("paired_motion_command")
        ref = command.reference
        self.N = ref.num_frames
        s = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in ref.sample(torch.arange(self.N, device=device)).items()}
        self.ref_pel_pos = s["agent_body_pos_w"][:, :, bl.index("pelvis")]                         # [N, A, 3] env-local
        self.ref_pel_quat = _xyzw_to_wxyz(s["agent_body_quat_w"][:, :, bl.index("pelvis")])          # [N, A, 4] wxyz
        pi = [bl.index(n) for n in sc.VR3_BODIES]
        self.ref_pts_pos = s["agent_body_pos_w"][:, :, pi]                                          # [N, A, 3, 3]
        self.ref_pts_quat = _xyzw_to_wxyz(s["agent_body_quat_w"][:, :, pi])                          # [N, A, 3, 4]
        self.ref_obj_pos, self.ref_obj_quat = s["object_pos_w"], _xyzw_to_wxyz(s["object_quat_w"])   # [N, 3], [N, 4]
        self.vr_off = np.array(sc.VR3_OFFSETS)
        if self.delta is not None:
            man = json.loads((CONTACTS / "manifest.json").read_text())["tasks"][case]
            z = np.load(ROOT / man["file"])
            if z["n_o"].shape[0] != self.N:
                raise ValueError("contact file frames differ from the reference")
            self.n_o, self.contact, self.contacts_sha256 = z["n_o"], z["contact"], man["sha256"]
        self.planner = sb.Planner()
        self.tracks = [sb.PlannerTrack(self.planner, self.ctl.joint_names) for _ in range(self.E * self.A)]
        self.H = sb.MAX_FRAMES_30 / 30.0 * 50.0
        self.need_init, self.step = True, 0

    def reset(self, ids=None):
        super().reset(ids)
        self.need_init, self.step = True, 0
        self.planner.calls, self.planner.seconds = 0, 0.0

    def episode_stats(self):
        return dict(planner_calls=self.planner.calls, planner_seconds=self.planner.seconds, b_variant=self.variant,
                    position_feedback=self.feedback)

    @torch.no_grad()
    def actions(self, command, dones=None, delta_q_h=None):
        if delta_q_h is not None:
            raise ValueError("scheme B takes no residual")
        E, A, dev = self.E, self.A, self.device
        t = command.time_steps.long().cpu().numpy()
        origin = self.sim.scene.env_origins.detach().cpu().numpy().astype(np.float64)
        rs = self.sim.agent_root_states.detach().cpu().numpy().astype(np.float64)                  # [E, A, 13] world, xyzw
        q, w_b, jq, jqd = self.robot_state()
        q_np = q.cpu().numpy().astype(np.float64)
        jq_np = jq.cpu().numpy().astype(np.float64)
        obj_pos = self.sim_object_pos(command) - origin                                             # [E, 3]
        obj_quat = _xyzw_to_wxyz(command.simulator_object_quat_w.detach().cpu().numpy().astype(np.float64))
        if self.need_init:
            for i, tr in enumerate(self.tracks):
                tr.initialize(rs[i // A, i % A, :3] - origin[i // A], q_np[i], jq_np[i])
            self.need_init = False
        else:
            for tr in self.tracks:
                tr.advance()
        lp, lv, pq, apos, aq, ppos, pquat = [], [], [], [], [], [], []
        for i, tr in enumerate(self.tracks):
            e, r = divmod(i, A)
            k = int(min(t[e], self.N - 1))
            pos_r = rs[e, r, :3] - origin[e]
            if self.step % sb.PLANNER_TICK == 0:
                proot, pqt = tr.root()
                shift = proot[:2] - pos_r[:2] if self.feedback else np.zeros(2)  # position feedback (preregistered)
                dyaw = sb.wrap(sb.yaw_of(pqt) - sb.yaw_of(q_np[i])) if self.feedback else 0.0
                wp, wy, _ = sb.waypoints(self.ref_pel_pos[:, r], self.ref_pel_quat[:, r], k, self.N, self.H, shift, dyaw)
                d = wp[-1, :2] - proot[:2]
                tr.replan(sb.MODE_WALK, sb.ref_planar_speed(self.ref_pel_pos[:, r], k, self.N),
                          np.r_[d / max(np.linalg.norm(d), 1e-9), 0.0], (np.cos(wy[-1]), np.sin(wy[-1]), 0.0),
                          float(self.ref_pel_pos[k, r, 2]), 1, wp, wy)
            a_, b_ = tr.future_window(self.ctl.n_future, self.ctl.skip)
            lp.append(a_); lv.append(b_)
            pq.append(tr.root()[1])
            pts = self.ref_pts_pos[k, r].copy()
            qts = self.ref_pts_quat[k, r]
            if self.delta is not None:   # B1: wrists (points 0, 1) follow the actual object pose
                Ra, Rr = _rot(obj_quat[e]), _rot(self.ref_obj_quat[k])
                for h in (0, 1):
                    vr = pts[h] + _rot(qts[h]) @ self.vr_off[h]                  # reference VR point (training offset)
                    vr = Ra @ (Rr.T @ (vr - self.ref_obj_pos[k])) + obj_pos[e]   # T_act * T_ref^-1 * p
                    if self.contact[k, r, h] and self.delta > 0:
                        vr = vr + self.delta * (Ra @ self.n_o[k, r, h])
                    pts[h] = vr - _rot(qts[h]) @ self.vr_off[h]                 # orientation kept from the reference
            ppos.append(pts); pquat.append(qts)
            apos.append(self.ref_pel_pos[k, r]); aq.append(self.ref_pel_quat[k, r])
        T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
        d = None if dones is None else dones.repeat_interleave(A).float()
        action, target_s, _ = self.ctl.act_teleop(q, w_b, jq, jqd, T(lp), T(lv), T(apos), T(aq), T(ppos), T(pquat),
                                                  dones=d, ori_anchor_quat=T(pq))
        self.step += 1
        q_target_h = self.ctl.to_holosoma(target_s).reshape(E, A, -1)
        self.term.joint_target_override = q_target_h
        self.last_q_target_h = q_target_h
        a_h = (q_target_h - self.default_h) / self.scale_h
        return a_h.reshape(E, -1), action.reshape(E, A, -1)

    @staticmethod
    def sim_object_pos(command):
        return command.simulator_object_pos_w.detach().cpu().numpy().astype(np.float64)
