"""Scheme A (static target + SONIC): drive both robots of a holosoma paired env with SONIC.

Each robot gets its own SONIC instance (shared weights, one batch of E*2). Inputs are
that robot's DuetMimic runtime reference (the same paired reference the paper used),
read from ``command.reference`` at the command's current frame plus SONIC's future
window. SONIC never sees the object. Joint targets q* = SONIC default + scale * a
reach holosoma's PD through the existing action term as a_h = (q* - default_h)/scale_h.

``sonic_pd_overrides`` gives the PD gains to use with SONIC (only hip_pitch differs
from ours); robot model limits stay ours.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from system1 import sonic_controller as sc

SONIC_HIP_PITCH_KP = 99.098427777  # STIFFNESS_7520_22 in gear_sonic g1.py
SONIC_HIP_PITCH_KD = 6.308801854  # DAMPING_7520_22
EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")


def with_sonic_pd(config):
    ctl = config.robot.control
    ctl = replace(ctl, stiffness={**ctl.stiffness, "hip_pitch": SONIC_HIP_PITCH_KP},
                  damping={**ctl.damping, "hip_pitch": SONIC_HIP_PITCH_KD})
    return replace(config, robot=replace(config.robot, control=ctl))


class SonicDualDriver:
    def __init__(self, env, device):
        sim = env.simulator
        self.env, self.sim = env, sim
        self.E, self.A = env.num_envs, env.num_agents
        self.ctl = sc.SonicController(batch=self.E * self.A, device=device, holosoma_joint_names=list(sim.dof_names))
        self.default_h = env.default_dof_pos.reshape(-1)[: len(sim.dof_names)]
        self.scale_h = env.action_scales.reshape(-1)
        bl = list(sim._body_list)
        self.pelvis = bl.index("pelvis")
        cmd = self.ctl.spec["cmd_body_names"]
        self.cmd_idx = torch.tensor([bl.index(b) for b in cmd], device=device)
        self.pel_in_cmd = cmd.index("pelvis")
        self.ee_in_cmd = torch.tensor([cmd.index(b) for b in EE_BODIES], device=device)
        self.device = device

    def reset(self):
        self.ctl.reset(torch.arange(self.E * self.A, device=self.device))

    def _flat(self, x):  # [E, A, ...] -> [E*A, ...]
        return x.reshape(self.E * self.A, *x.shape[2:])

    @torch.no_grad()
    def actions(self, command):
        E, A, F = self.E, self.A, self.ctl.n_future
        ref = command.reference
        idx = sc.future_indices(command.time_steps.long(), ref.num_frames, F, self.ctl.skip)  # [E, F]
        s = ref.sample(idx.reshape(-1))
        jp = s["agent_joint_pos"].reshape(E, F, A, -1).permute(0, 2, 1, 3).reshape(E * A, F, -1)
        jv = s["agent_joint_vel"].reshape(E, F, A, -1).permute(0, 2, 1, 3).reshape(E * A, F, -1)
        rq = s["agent_body_quat_w"][:, :, self.pelvis].reshape(E, F, A, 4).permute(0, 2, 1, 3).reshape(E * A, F, 4)
        rs = self._flat(self.sim.agent_root_states)  # xyzw
        q = sc.xyzw_to_wxyz(rs[:, 3:7])
        w_b = sc.quat_apply(sc.quat_conj(q), rs[:, 10:13])
        action, target_s, _ = self.ctl.act(
            q, w_b, self.ctl.to_sonic(self._flat(self.sim.agent_dof_pos)), self.ctl.to_sonic(self._flat(self.sim.agent_dof_vel)),
            self.ctl.to_sonic(jp), self.ctl.to_sonic(jv), sc.xyzw_to_wxyz(rq),
        )
        a_h = (self.ctl.to_holosoma(target_s) - self.default_h) / self.scale_h
        return a_h.reshape(E, A * a_h.shape[-1]), action.reshape(E, A, -1)

    @torch.no_grad()
    def tracking(self, command, frame: int):
        """Post-step SONIC tracking metrics per robot vs reference frame ``frame``."""
        E, A = self.E, self.A
        s = command.reference.sample(torch.tensor([frame], device=self.device))
        origin = self.sim.scene.env_origins[:, None, None, :]
        ref = s["agent_body_pos_w"][:, :, self.cmd_idx] + origin  # [E, A, 14, 3]
        body = self.sim.agent_rigid_body_pos[:, :, self.cmd_idx]
        err = torch.linalg.norm(body - ref, dim=-1)  # [E, A, 14]
        rq_ref = sc.xyzw_to_wxyz(s["agent_body_quat_w"][:, :, self.pelvis])
        rq = sc.xyzw_to_wxyz(self.sim.agent_rigid_body_rot[:, :, self.pelvis])
        dq = sc.quat_mul(sc.quat_conj(rq_ref.expand(E, A, 4)), rq)
        ang = 2 * torch.atan2(torch.linalg.norm(dq[..., 1:], dim=-1), dq[..., 0].abs())
        pel_dz = (body[:, :, self.pel_in_cmd, 2] - ref[:, :, self.pel_in_cmd, 2]).abs()
        ee_dz = (body[:, :, self.ee_in_cmd, 2] - ref[:, :, self.ee_in_cmd, 2]).abs().amax(dim=-1)
        fall = (pel_dz > 0.25) | (ang.square() > 1.0) | (ee_dz > 0.25)
        return dict(mpjpe_mm=err.mean(dim=-1) * 1000, pelvis_dz=pel_dz, pelvis_ori_err2=ang.square(), ee_dz=ee_dz, fall=fall)
