"""SONIC driving both robots of a holosoma paired env (scheme A; reused by C and B).

Each robot gets its own SONIC instance (shared weights, one batch of E*2). Inputs are
that robot's DuetMimic runtime reference (the paper's paired reference), read from
``command.reference`` at the command's current frame plus SONIC's future window.
SONIC never sees the object.

Low-level control (README "共同约定"):
* PD: only hip_pitch Kp/Kd are SONIC's; robot model limits stay ours;
* action scales stay the paper's (F2: ``frozen_action_scales``);
* the PD receives SONIC's q* verbatim via ``joint_target_override`` (F2); the action
  buffer gets a_h = (q* - default)/scale for logging/observations.
``hip_pitch_actuator='sonic'`` is the preregistered sensitivity variant: hip_pitch also
uses SONIC's effort 139, velocity 20 and armature 0.0251 (scales still frozen).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from system1 import sonic_controller as sc

# gear_sonic/envs/manager_env/robots/g1.py (actuators "legs", 7520_22 motor)
SONIC_HIP_PITCH = dict(kp=99.098427777, kd=6.308801854, effort=139.0, velocity=20.0, armature=0.025101925)
EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")


def paper_action_scales(robot_cfg):
    """Per-DOF action scales of the paper config, computed exactly as the action terms do."""
    ctl = robot_cfg.control
    out = []
    for i, name in enumerate(robot_cfg.dof_names):
        kp = None
        for key, val in ctl.stiffness.items():
            if key in name:
                kp = val
        if ctl.action_scales_by_effort_limit_over_p_gain:
            eff = robot_cfg.dof_effort_limit_list[i]
            # same arithmetic as the action terms: python float / float32 tensor element
            v = 0.0 if kp == 0.0 else ctl.action_scale * eff / torch.tensor(kp, dtype=torch.float32)
            out.append(float(torch.as_tensor(v, dtype=torch.float32)))
        else:
            out.append(float(ctl.action_scale))
    return tuple(out)


def with_sonic_pd(config, hip_pitch_actuator: str = "ours"):
    """SONIC low-level settings for schemes A-C: hip_pitch PD -> SONIC, action scales frozen."""
    robot = config.robot
    frozen = paper_action_scales(robot)
    ctl = replace(robot.control,
                  stiffness={**robot.control.stiffness, "hip_pitch": SONIC_HIP_PITCH["kp"]},
                  damping={**robot.control.damping, "hip_pitch": SONIC_HIP_PITCH["kd"]},
                  frozen_action_scales=frozen)
    robot = replace(robot, control=ctl)
    if hip_pitch_actuator == "sonic":
        idx = [i for i, n in enumerate(robot.dof_names) if "hip_pitch" in n]
        eff, vel, arm = list(robot.dof_effort_limit_list), list(robot.dof_vel_limit_list), list(robot.dof_armature_list)
        for i in idx:
            eff[i], vel[i], arm[i] = SONIC_HIP_PITCH["effort"], SONIC_HIP_PITCH["velocity"], SONIC_HIP_PITCH["armature"]
        robot = replace(robot, dof_effort_limit_list=eff, dof_vel_limit_list=vel, dof_armature_list=arm)
    elif hip_pitch_actuator != "ours":
        raise ValueError(hip_pitch_actuator)
    return replace(config, robot=robot)


def assert_reference_bodies(command, sim, names):
    """F4 (independent of the loader's index mapping): for each named body, the loaded
    reference at the simulator index equals the file data for that name — quaternions
    exactly, and positions relative to pelvis (cancels the Push lateral offset)."""
    from holosoma.utils.path import resolve_data_file_path

    bl = list(sim._body_list)
    path = command.paired_reference_file or command.motion_cfg.motion_file
    with np.load(resolve_data_file_path(str(path)), allow_pickle=False) as z:
        fnames = [str(x) for x in z["body_names"]]
        if "agent_body_quat_w" in z.files:
            fq, fp = z["agent_body_quat_w"], z["agent_body_pos_w"]  # [T, A, B, *]
        else:
            fq, fp = z["body_quat_w"][:, None], z["body_pos_w"][:, None]
    ref = command.reference
    lq = ref.agent_body_quat_w.detach().cpu().numpy()  # xyzw, [T, A, Bsim, 4]
    lp = ref.agent_body_pos_w.detach().cpu().numpy()
    fpel, spel = fnames.index("pelvis"), bl.index("pelvis")
    for n in names:
        j, i = fnames.index(n), bl.index(n)
        q_file = np.broadcast_to(fq[:, :, j][..., [1, 2, 3, 0]], lq[:, :, i].shape).astype(np.float32)
        if not np.array_equal(q_file, lq[:, :, i]):
            raise AssertionError(f"reference quaternion mismatch for body '{n}'")
        rel_file = np.broadcast_to(fp[:, :, j] - fp[:, :, fpel], lp[:, :, i].shape)
        if np.abs(rel_file - (lp[:, :, i] - lp[:, :, spel])).max() > 1e-5:
            raise AssertionError(f"reference position mismatch for body '{n}'")


def assert_body_correspondence(sim, reference, names):
    """F4: names used by the driver exist in the sim body list and the reference maps to them."""
    bl = list(sim._body_list)
    for n in names:
        if n not in bl:
            raise AssertionError(f"body '{n}' missing from simulator body list")
    ref_names = getattr(reference, "body_names", None)
    if ref_names is not None:  # reference arrays are stored in simulator body order
        for n in names:
            i = bl.index(n)
            if str(ref_names[i]) != n:
                raise AssertionError(f"reference body {i} is '{ref_names[i]}', simulator has '{n}'")


class SonicDualDriver:
    def __init__(self, env, device, tf32=False):
        sim = env.simulator
        self.env, self.sim = env, sim
        self.E, self.A = env.num_envs, env.num_agents
        self.ctl = sc.SonicController(batch=self.E * self.A, device=device, holosoma_joint_names=list(sim.dof_names), tf32=tf32)
        if list(sim.dof_names) != list(env.dof_names):
            raise AssertionError("simulator and env DOF order differ")
        self.default_h = env.default_dof_pos.reshape(-1)[: len(sim.dof_names)]
        self.scale_h = env.action_scales.reshape(-1)
        self.term = env.action_manager.get_term("joint_control")
        bl = list(sim._body_list)
        cmd = self.ctl.spec["cmd_body_names"]
        assert_body_correspondence(sim, None, ["pelvis", *cmd, *EE_BODIES])
        self.pelvis = bl.index("pelvis")
        self.cmd_idx = torch.tensor([bl.index(b) for b in cmd], device=device)
        self.pel_in_cmd = cmd.index("pelvis")
        self.ee_in_cmd = torch.tensor([cmd.index(b) for b in EE_BODIES], device=device)
        self.device = device
        assert_reference_bodies(env.command_manager.get_state("paired_motion_command"), sim, ["pelvis", *cmd, *EE_BODIES])
        self.last_q_target_h = None
        self.max_pd_target_diff = 0.0

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.E, device=self.device)
        ids = torch.as_tensor(ids, device=self.device)
        rob = (ids[:, None] * self.A + torch.arange(self.A, device=self.device)[None]).reshape(-1)
        self.ctl.reset(rob)

    def _flat(self, x):  # [E, A, ...] -> [E*A, ...]
        return x.reshape(self.E * self.A, *x.shape[2:])

    def reference_window(self, command, time_steps=None):
        """SONIC-order future windows [E*A, F, 29] / wxyz pelvis quats [E*A, F, 4]."""
        E, A, F = self.E, self.A, self.ctl.n_future
        ref = command.reference
        t = command.time_steps.long() if time_steps is None else time_steps
        idx = sc.future_indices(t, ref.num_frames, F, self.ctl.skip)  # [E, F]
        s = ref.sample(idx.reshape(-1))
        jp = s["agent_joint_pos"].reshape(E, F, A, -1).permute(0, 2, 1, 3).reshape(E * A, F, -1)
        jv = s["agent_joint_vel"].reshape(E, F, A, -1).permute(0, 2, 1, 3).reshape(E * A, F, -1)
        rq = s["agent_body_quat_w"][:, :, self.pelvis].reshape(E, F, A, 4).permute(0, 2, 1, 3).reshape(E * A, F, 4)
        return self.ctl.to_sonic(jp), self.ctl.to_sonic(jv), sc.xyzw_to_wxyz(rq), idx

    def robot_state(self):
        rs = self._flat(self.sim.agent_root_states)  # xyzw
        q = sc.xyzw_to_wxyz(rs[:, 3:7])
        w_b = sc.quat_apply(sc.quat_conj(q), rs[:, 10:13])
        return q, w_b, self.ctl.to_sonic(self._flat(self.sim.agent_dof_pos)), self.ctl.to_sonic(self._flat(self.sim.agent_dof_vel))

    @torch.no_grad()
    def actions(self, command, dones=None, delta_q_h=None):
        """Returns holosoma actions [E, A*29] and SONIC raw actions [E, A, 29].

        ``delta_q_h`` [E, A, 29] (holosoma order) is added to every reference frame of the
        window before SONIC (scheme C); None for scheme A.
        """
        jp, jv, rq, _ = self.reference_window(command)
        if delta_q_h is not None:
            jp = jp + self.ctl.to_sonic(self._flat(delta_q_h))[:, None, :]
        q, w_b, jq, jqd = self.robot_state()
        d = None if dones is None else dones.repeat_interleave(self.A).float()
        action, target_s, _ = self.ctl.act(q, w_b, jq, jqd, jp, jv, rq, d)
        q_target_h = self.ctl.to_holosoma(target_s).reshape(self.E, self.A, -1)
        self.term.joint_target_override = q_target_h  # PD uses q* verbatim (F2)
        self.last_q_target_h = q_target_h
        a_h = (q_target_h - self.default_h) / self.scale_h
        return a_h.reshape(self.E, -1), action.reshape(self.E, self.A, -1)

    def check_pd_target(self):
        """Max |PD target - q*| over the last control step (F2 acceptance: exactly 0)."""
        if self.term.last_pd_target is None or self.last_q_target_h is None:
            return None
        d = float((self.term.last_pd_target - self.last_q_target_h).abs().max().item())
        self.max_pd_target_diff = max(self.max_pd_target_diff, d)
        return d

    @torch.no_grad()
    def tracking(self, command, frame: int):
        """Post-step SONIC tracking metrics per robot vs reference frame ``frame``."""
        E, A = self.E, self.A
        s = command.reference.sample(torch.tensor([frame], device=self.device))
        origin = self.sim.scene.env_origins[:, None, None, :]
        ref = s["agent_body_pos_w"][:, :, self.cmd_idx] + origin  # [E, A, 14, 3]
        body = self.sim.agent_rigid_body_pos[:, :, self.cmd_idx]
        err = torch.linalg.norm(body - ref, dim=-1)  # [E, A, 14]
        rel = torch.linalg.norm((body - body[:, :, self.pel_in_cmd : self.pel_in_cmd + 1])
                                - (ref - ref[:, :, self.pel_in_cmd : self.pel_in_cmd + 1]), dim=-1)
        rq_ref = sc.xyzw_to_wxyz(s["agent_body_quat_w"][:, :, self.pelvis])
        rq = sc.xyzw_to_wxyz(self.sim.agent_rigid_body_rot[:, :, self.pelvis])
        dq = sc.quat_mul(sc.quat_conj(rq_ref.expand(E, A, 4)), rq)
        ang = 2 * torch.atan2(torch.linalg.norm(dq[..., 1:], dim=-1), dq[..., 0].abs())
        pel_dz = (body[:, :, self.pel_in_cmd, 2] - ref[:, :, self.pel_in_cmd, 2]).abs()
        pel_dxy = torch.linalg.norm(body[:, :, self.pel_in_cmd, :2] - ref[:, :, self.pel_in_cmd, :2], dim=-1)
        ee_dz = (body[:, :, self.ee_in_cmd, 2] - ref[:, :, self.ee_in_cmd, 2]).abs().amax(dim=-1)
        fall = (pel_dz > 0.25) | (ang.square() > 1.0) | (ee_dz > 0.25)
        return dict(mpjpe_mm=err.mean(dim=-1) * 1000, mpjpe_pelvis_rel_mm=rel.mean(dim=-1) * 1000,
                    pelvis_dxy_m=pel_dxy, pelvis_dz=pel_dz, pelvis_ori_err2=ang.square(), ee_dz=ee_dz, fall=fall)
