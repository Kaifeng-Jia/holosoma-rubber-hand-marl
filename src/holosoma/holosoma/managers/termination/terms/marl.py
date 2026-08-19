"""Joint termination terms for Plan 5 homogeneous multi-agent tracking."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.managers.command.terms.marl import PairedA1MotionCommand
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import quat_error_magnitude, quat_rotate_inverse


class JointBadTrackingZOnly(TerminationTermBase):
    """Reset the shared environment when either robot or the table loses tracking."""

    def __init__(self, cfg: TerminationTermCfg, env: Any):
        super().__init__(cfg, env)
        params = cfg.params
        self.ref_pos_threshold = float(params["bad_ref_pos_threshold"])
        self.ref_ori_threshold = float(params["bad_ref_ori_threshold"])
        self.body_pos_threshold = float(params["bad_motion_body_pos_threshold"])
        self.object_pos_threshold = float(params["bad_object_pos_threshold"])
        self.object_ori_threshold = float(params["bad_object_ori_threshold"])
        self.body_names_to_track = list(params["body_names_to_track"])
        names = list(params["bad_motion_body_pos_body_names"])
        self.body_pos_body_names = names
        self.body_indexes = torch.tensor(
            [self.body_names_to_track.index(name) for name in names],
            dtype=torch.long,
            device=env.device,
        )
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        command = env.command_manager.get_state("paired_motion_command")
        if not isinstance(command, PairedA1MotionCommand):
            raise TypeError(f"Expected PairedA1MotionCommand, got {type(command)}")
        if list(command.motion_cfg.body_names_to_track) != self.body_names_to_track:
            raise ValueError("Termination body_names_to_track must match paired motion command")

        actual_ref_pos = env.simulator.agent_rigid_body_pos[:, :, command.ref_body_index]
        ref_height_error = torch.abs(command.agent_ref_pos_w[..., 2] - actual_ref_pos[..., 2])
        bad_ref_pos = ref_height_error > self.ref_pos_threshold

        actual_ref_quat = env.simulator.agent_rigid_body_rot[:, :, command.ref_body_index]
        gravity = torch.zeros_like(command.agent_ref_pos_w)
        gravity[..., 2] = -1.0
        reference_gravity = quat_rotate_inverse(
            command.agent_ref_quat_w.reshape(-1, 4),
            gravity.reshape(-1, 3),
            w_last=True,
        ).reshape_as(gravity)
        actual_gravity = quat_rotate_inverse(
            actual_ref_quat.reshape(-1, 4),
            gravity.reshape(-1, 3),
            w_last=True,
        ).reshape_as(gravity)
        gravity_z_error = torch.abs(reference_gravity[..., 2] - actual_gravity[..., 2])
        bad_ref_ori = gravity_z_error > self.ref_ori_threshold

        body_error_z = torch.abs(
            command.agent_body_pos_relative_w[:, :, self.body_indexes, 2]
            - command.simulator_agent_body_pos_w[:, :, self.body_indexes, 2]
        )
        bad_body = torch.any(body_error_z > self.body_pos_threshold, dim=-1)
        bad_robot = torch.any(bad_ref_pos | bad_ref_ori | bad_body, dim=1)

        object_position_error = torch.linalg.vector_norm(
            command.object_pos_w - command.simulator_object_pos_w, dim=-1
        )
        bad_object_pos = object_position_error > self.object_pos_threshold
        object_orientation_error = quat_error_magnitude(
            command.object_quat_w, command.simulator_object_quat_w
        )
        bad_object_ori = object_orientation_error > self.object_ori_threshold
        self.last_diagnostics = {
            "bad_robot_ref_height_by_agent": bad_ref_pos.clone(),
            "bad_robot_orientation_by_agent": bad_ref_ori.clone(),
            "bad_robot_body_height_by_agent": bad_body.clone(),
            "bad_robot": bad_robot.clone(),
            "robot_ref_height_error_m_by_agent": ref_height_error.clone(),
            "robot_ref_height_reference_m_by_agent": command.agent_ref_pos_w[..., 2].clone(),
            "robot_ref_height_actual_m_by_agent": actual_ref_pos[..., 2].clone(),
            "robot_gravity_z_error_by_agent": gravity_z_error.clone(),
            "robot_reference_gravity_z_by_agent": reference_gravity[..., 2].clone(),
            "robot_actual_gravity_z_by_agent": actual_gravity[..., 2].clone(),
            "robot_body_height_error_m_by_agent": body_error_z.clone(),
            "robot_max_body_height_error_m_by_agent": body_error_z.max(dim=-1).values.clone(),
            "bad_object_position": bad_object_pos.clone(),
            "bad_object_orientation": bad_object_ori.clone(),
            "object_position_error_m": object_position_error.clone(),
            "object_orientation_error_rad": object_orientation_error.clone(),
            "reference_object_position": command.object_pos_w.clone(),
            "actual_object_position": command.simulator_object_pos_w.clone(),
            "reference_object_quaternion": command.object_quat_w.clone(),
            "actual_object_quaternion": command.simulator_object_quat_w.clone(),
            "reference_frame": command.time_steps.clone(),
        }
        return bad_robot | bad_object_pos | bad_object_ori

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        return
