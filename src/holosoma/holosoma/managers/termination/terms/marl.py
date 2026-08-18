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
        self.body_indexes = torch.tensor(
            [self.body_names_to_track.index(name) for name in names],
            dtype=torch.long,
            device=env.device,
        )

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        command = env.command_manager.get_state("paired_motion_command")
        if not isinstance(command, PairedA1MotionCommand):
            raise TypeError(f"Expected PairedA1MotionCommand, got {type(command)}")
        if list(command.motion_cfg.body_names_to_track) != self.body_names_to_track:
            raise ValueError("Termination body_names_to_track must match paired motion command")

        actual_ref_pos = env.simulator.agent_rigid_body_pos[:, :, command.ref_body_index]
        bad_ref_pos = torch.abs(command.agent_ref_pos_w[..., 2] - actual_ref_pos[..., 2]) > self.ref_pos_threshold

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
        bad_ref_ori = torch.abs(reference_gravity[..., 2] - actual_gravity[..., 2]) > self.ref_ori_threshold

        body_error_z = torch.abs(
            command.agent_body_pos_relative_w[:, :, self.body_indexes, 2]
            - command.simulator_agent_body_pos_w[:, :, self.body_indexes, 2]
        )
        bad_body = torch.any(body_error_z > self.body_pos_threshold, dim=-1)
        bad_robot = torch.any(bad_ref_pos | bad_ref_ori | bad_body, dim=1)

        bad_object_pos = (
            torch.linalg.vector_norm(command.object_pos_w - command.simulator_object_pos_w, dim=-1)
            > self.object_pos_threshold
        )
        bad_object_ori = (
            quat_error_magnitude(command.object_quat_w, command.simulator_object_quat_w)
            > self.object_ori_threshold
        )
        return bad_robot | bad_object_pos | bad_object_ori

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        return
