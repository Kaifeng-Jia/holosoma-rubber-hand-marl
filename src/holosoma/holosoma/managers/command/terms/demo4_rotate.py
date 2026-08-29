"""Static-table, non-looping paired Pull command for Demo 4."""

from __future__ import annotations

import math
from typing import Any

import torch

from holosoma.managers.command.terms.marl import PairedA1MotionCommand
from holosoma.utils.rotations import quaternion_to_matrix


def _table_heading_w(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    """Return normalized world-XY projection of the table's local X axis."""
    local_x_w = quaternion_to_matrix(quaternion_xyzw, w_last=True)[..., :, 0]
    heading = local_x_w[..., :2]
    return heading / torch.linalg.vector_norm(
        heading, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)


def _signed_planar_angle(start: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    cross_z = start[..., 0] * current[..., 1] - start[..., 1] * current[..., 0]
    dot = torch.sum(start * current, dim=-1)
    return torch.atan2(cross_z, dot)


class Demo4RotateMotionCommand(PairedA1MotionCommand):
    """Track two Pull priors without imposing a trajectory on the table.

    The paired runtime must carry a static table channel.  That channel is
    written once on reset for schema and placement only.  During the episode,
    :meth:`step` advances and clamps the robot-reference phase without writing
    any simulator state.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        if self.paired_reference_file is None:
            raise ValueError("Demo 4 rotation requires an explicit paired runtime")
        self.goal_yaw_radians = math.radians(
            float(cfg.params.get("goal_yaw_degrees", 90.0))
        )
        if not 0.0 < self.goal_yaw_radians <= math.pi:
            raise ValueError("Demo 4 goal yaw must lie in (0, 180] degrees")
        self.maximum_success_tilt_radians = math.radians(
            float(cfg.params.get("maximum_success_tilt_degrees", 60.0))
        )
        if not 0.0 < self.maximum_success_tilt_radians < math.pi:
            raise ValueError("Demo 4 success tilt must lie in (0, 180) degrees")

    def setup(self) -> None:
        super().setup()
        reference = self.reference
        if not torch.allclose(
            reference.object_pos_w,
            reference.object_pos_w[:1].expand_as(reference.object_pos_w),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError("Demo 4 paired runtime table position must be static")
        initial_quaternion = reference.object_quat_w[:1]
        alignment = torch.abs(torch.sum(reference.object_quat_w * initial_quaternion, dim=-1))
        if not torch.allclose(alignment, torch.ones_like(alignment), atol=1.0e-6, rtol=0.0):
            raise ValueError("Demo 4 paired runtime table orientation must be static")
        if not torch.allclose(
            reference.object_lin_vel_w,
            torch.zeros_like(reference.object_lin_vel_w),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError("Demo 4 paired runtime table velocity must be zero")

        dtype = reference.object_pos_w.dtype
        self.reset_object_pos_w = torch.zeros(self.num_envs, 3, dtype=dtype, device=self.device)
        self.reset_object_heading_w = torch.zeros(
            self.num_envs, 2, dtype=dtype, device=self.device
        )
        self.last_object_heading_w = torch.zeros(
            self.num_envs, 2, dtype=dtype, device=self.device
        )
        self.unwrapped_yaw_progress_radians = torch.zeros(
            self.num_envs, dtype=dtype, device=self.device
        )
        self.table_reset_write_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _write_reference_state(self, env_ids: torch.Tensor) -> None:
        super()._write_reference_state(env_ids)
        self.table_reset_write_count[env_ids] += 1

    def reset(self, env_ids: torch.Tensor | None) -> None:
        env_ids = self._ensure_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        super().reset(env_ids)
        sample = self.reference.sample(self.time_steps[env_ids])
        origins = self.env.simulator.scene.env_origins[env_ids]
        self.reset_object_pos_w[env_ids] = sample["object_pos_w"] + origins
        heading = _table_heading_w(sample["object_quat_w"])
        self.reset_object_heading_w[env_ids] = heading
        self.last_object_heading_w[env_ids] = heading
        self.unwrapped_yaw_progress_radians[env_ids] = 0.0

    def step(self) -> None:
        self.time_steps.add_(1)
        self.time_steps.clamp_(max=self.reference.num_frames - 1)

    def update_yaw_progress(self) -> None:
        """Accumulate per-step table yaw without the ``atan2`` wrap discontinuity.

        At 50 Hz the physical table cannot rotate by 180 degrees in one control
        interval, so the principal angle between adjacent headings is the
        unambiguous signed increment.  Summing those increments prevents an
        incorrect -180-degree rotation from aliasing to a positive success.
        """

        current_heading = _table_heading_w(self.simulator_object_quat_w)
        yaw_increment = _signed_planar_angle(
            self.last_object_heading_w,
            current_heading,
        )
        self.unwrapped_yaw_progress_radians.add_(yaw_increment)
        self.last_object_heading_w.copy_(current_heading)

    @property
    def signed_yaw_progress_radians(self) -> torch.Tensor:
        return self.unwrapped_yaw_progress_radians

    @property
    def normalized_yaw_progress(self) -> torch.Tensor:
        return torch.clamp(
            self.signed_yaw_progress_radians / self.goal_yaw_radians,
            min=-1.0,
            max=1.0,
        )

    @property
    def table_xy_drift(self) -> torch.Tensor:
        return torch.linalg.vector_norm(
            self.simulator_object_pos_w[:, :2] - self.reset_object_pos_w[:, :2],
            dim=-1,
        )

    @property
    def table_tilt_radians(self) -> torch.Tensor:
        # The Pull table URDF uses local Y as its vertical axis.
        local_y_w = quaternion_to_matrix(
            self.simulator_object_quat_w, w_last=True
        )[..., :, 1]
        return torch.acos(local_y_w[..., 2].clamp(-1.0, 1.0))

    @property
    def goal_reached_safely(self) -> torch.Tensor:
        return (
            self.normalized_yaw_progress >= 1.0 - 1.0e-6
        ) & (self.table_tilt_radians <= self.maximum_success_tilt_radians)

    def update_metrics(self) -> None:
        agent_ref_error = torch.linalg.vector_norm(
            self.agent_ref_pos_w
            - self.env.simulator.agent_rigid_body_pos[:, :, self.ref_body_index],
            dim=-1,
        )
        object_states = self.env.simulator.all_root_states[
            self.object_indices_in_simulator
        ]
        self.metrics["motion/error_ref_pos_mean"] = agent_ref_error.mean(dim=1)
        self.metrics["motion/error_ref_pos_max"] = agent_ref_error.max(dim=1).values
        self.metrics["rotate/yaw_progress_rad"] = (
            self.signed_yaw_progress_radians.clone()
        )
        self.metrics["rotate/yaw_progress_normalized"] = (
            self.normalized_yaw_progress.clone()
        )
        self.metrics["rotate/table_tilt_rad"] = self.table_tilt_radians
        self.metrics["rotate/table_xy_drift_m"] = self.table_xy_drift
        self.metrics["rotate/table_linear_speed_mps"] = torch.linalg.vector_norm(
            object_states[:, 7:10], dim=-1
        )
        self.metrics["rotate/table_angular_speed_radps"] = torch.linalg.vector_norm(
            object_states[:, 10:13], dim=-1
        )


__all__ = [
    "Demo4RotateMotionCommand",
    "_signed_planar_angle",
    "_table_heading_w",
]
