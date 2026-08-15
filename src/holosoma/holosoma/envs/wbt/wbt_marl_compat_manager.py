"""Single-robot WBT environments with explicit ghost-teammate interfaces."""

from __future__ import annotations

import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.utils.rotations import quat_rotate_inverse, yaw_quat


def _table_local_x_in_world_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return table local X expressed in world coordinates for xyzw quaternions."""
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape [N, 4], got {tuple(quaternion.shape)}")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm <= 0.0):
        raise ValueError("Object quaternion contains a zero-norm value")
    x, y, z, w = (quaternion / norm).unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ),
        dim=-1,
    )


def _finite_difference_by_clip(
    values: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Differentiate concatenated trajectories without crossing clip boundaries."""
    if values.ndim != 2:
        raise ValueError(f"Expected values shape [T, D], got {tuple(values.shape)}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    derivative = torch.zeros_like(values)
    for start_tensor, end_tensor in zip(starts, ends, strict=True):
        start = int(start_tensor.item())
        end = int(end_tensor.item())
        length = end - start
        if length < 2:
            continue
        if length == 2:
            derivative[start] = (values[start + 1] - values[start]) / dt
            derivative[end - 1] = derivative[start]
        else:
            derivative[start] = (-3.0 * values[start] + 4.0 * values[start + 1] - values[start + 2]) / (2.0 * dt)
            derivative[end - 1] = (
                3.0 * values[end - 1] - 4.0 * values[end - 2] + values[end - 3]
            ) / (2.0 * dt)
            derivative[start + 1 : end - 1] = (values[start + 2 : end] - values[start : end - 2]) / (2.0 * dt)
    return derivative


class GhostTeammateWholeBodyTrackingManager(WholeBodyTrackingManager):
    """Expose a four-value teammate buffer without changing standard WBT."""

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.teammate_relative_position_b = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.teammate_relative_velocity_b = torch.zeros_like(self.teammate_relative_position_b)


class TrajectoryGhostTeammateWholeBodyTrackingManager(GhostTeammateWholeBodyTrackingManager):
    """Observe a non-physical teammate following the opposite shifted A1 reference."""

    observer_side: int = 0
    lateral_spacing_m: float = 0.8

    def _init_buffers(self) -> None:
        super()._init_buffers()
        if self.observer_side not in (-1, 1):
            raise ValueError(f"observer_side must be -1 or +1, got {self.observer_side}")
        if self.lateral_spacing_m <= 0.0:
            raise ValueError(f"lateral_spacing_m must be positive, got {self.lateral_spacing_m}")
        self._ghost_relative_reference_position_w: torch.Tensor | None = None
        self._ghost_relative_reference_velocity_w: torch.Tensor | None = None

    def _build_ghost_reference_cache(self) -> None:
        motion_command = self.command_manager.get_state("motion_command")
        if motion_command is None:
            raise RuntimeError("motion_command is required for trajectory ghost observations")
        motion = motion_command.motion
        if not motion.has_object:
            raise ValueError("trajectory ghost observations require an object reference")

        table_axis_w = _table_local_x_in_world_xyzw(motion.object_quat_w)
        relative_position_w = -float(self.observer_side) * self.lateral_spacing_m * table_axis_w
        fps = int(torch.as_tensor(motion.fps).reshape(-1)[0].item())
        if fps <= 0:
            raise ValueError(f"Motion FPS must be positive, got {fps}")
        relative_velocity_w = _finite_difference_by_clip(
            relative_position_w,
            motion.motion_start_idx,
            motion.motion_end_idx,
            1.0 / float(fps),
        )
        self._ghost_relative_reference_position_w = relative_position_w
        self._ghost_relative_reference_velocity_w = relative_velocity_w

    def _update_trajectory_ghost_buffers(self) -> None:
        if self._ghost_relative_reference_position_w is None:
            self._build_ghost_reference_cache()
        assert self._ghost_relative_reference_position_w is not None
        assert self._ghost_relative_reference_velocity_w is not None

        motion_command = self.command_manager.get_state("motion_command")
        if motion_command is None:
            raise RuntimeError("motion_command is required for trajectory ghost observations")
        time_steps = motion_command.time_steps
        relative_reference_position_w = self._ghost_relative_reference_position_w[time_steps]
        relative_reference_velocity_w = self._ghost_relative_reference_velocity_w[time_steps]

        ghost_position_w = motion_command.root_pos_w + relative_reference_position_w
        ghost_velocity_w = motion_command.root_lin_vel_w + relative_reference_velocity_w
        self_position_w = self.simulator.robot_root_states[:, :3]
        self_velocity_w = self.simulator.robot_root_states[:, 7:10]
        self_yaw_w = yaw_quat(self.simulator.robot_root_states[:, 3:7], w_last=True)

        relative_position_b = quat_rotate_inverse(
            self_yaw_w,
            ghost_position_w - self_position_w,
            w_last=True,
        )
        relative_velocity_b = quat_rotate_inverse(
            self_yaw_w,
            ghost_velocity_w - self_velocity_w,
            w_last=True,
        )
        self.teammate_relative_position_b.copy_(relative_position_b[:, :2])
        self.teammate_relative_velocity_b.copy_(relative_velocity_b[:, :2])

    def _update_tasks_callback(self) -> None:
        super()._update_tasks_callback()
        self._update_trajectory_ghost_buffers()


class LeftTrajectoryGhostTeammateWholeBodyTrackingManager(TrajectoryGhostTeammateWholeBodyTrackingManager):
    """Trajectory ghost for the observer on table local negative X."""

    observer_side = -1


class RightTrajectoryGhostTeammateWholeBodyTrackingManager(TrajectoryGhostTeammateWholeBodyTrackingManager):
    """Trajectory ghost for the observer on table local positive X."""

    observer_side = 1
