"""In-memory paired reference derived from one frozen A1 motion."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.utils.rotations import quat_apply


class PairedA1Reference:
    """Create symmetric robot references while retaining one object trajectory."""

    num_agents = 2

    def __init__(
        self,
        *,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
        object_lin_vel_w: torch.Tensor,
        fps: int,
        lateral_spacing_m: float = 0.8,
    ) -> None:
        self._validate(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            object_pos_w=object_pos_w,
            object_quat_w=object_quat_w,
            object_lin_vel_w=object_lin_vel_w,
            fps=fps,
            lateral_spacing_m=lateral_spacing_m,
        )
        self.fps = fps
        self.lateral_spacing_m = lateral_spacing_m
        self.num_frames = joint_pos.shape[0]

        table_local_x = quat_apply(
            object_quat_w,
            object_quat_w.new_tensor([1.0, 0.0, 0.0]).expand(self.num_frames, -1),
            w_last=True,
        )
        signs = object_pos_w.new_tensor([-1.0, 1.0])
        self.lateral_offset_w = (
            0.5
            * lateral_spacing_m
            * signs.view(1, self.num_agents, 1)
            * table_local_x.unsqueeze(1)
        )
        self.lateral_offset_velocity_w = torch.gradient(
            self.lateral_offset_w,
            spacing=(1.0 / float(fps),),
            dim=(0,),
            edge_order=2,
        )[0]

        self.agent_joint_pos = joint_pos.unsqueeze(1).expand(-1, self.num_agents, -1)
        self.agent_joint_vel = joint_vel.unsqueeze(1).expand(-1, self.num_agents, -1)
        self.agent_body_pos_w = body_pos_w.unsqueeze(1) + self.lateral_offset_w.unsqueeze(2)
        self.agent_body_quat_w = body_quat_w.unsqueeze(1).expand(-1, self.num_agents, -1, -1)
        self.agent_body_lin_vel_w = (
            body_lin_vel_w.unsqueeze(1) + self.lateral_offset_velocity_w.unsqueeze(2)
        )
        self.agent_body_ang_vel_w = body_ang_vel_w.unsqueeze(1).expand(
            -1, self.num_agents, -1, -1
        )
        self.object_pos_w = object_pos_w
        self.object_quat_w = object_quat_w
        self.object_lin_vel_w = object_lin_vel_w

    @classmethod
    def from_motion_loader(
        cls,
        motion: Any,
        *,
        lateral_spacing_m: float = 0.8,
    ) -> PairedA1Reference:
        """Build from the existing WBT ``MotionLoader`` public tensor interface."""
        if not motion.has_object:
            raise ValueError("Paired A1 reference requires one object trajectory")
        fps = int(torch.as_tensor(motion.fps).reshape(-1)[0].item())
        return cls(
            joint_pos=motion.joint_pos,
            joint_vel=motion.joint_vel,
            body_pos_w=motion.body_pos_w,
            body_quat_w=motion.body_quat_w,
            body_lin_vel_w=motion.body_lin_vel_w,
            body_ang_vel_w=motion.body_ang_vel_w,
            object_pos_w=motion.object_pos_w,
            object_quat_w=motion.object_quat_w,
            object_lin_vel_w=motion.object_lin_vel_w,
            fps=fps,
            lateral_spacing_m=lateral_spacing_m,
        )

    @staticmethod
    def _validate(**values: Any) -> None:
        joint_pos = values["joint_pos"]
        joint_vel = values["joint_vel"]
        body_pos_w = values["body_pos_w"]
        body_quat_w = values["body_quat_w"]
        body_lin_vel_w = values["body_lin_vel_w"]
        body_ang_vel_w = values["body_ang_vel_w"]
        object_pos_w = values["object_pos_w"]
        object_quat_w = values["object_quat_w"]
        object_lin_vel_w = values["object_lin_vel_w"]
        fps = values["fps"]
        lateral_spacing_m = values["lateral_spacing_m"]

        if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
            raise ValueError("joint_pos and joint_vel must have matching [frames, dofs] shapes")
        frames = joint_pos.shape[0]
        if frames < 3:
            raise ValueError("Paired A1 reference requires at least three frames")
        if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
            raise ValueError("body_pos_w must have shape [frames, bodies, 3]")
        if body_quat_w.shape != (*body_pos_w.shape[:-1], 4):
            raise ValueError("body_quat_w must match body_pos_w with quaternion dimension 4")
        if body_lin_vel_w.shape != body_pos_w.shape:
            raise ValueError("body_lin_vel_w must match body_pos_w")
        if body_ang_vel_w.shape != body_pos_w.shape:
            raise ValueError("body_ang_vel_w must match body_pos_w")
        if object_pos_w.shape != (frames, 3):
            raise ValueError("object_pos_w must have shape [frames, 3]")
        if object_quat_w.shape != (frames, 4):
            raise ValueError("object_quat_w must have shape [frames, 4]")
        if object_lin_vel_w.shape != (frames, 3):
            raise ValueError("object_lin_vel_w must have shape [frames, 3]")
        if body_pos_w.shape[0] != frames:
            raise ValueError("robot and body reference frame counts must match")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if lateral_spacing_m <= 0.0:
            raise ValueError("lateral_spacing_m must be positive")

    def sample(self, time_steps: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return synchronized agent and shared-object reference channels."""
        if time_steps.ndim != 1:
            raise ValueError("time_steps must be one-dimensional")
        if torch.any(time_steps < 0) or torch.any(time_steps >= self.num_frames):
            raise IndexError("time_steps contains an out-of-range frame")
        return {
            "agent_joint_pos": self.agent_joint_pos[time_steps],
            "agent_joint_vel": self.agent_joint_vel[time_steps],
            "agent_body_pos_w": self.agent_body_pos_w[time_steps],
            "agent_body_quat_w": self.agent_body_quat_w[time_steps],
            "agent_body_lin_vel_w": self.agent_body_lin_vel_w[time_steps],
            "agent_body_ang_vel_w": self.agent_body_ang_vel_w[time_steps],
            "object_pos_w": self.object_pos_w[time_steps],
            "object_quat_w": self.object_quat_w[time_steps],
            "object_lin_vel_w": self.object_lin_vel_w[time_steps],
        }
