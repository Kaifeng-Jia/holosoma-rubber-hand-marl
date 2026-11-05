"""Non-looping paired command for the CORE4D small-table demo."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.envs.marl.core4d_smalltable_reference import (
    Core4DSmallTableReference,
)
from holosoma.managers.command.terms.marl import PairedA1MotionCommand
from holosoma.managers.command.terms.wbt import FAKE_BODY_NAME_ALIASES
from holosoma.utils.rotations import quat_apply


def object_origin_velocity_to_com_velocity(
    origin_velocity_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    com_position_b: torch.Tensor,
) -> torch.Tensor:
    """Convert a rigid actor-origin velocity to its center-of-mass velocity."""
    origin_to_com_w = quat_apply(object_quat_w, com_position_b, w_last=True)
    return origin_velocity_w + torch.cross(
        angular_velocity_w,
        origin_to_com_w,
        dim=-1,
    )


def object_com_velocity_to_origin_velocity(
    com_velocity_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    com_position_b: torch.Tensor,
) -> torch.Tensor:
    """Convert an Isaac rigid-body COM velocity to actor-origin velocity."""
    origin_to_com_w = quat_apply(object_quat_w, com_position_b, w_last=True)
    return com_velocity_w - torch.cross(
        angular_velocity_w,
        origin_to_com_w,
        dim=-1,
    )


class Core4DSmallTableMotionCommand(PairedA1MotionCommand):
    """Track two robot references and one physical table until the final frame.

    Unlike the legacy paired command, the terminal reference frame is clamped
    and an explicit termination term ends the episode.  The reference is never
    looped or written into physics after reset.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        if self.paired_reference_file is None:
            raise ValueError("CORE4D small-table command requires an explicit runtime")

    def setup(self) -> None:
        super().setup()
        simulator = self.env.simulator
        body_aliases = [
            FAKE_BODY_NAME_ALIASES.get(name, name)
            for name in simulator._body_list
        ]
        self.motion = Core4DSmallTableReference(
            str(self.paired_reference_file),
            body_aliases,
            simulator.dof_names,
            device=self.device,
        )
        self.reference = self.motion
        self.object_com_position_b = (
            simulator._object.root_physx_view.get_coms()[:, :3]
            .to(device=self.device)
            .clone()
        )
        self.table_reset_write_count = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

    def _write_reference_state(self, env_ids: torch.Tensor) -> None:
        sample = self.reference.sample(self.time_steps[env_ids])
        origins = self.env.simulator.scene.env_origins[env_ids]

        root_states = torch.zeros(
            len(env_ids),
            self.num_agents,
            13,
            device=self.device,
            dtype=sample["agent_body_pos_w"].dtype,
        )
        root_states[..., :3] = sample["agent_body_pos_w"][:, :, 0] + origins[:, None, :]
        root_states[..., 3:7] = sample["agent_body_quat_w"][:, :, 0]
        root_states[..., 7:10] = sample["agent_body_lin_vel_w"][:, :, 0]
        root_states[..., 10:13] = sample["agent_body_ang_vel_w"][:, :, 0]
        self.env.simulator.set_agent_root_states(env_ids, root_states)
        self.env.simulator.set_agent_dof_states(
            env_ids,
            sample["agent_joint_pos"],
            sample["agent_joint_vel"],
        )

        object_states = torch.zeros(
            len(env_ids),
            13,
            device=self.device,
            dtype=root_states.dtype,
        )
        object_states[:, :3] = sample["object_pos_w"] + origins
        object_states[:, 3:7] = sample["object_quat_w"]
        object_states[:, 7:10] = object_origin_velocity_to_com_velocity(
            sample["object_lin_vel_w"],
            sample["object_ang_vel_w"],
            sample["object_quat_w"],
            self.object_com_position_b[env_ids],
        )
        object_states[:, 10:13] = sample["object_ang_vel_w"]
        self.env.simulator.set_actor_states(
            [self.object_name],
            env_ids,
            object_states,
        )
        self.table_reset_write_count[env_ids] += 1

    def step(self) -> None:
        """Advance once and hold the final frame for horizon termination."""
        self.time_steps.add_(1)
        self.time_steps.clamp_(max=self.reference.num_frames - 1)

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        """Return velocity at the same actor origin used by the reference."""
        states = self.env.simulator.all_root_states[self.object_indices_in_simulator]
        return object_com_velocity_to_origin_velocity(
            states[:, 7:10],
            states[:, 10:13],
            states[:, 3:7],
            self.object_com_position_b,
        )


__all__ = [
    "Core4DSmallTableMotionCommand",
    "object_com_velocity_to_origin_velocity",
    "object_origin_velocity_to_com_velocity",
]
