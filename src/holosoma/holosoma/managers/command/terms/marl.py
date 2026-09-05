"""Command state for Plan 5 homogeneous multi-agent motion tracking."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.config_types.command import MotionConfig
from holosoma.envs.marl import PairedA1Reference, PairedMotionReference
from holosoma.managers.command.base import CommandTermBase
from holosoma.managers.command.terms.wbt import FAKE_BODY_NAME_ALIASES, MotionLoader
from holosoma.utils.rotations import quat_apply, quat_inverse, quat_mul, yaw_quat


class PairedA1MotionCommand(CommandTermBase):
    """Own one phase clock for two robot references and one object reference.

    This first-stage command deliberately performs exact reference resets. Pose
    noise and adaptive phase sampling belong to later training gates and are not
    mixed into the initial dual-entity physics validation.
    """

    num_agents = 2

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        motion_cfg = cfg.params["motion_config"]
        self.motion_cfg = motion_cfg if isinstance(motion_cfg, MotionConfig) else MotionConfig(**motion_cfg)
        self.lateral_spacing_m = float(cfg.params.get("lateral_spacing_m", 0.8))
        self.paired_reference_file = cfg.params.get("paired_reference_file")
        if self.paired_reference_file is not None and not str(self.paired_reference_file).strip():
            raise ValueError("paired_reference_file must be a non-empty path when provided")
        if self.motion_cfg.motion_dir or self.motion_cfg.motion_files:
            raise ValueError("Paired A1 command accepts exactly one frozen motion_file")
        if self.motion_cfg.noise_to_initial_pose.overall_noise_scale != 0.0:
            raise ValueError("Initial paired physics gate requires exact reset with zero pose noise")

    def setup(self) -> None:
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        simulator = self.env.simulator
        robot_body_names = simulator._body_list
        robot_body_aliases = [FAKE_BODY_NAME_ALIASES.get(name, name) for name in robot_body_names]

        if self.paired_reference_file is not None:
            self.motion = PairedMotionReference(
                str(self.paired_reference_file),
                robot_body_aliases,
                simulator.dof_names,
                device=self.device,
            )
            # Explicit paired files already contain both robot placements.  In
            # particular, do not apply the legacy A1 lateral offset a second time.
            self.reference = self.motion
        else:
            self.motion = MotionLoader(
                self.motion_cfg.motion_file,
                robot_body_aliases,
                simulator.dof_names,
                device=self.device,
            )
            if not self.motion.has_object:
                raise ValueError("Paired A1 command requires a motion with one object reference")
            self.reference = PairedA1Reference.from_motion_loader(
                self.motion,
                lateral_spacing_m=self.lateral_spacing_m,
            )
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])
        self.tracked_body_indexes = torch.tensor(
            [robot_body_names.index(name) for name in self.motion_cfg.body_names_to_track],
            dtype=torch.long,
            device=self.device,
        )
        self.object_name = "object"
        self.object_indices_in_simulator = simulator.get_actor_indices(self.object_name, env_ids=None)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics: dict[str, torch.Tensor] = {}

    def reset(self, env_ids: torch.Tensor | None) -> None:
        env_ids = self._ensure_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        if self.env.is_evaluating or self.motion_cfg.start_at_timestep_zero_prob >= 1.0:
            sampled_frames = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        else:
            sampled_frames = torch.randint(
                0,
                self.reference.num_frames - 1,
                (len(env_ids),),
                device=self.device,
            )
            start_prob = self.motion_cfg.start_at_timestep_zero_prob
            if start_prob > 0.0:
                choose_start = torch.rand(len(env_ids), device=self.device) < start_prob
                sampled_frames[choose_start] = 0
        self.time_steps[env_ids] = sampled_frames
        self._write_reference_state(env_ids)

    def step(self) -> None:
        self.time_steps += 1
        ended_env_ids = torch.where(self.time_steps >= self.reference.num_frames)[0]
        if ended_env_ids.numel() > 0:
            self.reset(ended_env_ids)

    def _write_reference_state(self, env_ids: torch.Tensor) -> None:
        sample = self.reference.sample(self.time_steps[env_ids])
        origins = self.env.simulator.scene.env_origins[env_ids]

        root_states = torch.zeros(
            len(env_ids), self.num_agents, 13, device=self.device, dtype=sample["agent_body_pos_w"].dtype
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

        object_states = torch.zeros(len(env_ids), 13, device=self.device, dtype=root_states.dtype)
        object_states[:, :3] = sample["object_pos_w"] + origins
        object_states[:, 3:7] = sample["object_quat_w"]
        object_states[:, 7:10] = sample["object_lin_vel_w"]
        self.env.simulator.set_actor_states([self.object_name], env_ids, object_states)

    def _ensure_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _sample(self) -> dict[str, torch.Tensor]:
        return self.reference.sample(self.time_steps)

    @property
    def command(self) -> torch.Tensor:
        sample = self._sample()
        return torch.cat((sample["agent_joint_pos"], sample["agent_joint_vel"]), dim=-1)

    @property
    def agent_joint_pos(self) -> torch.Tensor:
        return self._sample()["agent_joint_pos"]

    @property
    def agent_joint_vel(self) -> torch.Tensor:
        return self._sample()["agent_joint_vel"]

    @property
    def agent_body_pos_w(self) -> torch.Tensor:
        values = self._sample()["agent_body_pos_w"][:, :, self.tracked_body_indexes]
        return values + self.env.simulator.scene.env_origins[:, None, None, :]

    @property
    def agent_body_quat_w(self) -> torch.Tensor:
        return self._sample()["agent_body_quat_w"][:, :, self.tracked_body_indexes]

    @property
    def agent_body_pos_relative_w(self) -> torch.Tensor:
        """Reference bodies aligned to each simulated robot's planar heading.

        This is the paired-agent equivalent of ``MotionCommand.body_pos_relative_w``.
        It preserves the original WBT reward semantics independently for each robot.
        """
        ref_pos = self.agent_ref_pos_w[:, :, None, :]
        ref_quat = self.agent_ref_quat_w[:, :, None, :]
        robot_ref_pos = self.env.simulator.agent_rigid_body_pos[:, :, self.ref_body_index, None, :]
        robot_ref_quat = self.env.simulator.agent_rigid_body_rot[:, :, self.ref_body_index, None, :]
        delta_quat = yaw_quat(
            quat_mul(robot_ref_quat, quat_inverse(ref_quat, w_last=True), w_last=True),
            w_last=True,
        )
        delta_quat = delta_quat.expand(-1, -1, self.agent_body_pos_w.shape[2], -1)
        height_delta = ref_pos - robot_ref_pos
        height_delta = height_delta.clone()
        height_delta[..., :2] = 0.0
        return (
            robot_ref_pos
            + height_delta
            + quat_apply(delta_quat, self.agent_body_pos_w - ref_pos, w_last=True)
        )

    @property
    def agent_body_quat_relative_w(self) -> torch.Tensor:
        """Reference body orientations aligned to each robot's planar heading."""
        ref_quat = self.agent_ref_quat_w[:, :, None, :]
        robot_ref_quat = self.env.simulator.agent_rigid_body_rot[:, :, self.ref_body_index, None, :]
        delta_quat = yaw_quat(
            quat_mul(robot_ref_quat, quat_inverse(ref_quat, w_last=True), w_last=True),
            w_last=True,
        )
        delta_quat = delta_quat.expand(-1, -1, self.agent_body_quat_w.shape[2], -1)
        return quat_mul(delta_quat, self.agent_body_quat_w, w_last=True)

    @property
    def agent_body_lin_vel_w(self) -> torch.Tensor:
        return self._sample()["agent_body_lin_vel_w"][:, :, self.tracked_body_indexes]

    @property
    def agent_body_ang_vel_w(self) -> torch.Tensor:
        return self._sample()["agent_body_ang_vel_w"][:, :, self.tracked_body_indexes]

    @property
    def agent_ref_pos_w(self) -> torch.Tensor:
        values = self._sample()["agent_body_pos_w"][:, :, self.ref_body_index]
        return values + self.env.simulator.scene.env_origins[:, None, :]

    @property
    def agent_ref_quat_w(self) -> torch.Tensor:
        return self._sample()["agent_body_quat_w"][:, :, self.ref_body_index]

    @property
    def agent_root_pos_w(self) -> torch.Tensor:
        values = self._sample()["agent_body_pos_w"][:, :, 0]
        return values + self.env.simulator.scene.env_origins[:, None, :]

    @property
    def agent_root_quat_w(self) -> torch.Tensor:
        return self._sample()["agent_body_quat_w"][:, :, 0]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._sample()["object_pos_w"] + self.env.simulator.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._sample()["object_quat_w"]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._sample()["object_lin_vel_w"]

    @property
    def simulator_agent_joint_pos(self) -> torch.Tensor:
        return self.env.simulator.agent_dof_pos

    @property
    def simulator_agent_joint_vel(self) -> torch.Tensor:
        return self.env.simulator.agent_dof_vel

    @property
    def simulator_agent_body_pos_w(self) -> torch.Tensor:
        return self.env.simulator.agent_rigid_body_pos[:, :, self.tracked_body_indexes]

    @property
    def simulator_agent_body_quat_w(self) -> torch.Tensor:
        return self.env.simulator.agent_rigid_body_rot[:, :, self.tracked_body_indexes]

    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self.env.simulator.all_root_states[self.object_indices_in_simulator][:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self.env.simulator.all_root_states[self.object_indices_in_simulator][:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self.env.simulator.all_root_states[self.object_indices_in_simulator][:, 7:10]

    def update_metrics(self) -> None:
        agent_ref_error = torch.linalg.vector_norm(
            self.agent_ref_pos_w
            - self.env.simulator.agent_rigid_body_pos[:, :, self.ref_body_index],
            dim=-1,
        )
        self.metrics["motion/error_ref_pos_mean"] = agent_ref_error.mean(dim=1)
        self.metrics["motion/error_ref_pos_max"] = agent_ref_error.max(dim=1).values
        self.metrics["motion/error_object_pos"] = torch.linalg.vector_norm(
            self.object_pos_w - self.simulator_object_pos_w,
            dim=-1,
        )
