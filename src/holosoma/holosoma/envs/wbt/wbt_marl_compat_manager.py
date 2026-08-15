"""Single-robot WBT environments with explicit ghost-teammate interfaces."""

from __future__ import annotations

import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.utils.rotations import quat_error_magnitude, quat_rotate, quat_rotate_inverse, yaw_quat


_STAGE1B_DIAGNOSTIC_FLOAT_CHANNEL_SHAPES = {
    "stage1b_ref_pos_z_error_m": (),
    "stage1b_ref_ori_error": (),
    "stage1b_key_body_z_error_m": (4,),
    "stage1b_object_pos_error_m": (),
    "stage1b_object_pos_error_w_m": (3,),
    "stage1b_object_ori_error_rad": (),
    "stage1b_object_yaw_error_rad": (),
    "stage1b_ref_object_pos_w_m": (3,),
    "stage1b_object_pos_w_m": (3,),
    "stage1b_rubber_hand_origin_pos_w_m": (2, 3),
    "stage1b_rubber_hand_net_contact_force_w_n": (2, 3),
    "stage1b_table_force_proxy_w_n": (2, 3),
    "stage1b_table_yaw_moment_proxy_nm": (2,),
    "stage1b_table_total_yaw_moment_proxy_nm": (),
}

_STAGE1B_DIAGNOSTIC_BOOLEAN_CHANNELS = (
    "stage1b_bad_ref_pos",
    "stage1b_bad_ref_ori",
    "stage1b_bad_motion_body_pos",
    "stage1b_bad_object_pos",
    "stage1b_bad_object_ori",
    "stage1b_bad_tracking",
    "stage1b_timeout",
    "stage1b_reset",
)

_WBT_ACTOR_OBS_DIM = 154


def _mirror_joint_tensor(
    values: torch.Tensor,
    joint_index_map: torch.Tensor,
    sign_flip_mask: torch.Tensor,
) -> torch.Tensor:
    """Mirror G1 joint-space values using the robot configuration contract."""
    if values.shape[-1] != len(joint_index_map) or joint_index_map.shape != sign_flip_mask.shape:
        raise ValueError("Joint tensor and symmetry mapping dimensions do not match")
    return values[..., joint_index_map] * sign_flip_mask


def _mirror_rotation_6d_xz(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Mirror the WBT first-two-column rotation representation across the robot XZ plane."""
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected rotation_6d final dimension 6, got {tuple(rotation_6d.shape)}")
    # WBT encodes quaternion_to_matrix(...)[..., :2], which slices the
    # matrix's final dimension and therefore stores a row-major [3, 2] block.
    first_two_columns = rotation_6d.reshape(*rotation_6d.shape[:-1], 3, 2)
    third_column = torch.linalg.cross(
        first_two_columns[..., :, 0],
        first_two_columns[..., :, 1],
        dim=-1,
    )
    rotation = torch.cat((first_two_columns, third_column.unsqueeze(-1)), dim=-1)
    reflection = rotation.new_tensor([1.0, -1.0, 1.0])
    mirrored = reflection.unsqueeze(-1) * rotation * reflection.unsqueeze(-2)
    return mirrored[..., :2].reshape(*rotation_6d.shape[:-1], 6)


def _mirror_wbt_actor_observation(
    observation: torch.Tensor,
    joint_index_map: torch.Tensor,
    sign_flip_mask: torch.Tensor,
) -> torch.Tensor:
    """Map the 154-D WBT actor observation into its sagittal mirror."""
    if observation.shape[-1] != _WBT_ACTOR_OBS_DIM:
        raise ValueError(f"Expected {_WBT_ACTOR_OBS_DIM}-D WBT actor observation, got {observation.shape[-1]}")
    mirrored = observation.clone()
    # ObservationManager concatenates terms alphabetically:
    # actions, base_ang_vel, dof_pos, dof_vel, motion_command, motion_ref_ori_b.
    mirrored[..., 0:29] = _mirror_joint_tensor(observation[..., 0:29], joint_index_map, sign_flip_mask)
    mirrored[..., 29:32] = observation[..., 29:32] * observation.new_tensor([-1.0, 1.0, -1.0])
    mirrored[..., 32:61] = _mirror_joint_tensor(observation[..., 32:61], joint_index_map, sign_flip_mask)
    mirrored[..., 61:90] = _mirror_joint_tensor(observation[..., 61:90], joint_index_map, sign_flip_mask)
    mirrored[..., 90:119] = _mirror_joint_tensor(observation[..., 90:119], joint_index_map, sign_flip_mask)
    mirrored[..., 119:148] = _mirror_joint_tensor(observation[..., 119:148], joint_index_map, sign_flip_mask)
    mirrored[..., 148:154] = _mirror_rotation_6d_xz(observation[..., 148:154])
    return mirrored


def _mirror_teammate_observation(observation: torch.Tensor) -> torch.Tensor:
    """Mirror planar teammate position/velocity in the observer's yaw frame."""
    if observation.shape[-1] != 4:
        raise ValueError(f"Expected 4-D teammate observation, got {observation.shape[-1]}")
    return observation * observation.new_tensor([1.0, -1.0, 1.0, -1.0])


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


def _yaw_from_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return world yaw for normalized xyzw quaternions."""
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm <= 0.0):
        raise ValueError("Quaternion contains a zero-norm value")
    x, y, z, w = (quaternion / norm).unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _table_contact_moment_proxy(
    hand_origin_pos_w: torch.Tensor,
    hand_net_contact_force_w: torch.Tensor,
    table_com_pos_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate force and yaw moment applied to the table from hand net force.

    The contact sensor reports the net force on each hand body, not a
    hand-table pairwise force. The opposite force is used as a table-force
    proxy and the hand rigid-body origin is used as a contact-point proxy.
    """
    if hand_origin_pos_w.shape != hand_net_contact_force_w.shape:
        raise ValueError("Hand position and force tensors must have the same shape")
    if hand_origin_pos_w.ndim != 3 or hand_origin_pos_w.shape[-2:] != (2, 3):
        raise ValueError(f"Expected hand tensors shaped [N, 2, 3], got {tuple(hand_origin_pos_w.shape)}")
    if table_com_pos_w.shape != (hand_origin_pos_w.shape[0], 3):
        raise ValueError(f"Expected table COM shaped [N, 3], got {tuple(table_com_pos_w.shape)}")

    table_force_proxy_w = -hand_net_contact_force_w
    lever_arm_w = hand_origin_pos_w - table_com_pos_w.unsqueeze(1)
    yaw_moment_proxy = torch.linalg.cross(lever_arm_w, table_force_proxy_w, dim=-1)[..., 2]
    return table_force_proxy_w, yaw_moment_proxy


def _bad_tracking_term(env: WholeBodyTrackingManager):
    """Return the configured WBT bad-tracking term used by Stage 1B."""
    term = env.termination_manager._term_instances.get("bad_tracking")
    if term is None:
        raise RuntimeError("Stage 1B diagnostics require the stateful 'bad_tracking' termination term")
    required = (
        "bad_ref_pos_threshold",
        "bad_ref_ori_threshold",
        "bad_motion_body_pos_threshold",
        "bad_object_pos_threshold",
        "bad_object_ori_threshold",
        "bad_motion_body_pos_body_indexes",
        "bad_motion_body_pos_body_names",
    )
    missing = [name for name in required if not hasattr(term, name)]
    if missing:
        raise RuntimeError(f"Stage 1B bad-tracking term is missing diagnostic fields: {missing}")
    return term


def _compute_stage1b_termination_diagnostics(env: WholeBodyTrackingManager) -> dict[str, torch.Tensor]:
    """Recompute the exact WBT threshold operands before an in-step reset.

    This function deliberately mirrors ``BadTrackingZOnly`` without changing
    the termination result.  It exposes the individual scalar/vector errors
    that are otherwise collapsed into one ``bad_tracking`` boolean.
    """
    term = _bad_tracking_term(env)
    motion_command = env.command_manager.get_state("motion_command")
    if motion_command is None:
        raise RuntimeError("Stage 1B diagnostics require motion_command")
    if not motion_command.motion.has_object:
        raise RuntimeError("Stage 1B diagnostics require an object reference")

    ref_pos_z_error = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
    gravity_w = gravity_vector(env)
    motion_projected_gravity_b = quat_rotate_inverse(motion_command.ref_quat_w, gravity_w, w_last=True)
    robot_projected_gravity_b = quat_rotate_inverse(motion_command.robot_ref_quat_w, gravity_w, w_last=True)
    ref_ori_error = torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2])

    body_idx = term.bad_motion_body_pos_body_indexes
    key_body_z_error = torch.abs(
        motion_command.body_pos_relative_w[:, body_idx, -1]
        - motion_command.robot_body_pos_w[:, body_idx, -1]
    )
    object_pos_error = torch.linalg.vector_norm(
        motion_command.object_pos_w - motion_command.simulator_object_pos_w,
        dim=-1,
    )
    object_pos_error_w = motion_command.simulator_object_pos_w - motion_command.object_pos_w
    object_ori_error = quat_error_magnitude(
        motion_command.object_quat_w,
        motion_command.simulator_object_quat_w,
    )
    object_yaw_error = _yaw_from_xyzw(motion_command.simulator_object_quat_w) - _yaw_from_xyzw(
        motion_command.object_quat_w
    )
    object_yaw_error = torch.atan2(torch.sin(object_yaw_error), torch.cos(object_yaw_error))

    wrist_pos_w = motion_command.robot_body_pos_w[:, env.stage1b_wrist_tracked_indexes]
    wrist_quat_w = motion_command.robot_body_quat_w[:, env.stage1b_wrist_tracked_indexes]
    hand_offset_w = quat_rotate(
        wrist_quat_w.reshape(-1, 4),
        env.stage1b_hand_origin_offset_b.expand(env.num_envs, -1, -1).reshape(-1, 3),
        w_last=True,
    ).reshape(env.num_envs, 2, 3)
    hand_origin_pos_w = wrist_pos_w + hand_offset_w

    contact_sensor = env.simulator.contact_sensor
    hand_net_contact_force_w = contact_sensor.data.net_forces_w[:, env.stage1b_rubber_hand_sensor_indexes]
    table_force_proxy_w, table_yaw_moment_proxy = _table_contact_moment_proxy(
        hand_origin_pos_w,
        hand_net_contact_force_w,
        motion_command.simulator_object_pos_w,
    )

    bad_ref_pos = ref_pos_z_error > term.bad_ref_pos_threshold
    bad_ref_ori = ref_ori_error > term.bad_ref_ori_threshold
    bad_motion_body_pos = torch.any(key_body_z_error > term.bad_motion_body_pos_threshold, dim=-1)
    bad_object_pos = object_pos_error > term.bad_object_pos_threshold
    bad_object_ori = object_ori_error > term.bad_object_ori_threshold

    return {
        "stage1b_ref_pos_z_error_m": ref_pos_z_error,
        "stage1b_ref_ori_error": ref_ori_error,
        "stage1b_key_body_z_error_m": key_body_z_error,
        "stage1b_object_pos_error_m": object_pos_error,
        "stage1b_object_pos_error_w_m": object_pos_error_w,
        "stage1b_object_ori_error_rad": object_ori_error,
        "stage1b_object_yaw_error_rad": object_yaw_error,
        "stage1b_ref_object_pos_w_m": motion_command.object_pos_w.clone(),
        "stage1b_object_pos_w_m": motion_command.simulator_object_pos_w.clone(),
        "stage1b_rubber_hand_origin_pos_w_m": hand_origin_pos_w.clone(),
        "stage1b_rubber_hand_net_contact_force_w_n": hand_net_contact_force_w.clone(),
        "stage1b_table_force_proxy_w_n": table_force_proxy_w,
        "stage1b_table_yaw_moment_proxy_nm": table_yaw_moment_proxy,
        "stage1b_table_total_yaw_moment_proxy_nm": torch.sum(table_yaw_moment_proxy, dim=-1),
        "stage1b_bad_ref_pos": bad_ref_pos,
        "stage1b_bad_ref_ori": bad_ref_ori,
        "stage1b_bad_motion_body_pos": bad_motion_body_pos,
        "stage1b_bad_object_pos": bad_object_pos,
        "stage1b_bad_object_ori": bad_object_ori,
        "stage1b_bad_tracking": env.termination_manager.terminated.clone(),
        "stage1b_timeout": env.time_out_buf.to(dtype=torch.bool).clone(),
        "stage1b_reset": env.reset_buf.to(dtype=torch.bool).clone(),
    }


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

        term = _bad_tracking_term(self)
        if not hasattr(self.simulator, "contact_sensor"):
            raise RuntimeError("Stage 1B contact diagnostics require the Isaac Sim contact sensor")
        hand_names = ["left_rubber_hand_link", "right_rubber_hand_link"]
        sensor_body_names = list(self.simulator.contact_sensor.body_names)
        missing_hand_names = [name for name in hand_names if name not in sensor_body_names]
        if missing_hand_names:
            raise RuntimeError(f"Stage 1B contact sensor is missing rubber-hand bodies: {missing_hand_names}")
        self.stage1b_rubber_hand_sensor_indexes = torch.tensor(
            [sensor_body_names.index(name) for name in hand_names],
            dtype=torch.long,
            device=self.device,
        )
        wrist_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
        missing_wrist_names = [name for name in wrist_names if name not in term.body_names_to_track]
        if missing_wrist_names:
            raise RuntimeError(f"Stage 1B tracked bodies are missing wrist links: {missing_wrist_names}")
        self.stage1b_wrist_tracked_indexes = torch.tensor(
            [term.body_names_to_track.index(name) for name in wrist_names],
            dtype=torch.long,
            device=self.device,
        )
        # Fixed joints in main_mesh_collision_rubberhand.urdf:
        # wrist yaw -> rubber hand origin, with zero relative rotation.
        self.stage1b_hand_origin_offset_b = torch.tensor(
            [[0.0415, 0.003, 0.0], [0.0415, -0.003, 0.0]],
            dtype=torch.float,
            device=self.device,
        )
        self.stage1b_termination_diagnostic_metadata = {
            "semantics": "Post-physics WBT termination operands captured before reset.",
            "key_body_names": list(term.bad_motion_body_pos_body_names),
            "rubber_hand_body_names": hand_names,
            "rubber_hand_origin_semantics": (
                "Derived exactly from measured wrist-yaw poses and the fixed URDF joint offsets "
                "[0.0415, +0.003, 0] m (left), [0.0415, -0.003, 0] m (right)."
            ),
            "contact_moment_proxy_semantics": (
                "Comparison-only proxy: the opposite of each rubber-hand body's net external contact force "
                "is treated as force on the table, and the rubber-hand rigid-body origin is used instead of "
                "a pairwise hand-table contact point. It is not an exact pairwise contact moment."
            ),
            "thresholds": {
                "ref_pos_z_error_m": float(term.bad_ref_pos_threshold),
                "ref_ori_error": float(term.bad_ref_ori_threshold),
                "key_body_z_error_m": float(term.bad_motion_body_pos_threshold),
                "object_pos_error_m": float(term.bad_object_pos_threshold),
                "object_ori_error_rad": float(term.bad_object_ori_threshold),
            },
        }
        float_shapes = dict(_STAGE1B_DIAGNOSTIC_FLOAT_CHANNEL_SHAPES)
        float_shapes["stage1b_key_body_z_error_m"] = (len(term.bad_motion_body_pos_body_names),)
        self.stage1b_termination_diagnostics = {
            name: torch.zeros((self.num_envs, *shape), dtype=torch.float, device=self.device)
            for name, shape in float_shapes.items()
        }
        self.stage1b_termination_diagnostics.update(
            {
                name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                for name in _STAGE1B_DIAGNOSTIC_BOOLEAN_CHANNELS
            }
        )

    def _check_termination(self) -> None:
        super()._check_termination()
        # BaseTask resets terminated environments later in the same step. Keep
        # this snapshot alive so the post-step recorder sees the causal state.
        self.stage1b_termination_diagnostics = _compute_stage1b_termination_diagnostics(self)

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


class MirroredLeftTrajectoryGhostTeammateWholeBodyTrackingManager(
    LeftTrajectoryGhostTeammateWholeBodyTrackingManager
):
    """Run the frozen positive-side A1 actor through a sagittal policy-I/O mirror."""

    def _init_buffers(self) -> None:
        super()._init_buffers()
        dof_names = list(self.robot_config.dof_names)
        if len(dof_names) != 29:
            raise ValueError(f"Stage 1B A1 policy mirror requires 29 G1 DOFs, got {len(dof_names)}")
        name_to_index = {name: index for index, name in enumerate(dof_names)}
        joint_index_map = []
        for name in dof_names:
            mapped_name = self.robot_config.symmetry_joint_names.get(name)
            if mapped_name is None or mapped_name not in name_to_index:
                raise ValueError(f"Missing G1 symmetry partner for joint {name!r}")
            joint_index_map.append(name_to_index[mapped_name])
        flip_names = set(self.robot_config.flip_sign_joint_names)
        self.stage1b_policy_mirror_joint_index_map = torch.tensor(
            joint_index_map,
            dtype=torch.long,
            device=self.device,
        )
        self.stage1b_policy_mirror_sign_flip_mask = torch.tensor(
            [-1.0 if name in flip_names else 1.0 for name in dof_names],
            dtype=torch.float,
            device=self.device,
        )
        self.stage1b_policy_mirror_metadata = {
            "semantics": (
                "Left physical observations are mapped into the frozen A1 actor's positive-side canonical space; "
                "the actor output is mapped back into left physical joint space."
            ),
            "actor_observation_dim": _WBT_ACTOR_OBS_DIM,
            "teammate_observation_dim": 4,
            "joint_names": dof_names,
        }

    def _mirror_policy_observation_dict(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "actor_obs" not in observations or "teammate_obs" not in observations:
            raise KeyError("Stage 1B policy mirror requires actor_obs and teammate_obs groups")
        observations["actor_obs"] = _mirror_wbt_actor_observation(
            observations["actor_obs"],
            self.stage1b_policy_mirror_joint_index_map,
            self.stage1b_policy_mirror_sign_flip_mask,
        )
        observations["teammate_obs"] = _mirror_teammate_observation(observations["teammate_obs"])
        return observations

    def _compute_observations(self) -> None:
        super()._compute_observations()
        self._mirror_policy_observation_dict(self.obs_buf_dict)

    def _compute_final_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._compute_final_observations()
        return self._mirror_policy_observation_dict(observations)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        physical_actions = _mirror_joint_tensor(
            actions,
            self.stage1b_policy_mirror_joint_index_map,
            self.stage1b_policy_mirror_sign_flip_mask,
        )
        super()._pre_physics_step(physical_actions)


class LeftSpacing070TrajectoryGhostTeammateWholeBodyTrackingManager(
    LeftTrajectoryGhostTeammateWholeBodyTrackingManager
):
    """Stage 1B spacing candidate: left observer with a 0.7 m teammate separation."""

    lateral_spacing_m = 0.7


class RightSpacing070TrajectoryGhostTeammateWholeBodyTrackingManager(
    RightTrajectoryGhostTeammateWholeBodyTrackingManager
):
    """Stage 1B spacing candidate: right observer with a 0.7 m teammate separation."""

    lateral_spacing_m = 0.7
