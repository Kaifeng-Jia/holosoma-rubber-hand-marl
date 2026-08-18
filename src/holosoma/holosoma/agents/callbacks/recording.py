"""Eval callback that records per-step trajectory data to an NPZ file.

Besides the post-step robot state used by ``viser_eval_viewer.py``, WBT
evaluations record the pre-step reference/physical state.  The latter is
important because the environment resets a failed environment inside
``env.step``; reading only the post-step tensors would otherwise replace the
failure state with the next attempt's initial state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import RecordingConfig
from holosoma.utils.safe_torch_import import torch


class EvalRecordingCallback(RLEvalCallback):
    """Records per-step data during evaluation and saves to .npz on completion."""

    def __init__(
        self,
        config: RecordingConfig,
        training_loop: Any = None,
    ):
        super().__init__(config, training_loop)
        self.env_id = config.env_id

        output_path = config.output_path
        if not output_path.endswith(".npz"):
            output_path += ".npz"
        if training_loop is not None and hasattr(training_loop, "log_dir"):
            output_path = str(Path(training_loop.log_dir) / output_path)
        self.output_path = output_path

        self._buffers: dict[str, list[np.ndarray]] = {}
        self._metadata: dict[str, Any] = {}
        self._step_count = 0
        self._pre_step_count = 0

    def _get_env(self):
        """Get the unwrapped BaseTask environment."""
        return self.training_loop._unwrap_env()

    def _save(self) -> None:
        """Save recorded data to NPZ."""
        if self._step_count == 0:
            return

        arrays: dict[str, np.ndarray] = {}
        for name, values in self._buffers.items():
            if values:
                arrays[name] = np.stack(values, axis=0)

        arrays["_metadata_json"] = np.array(json.dumps(self._metadata))

        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **arrays)

        channel_summary = ", ".join(
            f"{name}{list(arr.shape)}" for name, arr in arrays.items() if name != "_metadata_json"
        )
        logger.info(f"EvalRecordingCallback: saved {self._step_count} steps to {path}\n  Channels: {channel_summary}")

    def on_pre_evaluate_policy(self) -> None:
        env = self._get_env()
        sim = env.simulator

        self._metadata["dt"] = float(env.dt)
        self._metadata["fps"] = round(1.0 / float(env.dt))
        self._metadata["sim_dt"] = float(env.sim_dt)
        self._metadata["sim_fps"] = round(1.0 / float(env.sim_dt))
        self._metadata["control_decimation"] = env.simulator.simulator_config.sim.control_decimation
        self._metadata["env_id"] = self.env_id
        actor_obs_keys = list(getattr(self.training_loop, "actor_obs_keys", []))
        if actor_obs_keys:
            self._metadata["actor_obs_keys"] = actor_obs_keys
            self._buffers["policy_actor_obs"] = []
        if hasattr(sim, "dof_names"):
            self._metadata["dof_names"] = list(sim.dof_names)
        if hasattr(sim, "body_names"):
            self._metadata["body_names"] = list(sim.body_names)

        # Static robot properties
        robot_cfg = env.robot_config
        self._metadata["effort_limits"] = list(robot_cfg.dof_effort_limit_list)
        self._metadata["dof_pos_lower_limits"] = list(robot_cfg.dof_pos_lower_limit_list)
        self._metadata["dof_pos_upper_limits"] = list(robot_cfg.dof_pos_upper_limit_list)
        self._metadata["velocity_limits"] = list(robot_cfg.dof_vel_limit_list)
        asset_cfg = robot_cfg.asset
        self._metadata["urdf_path"] = str(Path(asset_cfg.asset_root) / asset_cfg.urdf_file)

        object_asset = getattr(sim, "_object", None)
        if object_asset is not None:
            object_physx_view = object_asset.root_physx_view

            def _selected_property(value: torch.Tensor) -> Any:
                return value[self.env_id].detach().cpu().numpy().tolist()

            self._metadata["object_physics"] = {
                "object_urdf_path": robot_cfg.object.object_urdf_path,
                "mass_kg": _selected_property(object_physx_view.get_masses()),
                "inertia_kg_m2": _selected_property(object_physx_view.get_inertias()),
                "com_pose_b": _selected_property(object_physx_view.get_coms()),
                "material_properties": _selected_property(object_physx_view.get_material_properties()),
                "material_property_order": [
                    "static_friction",
                    "dynamic_friction",
                    "restitution",
                ],
            }

        for sensor_name in ("object_robot_contact_sensor", "object_hand_contact_sensor"):
            sensor = getattr(sim, sensor_name, None)
            if sensor is None:
                continue
            channel_prefix = sensor_name.removesuffix("_sensor")
            self._metadata[channel_prefix] = {
                "sensor_body_names": list(sensor.body_names),
                "filter_prim_paths_expr": list(sensor.cfg.filter_prim_paths_expr),
                "force_semantics": (
                    "World-frame contact force on the object sensor body from the filtered robot bodies."
                ),
                "contact_point_semantics": (
                    "World-frame aggregate contact point for each configured filter; NaN means no contact."
                ),
            }
            self._buffers[f"{channel_prefix}_force_matrix_w"] = []
            self._buffers[f"{channel_prefix}_force_matrix_history_w"] = []
            contact_position_channel = f"{channel_prefix.removesuffix('_contact')}_contact_pos_w"
            self._buffers[contact_position_channel] = []

        channel_names = [
            "dof_pos_target",
            "dof_pos",
            "dof_vel",
            "torques",
            "torques_substep",
            "dof_pos_substep",
            "dof_vel_substep",
            "actions",
            "root_pos",
            "root_quat_xyzw",
            "root_lin_vel",
            "root_ang_vel",
            "body_pos_w",
            "body_quat_xyzw",
            "commanded_velocity",
        ]
        for name in channel_names:
            self._buffers[name] = []

        if hasattr(env, "teammate_relative_position_b") and hasattr(env, "teammate_relative_velocity_b"):
            self._metadata["teammate_provider_class"] = type(env).__name__
            self._metadata["teammate_observer_side"] = getattr(env, "observer_side", None)
            self._metadata["teammate_lateral_spacing_m"] = getattr(env, "lateral_spacing_m", None)
            self._metadata["teammate_observation_frame"] = "observing robot yaw frame"
            self._metadata["teammate_observation_semantics"] = (
                "Planar relative position in meters and planar relative velocity in meters per second."
            )
            self._buffers["teammate_relative_position_b"] = []
            self._buffers["teammate_relative_velocity_b"] = []

        motion_command = self._get_motion_command(env)
        if motion_command is not None:
            tracked_body_names = list(motion_command.motion_cfg.body_names_to_track)
            self._metadata["tracked_body_names"] = tracked_body_names
            self._metadata["tracked_body_indexes"] = [
                int(index) for index in motion_command.tracked_body_indexes.detach().cpu().tolist()
            ]
            self._metadata["motion_fps"] = int(motion_command.motion.fps)
            self._metadata["motion_time_step_total"] = int(motion_command.motion.time_step_total)
            self._metadata["motion_has_object"] = bool(motion_command.motion.has_object)
            self._metadata["motion_files"] = list(motion_command.motion.motion_files)
            self._metadata["motion_names"] = list(motion_command.motion.motion_names)
            self._metadata["motion_sampling_probabilities"] = [
                float(value)
                for value in motion_command.motion_sampling_probabilities.detach().cpu().tolist()
            ]
            self._metadata["motion_sampling_weights_explicit"] = bool(
                motion_command.has_explicit_motion_sampling_weights
            )
            self._metadata["contact_force_semantics"] = (
                "Net external force on each contact-sensor body. Rubber-hand links are recorded "
                "directly when present. These are body-level net forces, not pairwise hand-object "
                "contact reports."
            )
            if hasattr(sim, "contact_sensor") and hasattr(sim.contact_sensor, "body_names"):
                self._metadata["contact_sensor_body_names"] = list(sim.contact_sensor.body_names)

            pre_step_channels = [
                "motion_time_step",
                "motion_id",
                "episode_step",
                "ref_joint_pos",
                "ref_joint_vel",
                "pre_dof_pos",
                "pre_dof_vel",
                "ref_body_pos_w",
                "ref_body_quat_xyzw",
                "pre_tracked_body_pos_w",
                "pre_tracked_body_quat_xyzw",
                "ref_root_pos_w",
                "ref_root_quat_xyzw",
                "pre_root_pos",
                "pre_root_quat_xyzw",
                "contact_forces_w",
                "contact_forces_history_w",
            ]
            if hasattr(sim, "contact_sensor"):
                pre_step_channels.extend(
                    [
                        "contact_sensor_forces_w",
                        "contact_sensor_forces_history_w",
                    ]
                )
            if motion_command.motion.has_object:
                pre_step_channels.extend(
                    [
                        "ref_object_pos_w",
                        "ref_object_quat_xyzw",
                        "ref_object_lin_vel_w",
                        "object_pos_w",
                        "object_quat_xyzw",
                        "object_lin_vel_w",
                        "object_ang_vel_w",
                    ]
                )
            for name in pre_step_channels:
                self._buffers[name] = []

        for name in ("reward", "done", "timeout", "terminated"):
            self._buffers[name] = []

        logger.info(f"EvalRecordingCallback: recording env_id={self.env_id}, output={self.output_path}")

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        """Record the state used to compute the current action.

        PPO calls the pre-step hook once before entering the evaluation loop.
        That warm-up call has no ``step`` key and is deliberately ignored.
        """
        if "step" not in actor_state:
            return actor_state

        env = self._get_env()
        motion_command = self._get_motion_command(env)
        if motion_command is None:
            return actor_state

        sim = env.simulator
        eid = self.env_id

        def _to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().copy()

        def _append(name: str, value: torch.Tensor) -> None:
            self._buffers[name].append(_to_np(value))

        if "policy_actor_obs" in self._buffers:
            actor_obs = torch.cat(
                [actor_state["obs"][key] for key in self._metadata["actor_obs_keys"]],
                dim=1,
            )
            _append("policy_actor_obs", actor_obs[eid])

        _append("motion_time_step", motion_command.time_steps[eid])
        _append("motion_id", motion_command.motion_ids[eid])
        _append("episode_step", env.episode_length_buf[eid])
        _append("ref_joint_pos", motion_command.joint_pos[eid])
        _append("ref_joint_vel", motion_command.joint_vel[eid])
        _append("pre_dof_pos", sim.dof_pos[eid])
        _append("pre_dof_vel", sim.dof_vel[eid])
        _append("ref_body_pos_w", motion_command.body_pos_relative_w[eid])
        _append("ref_body_quat_xyzw", motion_command.body_quat_relative_w[eid])
        _append("pre_tracked_body_pos_w", motion_command.robot_body_pos_w[eid])
        _append("pre_tracked_body_quat_xyzw", motion_command.robot_body_quat_w[eid])
        _append("ref_root_pos_w", motion_command.root_pos_w[eid])
        _append("ref_root_quat_xyzw", motion_command.root_quat_w[eid])
        _append("pre_root_pos", sim.robot_root_states[eid, :3])
        _append("pre_root_quat_xyzw", sim.robot_root_states[eid, 3:7])
        if "teammate_relative_position_b" in self._buffers:
            _append("teammate_relative_position_b", env.teammate_relative_position_b[eid])
            _append("teammate_relative_velocity_b", env.teammate_relative_velocity_b[eid])
        _append("contact_forces_w", sim.contact_forces[eid])
        _append("contact_forces_history_w", sim.contact_forces_history[eid])
        if hasattr(sim, "contact_sensor"):
            _append("contact_sensor_forces_w", sim.contact_sensor.data.net_forces_w[eid])
            _append("contact_sensor_forces_history_w", sim.contact_sensor.data.net_forces_w_history[eid])
        for sensor_name in ("object_robot_contact_sensor", "object_hand_contact_sensor"):
            sensor = getattr(sim, sensor_name, None)
            if sensor is None:
                continue
            channel_prefix = sensor_name.removesuffix("_sensor")
            sensor_data = sensor.data
            if sensor_data.force_matrix_w is None or sensor_data.force_matrix_w_history is None:
                raise RuntimeError(f"{sensor_name} did not initialize filtered force matrices")
            if sensor_data.contact_pos_w is None:
                raise RuntimeError(f"{sensor_name} did not initialize filtered contact positions")
            _append(f"{channel_prefix}_force_matrix_w", sensor_data.force_matrix_w[eid])
            _append(
                f"{channel_prefix}_force_matrix_history_w",
                sensor_data.force_matrix_w_history[eid],
            )
            contact_position_channel = f"{channel_prefix.removesuffix('_contact')}_contact_pos_w"
            _append(contact_position_channel, sensor_data.contact_pos_w[eid])

        if motion_command.motion.has_object:
            _append("ref_object_pos_w", motion_command.object_pos_w[eid])
            _append("ref_object_quat_xyzw", motion_command.object_quat_w[eid])
            _append("ref_object_lin_vel_w", motion_command.object_lin_vel_w[eid])
            _append("object_pos_w", motion_command.simulator_object_pos_w[eid])
            _append("object_quat_xyzw", motion_command.simulator_object_quat_w[eid])
            _append("object_lin_vel_w", motion_command.simulator_object_lin_vel_w[eid])
            object_states = sim.all_root_states[motion_command.object_indices_in_simulator]
            _append("object_ang_vel_w", object_states[eid, 10:13])

        self._pre_step_count += 1
        return actor_state

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        env = self._get_env()
        sim = env.simulator
        eid = self.env_id

        def _to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().copy()

        self._buffers["dof_pos"].append(_to_np(sim.dof_pos[eid]))  # post_eval_env_step, so after 4 decimation
        self._buffers["dof_vel"].append(_to_np(sim.dof_vel[eid]))
        self._buffers["torques"].append(
            _to_np(self._extract_torques(env, eid))
        )  # pre_eval_env_step, so the torques is the last decimation

        # robot_root_states: [num_envs, 13] = pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)
        root = sim.robot_root_states[eid]
        self._buffers["root_pos"].append(_to_np(root[:3]))
        self._buffers["root_quat_xyzw"].append(_to_np(root[3:7]))
        self._buffers["root_lin_vel"].append(_to_np(root[7:10]))
        self._buffers["root_ang_vel"].append(_to_np(root[10:13]))

        self._buffers["body_pos_w"].append(_to_np(sim._rigid_body_pos[eid]))
        self._buffers["body_quat_xyzw"].append(_to_np(sim._rigid_body_rot[eid]))

        # substep tensors: [decimation, num_dof] — one row per physics sub-step
        torques_substep, dof_pos_substep, dof_vel_substep = self._extract_substep_data(env, eid)
        self._buffers["torques_substep"].append(_to_np(torques_substep))
        self._buffers["dof_pos_substep"].append(_to_np(dof_pos_substep))
        self._buffers["dof_vel_substep"].append(_to_np(dof_vel_substep))

        if "actions" in actor_state and actor_state["actions"] is not None:
            self._buffers["actions"].append(_to_np(actor_state["actions"][eid]))

        # Record desired target joint positions (PD setpoint)
        self._buffers["dof_pos_target"].append(_to_np(self._extract_dof_pos_target(env, eid)))

        # Record commanded velocity [lin_vel_x, lin_vel_y, ang_vel_yaw]
        if hasattr(env, "command_manager") and env.command_manager is not None:
            try:
                self._buffers["commanded_velocity"].append(_to_np(env.command_manager.commands[eid]))
            except (AttributeError, IndexError):
                pass

        reward = actor_state["rewards"][eid]
        done = actor_state["dones"][eid].to(dtype=torch.bool)
        timeouts = actor_state.get("extras", {}).get("time_outs")
        timeout = (
            timeouts[eid].to(dtype=torch.bool)
            if isinstance(timeouts, torch.Tensor)
            else torch.zeros((), dtype=torch.bool, device=done.device)
        )
        self._buffers["reward"].append(_to_np(reward))
        self._buffers["done"].append(_to_np(done))
        self._buffers["timeout"].append(_to_np(timeout))
        self._buffers["terminated"].append(_to_np(done & ~timeout))

        self._step_count += 1
        return actor_state

    @staticmethod
    def _get_motion_command(env: Any) -> Any | None:
        if not hasattr(env, "command_manager") or env.command_manager is None:
            return None
        try:
            return env.command_manager.get_state("motion_command")
        except (KeyError, AttributeError):
            return None

    def _extract_dof_pos_target(self, env: Any, env_id: int) -> torch.Tensor:
        """Extract desired target joint positions from the action manager's joint control term.

        The PD target is: actions_after_delay * action_scales + default_dof_pos.
        Returns shape [num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "_actions_after_delay") and hasattr(term, "action_scales"):
                return term._actions_after_delay[env_id] * term.action_scales + env.default_dof_pos[env_id]
        raise RuntimeError("No action term with _actions_after_delay found")

    def _extract_torques(self, env: Any, env_id: int) -> torch.Tensor:
        """Extract torques from the action manager's joint control term.

        Returns torques, shape [num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "torques"):
                return term.torques[env_id]
        raise RuntimeError("No action term with torques found")

    def _extract_substep_data(self, env: Any, env_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract sub-step torques, dof_pos, and dof_vel from the action manager's joint control term.

        Returns (torques_substep, dof_pos_substep, dof_vel_substep), each shape [decimation, num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "torques_substep"):
                return term.torques_substep[env_id], term.dof_pos_substep[env_id], term.dof_vel_substep[env_id]
        raise RuntimeError("No action term with torques_substep found")

    def on_post_evaluate_policy(self) -> None:
        if self._pre_step_count not in (0, self._step_count):
            raise RuntimeError(
                "EvalRecordingCallback recorded mismatched pre/post step counts: "
                f"{self._pre_step_count} pre-step states versus {self._step_count} transitions"
            )
        self._save()
