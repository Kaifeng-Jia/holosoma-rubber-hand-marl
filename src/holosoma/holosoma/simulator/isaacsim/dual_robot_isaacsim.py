"""Isaac Sim adapter for two homogeneous articulations in every environment."""

from __future__ import annotations

import copy

import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from holosoma.simulator.isaacsim.isaacsim import IsaacSim
from holosoma.simulator.isaacsim.proxy_utils import RootStatesProxy


class DualRobotIsaacSim(IsaacSim):
    """Keep the legacy primary robot while adding a second physical G1."""

    num_agents = 2

    def _setup_additional_robot_articulations(
        self,
        robot_articulation_config: ArticulationCfg,
        contact_sensor_config: ContactSensorCfg,
    ) -> None:
        secondary_spawn = copy.deepcopy(robot_articulation_config.spawn)
        if secondary_spawn is not None and hasattr(secondary_spawn, "force_usd_conversion"):
            secondary_spawn = secondary_spawn.replace(force_usd_conversion=False)

        secondary_position = list(self.robot_config.init_state.pos)
        secondary_position[0] += 0.8
        secondary_init_state = robot_articulation_config.init_state.replace(
            pos=tuple(secondary_position)
        )
        secondary_config = robot_articulation_config.replace(
            prim_path="/World/envs/env_.*/Robot_1",
            spawn=secondary_spawn,
            init_state=secondary_init_state,
        )
        self._robot_1 = Articulation(secondary_config)
        self.scene.articulations["robot_1"] = self._robot_1

        secondary_contact_config = contact_sensor_config.replace(
            prim_path="/World/envs/env_.*/Robot_1/.*"
        )
        self._robot_1_contact_sensor = ContactSensor(secondary_contact_config)
        self.scene.sensors["robot_1_contact_sensor"] = self._robot_1_contact_sensor

    def _robot_prim_path_expressions(self) -> list[str]:
        return [
            "/World/envs/env_.*/Robot",
            "/World/envs/env_.*/Robot_1",
        ]

    def load_assets(self) -> None:
        super().load_assets()
        self._robot_1_dof_ids, secondary_dof_names = self._robot_1.find_joints(
            self.robot_config.dof_names,
            preserve_order=True,
        )
        self._robot_1_body_ids, secondary_body_names = self._robot_1.find_bodies(
            self.robot_config.body_names,
            preserve_order=True,
        )
        if secondary_dof_names != self.dof_names:
            raise RuntimeError("Secondary robot joint order differs from the primary robot")
        if secondary_body_names != self.body_names:
            raise RuntimeError("Secondary robot body order differs from the primary robot")

    def refresh_sim_tensors(self) -> None:
        super().refresh_sim_tensors()
        if not hasattr(self, "_robot_1_dof_ids"):
            return

        secondary_root_wxyz = self._robot_1.data.root_state_w
        if hasattr(self, "robot_1_root_states"):
            self.robot_1_root_states.reset(secondary_root_wxyz)
        else:
            self.robot_1_root_states = RootStatesProxy(secondary_root_wxyz)

        secondary_dof_pos = self._robot_1.data.joint_pos[:, self._robot_1_dof_ids]
        secondary_dof_vel = self._robot_1.data.joint_vel[:, self._robot_1_dof_ids]
        secondary_body_pos = self._robot_1.data.body_pos_w[:, self._robot_1_body_ids]
        secondary_body_rot = self._robot_1.data.body_quat_w[:, self._robot_1_body_ids][
            :, :, [1, 2, 3, 0]
        ]
        secondary_body_vel = self._robot_1.data.body_lin_vel_w[:, self._robot_1_body_ids]
        secondary_body_ang_vel = self._robot_1.data.body_ang_vel_w[:, self._robot_1_body_ids]
        secondary_contact = self._robot_1_contact_sensor.data.net_forces_w[
            :, self._contact_to_robot_body_ids
        ]

        control_decimation = self.simulator_config.sim.control_decimation
        history_length = self.simulator_config.contact_sensor_history_length
        effective_history_length = min(control_decimation, history_length)
        if not hasattr(self, "_robot_1_contact_forces_history"):
            self._robot_1_contact_forces_history = torch.zeros(
                self.num_envs,
                history_length,
                self.num_bodies,
                3,
                device=self.device,
            )
        self._robot_1_contact_forces_history[:, :effective_history_length] = (
            self._robot_1_contact_sensor.data.net_forces_w_history[
                :, :effective_history_length, self._contact_to_robot_body_ids
            ]
        )

        self.agent_root_states = torch.stack(
            (self.robot_root_states.tensor_xyzw, self.robot_1_root_states.tensor_xyzw),
            dim=1,
        )
        self.agent_dof_pos = torch.stack((self.dof_pos, secondary_dof_pos), dim=1)
        self.agent_dof_vel = torch.stack((self.dof_vel, secondary_dof_vel), dim=1)
        self.agent_rigid_body_pos = torch.stack((self._rigid_body_pos, secondary_body_pos), dim=1)
        self.agent_rigid_body_rot = torch.stack((self._rigid_body_rot, secondary_body_rot), dim=1)
        self.agent_rigid_body_vel = torch.stack((self._rigid_body_vel, secondary_body_vel), dim=1)
        self.agent_rigid_body_ang_vel = torch.stack(
            (self._rigid_body_ang_vel, secondary_body_ang_vel),
            dim=1,
        )
        self.agent_contact_forces = torch.stack((self.contact_forces, secondary_contact), dim=1)
        self.agent_contact_forces_history = torch.stack(
            (self.contact_forces_history, self._robot_1_contact_forces_history),
            dim=1,
        )

    def get_agent_dof_control_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the latest per-agent joint state for low-level control.

        ``agent_dof_pos`` and ``agent_dof_vel`` are control-step snapshots used
        by observations, rewards, and the critic.  A PD controller instead
        needs the state refreshed by Isaac Lab after every physics substep.
        Reading the articulation buffers here keeps those two timing contracts
        separate while preserving the public ``[env, agent, dof]`` layout.
        """
        primary_dof_pos = self._robot.data.joint_pos[:, self.dof_ids]
        primary_dof_vel = self._robot.data.joint_vel[:, self.dof_ids]
        secondary_dof_pos = self._robot_1.data.joint_pos[:, self._robot_1_dof_ids]
        secondary_dof_vel = self._robot_1.data.joint_vel[:, self._robot_1_dof_ids]
        return (
            torch.stack((primary_dof_pos, secondary_dof_pos), dim=1),
            torch.stack((primary_dof_vel, secondary_dof_vel), dim=1),
        )

    def apply_agent_torques(self, torques: torch.Tensor) -> None:
        expected = (self.num_envs, self.num_agents, self.num_dof)
        if torques.shape != expected:
            raise ValueError(f"Agent torques must have shape {expected}, got {tuple(torques.shape)}")
        self._robot.set_joint_effort_target(torques[:, 0], joint_ids=self.dof_ids)
        self._robot_1.set_joint_effort_target(
            torques[:, 1],
            joint_ids=self._robot_1_dof_ids,
        )

    def set_agent_root_states(self, env_ids: torch.Tensor, root_states: torch.Tensor) -> None:
        expected = (len(env_ids), self.num_agents, 13)
        if root_states.shape != expected:
            raise ValueError(
                f"Agent root states must have shape {expected}, got {tuple(root_states.shape)}"
            )
        for agent_index, (robot, proxy) in enumerate(
            (
                (self._robot, self.robot_root_states),
                (self._robot_1, self.robot_1_root_states),
            )
        ):
            proxy[env_ids] = root_states[:, agent_index]
            states_wxyz = proxy._get_wxyz(env_ids)
            robot.write_root_pose_to_sim(states_wxyz[:, :7], env_ids)
            robot.write_root_velocity_to_sim(states_wxyz[:, 7:], env_ids)

    def set_agent_dof_states(
        self,
        env_ids: torch.Tensor,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
    ) -> None:
        expected = (len(env_ids), self.num_agents, self.num_dof)
        if dof_pos.shape != expected or dof_vel.shape != expected:
            raise ValueError(
                "Agent DOF states must both have shape "
                f"{expected}, got {tuple(dof_pos.shape)} and {tuple(dof_vel.shape)}"
            )
        self._robot.write_joint_state_to_sim(
            dof_pos[:, 0],
            dof_vel[:, 0],
            self.dof_ids,
            env_ids,
        )
        self._robot_1.write_joint_state_to_sim(
            dof_pos[:, 1],
            dof_vel[:, 1],
            self._robot_1_dof_ids,
            env_ids,
        )

    def clear_contact_forces_history(self, env_ids: torch.Tensor) -> None:
        super().clear_contact_forces_history(env_ids)
        if len(env_ids) > 0 and hasattr(self, "_robot_1_contact_forces_history"):
            self._robot_1_contact_forces_history[env_ids] = 0.0
