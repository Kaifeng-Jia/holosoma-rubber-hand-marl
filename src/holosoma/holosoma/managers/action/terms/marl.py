"""Action terms for homogeneous multi-agent control."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from holosoma.managers.action.base import ActionTermBase

if TYPE_CHECKING:
    from holosoma.config_types.action import ActionTermCfg


class DualJointPositionActionTerm(ActionTermBase):
    """Route two 29-DoF actor outputs to two physical robots.

    The public action layout is ``[env, agent-major action]``. Internally all
    controller state is kept as ``[env, agent, dof]`` so that no operation can
    accidentally mix the two robots.
    """

    num_agents = 2

    def __init__(self, cfg: ActionTermCfg, env: Any):
        super().__init__(cfg, env)
        self._per_agent_action_dim = env.num_dof
        self._action_dim = self.num_agents * self._per_agent_action_dim
        flat_shape = (env.num_envs, self._action_dim)
        agent_shape = (env.num_envs, self.num_agents, self._per_agent_action_dim)

        self._raw_actions = torch.zeros(flat_shape, device=env.device)
        self._processed_actions = torch.zeros(flat_shape, device=env.device)
        self._actions_after_delay = torch.zeros(agent_shape, device=env.device)
        self.torques = torch.zeros(agent_shape, device=env.device)
        self._prev_dof_vel = torch.zeros(agent_shape, device=env.device)

        self._kp_scale = torch.ones(agent_shape, device=env.device)
        self._kd_scale = torch.ones(agent_shape, device=env.device)
        self._rfi_lim_scale = torch.ones(agent_shape, device=env.device)
        self._rfi_lim = 0.0
        self._randomize_torque_rfi = False

        self.p_gains = torch.zeros(self._per_agent_action_dim, device=env.device)
        self.d_gains = torch.zeros_like(self.p_gains)
        self.i_gains = torch.zeros_like(self.p_gains)
        self.action_scales = torch.zeros_like(self.p_gains)
        self._configure_pd_gains(env)
        self._configure_action_scales(env)

        # Preserve the established environment-level controller interface.
        env.p_gains = self.p_gains
        env.d_gains = self.d_gains
        env.i_gains = self.i_gains
        env.action_scales = self.action_scales

        self.action_queue: torch.Tensor | None = None
        self._substep_idx = 0

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def per_agent_action_dim(self) -> int:
        return self._per_agent_action_dim

    @property
    def agent_actions(self) -> torch.Tensor:
        """Return processed actions as ``[env, agent, dof]``."""
        assert self._processed_actions is not None
        return self._processed_actions.view(
            self.env.num_envs, self.num_agents, self._per_agent_action_dim
        )

    def setup(self) -> None:
        super().setup()
        if getattr(self.env, "_randomize_ctrl_delay", False):
            max_delay = self.env._ctrl_delay_step_range[1]
            self.action_queue = torch.zeros(
                self.env.num_envs,
                max_delay + 1,
                self.num_agents,
                self._per_agent_action_dim,
                device=self.env.device,
            )

        decimation = self.env.simulator.simulator_config.sim.control_decimation
        history_shape = (
            self.env.num_envs,
            decimation,
            self.num_agents,
            self._per_agent_action_dim,
        )
        self.torques_substep = torch.zeros(history_shape, device=self.env.device)
        self.dof_pos_substep = torch.zeros(history_shape, device=self.env.device)
        self.dof_vel_substep = torch.zeros(history_shape, device=self.env.device)

        self._attach_actuator_randomizer_scales()
        enabled, rfi_lim = self.env._pending_torque_rfi
        self.configure_torque_rfi(enabled=enabled, rfi_lim=rfi_lim)
        self.env._pending_torque_rfi = (False, 0.0)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._substep_idx = 0
        assert self._raw_actions is not None
        assert self._processed_actions is not None
        self._raw_actions[:] = actions

        if self.env.robot_config.control.clip_actions:
            clip_limit = self.env.robot_config.control.action_clip_value
            self._processed_actions[:] = torch.clip(actions, -clip_limit, clip_limit)
            self.env.log_dict["action_clip_frac"] = (
                self._processed_actions.abs() == clip_limit
            ).sum() / self._processed_actions.numel()
        else:
            self._processed_actions[:] = actions
            self.env.log_dict["action_clip_frac"] = torch.tensor(0.0, device=self.env.device)

        agent_actions = self.agent_actions
        if getattr(self.env, "_randomize_ctrl_delay", False):
            self._apply_action_delay(agent_actions)
        else:
            self._actions_after_delay[:] = agent_actions

    def _apply_action_delay(self, actions: torch.Tensor) -> None:
        assert self.action_queue is not None, "action_queue must be initialized in setup()"
        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = actions
        self._actions_after_delay[:] = self.action_queue[
            torch.arange(self.env.num_envs, device=self.env.device), self.env.action_delay_idx
        ]

    def apply_actions(self) -> None:
        dof_pos, dof_vel = self.env.simulator.get_agent_dof_control_state()
        self.torques[:] = self._compute_torques(
            self._actions_after_delay,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
        )
        self.torques_substep[:, self._substep_idx] = self.torques
        self.dof_pos_substep[:, self._substep_idx] = dof_pos
        self.dof_vel_substep[:, self._substep_idx] = dof_vel
        self._substep_idx += 1
        self.env.simulator.apply_agent_torques(self.torques)
        self._prev_dof_vel.copy_(dof_vel)

    def _compute_torques(
        self,
        actions: torch.Tensor,
        *,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
    ) -> torch.Tensor:
        actions_scaled = actions * self.action_scales
        control_type = self.env.robot_config.control.control_type

        if control_type == "P":
            default_dof_pos = self.env.default_dof_pos
            if default_dof_pos.ndim == 2:
                default_dof_pos = default_dof_pos.unsqueeze(1)
            torques = (
                self._kp_scale
                * self.p_gains
                * (actions_scaled + default_dof_pos - dof_pos)
                - self._kd_scale * self.d_gains * dof_vel
            )
        elif control_type == "V":
            torques = (
                self._kp_scale
                * self.p_gains
                * (actions_scaled - dof_vel)
                - self._kd_scale
                * self.d_gains
                * (dof_vel - self._prev_dof_vel)
                / self.env.sim_dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise ValueError(f"Unknown controller type: {control_type}")

        if self._randomize_torque_rfi:
            torques = torques + (
                (torch.rand_like(torques) * 2.0 - 1.0)
                * self._rfi_lim
                * self._rfi_lim_scale
                * self.env.torque_limits
            )
        if self.env.robot_config.control.clip_torques:
            torques = torch.clip(torques, -self.env.torque_limits, self.env.torque_limits)
        return torques

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        buffers = [self._actions_after_delay, self.torques, self._prev_dof_vel]
        if env_ids is None:
            for buffer in buffers:
                buffer.zero_()
            if self.action_queue is not None:
                self.action_queue.zero_()
        else:
            for buffer in buffers:
                buffer[env_ids] = 0.0
            if self.action_queue is not None:
                self.action_queue[env_ids] = 0.0

    def attach_actuator_scales(
        self, kp_scale: torch.Tensor, kd_scale: torch.Tensor, rfi_lim_scale: torch.Tensor
    ) -> None:
        self._kp_scale = self._as_agent_scale(kp_scale)
        self._kd_scale = self._as_agent_scale(kd_scale)
        self._rfi_lim_scale = self._as_agent_scale(rfi_lim_scale)

    def update_pd_scales(
        self, env_ids: torch.Tensor, kp_values: torch.Tensor, kd_values: torch.Tensor
    ) -> None:
        self._kp_scale[env_ids] = self._as_agent_scale(kp_values, len(env_ids))
        self._kd_scale[env_ids] = self._as_agent_scale(kd_values, len(env_ids))

    def update_rfi_scales(self, env_ids: torch.Tensor, rfi_values: torch.Tensor) -> None:
        self._rfi_lim_scale[env_ids] = self._as_agent_scale(rfi_values, len(env_ids))

    def configure_torque_rfi(self, *, enabled: bool, rfi_lim: float | None = None) -> None:
        self._randomize_torque_rfi = enabled
        if rfi_lim is not None:
            self._rfi_lim = float(rfi_lim)

    def get_pd_scale_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._kp_scale, self._kd_scale

    def get_rfi_scale_tensor(self) -> torch.Tensor:
        return self._rfi_lim_scale

    def get_prev_dof_vel(self) -> torch.Tensor:
        return self._prev_dof_vel

    def _as_agent_scale(self, value: torch.Tensor, num_envs: int | None = None) -> torch.Tensor:
        if value.ndim == 3:
            return value
        if value.ndim != 2:
            raise ValueError(f"Expected actuator scale [env, dof] or [env, agent, dof], got {value.shape}")
        expected_envs = self.env.num_envs if num_envs is None else num_envs
        if value.shape != (expected_envs, self._per_agent_action_dim):
            raise ValueError(
                f"Expected actuator scale [{expected_envs}, {self._per_agent_action_dim}], got {value.shape}"
            )
        return value.unsqueeze(1).expand(-1, self.num_agents, -1).clone()

    def _attach_actuator_randomizer_scales(self) -> None:
        rand_manager = getattr(self.env, "randomization_manager", None)
        get_state = getattr(rand_manager, "get_state", None)
        if not callable(get_state):
            return
        state = get_state("actuator_randomizer_state")
        if state is not None:
            self.attach_actuator_scales(
                state.kp_scale_tensor,
                state.kd_scale_tensor,
                state.rfi_lim_scale_tensor,
            )

    def _configure_pd_gains(self, env: Any) -> None:
        control_cfg = env.robot_config.control
        for i, name in enumerate(env.dof_names):
            if name not in env.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            matched = False
            for dof_name, stiffness in control_cfg.stiffness.items():
                if dof_name in name:
                    self.p_gains[i] = stiffness
                    self.d_gains[i] = control_cfg.damping[dof_name]
                    self.i_gains[i] = getattr(control_cfg, "integral", {}).get(dof_name, 0.0)
                    matched = True
            if not matched and control_cfg.control_type in ["P", "V"]:
                raise ValueError(f"PD gains for joint '{name}' were not defined.")

    def _configure_action_scales(self, env: Any) -> None:
        control_cfg = env.robot_config.control
        if control_cfg.action_scales_by_effort_limit_over_p_gain:
            for i, effort in enumerate(env.robot_config.dof_effort_limit_list):
                stiffness = self.p_gains[i]
                self.action_scales[i] = 0.0 if stiffness == 0.0 else control_cfg.action_scale * effort / stiffness
        else:
            self.action_scales[:] = control_cfg.action_scale
