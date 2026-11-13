"""Minimal online environment shell for the first Plan 5 Push A1 gate."""

from __future__ import annotations

import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


class Plan5PushManager(WholeBodyTrackingManager):
    """Connect two shared-policy actions to two physical rubber-hand G1 robots."""

    num_agents = 2

    def _policy_dim(self) -> int:
        """Per-robot policy action width: num_dof, or the action term's policy_dim (System1 T06/T07 hands)."""
        term = self.action_manager.get_term("joint_control") if "joint_control" in getattr(self.action_manager, "active_terms", ()) else None
        return int(getattr(term, "policy_dim", self.num_dof))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        width = self._policy_dim()
        expected_agent_shape = (self.num_envs, self.num_agents, width)
        expected_flat_shape = (self.num_envs, self.num_agents * width)
        if actions.shape == expected_agent_shape:
            flat_actions = actions.reshape(expected_flat_shape)
        elif actions.shape == expected_flat_shape:
            flat_actions = actions
        else:
            raise ValueError(
                "Plan 5 actions must have shape "
                f"{expected_agent_shape} or {expected_flat_shape}, got {tuple(actions.shape)}"
            )
        self.action_manager.process_actions(flat_actions)

    def reset_all(self):
        command = self.command_manager.get_state("paired_motion_command")
        if command is None:
            raise RuntimeError("Plan 5 environment requires paired_motion_command")
        command.time_steps.zero_()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_envs_idx(env_ids)
        self._refresh_envs_after_reset(env_ids)
        actions = torch.zeros(
            self.num_envs,
            self.num_agents,
            self._policy_dim(),
            device=self.device,
        )
        actor_state = {"actions": actions}
        obs_dict, _, _, _ = self.step(actor_state)
        return obs_dict

    def _refresh_envs_after_reset(self, env_ids):
        """Refresh states already written jointly by the paired command."""
        self.simulator.clear_contact_forces_history(env_ids)
        self.simulator.refresh_sim_tensors()
        self.need_to_refresh_envs[env_ids] = False
        self._pre_compute_observations_callback()

    def _update_log_dict(self):
        command = self.command_manager.get_state("paired_motion_command")
        if command is not None:
            command.update_metrics()
            self.log_dict.update(command.metrics)
