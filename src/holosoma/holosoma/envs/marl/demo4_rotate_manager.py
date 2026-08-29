"""Isolated cooperative environment for Demo 4 table rotation."""

from __future__ import annotations

from holosoma.envs.marl.plan5_push_manager import Plan5PushManager


class Demo4RotateManager(Plan5PushManager):
    """Run two shared-policy rubber-hand G1s around one physical table.

    The paired command owns reset placement and the two robot motion priors.
    After reset, the table is controlled only by simulator contacts.
    """

    def _pre_compute_observations_callback(self) -> None:
        super()._pre_compute_observations_callback()
        command = self.command_manager.get_state("paired_motion_command")
        if command is not None:
            command.update_yaw_progress()

    def _check_termination(self) -> None:
        """Give physical/success terminals priority over the horizon timeout."""

        super()._check_termination()
        results = self.termination_manager.term_results
        goal = results.get("yaw_goal_success")
        horizon = results.get("reference_horizon")
        robot_fall = results.get("clear_robot_fall")
        table_safety = results.get("table_physical_safety")
        if (
            goal is None
            or horizon is None
            or robot_fall is None
            or table_safety is None
        ):
            raise RuntimeError("Demo 4 termination contract is incomplete")

        unsafe = robot_fall | table_safety
        safe_goal = goal & ~unsafe
        results["yaw_goal_success"] = safe_goal
        self.extras["termination_terms"]["yaw_goal_success"] = safe_goal.clone()

        non_timeout_terminal = safe_goal | unsafe
        resolved_horizon = horizon & ~non_timeout_terminal
        results["reference_horizon"] = resolved_horizon
        self.extras["termination_terms"]["reference_horizon"] = (
            resolved_horizon.clone()
        )
        self.time_out_buf &= ~non_timeout_terminal
        self.reset_buf |= non_timeout_terminal.to(dtype=self.reset_buf.dtype)

    def _update_log_dict(self) -> None:
        command = self.command_manager.get_state("paired_motion_command")
        if command is not None:
            command.update_metrics()
            self.log_dict.update(command.metrics)


__all__ = ["Demo4RotateManager"]
