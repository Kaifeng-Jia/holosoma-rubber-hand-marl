"""Isolated physical environment for the CORE4D small-table demo."""

from __future__ import annotations

from holosoma.envs.marl.plan5_push_manager import Plan5PushManager


class Core4DSmallTableManager(Plan5PushManager):
    """Two rubber-hand G1 robots track one paired clip around one table."""

    def _check_termination(self) -> None:
        """Do not bootstrap a simultaneous physical tracking failure as timeout."""
        super()._check_termination()
        results = self.termination_manager.term_results
        horizon = results.get("reference_horizon")
        bad_tracking = results.get("joint_bad_tracking")
        if horizon is None or bad_tracking is None:
            raise RuntimeError("CORE4D small-table termination contract is incomplete")
        resolved_horizon = horizon & ~bad_tracking
        results["reference_horizon"] = resolved_horizon
        self.extras["termination_terms"]["reference_horizon"] = (
            resolved_horizon.clone()
        )
        self.time_out_buf &= ~bad_tracking


__all__ = ["Core4DSmallTableManager"]
