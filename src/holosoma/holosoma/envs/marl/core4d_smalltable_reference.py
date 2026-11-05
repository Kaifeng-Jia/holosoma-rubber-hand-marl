"""Strict runtime reference contract for the CORE4D small-table demo."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from holosoma.envs.marl.paired_motion_reference import PairedMotionReference
from holosoma.utils.file_cache import cached_open


class Core4DSmallTableReference(PairedMotionReference):
    """Extend the paired runtime with the table angular velocity channel."""

    _REQUIRED_KEYS = PairedMotionReference._REQUIRED_KEYS | {"object_ang_vel_w"}
    _SAMPLE_KEYS = (*PairedMotionReference._SAMPLE_KEYS, "object_ang_vel_w")

    def __init__(
        self,
        paired_reference_file: str,
        robot_body_names: Sequence[str],
        robot_joint_names: Sequence[str],
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            paired_reference_file,
            robot_body_names,
            robot_joint_names,
            device=device,
        )
        resolved_file = Path(self.motion_files[0])
        with cached_open(str(resolved_file), "rb") as file_handle, np.load(
            file_handle,
            allow_pickle=False,
        ) as data:
            angular_velocity = np.asarray(data["object_ang_vel_w"])
        expected = (self.num_frames, 3)
        if angular_velocity.shape != expected:
            raise ValueError(
                "object_ang_vel_w must have shape "
                f"{expected}, got {angular_velocity.shape}"
            )
        if not np.issubdtype(angular_velocity.dtype, np.number):
            raise ValueError("object_ang_vel_w must contain numeric values")
        if not np.all(np.isfinite(angular_velocity)):
            raise ValueError("object_ang_vel_w contains non-finite values")
        self.object_ang_vel_w = self._tensor(angular_velocity, device)


__all__ = ["Core4DSmallTableReference"]
