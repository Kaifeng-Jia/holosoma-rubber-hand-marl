"""Strict loader for an explicit two-agent motion reference NPZ."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from holosoma.utils.file_cache import cached_open
from holosoma.utils.path import resolve_data_file_path


class PairedMotionReference:
    """Load two robot references and one shared object reference.

    The file contract is deliberately separate from the five-channel ViSER
    rollout format.  It contains the complete body and velocity channels used
    by WBT rewards and termination terms.  Quaternions are stored as WXYZ at
    the NPZ boundary, matching the existing Holosoma motion-file convention,
    and converted to the simulator's XYZW convention while loading.
    """

    num_agents = 2
    _REQUIRED_KEYS = {
        "fps",
        "joint_names",
        "body_names",
        "agent_joint_pos",
        "agent_joint_vel",
        "agent_body_pos_w",
        "agent_body_quat_w",
        "agent_body_lin_vel_w",
        "agent_body_ang_vel_w",
        "object_pos_w",
        "object_quat_w",
        "object_lin_vel_w",
    }
    _SAMPLE_KEYS = (
        "agent_joint_pos",
        "agent_joint_vel",
        "agent_body_pos_w",
        "agent_body_quat_w",
        "agent_body_lin_vel_w",
        "agent_body_ang_vel_w",
        "object_pos_w",
        "object_quat_w",
        "object_lin_vel_w",
    )

    def __init__(
        self,
        paired_reference_file: str,
        robot_body_names: Sequence[str],
        robot_joint_names: Sequence[str],
        device: str | torch.device = "cpu",
    ) -> None:
        resolved_file = resolve_data_file_path(str(paired_reference_file))
        self.motion_files = [resolved_file]
        self.motion_names = [Path(resolved_file).stem]
        self.has_object = True

        with cached_open(resolved_file, "rb") as file_handle, np.load(
            file_handle, allow_pickle=False
        ) as data:
            missing = self._REQUIRED_KEYS - set(data.files)
            if missing:
                raise ValueError(
                    f"Explicit paired reference '{resolved_file}' is missing keys: "
                    f"{sorted(missing)}"
                )

            body_names = self._read_names(data["body_names"], "body_names")
            joint_names = self._read_names(data["joint_names"], "joint_names")
            body_indexes = self._indexes_for(
                robot_body_names,
                body_names,
                "body",
                device=device,
                allow_requested_duplicates=True,
            )
            joint_indexes = self._indexes_for(
                robot_joint_names, joint_names, "joint", device=device
            )

            fps = self._read_fps(data["fps"])
            arrays = {key: np.asarray(data[key]) for key in self._SAMPLE_KEYS}
            self._validate_arrays(
                arrays,
                body_count=len(body_names),
                joint_count=len(joint_names),
            )

            self.fps = fps
            self.time_step_total = int(arrays["agent_joint_pos"].shape[0])
            self.num_frames = self.time_step_total

            self.agent_joint_pos = self._tensor(
                arrays["agent_joint_pos"], device
            ).index_select(2, joint_indexes)
            self.agent_joint_vel = self._tensor(
                arrays["agent_joint_vel"], device
            ).index_select(2, joint_indexes)
            self.agent_body_pos_w = self._tensor(
                arrays["agent_body_pos_w"], device
            ).index_select(2, body_indexes)
            self.agent_body_quat_w = self._wxyz_to_xyzw(
                self._tensor(arrays["agent_body_quat_w"], device)
            ).index_select(2, body_indexes)
            self.agent_body_lin_vel_w = self._tensor(
                arrays["agent_body_lin_vel_w"], device
            ).index_select(2, body_indexes)
            self.agent_body_ang_vel_w = self._tensor(
                arrays["agent_body_ang_vel_w"], device
            ).index_select(2, body_indexes)
            self.object_pos_w = self._tensor(arrays["object_pos_w"], device)
            self.object_quat_w = self._wxyz_to_xyzw(
                self._tensor(arrays["object_quat_w"], device)
            )
            self.object_lin_vel_w = self._tensor(
                arrays["object_lin_vel_w"], device
            )

    @staticmethod
    def _tensor(array: np.ndarray, device: str | torch.device) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    @staticmethod
    def _wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
        return quaternion[..., [1, 2, 3, 0]]

    @staticmethod
    def _read_names(raw_names: np.ndarray, field_name: str) -> list[str]:
        names_array = np.asarray(raw_names)
        if names_array.ndim != 1:
            raise ValueError(f"{field_name} must be one-dimensional")
        names: list[str] = []
        for raw_name in names_array.tolist():
            if isinstance(raw_name, bytes):
                name = raw_name.decode("utf-8")
            elif isinstance(raw_name, str):
                name = raw_name
            else:
                raise ValueError(f"{field_name} entries must be strings")
            if not name:
                raise ValueError(f"{field_name} must not contain empty names")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError(f"{field_name} must not contain duplicate names")
        return names

    @staticmethod
    def _indexes_for(
        requested_names: Sequence[str],
        stored_names: Sequence[str],
        kind: str,
        *,
        device: str | torch.device,
        allow_requested_duplicates: bool = False,
    ) -> torch.Tensor:
        requested = list(requested_names)
        if not allow_requested_duplicates and len(set(requested)) != len(requested):
            raise ValueError(f"Requested simulator {kind} names must be unique")
        stored_index = {name: index for index, name in enumerate(stored_names)}
        missing = [name for name in requested if name not in stored_index]
        if missing:
            raise ValueError(
                f"Explicit paired reference is missing simulator {kind} names: {missing}"
            )
        return torch.tensor(
            [stored_index[name] for name in requested],
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def _read_fps(raw_fps: np.ndarray) -> int:
        fps_array = np.asarray(raw_fps).reshape(-1)
        if fps_array.size != 1:
            raise ValueError("fps must contain exactly one scalar value")
        fps_value = float(fps_array[0])
        if not np.isfinite(fps_value) or fps_value <= 0.0 or not fps_value.is_integer():
            raise ValueError("fps must be a positive integer")
        return int(fps_value)

    @classmethod
    def _validate_arrays(
        cls,
        arrays: dict[str, np.ndarray],
        *,
        body_count: int,
        joint_count: int,
    ) -> None:
        joint_pos = arrays["agent_joint_pos"]
        if joint_pos.ndim != 3 or joint_pos.shape[1:] != (cls.num_agents, joint_count):
            raise ValueError(
                "agent_joint_pos must have shape "
                f"[frames, {cls.num_agents}, {joint_count}], got {joint_pos.shape}"
            )
        frames = joint_pos.shape[0]
        if frames < 3:
            raise ValueError("Explicit paired reference requires at least three frames")

        expected_shapes = {
            "agent_joint_vel": joint_pos.shape,
            "agent_body_pos_w": (frames, cls.num_agents, body_count, 3),
            "agent_body_quat_w": (frames, cls.num_agents, body_count, 4),
            "agent_body_lin_vel_w": (frames, cls.num_agents, body_count, 3),
            "agent_body_ang_vel_w": (frames, cls.num_agents, body_count, 3),
            "object_pos_w": (frames, 3),
            "object_quat_w": (frames, 4),
            "object_lin_vel_w": (frames, 3),
        }
        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {arrays[name].shape}"
                )

        for name, array in arrays.items():
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"{name} must contain numeric values")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")

        for name in ("agent_body_quat_w", "object_quat_w"):
            norms = np.linalg.norm(arrays[name].astype(np.float64), axis=-1)
            if not np.allclose(norms, 1.0, rtol=1.0e-4, atol=1.0e-4):
                raise ValueError(f"{name} must contain unit WXYZ quaternions")

    def sample(self, time_steps: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return synchronized robot and shared-object channels at frame indices."""
        if time_steps.ndim != 1:
            raise ValueError("time_steps must be one-dimensional")
        if torch.any(time_steps < 0) or torch.any(time_steps >= self.num_frames):
            raise IndexError("time_steps contains an out-of-range frame")
        return {key: getattr(self, key)[time_steps] for key in self._SAMPLE_KEYS}
