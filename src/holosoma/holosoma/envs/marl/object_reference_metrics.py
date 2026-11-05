"""Object-centric trajectory metrics for Plan 5 evaluation."""

from __future__ import annotations

import math

import torch

from holosoma.utils.rotations import get_euler_xyz, wrap_to_pi


def _validate_history(name: str, values: torch.Tensor, width: int) -> None:
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape [frames, {width}], got {tuple(values.shape)}")
    if values.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one frame")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def compute_object_reference_metrics(
    *,
    reference_positions: torch.Tensor,
    actual_positions: torch.Tensor,
    reference_quaternions: torch.Tensor,
    actual_quaternions: torch.Tensor,
    full_reference_positions: torch.Tensor,
    completed_reference: bool,
) -> dict[str, float | int | bool]:
    """Summarize one continuous object rollout without crossing an environment reset."""
    _validate_history("reference_positions", reference_positions, 3)
    _validate_history("actual_positions", actual_positions, 3)
    _validate_history("reference_quaternions", reference_quaternions, 4)
    _validate_history("actual_quaternions", actual_quaternions, 4)
    _validate_history("full_reference_positions", full_reference_positions, 3)
    frames = reference_positions.shape[0]
    if actual_positions.shape[0] != frames:
        raise ValueError("Reference and actual position histories must have equal length")
    if reference_quaternions.shape[0] != frames or actual_quaternions.shape[0] != frames:
        raise ValueError("Position and quaternion histories must have equal length")
    total_frames = full_reference_positions.shape[0]
    if total_frames < 2:
        raise ValueError("The full object reference must contain at least two frames")
    if frames > total_frames:
        raise ValueError("Evaluated history cannot exceed the full object reference")
    if completed_reference and frames != total_frames:
        raise ValueError("A completed reference must contain every reference frame")

    reference_xy = reference_positions[:, :2]
    actual_xy = actual_positions[:, :2]
    full_reference_xy = full_reference_positions[:, :2]
    full_reference_deltas = full_reference_xy[1:] - full_reference_xy[:-1]
    reference_path_length = torch.linalg.vector_norm(full_reference_deltas, dim=-1).sum()
    reference_displacement = full_reference_xy[-1] - full_reference_xy[0]
    reference_net_displacement = torch.linalg.vector_norm(reference_displacement)
    if reference_net_displacement <= 1.0e-8:
        raise ValueError("The full object reference must have non-zero planar displacement")
    forward = reference_displacement / reference_net_displacement
    lateral = torch.stack((-forward[1], forward[0]))

    error_xy = actual_xy - reference_xy
    planar_error = torch.linalg.vector_norm(error_xy, dim=-1)
    along_error = error_xy @ forward
    lateral_error = error_xy @ lateral
    actual_deltas = actual_xy[1:] - actual_xy[:-1]
    actual_path_length = (
        torch.linalg.vector_norm(actual_deltas, dim=-1).sum()
        if frames > 1
        else actual_xy.new_zeros(())
    )
    actual_displacement = actual_xy[-1] - actual_xy[0]
    actual_net_displacement = torch.linalg.vector_norm(actual_displacement)
    along_track_progress = actual_displacement @ forward
    direction_defined = bool(actual_net_displacement > 1.0e-8)
    direction_cosine = (
        (actual_displacement @ forward) / actual_net_displacement
        if direction_defined
        else actual_xy.new_zeros(())
    )

    _, _, reference_yaw = get_euler_xyz(reference_quaternions, w_last=True)
    _, _, actual_yaw = get_euler_xyz(actual_quaternions, w_last=True)
    yaw_error = wrap_to_pi((actual_yaw - reference_yaw).clone())
    yaw_error_deg = torch.rad2deg(yaw_error)

    return {
        "reference_frames_total": int(total_frames),
        "reference_frames_evaluated": int(frames),
        "completion_ratio": float(frames / total_frames),
        "completed_reference": bool(completed_reference),
        "reference_path_length_m": float(reference_path_length.item()),
        "reference_net_displacement_m": float(reference_net_displacement.item()),
        "actual_path_length_m": float(actual_path_length.item()),
        "actual_net_displacement_m": float(actual_net_displacement.item()),
        "along_track_progress_m": float(along_track_progress.item()),
        "along_track_progress_ratio": float(
            (along_track_progress / reference_net_displacement).item()
        ),
        "direction_cosine": float(direction_cosine.item()),
        "direction_defined": direction_defined,
        "planar_error_mean_m": float(planar_error.mean().item()),
        "planar_error_rmse_m": float(torch.sqrt(torch.mean(planar_error.square())).item()),
        "planar_error_max_m": float(planar_error.max().item()),
        "final_planar_error_m": float(planar_error[-1].item()),
        "along_track_error_rmse_m": float(torch.sqrt(torch.mean(along_error.square())).item()),
        "lateral_error_rmse_m": float(torch.sqrt(torch.mean(lateral_error.square())).item()),
        "yaw_error_mean_abs_deg": float(yaw_error_deg.abs().mean().item()),
        "yaw_error_rmse_deg": float(torch.sqrt(torch.mean(yaw_error_deg.square())).item()),
        "yaw_error_max_abs_deg": float(yaw_error_deg.abs().max().item()),
        "final_yaw_error_abs_deg": float(yaw_error_deg[-1].abs().item()),
    }
