from __future__ import annotations

import math

import pytest
import torch

from holosoma.envs.marl.object_reference_metrics import compute_object_reference_metrics


def _yaw_quaternions(degrees: list[float]) -> torch.Tensor:
    yaw = torch.deg2rad(torch.tensor(degrees))
    result = torch.zeros(len(degrees), 4)
    result[:, 2] = torch.sin(yaw / 2.0)
    result[:, 3] = torch.cos(yaw / 2.0)
    return result


def test_complete_straight_object_reference_metrics() -> None:
    reference = torch.tensor([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [2.0, 0.0, 0.5]])
    actual = torch.tensor([[0.0, 0.0, 0.5], [0.8, 0.1, 0.5], [1.8, 0.2, 0.5]])

    metrics = compute_object_reference_metrics(
        reference_positions=reference,
        actual_positions=actual,
        reference_quaternions=_yaw_quaternions([0.0, 0.0, 0.0]),
        actual_quaternions=_yaw_quaternions([0.0, 5.0, -10.0]),
        full_reference_positions=reference,
        completed_reference=True,
    )

    assert metrics["completed_reference"] is True
    assert metrics["completion_ratio"] == 1.0
    assert metrics["reference_path_length_m"] == pytest.approx(2.0)
    assert metrics["along_track_progress_m"] == pytest.approx(1.8)
    assert metrics["along_track_progress_ratio"] == pytest.approx(0.9)
    assert metrics["direction_cosine"] == pytest.approx(1.8 / math.sqrt(1.8**2 + 0.2**2))
    assert metrics["planar_error_rmse_m"] == pytest.approx(math.sqrt((0.0 + 0.05 + 0.08) / 3.0))
    assert metrics["lateral_error_rmse_m"] == pytest.approx(math.sqrt((0.0 + 0.01 + 0.04) / 3.0))
    assert metrics["yaw_error_rmse_deg"] == pytest.approx(math.sqrt(125.0 / 3.0), abs=1.0e-4)
    assert metrics["final_yaw_error_abs_deg"] == pytest.approx(10.0, abs=1.0e-4)


def test_partial_reference_reports_completion_without_extrapolation() -> None:
    full_reference = torch.tensor(
        [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [2.0, 0.0, 0.5], [3.0, 0.0, 0.5]]
    )
    evaluated_reference = full_reference[:2]
    quaternions = _yaw_quaternions([0.0, 0.0])

    metrics = compute_object_reference_metrics(
        reference_positions=evaluated_reference,
        actual_positions=evaluated_reference.clone(),
        reference_quaternions=quaternions,
        actual_quaternions=quaternions.clone(),
        full_reference_positions=full_reference,
        completed_reference=False,
    )

    assert metrics["reference_frames_evaluated"] == 2
    assert metrics["reference_frames_total"] == 4
    assert metrics["completion_ratio"] == 0.5
    assert metrics["completed_reference"] is False
    assert metrics["along_track_progress_ratio"] == pytest.approx(1.0 / 3.0)


def test_completed_reference_requires_every_frame() -> None:
    full_reference = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    quaternion = _yaw_quaternions([0.0])

    with pytest.raises(ValueError, match="every reference frame"):
        compute_object_reference_metrics(
            reference_positions=full_reference[:1],
            actual_positions=full_reference[:1],
            reference_quaternions=quaternion,
            actual_quaternions=quaternion,
            full_reference_positions=full_reference,
            completed_reference=True,
        )
