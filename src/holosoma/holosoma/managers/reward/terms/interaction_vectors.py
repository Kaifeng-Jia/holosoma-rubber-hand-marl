"""Pure-tensor, distance-weighted body/object vector comparison.

This is a separate candidate from the frozen Laplacian reward. It does not
read simulator state, choose object samples, apply exp/weight/dt, or install a
reward. Object sample indices must denote the same fixed object-local points
in actual and reference states, transformed with their respective object poses.
Formula source: InterMimic's compute_ig_reward (inverse-distance branches):
https://github.com/Sirui-Xu/InterMimic/blob/main/isaacgym/src/intermimic/env/tasks/intermimic.py
The optional body-group sampling prior is our explicit adaptation.
"""

from __future__ import annotations

import math

import torch


def weighted_body_object_vector_error(
    actual_body_points_w: torch.Tensor,
    reference_body_points_w: torch.Tensor,
    actual_object_points_w: torch.Tensor,
    reference_object_points_w: torch.Tensor,
    *,
    body_point_weights: torch.Tensor | None = None,
    distance_floor_m: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return per-leading-index squared vector error in m² and diagnostics.

    Bodies have shape [..., P, 3], objects [..., K, 3]; leading dimensions
    broadcast. For bodies [batch, agent, P, 3] and one shared object per batch,
    pass object points as [batch, 1, K, 3]. Every agent is normalized separately,
    over P and K only. All four point sets use the SAME world-coordinate axes;
    do not rotate each relation into its own actual/reference object frame.

    Positive distances use inverse squared weights, capped below
    distance_floor_m². The 0.1 m default reproduces InterMimic's 0.01 m² floor;
    it is a source convention, not a calibrated setting for this task.
    Actual/reference weights are normalized separately and then averaged.

    body_point_weights is a nonnegative [P] sampling prior. Use 1 for singleton
    body groups and 1/3 for each point in a three-point hand group. This removes
    the threefold sampling-count advantage, but does NOT freeze a hand's final
    normalized weight: its geometry still determines distance-based salience.
    Finite floating-point coordinates in metres are expected.
    """
    if not math.isfinite(distance_floor_m) or distance_floor_m <= 0:
        raise ValueError("distance_floor_m must be finite and positive")
    for actual, reference in (
        (actual_body_points_w, reference_body_points_w),
        (actual_object_points_w, reference_object_points_w),
    ):
        if (
            actual.ndim < 2 or reference.ndim < 2
            or actual.shape[-1] != 3 or actual.shape[-2] == 0
            or actual.shape[-2:] != reference.shape[-2:]
        ):
            raise ValueError("Actual/reference point axes must match nonempty [N, 3]")

    point_count = actual_body_points_w.shape[-2]
    if body_point_weights is None:
        prior = actual_body_points_w.new_ones(point_count)
    else:
        prior = body_point_weights.to(actual_body_points_w)
        if (
            prior.shape != (point_count,) or not torch.isfinite(prior).all()
            or (prior < 0).any() or prior.sum() <= 0
        ):
            raise ValueError("body_point_weights must be finite nonnegative [P] with positive sum")

    actual_vectors = actual_body_points_w[..., :, None, :] - actual_object_points_w[..., None, :, :]
    reference_vectors = reference_body_points_w[..., :, None, :] - reference_object_points_w[..., None, :, :]

    def normalized_weights(vectors: torch.Tensor) -> torch.Tensor:
        distance_sq = vectors.square().sum(dim=-1)
        weights = prior[:, None] / distance_sq.clamp_min(distance_floor_m**2)
        return weights / weights.sum(dim=(-2, -1), keepdim=True)

    actual_weights = normalized_weights(actual_vectors)
    reference_weights = normalized_weights(reference_vectors)
    weights = 0.5 * (actual_weights + reference_weights)
    edge_error_m2 = (actual_vectors - reference_vectors).square().sum(dim=-1)
    point_error_m2 = (weights * edge_error_m2).sum(dim=-1)
    error_m2 = point_error_m2.sum(dim=-1)
    return error_m2, {
        "actual_weights": actual_weights,
        "reference_weights": reference_weights,
        "point_weight": weights.sum(dim=-1),
        "point_error_m2": point_error_m2,
        "actual_weighted_error_m2": (actual_weights * edge_error_m2).sum(dim=(-2, -1)),
        "reference_weighted_error_m2": (reference_weights * edge_error_m2).sum(dim=(-2, -1)),
    }


__all__ = ["weighted_body_object_vector_error"]
