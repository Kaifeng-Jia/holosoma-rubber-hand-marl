"""CPU-only contracts for the offline interaction-vector candidate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


# Load this pure-tensor module directly, without initializing Isaac or an env.
MODULE_PATH = Path(__file__).resolve().parents[3] / "holosoma/managers/reward/terms/interaction_vectors.py"
_spec = importlib.util.spec_from_file_location("interaction_vectors", MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
score = _module.weighted_body_object_vector_error


def _fixture():
    generator = torch.Generator().manual_seed(721)
    body = torch.randn(2, 2, 5, 3, generator=generator, dtype=torch.float64) * 0.3
    obj = torch.randn(2, 1, 7, 3, generator=generator, dtype=torch.float64) * 0.2
    return body, obj


def test_identity_zero_distance_and_per_agent_normalization():
    body, obj = _fixture()
    body[:, :, 0] = obj[:, :, 0]
    body_before, object_before = body.clone(), obj.clone()
    error, diag = score(body, body, obj, obj)
    torch.testing.assert_close(error, torch.zeros(2, 2, dtype=body.dtype), atol=0, rtol=0)
    for key in ("actual_weights", "reference_weights"):
        assert torch.isfinite(diag[key]).all()
        torch.testing.assert_close(diag[key].sum(dim=(-2, -1)), torch.ones_like(error))
    torch.testing.assert_close(diag["point_error_m2"].sum(dim=-1), error)
    torch.testing.assert_close(body, body_before)
    torch.testing.assert_close(obj, object_before)


def test_matches_literal_separately_normalized_inverse_square_formula():
    reference, ref_obj = _fixture()
    actual, obj = reference.clone(), ref_obj.clone()
    actual[..., 0, 0] += 0.23
    obj[..., 2, 1] -= 0.14
    prior = torch.tensor([1, 1, 1 / 3, 1 / 3, 1 / 3], dtype=actual.dtype)
    error, diag = score(actual, reference, obj, ref_obj, body_point_weights=prior)
    expected = torch.zeros_like(error)
    for batch in range(2):
        for agent in range(2):
            raw_actual, raw_reference, residual = [], [], []
            for i in range(5):
                for j in range(7):
                    va = (actual[batch, agent, i] - obj[batch, 0, j]).tolist()
                    vr = (reference[batch, agent, i] - ref_obj[batch, 0, j]).tolist()
                    raw_actual.append(float(prior[i]) / max(sum(x*x for x in va), 0.01))
                    raw_reference.append(float(prior[i]) / max(sum(x*x for x in vr), 0.01))
                    residual.append(sum((a-r)**2 for a, r in zip(va, vr)))
            expected[batch, agent] = sum(
                0.5 * (wa / sum(raw_actual) + wr / sum(raw_reference)) * delta
                for wa, wr, delta in zip(raw_actual, raw_reference, residual)
            )
    torch.testing.assert_close(error, expected, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(
        error, 0.5 * (diag["actual_weighted_error_m2"] + diag["reference_weighted_error_m2"]),
    )


def test_three_hand_points_share_one_sampling_budget():
    reference = torch.zeros(2, 15, 3, dtype=torch.float64)
    reference[..., 0] = 0.4
    actual = reference.clone()
    actual[:, 13, 0] += 0.05
    obj = torch.zeros(1, 2, 3, dtype=reference.dtype)
    obj[:, 1, 1] = 0.1
    original, _ = score(actual, reference, obj, obj)
    # Existing 19-point ordering: wrists 13/14, extra hand points 15..18.
    indexes = list(range(15)) + [13, 13, 14, 14]
    prior = torch.ones(19, dtype=reference.dtype)
    prior[[13, 14, 15, 16, 17, 18]] = 1 / 3
    expanded, diag = score(
        actual[:, indexes], reference[:, indexes], obj, obj, body_point_weights=prior,
    )
    torch.testing.assert_close(expanded, original, atol=1e-14, rtol=1e-12)
    ref_point_weights = diag["reference_weights"].sum(dim=-1)
    torch.testing.assert_close(ref_point_weights[:, [13, 15, 16]].sum(dim=-1), ref_point_weights[:, 0])


def test_world_axes_rotation_and_fixed_object_correspondence():
    reference, obj = _fixture()
    actual = reference.clone()
    actual[..., 0] += 0.03
    before, _ = score(actual, reference, obj, obj)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=actual.dtype)
    translation = torch.tensor([10., -7., 2.], dtype=actual.dtype)
    def transform(points):
        return points @ rotation.T + translation
    after, _ = score(transform(actual), transform(reference), transform(obj), transform(obj))
    torch.testing.assert_close(after, before, atol=1e-14, rtol=1e-11)
    # Rotating only actual body + object preserves actual distances, not vectors.
    rotated, _ = score(reference @ rotation.T, reference, obj @ rotation.T, obj)
    assert (rotated > 0).all()
    permutation = torch.arange(obj.shape[-2] - 1, -1, -1)
    matched, _ = score(reference, reference, obj[..., permutation, :], obj[..., permutation, :])
    mismatched, _ = score(reference, reference, obj[..., permutation, :], obj)
    assert not matched.any()
    assert (mismatched > 0).all()


def test_five_cm_error_scale_and_agent_independence_with_extra_batch_axis():
    reference, obj = _fixture()
    reference = reference.unsqueeze(0).expand(3, -1, -1, -1, -1)
    obj = obj.unsqueeze(0).expand(3, -1, -1, -1, -1)
    actual = reference.clone()
    actual[:, :, 0, :, 0] += 0.05
    error, diag = score(actual, reference, obj, obj)
    assert error.shape == (3, 2, 2)
    torch.testing.assert_close(error[..., 0], torch.full((3, 2), 0.05**2, dtype=error.dtype))
    torch.testing.assert_close(error[..., 1], torch.zeros(3, 2, dtype=error.dtype), atol=0, rtol=0)
    # Weights normalized over points/objects only: altering agent 0 cannot alter 1.
    _, original_diag = score(reference, reference, obj, obj)
    torch.testing.assert_close(diag["actual_weights"][:, :, 1], original_diag["actual_weights"][:, :, 1])
    # No hidden xyz/point/object/agent averaging in the per-agent m² metric.
    torch.testing.assert_close(error.mean(dim=-1), torch.full((3, 2), 0.00125, dtype=error.dtype))


def test_reference_weight_lower_bound_and_release_follows_reference():
    reference = torch.zeros(2, 19, 3, dtype=torch.float64)
    reference[..., 0] = 0.8
    hand = [13, 15, 16]
    reference[:, hand, 0] = 0.02
    obj = torch.zeros(1, 1, 3, dtype=reference.dtype)
    prior = torch.ones(19, dtype=reference.dtype)
    prior[[13, 14, 15, 16, 17, 18]] = 1 / 3
    actual = reference.clone()
    actual[:, hand, 0] = 2.0
    error, diag = score(actual, reference, obj, obj, body_point_weights=prior)
    actual_mass = diag["actual_weights"][:, hand].sum(dim=(-2, -1))
    reference_mass = diag["reference_weights"][:, hand].sum(dim=(-2, -1))
    assert (actual_mass < reference_mass).all()
    assert (error >= 0.5 * diag["reference_weighted_error_m2"] - 1e-14).all()
    assert (error > 0).all()
    # In a release frame, the target moves away; following it remains perfect.
    release_reference = reference.clone()
    release_reference[:, hand, 0] = 0.5
    following, _ = score(
        release_reference, release_reference, obj, obj, body_point_weights=prior,
    )
    stuck, _ = score(reference, release_reference, obj, obj, body_point_weights=prior)
    assert not following.any()
    assert (stuck > 0).all()
