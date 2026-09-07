"""Small invariance and normalization tests; no Isaac or policy involved."""

import numpy as np
from scipy.spatial.transform import Rotation

from holosoma_retargeting.interaction_mesh_audit import (
    frozen_mesh,
    mesh_errors,
    points_in_object_frame,
)


def scene():
    body = np.array([[0.2, 0.0, 0.9], [-0.2, 0.1, 0.8], [0.0, -0.4, 0.7]])
    obj = np.array([[x, y, z] for x in (-0.3, 0.3) for y in (-0.2, 0.2) for z in (-0.1, 0.1)])
    lap, adjacency = frozen_mesh(body, obj)
    return body, obj, lap, adjacency


def test_identity_and_uniform_laplacian():
    body, obj, lap, adjacency = scene()
    np.testing.assert_allclose(lap.sum(axis=1), 0.0, atol=1e-15)
    result = mesh_errors(lap, adjacency, body, body, obj, ["a", "b", "c"])
    assert result["grouped_mse_m2"] == 0
    assert result["omni_sum_m2"] == 0


def test_frozen_graph_shift_has_quadratic_response():
    body, obj, lap, adjacency = scene()
    values = []
    for shift in (0.02, 0.04, 0.08):
        result = mesh_errors(lap, adjacency, body, body + np.array([shift, 0, 0]), obj, ["a", "b", "c"])
        values.append(result["grouped_mse_m2"])
        assert result["active_object_mse_m2"] > 0
    np.testing.assert_allclose(np.array(values) / values[0], [1, 4, 16])


def test_common_rigid_transform_preserves_object_coordinates():
    body, obj, _, _ = scene()
    points = body[None, None]
    identity = np.array([[0, 0, 0, 1, 0, 0, 0]])
    rotation = Rotation.from_euler("xyz", [0.3, 0.5, 0.7])
    translation = np.array([3, -2, 0.1])
    transformed = rotation.apply(body) + translation
    q = rotation.as_quat()[[3, 0, 1, 2]]
    pose = np.concatenate((translation, q))[None]
    np.testing.assert_allclose(points_in_object_frame(points, identity), points_in_object_frame(transformed[None, None], pose), atol=1e-14)


def test_object_only_zero_rows_do_not_dilute_grouped_score():
    body, obj, lap, adjacency = scene()
    actual = body.copy()
    actual[0, 0] += 0.1
    original = mesh_errors(lap, adjacency, body, actual, obj, ["a", "b", "c"])
    expanded_lap = np.zeros((len(lap) + 2, len(lap) + 2))
    expanded_lap[:-2, :-2] = lap
    expanded_lap[-2:, -2:] = [[1, -1], [-1, 1]]
    expanded_adj = np.zeros_like(expanded_lap, dtype=bool)
    expanded_adj[:-2, :-2] = adjacency
    expanded_adj[-2:, -2:] = [[False, True], [True, False]]
    expanded = mesh_errors(expanded_lap, expanded_adj, body, actual, np.vstack([obj, [2, 0, 0], [3, 0, 0]]), ["a", "b", "c"])
    assert expanded["grouped_mse_m2"] == original["grouped_mse_m2"]
    assert expanded["all_node_mean_m2"] < original["all_node_mean_m2"]


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} checks passed")
