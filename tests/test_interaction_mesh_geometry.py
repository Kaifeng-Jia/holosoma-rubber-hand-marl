"""Offline checks; no Isaac or training process is launched."""

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from holosoma_retargeting.interaction_mesh_geometry import (
    build_robot_landmarks,
    coverage_report,
    landmarks_world,
    load_box_object_mesh,
    sample_object_points,
)


ROOT = next(path for path in (*Path(__file__).resolve().parents, Path.cwd()) if (path / "src/holosoma_retargeting").is_dir())
XML = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
URDF = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_core4d_desk001_small_training.urdf"


def test_mapping_and_non_collinear_hand_vertices():
    from holosoma_retargeting.config_types.data_type import JOINTS_MAPPINGS

    specs, metadata = build_robot_landmarks(XML)
    mapping = JOINTS_MAPPINGS[("smplx", "g1")]
    assert len(specs) == 19
    assert [entry["name"] for entry in specs[:15]] == list(mapping)
    assert [entry["body"] for entry in specs[:15]] == list(mapping.values())
    assert all(entry["local_xyz"] == [0.0, 0.0, 0.0] for entry in specs[:15])
    assert not any("head" in entry["name"] or "sphere_hand" in entry["body"] for entry in specs)
    for side in ("left", "right"):
        assert sum(entry["group"] == f"{side}_hand" for entry in specs) == 3
        offsets = np.asarray([entry["local_xyz"] for entry in specs[15:] if entry["body"] == f"{side}_rubber_hand_link"])
        area = 0.5 * np.linalg.norm(np.cross(offsets[0], offsets[1]))
        assert area > 1e-8
        assert np.isclose(area, metadata["hands"][side]["origin_surface_surface_triangle_area_m2"])


def test_mesh_vertices_are_reconstructed_in_correct_body_frame():
    specs, metadata = build_robot_landmarks(XML)
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.7, -0.2, 1.2]
    data.qpos[3:7] = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    actual = landmarks_world(data.xpos[None, None], data.xquat[None, None], names, specs)[0, 0]
    for index, spec in enumerate(specs[15:], start=15):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, spec["body"])
        mesh = int(model.geom_dataid[geom])
        vertex = model.mesh_vert[int(model.mesh_vertadr[mesh]) + spec["mesh_vertex_index"]]
        expected = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ vertex
        np.testing.assert_allclose(actual[index], expected, atol=2e-8)


def test_landmark_rigid_transform():
    specs = [{"name": "p", "body": "b", "local_xyz": [1.0, 0, 0], "group": "b"}]
    pos = np.zeros((3, 2, 1, 3)) + [4, 5, 6]
    quat = np.zeros((3, 2, 1, 4))
    quat[...] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    actual = landmarks_world(pos, quat, ["b"], specs)
    assert actual.shape == (3, 2, 1, 3)
    np.testing.assert_allclose(actual, np.broadcast_to([4, 6, 6], actual.shape), atol=1e-12)


def test_exterior_union_mesh_and_sampling():
    mesh, metadata = load_box_object_mesh(URDF)
    assert len(metadata["components"]) == 5
    assert metadata["removed_buried_or_duplicate_area_m2"] > 0
    assert len(mesh.faces) == len(metadata["face_component_ids"])
    first, report = sample_object_points(mesh, metadata, 64)
    second, repeated = sample_object_points(mesh, metadata, 64)
    np.testing.assert_array_equal(first, second)
    assert report == repeated
    assert 0 < len(first) <= 64
    assert not report["filled_shortfall"]
    assert len(report["per_component"]) == 5
    assert sum(entry["count"] for entry in report["per_component"].values()) == len(first)
    for component in report["per_component"].values():
        assert sum(component["face_axis_counts"].values()) == component["count"]

    # Every point is on its recorded source box boundary and not in any box
    # interior. Test the actual collision geometry, not a visual proxy.
    lower = np.asarray([item["lower_xyz"] for item in metadata["components"]])
    upper = np.asarray([item["upper_xyz"] for item in metadata["components"]])
    owner = np.asarray(metadata["face_component_ids"])[report["face_indices"]]
    assert np.all(first >= lower[owner] - 1e-10)
    assert np.all(first <= upper[owner] + 1e-10)
    on_face = np.isclose(first, lower[owner], atol=1e-10) | np.isclose(first, upper[owner], atol=1e-10)
    assert np.all(on_face.any(axis=1))
    inside_any = np.all(first[:, None] > lower[None] + 1e-10, axis=-1) & np.all(first[:, None] < upper[None] - 1e-10, axis=-1)
    assert not inside_any.any()


def test_dense_coverage_pool_is_fixed_across_budgets():
    mesh, metadata = load_box_object_mesh(URDF)
    small, _ = sample_object_points(mesh, metadata, 64)
    large, _ = sample_object_points(mesh, metadata, 100)
    first = coverage_report(mesh, metadata, small)
    second = coverage_report(mesh, metadata, large)
    assert first["dense_points_sha256"] == second["dense_points_sha256"]
    assert first["dense_count"] == 4096
    assert len(first["per_component"]) == 5
    assert first["overall"]["max_m"] >= first["overall"]["p95_m"] >= 0
    assert sum(item["dense_count"] for item in first["per_component"].values()) == 4096


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} checks passed")
