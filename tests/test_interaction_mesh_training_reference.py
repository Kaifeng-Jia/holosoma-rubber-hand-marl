"""Offline compiler checks, runnable directly without pytest or Isaac."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from holosoma_retargeting.interaction_mesh_audit import frozen_mesh, mesh_errors
from holosoma_retargeting.interaction_mesh_geometry import build_robot_landmarks, landmarks_world
from holosoma_retargeting.interaction_mesh_training_reference import (
    grouped_q_matrix, map_landmarks_to_training, score_q, training_body_names,
    urdf_body_transforms,
)


ROOT = next(path for path in (*Path(__file__).resolve().parents, Path.cwd()) if (path / "src/holosoma_retargeting").is_dir())
XML = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
CONFIG = ROOT / "src/holosoma/holosoma/config_values/robot.py"


def test_folded_score_matches_full_graph():
    rng = np.random.default_rng(721)
    body, obj = rng.normal(size=(19, 3)), rng.normal(size=(85, 3))
    groups = [f"body_{i}" for i in range(13)] + ["left_hand", "right_hand", "left_hand", "left_hand", "right_hand", "right_hand"]
    lap, adjacency = frozen_mesh(body, obj)
    q = grouped_q_matrix(lap, adjacency, groups)
    np.testing.assert_allclose(q, q.T, atol=1e-15)
    assert np.linalg.eigvalsh(q).min() > -1e-12
    for delta in (np.zeros_like(body), np.ones_like(body) * 0.05, rng.normal(size=body.shape)):
        expected = mesh_errors(lap, adjacency, body, body + delta, obj, groups)["grouped_mse_m2"]
        np.testing.assert_allclose(score_q(delta, q), expected, atol=1e-12)


def test_grouping_does_not_triple_hand_weight():
    groups = [f"body_{i}" for i in range(13)] + ["left_hand", "right_hand", "left_hand", "left_hand", "right_hand", "right_hand"]
    lap = np.eye(20)
    adjacency = np.zeros((20, 20), dtype=bool)
    q = grouped_q_matrix(lap, adjacency, groups)
    np.testing.assert_allclose(q[0, 0], 0.5 / 15)
    np.testing.assert_allclose(q[13, 13], 0.5 / 45)
    np.testing.assert_allclose(sum(q[i, i] for i in [13, 15, 16]), q[0, 0])


def test_mapping_uses_fixed_chains_not_joint_origins():
    specs, _ = build_robot_landmarks(XML)
    mapped, _ = map_landmarks_to_training(specs, XML, URDF, training_body_names(CONFIG))
    by_name = {item["name"]: item for item in mapped}
    assert by_name["L_Ankle"]["body"] == "left_knee_link"
    np.testing.assert_allclose(by_name["L_Ankle"]["local_xyz"], [0, 0, -0.28])
    assert by_name["L_Foot"]["body"] == "left_ankle_roll_link"
    np.testing.assert_allclose(by_name["L_Foot"]["local_xyz"], [0.14, 0, -0.03])
    assert by_name["L_Wrist"]["body"] == "left_wrist_yaw_link"
    np.testing.assert_allclose(by_name["L_Wrist"]["local_xyz"], [0.0415, 0.003, 0])
    assert all(item["body"] in training_body_names(CONFIG) for item in mapped)


def test_training_fixed_points_agree_with_independent_urdf_fk():
    import yourdfpy

    specs, _ = build_robot_landmarks(XML)
    mapped, _ = map_landmarks_to_training(specs, XML, URDF, training_body_names(CONFIG))
    parsed = ET.parse(URDF).getroot()
    movable = [item.get("name") for item in parsed.findall("joint") if item.get("type") != "fixed"]
    oracle = yourdfpy.URDF.load(str(URDF), load_meshes=False, load_collision_meshes=False)
    rng = np.random.default_rng(721)
    for _ in range(5):
        joints = dict(zip(movable, rng.uniform(-0.5, 0.5, len(movable))))
        oracle.update_cfg(joints)
        poses = urdf_body_transforms(URDF, joints)
        for name in training_body_names(CONFIG):
            np.testing.assert_allclose(poses[name], oracle.get_transform(name), atol=1e-10)
        names = training_body_names(CONFIG)
        pos = np.stack([poses[name][:3, 3] for name in names])
        quat = Rotation.from_matrix(np.stack([poses[name][:3, :3] for name in names])).as_quat()[:, [3, 0, 1, 2]]
        actual = landmarks_world(pos, quat, names, mapped)
        for i, original in enumerate(specs):
            if original["body"] not in poses:
                # Virtual foot helpers have no physical training link; their
                # reviewed fixed offset is asserted separately above.
                continue
            expected = oracle.get_transform(original["body"]) @ np.r_[original["local_xyz"], 1.0]
            np.testing.assert_allclose(actual[i], expected[:3], atol=1e-10)


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} offline compiler checks passed")
