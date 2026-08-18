import numpy as np
import pytest

from holosoma.analyze_stage1b_contact_audit import analyze_recording


def _audit_fixture() -> tuple[dict[str, np.ndarray], dict]:
    samples = 4
    total_force = np.zeros((samples, 1, 2, 3), dtype=np.float32)
    total_force[:, :, 0, 0] = 6.0
    total_force[:, :, 1, 0] = 4.0
    hand_force = np.zeros((samples, 1, 1, 3), dtype=np.float32)
    hand_force[..., 0] = 6.0
    total_history = np.repeat(total_force[:, None], 3, axis=1)
    hand_history = np.repeat(hand_force[:, None], 3, axis=1)
    contact_pos = np.zeros_like(total_force)
    contact_pos[..., 1] = 0.2
    hand_contact_pos = np.zeros_like(hand_force)
    hand_contact_pos[..., 1] = 0.2
    identity_quat = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (samples, 1))
    position = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    recording = {
        "motion_time_step": np.asarray([0, 1, 2, 0], dtype=np.int64),
        "done": np.zeros(samples, dtype=bool),
        "ref_object_pos_w": position,
        "object_pos_w": position.copy(),
        "ref_object_quat_xyzw": identity_quat,
        "object_quat_xyzw": identity_quat.copy(),
        "object_robot_contact_force_matrix_w": total_force,
        "object_robot_contact_force_matrix_history_w": total_history,
        "object_robot_contact_pos_w": contact_pos,
        "object_hand_contact_force_matrix_w": hand_force,
        "object_hand_contact_force_matrix_history_w": hand_history,
        "object_hand_contact_pos_w": hand_contact_pos,
    }
    metadata = {
        "dt": 0.02,
        "sim_dt": 0.005,
        "motion_time_step_total": 3,
        "object_physics": {
            "mass_kg": [2.6],
            "inertia_kg_m2": [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            "com_pose_b": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            "material_properties": [[[0.5, 0.5, 0.0]]],
        },
        "object_robot_contact": {
            "filter_prim_paths_expr": [
                "/World/envs/env_.*/Robot/left_rubber_hand_link",
                "/World/envs/env_.*/Robot/left_hip_yaw_link",
            ]
        },
    }
    return recording, metadata


def test_contact_audit_separates_hand_and_nonhand_propulsion() -> None:
    recording, metadata = _audit_fixture()
    result = analyze_recording(recording, metadata)

    assert result["recording"]["attempts"] == 1
    assert result["recording"]["completed"] == 1
    attempt = result["attempt_details"][0]
    assert attempt["hand_propulsion_fraction"] == pytest.approx(0.6)
    assert attempt["rubber_hand_propulsion_dominant"]
    assert attempt["endpoint_yaw_pass"]
    assert attempt["top_propulsive_contact_bodies"] == [
        {
            "body_name": "left_rubber_hand_link",
            "positive_propulsive_impulse_n_s": pytest.approx(0.27),
        },
        {
            "body_name": "left_hip_yaw_link",
            "positive_propulsive_impulse_n_s": pytest.approx(0.18),
        },
    ]
    assert result["summary"]["rubber_hand_dominant_attempt_rate"] == pytest.approx(1.0)


def test_contact_audit_requires_runtime_object_physics() -> None:
    recording, metadata = _audit_fixture()
    del metadata["object_physics"]

    with pytest.raises(ValueError, match="object_physics"):
        analyze_recording(recording, metadata)


def test_hand_fraction_remains_bounded_when_nonhand_force_opposes_motion() -> None:
    recording, metadata = _audit_fixture()
    recording["object_robot_contact_force_matrix_w"][:, :, 1, 0] = -2.0
    recording["object_robot_contact_force_matrix_history_w"][:, :, :, 1, 0] = -2.0

    result = analyze_recording(recording, metadata)

    attempt = result["attempt_details"][0]
    assert attempt["hand_propulsion_fraction"] == pytest.approx(1.0)
    assert attempt["rubber_hand_propulsion_dominant"]
