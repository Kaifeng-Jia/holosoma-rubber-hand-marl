from __future__ import annotations

import numpy as np

from holosoma.analyze_wbt_eval_recording import quaternion_angle_error, summarize_recording


def test_quaternion_angle_error_uses_shortest_rotation() -> None:
    identity = np.array([[0.0, 0.0, 0.0, 1.0]])
    negated_identity = -identity
    half_turn_z = np.array([[0.0, 0.0, 1.0, 0.0]])

    np.testing.assert_allclose(quaternion_angle_error(identity, negated_identity), 0.0)
    np.testing.assert_allclose(quaternion_angle_error(identity, half_turn_z), np.pi)


def test_summary_separates_completed_and_failed_motion_attempts() -> None:
    steps = 7
    dofs = 2
    tracked_bodies = 1
    robot_bodies = 3
    history = 2

    motion_time_step = np.array([0, 1, 2, 3, 0, 1, 0])
    done = np.array([False, False, False, False, False, True, False])
    zeros_dof = np.zeros((steps, dofs))
    zeros_body_pos = np.zeros((steps, tracked_bodies, 3))
    identity_body = np.zeros((steps, tracked_bodies, 4))
    identity_body[..., 3] = 1.0
    identity_root = np.zeros((steps, 4))
    identity_root[..., 3] = 1.0
    object_pos = np.zeros((steps, 3))
    object_pos[:, 0] = np.array([0.0, 0.02, 0.04, 0.06, 0.0, -0.02, 0.0])
    ref_object_pos = np.zeros((steps, 3))
    ref_object_pos[:, 0] = np.array([0.0, 0.02, 0.04, 0.06, 0.0, 0.02, 0.0])
    object_quat = np.zeros((steps, 4))
    object_quat[..., 3] = 1.0

    contact_history = np.zeros((steps, history, robot_bodies, 3))
    contact_history[2, 0, 0, 0] = 8.0
    contact_history[5, 0, 1, 0] = 7.0
    sensor_contact_history = np.zeros((steps, history, 5, 3))
    sensor_contact_history[:, 0, 0, 0] = 20.0  # Wrist force must not override a present hand link.
    sensor_contact_history[2, 0, 1, 0] = 8.0
    sensor_contact_history[5, 0, 3, 0] = 7.0

    recording = {
        "motion_time_step": motion_time_step,
        "done": done,
        "timeout": np.zeros(steps, dtype=bool),
        "ref_joint_pos": zeros_dof,
        "pre_dof_pos": zeros_dof + 0.1,
        "ref_body_pos_w": zeros_body_pos,
        "pre_tracked_body_pos_w": zeros_body_pos + 0.2,
        "ref_body_quat_xyzw": identity_body,
        "pre_tracked_body_quat_xyzw": identity_body,
        "ref_root_pos_w": np.tile([0.0, 0.0, 0.8], (steps, 1)),
        "pre_root_pos": np.tile([0.0, 0.0, 0.8], (steps, 1)),
        "pre_root_quat_xyzw": identity_root,
        "contact_forces_w": contact_history[:, 0],
        "contact_forces_history_w": contact_history,
        "contact_sensor_forces_w": sensor_contact_history[:, 0],
        "contact_sensor_forces_history_w": sensor_contact_history,
        "actions": zeros_dof,
        "torques": zeros_dof,
        "ref_object_pos_w": ref_object_pos,
        "object_pos_w": object_pos,
        "ref_object_quat_xyzw": object_quat,
        "object_quat_xyzw": object_quat,
    }
    metadata = {
        "dt": 0.02,
        "motion_fps": 50,
        "motion_time_step_total": 5,
        "body_names": [
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "pelvis",
        ],
        "contact_sensor_body_names": [
            "left_wrist_yaw_link",
            "left_rubber_hand_link",
            "right_wrist_yaw_link",
            "right_rubber_hand_link",
            "pelvis",
        ],
        "effort_limits": [10.0, 20.0],
        "contact_force_semantics": "test",
    }

    summary = summarize_recording(recording, metadata)

    assert summary["attempts"]["closed"] == 2
    assert summary["attempts"]["completed"] == 1
    assert summary["attempts"]["early_terminated"] == 1
    assert summary["attempts"]["completion_rate"] == 0.5
    assert summary["contact"]["attempt_contact_success_rate"] == 1.0
    assert summary["contact"]["left_contact_bodies"] == ["left_rubber_hand_link"]
    assert summary["contact"]["left_hand_step_fraction"] == 1 / 6
    assert summary["object"]["direction_evaluable_attempts"] == 2
    assert summary["object"]["correct_displacement_direction_rate"] == 0.5
    np.testing.assert_allclose(summary["tracking"]["joint_position_rmse_rad"], 0.1)
    np.testing.assert_allclose(summary["tracking"]["key_body_position_rmse_m"], 0.2)
