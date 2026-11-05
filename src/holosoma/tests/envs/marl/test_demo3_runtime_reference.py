"""Frozen full-body reference contract for the isolated Demo 3 task."""

import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
REFERENCE = (
    REPO_ROOT
    / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/demo3_tug"
    / "sub3_010_diagonal_tug_runtime.npz"
)


def test_demo3_runtime_reference_is_full_body_opposing_and_not_an_object_target() -> None:
    with np.load(REFERENCE, allow_pickle=False) as data:
        assert data["agent_joint_pos"].shape == (317, 2, 29)
        assert data["agent_joint_vel"].shape == (317, 2, 29)
        assert data["agent_body_pos_w"].shape == (317, 2, 51, 3)
        assert data["agent_body_quat_w"].shape == (317, 2, 51, 4)
        assert int(data["fps"]) == 50
        provenance = json.loads(str(data["provenance"]))
        assert provenance["demo"] == "Demo3Tug"
        assert provenance["opponent_transform"] == "world_z_half_turn"
        assert provenance["table_channel_role"] == "reset_and_schema_only_not_tracking_target"

        body_names = data["body_names"].astype(str).tolist()
        pelvis = body_names.index("pelvis")
        table_xy = data["object_pos_w"][:, :2]
        relative = data["agent_body_pos_w"][:, :, pelvis, :2] - table_xy[:, None]
        np.testing.assert_allclose(relative[:, 0], -relative[:, 1], atol=1.0e-9)
        np.testing.assert_allclose(
            data["agent_joint_pos"][:, 0], data["agent_joint_pos"][:, 1], atol=0.0
        )
