"""Demo 4 checkpoint isolation and lossless expansion tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from holosoma.agents.mappo.demo4_checkpoint import (
    DEMO4_ACTOR_COMPATIBILITY_VERSION,
    DEMO4_CHECKPOINT_INTERVAL,
    DEMO4_OBJECT_MASS_KG,
    DEMO4_OBJECT_MATERIAL,
    DEMO4_REFERENCE_FPS,
    DEMO4_REFERENCE_FRAMES,
    DEMO4_ROBOT_ASSET,
    DEMO4_STATIC_RUNTIME_SHA256,
    DEMO4_SUCCESS_BONUS_WEIGHT,
    DEMO4_TABLE_OBS_CONTAINS_YAW_RATE,
    DEMO4_TARGET_YAW_DEGREES,
    DEMO4_YAW_PROGRESS_REWARD_WEIGHT,
    Demo4TableObservationExpansion,
    demo4_static_training_contract,
    expand_plan5_pull_actor_for_demo4,
    is_demo4_periodic_checkpoint,
    validate_demo4_static_runtime,
    validate_demo4_lossless_expansion,
    demo4_static_training_contract,
)
from holosoma.agents.mappo.demo4_initialization import (
    DEMO4_SOURCE_PULL_CHECKPOINT_SHA256,
)
from holosoma.agents.ppo.checkpoint_compat import ACTOR_FIRST_WEIGHT


REPO_ROOT = Path(__file__).resolve().parents[5]
PULL_08050 = (
    REPO_ROOT
    / "logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt"
)
DEMO4_RUNTIME = (
    REPO_ROOT
    / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
    / "demo4_rotate/demo4_pull_pull_static_table_runtime.npz"
)


def _source() -> dict:
    return torch.load(PULL_08050, map_location="cpu", weights_only=False)


def test_demo4_expansion_is_lossless_and_strips_old_training_state() -> None:
    source = _source()
    frozen_source = copy.deepcopy(source)

    converted = expand_plan5_pull_actor_for_demo4(
        source,
        source_file_sha256=DEMO4_SOURCE_PULL_CHECKPOINT_SHA256,
    )

    validate_demo4_lossless_expansion(source, converted)
    validate_demo4_lossless_expansion(frozen_source, converted)
    spec = Demo4TableObservationExpansion()
    assert source["actor_model_state_dict"][ACTOR_FIRST_WEIGHT].shape[1] == 158
    assert converted["actor_model_state_dict"][ACTOR_FIRST_WEIGHT].shape[1] == 164
    assert converted["demo4_actor_compatibility"]["version"] == (
        DEMO4_ACTOR_COMPATIBILITY_VERSION
    )
    assert converted["demo4_actor_compatibility"]["contains_yaw_rate"] is False
    assert converted["demo4_actor_compatibility"]["table_dim"] == 6
    assert spec.target_dim == 164
    for forbidden in (
        "plan5_mappo",
        "critic_model_state_dict",
        "critic_obs_normalizer_state_dict",
        "actor_optimizer_state_dict",
        "critic_optimizer_state_dict",
    ):
        assert forbidden not in converted


def test_demo4_expansion_rejects_incompatible_plan5_metadata() -> None:
    source = _source()
    source["plan5_mappo"]["actor_obs_dim"] = 164

    with pytest.raises(ValueError, match="Incompatible Plan 5 Pull checkpoint metadata"):
        expand_plan5_pull_actor_for_demo4(
            source,
            source_file_sha256=DEMO4_SOURCE_PULL_CHECKPOINT_SHA256,
        )


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(-1, False), (0, False), (149, False), (150, True), (300, True), (301, False)],
)
def test_demo4_checkpoint_cadence_defaults_to_150(
    iteration: int,
    expected: bool,
) -> None:
    assert DEMO4_CHECKPOINT_INTERVAL == 150
    assert is_demo4_periodic_checkpoint(iteration) is expected


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(0, False), (999, False), (1000, True), (2000, True), (2001, False)],
)
def test_demo4_checkpoint_cadence_accepts_explicit_interval(
    iteration: int,
    expected: bool,
) -> None:
    assert is_demo4_periodic_checkpoint(iteration, interval=1000) is expected


@pytest.mark.parametrize("interval", [0, -1])
def test_demo4_checkpoint_cadence_rejects_non_positive_interval(
    interval: int,
) -> None:
    with pytest.raises(ValueError, match="interval must be positive"):
        is_demo4_periodic_checkpoint(150, interval=interval)


def test_demo4_training_contract_uses_unwrapped_safe_yaw() -> None:
    contract = demo4_static_training_contract()
    assert contract["target_yaw_degrees"] == 90.0
    assert contract["maximum_success_tilt_degrees"] == 60.0
    assert contract["yaw_progress_representation"] == (
        "per_step_unwrapped_heading_accumulator"
    )
    assert contract["unsafe_terminal_task_reward"] is False


def test_demo4_static_training_contract_locks_approved_task_facts() -> None:
    contract = demo4_static_training_contract()

    assert contract == {
        "static_runtime_sha256": DEMO4_STATIC_RUNTIME_SHA256,
        "reference_frames": DEMO4_REFERENCE_FRAMES,
        "reference_fps": DEMO4_REFERENCE_FPS,
        "target_yaw_degrees": DEMO4_TARGET_YAW_DEGREES,
        "maximum_success_tilt_degrees": 60.0,
        "yaw_progress_representation": (
            "per_step_unwrapped_heading_accumulator"
        ),
        "unsafe_terminal_task_reward": False,
        "object_mass_kg": DEMO4_OBJECT_MASS_KG,
        "object_material_static_dynamic_restitution": list(DEMO4_OBJECT_MATERIAL),
        "yaw_progress_reward_weight": DEMO4_YAW_PROGRESS_REWARD_WEIGHT,
        "success_bonus_weight": DEMO4_SUCCESS_BONUS_WEIGHT,
        "robot_asset": DEMO4_ROBOT_ASSET,
        "rubber_hand_collision": True,
        "table_obs_contains_yaw_rate": DEMO4_TABLE_OBS_CONTAINS_YAW_RATE,
        "table_reference_role": "reset_and_schema_only_not_tracking_target",
    }
    assert DEMO4_STATIC_RUNTIME_SHA256 == (
        "3eb482e1bdb9072b50054c5501cda1d4646bef39ef754df55c03c281626199f3"
    )
    assert (DEMO4_REFERENCE_FRAMES, DEMO4_REFERENCE_FPS) == (316, 50)
    assert DEMO4_TARGET_YAW_DEGREES == 90.0
    assert DEMO4_OBJECT_MASS_KG == 20.0
    assert DEMO4_OBJECT_MATERIAL == (0.5, 0.5, 0.0)
    assert DEMO4_YAW_PROGRESS_REWARD_WEIGHT == 10.0
    assert DEMO4_SUCCESS_BONUS_WEIGHT == 5.0
    assert DEMO4_ROBOT_ASSET.endswith("rubberhand.urdf")
    assert DEMO4_TABLE_OBS_CONTAINS_YAW_RATE is False


def test_demo4_static_runtime_hash_validation_fails_closed(tmp_path) -> None:
    assert validate_demo4_static_runtime(DEMO4_RUNTIME) == (
        DEMO4_STATIC_RUNTIME_SHA256
    )

    corrupted = tmp_path / "corrupted_runtime.npz"
    corrupted.write_bytes(DEMO4_RUNTIME.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_demo4_static_runtime(corrupted)
