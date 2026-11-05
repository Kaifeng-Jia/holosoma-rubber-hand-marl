"""CPU tests for the isolated Demo 3 deterministic evaluator."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from holosoma.agents.mappo.demo3_evaluation import (
    DEMO3_WIN_THRESHOLD_M,
    Demo3EpisodeResult,
    classify_demo3_winner,
    deterministic_demo3_actions,
    expected_demo3_metadata,
    representative_episode_indices,
    save_viser_episode,
    validate_demo3_evaluation_assets,
    validate_demo3_evaluation_checkpoint,
    validate_viser_five_channels,
)


class _Normalizer(nn.Module):
    def forward(self, values: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        assert update is False
        return values


class _Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(164, 29, bias=False)

    def act_inference(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.linear(state["actor_obs"])


def _checkpoint() -> dict:
    return {
        "demo3_mappo": expected_demo3_metadata(),
        "actor_model_state_dict": {},
        "critic_model_state_dict": {},
        "actor_optimizer_state_dict": {},
        "critic_optimizer_state_dict": {},
        "actor_obs_normalizer_state_dict": {},
        "critic_obs_normalizer_state_dict": {},
        "iter": 8000,
    }


def _channels(frames: int = 4) -> dict[str, np.ndarray]:
    root_quat = np.zeros((frames, 2, 4), dtype=np.float32)
    root_quat[..., 3] = 1.0
    object_quat = np.zeros((frames, 4), dtype=np.float32)
    object_quat[..., 3] = 1.0
    return {
        "root_pos": np.zeros((frames, 2, 3), dtype=np.float32),
        "root_quat_xyzw": root_quat,
        "dof_pos": np.zeros((frames, 2, 29), dtype=np.float32),
        "object_pos_w": np.zeros((frames, 3), dtype=np.float32),
        "object_quat_xyzw": object_quat,
    }


def _result(episode: int, displacement: float) -> Demo3EpisodeResult:
    return Demo3EpisodeResult(
        episode=episode,
        steps=317,
        winner=classify_demo3_winner(displacement),
        signed_displacement_m=displacement,
        planar_displacement_xy_m=(displacement, 0.0),
        fall=False,
        timeout=True,
        reward_sum_agent_a=1.0,
        reward_sum_agent_b=-1.0,
    )


def test_checkpoint_validation_is_strict_and_demo3_only() -> None:
    state = _checkpoint()
    assert validate_demo3_evaluation_checkpoint(state) == 8000

    contaminated = dict(state)
    contaminated["demo4_mappo"] = {}
    with pytest.raises(ValueError, match="cross-demo"):
        validate_demo3_evaluation_checkpoint(contaminated)

    incompatible = copy.deepcopy(state)
    incompatible["demo3_mappo"]["source_sha256"] = "wrong-source"
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_demo3_evaluation_checkpoint(incompatible)

    incompatible_ppo = copy.deepcopy(state)
    incompatible_ppo["demo3_mappo"]["ppo_contract"]["gamma"] = 0.5
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_demo3_evaluation_checkpoint(incompatible_ppo)

    custom_schedule = _checkpoint()
    custom_schedule["demo3_mappo"] = expected_demo3_metadata(
        critic_only_iterations=5,
        full_actor_iterations=20,
        checkpoint_interval=10,
    )
    assert validate_demo3_evaluation_checkpoint(custom_schedule) == 8000

    custom_steps = _checkpoint()
    custom_steps["demo3_mappo"] = expected_demo3_metadata(num_steps_per_env=4)
    assert validate_demo3_evaluation_checkpoint(custom_steps) == 8000

    inconsistent_contract = copy.deepcopy(custom_schedule)
    inconsistent_contract["demo3_mappo"]["reference_frames"] = 999
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_demo3_evaluation_checkpoint(inconsistent_contract)


def test_deterministic_inference_folds_two_agents_through_actor_only() -> None:
    torch.manual_seed(721)
    models = SimpleNamespace(actor=_Actor(), actor_obs_normalizer=_Normalizer())
    observations = {
        "actor_obs": torch.randn(3, 2, 154),
        "teammate_obs": torch.randn(3, 2, 4),
        "table_obs": torch.randn(3, 2, 6),
        # Deliberately invalid Critic input: actor-only inference must ignore it.
        "critic_obs": torch.full((3, 2, 527), float("nan")),
    }
    actions_a = deterministic_demo3_actions(models, observations)
    actions_b = deterministic_demo3_actions(models, observations)

    assert actions_a.shape == (3, 2, 29)
    torch.testing.assert_close(actions_a, actions_b)
    assert torch.isfinite(actions_a).all()


def test_winner_threshold_is_strict_and_representatives_are_balanced() -> None:
    assert classify_demo3_winner(DEMO3_WIN_THRESHOLD_M) == "draw"
    assert classify_demo3_winner(-DEMO3_WIN_THRESHOLD_M) == "draw"
    assert classify_demo3_winner(DEMO3_WIN_THRESHOLD_M + 1.0e-6) == "agent_a"
    assert classify_demo3_winner(-DEMO3_WIN_THRESHOLD_M - 1.0e-6) == "agent_b"

    results = [
        _result(0, 0.01),
        _result(1, 0.08),
        _result(2, 0.12),
        _result(3, -0.09),
        _result(4, -0.15),
    ]
    assert representative_episode_indices(results) == {
        "max_absolute_displacement": 4,
        "agent_a_win": 2,
        "agent_b_win": 4,
    }


def test_real_runtime_and_asset_contract_is_frozen() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    assets = validate_demo3_evaluation_assets(repo_root)
    assert set(assets) == {
        "runtime_reference",
        "rubber_hand_robot_urdf",
        "square_table_urdf",
        "actor164_warm_start",
    }


def test_viser_five_channel_recording(tmp_path: Path) -> None:
    channels = _channels()
    assert validate_viser_five_channels(channels) == 4
    output = save_viser_episode(
        tmp_path / "episode.npz",
        channels,
        metadata={"fps": 50, "selection_labels": ["agent_a_win"]},
    )
    with np.load(output, allow_pickle=False) as saved:
        assert set(channels).issubset(saved.files)
        metadata = json.loads(str(saved["_metadata_json"]))
        assert metadata["fps"] == 50
        assert metadata["selection_labels"] == ["agent_a_win"]


def test_evaluation_entry_help_has_no_isaac_dependency() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    script = repo_root / "scripts" / "evaluate_demo3_tug.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--episodes" in result.stdout
    assert "--seed" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--warm-start-checkpoint" in result.stdout
