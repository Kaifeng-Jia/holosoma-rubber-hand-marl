"""CPU tests for the isolated Demo 4 deterministic evaluator."""

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

from holosoma.agents.mappo.demo4_evaluation import (
    Demo4EpisodeResult,
    deterministic_demo4_actions,
    expected_demo4_metadata,
    representative_episode_index,
    save_viser_episode,
    validate_demo4_evaluation_checkpoint,
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
        "demo4_mappo": expected_demo4_metadata(),
        "actor_model_state_dict": {},
        "critic_model_state_dict": {},
        "actor_optimizer_state_dict": {},
        "critic_optimizer_state_dict": {},
        "actor_obs_normalizer_state_dict": {},
        "critic_obs_normalizer_state_dict": {},
        "iter": 3000,
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


def test_checkpoint_validation_is_strict_and_demo4_only() -> None:
    state = _checkpoint()
    assert validate_demo4_evaluation_checkpoint(state) == 3000

    contaminated = dict(state)
    contaminated["plan5_mappo"] = {}
    with pytest.raises(ValueError, match="cross-demo"):
        validate_demo4_evaluation_checkpoint(contaminated)

    incompatible = copy.deepcopy(state)
    incompatible["demo4_mappo"]["target_yaw_degrees"] = 45.0
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_demo4_evaluation_checkpoint(incompatible)


def test_deterministic_inference_folds_two_agents_through_one_actor() -> None:
    torch.manual_seed(721)
    models = SimpleNamespace(actor=_Actor(), actor_obs_normalizer=_Normalizer())
    observations = {
        "actor_obs": torch.randn(3, 2, 154),
        "teammate_obs": torch.randn(3, 2, 4),
        "table_obs": torch.randn(3, 2, 6),
        "critic_obs": torch.full((3, 527), float("nan")),
    }
    actions_a = deterministic_demo4_actions(models, observations)
    actions_b = deterministic_demo4_actions(models, observations)

    assert actions_a.shape == (3, 2, 29)
    torch.testing.assert_close(actions_a, actions_b)
    assert torch.isfinite(actions_a).all()


def test_viser_five_channel_recording_and_representative_selection(tmp_path: Path) -> None:
    channels = _channels()
    assert validate_viser_five_channels(channels) == 4
    output = save_viser_episode(tmp_path / "episode.npz", channels, metadata={"fps": 50})
    with np.load(output, allow_pickle=False) as saved:
        assert set(channels).issubset(saved.files)
        assert json.loads(str(saved["_metadata_json"]))["fps"] == 50

    results = [
        Demo4EpisodeResult(0, 316, False, False, False, True, 1.2, 1.3, 2.0),
        Demo4EpisodeResult(1, 250, True, False, False, False, 1.57, 1.57, 8.0),
        Demo4EpisodeResult(2, 260, True, False, False, False, 1.58, 1.62, 7.0),
    ]
    assert representative_episode_index(results) == 2


def test_evaluation_entry_help_has_no_isaac_dependency() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    script = repo_root / "scripts" / "evaluate_demo4_rotate.py"
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
