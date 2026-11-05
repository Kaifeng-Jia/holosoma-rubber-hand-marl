from __future__ import annotations

import numpy as np
import pytest
import torch
import tyro

from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.managers.command.terms.wbt import (
    HierarchicalAdaptiveTimestepsSampler,
    MultiMotionLoader,
    normalize_motion_sampling_weights,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _write_motion(path, frames: int) -> None:
    joint_names = np.array(["joint"])
    body_names = np.array(["pelvis", "link"])
    body_quat = np.zeros((frames, 2, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    object_quat = np.zeros((frames, 4), dtype=np.float32)
    object_quat[..., 0] = 1.0

    np.savez(
        path,
        fps=np.array([50], dtype=np.int64),
        joint_pos=np.zeros((frames, 8), dtype=np.float32),
        joint_vel=np.zeros((frames, 7), dtype=np.float32),
        body_pos_w=np.zeros((frames, 2, 3), dtype=np.float32),
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((frames, 2, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 2, 3), dtype=np.float32),
        object_pos_w=np.zeros((frames, 3), dtype=np.float32),
        object_quat_w=object_quat,
        object_lin_vel_w=np.zeros((frames, 3), dtype=np.float32),
        object_ang_vel_w=np.zeros((frames, 3), dtype=np.float32),
        joint_names=joint_names,
        body_names=body_names,
    )


def test_explicit_motion_files_preserve_order_and_boundaries(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_motion(first, frames=7)
    _write_motion(second, frames=11)

    loader = MultiMotionLoader(
        "",
        robot_body_names=["pelvis", "link"],
        robot_joint_names=["joint"],
        motion_files=[str(second), str(first)],
    )

    assert loader.motion_names == ["second", "first"]
    assert loader.motion_files == [str(second), str(first)]
    assert loader.motion_start_idx.tolist() == [0, 11]
    assert loader.motion_end_idx.tolist() == [11, 18]
    assert loader.time_step_total == 18
    assert loader.has_object


def test_explicit_motion_files_fail_closed(tmp_path) -> None:
    valid = tmp_path / "valid.npz"
    invalid = tmp_path / "invalid.npz"
    _write_motion(valid, frames=7)
    np.savez(invalid, fps=np.array([50], dtype=np.int64))

    with pytest.raises(ValueError, match="Failed to load frozen motion file"):
        MultiMotionLoader(
            "",
            robot_body_names=["pelvis", "link"],
            robot_joint_names=["joint"],
            motion_files=[str(valid), str(invalid)],
        )


def test_explicit_motion_files_reject_duplicates(tmp_path) -> None:
    motion = tmp_path / "motion.npz"
    _write_motion(motion, frames=7)

    with pytest.raises(ValueError, match="duplicate"):
        MultiMotionLoader(
            "",
            robot_body_names=["pelvis", "link"],
            robot_joint_names=["joint"],
            motion_files=[str(motion), str(motion)],
        )


def test_tyro_cli_parses_explicit_motion_files_and_weights() -> None:
    key = "--command.setup-terms.motion-command.params.motion-config."
    config = tyro.cli(
        AnnotatedExperimentConfig,
        args=[
            "exp:g1-29dof-wbt-w-object",
            f"{key}motion-files",
            "['a.npz','b.npz','c.npz','d.npz']",
            f"{key}motion-sampling-weights",
            "[1,1,2,2]",
        ],
        config=TYRO_CONIFG,
    )
    motion_config = config.command.setup_terms["motion_command"].params["motion_config"]

    assert motion_config.motion_files == ["a.npz", "b.npz", "c.npz", "d.npz"]
    assert motion_config.motion_sampling_weights == [1.0, 1.0, 2.0, 2.0]


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0], "exactly one value"),
        ([1.0, -1.0], "non-negative"),
        ([0.0, 0.0], "at least one positive"),
        ([1.0, float("nan")], "finite"),
    ],
)
def test_motion_sampling_weights_fail_closed(weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_motion_sampling_weights(weights, num_motions=2, device="cpu")


def test_hierarchical_sampler_preserves_motion_probabilities_and_boundaries() -> None:
    torch.manual_seed(7)
    starts = torch.tensor([0, 309, 618, 917], dtype=torch.long)
    ends = torch.tensor([309, 618, 917, 1234], dtype=torch.long)
    weights = normalize_motion_sampling_weights([1.0, 1.0, 2.0, 2.0], 4, "cpu")
    sampler = HierarchicalAdaptiveTimestepsSampler(starts, ends, weights, "cpu", env_fps=50)

    motion_ids, global_steps = sampler.sample(120_000)
    observed = torch.bincount(motion_ids, minlength=4).float() / motion_ids.numel()

    assert torch.allclose(observed, weights, atol=0.005)
    assert torch.all(global_steps >= starts[motion_ids])
    assert torch.all(global_steps < ends[motion_ids])


def test_adaptive_failures_stay_inside_selected_motion() -> None:
    torch.manual_seed(11)
    starts = torch.tensor([0, 100, 200], dtype=torch.long)
    ends = torch.tensor([100, 200, 300], dtype=torch.long)
    weights = normalize_motion_sampling_weights([0.2, 0.3, 0.5], 3, "cpu")
    sampler = HierarchicalAdaptiveTimestepsSampler(starts, ends, weights, "cpu", env_fps=50)

    untouched_before = [inner.sampling_probabilities.clone() for inner in sampler.samplers[1:]]
    sampler.update_current_bin_failed_count(
        failed_motion_ids=torch.zeros(1_000, dtype=torch.long),
        failed_at_time_step=torch.full((1_000,), 75, dtype=torch.long),
    )
    sampler.update_bin_failed_count()

    assert sampler.samplers[0].sampling_probabilities.max() > (1.0 / sampler.samplers[0].num_bins)
    for before, inner in zip(untouched_before, sampler.samplers[1:]):
        assert torch.allclose(inner.sampling_probabilities, before)

    motion_ids, _ = sampler.sample(100_000)
    observed = torch.bincount(motion_ids, minlength=3).float() / motion_ids.numel()
    assert torch.allclose(observed, weights, atol=0.005)
