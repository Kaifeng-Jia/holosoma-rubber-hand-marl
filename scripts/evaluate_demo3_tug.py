#!/usr/bin/env python3
"""Deterministically evaluate and record an isolated trained Demo 3 policy."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))
INITIAL_RESET_SANITY_ATOL_BY_CHANNEL = {
    "root_pos": 1.0e-3,
    "root_quat_xyzw": 1.0e-2,
    "dof_pos": 1.0e-2,
    "object_pos_w": 1.0e-3,
    "object_quat_xyzw": 1.0e-3,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / "logs/Demo3Tug/checkpoints/model_07999_actor164_table_neutral.pt"
        ),
        help=(
            "Frozen 164-D Demo 3 warm-start used only to construct the shared "
            "Actor architecture; its SHA256 is validated."
        ),
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    return args


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _snapshot(env: Any, command: Any) -> dict[str, np.ndarray]:
    simulator = env.simulator

    def _array(value: Any) -> np.ndarray:
        return value[0].detach().cpu().numpy().copy()

    return {
        "root_pos": _array(simulator.agent_root_states[..., :3]),
        "root_quat_xyzw": _array(simulator.agent_root_states[..., 3:7]),
        "dof_pos": _array(simulator.agent_dof_pos),
        "object_pos_w": _array(command.simulator_object_pos_w),
        "object_quat_xyzw": _array(command.simulator_object_quat_w),
    }


def _stack_snapshots(snapshots: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not snapshots:
        raise ValueError("Cannot stack an empty Demo 3 episode")
    return {
        name: np.stack([frame[name] for frame in snapshots], axis=0)
        for name in snapshots[0]
    }


def _assert_initial_reset_sanity(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    episode: int,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name, expected_values in expected.items():
        actual_values = actual[name]
        maximum_error = float(np.max(np.abs(actual_values - expected_values)))
        errors[name] = maximum_error
        tolerance = INITIAL_RESET_SANITY_ATOL_BY_CHANNEL[name]
        if not np.allclose(
            actual_values,
            expected_values,
            rtol=0.0,
            atol=tolerance,
        ):
            raise RuntimeError(
                f"Demo 3 episode {episode} initial {name} differs from episode 0; "
                f"maximum absolute error={maximum_error}, "
                f"tolerance={tolerance}"
            )
    return errors


def main() -> int:
    args = _parse_args()

    # All byte-level and checkpoint checks happen before Isaac Sim starts.
    import torch

    from holosoma.agents.mappo.demo3_evaluation import (
        DEMO3_REFERENCE_FRAMES,
        DEMO3_RUNTIME_REFERENCE_RELATIVE_PATH,
        DEMO3_TERMINATION_NAMES,
        DEMO3_WIN_THRESHOLD_M,
        VISER_FIVE_CHANNEL_SHAPES,
        Demo3EpisodeResult,
        classify_demo3_winner,
        deterministic_demo3_actions,
        representative_episode_indices,
        save_viser_episode,
        validate_demo3_evaluation_assets,
        validate_demo3_evaluation_checkpoint,
    )
    from holosoma.config_values.marl.g1.demo3_experiment import (
        g1_29dof_demo3_tug_smoke,
    )
    from holosoma.utils.eval_utils import init_sim_imports

    checkpoint = args.checkpoint.expanduser().resolve()
    warm_start = args.warm_start_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Demo 3 checkpoint does not exist: {checkpoint}")
    asset_paths = validate_demo3_evaluation_assets(
        REPO_ROOT,
        warm_start_path=warm_start,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_iteration = validate_demo3_evaluation_checkpoint(state)

    config = replace(
        g1_29dof_demo3_tug_smoke,
        training=replace(
            g1_29dof_demo3_tug_smoke.training,
            num_envs=1,
            headless=True,
            seed=args.seed,
            project="Demo3TugEval",
            name="deterministic_actor_evaluation",
        ),
    )
    command_term = config.command.setup_terms["paired_motion_command"]
    runtime_label = command_term.params.get("paired_reference_file")
    runtime_path = (REPO_ROOT / "src" / "holosoma" / str(runtime_label)).resolve()
    if runtime_path != Path(asset_paths["runtime_reference"]):
        raise ValueError(
            "Demo 3 config runtime reference differs from the evaluated contract: "
            f"{runtime_path}"
        )
    expected_runtime = (REPO_ROOT / DEMO3_RUNTIME_REFERENCE_RELATIVE_PATH).resolve()
    if runtime_path != expected_runtime:
        raise ValueError(f"Unexpected Demo 3 runtime reference: {runtime_path}")
    table_path = (REPO_ROOT / "src" / "holosoma" / str(config.robot.object.object_urdf_path)).resolve()
    if table_path != Path(asset_paths["square_table_urdf"]):
        raise ValueError(f"Unexpected Demo 3 table URDF: {table_path}")
    if config.robot.asset.urdf_file != "g1/main_mesh_collision_rubberhand.urdf":
        raise ValueError(f"Unexpected Demo 3 robot URDF: {config.robot.asset.urdf_file}")

    simulation_app = init_sim_imports(config)

    from holosoma.agents.mappo.demo3_initialization import (
        initialize_demo3_model_bundle,
    )
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app

    env = None
    failure: BaseException | None = None
    try:
        torch.manual_seed(args.seed)
        env_class = get_class(config.env_class)
        env = env_class(get_tyro_env_config(config), device="cuda:0")
        env.set_is_evaluating()
        if env.num_envs != 1:
            raise RuntimeError(f"Demo 3 evaluation requires one environment, got {env.num_envs}")
        command = env.command_manager.get_state("paired_motion_command")
        if command is None:
            raise RuntimeError("Demo 3 paired_motion_command was not created")
        if int(command.reference.num_frames) != DEMO3_REFERENCE_FRAMES:
            raise RuntimeError(
                f"Demo 3 runtime frame count changed: {command.reference.num_frames}"
            )

        models = initialize_demo3_model_bundle(
            warm_start,
            config.algo.config,
            expected_sha256=state["demo3_mappo"]["source_sha256"],
            device=env.device,
        )
        models.actor.load_state_dict(state["actor_model_state_dict"], strict=True)
        models.actor_obs_normalizer.load_state_dict(
            state["actor_obs_normalizer_state_dict"],
            strict=True,
        )
        models.actor.eval()
        models.actor_obs_normalizer.eval()

        # BaseTask auto-resets during env.step. Capture physical terminal state
        # immediately before that reset; no post-reset pose enters the episode.
        terminal_snapshot: dict[str, np.ndarray] | None = None
        capture_terminal = False
        original_reset = env.reset_envs_idx

        def _capturing_reset(env_ids, target_states=None, target_buf=None):
            nonlocal terminal_snapshot
            ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
            if capture_terminal and bool(torch.any(ids == 0).item()):
                terminal_snapshot = _snapshot(env, command)
            return original_reset(env_ids, target_states, target_buf)

        env.reset_envs_idx = _capturing_reset

        results: list[Demo3EpisodeResult] = []
        trajectories: list[dict[str, np.ndarray]] = []
        pull_axes: list[tuple[float, float]] = []
        expected_initial_snapshot: dict[str, np.ndarray] | None = None
        initial_reset_max_abs_error = {
            name: 0.0 for name in VISER_FIVE_CHANNEL_SHAPES
        }
        expected_start_phase: int | None = None
        max_steps = int(command.reference.num_frames) + 1
        for episode in range(args.episodes):
            # Use the same explicit reset + zero-action settling step for every
            # episode.  Reusing the observation returned by an auto-reset would
            # make episode 0 and later episodes start from different physics paths.
            observations = env.reset_all()
            start_phase = int(command.time_steps[0].item())
            if expected_start_phase is None:
                expected_start_phase = start_phase
            elif start_phase != expected_start_phase:
                raise RuntimeError(
                    "Demo 3 episodes started from different reference phases: "
                    f"episode 0={expected_start_phase}, "
                    f"episode {episode}={start_phase}"
                )
            initial_snapshot = _snapshot(env, command)
            if expected_initial_snapshot is None:
                expected_initial_snapshot = {
                    name: values.copy() for name, values in initial_snapshot.items()
                }
            else:
                initial_errors = _assert_initial_reset_sanity(
                    expected_initial_snapshot,
                    initial_snapshot,
                    episode=episode,
                )
                for name, error in initial_errors.items():
                    initial_reset_max_abs_error[name] = max(
                        initial_reset_max_abs_error[name],
                        error,
                    )
            initial_table_xy = initial_snapshot["object_pos_w"][:2].copy()
            initial_pull_axis_a = (
                command.agent_pull_axis_w[0, 0].detach().cpu().numpy().copy()
            )
            if not np.isclose(np.linalg.norm(initial_pull_axis_a), 1.0, atol=1.0e-5):
                raise RuntimeError("Demo 3 Agent A initial Pull axis is not unit length")
            frames = [initial_snapshot]
            terminal_snapshot = None
            capture_terminal = True
            reward_sums = np.zeros(2, dtype=np.float64)
            terminal_terms = {name: False for name in DEMO3_TERMINATION_NAMES}
            for step in range(1, max_steps + 1):
                actions = deterministic_demo3_actions(models, observations)
                observations, rewards, dones, extras = env.step({"actions": actions})
                reward_array = rewards[0].detach().cpu().numpy()
                if reward_array.shape != (2,):
                    raise RuntimeError(
                        f"Demo 3 per-agent rewards must have shape (2,), got {reward_array.shape}"
                    )
                reward_sums += reward_array
                done = bool(dones[0].item())
                if done:
                    terms = extras.get("termination_terms", {})
                    terminal_terms = {
                        name: bool(
                            terms.get(name, torch.zeros(1, device=env.device))[0].item()
                        )
                        for name in DEMO3_TERMINATION_NAMES
                    }
                    time_outs = extras.get("time_outs")
                    if isinstance(time_outs, torch.Tensor):
                        terminal_terms["reference_horizon"] = bool(time_outs[0].item())
                    if terminal_snapshot is None:
                        raise RuntimeError(
                            "Demo 3 terminal state was not captured before auto-reset"
                        )
                    frames.append(terminal_snapshot)
                    break
                frames.append(_snapshot(env, command))
            else:
                raise RuntimeError(
                    f"Demo 3 episode {episode} exceeded {max_steps} steps without termination"
                )
            capture_terminal = False
            expected_horizon_steps = int(command.reference.num_frames) - start_phase
            if terminal_terms["reference_horizon"] and step != expected_horizon_steps:
                raise RuntimeError(
                    "Demo 3 horizon episode used the wrong number of Actor steps: "
                    f"expected {expected_horizon_steps}, got {step}"
                )

            planar_delta = terminal_snapshot["object_pos_w"][:2] - initial_table_xy
            signed_displacement = float(
                np.dot(planar_delta, initial_pull_axis_a[:2])
            )
            winner = classify_demo3_winner(
                signed_displacement,
                threshold_m=DEMO3_WIN_THRESHOLD_M,
            )
            result = Demo3EpisodeResult(
                episode=episode,
                steps=step,
                winner=winner,
                signed_displacement_m=signed_displacement,
                planar_displacement_xy_m=(float(planar_delta[0]), float(planar_delta[1])),
                fall=terminal_terms["clear_robot_fall"],
                timeout=terminal_terms["reference_horizon"],
                reward_sum_agent_a=float(reward_sums[0]),
                reward_sum_agent_b=float(reward_sums[1]),
            )
            results.append(result)
            trajectories.append(_stack_snapshots(frames))
            pull_axes.append(
                (float(initial_pull_axis_a[0]), float(initial_pull_axis_a[1]))
            )
            print(json.dumps(result.as_dict(), sort_keys=True), flush=True)

        selection_indices = representative_episode_indices(results)
        selected_paths: dict[int, Path] = {}
        representatives: dict[str, dict[str, Any]] = {}
        labels_by_index: dict[int, list[str]] = {}
        for label, index in selection_indices.items():
            labels_by_index.setdefault(index, []).append(label)
        for index, labels in labels_by_index.items():
            result = results[index]
            path = save_viser_episode(
                output_dir / f"representative_episode_{result.episode:03d}.npz",
                trajectories[index],
                metadata={
                    "scenario": "competitive_square_table_diagonal_tug",
                    "checkpoint": str(checkpoint),
                    "checkpoint_iteration": checkpoint_iteration,
                    "seed": args.seed,
                    "episode": result.episode,
                    "fps": round(1.0 / float(env.dt)),
                    "selection_labels": sorted(labels),
                    "win_axis": "agent_a_initial_pull_axis_world_xy",
                    "agent_a_initial_pull_axis_world_xy": list(pull_axes[index]),
                    "strict_win_threshold_m": DEMO3_WIN_THRESHOLD_M,
                    "result": result.as_dict(),
                },
            )
            selected_paths[index] = path
        for label, index in selection_indices.items():
            representatives[label] = {
                "episode": results[index].episode,
                "path": str(selected_paths[index]),
            }

        counts = {
            winner: sum(result.winner == winner for result in results)
            for winner in ("agent_a", "agent_b", "draw")
        }
        signed_values = np.asarray(
            [result.signed_displacement_m for result in results], dtype=np.float64
        )
        summary = {
            "scenario": "competitive_square_table_diagonal_tug",
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": checkpoint_iteration,
            "seed": args.seed,
            "episodes": args.episodes,
            "deterministic_inference": True,
            "inference_network": "shared_actor_only_critic_not_called",
            "episode_initialization": (
                "explicit_reset_all_with_zero_action_settle_before_every_episode"
            ),
            "consistent_explicit_reset_path": True,
            "equal_reference_start_phase_assertion": True,
            "reference_start_phase": expected_start_phase,
            "initial_reset_sanity_assertion": True,
            "initial_reset_sanity_absolute_tolerance_by_channel": (
                INITIAL_RESET_SANITY_ATOL_BY_CHANNEL
            ),
            "initial_reset_max_absolute_error_by_channel": (
                initial_reset_max_abs_error
            ),
            "horizon_actor_step_count_assertion": True,
            "winner_definition": {
                "axis": "agent_a_initial_pull_axis_world_xy",
                "agent_a": f"signed_displacement_m > {DEMO3_WIN_THRESHOLD_M}",
                "agent_b": f"signed_displacement_m < {-DEMO3_WIN_THRESHOLD_M}",
                "draw": (
                    f"{-DEMO3_WIN_THRESHOLD_M} <= signed_displacement_m "
                    f"<= {DEMO3_WIN_THRESHOLD_M}"
                ),
            },
            "win_counts": counts,
            "win_rates": {
                name: value / args.episodes for name, value in counts.items()
            },
            "fall_count": sum(result.fall for result in results),
            "timeout_count": sum(result.timeout for result in results),
            "signed_displacement_mean_m": float(np.mean(signed_values)),
            "signed_displacement_min_m": float(np.min(signed_values)),
            "signed_displacement_max_m": float(np.max(signed_values)),
            "representative_rule": (
                "strongest_agent_a_win_and_strongest_agent_b_win_when_present_plus_"
                "largest_absolute_displacement; duplicate episodes share one NPZ"
            ),
            "representatives": representatives,
            "validated_assets": asset_paths,
            "episode_results": [result.as_dict() for result in results],
        }
        _write_json(output_dir / "evaluation.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        failure = error
        traceback.print_exc()
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(simulation_app)
        if failure is not None:
            print(f"Demo 3 evaluation failed: {failure}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
