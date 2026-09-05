#!/usr/bin/env python3
"""Deterministically evaluate and record an isolated trained Demo 4 policy."""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / "logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/"
            "model_08050.pt"
        ),
        help="Frozen Pull08050 source used only to construct the Demo 4 Actor architecture.",
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
        raise ValueError("Cannot stack an empty Demo 4 episode")
    return {
        name: np.stack([frame[name] for frame in snapshots], axis=0)
        for name in snapshots[0]
    }


def main() -> int:
    args = _parse_args()

    from holosoma.agents.mappo.demo4_evaluation import (
        DEMO4_TERMINATION_NAMES,
        Demo4EpisodeResult,
        deterministic_demo4_actions,
        representative_episode_index,
        save_viser_episode,
        validate_demo4_evaluation_checkpoint,
    )
    from holosoma.config_values.marl.g1.demo4_experiment import (
        g1_29dof_demo4_rotate_smoke,
    )
    from holosoma.agents.mappo.demo4_checkpoint import (
        validate_demo4_static_runtime,
    )
    from holosoma.utils.eval_utils import init_sim_imports

    checkpoint = args.checkpoint.expanduser().resolve()
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Demo 4 checkpoint does not exist: {checkpoint}")
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Pull08050 source checkpoint does not exist: {source_checkpoint}")

    config = replace(
        g1_29dof_demo4_rotate_smoke,
        training=replace(
            g1_29dof_demo4_rotate_smoke.training,
            num_envs=1,
            headless=True,
            seed=args.seed,
            project="Demo4RotateEval",
            name="deterministic_actor_evaluation",
        ),
    )
    command_term = config.command.setup_terms["paired_motion_command"]
    runtime_reference_label = command_term.params.get("paired_reference_file")
    if not isinstance(runtime_reference_label, str) or not runtime_reference_label:
        raise ValueError("Demo 4 paired_reference_file must be a non-empty string")
    runtime_reference_path = (
        REPO_ROOT / "src" / "holosoma" / runtime_reference_label
    ).resolve()
    validate_demo4_static_runtime(runtime_reference_path)
    simulation_app = init_sim_imports(config)

    import torch

    from holosoma.agents.mappo.demo4_initialization import (
        initialize_demo4_model_bundle,
    )
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app

    env = None
    failure: BaseException | None = None
    try:
        torch.manual_seed(args.seed)
        state = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        checkpoint_iteration = validate_demo4_evaluation_checkpoint(state)

        env_class = get_class(config.env_class)
        env = env_class(get_tyro_env_config(config), device="cuda:0")
        env.set_is_evaluating()
        observations = env.reset_all()
        if env.num_envs != 1:
            raise RuntimeError(f"Demo 4 evaluation requires one environment, got {env.num_envs}")
        command = env.command_manager.get_state("paired_motion_command")
        if command is None:
            raise RuntimeError("Demo 4 paired_motion_command was not created")

        models = initialize_demo4_model_bundle(
            source_checkpoint,
            config.algo.config,
            device=env.device,
        )
        models.actor.load_state_dict(state["actor_model_state_dict"], strict=True)
        models.actor_obs_normalizer.load_state_dict(
            state["actor_obs_normalizer_state_dict"],
            strict=True,
        )
        models.actor.eval()
        models.actor_obs_normalizer.eval()

        # BaseTask auto-resets inside env.step.  Capture the terminal physical
        # state immediately before that reset so the ViSER trajectory does not
        # end with the next episode's initial pose.
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

        results: list[Demo4EpisodeResult] = []
        trajectories: list[dict[str, np.ndarray]] = []
        max_steps = int(command.reference.num_frames) + 1
        for episode in range(args.episodes):
            frames = [_snapshot(env, command)]
            terminal_snapshot = None
            capture_terminal = True
            reward_sum = 0.0
            max_yaw = float(command.signed_yaw_progress_radians[0].item())
            terminal_terms = {name: False for name in DEMO4_TERMINATION_NAMES}
            final_yaw = max_yaw
            for step in range(1, max_steps + 1):
                actions = deterministic_demo4_actions(models, observations)
                observations, rewards, dones, extras = env.step({"actions": actions})
                reward_sum += float(rewards[0].item())
                to_log = extras.get("to_log", {})
                yaw_value = to_log.get("rotate/yaw_progress_rad")
                if isinstance(yaw_value, torch.Tensor):
                    final_yaw = float(yaw_value[0].item())
                else:
                    final_yaw = float(command.signed_yaw_progress_radians[0].item())
                max_yaw = max(max_yaw, final_yaw)
                done = bool(dones[0].item())
                if done:
                    terms = extras.get("termination_terms", {})
                    terminal_terms = {
                        name: bool(terms.get(name, torch.zeros(1, device=env.device))[0].item())
                        for name in DEMO4_TERMINATION_NAMES
                    }
                    time_outs = extras.get("time_outs")
                    if isinstance(time_outs, torch.Tensor):
                        # Demo4RotateManager masks a simultaneous horizon when
                        # a physical success/failure takes precedence.  Report
                        # the effective timeout, not the raw horizon predicate.
                        terminal_terms["reference_horizon"] = bool(time_outs[0].item())
                    if terminal_snapshot is None:
                        raise RuntimeError("Demo 4 terminal state was not captured before reset")
                    frames.append(terminal_snapshot)
                    break
                frames.append(_snapshot(env, command))
            else:
                raise RuntimeError(
                    f"Demo 4 episode {episode} exceeded {max_steps} steps without termination"
                )
            capture_terminal = False
            result = Demo4EpisodeResult(
                episode=episode,
                steps=step,
                success=terminal_terms["yaw_goal_success"],
                fall=terminal_terms["clear_robot_fall"],
                table_safety=terminal_terms["table_physical_safety"],
                timeout=terminal_terms["reference_horizon"],
                final_unwrapped_yaw_rad=final_yaw,
                max_unwrapped_yaw_rad=max_yaw,
                reward_sum=reward_sum,
            )
            results.append(result)
            trajectories.append(_stack_snapshots(frames))
            print(json.dumps(result.as_dict(), sort_keys=True), flush=True)

        representative_index = representative_episode_index(results)
        representative = results[representative_index]
        viser_path = save_viser_episode(
            output_dir / "representative_episode.npz",
            trajectories[representative_index],
            metadata={
                "scenario": "cooperative_rectangular_table_rotate_90deg",
                "checkpoint": str(checkpoint),
                "checkpoint_iteration": checkpoint_iteration,
                "seed": args.seed,
                "episode": representative.episode,
                "fps": round(1.0 / float(env.dt)),
                "selection": "success_then_max_unwrapped_yaw",
                "result": representative.as_dict(),
            },
        )
        episode_payload = [result.as_dict() for result in results]
        summary = {
            "scenario": "cooperative_rectangular_table_rotate_90deg",
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": checkpoint_iteration,
            "seed": args.seed,
            "episodes": args.episodes,
            "deterministic_inference": True,
            "inference_network": "shared_actor_only_critic_not_called",
            "success_count": sum(result.success for result in results),
            "success_rate": sum(result.success for result in results) / args.episodes,
            "fall_count": sum(result.fall for result in results),
            "table_safety_count": sum(result.table_safety for result in results),
            "timeout_count": sum(result.timeout for result in results),
            "representative_episode": representative.episode,
            "representative_viser_npz": str(viser_path),
            "episode_results": episode_payload,
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
            print(f"Demo 4 evaluation failed: {failure}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
