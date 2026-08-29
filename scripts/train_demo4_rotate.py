#!/usr/bin/env python3
"""Train the isolated cooperative Demo 4 rectangular-table rotation policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--iterations", type=int, default=15000)
PARSER.add_argument("--num-envs", type=int, default=4096)
PARSER.add_argument("--steps-per-env", type=int, default=24)
PARSER.add_argument("--seed", type=int, default=721)
PARSER.add_argument("--checkpoint-interval", type=int, default=150)
PARSER.add_argument("--output-dir", type=Path, default=None)
PARSER.add_argument("--resume", type=Path, default=None)
PARSER.add_argument("--actor-learning-rate", type=float, default=None)
PARSER.add_argument("--critic-learning-rate", type=float, default=None)
ARGS = PARSER.parse_args()
if ARGS.output_dir is None:
    ARGS.output_dir = (
        REPO_ROOT
        / "logs/Demo4Rotate"
        / (
            "rectangular_pull_pull_rotate90_"
            f"seed{ARGS.seed}_env{ARGS.num_envs}"
        )
    )
for name in ("iterations", "num_envs", "steps_per_env", "checkpoint_interval"):
    if getattr(ARGS, name) < 1:
        PARSER.error(f"--{name.replace('_', '-')} must be positive")
for name in ("actor_learning_rate", "critic_learning_rate"):
    value = getattr(ARGS, name)
    if value is not None and value <= 0.0:
        PARSER.error(f"--{name.replace('_', '-')} must be positive")
try:
    RUNTIME_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
except ValueError:
    PARSER.error("WORLD_SIZE must be an integer")
if RUNTIME_WORLD_SIZE != 1:
    PARSER.error(
        "Demo 4 uses one independent single-GPU process; WORLD_SIZE must be 1"
    )

from holosoma.config_values.marl.g1.demo4_experiment import (  # noqa: E402
    g1_29dof_demo4_rotate_baseline,
)
from holosoma.agents.mappo.demo4_checkpoint import (  # noqa: E402
    DEMO4_OBJECT_MASS_KG,
    DEMO4_OBJECT_MATERIAL,
    DEMO4_STATIC_RUNTIME_SHA256,
    is_demo4_periodic_checkpoint,
    validate_demo4_static_runtime,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402


CONFIG = replace(
    g1_29dof_demo4_rotate_baseline,
    training=replace(
        g1_29dof_demo4_rotate_baseline.training,
        num_envs=ARGS.num_envs,
        headless=True,
        seed=ARGS.seed,
        project="Demo4Rotate",
        name="rectangular_pull_pull_rotate90_mappo",
    ),
)
COMMAND_TERM = CONFIG.command.setup_terms["paired_motion_command"]
RUNTIME_REFERENCE_LABEL = COMMAND_TERM.params.get("paired_reference_file")
if not isinstance(RUNTIME_REFERENCE_LABEL, str) or not RUNTIME_REFERENCE_LABEL:
    raise ValueError("Demo 4 paired_reference_file must be a non-empty string")
RUNTIME_REFERENCE_PATH = (
    REPO_ROOT / "src" / "holosoma" / RUNTIME_REFERENCE_LABEL
).resolve()
RUNTIME_REFERENCE_ACTUAL_SHA256 = validate_demo4_static_runtime(
    RUNTIME_REFERENCE_PATH
)
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.agents.mappo.demo4_initialization import (  # noqa: E402
    DEMO4_SOURCE_PULL_CHECKPOINT_SHA256,
    initialize_demo4_model_bundle,
)
from holosoma.agents.mappo.demo4_ppo import Demo4PPO  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt"
)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _runtime_object_physics(env) -> dict[str, object]:
    view = env.simulator._object.root_physx_view
    mass = view.get_masses().reshape(-1)
    material = view.get_material_properties()
    expected_mass = torch.full_like(mass, DEMO4_OBJECT_MASS_KG)
    expected_material = torch.as_tensor(
        DEMO4_OBJECT_MATERIAL,
        device=material.device,
        dtype=material.dtype,
    ).expand_as(material)
    if not torch.allclose(mass, expected_mass, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError(f"Demo 4 object mass mismatch: {mass.detach().cpu().tolist()}")
    if not torch.allclose(material, expected_material, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError(
            "Demo 4 object material mismatch: "
            f"{material.detach().cpu().tolist()}"
        )
    return {
        "mass_kg": mass.detach().cpu().tolist(),
        "material_static_dynamic_restitution": material.detach().cpu().tolist(),
        "inertia_kg_m2_row_major": view.get_inertias().detach().cpu().tolist(),
        "com_pose_body_xyzw": view.get_coms().detach().cpu().tolist(),
    }


def main() -> int:
    env = None
    failure: BaseException | None = None
    output_dir = ARGS.output_dir.expanduser().resolve()
    owns_output_dir = False
    try:
        torch.manual_seed(ARGS.seed)
        if ARGS.resume is None and output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose another path or use --resume."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        owns_output_dir = True
        metrics_path = output_dir / "metrics.jsonl"

        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        observations = env.reset_all()
        physics = _runtime_object_physics(env)
        ppo_overrides = {"num_steps_per_env": ARGS.steps_per_env}
        if ARGS.actor_learning_rate is not None:
            ppo_overrides["actor_learning_rate"] = ARGS.actor_learning_rate
        if ARGS.critic_learning_rate is not None:
            ppo_overrides["critic_learning_rate"] = ARGS.critic_learning_rate
        ppo_config = replace(CONFIG.algo.config, **ppo_overrides)
        models = initialize_demo4_model_bundle(
            SOURCE_CHECKPOINT,
            ppo_config,
            device=env.device,
        )
        learner = Demo4PPO(
            models,
            ppo_config,
            num_envs=env.num_envs,
            num_steps_per_env=ARGS.steps_per_env,
            device=env.device,
        )

        start_iteration = 0
        if ARGS.resume is not None:
            resume_state = torch.load(
                ARGS.resume.expanduser().resolve(),
                map_location=env.device,
                weights_only=False,
            )
            start_iteration = learner.load_training_state_dict(
                resume_state,
                actor_learning_rate=ARGS.actor_learning_rate,
                critic_learning_rate=ARGS.critic_learning_rate,
            )

        actor_lr_source = (
            "command_line_override"
            if ARGS.actor_learning_rate is not None
            else "restored_checkpoint"
            if ARGS.resume is not None
            else "configured_default"
        )
        critic_lr_source = (
            "command_line_override"
            if ARGS.critic_learning_rate is not None
            else "restored_checkpoint"
            if ARGS.resume is not None
            else "configured_default"
        )
        run_config = {
            "git_commit": _git_commit(),
            "started_unix_time": time.time(),
            "scenario": "cooperative_rectangular_table_rotate_90deg",
            "config": "g1_29dof_demo4_rotate_baseline",
            "output_dir": str(output_dir),
            "seed": ARGS.seed,
            "iterations_requested": ARGS.iterations,
            "start_iteration": start_iteration,
            "num_envs": env.num_envs,
            "num_agents": 2,
            "steps_per_env": ARGS.steps_per_env,
            "checkpoint_interval": ARGS.checkpoint_interval,
            "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(REPO_ROOT)),
            "source_checkpoint_sha256": models.source_sha256,
            "expected_source_checkpoint_sha256": DEMO4_SOURCE_PULL_CHECKPOINT_SHA256,
            "source_iteration": models.source_iteration,
            "source_state_loaded": "actor_and_actor_normalizer_only",
            "critic_initialization": "fresh",
            "optimizer_initialization": (
                "restored_from_demo4_resume" if ARGS.resume is not None else "fresh"
            ),
            "actor_parameters_shared": True,
            "actor_forward_contract": "[E,2,164] -> [E*2,164] -> [E,2,29]",
            "actor_observation_groups": {
                "actor_obs": 154,
                "teammate_obs": 4,
                "table_obs": 6,
            },
            "table_observation_contains_yaw_rate": False,
            "critic_observation_dim": 527,
            "critic_forward_contract": "[E,527] -> [E,1]",
            "return_model": "team_reward_team_value_team_gae",
            "goal_yaw_degrees": COMMAND_TERM.params.get("goal_yaw_degrees"),
            "reference_file": RUNTIME_REFERENCE_LABEL,
            "reference_file_sha256_expected": DEMO4_STATIC_RUNTIME_SHA256,
            "reference_file_sha256_actual": RUNTIME_REFERENCE_ACTUAL_SHA256,
            "reference_table_policy": "reset_only_static_no_trajectory_tracking",
            "robot_urdf": CONFIG.robot.asset.urdf_file,
            "object_urdf": CONFIG.robot.object.object_urdf_path,
            "object_physics": physics,
            "reward_terms": list(env.reward_manager.active_terms),
            "termination_terms": list(env.termination_manager.active_terms),
            "gamma": ppo_config.gamma,
            "lambda": ppo_config.lam,
            "clip_param": ppo_config.clip_param,
            "entropy_coef": ppo_config.entropy_coef,
            "actor_learning_rate": learner.actor_learning_rate,
            "critic_learning_rate": learner.critic_learning_rate,
            "learning_rates": {
                "actor": {
                    "requested_override": ARGS.actor_learning_rate,
                    "actual": learner.actor_learning_rate,
                    "source": actor_lr_source,
                },
                "critic": {
                    "requested_override": ARGS.critic_learning_rate,
                    "actual": learner.critic_learning_rate,
                    "source": critic_lr_source,
                },
            },
            "num_learning_epochs": ppo_config.num_learning_epochs,
            "num_mini_batches": ppo_config.num_mini_batches,
            "runtime_process": {
                "distributed_data_parallel": False,
                "device_within_visible_set": "cuda:0",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "world_size": RUNTIME_WORLD_SIZE,
            },
        }
        _write_json(output_dir / "run_config.json", run_config)
        if ARGS.resume is None:
            torch.save(
                learner.training_state_dict(iteration=0),
                output_dir / "model_00000.pt",
            )

        final_iteration = start_iteration + ARGS.iterations
        for iteration in range(start_iteration + 1, final_iteration + 1):
            iteration_started = time.perf_counter()
            fixed_observations = {
                key: value.clone() for key, value in observations.items()
            }
            action_before = learner.runner.decide(
                fixed_observations,
                update_critic_normalizer=False,
            ).actions
            observations = learner.collect_rollout(env, observations)
            rewards = learner.storage.team("rewards").clone()
            dones = learner.storage.team("dones").clone()
            timeouts = learner.storage.team("timeouts").clone()
            advantages = learner.storage.team("advantages").clone()
            rollout_diagnostics = learner.last_rollout_diagnostics
            update_metrics = learner.update()
            action_after = learner.runner.decide(
                fixed_observations,
                update_critic_normalizer=False,
            ).actions
            policy_drift = torch.abs(action_after - action_before)
            termination_counts = rollout_diagnostics["termination_counts"]
            metrics = {
                "iteration": iteration,
                "elapsed_seconds": time.perf_counter() - iteration_started,
                "reward_mean": rewards.mean().item(),
                "reward_min": rewards.min().item(),
                "reward_max": rewards.max().item(),
                "reset_count": int(dones.count_nonzero().item()),
                "timeout_count": int(timeouts.count_nonzero().item()),
                "completed_episodes": rollout_diagnostics["completed_episodes"],
                "success_count": rollout_diagnostics["success_count"],
                "robot_fall_count": termination_counts["clear_robot_fall"],
                "table_safety_count": termination_counts[
                    "table_physical_safety"
                ],
                "reference_horizon_count": termination_counts[
                    "reference_horizon"
                ],
                "success_rate": rollout_diagnostics["success_rate"],
                "advantage_mean": advantages.mean().item(),
                "advantage_std": advantages.std(unbiased=False).item(),
                "policy_drift_mean_abs": policy_drift.mean().item(),
                "policy_drift_max_abs": policy_drift.max().item(),
                "yaw_sample_count": rollout_diagnostics["yaw_sample_count"],
                "yaw_sample_mean_rad": rollout_diagnostics[
                    "yaw_sample_mean_rad"
                ],
                "yaw_sample_min_rad": rollout_diagnostics[
                    "yaw_sample_min_rad"
                ],
                "yaw_sample_max_rad": rollout_diagnostics[
                    "yaw_sample_max_rad"
                ],
                "terminal_yaw_sample_count": rollout_diagnostics[
                    "terminal_yaw_sample_count"
                ],
                "terminal_yaw_mean_rad": rollout_diagnostics[
                    "terminal_yaw_mean_rad"
                ],
                "terminal_yaw_min_rad": rollout_diagnostics[
                    "terminal_yaw_min_rad"
                ],
                "terminal_yaw_max_rad": rollout_diagnostics[
                    "terminal_yaw_max_rad"
                ],
                **update_metrics.__dict__,
            }
            scalar_values = [
                value
                for value in metrics.values()
                if isinstance(value, (int, float))
            ]
            if not all(torch.isfinite(torch.tensor(value)) for value in scalar_values):
                raise RuntimeError(
                    f"Non-finite Demo 4 training metric at iteration {iteration}"
                )
            _append_jsonl(metrics_path, metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)

            if (
                is_demo4_periodic_checkpoint(
                    iteration,
                    interval=ARGS.checkpoint_interval,
                )
                or iteration == final_iteration
            ):
                torch.save(
                    learner.training_state_dict(iteration=iteration),
                    output_dir / f"model_{iteration:05d}.pt",
                )

        status = {
            "passed": True,
            "output_dir": str(output_dir),
            "final_iteration": final_iteration,
            "final_checkpoint": str(output_dir / f"model_{final_iteration:05d}.pt"),
            "periodic_checkpoint_interval": ARGS.checkpoint_interval,
        }
        _write_json(output_dir / "status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
        status = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "periodic_checkpoint_interval": ARGS.checkpoint_interval,
        }
        if owns_output_dir:
            _write_json(output_dir / "status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(SIMULATION_APP)
        if failure is not None:
            print(f"Demo 4 training failed: {failure}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
