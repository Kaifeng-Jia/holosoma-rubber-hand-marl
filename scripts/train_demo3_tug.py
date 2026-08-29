#!/usr/bin/env python3
"""Train the isolated competitive Demo 3 square-table tug policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
PARSER.add_argument("--critic-only-iterations", type=int, default=50)
PARSER.add_argument("--full-actor-iterations", type=int, default=8000)
PARSER.add_argument("--num-envs", type=int, default=2048)
PARSER.add_argument("--steps-per-env", type=int, default=24)
PARSER.add_argument("--seed", type=int, default=721)
PARSER.add_argument("--checkpoint-interval", type=int, default=1000)
PARSER.add_argument("--output-dir", type=Path, default=None)
PARSER.add_argument("--resume", type=Path, default=None)
PARSER.add_argument("--actor-learning-rate", type=float, default=None)
PARSER.add_argument("--critic-learning-rate", type=float, default=None)
ARGS = PARSER.parse_args()
if ARGS.critic_only_iterations < 0:
    PARSER.error("--critic-only-iterations must be non-negative")
for name in (
    "full_actor_iterations",
    "num_envs",
    "steps_per_env",
    "checkpoint_interval",
):
    if getattr(ARGS, name) < 1:
        PARSER.error(f"--{name.replace('_', '-')} must be positive")
for name in ("actor_learning_rate", "critic_learning_rate"):
    value = getattr(ARGS, name)
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        PARSER.error(f"--{name.replace('_', '-')} must be finite and positive")
try:
    RUNTIME_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
except ValueError:
    PARSER.error("WORLD_SIZE must be an integer")
if RUNTIME_WORLD_SIZE != 1:
    PARSER.error(
        "Demo 3 uses one independent single-GPU process; WORLD_SIZE must be 1"
    )
if ARGS.output_dir is None:
    ARGS.output_dir = (
        REPO_ROOT
        / "logs/Demo3Tug"
        / f"square_table_diagonal_tug_seed{ARGS.seed}_env{ARGS.num_envs}"
    )

from holosoma.agents.mappo.demo3_checkpoint import (  # noqa: E402
    DEMO3_OBJECT_COM_M,
    DEMO3_OBJECT_INERTIA_KG_M2,
    DEMO3_OBJECT_MASS_KG,
    DEMO3_OBJECT_MATERIAL,
    DEMO3_ROBOT_URDF_SHA256,
    DEMO3_RUNTIME_REFERENCE_SHA256,
    DEMO3_SQUARE_TABLE_URDF_SHA256,
    DEMO3_WARM_START_SHA256,
    demo3_ppo_contract,
    demo3_training_contract,
    is_demo3_periodic_checkpoint,
    validate_demo3_asset,
)
from holosoma.config_values.marl.g1.demo3_command import (  # noqa: E402
    DEMO3_TUG_RUNTIME_REFERENCE_FILE,
)
from holosoma.config_values.marl.g1.demo3_experiment import (  # noqa: E402
    g1_29dof_demo3_tug_baseline,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402


SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "logs/Demo3Tug/checkpoints/model_07999_actor164_table_neutral.pt"
)
RUNTIME_REFERENCE = (
    REPO_ROOT / "src/holosoma" / DEMO3_TUG_RUNTIME_REFERENCE_FILE
).resolve()
ROBOT_URDF = (
    REPO_ROOT
    / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
)
TABLE_URDF = (
    REPO_ROOT
    / (
        "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/"
        "objects_squaretable_demo3_training.urdf"
    )
)
ASSET_SHA256 = {
    "warm_start": validate_demo3_asset(
        SOURCE_CHECKPOINT,
        expected_sha256=DEMO3_WARM_START_SHA256,
        label="warm-start checkpoint",
    ),
    "runtime_reference": validate_demo3_asset(
        RUNTIME_REFERENCE,
        expected_sha256=DEMO3_RUNTIME_REFERENCE_SHA256,
        label="runtime reference",
    ),
    "robot_urdf": validate_demo3_asset(
        ROBOT_URDF,
        expected_sha256=DEMO3_ROBOT_URDF_SHA256,
        label="rubber-hand robot URDF",
    ),
    "table_urdf": validate_demo3_asset(
        TABLE_URDF,
        expected_sha256=DEMO3_SQUARE_TABLE_URDF_SHA256,
        label="square-table URDF",
    ),
}
TRAINING_CONTRACT = demo3_training_contract(
    critic_only_iterations=ARGS.critic_only_iterations,
    full_actor_iterations=ARGS.full_actor_iterations,
    checkpoint_interval=ARGS.checkpoint_interval,
)
CONFIG = replace(
    g1_29dof_demo3_tug_baseline,
    training=replace(
        g1_29dof_demo3_tug_baseline.training,
        num_envs=ARGS.num_envs,
        headless=True,
        seed=ARGS.seed,
        project="Demo3Tug",
        name="square_table_diagonal_tug_mappo",
    ),
)


def _validate_config_contract() -> None:
    actual_reward_weights = {
        name: cfg.weight for name, cfg in CONFIG.reward.terms.items()
    }
    if actual_reward_weights != TRAINING_CONTRACT["reward_weights"]:
        raise ValueError(
            "Demo 3 reward weights no longer match the frozen training contract"
        )
    horizon = CONFIG.termination.terms["reference_horizon"]
    fall = CONFIG.termination.terms["clear_robot_fall"]
    expected_termination = TRAINING_CONTRACT["termination"]
    if horizon.is_timeout is not True or fall.params != {
        "minimum_ref_body_height": expected_termination[
            "minimum_ref_body_height_m"
        ],
        "maximum_gravity_z": expected_termination["maximum_gravity_z"],
    }:
        raise ValueError(
            "Demo 3 termination settings no longer match the frozen contract"
        )
    command_file = CONFIG.command.setup_terms["paired_motion_command"].params.get(
        "paired_reference_file"
    )
    if command_file != DEMO3_TUG_RUNTIME_REFERENCE_FILE:
        raise ValueError("Demo 3 command no longer uses the frozen runtime reference")
    if not CONFIG.robot.asset.urdf_file.endswith(TRAINING_CONTRACT["robot_asset"]):
        raise ValueError("Demo 3 no longer uses the frozen rubber-hand robot asset")
    if not CONFIG.robot.object.object_urdf_path.endswith(
        TRAINING_CONTRACT["table_asset"]
    ):
        raise ValueError("Demo 3 no longer uses the frozen square-table asset")
    actor_inputs = CONFIG.algo.config.module_dict.actor.input_dim
    if actor_inputs != ["actor_obs", "teammate_obs", "table_obs"]:
        raise ValueError("Demo 3 Actor observation group order changed")


_validate_config_contract()
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.agents.mappo.demo3_initialization import (  # noqa: E402
    initialize_demo3_model_bundle,
)
from holosoma.agents.mappo.demo3_ppo import Demo3PPO  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


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


_RESUME_INVARIANT_FIELDS = (
    "scenario",
    "seed",
    "critic_only_iterations",
    "full_actor_iterations",
    "num_envs",
    "num_agents",
    "steps_per_env",
    "checkpoint_interval",
    "source_checkpoint_sha256",
    "source_iteration",
    "actor_forward_contract",
    "actor_observation_groups",
    "critic_observation_dim",
    "critic_forward_contract",
    "return_model",
    "done_model",
    "reference_file",
    "asset_sha256",
    "training_contract",
    "ppo_contract",
)
_MODEL_NAME = re.compile(r"model_(\d+)\.pt")


def _prepare_resume_ledger(
    output_dir: Path,
    *,
    start_iteration: int,
    current_run_config: dict[str, object],
) -> tuple[Path, Path, Path, str | None]:
    """Preserve the original ledger and reject ambiguous checkpoint rewinds."""

    base_config_path = output_dir / "run_config.json"
    if not base_config_path.is_file():
        raise FileNotFoundError(
            f"Demo 3 resume requires the original run_config.json: {base_config_path}"
        )
    try:
        base_config = json.loads(base_config_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            f"Demo 3 original run_config.json is unreadable: {base_config_path}"
        ) from error
    if not isinstance(base_config, dict):
        raise ValueError("Demo 3 original run_config.json must contain an object")
    mismatches = {
        name: {
            "original": base_config.get(name),
            "requested": current_run_config.get(name),
        }
        for name in _RESUME_INVARIANT_FIELDS
        if base_config.get(name) != current_run_config.get(name)
    }
    if mismatches:
        raise ValueError(
            "Demo 3 resume configuration differs from the original run: "
            f"{mismatches}"
        )

    later_checkpoints: list[Path] = []
    for path in output_dir.glob("model_*.pt"):
        match = _MODEL_NAME.fullmatch(path.name)
        if match is not None and int(match.group(1)) > start_iteration:
            later_checkpoints.append(path)
    if later_checkpoints:
        raise FileExistsError(
            "Demo 3 refuses to rewind over newer checkpoints: "
            f"{[str(path) for path in sorted(later_checkpoints)]}"
        )

    suffix = f"resume_from_{start_iteration:05d}"
    run_config_path = output_dir / f"run_config_{suffix}.json"
    metrics_path = output_dir / f"metrics_{suffix}.jsonl"
    status_path = output_dir / f"status_resume_from_model_{start_iteration:05d}.json"
    occupied = [
        path
        for path in (run_config_path, metrics_path, status_path)
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "Demo 3 resume ledger segment already exists; inspect it before retrying: "
            f"{[str(path) for path in occupied]}"
        )
    original_output_dir = base_config.get("output_dir")
    if original_output_dir is not None and not isinstance(original_output_dir, str):
        raise ValueError("Demo 3 original output_dir must be a string or null")
    return run_config_path, metrics_path, status_path, original_output_dir


def _runtime_object_physics(env) -> dict[str, object]:
    view = env.simulator._object.root_physx_view
    mass = view.get_masses().reshape(-1)
    inertia = view.get_inertias()
    com = view.get_coms()
    material = view.get_material_properties()
    checks = {
        "mass": (
            mass,
            torch.full_like(mass, DEMO3_OBJECT_MASS_KG),
        ),
        "center of mass": (
            com[..., :3],
            torch.as_tensor(
                DEMO3_OBJECT_COM_M,
                device=com.device,
                dtype=com.dtype,
            ).expand_as(com[..., :3]),
        ),
        "inertia": (
            inertia,
            torch.as_tensor(
                DEMO3_OBJECT_INERTIA_KG_M2,
                device=inertia.device,
                dtype=inertia.dtype,
            ).expand_as(inertia),
        ),
        "material": (
            material,
            torch.as_tensor(
                DEMO3_OBJECT_MATERIAL,
                device=material.device,
                dtype=material.dtype,
            ).expand_as(material),
        ),
    }
    for name, (actual, expected) in checks.items():
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"Demo 3 object {name} contains non-finite values")
        if not torch.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6):
            raise RuntimeError(
                f"Demo 3 object {name} mismatch: "
                f"actual={actual.detach().cpu().tolist()}"
            )
    return {
        "mass_kg": mass.detach().cpu().tolist(),
        "com_pose_body_xyzw": com.detach().cpu().tolist(),
        "inertia_kg_m2_row_major": inertia.detach().cpu().tolist(),
        "material_static_dynamic_restitution": material.detach().cpu().tolist(),
    }


def _learning_rate_source(*, override: float | None, resumed: bool) -> str:
    if override is not None:
        return "command_line_override"
    return "restored_checkpoint" if resumed else "configured_default"


def _is_finite_metrics(metrics: dict[str, object]) -> bool:
    values = [
        value
        for value in metrics.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return all(torch.isfinite(torch.tensor(value)) for value in values)


def main() -> int:
    env = None
    failure: BaseException | None = None
    output_dir = ARGS.output_dir.expanduser().resolve()
    status_path = output_dir / "status.json"
    status_path_is_reserved = False
    try:
        torch.manual_seed(ARGS.seed)
        resume_path = (
            None if ARGS.resume is None else ARGS.resume.expanduser().resolve()
        )
        if resume_path is None:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}. "
                    "Choose another path or use --resume."
                )
        else:
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
            if resume_path.parent != output_dir:
                raise ValueError(
                    "Demo 3 resume checkpoint must be inside --output-dir so the "
                    "run ledger remains contiguous"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        if resume_path is None:
            status_path_is_reserved = True

        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        observations = env.reset_all()
        physics = _runtime_object_physics(env)

        ppo_overrides: dict[str, object] = {
            "num_steps_per_env": ARGS.steps_per_env,
        }
        if ARGS.actor_learning_rate is not None:
            ppo_overrides["actor_learning_rate"] = ARGS.actor_learning_rate
        if ARGS.critic_learning_rate is not None:
            ppo_overrides["critic_learning_rate"] = ARGS.critic_learning_rate
        ppo_config = replace(CONFIG.algo.config, **ppo_overrides)
        ppo_contract = demo3_ppo_contract(
            ppo_config,
            num_steps_per_env=ARGS.steps_per_env,
        )
        models = initialize_demo3_model_bundle(
            SOURCE_CHECKPOINT,
            ppo_config,
            expected_sha256=DEMO3_WARM_START_SHA256,
            device=env.device,
        )
        learner = Demo3PPO(
            models,
            ppo_config,
            num_envs=env.num_envs,
            num_steps_per_env=ARGS.steps_per_env,
            device=env.device,
            training_contract=TRAINING_CONTRACT,
        )

        start_iteration = 0
        if resume_path is not None:
            state = torch.load(
                resume_path,
                map_location=env.device,
                weights_only=False,
            )
            start_iteration = learner.load_training_state_dict(
                state,
                actor_learning_rate=ARGS.actor_learning_rate,
                critic_learning_rate=ARGS.critic_learning_rate,
            )

        final_iteration = (
            ARGS.critic_only_iterations + ARGS.full_actor_iterations
        )
        if start_iteration < 0 or start_iteration >= final_iteration:
            raise ValueError(
                "Resume iteration must be within the requested unfinished schedule: "
                f"start={start_iteration}, final={final_iteration}"
            )

        command_term = CONFIG.command.setup_terms["paired_motion_command"]
        run_config = {
            "git_commit": _git_commit(),
            "started_unix_time": time.time(),
            "scenario": "competitive_square_table_diagonal_tug",
            "config": "g1_29dof_demo3_tug_baseline",
            "output_dir": str(output_dir),
            "seed": ARGS.seed,
            "start_iteration": start_iteration,
            "final_iteration": final_iteration,
            "critic_only_iterations": ARGS.critic_only_iterations,
            "full_actor_iterations": ARGS.full_actor_iterations,
            "num_envs": env.num_envs,
            "num_agents": 2,
            "steps_per_env": ARGS.steps_per_env,
            "checkpoint_interval": ARGS.checkpoint_interval,
            "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(REPO_ROOT)),
            "source_checkpoint_sha256": models.source_sha256,
            "source_iteration": models.source_iteration,
            "source_state_loaded": (
                "full_demo3_resume_after_warm_start_architecture"
                if resume_path is not None
                else "actor_and_actor_normalizer_only"
            ),
            "critic_initialization": (
                "restored_from_demo3_resume" if resume_path is not None else "fresh"
            ),
            "optimizer_initialization": (
                "restored_from_demo3_resume" if resume_path is not None else "fresh"
            ),
            "actor_parameters_shared": True,
            "opponent_update": "synchronous_current_shared_actor",
            "actor_forward_contract": "[E,2,164] -> [E*2,164] -> [E,2,29]",
            "actor_observation_groups": {
                "actor_obs": 154,
                "teammate_obs": 4,
                "table_obs": 6,
            },
            "table_observation_contains_yaw_rate": False,
            "critic_observation_dim": 527,
            "critic_forward_contract": "[E,2,527] -> [E*2,527] -> [E,2,1]",
            "return_model": "per_agent_reward_value_return_advantage",
            "done_model": "shared_physical_episode",
            "reference_file": command_term.params.get("paired_reference_file"),
            "asset_sha256": ASSET_SHA256,
            "reference_table_policy": "reset_only_no_trajectory_tracking",
            "robot_urdf": CONFIG.robot.asset.urdf_file,
            "object_urdf": CONFIG.robot.object.object_urdf_path,
            "object_physics": physics,
            "reward_terms": list(env.reward_manager.active_terms),
            "reward_weights": {
                name: cfg.weight for name, cfg in CONFIG.reward.terms.items()
            },
            "termination_terms": list(env.termination_manager.active_terms),
            "training_contract": TRAINING_CONTRACT,
            "ppo_contract": ppo_contract,
            "gamma": ppo_config.gamma,
            "lambda": ppo_config.lam,
            "clip_param": ppo_config.clip_param,
            "entropy_coef": ppo_config.entropy_coef,
            "num_learning_epochs": ppo_config.num_learning_epochs,
            "num_mini_batches": ppo_config.num_mini_batches,
            "learning_rates": {
                "actor": {
                    "requested_override": ARGS.actor_learning_rate,
                    "actual": learner.actor_learning_rate,
                    "source": _learning_rate_source(
                        override=ARGS.actor_learning_rate,
                        resumed=resume_path is not None,
                    ),
                },
                "critic": {
                    "requested_override": ARGS.critic_learning_rate,
                    "actual": learner.critic_learning_rate,
                    "source": _learning_rate_source(
                        override=ARGS.critic_learning_rate,
                        resumed=resume_path is not None,
                    ),
                },
            },
            "runtime_process": {
                "distributed_data_parallel": False,
                "device_within_visible_set": "cuda:0",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "world_size": RUNTIME_WORLD_SIZE,
            },
        }
        if resume_path is None:
            run_config_path = output_dir / "run_config.json"
            metrics_path = output_dir / "metrics.jsonl"
        else:
            (
                run_config_path,
                metrics_path,
                status_path,
                original_output_dir,
            ) = _prepare_resume_ledger(
                output_dir,
                start_iteration=start_iteration,
                current_run_config=run_config,
            )
            status_path_is_reserved = True
            run_config["resume_checkpoint"] = str(resume_path)
            run_config["original_run_config"] = str(output_dir / "run_config.json")
            run_config["ledger_origin_output_dir"] = original_output_dir
            run_config["ledger_relocated"] = (
                original_output_dir is not None
                and Path(original_output_dir).expanduser().resolve() != output_dir
            )
        run_config["run_config_path"] = str(run_config_path)
        run_config["metrics_path"] = str(metrics_path)
        _write_json(run_config_path, run_config)
        if resume_path is None:
            torch.save(
                learner.training_state_dict(iteration=0),
                output_dir / "model_00000.pt",
            )

        for iteration in range(start_iteration + 1, final_iteration + 1):
            iteration_started = time.perf_counter()
            update_actor = iteration > ARGS.critic_only_iterations
            full_actor_iteration = max(
                0,
                iteration - ARGS.critic_only_iterations,
            )
            fixed_observations = {
                key: value.clone() for key, value in observations.items()
            }
            action_before = learner.runner.decide(
                fixed_observations,
                update_critic_normalizer=False,
            ).actions
            observations = learner.collect_rollout(env, observations)
            rewards = learner.storage.agent("rewards").clone()
            dones = learner.storage.shared("dones").clone()
            timeouts = learner.storage.shared("timeouts").clone()
            advantages = learner.storage.agent("advantages").clone()
            diagnostics = learner.last_rollout_diagnostics
            update_metrics = learner.update(update_actor=update_actor)
            action_after = learner.runner.decide(
                fixed_observations,
                update_critic_normalizer=False,
            ).actions
            policy_drift = torch.abs(action_after - action_before)
            termination_counts = diagnostics["termination_counts"]
            metrics = {
                "iteration": iteration,
                "phase": "full_actor" if update_actor else "critic_only",
                "full_actor_iteration": full_actor_iteration,
                "elapsed_seconds": time.perf_counter() - iteration_started,
                "reward_agent_a_mean": rewards[..., 0, :].mean().item(),
                "reward_agent_b_mean": rewards[..., 1, :].mean().item(),
                "reward_agent_a_min": rewards[..., 0, :].min().item(),
                "reward_agent_b_min": rewards[..., 1, :].min().item(),
                "reward_agent_a_max": rewards[..., 0, :].max().item(),
                "reward_agent_b_max": rewards[..., 1, :].max().item(),
                "reset_count": int(dones.count_nonzero().item()),
                "timeout_count": int(timeouts.count_nonzero().item()),
                "completed_episodes": diagnostics["completed_episodes"],
                "clear_robot_fall_count": termination_counts["clear_robot_fall"],
                "reference_horizon_count": termination_counts["reference_horizon"],
                "advantage_agent_a_mean": advantages[..., 0, :].mean().item(),
                "advantage_agent_b_mean": advantages[..., 1, :].mean().item(),
                "advantage_agent_a_std": advantages[..., 0, :].std(
                    unbiased=False
                ).item(),
                "advantage_agent_b_std": advantages[..., 1, :].std(
                    unbiased=False
                ).item(),
                "policy_drift_mean_abs": policy_drift.mean().item(),
                "policy_drift_max_abs": policy_drift.max().item(),
                "actor_learning_rate": learner.actor_learning_rate,
                "critic_learning_rate": learner.critic_learning_rate,
                **{
                    key: value
                    for key, value in diagnostics.items()
                    if key not in {"termination_counts", "completed_episodes"}
                },
                **update_metrics.__dict__,
            }
            if not _is_finite_metrics(metrics):
                raise RuntimeError(
                    f"Non-finite Demo 3 training metric at iteration {iteration}"
                )
            _append_jsonl(metrics_path, metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)

            if (
                is_demo3_periodic_checkpoint(
                    iteration,
                    critic_only_iterations=ARGS.critic_only_iterations,
                    checkpoint_interval=ARGS.checkpoint_interval,
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
            "critic_only_boundary_checkpoint": str(
                output_dir / f"model_{ARGS.critic_only_iterations:05d}.pt"
            ),
            "final_checkpoint": str(output_dir / f"model_{final_iteration:05d}.pt"),
            "checkpoint_interval_full_actor_iterations": ARGS.checkpoint_interval,
            "run_config": str(run_config_path),
            "metrics": str(metrics_path),
        }
        _write_json(status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
        status = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if status_path_is_reserved:
            _write_json(status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(SIMULATION_APP)
        if failure is not None:
            print(f"Demo 3 training failed: {failure}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
