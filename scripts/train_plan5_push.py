#!/usr/bin/env python3
"""Train one bounded Plan 5 cooperative Push, Pull, or Kick MAPPO baseline."""

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

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--skill", choices=("push", "pull", "kick"), default="push")
PARSER.add_argument("--iterations", type=int, default=50)
PARSER.add_argument("--num-envs", type=int, default=8)
PARSER.add_argument("--steps-per-env", type=int, default=24)
PARSER.add_argument("--seed", type=int, default=721)
PARSER.add_argument("--actor-learning-rate", type=float, default=None)
PARSER.add_argument("--max-actor-learning-rate", type=float, default=None)
PARSER.add_argument("--critic-learning-rate", type=float, default=None)
UPDATE_MODE = PARSER.add_mutually_exclusive_group()
UPDATE_MODE.add_argument("--critic-only", action="store_true")
UPDATE_MODE.add_argument("--teammate-input-only", action="store_true")
PARSER.add_argument(
    "--output-dir",
    type=Path,
    default=None,
)
CHECKPOINT_MODE = PARSER.add_mutually_exclusive_group()
CHECKPOINT_MODE.add_argument("--resume", type=Path, default=None)
CHECKPOINT_MODE.add_argument("--fine-tune-from", type=Path, default=None)
PARSER.add_argument("--joint-acceleration-weight", type=float, default=None)
PARSER.add_argument("--joint-acceleration-ramp-iterations", type=int, default=500)
PARSER.add_argument("--save-interval", type=int, default=1000)
ARGS = PARSER.parse_args()
try:
    RUNTIME_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
except ValueError:
    PARSER.error("WORLD_SIZE must be an integer")
if RUNTIME_WORLD_SIZE != 1:
    PARSER.error(
        "This entry point runs one independent single-GPU process; WORLD_SIZE must be 1"
    )
if ARGS.output_dir is None:
    project_dir = {
        "push": "Plan5Push",
        "pull": "Plan5Pull",
        "kick": "Plan5Kick",
    }[ARGS.skill]
    default_run_name = {
        "push": "a1_mappo_formal50_seed721",
        "pull": "pull_mappo_formal50_seed721",
        "kick": "mirrored_kick_mappo_formal50_seed721",
    }[ARGS.skill]
    ARGS.output_dir = (
        REPO_ROOT
        / "logs"
        / project_dir
        / default_run_name
    )
for name in ("iterations", "num_envs", "steps_per_env", "save_interval"):
    if getattr(ARGS, name) < 1:
        PARSER.error(f"--{name.replace('_', '-')} must be at least 1")
for name in ("actor_learning_rate", "max_actor_learning_rate", "critic_learning_rate"):
    value = getattr(ARGS, name)
    if value is not None and value <= 0.0:
        PARSER.error(f"--{name.replace('_', '-')} must be positive")
if ARGS.joint_acceleration_ramp_iterations < 0:
    PARSER.error("--joint-acceleration-ramp-iterations must be non-negative")
if ARGS.joint_acceleration_weight is not None and ARGS.joint_acceleration_weight >= 0.0:
    PARSER.error("--joint-acceleration-weight must be negative")
if ARGS.fine_tune_from is not None and ARGS.joint_acceleration_weight is None:
    PARSER.error("--fine-tune-from requires --joint-acceleration-weight")
if ARGS.skill != "push" and (
    ARGS.joint_acceleration_weight is not None or ARGS.fine_tune_from is not None
):
    PARSER.error(
        f"{ARGS.skill.title()} currently supports baseline training only; "
        "smooth fine-tuning is Push-only"
    )
if (
    ARGS.actor_learning_rate is not None
    and ARGS.max_actor_learning_rate is not None
    and ARGS.max_actor_learning_rate < ARGS.actor_learning_rate
):
    PARSER.error("--max-actor-learning-rate must be at least --actor-learning-rate")
if ARGS.teammate_input_only and ARGS.actor_learning_rate is None:
    PARSER.error("--teammate-input-only requires an explicit --actor-learning-rate")

from holosoma.config_values.marl.g1.experiment import (
    g1_29dof_plan5_pull_baseline,
    g1_29dof_plan5_push_baseline,
    g1_29dof_plan5_push_smooth,
)
from holosoma.config_values.marl.g1.kick_experiment import (
    g1_29dof_plan5_kick_baseline,
)
from holosoma.utils.eval_utils import init_sim_imports


SMOOTH_FINETUNE = ARGS.joint_acceleration_weight is not None
if ARGS.skill == "kick":
    EXPERIMENT = g1_29dof_plan5_kick_baseline
    CONFIG_LABEL = "g1_29dof_plan5_kick_baseline"
    TRAINING_PROJECT = "Plan5Kick"
    TRAINING_NAME = "mirrored_kick_mappo_formal"
    SOURCE_CHECKPOINT = (
        REPO_ROOT / "logs/WholeBodyTracking/marl_compat_kick_v1/model_07999_actor158.pt"
    )
    EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
        "1555968f678c2b69fcd6f09c64d0a6252683eab902edd773a84acc205d0f5491"
    )
elif ARGS.skill == "pull":
    EXPERIMENT = g1_29dof_plan5_pull_baseline
    CONFIG_LABEL = "g1_29dof_plan5_pull_baseline"
    TRAINING_PROJECT = "Plan5Pull"
    TRAINING_NAME = "pull_mappo_formal"
    SOURCE_CHECKPOINT = (
        REPO_ROOT / "logs/WholeBodyTracking/marl_compat_pull_v1/model_07999_actor158.pt"
    )
    EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
        "f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364"
    )
else:
    EXPERIMENT = g1_29dof_plan5_push_smooth if SMOOTH_FINETUNE else g1_29dof_plan5_push_baseline
    CONFIG_LABEL = (
        "g1_29dof_plan5_push_smooth"
        if SMOOTH_FINETUNE
        else "g1_29dof_plan5_push_baseline"
    )
    TRAINING_PROJECT = "Plan5Push"
    TRAINING_NAME = "a1_mappo_formal"
    SOURCE_CHECKPOINT = (
        REPO_ROOT / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"
    )
    EXPECTED_SOURCE_CHECKPOINT_SHA256 = None
CONFIG = replace(
    EXPERIMENT,
    training=replace(
        EXPERIMENT.training,
        num_envs=ARGS.num_envs,
        headless=True,
        seed=ARGS.seed,
        project=TRAINING_PROJECT,
        name=TRAINING_NAME,
    ),
)
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle  # noqa: E402
from holosoma.agents.mappo.ppo import Plan5PPO  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


EXPECTED_OBJECT_MASS_KG = 20.0
EXPECTED_OBJECT_COM_M = (0.0, 0.015111745244133, 0.0)
EXPECTED_OBJECT_INERTIA_KG_M2 = {
    "push": (
        0.62774975216702,
        0.0,
        0.0,
        0.0,
        4.36041519206568,
        0.0,
        0.0,
        0.0,
        3.95048914424628,
    ),
    "pull": (
        3.95048914424628,
        0.0,
        0.0,
        0.0,
        4.36041519206568,
        0.0,
        0.0,
        0.0,
        0.62774975216702,
    ),
    "kick": (
        3.95048914424628,
        0.0,
        0.0,
        0.0,
        4.36041519206568,
        0.0,
        0.0,
        0.0,
        0.62774975216702,
    ),
}
EXPECTED_OBJECT_MATERIAL = (0.5, 0.5, 0.0)


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


def _reward_term_snapshot(env) -> dict[str, float]:
    result = {}
    for name, config in zip(env.reward_manager._term_names, env.reward_manager._term_cfgs):
        if name in env.reward_manager._term_instances:
            instance = env.reward_manager._term_instances[name]
            snapshot = getattr(instance, "snapshot", None)
            raw = snapshot() if callable(snapshot) else instance(env, **config.params)
        else:
            raw = env.reward_manager._term_funcs[name](env, **config.params)
        result[name] = raw.mean().item()
    return result


def _set_reward_weight(env, name: str, weight: float) -> None:
    config = env.reward_manager.get_term_cfg(name)
    env.reward_manager.set_term_cfg(name, replace(config, weight=weight))


def _ramped_weight(*, iteration: int, start_iteration: int, target: float, ramp: int) -> float:
    if ramp == 0:
        return target
    progress = min(max((iteration - start_iteration) / ramp, 0.0), 1.0)
    return target * progress


def _checkpoint_state(learner, *, iteration: int, smooth_schedule: dict | None) -> dict:
    state = learner.training_state_dict(iteration=iteration)
    if smooth_schedule is not None:
        state["joint_acceleration_schedule"] = smooth_schedule
    return state


def _runtime_object_physics(env, *, skill: str) -> dict[str, dict[str, object]]:
    """Read, validate, and serialize the formal table properties from PhysX."""
    physx_view = env.simulator._object.root_physx_view
    tensors = {
        "mass_kg": physx_view.get_masses(),
        "inertia_kg_m2_row_major": physx_view.get_inertias(),
        "com_pose_body_xyzw": physx_view.get_coms(),
        "material_static_dynamic_restitution": physx_view.get_material_properties(),
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("Non-finite runtime object physics properties")

    checks = {
        "mass": (
            tensors["mass_kg"].reshape(-1),
            torch.full_like(tensors["mass_kg"].reshape(-1), EXPECTED_OBJECT_MASS_KG),
        ),
        "center of mass": (
            tensors["com_pose_body_xyzw"][..., :3],
            torch.as_tensor(
                EXPECTED_OBJECT_COM_M,
                device=tensors["com_pose_body_xyzw"].device,
                dtype=tensors["com_pose_body_xyzw"].dtype,
            ).expand_as(tensors["com_pose_body_xyzw"][..., :3]),
        ),
        "inertia": (
            tensors["inertia_kg_m2_row_major"],
            torch.as_tensor(
                EXPECTED_OBJECT_INERTIA_KG_M2[skill],
                device=tensors["inertia_kg_m2_row_major"].device,
                dtype=tensors["inertia_kg_m2_row_major"].dtype,
            ).expand_as(tensors["inertia_kg_m2_row_major"]),
        ),
        "collision material": (
            tensors["material_static_dynamic_restitution"],
            torch.as_tensor(
                EXPECTED_OBJECT_MATERIAL,
                device=tensors["material_static_dynamic_restitution"].device,
                dtype=tensors["material_static_dynamic_restitution"].dtype,
            ).expand_as(tensors["material_static_dynamic_restitution"]),
        ),
    }
    for name, (actual, expected) in checks.items():
        if not torch.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6):
            raise RuntimeError(
                f"Formal Plan 5 object {name} does not match the frozen setup: "
                f"actual={actual.detach().cpu().tolist()}, "
                f"expected={expected.detach().cpu().tolist()}"
            )

    return {
        name: {
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
        for name, value in tensors.items()
    }


def main() -> None:
    env = None
    failure: BaseException | None = None
    output_dir = ARGS.output_dir.expanduser().resolve()
    owns_output_dir = False
    try:
        torch.manual_seed(ARGS.seed)
        if ARGS.resume is None and output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Choose another path or use --resume."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        owns_output_dir = True
        metrics_path = output_dir / "metrics.jsonl"

        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        observations = env.reset_all()
        runtime_object_physics = _runtime_object_physics(env, skill=ARGS.skill)
        if g1_29dof_wbt_w_object.algo.config.init_at_random_ep_len:
            env.episode_length_buf = torch.randint_like(
                env.episode_length_buf,
                high=int(env.max_episode_length),
            )

        ppo_overrides = {"num_steps_per_env": ARGS.steps_per_env}
        actor_learning_rate = ARGS.actor_learning_rate
        max_actor_learning_rate = ARGS.max_actor_learning_rate
        if SMOOTH_FINETUNE:
            actor_learning_rate = actor_learning_rate or 1.0e-5
            max_actor_learning_rate = max_actor_learning_rate or 5.0e-5
        if actor_learning_rate is not None:
            ppo_overrides["actor_learning_rate"] = actor_learning_rate
        if max_actor_learning_rate is not None:
            ppo_overrides["max_actor_learning_rate"] = max_actor_learning_rate
        if ARGS.critic_learning_rate is not None:
            ppo_overrides["critic_learning_rate"] = ARGS.critic_learning_rate
        if ARGS.teammate_input_only:
            ppo_overrides["schedule"] = "fixed"
        ppo_config = replace(g1_29dof_wbt_w_object.algo.config, **ppo_overrides)
        initialization_kwargs = {}
        if EXPECTED_SOURCE_CHECKPOINT_SHA256 is not None:
            initialization_kwargs["expected_sha256"] = EXPECTED_SOURCE_CHECKPOINT_SHA256
        models = initialize_plan5_model_bundle(
            SOURCE_CHECKPOINT,
            ppo_config,
            device=env.device,
            **initialization_kwargs,
        )
        learner = Plan5PPO(
            models,
            ppo_config,
            num_envs=env.num_envs,
            num_steps_per_env=ARGS.steps_per_env,
            device=env.device,
        )
        start_iteration = 0
        smooth_ramp_start_iteration = 0 if SMOOTH_FINETUNE else None
        if ARGS.resume is not None:
            resume_state = torch.load(ARGS.resume.expanduser().resolve(), map_location=env.device)
            start_iteration = learner.load_training_state_dict(resume_state)
            if SMOOTH_FINETUNE:
                saved_schedule = resume_state.get("joint_acceleration_schedule")
                if not isinstance(saved_schedule, dict):
                    raise ValueError(
                        "Smooth resume checkpoint has no joint-acceleration schedule; "
                        "use --fine-tune-from for an existing non-smooth checkpoint"
                    )
                expected_schedule = {
                    "target_weight": ARGS.joint_acceleration_weight,
                    "ramp_iterations": ARGS.joint_acceleration_ramp_iterations,
                }
                for key, expected in expected_schedule.items():
                    if saved_schedule.get(key) != expected:
                        raise ValueError(
                            f"Smooth resume schedule mismatch for {key}: "
                            f"checkpoint={saved_schedule.get(key)!r}, requested={expected!r}"
                        )
                smooth_ramp_start_iteration = int(saved_schedule["start_iteration"])
        elif ARGS.fine_tune_from is not None:
            fine_tune_state = torch.load(
                ARGS.fine_tune_from.expanduser().resolve(),
                map_location=env.device,
            )
            start_iteration = learner.load_fine_tune_state_dict(fine_tune_state)
            smooth_ramp_start_iteration = start_iteration

        smooth_schedule = None
        if SMOOTH_FINETUNE:
            assert smooth_ramp_start_iteration is not None
            smooth_schedule = {
                "target_weight": ARGS.joint_acceleration_weight,
                "ramp_iterations": ARGS.joint_acceleration_ramp_iterations,
                "start_iteration": smooth_ramp_start_iteration,
            }
            initial_weight = _ramped_weight(
                iteration=start_iteration,
                start_iteration=smooth_ramp_start_iteration,
                target=ARGS.joint_acceleration_weight,
                ramp=ARGS.joint_acceleration_ramp_iterations,
            )
            _set_reward_weight(env, "joint_acceleration_l2", initial_weight)

        command_term = CONFIG.command.setup_terms["paired_motion_command"]
        command_params = command_term.params
        selected_motion_config = command_params["motion_config"]
        run_config = {
            "git_commit": _git_commit(),
            "started_unix_time": time.time(),
            "skill": ARGS.skill,
            "config": CONFIG_LABEL,
            "training_project": TRAINING_PROJECT,
            "training_name": TRAINING_NAME,
            "output_dir": str(output_dir),
            "seed": ARGS.seed,
            "iterations_requested": ARGS.iterations,
            "start_iteration": start_iteration,
            "num_envs": env.num_envs,
            "num_agents": 2,
            "steps_per_env": ARGS.steps_per_env,
            "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(REPO_ROOT)),
            "source_checkpoint_sha256": models.source_sha256,
            "expected_source_checkpoint_sha256": (
                EXPECTED_SOURCE_CHECKPOINT_SHA256 or models.source_sha256
            ),
            "fine_tune_checkpoint": (
                str(ARGS.fine_tune_from.expanduser().resolve())
                if ARGS.fine_tune_from is not None
                else None
            ),
            "optimizer_state_restored": ARGS.resume is not None,
            "motion_file": selected_motion_config.motion_file,
            "paired_reference_file": command_params.get("paired_reference_file"),
            "lateral_spacing_m": command_params.get("lateral_spacing_m"),
            "command_func": command_term.func,
            "robot_urdf": CONFIG.robot.asset.urdf_file,
            "object_urdf": CONFIG.robot.object.object_urdf_path,
            "runtime_object_physics": runtime_object_physics,
            "reward_terms": list(env.reward_manager.active_terms),
            "termination_terms": list(env.termination_manager.active_terms),
            "actor_obs_dim": 158,
            "critic_obs_dim": 527,
            "action_dim_per_agent": 29,
            "actor_learning_rate": ppo_config.actor_learning_rate,
            "max_actor_learning_rate": ppo_config.max_actor_learning_rate,
            "critic_learning_rate": ppo_config.critic_learning_rate,
            "learning_rate_schedule": ppo_config.schedule,
            "critic_only": ARGS.critic_only,
            "teammate_input_only": ARGS.teammate_input_only,
            "actor_update_mode": (
                "critic_only"
                if ARGS.critic_only
                else "teammate_input_only"
                if ARGS.teammate_input_only
                else "full"
            ),
            "gamma": ppo_config.gamma,
            "lambda": ppo_config.lam,
            "clip_param": ppo_config.clip_param,
            "num_learning_epochs": ppo_config.num_learning_epochs,
            "num_mini_batches": ppo_config.num_mini_batches,
            "joint_acceleration_schedule": smooth_schedule,
            "runtime_process": {
                "distributed_data_parallel": False,
                "device_within_visible_set": "cuda:0",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "world_size": RUNTIME_WORLD_SIZE,
            },
        }
        _write_json(output_dir / "run_config.json", run_config)
        if ARGS.resume is None:
            torch.save(
                _checkpoint_state(
                    learner,
                    iteration=start_iteration,
                    smooth_schedule=smooth_schedule,
                ),
                output_dir / f"model_{start_iteration:05d}.pt",
            )

        final_iteration = start_iteration + ARGS.iterations
        for iteration in range(start_iteration + 1, final_iteration + 1):
            iteration_started = time.perf_counter()
            current_joint_acceleration_weight = None
            if smooth_schedule is not None:
                current_joint_acceleration_weight = _ramped_weight(
                    iteration=iteration,
                    start_iteration=smooth_schedule["start_iteration"],
                    target=smooth_schedule["target_weight"],
                    ramp=smooth_schedule["ramp_iterations"],
                )
                _set_reward_weight(
                    env,
                    "joint_acceleration_l2",
                    current_joint_acceleration_weight,
                )
            fixed_observations = {
                key: value.clone() for key, value in observations.items()
            }
            action_mean_before = learner.runner.decide(
                fixed_observations,
                update_critic_normalizer=False,
            ).actions
            observations = learner.collect_rollout(env, observations)
            rewards = learner.storage.team("rewards").clone()
            dones = learner.storage.team("dones").clone()
            timeouts = learner.storage.team("timeouts").clone()
            advantages = learner.storage.team("advantages").clone()
            reward_terms = _reward_term_snapshot(env)
            update_metrics = learner.update(
                update_actor=not ARGS.critic_only,
                teammate_input_only=ARGS.teammate_input_only,
            )
            action_mean_after = learner.runner.decide(
                fixed_observations,
                update_critic_normalizer=False,
            ).actions
            policy_drift = torch.abs(action_mean_after - action_mean_before)
            elapsed = time.perf_counter() - iteration_started
            metrics = {
                "iteration": iteration,
                "elapsed_seconds": elapsed,
                "reward_mean": rewards.mean().item(),
                "reward_min": rewards.min().item(),
                "reward_max": rewards.max().item(),
                "reset_count": int(dones.count_nonzero().item()),
                "timeout_count": int(timeouts.count_nonzero().item()),
                "tracking_failure_count": int((dones & ~timeouts).count_nonzero().item()),
                "advantage_mean": advantages.mean().item(),
                "advantage_std": advantages.std().item(),
                "policy_drift_mean_abs": policy_drift.mean().item(),
                "policy_drift_max_abs": policy_drift.max().item(),
                "action_noise_std_mean": models.actor.std.mean().item(),
                "actor_learning_rate": learner.actor_learning_rate,
                "critic_learning_rate": learner.critic_learning_rate,
                "joint_acceleration_weight": current_joint_acceleration_weight,
                "reward_terms_raw_mean": reward_terms,
                **update_metrics.__dict__,
            }
            if not all(
                torch.isfinite(torch.tensor(value))
                for key, value in metrics.items()
                if key
                not in {
                    "iteration",
                    "reset_count",
                    "timeout_count",
                    "tracking_failure_count",
                    "joint_acceleration_weight",
                    "reward_terms_raw_mean",
                }
            ):
                raise RuntimeError(f"Non-finite training metric at iteration {iteration}: {metrics}")
            if not all(torch.isfinite(torch.tensor(value)) for value in reward_terms.values()):
                raise RuntimeError(f"Non-finite reward term at iteration {iteration}: {reward_terms}")
            _append_jsonl(metrics_path, metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)

            fine_tune_iteration = iteration - start_iteration
            if fine_tune_iteration % ARGS.save_interval == 0 or iteration == final_iteration:
                torch.save(
                    _checkpoint_state(
                        learner,
                        iteration=iteration,
                        smooth_schedule=smooth_schedule,
                    ),
                    output_dir / f"model_{iteration:05d}.pt",
                )

        final_report = {
            "passed": True,
            "skill": ARGS.skill,
            "config": CONFIG_LABEL,
            "output_dir": str(output_dir),
            "final_iteration": final_iteration,
            "final_checkpoint": str(output_dir / f"model_{final_iteration:05d}.pt"),
        }
        _write_json(output_dir / "status.json", final_report)
        print(json.dumps(final_report, indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
        failure_report = {
            "passed": False,
            "skill": ARGS.skill,
            "config": CONFIG_LABEL,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if owns_output_dir:
            _write_json(output_dir / "status.json", failure_report)
        print(json.dumps(failure_report, indent=2, sort_keys=True), flush=True)
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(SIMULATION_APP)
    if failure is not None:
        raise RuntimeError(f"Plan 5 {ARGS.skill.title()} training failed") from failure


if __name__ == "__main__":
    main()
