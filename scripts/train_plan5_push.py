#!/usr/bin/env python3
"""Train the bounded Plan 5 cooperative Push A1 MAPPO baseline."""

from __future__ import annotations

import argparse
import json
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
PARSER.add_argument("--iterations", type=int, default=50)
PARSER.add_argument("--num-envs", type=int, default=8)
PARSER.add_argument("--steps-per-env", type=int, default=24)
PARSER.add_argument("--seed", type=int, default=721)
PARSER.add_argument("--actor-learning-rate", type=float, default=None)
PARSER.add_argument("--critic-learning-rate", type=float, default=None)
PARSER.add_argument("--critic-only", action="store_true")
PARSER.add_argument(
    "--output-dir",
    type=Path,
    default=REPO_ROOT / "logs" / "Plan5Push" / "a1_mappo_smoke50_seed721",
)
PARSER.add_argument("--resume", type=Path, default=None)
PARSER.add_argument("--save-interval", type=int, default=50)
ARGS = PARSER.parse_args()
for name in ("iterations", "num_envs", "steps_per_env", "save_interval"):
    if getattr(ARGS, name) < 1:
        PARSER.error(f"--{name.replace('_', '-')} must be at least 1")
for name in ("actor_learning_rate", "critic_learning_rate"):
    value = getattr(ARGS, name)
    if value is not None and value <= 0.0:
        PARSER.error(f"--{name.replace('_', '-')} must be positive")

from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_push_baseline
from holosoma.utils.eval_utils import init_sim_imports


CONFIG = replace(
    g1_29dof_plan5_push_baseline,
    training=replace(
        g1_29dof_plan5_push_baseline.training,
        num_envs=ARGS.num_envs,
        headless=True,
        seed=ARGS.seed,
        project="Plan5Push",
        name="a1_mappo_smoke",
    ),
)
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle  # noqa: E402
from holosoma.agents.mappo.ppo import Plan5PPO  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.config_values.marl.g1.command import motion_config  # noqa: E402
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object  # noqa: E402
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


def _reward_term_snapshot(env) -> dict[str, float]:
    result = {}
    for name, config in zip(env.reward_manager._term_names, env.reward_manager._term_cfgs):
        if name in env.reward_manager._term_instances:
            raw = env.reward_manager._term_instances[name](env, **config.params)
        else:
            raw = env.reward_manager._term_funcs[name](env, **config.params)
        result[name] = raw.mean().item()
    return result


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
        if g1_29dof_wbt_w_object.algo.config.init_at_random_ep_len:
            env.episode_length_buf = torch.randint_like(
                env.episode_length_buf,
                high=int(env.max_episode_length),
            )

        ppo_overrides = {"num_steps_per_env": ARGS.steps_per_env}
        if ARGS.actor_learning_rate is not None:
            ppo_overrides["actor_learning_rate"] = ARGS.actor_learning_rate
        if ARGS.critic_learning_rate is not None:
            ppo_overrides["critic_learning_rate"] = ARGS.critic_learning_rate
        ppo_config = replace(g1_29dof_wbt_w_object.algo.config, **ppo_overrides)
        source_checkpoint = (
            REPO_ROOT / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"
        )
        models = initialize_plan5_model_bundle(source_checkpoint, ppo_config, device=env.device)
        learner = Plan5PPO(
            models,
            ppo_config,
            num_envs=env.num_envs,
            num_steps_per_env=ARGS.steps_per_env,
            device=env.device,
        )
        start_iteration = 0
        if ARGS.resume is not None:
            resume_state = torch.load(ARGS.resume.expanduser().resolve(), map_location=env.device)
            start_iteration = learner.load_training_state_dict(resume_state)

        run_config = {
            "git_commit": _git_commit(),
            "started_unix_time": time.time(),
            "seed": ARGS.seed,
            "iterations_requested": ARGS.iterations,
            "start_iteration": start_iteration,
            "num_envs": env.num_envs,
            "num_agents": 2,
            "steps_per_env": ARGS.steps_per_env,
            "source_checkpoint": str(source_checkpoint.relative_to(REPO_ROOT)),
            "motion_file": motion_config.motion_file,
            "robot_urdf": CONFIG.robot.asset.urdf_file,
            "object_urdf": CONFIG.robot.object.object_urdf_path,
            "reward_terms": list(env.reward_manager.active_terms),
            "termination_terms": list(env.termination_manager.active_terms),
            "actor_obs_dim": 158,
            "critic_obs_dim": 527,
            "action_dim_per_agent": 29,
            "actor_learning_rate": ppo_config.actor_learning_rate,
            "critic_learning_rate": ppo_config.critic_learning_rate,
            "critic_only": ARGS.critic_only,
            "gamma": ppo_config.gamma,
            "lambda": ppo_config.lam,
            "clip_param": ppo_config.clip_param,
            "num_learning_epochs": ppo_config.num_learning_epochs,
            "num_mini_batches": ppo_config.num_mini_batches,
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
            update_metrics = learner.update(update_actor=not ARGS.critic_only)
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
                    "reward_terms_raw_mean",
                }
            ):
                raise RuntimeError(f"Non-finite training metric at iteration {iteration}: {metrics}")
            if not all(torch.isfinite(torch.tensor(value)) for value in reward_terms.values()):
                raise RuntimeError(f"Non-finite reward term at iteration {iteration}: {reward_terms}")
            _append_jsonl(metrics_path, metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)

            if iteration % ARGS.save_interval == 0 or iteration == final_iteration:
                torch.save(
                    learner.training_state_dict(iteration=iteration),
                    output_dir / f"model_{iteration:05d}.pt",
                )

        final_report = {
            "passed": True,
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
        raise RuntimeError("Plan 5 Push training failed") from failure


if __name__ == "__main__":
    main()
