#!/usr/bin/env python3
"""Train the isolated CORE4D small-table shared-Actor MAPPO baseline."""

from __future__ import annotations

import argparse
import hashlib
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
PARSER.add_argument("--iterations", type=int, default=12000)
PARSER.add_argument("--num-envs", type=int, default=2048)
PARSER.add_argument("--steps-per-env", type=int, default=24)
PARSER.add_argument("--seed", type=int, default=721)
PARSER.add_argument("--save-interval", type=int, default=2000)
PARSER.add_argument("--output-dir", type=Path, default=None)
PARSER.add_argument("--resume", type=Path, default=None)
ARGS = PARSER.parse_args()
if ARGS.output_dir is None:
    ARGS.output_dir = (
        REPO_ROOT
        / "logs/Core4DSmallTable"
        / f"paired_reference_fresh_seed{ARGS.seed}_env{ARGS.num_envs}"
    )
for name in ("iterations", "num_envs", "steps_per_env", "save_interval"):
    if getattr(ARGS, name) < 1:
        PARSER.error(f"--{name.replace('_', '-')} must be positive")
if int(os.environ.get("WORLD_SIZE", "1")) != 1:
    PARSER.error("CORE4D small-table training is one independent single-GPU process")

from holosoma.agents.mappo.core4d_smalltable_initialization import (  # noqa: E402
    initialize_core4d_smalltable_model_bundle,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (  # noqa: E402
    CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
    CORE4D_SMALLTABLE_PHYSICS_CONTRACT,
    CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
    CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
    Core4DSmallTablePPO,
)
from holosoma.config_values.marl.g1.core4d_smalltable_command import (  # noqa: E402
    CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (  # noqa: E402
    CORE4D_SMALLTABLE_OBJECT_URDF,
    g1_29dof_core4d_smalltable_baseline,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402


CONFIG = replace(
    g1_29dof_core4d_smalltable_baseline,
    training=replace(
        g1_29dof_core4d_smalltable_baseline.training,
        num_envs=ARGS.num_envs,
        headless=True,
        seed=ARGS.seed,
    ),
)
RUNTIME_REFERENCE_PATH = (
    REPO_ROOT / "src" / "holosoma" / CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE
).resolve()
OBJECT_URDF_PATH = (
    REPO_ROOT / "src" / "holosoma" / CORE4D_SMALLTABLE_OBJECT_URDF
).resolve()
TRAINING_PROMOTION_PATH = RUNTIME_REFERENCE_PATH.with_name(
    "training_asset_manifest.json"
)
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


EXPECTED_OBJECT_MASS_KG = CORE4D_SMALLTABLE_PHYSICS_CONTRACT["object_mass_kg"]
EXPECTED_OBJECT_MATERIAL = CORE4D_SMALLTABLE_PHYSICS_CONTRACT[
    "material_static_dynamic_restitution"
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"CORE4D {label} is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"CORE4D {label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _object_physics(env) -> dict[str, object]:
    view = env.simulator._object.root_physx_view
    mass = view.get_masses().reshape(-1)
    material = view.get_material_properties()
    expected_mass = torch.full_like(mass, EXPECTED_OBJECT_MASS_KG)
    expected_material = torch.as_tensor(
        EXPECTED_OBJECT_MATERIAL,
        device=material.device,
        dtype=material.dtype,
    ).expand_as(material)
    if not torch.allclose(mass, expected_mass, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError(f"CORE4D table mass mismatch: {mass.detach().cpu().tolist()}")
    if not torch.allclose(material, expected_material, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError(
            "CORE4D table material mismatch: "
            f"{material.detach().cpu().tolist()}"
        )
    return {
        "mass_kg": mass.detach().cpu().tolist(),
        "material_static_dynamic_restitution": material.detach().cpu().tolist(),
        "inertia_kg_m2_row_major": view.get_inertias().detach().cpu().tolist(),
        "com_pose_body_xyzw": view.get_coms().detach().cpu().tolist(),
    }


def _simulation_rates(env) -> tuple[float, float]:
    physics_hz = 1.0 / float(env.sim_dt)
    control_hz = 1.0 / float(env.dt)
    expected = (
        float(CORE4D_SMALLTABLE_PHYSICS_CONTRACT["physics_hz"]),
        float(CORE4D_SMALLTABLE_PHYSICS_CONTRACT["control_hz"]),
    )
    if abs(physics_hz - expected[0]) > 1.0e-6 or abs(control_hz - expected[1]) > 1.0e-6:
        raise RuntimeError(
            "CORE4D simulation-rate mismatch: "
            f"expected {expected}, got {(physics_hz, control_hz)}"
        )
    return physics_hz, control_hz


def main() -> int:
    env = None
    output_dir = ARGS.output_dir.expanduser().resolve()
    output_created = False
    try:
        runtime_reference_sha256 = _require_sha256(
            RUNTIME_REFERENCE_PATH,
            CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
            "runtime reference",
        )
        object_urdf_sha256 = _require_sha256(
            OBJECT_URDF_PATH,
            CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
            "object URDF",
        )
        training_promotion_sha256 = _require_sha256(
            TRAINING_PROMOTION_PATH,
            CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
            "training-asset promotion",
        )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                "Output directory is not empty: "
                f"{output_dir}; every fresh or resumed segment needs a new directory"
            )
        resume_path = (
            None if ARGS.resume is None else ARGS.resume.expanduser().resolve()
        )
        if resume_path is not None and not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_created = True
        torch.manual_seed(ARGS.seed)

        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        observations = env.reset_all()
        physics = _object_physics(env)
        physics_hz, control_hz = _simulation_rates(env)
        ppo_config = replace(
            CONFIG.algo.config,
            num_learning_iterations=ARGS.iterations,
            num_steps_per_env=ARGS.steps_per_env,
            save_interval=ARGS.save_interval,
        )
        models = initialize_core4d_smalltable_model_bundle(
            ppo_config,
            device=env.device,
        )
        learner = Core4DSmallTablePPO(
            models,
            ppo_config,
            num_envs=env.num_envs,
            num_steps_per_env=ARGS.steps_per_env,
            device=env.device,
        )

        start_iteration = 0
        resume_checkpoint_sha256 = None
        if resume_path is not None:
            state = torch.load(
                resume_path,
                map_location=env.device,
                weights_only=False,
            )
            start_iteration = learner.load_training_state_dict(state)
            resume_checkpoint_sha256 = _sha256(resume_path)

        run_config = {
            "git_commit": _git_commit(),
            "started_unix_time": time.time(),
            "scenario": "core4d_paired_small_table_reference_tracking",
            "initialization": "fresh" if resume_path is None else "resumed",
            "resume_checkpoint": None if resume_path is None else str(resume_path),
            "resume_checkpoint_sha256": resume_checkpoint_sha256,
            "seed": ARGS.seed,
            "iterations_requested": ARGS.iterations,
            "start_iteration": start_iteration,
            "num_envs": env.num_envs,
            "num_agents": 2,
            "steps_per_env": ARGS.steps_per_env,
            "save_interval": ARGS.save_interval,
            "actor_observation_groups": {
                "actor_obs": 154,
                "teammate_obs": 4,
            },
            "actor_input_dim": 158,
            "action_dim_per_agent": 29,
            "critic_input_dim": 527,
            "shared_actor": True,
            "runtime_reference": str(RUNTIME_REFERENCE_PATH.relative_to(REPO_ROOT)),
            "runtime_reference_sha256": runtime_reference_sha256,
            "reference_policy": "non_looping_position_and_orientation_tracking",
            "object_urdf": CONFIG.robot.object.object_urdf_path,
            "object_urdf_sha256": object_urdf_sha256,
            "training_promotion": str(TRAINING_PROMOTION_PATH.relative_to(REPO_ROOT)),
            "training_promotion_sha256": training_promotion_sha256,
            "object_physics": physics,
            "physics_hz": physics_hz,
            "control_hz": control_hz,
            "reward_terms": list(env.reward_manager.active_terms),
            "termination_terms": list(env.termination_manager.active_terms),
            "gamma": ppo_config.gamma,
            "lambda": ppo_config.lam,
            "clip_param": ppo_config.clip_param,
            "entropy_coef": ppo_config.entropy_coef,
            "actor_learning_rate": learner.actor_learning_rate,
            "critic_learning_rate": learner.critic_learning_rate,
        }
        _write_json(output_dir / "run_config.json", run_config)
        torch.save(
            learner.training_state_dict(iteration=start_iteration),
            output_dir / f"model_{start_iteration:05d}.pt",
        )

        metrics_path = output_dir / "metrics.jsonl"
        final_iteration = start_iteration + ARGS.iterations
        for iteration in range(start_iteration + 1, final_iteration + 1):
            started = time.perf_counter()
            observations = learner.collect_rollout(env, observations)
            rewards = learner.storage.team("rewards").clone()
            dones = learner.storage.team("dones").clone()
            timeouts = learner.storage.team("timeouts").clone()
            metrics = learner.update()
            reset_count = int(dones.count_nonzero().item())
            completion_count = int(timeouts.count_nonzero().item())
            physical_failure_count = reset_count - completion_count
            command_metrics = env.log_dict
            record = {
                "iteration": iteration,
                "elapsed_seconds": time.perf_counter() - started,
                "reward_mean": rewards.mean().item(),
                "reward_min": rewards.min().item(),
                "reward_max": rewards.max().item(),
                "reset_count": reset_count,
                "timeout_count": completion_count,
                "reference_completion_count": completion_count,
                "physical_failure_count": physical_failure_count,
                "reference_completion_fraction_of_resets": (
                    completion_count / reset_count if reset_count else None
                ),
                "last_step_agent_ref_position_error_mean_m": float(
                    command_metrics["motion/error_ref_pos_mean"].mean().item()
                ),
                "last_step_agent_ref_position_error_max_m": float(
                    command_metrics["motion/error_ref_pos_max"].max().item()
                ),
                "last_step_object_position_error_mean_m": float(
                    command_metrics["motion/error_object_pos"].mean().item()
                ),
                "actor_learning_rate": learner.actor_learning_rate,
                "critic_learning_rate": learner.critic_learning_rate,
                **metrics.__dict__,
            }
            numeric = [value for value in record.values() if isinstance(value, (int, float))]
            if not all(torch.isfinite(torch.tensor(value)) for value in numeric):
                raise RuntimeError(f"Non-finite metric at iteration {iteration}")
            _append_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)

            if iteration % ARGS.save_interval == 0 or iteration == final_iteration:
                torch.save(
                    learner.training_state_dict(iteration=iteration),
                    output_dir / f"model_{iteration:05d}.pt",
                )

        status = {
            "passed": True,
            "final_iteration": final_iteration,
            "final_checkpoint": str(output_dir / f"model_{final_iteration:05d}.pt"),
        }
        _write_json(output_dir / "status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        traceback.print_exc()
        if output_created:
            _write_json(
                output_dir / "status.json",
                {"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
            )
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(SIMULATION_APP)


if __name__ == "__main__":
    raise SystemExit(main())
