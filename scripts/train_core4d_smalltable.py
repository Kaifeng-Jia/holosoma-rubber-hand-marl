#!/usr/bin/env python3
"""Train a selected CORE4D paired-reference shared-Actor MAPPO baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--experiment", choices=("smalltable", "chair021"), default="smalltable")
PARSER.add_argument("--iterations", type=int, default=12000)
PARSER.add_argument("--num-envs", type=int, default=2048)
PARSER.add_argument("--steps-per-env", type=int, default=24)
PARSER.add_argument("--seed", type=int, default=721)
PARSER.add_argument("--save-interval", type=int, default=2000)
PARSER.add_argument("--output-dir", type=Path, default=None)
PARSER.add_argument("--resume", type=Path, default=None)
PARSER.add_argument("--reward-variant", choices=("baseline", "interaction_mesh"), default="baseline")
PARSER.add_argument("--interaction-reference", type=Path, default=None)
PARSER.add_argument(
    "--object-z-error-weight", type=float, default=1.0,
    help="Coefficient of squared world-z position error; x/y stay 1, smalltable only if non-default",
)
ARGS = PARSER.parse_args()
if not math.isfinite(ARGS.object_z_error_weight) or ARGS.object_z_error_weight <= 0.0:
    PARSER.error("--object-z-error-weight must be positive and finite")
if ARGS.experiment != "smalltable" and ARGS.object_z_error_weight != 1.0:
    PARSER.error("Non-default --object-z-error-weight is restricted to --experiment smalltable")
if ARGS.reward_variant == "interaction_mesh" and ARGS.interaction_reference is None:
    PARSER.error("--reward-variant interaction_mesh requires --interaction-reference")
if ARGS.reward_variant == "baseline" and ARGS.interaction_reference is not None:
    PARSER.error("--interaction-reference requires the explicit interaction_mesh reward variant")
for name in ("iterations", "num_envs", "steps_per_env", "save_interval"):
    if getattr(ARGS, name) < 1:
        PARSER.error(f"--{name.replace('_', '-')} must be positive")
if int(os.environ.get("WORLD_SIZE", "1")) != 1:
    PARSER.error("CORE4D paired training is one independent single-GPU process")

from holosoma.agents.mappo.core4d_smalltable_initialization import (  # noqa: E402
    initialize_core4d_smalltable_model_bundle,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (  # noqa: E402
    Core4DSmallTablePPO,
    build_core4d_interaction_contract,
    core4d_object_position_tracking_contract,
)
from holosoma.config_values.marl.g1.core4d_pair_experiments import (  # noqa: E402
    get_core4d_pair_experiment,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (  # noqa: E402
    g1_29dof_core4d_smalltable_baseline,
    with_interaction_mesh_reward,
    with_pair_experiment,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402
from holosoma.config_values.marl.g1.core4d_smalltable_reward import (  # noqa: E402
    with_object_z_error_weight,
)


EXPERIMENT = get_core4d_pair_experiment(ARGS.experiment)
if ARGS.reward_variant == "interaction_mesh" and not EXPERIMENT.allow_interaction_mesh:
    PARSER.error(f"--experiment {ARGS.experiment} does not allow interaction_mesh")
if not EXPERIMENT.training_ready:
    PARSER.error(f"--experiment {ARGS.experiment} is not training_ready; complete asset/physics preparation first")
if ARGS.output_dir is None:
    variant_suffix = "" if ARGS.reward_variant == "baseline" else "_interaction_mesh"
    if ARGS.object_z_error_weight != 1.0:
        variant_suffix += f"_zweight{ARGS.object_z_error_weight:g}"
    ARGS.output_dir = (
        REPO_ROOT
        / "logs" / EXPERIMENT.project
        / f"paired_reference_fresh{variant_suffix}_seed{ARGS.seed}_env{ARGS.num_envs}"
    )
INTERACTION_REFERENCE = (
    None if ARGS.interaction_reference is None else ARGS.interaction_reference.expanduser().resolve()
)
INTERACTION_CONTRACT = (
    None if INTERACTION_REFERENCE is None else build_core4d_interaction_contract(INTERACTION_REFERENCE)
)
PAIR_CONFIG = with_pair_experiment(g1_29dof_core4d_smalltable_baseline, EXPERIMENT)
BASE_CONFIG = (
    PAIR_CONFIG
    if INTERACTION_REFERENCE is None
    else with_interaction_mesh_reward(PAIR_CONFIG, str(INTERACTION_REFERENCE))
)
if ARGS.object_z_error_weight != 1.0:
    BASE_CONFIG = replace(
        BASE_CONFIG,
        reward=with_object_z_error_weight(BASE_CONFIG.reward, ARGS.object_z_error_weight),
    )
CONFIG = replace(
    BASE_CONFIG,
    training=replace(
        BASE_CONFIG.training,
        num_envs=ARGS.num_envs,
        headless=True,
        seed=ARGS.seed,
    ),
)
RUNTIME_REFERENCE_PATH = (
    REPO_ROOT / "src" / "holosoma" / EXPERIMENT.runtime_reference_file
).resolve()
OBJECT_URDF_PATH = (
    REPO_ROOT / "src" / "holosoma" / EXPERIMENT.object_urdf_file
).resolve()
TRAINING_PROMOTION_PATH = (
    REPO_ROOT / "src" / "holosoma" / EXPERIMENT.training_promotion_file
).resolve()
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


EXPECTED_OBJECT_MASS_KG = EXPERIMENT.object_mass_kg
EXPECTED_OBJECT_MATERIAL = EXPERIMENT.material_static_dynamic_restitution


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


def _save_checkpoint_atomic(path: Path, state: dict) -> None:
    """Expose a .pt file to cloud sync only after serialization has completed."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
        float(EXPERIMENT.physics_hz),
        float(EXPERIMENT.control_hz),
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
            EXPERIMENT.runtime_reference_sha256,
            "runtime reference",
        )
        object_urdf_sha256 = _require_sha256(
            OBJECT_URDF_PATH,
            EXPERIMENT.object_urdf_sha256,
            "object URDF",
        )
        training_promotion_sha256 = _require_sha256(
            TRAINING_PROMOTION_PATH,
            EXPERIMENT.training_promotion_sha256,
            "training-asset promotion",
        )
        training_robot_urdf_sha256 = None
        if INTERACTION_CONTRACT is not None:
            asset_root = CONFIG.robot.asset.asset_root
            if asset_root.startswith("@holosoma/"):
                asset_root = asset_root.replace("@holosoma", get_holosoma_root())
            robot_path = (Path(asset_root) / CONFIG.robot.asset.urdf_file).resolve()
            training_robot_urdf_sha256 = _require_sha256(
                robot_path, INTERACTION_CONTRACT["training_robot_urdf_sha256"], "interaction robot URDF"
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
            interaction_contract=INTERACTION_CONTRACT,
            experiment_contract=EXPERIMENT.checkpoint_contract,
            object_z_error_weight=ARGS.object_z_error_weight,
        )
        interaction_term = (
            env.reward_manager.get_term("interaction_mesh")
            if "interaction_mesh" in env.reward_manager.active_terms else None
        )
        if (interaction_term is not None) != (INTERACTION_CONTRACT is not None):
            raise RuntimeError("Interaction reward and checkpoint contract disagree")

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
            "experiment": EXPERIMENT.experiment_id,
            "project": EXPERIMENT.project,
            "scenario": EXPERIMENT.scenario,
            "experiment_contract": EXPERIMENT.checkpoint_contract,
            "reference_frames": EXPERIMENT.reference_frames,
            "reference_fps": EXPERIMENT.reference_fps,
            "object_collider_type": EXPERIMENT.object_collider_type,
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
            "reward_variant": ARGS.reward_variant,
            "object_z_error_weight": ARGS.object_z_error_weight,
            "object_position_tracking_contract": core4d_object_position_tracking_contract(
                ARGS.object_z_error_weight
            ),
            "interaction_reference": None if INTERACTION_REFERENCE is None else str(INTERACTION_REFERENCE),
            "interaction_contract": INTERACTION_CONTRACT,
            "training_robot_urdf_sha256": training_robot_urdf_sha256,
            "termination_terms": list(env.termination_manager.active_terms),
            "gamma": ppo_config.gamma,
            "lambda": ppo_config.lam,
            "clip_param": ppo_config.clip_param,
            "entropy_coef": ppo_config.entropy_coef,
            "actor_learning_rate": learner.actor_learning_rate,
            "critic_learning_rate": learner.critic_learning_rate,
        }
        _write_json(output_dir / "run_config.json", run_config)
        _save_checkpoint_atomic(
            output_dir / f"model_{start_iteration:05d}.pt",
            learner.training_state_dict(iteration=start_iteration),
        )

        metrics_path = output_dir / "metrics.jsonl"
        if interaction_term is not None:
            interaction_term.get_iteration_diagnostics(reset=True)
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
            if interaction_term is not None:
                record.update({
                    name: float(value.item())
                    for name, value in interaction_term.get_iteration_diagnostics(reset=True).items()
                })
            numeric = [value for value in record.values() if isinstance(value, (int, float))]
            if not all(torch.isfinite(torch.tensor(value)) for value in numeric):
                raise RuntimeError(f"Non-finite metric at iteration {iteration}")
            _append_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)

            if iteration % ARGS.save_interval == 0 or iteration == final_iteration:
                _save_checkpoint_atomic(
                    output_dir / f"model_{iteration:05d}.pt",
                    learner.training_state_dict(iteration=iteration),
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
