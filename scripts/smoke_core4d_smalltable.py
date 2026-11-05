#!/usr/bin/env python3
"""Run bounded physics and optional one-update checks for selected CORE4D paired MAPPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--experiment", choices=("smalltable", "chair021", "bucket003", "smalltable5kg_A"), default="smalltable")
PARSER.add_argument("--num-envs", type=int, default=1)
PARSER.add_argument("--steps", type=int, default=8)
PARSER.add_argument("--reference-frame", type=int, default=0,
                    help="Diagnostic reset phase only; does not change formal training initialization")
PARSER.add_argument("--seed", type=int, default=721)
PARSER.add_argument("--ppo-update", action="store_true")
PARSER.add_argument("--output", type=Path, default=None)
BUCKET_REWARD_CHOICES = ("bucket_A", "bucket_B", "bucket_A_no_rel", "bucket_A_no_height", "bucket_A_no_rel_no_height")
PARSER.add_argument("--reward-variant", choices=("baseline", "interaction_mesh", *BUCKET_REWARD_CHOICES), default="baseline")
PARSER.add_argument("--bucket-reference", type=Path, default=None)
PARSER.add_argument("--interaction-reference", type=Path, default=None)
ARGS = PARSER.parse_args()
IS_BUCKET = ARGS.reward_variant in BUCKET_REWARD_CHOICES
if ARGS.experiment == "smalltable5kg_A" and ARGS.reward_variant not in ("bucket_A", "bucket_A_no_rel_no_height"):
    PARSER.error("smalltable5kg_A requires bucket_A or bucket_A_no_rel_no_height")
if IS_BUCKET != (ARGS.experiment in ("bucket003", "smalltable5kg_A")):
    PARSER.error(f"bucket003 smoke requires an explicit reward variant from {BUCKET_REWARD_CHOICES}")
if ARGS.bucket_reference is not None and not IS_BUCKET:
    PARSER.error("--bucket-reference is only for bucket reward variants")
if IS_BUCKET and ARGS.interaction_reference is not None:
    PARSER.error("Bucket vectors use --bucket-reference, not the legacy Laplacian artifact")
if ARGS.num_envs < 1 or ARGS.steps < 1:
    PARSER.error("--num-envs and --steps must be positive")
if ARGS.reward_variant == "interaction_mesh" and ARGS.interaction_reference is None:
    PARSER.error("--reward-variant interaction_mesh requires --interaction-reference")
if ARGS.reward_variant == "baseline" and ARGS.interaction_reference is not None:
    PARSER.error("--interaction-reference requires the explicit interaction_mesh reward variant")

from holosoma.agents.mappo.core4d_smalltable_initialization import (  # noqa: E402
    initialize_core4d_smalltable_model_bundle,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (  # noqa: E402
    Core4DSmallTablePPO,
    build_core4d_interaction_contract,
)
from holosoma.agents.mappo.core4d_smalltable_runner import (  # noqa: E402
    Core4DSmallTablePolicyRunner,
)
from holosoma.config_values.marl.g1.core4d_pair_experiments import (  # noqa: E402
    get_core4d_pair_experiment,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (  # noqa: E402
    g1_29dof_core4d_smalltable_smoke,
    with_interaction_mesh_reward,
    with_pair_experiment,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402
from holosoma.config_values.marl.g1.core4d_bucket_contract import (  # noqa: E402
    BUCKET_DATA_DIR, build_bucket_reward_contract,
)


EXPERIMENT = get_core4d_pair_experiment(ARGS.experiment)
if not 0 <= ARGS.reference_frame < EXPERIMENT.reference_frames:
    PARSER.error("--reference-frame must be inside the selected clip")
if ARGS.reward_variant == "interaction_mesh" and not EXPERIMENT.allow_interaction_mesh:
    PARSER.error(f"--experiment {ARGS.experiment} does not allow interaction_mesh")
if ARGS.ppo_update and not EXPERIMENT.training_ready:
    PARSER.error(f"--experiment {ARGS.experiment} is not training_ready; run physics smoke without --ppo-update first")
INTERACTION_REFERENCE = (
    None if ARGS.interaction_reference is None else ARGS.interaction_reference.expanduser().resolve()
)
INTERACTION_CONTRACT = (
    None if INTERACTION_REFERENCE is None else build_core4d_interaction_contract(INTERACTION_REFERENCE)
)
PAIR_CONFIG = with_pair_experiment(g1_29dof_core4d_smalltable_smoke, EXPERIMENT)
BASE_CONFIG = (
    PAIR_CONFIG
    if INTERACTION_REFERENCE is None
    else with_interaction_mesh_reward(PAIR_CONFIG, str(INTERACTION_REFERENCE))
)
BUCKET_CONTRACT = None
if IS_BUCKET:
    from holosoma.config_values.marl.g1.core4d_bucket_reward import with_bucket_reward
    bucket_reference = (
        ARGS.bucket_reference.expanduser().resolve() if ARGS.bucket_reference is not None
        else REPO_ROOT / "src/holosoma" / Path(EXPERIMENT.training_promotion_file).parent / "interaction_vectors_v1.npz"
    )
    bucket_variant = ARGS.reward_variant.removeprefix("bucket_")
    BUCKET_CONTRACT = build_bucket_reward_contract(bucket_reference, bucket_variant, experiment=EXPERIMENT)
    BASE_CONFIG = with_bucket_reward(BASE_CONFIG, str(bucket_reference), bucket_variant)
CONFIG = replace(
    BASE_CONFIG,
    training=replace(
        BASE_CONFIG.training,
        num_envs=ARGS.num_envs,
        seed=ARGS.seed,
    ),
)
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.managers.command.terms.core4d_smalltable import (  # noqa: E402
    Core4DSmallTableMotionCommand,
    object_origin_velocity_to_com_velocity,
)
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


EXPECTED_MASS_KG = EXPERIMENT.object_mass_kg
EXPECTED_MATERIAL = EXPERIMENT.material_static_dynamic_restitution


def main() -> int:
    env = None
    try:
        for relative_path, expected in (
            (EXPERIMENT.runtime_reference_file, EXPERIMENT.runtime_reference_sha256),
            (EXPERIMENT.object_urdf_file, EXPERIMENT.object_urdf_sha256),
            (EXPERIMENT.training_promotion_file, EXPERIMENT.training_promotion_sha256),
        ):
            path = REPO_ROOT / "src" / "holosoma" / relative_path
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"CORE4D experiment asset SHA-256 mismatch: {path}")
        torch.manual_seed(ARGS.seed)
        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        env.set_is_evaluating()
        observations = env.reset_all()
        command = env.command_manager.get_state("paired_motion_command")
        if not isinstance(command, Core4DSmallTableMotionCommand):
            raise RuntimeError(f"Unexpected command type: {type(command)}")

        expected_shapes = {
            "actor_obs": (env.num_envs, 2, 154),
            "teammate_obs": (env.num_envs, 2, 4),
            "critic_obs": (env.num_envs, 527),
        }
        for name, expected in expected_shapes.items():
            if observations[name].shape != expected:
                raise RuntimeError(
                    f"{name} shape mismatch: {tuple(observations[name].shape)} vs {expected}"
                )
        if command.reference.fps != EXPERIMENT.reference_fps:
            raise RuntimeError(f"Runtime FPS must be {EXPERIMENT.reference_fps}, got {command.reference.fps}")
        if command.reference.num_frames != EXPERIMENT.reference_frames:
            raise RuntimeError(f"Runtime frames must be {EXPERIMENT.reference_frames}, got {command.reference.num_frames}")
        actual_rates = (1.0 / float(env.sim_dt), 1.0 / float(env.dt))
        expected_rates = (EXPERIMENT.physics_hz, EXPERIMENT.control_hz)
        if any(abs(actual - expected) > 1.0e-6 for actual, expected in zip(actual_rates, expected_rates)):
            raise RuntimeError(f"Simulation rates must be {expected_rates}, got {actual_rates}")

        # Verify the exact reset write before taking another physics step.
        env_ids = torch.arange(env.num_envs, device=env.device)
        env.reset_envs_idx(env_ids)
        if ARGS.reference_frame:
            command.time_steps.fill_(ARGS.reference_frame)
            command._write_reference_state(env_ids)
            env.simulator.write_state_updates()
        env._refresh_envs_after_reset(env_ids)
        if not bool(torch.all(command.time_steps == ARGS.reference_frame).item()):
            raise RuntimeError("Smoke diagnostic phase did not survive reset")
        object_state = env.simulator.all_root_states[command.object_indices_in_simulator]
        sample = command.reference.sample(command.time_steps)
        expected_com_velocity = object_origin_velocity_to_com_velocity(
            sample["object_lin_vel_w"],
            sample["object_ang_vel_w"],
            sample["object_quat_w"],
            command.object_com_position_b,
        )
        torch.testing.assert_close(
            object_state[:, 7:10], expected_com_velocity, rtol=0.0, atol=1.0e-5
        )
        torch.testing.assert_close(
            command.simulator_object_lin_vel_w,
            sample["object_lin_vel_w"],
            rtol=0.0,
            atol=1.0e-5,
        )
        torch.testing.assert_close(
            object_state[:, 10:13], sample["object_ang_vel_w"], rtol=0.0, atol=1.0e-5
        )
        env._compute_observations()
        observations = env.obs_buf_dict

        interaction_term = (
            env.reward_manager.get_term("interaction_mesh")
            if "interaction_mesh" in env.reward_manager.active_terms else None
        )
        interaction_check = None
        bucket_term = env.reward_manager.get_term("bucket_interaction") if IS_BUCKET else None
        if interaction_term is not None:
            params = CONFIG.reward.terms["interaction_mesh"].params
            raw = interaction_term(env, **params)
            if raw.shape != (env.num_envs,) or not torch.isfinite(raw).all():
                raise RuntimeError("Invalid interaction reward shape/value")
            # A weight-zero counterpart must differ by precisely dt * this raw
            # term. The switch is temporary and restored before PPO rollout.
            weighted = env.reward_manager.compute(env.dt).clone()
            term_cfg = env.reward_manager.get_term_cfg("interaction_mesh")
            env.reward_manager.set_term_cfg("interaction_mesh", replace(term_cfg, weight=0.0))
            try:
                without = env.reward_manager.compute(env.dt).clone()
            finally:
                env.reward_manager.set_term_cfg("interaction_mesh", term_cfg)
            torch.testing.assert_close(weighted - without, raw * term_cfg.weight * env.dt, atol=1.0e-6, rtol=1.0e-5)
            interaction_check = {
                "raw_reward_shape": list(raw.shape),
                "initial_raw_reward_mean": raw.mean().item(),
                "initial_error_m2_per_agent": interaction_term.last_error_m2.mean(dim=0).cpu().tolist(),
                "additive_dt_weight_verified": True,
            }
            interaction_term.get_iteration_diagnostics(reset=True)

        view = env.simulator._object.root_physx_view
        mass = view.get_masses().reshape(-1)
        material = view.get_material_properties()
        torch.testing.assert_close(
            mass,
            torch.full_like(mass, EXPECTED_MASS_KG),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            material,
            torch.as_tensor(
                EXPECTED_MATERIAL,
                dtype=material.dtype,
                device=material.device,
            ).expand_as(material),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

        ppo_config = replace(CONFIG.algo.config, num_steps_per_env=ARGS.steps)
        models = initialize_core4d_smalltable_model_bundle(
            ppo_config,
            device=env.device,
        )
        runner = Core4DSmallTablePolicyRunner(models)
        decision = runner.decide(observations)
        if decision.actions.shape != (env.num_envs, 2, 29):
            raise RuntimeError(f"Unexpected action shape: {tuple(decision.actions.shape)}")

        update_metrics = None
        if bucket_term is not None:
            bucket_term.get_iteration_diagnostics(reset=True)
        if ARGS.ppo_update:
            learner = Core4DSmallTablePPO(
                models,
                ppo_config,
                num_envs=env.num_envs,
                num_steps_per_env=ARGS.steps,
                device=env.device,
                interaction_contract=INTERACTION_CONTRACT,
                experiment_contract=EXPERIMENT.checkpoint_contract,
                bucket_contract=BUCKET_CONTRACT,
            )
            observations = learner.collect_rollout(env, observations)
            update_metrics = learner.update().__dict__
        else:
            for _ in range(ARGS.steps):
                decision = runner.decide(observations)
                observations, _, _, _ = env.step({"actions": decision.actions})

        robot_urdf = CONFIG.robot.asset.urdf_file.lower()
        if "rubberhand" not in robot_urdf or "hemisphere" in robot_urdf:
            raise RuntimeError(f"Unexpected robot asset: {CONFIG.robot.asset.urdf_file}")
        # Record the cooked USD configuration; this is diagnostic evidence, not
        # a new success condition or a claim that every mesh opening is retained.
        from pxr import Usd, UsdPhysics
        import omni.usd

        object_root = omni.usd.get_context().get_stage().GetPrimAtPath("/World/envs/env_0/Object")
        object_mesh_collision_approximations = []
        if object_root.IsValid():
            for prim in Usd.PrimRange(object_root, Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    object_mesh_collision_approximations.append({
                        "prim_path": str(prim.GetPath()),
                        "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                    })
        report = {
            "passed": True,
            "experiment": EXPERIMENT.experiment_id,
            "scenario": EXPERIMENT.scenario,
            "experiment_contract": EXPERIMENT.checkpoint_contract,
            "training_ready": EXPERIMENT.training_ready,
            "runtime_reference_file": EXPERIMENT.runtime_reference_file,
            "object_collider_type": EXPERIMENT.object_collider_type,
            "object_mesh_collision_approximations": object_mesh_collision_approximations,
            "num_envs": env.num_envs,
            "steps": ARGS.steps,
            "diagnostic_initial_reference_frame": ARGS.reference_frame,
            "ppo_update": ARGS.ppo_update,
            "ppo_update_metrics": update_metrics,
            "reference_frames": command.reference.num_frames,
            "reference_fps": command.reference.fps,
            "reference_non_looping": True,
            "actor_input_shape": [env.num_envs, 2, 158],
            "critic_input_shape": [env.num_envs, 527],
            "action_shape": [env.num_envs, 2, 29],
            "object_angular_velocity_reset_verified": True,
            "object_origin_com_velocity_conversion_verified": True,
            "table_reset_write_count": command.table_reset_write_count.cpu().tolist(),
            "physics_hz": 1.0 / float(env.sim_dt),
            "control_hz": 1.0 / float(env.dt),
            "mass_kg": mass.cpu().tolist(),
            "material_static_dynamic_restitution": material.cpu().tolist(),
            "robot_urdf": CONFIG.robot.asset.urdf_file,
            "object_urdf": CONFIG.robot.object.object_urdf_path,
            "reward_variant": ARGS.reward_variant,
            "interaction_contract": INTERACTION_CONTRACT,
            "interaction_initial_check": interaction_check,
            "bucket_reward_contract": BUCKET_CONTRACT,
            "bucket_rollout_diagnostics": (
                None if bucket_term is None else {
                    name: value.item() for name, value in
                    bucket_term.get_iteration_diagnostics(reset=True).items()
                }
            ),
            "interaction_rollout_diagnostics": (
                None if interaction_term is None else {
                    name: value.item()
                    for name, value in interaction_term.get_iteration_diagnostics(reset=True).items()
                }
            ),
        }
        text = json.dumps(report, indent=2, sort_keys=True)
        print(text, flush=True)
        if ARGS.output is not None:
            output = ARGS.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n")
        return 0
    except BaseException as exc:
        traceback.print_exc()
        if ARGS.output is not None:
            output = ARGS.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"passed": False, "error_type": type(exc).__name__,
                                          "error": str(exc)}, indent=2) + "\n")
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(SIMULATION_APP)


if __name__ == "__main__":
    raise SystemExit(main())
