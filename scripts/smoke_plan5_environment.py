#!/usr/bin/env python3
"""Run real-CUDA one-step or multi-step gates for the Plan 5 environment."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma.config_values.marl.g1.experiment import (
    g1_29dof_plan5_pull_baseline,
    g1_29dof_plan5_pull_smoke,
    g1_29dof_plan5_push_baseline,
    g1_29dof_plan5_push_smoke,
)
from holosoma.utils.eval_utils import init_sim_imports


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--skill", choices=("push", "pull"), default="push")
PARSER.add_argument("--baseline-reward", action="store_true")
PARSER.add_argument("--steps", type=int, default=1)
PARSER.add_argument("--record-output", type=str, default=None)
PARSER.add_argument("--mappo-checkpoint", type=str, default=None)
PARSER.add_argument("--seed", type=int, default=42)
PARSER.add_argument("--object-centric", action="store_true")
ARGS = PARSER.parse_args()
if ARGS.steps < 1:
    PARSER.error("--steps must be at least 1")
USE_BASELINE_CONFIG = ARGS.baseline_reward or ARGS.object_centric
if ARGS.skill == "pull":
    CONFIG = g1_29dof_plan5_pull_baseline if USE_BASELINE_CONFIG else g1_29dof_plan5_pull_smoke
    CONFIG_LABEL = (
        "g1_29dof_plan5_pull_baseline"
        if USE_BASELINE_CONFIG
        else "g1_29dof_plan5_pull_smoke"
    )
    ACTOR_CHECKPOINT = (
        REPO_ROOT / "logs/WholeBodyTracking/marl_compat_pull_v1/model_07999_actor158.pt"
    )
    EXPECTED_ACTOR_SHA256 = (
        "f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364"
    )
else:
    CONFIG = g1_29dof_plan5_push_baseline if USE_BASELINE_CONFIG else g1_29dof_plan5_push_smoke
    CONFIG_LABEL = (
        "g1_29dof_plan5_push_baseline"
        if USE_BASELINE_CONFIG
        else "g1_29dof_plan5_push_smoke"
    )
    ACTOR_CHECKPOINT = (
        REPO_ROOT / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"
    )
    EXPECTED_ACTOR_SHA256 = None
CONFIG = replace(CONFIG, training=replace(CONFIG.training, seed=ARGS.seed))
if ARGS.record_output is not None:
    CONFIG = replace(
        CONFIG,
        simulator=replace(
            CONFIG.simulator,
            config=replace(
                CONFIG.simulator.config,
                enable_object_contact_diagnostics=True,
            ),
        ),
    )
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle  # noqa: E402
from holosoma.agents.mappo.ppo import Plan5PPO  # noqa: E402
from holosoma.agents.mappo.runner import Plan5PolicyRunner  # noqa: E402
from holosoma.agents.callbacks.recording import EvalRecordingCallback  # noqa: E402
from holosoma.config_types.eval_callback import RecordingConfig  # noqa: E402
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object  # noqa: E402
from holosoma.envs.marl import compute_object_reference_metrics  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


def main() -> None:
    env = None
    failure: BaseException | None = None
    try:
        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        env.set_is_evaluating()
        env_ids = torch.arange(env.num_envs, device=env.device)
        env.reset_envs_idx(env_ids)
        env._refresh_envs_after_reset(env_ids)

        simulator = env.simulator
        command = env.command_manager.get_state("paired_motion_command")
        if command is None:
            raise RuntimeError("paired_motion_command was not created")
        joint_tracking_term = env.termination_manager._term_instances.get("joint_bad_tracking")
        if ARGS.object_centric and (
            joint_tracking_term is None or not hasattr(joint_tracking_term, "last_diagnostics")
        ):
            raise RuntimeError("Object-centric evaluation requires joint tracking diagnostics")

        initial_root = simulator.agent_root_states.clone()
        initial_dof_pos = simulator.agent_dof_pos.clone()
        initial_object_pos = command.simulator_object_pos_w.clone()
        object_physx_view = simulator._object.root_physx_view
        runtime_object_tensors = {
            "mass_kg": object_physx_view.get_masses(),
            "inertia_kg_m2_row_major": object_physx_view.get_inertias(),
            "com_pose_body_xyzw": object_physx_view.get_coms(),
            "material_static_dynamic_restitution": object_physx_view.get_material_properties(),
        }
        if not all(torch.isfinite(value).all() for value in runtime_object_tensors.values()):
            raise RuntimeError("Non-finite runtime object physics properties")
        runtime_object_physics = {
            name: {
                "shape": list(value.shape),
                "values": value.detach().cpu().tolist(),
            }
            for name, value in runtime_object_tensors.items()
        }
        lateral_spacing = torch.linalg.vector_norm(
            initial_root[:, 1, :3] - initial_root[:, 0, :3], dim=-1
        )
        reference_initial_root_pos = command.agent_root_pos_w.clone()
        robot_urdf = (
            REPO_ROOT
            / "src"
            / "holosoma"
            / "holosoma"
            / "data"
            / "robots"
            / CONFIG.robot.asset.urdf_file
        )
        robot_urdf_text = robot_urdf.read_text()
        required_asset_tokens = {
            "left_rubber_hand_link",
            "right_rubber_hand_link",
            "left_rubber_hand.STL",
            "right_rubber_hand.STL",
        }
        missing_asset_tokens = sorted(
            token for token in required_asset_tokens if token not in robot_urdf_text
        )
        forbidden_asset_tokens = sorted(
            token for token in {"hemisphere", "hemispherical"} if token in robot_urdf_text.lower()
        )

        if initial_root.shape != (1, 2, 13):
            raise RuntimeError(f"Unexpected root shape: {tuple(initial_root.shape)}")
        if initial_dof_pos.shape != (1, 2, 29):
            raise RuntimeError(f"Unexpected DOF shape: {tuple(initial_dof_pos.shape)}")
        if not torch.isfinite(initial_root).all() or not torch.isfinite(initial_dof_pos).all():
            raise RuntimeError("Non-finite robot state after paired reset")
        if not torch.isfinite(initial_object_pos).all():
            raise RuntimeError("Non-finite object state after paired reset")
        if ARGS.skill == "push":
            if not torch.allclose(
                lateral_spacing, torch.full_like(lateral_spacing, 0.8), atol=1.0e-4
            ):
                raise RuntimeError(f"Unexpected robot spacing: {lateral_spacing.tolist()}")
        else:
            if command.paired_reference_file is None:
                raise RuntimeError("Pull smoke requires an explicit paired reference file")
            if not torch.allclose(
                initial_root[..., :3], reference_initial_root_pos, atol=1.0e-4
            ):
                raise RuntimeError(
                    "Pull reset positions do not match the explicit paired reference"
                )
        if missing_asset_tokens:
            raise RuntimeError(f"Rubber-hand URDF is incomplete: {missing_asset_tokens}")
        if forbidden_asset_tokens:
            raise RuntimeError(f"Hemisphere-hand tokens are present: {forbidden_asset_tokens}")

        observations = env.observation_manager.compute()
        initialization_kwargs = {}
        if EXPECTED_ACTOR_SHA256 is not None:
            initialization_kwargs["expected_sha256"] = EXPECTED_ACTOR_SHA256
        models = initialize_plan5_model_bundle(
            ACTOR_CHECKPOINT,
            g1_29dof_wbt_w_object.algo.config,
            device=env.device,
            **initialization_kwargs,
        )
        restored_iteration = None
        if ARGS.mappo_checkpoint is not None:
            mappo_checkpoint = Path(ARGS.mappo_checkpoint).expanduser().resolve()
            learner = Plan5PPO(
                models,
                g1_29dof_wbt_w_object.algo.config,
                num_envs=env.num_envs,
                device=env.device,
            )
            state = torch.load(mappo_checkpoint, map_location=env.device, weights_only=False)
            restored_iteration = learner.load_training_state_dict(state)
        runner = Plan5PolicyRunner(models)
        recorder = None
        if ARGS.record_output is not None:
            training_loop = SimpleNamespace(
                device=env.device,
                actor_obs_keys=["actor_obs", "teammate_obs"],
                _unwrap_env=lambda: env,
            )
            recorder = EvalRecordingCallback(
                RecordingConfig(
                    enabled=True,
                    output_path=ARGS.record_output,
                    env_id=0,
                ),
                training_loop,
            )
            recorder.on_pre_evaluate_policy()
        rewards = []
        resets = []
        term_samples: dict[str, list[torch.Tensor]] = {
            name: [] for name in env.reward_manager.active_terms
        }
        termination_samples: dict[str, list[torch.Tensor]] = {
            name: [] for name in env.termination_manager.active_terms
        }
        reference_object_positions: list[torch.Tensor] = []
        actual_object_positions: list[torch.Tensor] = []
        reference_object_quaternions: list[torch.Tensor] = []
        actual_object_quaternions: list[torch.Tensor] = []
        evaluated_reference_frames: list[torch.Tensor] = []
        first_termination_components = None
        decision = None
        rollout_step_limit = command.reference.num_frames if ARGS.object_centric else ARGS.steps
        executed_steps = 0
        for step in range(rollout_step_limit):
            decision = runner.decide(
                observations,
                update_critic_normalizer=False,
            )
            actor_state = {
                "step": step,
                "obs": observations,
                "actions": decision.actions,
            }
            if recorder is not None:
                recorder.on_pre_eval_env_step(actor_state)
            observations, reward, done, extras = env.step({"actions": decision.actions})
            executed_steps = step + 1
            actor_state.update(
                obs=observations,
                rewards=reward,
                dones=done,
                extras=extras,
            )
            if recorder is not None:
                recorder.on_post_eval_env_step(actor_state)
            rewards.append(reward.detach().clone())
            resets.append(done.detach().clone())
            for name, value in extras.get("termination_terms", {}).items():
                termination_samples[name].append(value.detach().clone())
            for name, cfg in zip(
                env.reward_manager._term_names,
                env.reward_manager._term_cfgs,
            ):
                if name in env.reward_manager._term_instances:
                    raw = env.reward_manager._term_instances[name](env, **cfg.params)
                else:
                    raw = env.reward_manager._term_funcs[name](env, **cfg.params)
                if raw.shape != (env.num_envs,) or not torch.isfinite(raw).all():
                    raise RuntimeError(f"Invalid diagnostic reward term {name}: {raw}")
                term_samples[name].append(raw.detach().clone())
            if ARGS.object_centric:
                diagnostics = joint_tracking_term.last_diagnostics
                required_diagnostics = {
                    "bad_robot_ref_height_by_agent",
                    "bad_robot_low_height_by_agent",
                    "bad_robot_orientation_by_agent",
                    "bad_robot_body_height_by_agent",
                    "bad_robot",
                    "robot_ref_height_error_m_by_agent",
                    "robot_ref_height_reference_m_by_agent",
                    "robot_ref_height_actual_m_by_agent",
                    "robot_gravity_z_error_by_agent",
                    "robot_reference_gravity_z_by_agent",
                    "robot_actual_gravity_z_by_agent",
                    "robot_body_height_error_m_by_agent",
                    "robot_max_body_height_error_m_by_agent",
                    "bad_object_position",
                    "bad_object_orientation",
                    "object_position_error_m",
                    "object_orientation_error_rad",
                    "reference_object_position",
                    "actual_object_position",
                    "reference_object_quaternion",
                    "actual_object_quaternion",
                    "reference_frame",
                }
                if set(diagnostics) != required_diagnostics:
                    raise RuntimeError(
                        "Incomplete joint tracking diagnostics: "
                        f"{sorted(set(diagnostics) ^ required_diagnostics)}"
                    )
                reference_object_positions.append(
                    diagnostics["reference_object_position"][0].detach().clone()
                )
                actual_object_positions.append(
                    diagnostics["actual_object_position"][0].detach().clone()
                )
                reference_object_quaternions.append(
                    diagnostics["reference_object_quaternion"][0].detach().clone()
                )
                actual_object_quaternions.append(
                    diagnostics["actual_object_quaternion"][0].detach().clone()
                )
                evaluated_reference_frames.append(diagnostics["reference_frame"][0].detach().clone())
                if bool(done[0].item()):
                    first_termination_components = {
                        "reference_frame": int(diagnostics["reference_frame"][0].item()),
                        "bad_robot_ref_height_by_agent": diagnostics[
                            "bad_robot_ref_height_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "bad_robot_low_height_by_agent": diagnostics[
                            "bad_robot_low_height_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "bad_robot_orientation_by_agent": diagnostics[
                            "bad_robot_orientation_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "bad_robot_body_height_by_agent": diagnostics[
                            "bad_robot_body_height_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "bad_robot": bool(diagnostics["bad_robot"][0].item()),
                        "robot_ref_height_error_m_by_agent": diagnostics[
                            "robot_ref_height_error_m_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "robot_ref_height_reference_m_by_agent": diagnostics[
                            "robot_ref_height_reference_m_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "robot_ref_height_actual_m_by_agent": diagnostics[
                            "robot_ref_height_actual_m_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "robot_gravity_z_error_by_agent": diagnostics[
                            "robot_gravity_z_error_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "robot_reference_gravity_z_by_agent": diagnostics[
                            "robot_reference_gravity_z_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "robot_actual_gravity_z_by_agent": diagnostics[
                            "robot_actual_gravity_z_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "robot_body_height_names": joint_tracking_term.body_pos_body_names,
                        "robot_body_height_error_m_by_agent": diagnostics[
                            "robot_body_height_error_m_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "robot_max_body_height_error_m_by_agent": diagnostics[
                            "robot_max_body_height_error_m_by_agent"
                        ][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "bad_object_position": bool(
                            diagnostics["bad_object_position"][0].item()
                        ),
                        "bad_object_orientation": bool(
                            diagnostics["bad_object_orientation"][0].item()
                        ),
                        "object_position_error_m": float(
                            diagnostics["object_position_error_m"][0].item()
                        ),
                        "object_orientation_error_rad": float(
                            diagnostics["object_orientation_error_rad"][0].item()
                        ),
                        "thresholds": {
                            "robot_ref_height_m": joint_tracking_term.ref_pos_threshold,
                            "robot_minimum_ref_body_height_m": (
                                joint_tracking_term.minimum_ref_body_height
                            ),
                            "robot_gravity_z": joint_tracking_term.ref_ori_threshold,
                            "robot_body_height_m": joint_tracking_term.body_pos_threshold,
                            "object_position_m": joint_tracking_term.object_pos_threshold,
                            "object_orientation_rad": joint_tracking_term.object_ori_threshold,
                        },
                        "timeout": bool(extras["time_outs"][0].item()),
                    }
                    break
        assert decision is not None
        if recorder is not None:
            recorder.on_post_evaluate_policy()
        reward_history = torch.stack(rewards)
        reset_history = torch.stack(resets)
        policy_actions = decision.actions
        critic_values = decision.values
        simulator.refresh_sim_tensors()
        actor_obs = observations["actor_obs"]
        teammate_obs = observations["teammate_obs"]
        critic_obs = observations["critic_obs"]
        combined_actor_obs = torch.cat((actor_obs, teammate_obs), dim=-1)
        if actor_obs.shape != (1, 2, 154):
            raise RuntimeError(f"Unexpected actor observation shape: {tuple(actor_obs.shape)}")
        if teammate_obs.shape != (1, 2, 4):
            raise RuntimeError(f"Unexpected teammate observation shape: {tuple(teammate_obs.shape)}")
        if combined_actor_obs.shape != (1, 2, 158):
            raise RuntimeError(f"Unexpected combined observation shape: {tuple(combined_actor_obs.shape)}")
        if critic_obs.shape != (1, 527):
            raise RuntimeError(f"Unexpected centralized critic observation shape: {tuple(critic_obs.shape)}")
        if policy_actions.shape != (1, 2, 29) or critic_values.shape != (1, 1):
            raise RuntimeError(
                "Unexpected online policy output: "
                f"actions={tuple(policy_actions.shape)}, values={tuple(critic_values.shape)}"
            )
        finite_after_step = bool(
            torch.isfinite(simulator.agent_root_states).all()
            and torch.isfinite(simulator.agent_dof_pos).all()
            and torch.isfinite(command.simulator_object_pos_w).all()
            and torch.isfinite(combined_actor_obs).all()
            and torch.isfinite(critic_obs).all()
            and torch.isfinite(policy_actions).all()
            and torch.isfinite(critic_values).all()
        )
        if not finite_after_step:
            raise RuntimeError("Non-finite state after one control step")

        recording_shapes = None
        if recorder is not None:
            with np.load(recorder.output_path) as recording:
                required_shapes = {
                    "policy_actor_obs": (executed_steps, 2, 158),
                    "dof_pos": (executed_steps, 2, 29),
                    "root_pos": (executed_steps, 2, 3),
                    "ref_object_pos_w": (executed_steps, 3),
                    "termination_term_joint_bad_tracking": (executed_steps,),
                }
                for name, expected_shape in required_shapes.items():
                    actual_shape = recording[name].shape
                    if actual_shape != expected_shape:
                        raise RuntimeError(
                            f"Unexpected recording channel {name}: {actual_shape}, expected {expected_shape}"
                        )
                recording_shapes = {
                    name: list(recording[name].shape) for name in required_shapes
                }
                diagnostic_channels = [
                    "object_robot_contact_force_matrix_w",
                    "object_hand_contact_force_matrix_w",
                    "object_robot_contact_pos_w",
                    "object_hand_contact_pos_w",
                ]
                for name in diagnostic_channels:
                    if name not in recording or recording[name].shape[0] != executed_steps:
                        raise RuntimeError(f"Missing or invalid object contact diagnostic: {name}")
                    recording_shapes[name] = list(recording[name].shape)
                metadata = json.loads(recording["_metadata_json"].item())
                hand_filters = metadata["object_hand_contact"]["filter_prim_paths_expr"]
                if not any("/Robot/" in path for path in hand_filters) or not any(
                    "/Robot_1/" in path for path in hand_filters
                ):
                    raise RuntimeError("Object-hand diagnostics do not cover both physical robots")

        object_reference_metrics = None
        if ARGS.object_centric:
            reference_frames = torch.stack(evaluated_reference_frames)
            expected_frames = torch.arange(
                reference_frames.shape[0],
                device=reference_frames.device,
                dtype=reference_frames.dtype,
            )
            if not torch.equal(reference_frames, expected_frames):
                raise RuntimeError(
                    "Object-centric evaluation crossed a reset or skipped reference frames: "
                    f"{reference_frames.detach().cpu().tolist()}"
                )
            full_reference_positions = (
                command.reference.object_pos_w + simulator.scene.env_origins[0]
            )
            object_reference_metrics = compute_object_reference_metrics(
                reference_positions=torch.stack(reference_object_positions),
                actual_positions=torch.stack(actual_object_positions),
                reference_quaternions=torch.stack(reference_object_quaternions),
                actual_quaternions=torch.stack(actual_object_quaternions),
                full_reference_positions=full_reference_positions,
                completed_reference=(
                    first_termination_components is None
                    and executed_steps == command.reference.num_frames
                ),
            )

        report = {
            "passed": True,
            "skill": ARGS.skill,
            "config": CONFIG_LABEL,
            "actor_checkpoint": str(ACTOR_CHECKPOINT.relative_to(REPO_ROOT)),
            "actor_checkpoint_sha256": models.source_sha256,
            "expected_actor_checkpoint_sha256": EXPECTED_ACTOR_SHA256,
            "num_envs": env.num_envs,
            "num_agents": 2,
            "agent_root_shape": list(initial_root.shape),
            "agent_dof_shape": list(initial_dof_pos.shape),
            "actor_observation_shape": list(actor_obs.shape),
            "teammate_observation_shape": list(teammate_obs.shape),
            "combined_actor_observation_shape": list(combined_actor_obs.shape),
            "centralized_critic_observation_shape": list(critic_obs.shape),
            "online_policy_action_shape": list(policy_actions.shape),
            "online_critic_value_shape": list(critic_values.shape),
            "lateral_spacing_m": lateral_spacing.detach().cpu().tolist(),
            "object_position_w": initial_object_pos.detach().cpu().tolist(),
            "runtime_object_physics": runtime_object_physics,
            "robot_urdf": str(robot_urdf.relative_to(REPO_ROOT)),
            "rubber_hand_asset_tokens": sorted(required_asset_tokens),
            "hemisphere_asset_tokens": forbidden_asset_tokens,
            "baseline_reward_enabled": ARGS.baseline_reward,
            "seed": ARGS.seed,
            "mappo_checkpoint": ARGS.mappo_checkpoint,
            "restored_iteration": restored_iteration,
            "object_centric": ARGS.object_centric,
            "rollout_step_limit": rollout_step_limit,
            "rollout_steps": executed_steps,
            "reward_min": reward_history.min().item(),
            "reward_max": reward_history.max().item(),
            "reward_mean": reward_history.mean().item(),
            "reset_count": int(reset_history.count_nonzero().item()),
            "termination_counts": {
                name: int(torch.stack(samples).count_nonzero().item())
                for name, samples in termination_samples.items()
                if samples
            },
            "reward_terms": {
                name: {
                    "raw_min": torch.stack(samples).min().item(),
                    "raw_max": torch.stack(samples).max().item(),
                    "raw_mean": torch.stack(samples).mean().item(),
                    "weight": env.reward_manager.cfg.terms[name].weight,
                }
                for name, samples in term_samples.items()
            },
            "recording_output": recorder.output_path if recorder is not None else None,
            "recording_shapes": recording_shapes,
            "object_reference_metrics": object_reference_metrics,
            "first_termination_components": first_termination_components,
            "finite_after_rollout": finite_after_step,
        }
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
        print(
            json.dumps(
                {
                    "passed": False,
                    "skill": ARGS.skill,
                    "config": CONFIG_LABEL,
                    "actor_checkpoint": str(ACTOR_CHECKPOINT.relative_to(REPO_ROOT)),
                    "expected_actor_checkpoint_sha256": EXPECTED_ACTOR_SHA256,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(SIMULATION_APP)
    if failure is not None:
        raise RuntimeError("Plan 5 CUDA smoke failed") from failure


if __name__ == "__main__":
    main()
