#!/usr/bin/env python3
"""Run real-CUDA one-step or multi-step gates for the Plan 5 environment."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma.config_values.marl.g1.experiment import (
    g1_29dof_plan5_push_baseline,
    g1_29dof_plan5_push_smoke,
)
from holosoma.utils.eval_utils import init_sim_imports


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--baseline-reward", action="store_true")
PARSER.add_argument("--steps", type=int, default=1)
ARGS = PARSER.parse_args()
if ARGS.steps < 1:
    PARSER.error("--steps must be at least 1")
CONFIG = g1_29dof_plan5_push_baseline if ARGS.baseline_reward else g1_29dof_plan5_push_smoke
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle  # noqa: E402
from holosoma.agents.mappo.runner import Plan5PolicyRunner  # noqa: E402
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object  # noqa: E402
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

        initial_root = simulator.agent_root_states.clone()
        initial_dof_pos = simulator.agent_dof_pos.clone()
        initial_object_pos = command.simulator_object_pos_w.clone()
        lateral_spacing = torch.linalg.vector_norm(
            initial_root[:, 1, :3] - initial_root[:, 0, :3], dim=-1
        )
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
        if not torch.allclose(lateral_spacing, torch.full_like(lateral_spacing, 0.8), atol=1.0e-4):
            raise RuntimeError(f"Unexpected robot spacing: {lateral_spacing.tolist()}")
        if missing_asset_tokens:
            raise RuntimeError(f"Rubber-hand URDF is incomplete: {missing_asset_tokens}")
        if forbidden_asset_tokens:
            raise RuntimeError(f"Hemisphere-hand tokens are present: {forbidden_asset_tokens}")

        observations = env.observation_manager.compute()
        checkpoint = REPO_ROOT / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"
        models = initialize_plan5_model_bundle(
            checkpoint,
            g1_29dof_wbt_w_object.algo.config,
            device=env.device,
        )
        runner = Plan5PolicyRunner(models)
        rewards = []
        resets = []
        term_samples: dict[str, list[torch.Tensor]] = {
            name: [] for name in env.reward_manager.active_terms
        }
        transition = None
        for _ in range(ARGS.steps):
            transition = runner.step_environment(
                env,
                observations,
                update_critic_normalizer=False,
            )
            observations = transition.observations
            rewards.append(transition.rewards.detach().clone())
            resets.append(transition.dones.detach().clone())
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
        assert transition is not None
        reward_history = torch.stack(rewards)
        reset_history = torch.stack(resets)
        policy_actions = transition.decision.actions
        critic_values = transition.decision.values
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

        report = {
            "passed": True,
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
            "robot_urdf": str(robot_urdf.relative_to(REPO_ROOT)),
            "rubber_hand_asset_tokens": sorted(required_asset_tokens),
            "hemisphere_asset_tokens": forbidden_asset_tokens,
            "baseline_reward_enabled": ARGS.baseline_reward,
            "rollout_steps": ARGS.steps,
            "reward_min": reward_history.min().item(),
            "reward_max": reward_history.max().item(),
            "reset_count": int(reset_history.count_nonzero().item()),
            "reward_terms": {
                name: {
                    "raw_min": torch.stack(samples).min().item(),
                    "raw_max": torch.stack(samples).max().item(),
                    "weight": env.reward_manager.cfg.terms[name].weight,
                }
                for name, samples in term_samples.items()
            },
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
