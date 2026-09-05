#!/usr/bin/env python3
"""Run one real-CUDA Plan 5 stochastic rollout and PPO update."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_push_baseline
from holosoma.utils.eval_utils import init_sim_imports


CONFIG = g1_29dof_plan5_push_baseline
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle  # noqa: E402
from holosoma.agents.mappo.ppo import Plan5PPO  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


def _maximum_parameter_change(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> float:
    return max(torch.max(torch.abs(before[key] - after[key])).item() for key in before)


def main() -> None:
    env = None
    failure: BaseException | None = None
    try:
        torch.manual_seed(721)
        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        env.set_is_evaluating()
        observations = env.reset_all()

        config = replace(
            g1_29dof_wbt_w_object.algo.config,
            num_learning_epochs=1,
            num_mini_batches=1,
        )
        checkpoint = REPO_ROOT / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"
        models = initialize_plan5_model_bundle(checkpoint, config, device=env.device)
        learner = Plan5PPO(
            models,
            config,
            num_envs=env.num_envs,
            num_steps_per_env=4,
            device=env.device,
        )
        actor_before = {key: value.clone() for key, value in models.actor.state_dict().items()}
        critic_before = {key: value.clone() for key, value in models.critic.state_dict().items()}
        actor_normalizer_before = {
            key: value.clone() for key, value in models.actor_obs_normalizer.state_dict().items()
        }

        observations = learner.collect_rollout(env, observations)
        reward_history = learner.storage.team("rewards").clone()
        done_history = learner.storage.team("dones").clone()
        log_prob_history = learner.storage.agent("actions_log_prob").clone()
        metrics = learner.update()

        actor_change = _maximum_parameter_change(actor_before, models.actor.state_dict())
        critic_change = _maximum_parameter_change(critic_before, models.critic.state_dict())
        if actor_change <= 0.0 or critic_change <= 0.0:
            raise RuntimeError(
                f"PPO update did not change both models: actor={actor_change}, critic={critic_change}"
            )
        for key, value in actor_normalizer_before.items():
            torch.testing.assert_close(value, models.actor_obs_normalizer.state_dict()[key])

        state = learner.training_state_dict(iteration=1)
        restored_models = initialize_plan5_model_bundle(checkpoint, config, device=env.device)
        restored = Plan5PPO(
            restored_models,
            config,
            num_envs=env.num_envs,
            num_steps_per_env=4,
            device=env.device,
        )
        restored_iteration = restored.load_training_state_dict(state)
        if restored_iteration != 1:
            raise RuntimeError(f"Unexpected restored iteration: {restored_iteration}")
        for key, value in models.actor.state_dict().items():
            torch.testing.assert_close(value, restored_models.actor.state_dict()[key])
        for key, value in models.critic.state_dict().items():
            torch.testing.assert_close(value, restored_models.critic.state_dict()[key])

        finite = all(
            torch.isfinite(tensor).all()
            for tensor in (
                env.simulator.agent_root_states,
                env.simulator.agent_dof_pos,
                observations["actor_obs"],
                observations["teammate_obs"],
                observations["critic_obs"],
                reward_history,
                log_prob_history,
            )
        ) and all(torch.isfinite(torch.tensor(value)) for value in metrics.__dict__.values())
        if not finite:
            raise RuntimeError("Non-finite state, distribution statistic, or PPO metric")
        robot_urdf = CONFIG.robot.asset.urdf_file.lower()
        if "rubberhand" not in robot_urdf or "hemisphere" in robot_urdf:
            raise RuntimeError(f"Unexpected robot asset: {CONFIG.robot.asset.urdf_file}")

        report = {
            "passed": True,
            "rollout_steps": 4,
            "num_envs": env.num_envs,
            "num_agents": 2,
            "actor_observation_shape": list(learner.storage.agent("actor_obs").shape),
            "critic_observation_shape": list(learner.storage.team("critic_obs").shape),
            "action_shape": list(learner.storage.agent("actions").shape),
            "team_reward_shape": list(reward_history.shape),
            "team_advantage_shape": list(learner.storage.team("advantages").shape),
            "reward_min": reward_history.min().item(),
            "reward_max": reward_history.max().item(),
            "reset_count": int(done_history.count_nonzero().item()),
            "log_prob_min": log_prob_history.min().item(),
            "log_prob_max": log_prob_history.max().item(),
            "actor_max_parameter_change": actor_change,
            "critic_max_parameter_change": critic_change,
            "actor_normalizer_frozen": True,
            "checkpoint_round_trip": True,
            "robot_urdf": CONFIG.robot.asset.urdf_file,
            "metrics": metrics.__dict__,
        }
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
        print(
            json.dumps(
                {"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
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
        raise RuntimeError("Plan 5 PPO update smoke failed") from failure


if __name__ == "__main__":
    main()
