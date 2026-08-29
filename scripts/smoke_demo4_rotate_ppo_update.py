#!/usr/bin/env python3
"""Run one real-CUDA Demo 4 rollout, cooperative PPO update, and restore."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma.config_values.marl.g1.demo4_experiment import (  # noqa: E402
    g1_29dof_demo4_rotate_smoke,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402


CONFIG = g1_29dof_demo4_rotate_smoke
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.agents.mappo.demo4_initialization import (  # noqa: E402
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


def _maximum_parameter_change(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> float:
    return max(torch.max(torch.abs(before[key] - after[key])).item() for key in before)


def main() -> int:
    env = None
    failure: BaseException | None = None
    try:
        torch.manual_seed(721)
        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        env.set_is_evaluating()
        observations = env.reset_all()
        config = replace(
            CONFIG.algo.config,
            num_steps_per_env=4,
            num_learning_epochs=1,
            num_mini_batches=1,
        )
        models = initialize_demo4_model_bundle(
            SOURCE_CHECKPOINT,
            config,
            device=env.device,
        )
        learner = Demo4PPO(
            models,
            config,
            num_envs=env.num_envs,
            num_steps_per_env=4,
            device=env.device,
        )
        actor_before = {
            key: value.clone() for key, value in models.actor.state_dict().items()
        }
        critic_before = {
            key: value.clone() for key, value in models.critic.state_dict().items()
        }
        actor_normalizer_before = {
            key: value.clone()
            for key, value in models.actor_obs_normalizer.state_dict().items()
        }

        observations = learner.collect_rollout(env, observations)
        rewards = learner.storage.team("rewards").clone()
        advantages = learner.storage.team("advantages").clone()
        metrics = learner.update()
        actor_change = _maximum_parameter_change(
            actor_before,
            models.actor.state_dict(),
        )
        critic_change = _maximum_parameter_change(
            critic_before,
            models.critic.state_dict(),
        )
        if actor_change <= 0.0 or critic_change <= 0.0:
            raise RuntimeError(
                f"Demo 4 PPO did not update both networks: {actor_change}, {critic_change}"
            )
        for key, value in actor_normalizer_before.items():
            torch.testing.assert_close(
                value,
                models.actor_obs_normalizer.state_dict()[key],
            )

        state = learner.training_state_dict(iteration=1)
        if "plan5_mappo" in state or "demo3_mappo" in state:
            raise RuntimeError("Demo 4 checkpoint contains cross-demo metadata")
        restored_models = initialize_demo4_model_bundle(
            SOURCE_CHECKPOINT,
            config,
            device=env.device,
        )
        restored = Demo4PPO(
            restored_models,
            config,
            num_envs=env.num_envs,
            num_steps_per_env=4,
            device=env.device,
        )
        if restored.load_training_state_dict(state) != 1:
            raise RuntimeError("Demo 4 checkpoint restored the wrong iteration")
        for key, value in models.actor.state_dict().items():
            torch.testing.assert_close(value, restored_models.actor.state_dict()[key])
        for key, value in models.critic.state_dict().items():
            torch.testing.assert_close(value, restored_models.critic.state_dict()[key])

        finite = all(
            torch.isfinite(tensor).all()
            for tensor in (
                observations["actor_obs"],
                observations["teammate_obs"],
                observations["table_obs"],
                observations["critic_obs"],
                rewards,
                advantages,
            )
        ) and all(torch.isfinite(torch.tensor(value)) for value in metrics.__dict__.values())
        if not finite:
            raise RuntimeError("Demo 4 PPO smoke produced non-finite state")

        report = {
            "passed": True,
            "rollout_steps": 4,
            "num_envs": env.num_envs,
            "num_agents": 2,
            "actor_observation_shape": list(learner.storage.agent("actor_obs").shape),
            "critic_observation_shape": list(learner.storage.team("critic_obs").shape),
            "action_shape": list(learner.storage.agent("actions").shape),
            "team_reward_shape": list(rewards.shape),
            "team_advantage_shape": list(advantages.shape),
            "actor_max_parameter_change": actor_change,
            "critic_max_parameter_change": critic_change,
            "actor_normalizer_frozen": True,
            "checkpoint_round_trip": True,
            "checkpoint_metadata": state["demo4_mappo"],
            "metrics": metrics.__dict__,
        }
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(SIMULATION_APP)
        if failure is not None:
            print(f"Demo 4 PPO update smoke failed: {failure}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
