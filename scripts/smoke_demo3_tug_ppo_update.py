#!/usr/bin/env python3
"""Run one real-CUDA Demo 3 rollout, PPO update, and checkpoint restore."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma.config_values.marl.g1.demo3_experiment import (  # noqa: E402
    g1_29dof_demo3_tug_smoke,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402


CONFIG = g1_29dof_demo3_tug_smoke
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.agents.mappo.demo3_checkpoint import (  # noqa: E402
    DEMO3_MAPPO_CHECKPOINT_VERSION,
    DEMO3_WARM_START_SHA256,
    demo3_training_contract,
)
from holosoma.agents.mappo.demo3_initialization import (  # noqa: E402
    initialize_demo3_model_bundle,
)
from holosoma.agents.mappo.demo3_ppo import Demo3PPO  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


SOURCE_CHECKPOINT = (
    REPO_ROOT / "logs/Demo3Tug/checkpoints/model_07999_actor164_table_neutral.pt"
)
NUM_ROLLOUT_STEPS = 4
EXPECTED_AGENT_SCALAR_SHAPE = (NUM_ROLLOUT_STEPS, 1, 2, 1)
EXPECTED_SHARED_SCALAR_SHAPE = (NUM_ROLLOUT_STEPS, 1, 1)


def _maximum_parameter_change(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> float:
    return max(torch.max(torch.abs(before[key] - after[key])).item() for key in before)


def _assert_shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise RuntimeError(
            f"Demo 3 {name} shape mismatch: {tuple(value.shape)} vs {expected}"
        )


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Demo 3 PPO smoke produced non-finite {name}")


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
            num_steps_per_env=NUM_ROLLOUT_STEPS,
            num_learning_epochs=1,
            num_mini_batches=1,
        )
        models = initialize_demo3_model_bundle(
            SOURCE_CHECKPOINT,
            config,
            expected_sha256=DEMO3_WARM_START_SHA256,
            device=env.device,
        )
        contract = demo3_training_contract()
        learner = Demo3PPO(
            models,
            config,
            num_envs=env.num_envs,
            num_steps_per_env=NUM_ROLLOUT_STEPS,
            device=env.device,
            training_contract=contract,
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
        rewards = learner.storage.agent("rewards").clone()
        values = learner.storage.agent("values").clone()
        returns = learner.storage.agent("returns").clone()
        advantages = learner.storage.agent("advantages").clone()
        dones = learner.storage.shared("dones").clone()
        timeouts = learner.storage.shared("timeouts").clone()

        for name, tensor in {
            "per-agent rewards": rewards,
            "per-agent values": values,
            "per-agent returns": returns,
            "per-agent advantages": advantages,
        }.items():
            _assert_shape(name, tensor, EXPECTED_AGENT_SCALAR_SHAPE)
        for name, tensor in {
            "shared dones": dones,
            "shared timeouts": timeouts,
        }.items():
            _assert_shape(name, tensor, EXPECTED_SHARED_SCALAR_SHAPE)

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
                "Demo 3 PPO did not update both networks: "
                f"actor={actor_change}, critic={critic_change}"
            )
        for key, value in actor_normalizer_before.items():
            torch.testing.assert_close(
                value,
                models.actor_obs_normalizer.state_dict()[key],
            )

        state = learner.training_state_dict(iteration=1)
        contaminants = sorted({"plan5_mappo", "demo4_mappo"}.intersection(state))
        if contaminants:
            raise RuntimeError(
                f"Demo 3 checkpoint contains cross-demo metadata: {contaminants}"
            )
        metadata = state.get("demo3_mappo")
        if not isinstance(metadata, dict):
            raise RuntimeError("Demo 3 checkpoint is missing demo3_mappo metadata")
        if metadata.get("version") != DEMO3_MAPPO_CHECKPOINT_VERSION:
            raise RuntimeError(
                "Demo 3 checkpoint version mismatch: "
                f"{metadata.get('version')!r} vs {DEMO3_MAPPO_CHECKPOINT_VERSION!r}"
            )
        if metadata.get("reward_layout") != "per_agent":
            raise RuntimeError("Demo 3 checkpoint lost its per-agent reward contract")
        if metadata.get("done_layout") != "shared_environment":
            raise RuntimeError("Demo 3 checkpoint lost its shared-done contract")

        restored_models = initialize_demo3_model_bundle(
            SOURCE_CHECKPOINT,
            config,
            expected_sha256=DEMO3_WARM_START_SHA256,
            device=env.device,
        )
        restored = Demo3PPO(
            restored_models,
            config,
            num_envs=env.num_envs,
            num_steps_per_env=NUM_ROLLOUT_STEPS,
            device=env.device,
            training_contract=contract,
        )
        if restored.load_training_state_dict(state) != 1:
            raise RuntimeError("Demo 3 checkpoint restored the wrong iteration")
        for key, value in models.actor.state_dict().items():
            torch.testing.assert_close(value, restored_models.actor.state_dict()[key])
        for key, value in models.critic.state_dict().items():
            torch.testing.assert_close(value, restored_models.critic.state_dict()[key])
        for key, value in models.actor_obs_normalizer.state_dict().items():
            torch.testing.assert_close(
                value,
                restored_models.actor_obs_normalizer.state_dict()[key],
            )
        for key, value in models.critic_obs_normalizer.state_dict().items():
            torch.testing.assert_close(
                value,
                restored_models.critic_obs_normalizer.state_dict()[key],
            )

        for name, tensor in {
            "actor observations": observations["actor_obs"],
            "teammate observations": observations["teammate_obs"],
            "table observations": observations["table_obs"],
            "critic observations": observations["critic_obs"],
            "rewards": rewards,
            "values": values,
            "returns": returns,
            "advantages": advantages,
        }.items():
            _assert_finite(name, tensor)
        for name, value in metrics.__dict__.items():
            if not torch.isfinite(torch.tensor(value)):
                raise RuntimeError(f"Demo 3 PPO smoke produced non-finite metric {name}")

        report = {
            "passed": True,
            "rollout_steps": NUM_ROLLOUT_STEPS,
            "num_envs": env.num_envs,
            "num_agents": 2,
            "actor_observation_shape": list(learner.storage.agent("actor_obs").shape),
            "critic_observation_shape": list(learner.storage.agent("critic_obs").shape),
            "action_shape": list(learner.storage.agent("actions").shape),
            "per_agent_reward_shape": list(rewards.shape),
            "per_agent_value_shape": list(values.shape),
            "per_agent_return_shape": list(returns.shape),
            "per_agent_advantage_shape": list(advantages.shape),
            "shared_done_shape": list(dones.shape),
            "shared_timeout_shape": list(timeouts.shape),
            "actor_max_parameter_change": actor_change,
            "critic_max_parameter_change": critic_change,
            "actor_normalizer_frozen": True,
            "checkpoint_round_trip": True,
            "checkpoint_metadata": metadata,
            "rollout_diagnostics": learner.last_rollout_diagnostics,
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
            print(f"Demo 3 PPO update smoke failed: {failure}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
