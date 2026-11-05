"""Warm-start the Demo 4 Actor and construct a fresh team Critic."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from holosoma.agents.mappo.demo4_checkpoint import (
    DEMO4_ACTOR_COMPATIBILITY_VERSION,
    expand_plan5_pull_actor_for_demo4,
    validate_demo4_lossless_expansion,
)
from holosoma.agents.mappo.initialization import (
    FrozenEmpiricalNormalization,
    Plan5ModelBundle,
)
from holosoma.agents.modules.module_utils import (
    setup_ppo_actor_module,
    setup_ppo_critic_module,
)
from holosoma.agents.ppo.ppo import EmpiricalNormalization
from holosoma.config_types.algo import PPOConfig
from holosoma.utils.helpers import instantiate


DEMO4_NUM_AGENTS = 2
DEMO4_ACTOR_OBS_DIM = 164
DEMO4_CRITIC_OBS_DIM = 527
DEMO4_ACTION_DIM = 29
DEMO4_SOURCE_PULL_ITERATION = 8050
DEMO4_SOURCE_PULL_CHECKPOINT_SHA256 = (
    "727630e9cec654d88bfb454d5eaabd70d7db629478598a2039140653a57cbf43"
)


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a checkpoint without loading any of its state."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_demo4_model_bundle(
    checkpoint_path: str | Path,
    config: PPOConfig,
    *,
    device: str = "cpu",
    expected_sha256: str = DEMO4_SOURCE_PULL_CHECKPOINT_SHA256,
) -> Plan5ModelBundle:
    """Expand only the accepted Pull Actor and create all training state fresh."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    source_sha256 = checkpoint_sha256(checkpoint_path)
    if source_sha256 != expected_sha256:
        raise ValueError(
            "Demo 4 source Pull checkpoint SHA256 mismatch: "
            f"expected {expected_sha256}, got {source_sha256}"
        )

    source = torch.load(checkpoint_path, map_location=device, weights_only=False)
    converted = expand_plan5_pull_actor_for_demo4(
        source,
        source_file_sha256=source_sha256,
    )
    validate_demo4_lossless_expansion(source, converted)
    compatibility = converted["demo4_actor_compatibility"]
    if compatibility.get("version") != DEMO4_ACTOR_COMPATIBILITY_VERSION:
        raise ValueError("Demo 4 Actor compatibility version mismatch")
    source_iteration = int(converted["iter"])
    if source_iteration != DEMO4_SOURCE_PULL_ITERATION:
        raise ValueError(
            f"Demo 4 requires accepted Pull iteration {DEMO4_SOURCE_PULL_ITERATION}, "
            f"got {source_iteration}"
        )

    obs_dim_dict = {
        "actor_obs": 154,
        "teammate_obs": 4,
        "table_obs": 6,
        "critic_obs": DEMO4_CRITIC_OBS_DIM,
    }
    history_length = {name: 1 for name in obs_dim_dict}
    actor_config = replace(
        copy.deepcopy(config.module_dict.actor),
        input_dim=["actor_obs", "teammate_obs", "table_obs"],
    )
    critic_config = replace(
        copy.deepcopy(config.module_dict.critic),
        input_dim=["critic_obs"],
    )
    actor = setup_ppo_actor_module(
        obs_dim_dict=obs_dim_dict,
        module_config=actor_config,
        num_actions=DEMO4_ACTION_DIM,
        init_noise_std=config.init_noise_std,
        device=device,
        history_length=history_length,
    )
    actor.load_state_dict(converted["actor_model_state_dict"], strict=True)
    critic = setup_ppo_critic_module(
        obs_dim_dict=obs_dim_dict,
        module_config=critic_config,
        device=device,
        history_length=history_length,
    )

    actor_obs_normalizer = FrozenEmpiricalNormalization(
        DEMO4_ACTOR_OBS_DIM,
        device=device,
    )
    actor_obs_normalizer.load_state_dict(
        converted["actor_obs_normalizer_state_dict"],
        strict=True,
    )
    actor_obs_normalizer.eval()
    critic_obs_normalizer = EmpiricalNormalization(
        DEMO4_CRITIC_OBS_DIM,
        device=device,
    )
    critic_obs_normalizer.train()

    # Neither Adam state is loaded from Plan 5.  Demo 4 starts two fresh
    # optimizers even though the Actor parameters carry the accepted skill.
    actor_optimizer = instantiate(
        config.actor_optimizer,
        params=actor.parameters(),
        lr=config.actor_learning_rate,
    )
    critic_optimizer = instantiate(
        config.critic_optimizer,
        params=critic.parameters(),
        lr=config.critic_learning_rate,
    )
    return Plan5ModelBundle(
        actor=actor,
        critic=critic,
        actor_obs_normalizer=actor_obs_normalizer,
        critic_obs_normalizer=critic_obs_normalizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        source_iteration=source_iteration,
        source_sha256=source_sha256,
    )


__all__ = [
    "DEMO4_ACTION_DIM",
    "DEMO4_ACTOR_OBS_DIM",
    "DEMO4_CRITIC_OBS_DIM",
    "DEMO4_NUM_AGENTS",
    "DEMO4_SOURCE_PULL_CHECKPOINT_SHA256",
    "DEMO4_SOURCE_PULL_ITERATION",
    "checkpoint_sha256",
    "initialize_demo4_model_bundle",
]
