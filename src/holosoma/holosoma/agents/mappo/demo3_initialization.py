"""Warm-start a 164-D Demo 3 Actor and a fresh centralized critic."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from holosoma.agents.mappo.demo3_checkpoint import DEMO3_COMPATIBILITY_VERSION
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


DEMO3_ACTOR_OBS_DIM = 164
DEMO3_CRITIC_OBS_DIM = 527
DEMO3_ACTION_DIM = 29


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_demo3_model_bundle(
    checkpoint_path: str | Path,
    config: PPOConfig,
    *,
    expected_sha256: str,
    device: str = "cpu",
) -> Plan5ModelBundle:
    """Load only the expanded Actor and initialize all critic-side state fresh."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    source_sha256 = _sha256(checkpoint_path)
    if source_sha256 != expected_sha256:
        raise ValueError(
            f"Demo 3 Actor checkpoint SHA256 mismatch: expected {expected_sha256}, "
            f"got {source_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    compatibility = checkpoint.get("demo3_compatibility", {})
    if compatibility.get("version") != DEMO3_COMPATIBILITY_VERSION:
        raise ValueError(
            "Demo 3 Actor compatibility mismatch: "
            f"expected {DEMO3_COMPATIBILITY_VERSION!r}, "
            f"got {compatibility.get('version')!r}"
        )
    if checkpoint.get("actor_obs_normalizer_state_dict") is None:
        raise ValueError("Demo 3 Actor checkpoint must contain its 164-D normalizer")

    obs_dim_dict = {
        "actor_obs": 154,
        "teammate_obs": 4,
        "table_obs": 6,
        "critic_obs": DEMO3_CRITIC_OBS_DIM,
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
        num_actions=DEMO3_ACTION_DIM,
        init_noise_std=config.init_noise_std,
        device=device,
        history_length=history_length,
    )
    actor.load_state_dict(checkpoint["actor_model_state_dict"], strict=True)
    critic = setup_ppo_critic_module(
        obs_dim_dict=obs_dim_dict,
        module_config=critic_config,
        device=device,
        history_length=history_length,
    )
    actor_obs_normalizer = FrozenEmpiricalNormalization(DEMO3_ACTOR_OBS_DIM, device=device)
    actor_obs_normalizer.load_state_dict(
        checkpoint["actor_obs_normalizer_state_dict"],
        strict=True,
    )
    actor_obs_normalizer.eval()
    critic_obs_normalizer = EmpiricalNormalization(DEMO3_CRITIC_OBS_DIM, device=device)
    critic_obs_normalizer.train()
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
        source_iteration=int(checkpoint.get("iter", checkpoint.get("iteration", 0))),
        source_sha256=source_sha256,
    )


__all__ = [
    "DEMO3_ACTION_DIM",
    "DEMO3_ACTOR_OBS_DIM",
    "DEMO3_CRITIC_OBS_DIM",
    "initialize_demo3_model_bundle",
]
