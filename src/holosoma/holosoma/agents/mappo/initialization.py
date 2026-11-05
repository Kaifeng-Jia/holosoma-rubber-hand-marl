"""Safe warm-start contract for the first Plan 5 MAPPO model bundle."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from holosoma.agents.modules.module_utils import (
    setup_ppo_actor_module,
    setup_ppo_critic_module,
)
from holosoma.agents.ppo.ppo import EmpiricalNormalization
from holosoma.config_types.algo import PPOConfig
from holosoma.utils.helpers import instantiate


PLAN5_ACTOR_OBS_DIM = 158
PLAN5_CRITIC_OBS_DIM = 527
PLAN5_ACTION_DIM = 29
PLAN5_COMPATIBILITY_VERSION = "a1_actor_obs_154_to_158_v1"
PLAN5_ACTOR_CHECKPOINT_SHA256 = "11ef1fe7a4b340a47218f040e7675af4073067c5ce931e169818f903150e3224"


class FrozenEmpiricalNormalization(EmpiricalNormalization):
    """Empirical normalizer whose checkpoint statistics can never update."""

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        return super().forward(x, center=center, update=False)


@dataclass
class Plan5ModelBundle:
    actor: nn.Module
    critic: nn.Module
    actor_obs_normalizer: FrozenEmpiricalNormalization
    critic_obs_normalizer: EmpiricalNormalization
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    source_iteration: int
    source_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_plan5_model_bundle(
    checkpoint_path: str | Path,
    config: PPOConfig,
    *,
    device: str = "cpu",
    expected_sha256: str = PLAN5_ACTOR_CHECKPOINT_SHA256,
) -> Plan5ModelBundle:
    """Load only the 158-D actor prior and create every critic-side state fresh."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    source_sha256 = _sha256(checkpoint_path)
    if source_sha256 != expected_sha256:
        raise ValueError(
            f"Plan 5 actor checkpoint SHA256 mismatch: expected {expected_sha256}, got {source_sha256}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    compatibility = checkpoint.get("marl_compatibility", {})
    if compatibility.get("version") != PLAN5_COMPATIBILITY_VERSION:
        raise ValueError(
            "Plan 5 actor checkpoint compatibility mismatch: "
            f"expected {PLAN5_COMPATIBILITY_VERSION!r}, got {compatibility.get('version')!r}"
        )
    if checkpoint.get("actor_obs_normalizer_state_dict") is None:
        raise ValueError("Plan 5 actor checkpoint must contain its 158-D actor normalizer")

    obs_dim_dict = {
        "actor_obs": 154,
        "teammate_obs": 4,
        "critic_obs": PLAN5_CRITIC_OBS_DIM,
    }
    history_length = {name: 1 for name in obs_dim_dict}
    actor_config = replace(
        copy.deepcopy(config.module_dict.actor),
        input_dim=["actor_obs", "teammate_obs"],
    )
    critic_config = replace(
        copy.deepcopy(config.module_dict.critic),
        input_dim=["critic_obs"],
    )

    actor = setup_ppo_actor_module(
        obs_dim_dict=obs_dim_dict,
        module_config=actor_config,
        num_actions=PLAN5_ACTION_DIM,
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

    actor_obs_normalizer = FrozenEmpiricalNormalization(PLAN5_ACTOR_OBS_DIM, device=device)
    actor_obs_normalizer.load_state_dict(checkpoint["actor_obs_normalizer_state_dict"], strict=True)
    actor_obs_normalizer.eval()
    critic_obs_normalizer = EmpiricalNormalization(PLAN5_CRITIC_OBS_DIM, device=device)
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
