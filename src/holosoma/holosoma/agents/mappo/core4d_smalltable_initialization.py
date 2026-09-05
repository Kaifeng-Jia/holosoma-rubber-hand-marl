"""Fresh shared-Actor/team-Critic initialization for CORE4D small-table MAPPO."""

from __future__ import annotations

import copy
from dataclasses import replace

from holosoma.agents.mappo.initialization import Plan5ModelBundle
from holosoma.agents.modules.module_utils import (
    setup_ppo_actor_module,
    setup_ppo_critic_module,
)
from holosoma.agents.ppo.ppo import EmpiricalNormalization
from holosoma.config_types.algo import PPOConfig
from holosoma.utils.helpers import instantiate


CORE4D_SMALLTABLE_NUM_AGENTS = 2
CORE4D_SMALLTABLE_ACTOR_OBS_DIM = 164
CORE4D_SMALLTABLE_CRITIC_OBS_DIM = 527
CORE4D_SMALLTABLE_ACTION_DIM = 29
CORE4D_SMALLTABLE_INITIALIZATION = "fresh_random_initialization_v1"


def initialize_core4d_smalltable_model_bundle(
    config: PPOConfig,
    *,
    device: str = "cpu",
) -> Plan5ModelBundle:
    """Create every network, normalizer, and optimizer from scratch."""
    obs_dim_dict = {
        "actor_obs": 154,
        "teammate_obs": 4,
        "table_obs": 6,
        "critic_obs": CORE4D_SMALLTABLE_CRITIC_OBS_DIM,
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
        num_actions=CORE4D_SMALLTABLE_ACTION_DIM,
        init_noise_std=config.init_noise_std,
        device=device,
        history_length=history_length,
    )
    critic = setup_ppo_critic_module(
        obs_dim_dict=obs_dim_dict,
        module_config=critic_config,
        device=device,
        history_length=history_length,
    )
    actor_obs_normalizer = EmpiricalNormalization(
        CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
        device=device,
    )
    critic_obs_normalizer = EmpiricalNormalization(
        CORE4D_SMALLTABLE_CRITIC_OBS_DIM,
        device=device,
    )
    actor_obs_normalizer.train()
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
        source_iteration=0,
        source_sha256=CORE4D_SMALLTABLE_INITIALIZATION,
    )


__all__ = [
    "CORE4D_SMALLTABLE_ACTION_DIM",
    "CORE4D_SMALLTABLE_ACTOR_OBS_DIM",
    "CORE4D_SMALLTABLE_CRITIC_OBS_DIM",
    "CORE4D_SMALLTABLE_INITIALIZATION",
    "CORE4D_SMALLTABLE_NUM_AGENTS",
    "initialize_core4d_smalltable_model_bundle",
]
