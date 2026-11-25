"""T13 v4: MAPPO sizes and initialization for observation v1 / critic v1 (switches; default sizes are untouched).

The MAPPO modules carry the legacy sizes as module constants (actor 154 + teammate 4 = 158, critic 527). With the v1
switches, patch_dims() sets them in every module that reads them at call time (same mechanism as system1/hand/hand_mappo.py):
  CORE4D path (Table): core4d_smalltable_{initialization,runner,ppo,evaluation};
  Plan5 path (Push):   runner.PLAN5_RUNNER_*, ppo.PLAN5_PPO_CRITIC_OBS_DIM, initialization.PLAN5_*.
Initializers:
  * initialize_core4d_v1_model_bundle: as initialize_core4d_smalltable_model_bundle (fresh networks / normalizers /
    optimizers), with the v1 sizes;
  * initialize_plan5_v1_model_bundle: D-Push (T13 section 1: "the new inputs' weights are zero, the rest loaded as is"):
    loads the frozen 158-D A1 prior (same checkpoint and sha256 check as initialize_plan5_model_bundle) into a 159-D actor:
    columns 0..153 (actor block) copied, the 2 partner-position columns copied from the prior's 2 teammate-position
    columns, the prior's 2 teammate-velocity columns dropped (obs v1 removes that input), the 3 object columns zero; the
    frozen normalizer gets the prior's mean / var for the copied inputs and mean 0, var 1 for the object inputs.
    fresh_actor=True (T13 v4 round 2 C-out-Push, the T03 C-out Pull recipe on the Plan5 path): the actor keeps its random
    initialization (not loaded), the frozen normalizer is built from the prior exactly as above.
"""
from __future__ import annotations

import copy
import importlib
from dataclasses import replace

import torch

LEGACY = dict(actor=154, teammate=4, critic=527)
_CORE4D = ("holosoma.agents.mappo.core4d_smalltable_initialization", "holosoma.agents.mappo.core4d_smalltable_runner",
           "holosoma.agents.mappo.core4d_smalltable_ppo", "holosoma.agents.mappo.core4d_smalltable_evaluation")


def patch_dims(*, teammate: int, critic: int, actor: int = 154) -> dict:
    vals = dict(CORE4D_SMALLTABLE_ACTOR_OBS_DIM=actor + teammate, CORE4D_SMALLTABLE_CRITIC_OBS_DIM=critic)
    for name in _CORE4D:
        m = importlib.import_module(name)
        for k, v in vals.items():
            if hasattr(m, k):
                setattr(m, k, v)
        if hasattr(m, "CORE4D_SMALLTABLE_ACTOR_GROUPS"):
            m.CORE4D_SMALLTABLE_ACTOR_GROUPS = (("actor_obs", actor), ("teammate_obs", teammate))
    runner = importlib.import_module("holosoma.agents.mappo.runner")
    runner.PLAN5_RUNNER_ACTOR_OBS_DIM, runner.PLAN5_RUNNER_TEAMMATE_OBS_DIM, runner.PLAN5_RUNNER_CRITIC_OBS_DIM = actor, teammate, critic
    importlib.import_module("holosoma.agents.mappo.ppo").PLAN5_PPO_CRITIC_OBS_DIM = critic
    init = importlib.import_module("holosoma.agents.mappo.initialization")
    init.PLAN5_ACTOR_OBS_DIM, init.PLAN5_CRITIC_OBS_DIM = actor + teammate, critic
    return dict(actor_obs=actor, teammate_obs=teammate, actor_input=actor + teammate, critic_obs=critic)


def _networks(config, obs_dim_dict, num_actions, device):
    from holosoma.agents.modules.module_utils import setup_ppo_actor_module, setup_ppo_critic_module
    history_length = {name: 1 for name in obs_dim_dict}
    actor = setup_ppo_actor_module(obs_dim_dict=obs_dim_dict,
                                   module_config=replace(copy.deepcopy(config.module_dict.actor), input_dim=["actor_obs", "teammate_obs"]),
                                   num_actions=num_actions, init_noise_std=config.init_noise_std, device=device, history_length=history_length)
    critic = setup_ppo_critic_module(obs_dim_dict=obs_dim_dict,
                                     module_config=replace(copy.deepcopy(config.module_dict.critic), input_dim=["critic_obs"]),
                                     device=device, history_length=history_length)
    return actor, critic


def initialize_core4d_v1_model_bundle(config, *, teammate: int, critic: int, device: str = "cpu"):
    from holosoma.agents.mappo.initialization import Plan5ModelBundle
    from holosoma.agents.ppo.ppo import EmpiricalNormalization
    from holosoma.utils.helpers import instantiate
    obs_dim_dict = {"actor_obs": 154, "teammate_obs": teammate, "critic_obs": critic}
    actor, critic_net = _networks(config, obs_dim_dict, 29, device)
    an, cn = EmpiricalNormalization(154 + teammate, device=device), EmpiricalNormalization(critic, device=device)
    an.train()
    cn.train()
    return Plan5ModelBundle(actor=actor, critic=critic_net, actor_obs_normalizer=an, critic_obs_normalizer=cn,
                            actor_optimizer=instantiate(config.actor_optimizer, params=actor.parameters(), lr=config.actor_learning_rate),
                            critic_optimizer=instantiate(config.critic_optimizer, params=critic_net.parameters(), lr=config.critic_learning_rate),
                            source_iteration=0, source_sha256=f"system1_t13v4_fresh_teammate{teammate}_critic{critic}_v1")


def _first_linear_key(state_dict: dict, in_features: int) -> str:
    keys = [k for k, v in state_dict.items() if k.endswith("weight") and v.ndim == 2 and v.shape[1] == in_features]
    if len(keys) != 1:
        raise ValueError(f"expected exactly one actor input layer with {in_features} inputs, found {keys}")
    return keys[0]


def remap_prior_actor_state(state_dict: dict, *, teammate: int) -> dict:
    """158-D prior (154 actor + teammate pos 2 + teammate vel 2) -> 154 + v1 teammate (partner pos 2 + object 3)."""
    if teammate != 5:
        raise ValueError("only the obs-v1 layout (partner 2 + object 3) is defined")
    out = dict(state_dict)
    key = _first_linear_key(state_dict, 158)
    w = state_dict[key]
    new = torch.zeros(w.shape[0], 159, dtype=w.dtype, device=w.device)
    new[:, :154] = w[:, :154]
    new[:, 154:156] = w[:, 154:156]  # partner relative position <- prior teammate relative position
    out[key] = new  # prior teammate velocity columns (156, 157) dropped; object columns 156..158 zero
    return out


def remap_prior_normalizer(state: dict) -> dict:
    out = dict(state)
    for name, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        if name in state:
            t = state[name]
            new = torch.full((*t.shape[:-1], 159), fill, dtype=t.dtype, device=t.device)
            new[..., :156] = t[..., :156]
            out[name] = new
    return out


def initialize_plan5_v1_model_bundle(checkpoint_path, config, *, critic: int, device: str = "cpu", expected_sha256: str | None = None,
                                     fresh_actor: bool = False):
    import holosoma.agents.mappo.initialization as init_mod
    from holosoma.agents.mappo.initialization import FrozenEmpiricalNormalization, Plan5ModelBundle, _sha256
    from holosoma.agents.ppo.ppo import EmpiricalNormalization
    from holosoma.utils.helpers import instantiate
    from pathlib import Path
    path = Path(checkpoint_path).expanduser().resolve()
    sha = _sha256(path)
    expected = expected_sha256 or init_mod.PLAN5_ACTOR_CHECKPOINT_SHA256
    if sha != expected:
        raise ValueError(f"Plan 5 actor checkpoint SHA256 mismatch: expected {expected}, got {sha}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("marl_compatibility", {}).get("version") != init_mod.PLAN5_COMPATIBILITY_VERSION:
        raise ValueError("Plan 5 actor checkpoint compatibility mismatch")
    obs_dim_dict = {"actor_obs": 154, "teammate_obs": 5, "critic_obs": critic}
    actor, critic_net = _networks(config, obs_dim_dict, 29, device)
    if not fresh_actor:  # fresh_actor (T13 v4 round 2 C-out-Push): random actor as T03 C-out Pull (--from-scratch), normalizer below as is
        actor.load_state_dict(remap_prior_actor_state(ckpt["actor_model_state_dict"], teammate=5), strict=True)
    an = FrozenEmpiricalNormalization(159, device=device)
    an.load_state_dict(remap_prior_normalizer(ckpt["actor_obs_normalizer_state_dict"]), strict=True)
    an.eval()
    cn = EmpiricalNormalization(critic, device=device)
    cn.train()
    return Plan5ModelBundle(actor=actor, critic=critic_net, actor_obs_normalizer=an, critic_obs_normalizer=cn,
                            actor_optimizer=instantiate(config.actor_optimizer, params=actor.parameters(), lr=config.actor_learning_rate),
                            critic_optimizer=instantiate(config.critic_optimizer, params=critic_net.parameters(), lr=config.critic_learning_rate),
                            source_iteration=int(ckpt.get("iter", ckpt.get("iteration", 0))), source_sha256=sha)


__all__ = ["LEGACY", "initialize_core4d_v1_model_bundle", "initialize_plan5_v1_model_bundle", "patch_dims",
           "remap_prior_actor_state", "remap_prior_normalizer"]
