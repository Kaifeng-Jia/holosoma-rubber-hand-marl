"""T06 / T07: MAPPO sizes for the G1 + Sharpa robot (CORE4D pair training path, Table / Bin).

The CORE4D MAPPO modules carry D's sizes as module constants (actor 158, critic 527, action 29) and the initializer
hard-codes the 154-D actor term block. With the hand switch on:
  * patch_core4d_dims sets those constants to the hand sizes in every module that reads them at call time
    (initialization, runner, ppo, evaluation) - the modules' functions look the names up at call time;
  * initialize_hand_model_bundle builds the same networks / normalizers / optimizers as
    initialize_core4d_smalltable_model_bundle with the hand sizes;
  * learn mode: the 44 finger rows of the actor's output layer are set to 0 (weights and bias), so the action mean of
    the finger part is 0 at iteration 0 and the finger PD targets equal the geometric grip, as in track mode.
HandTrainDiag: every 100 iterations, finger correction |scale * a_f| mean, fraction of finger targets clamped at a
joint limit, and the actor std of the finger dimensions (learn).
"""
from __future__ import annotations

import copy
from dataclasses import replace

import torch

from system1.hand.hand_config import HAND_ACTOR_OBS, HAND_CRITIC_OBS, HAND_TEAMMATE_OBS, policy_action_dim
from system1.hand.hand_env import BODY_DOFS

_MODULES = ("holosoma.agents.mappo.core4d_smalltable_initialization", "holosoma.agents.mappo.core4d_smalltable_runner",
            "holosoma.agents.mappo.core4d_smalltable_ppo", "holosoma.agents.mappo.core4d_smalltable_evaluation")


def patch_core4d_dims(finger_mode: str) -> dict:
    import importlib
    vals = dict(CORE4D_SMALLTABLE_ACTOR_OBS_DIM=HAND_ACTOR_OBS + HAND_TEAMMATE_OBS,
                CORE4D_SMALLTABLE_CRITIC_OBS_DIM=HAND_CRITIC_OBS,
                CORE4D_SMALLTABLE_ACTION_DIM=policy_action_dim(finger_mode))
    for name in _MODULES:
        m = importlib.import_module(name)
        for k, v in vals.items():
            if hasattr(m, k):
                setattr(m, k, v)
    return vals


def zero_finger_rows(actor) -> int:
    last = [m for m in actor.actor_module.module.modules() if isinstance(m, torch.nn.Linear)][-1]
    if last.out_features != BODY_DOFS + 44:
        raise ValueError(f"expected a 73-output actor, got {last.out_features}")
    with torch.no_grad():
        last.weight[BODY_DOFS:] = 0.0
        last.bias[BODY_DOFS:] = 0.0
    return last.out_features - BODY_DOFS


def initialize_hand_model_bundle(config, *, finger_mode: str, device: str = "cpu"):
    from holosoma.agents.mappo.initialization import Plan5ModelBundle
    from holosoma.agents.modules.module_utils import setup_ppo_actor_module, setup_ppo_critic_module
    from holosoma.agents.ppo.ppo import EmpiricalNormalization
    from holosoma.utils.helpers import instantiate
    obs_dim_dict = {"actor_obs": HAND_ACTOR_OBS, "teammate_obs": HAND_TEAMMATE_OBS, "critic_obs": HAND_CRITIC_OBS}
    history_length = {name: 1 for name in obs_dim_dict}
    actor = setup_ppo_actor_module(obs_dim_dict=obs_dim_dict,
                                   module_config=replace(copy.deepcopy(config.module_dict.actor), input_dim=["actor_obs", "teammate_obs"]),
                                   num_actions=policy_action_dim(finger_mode), init_noise_std=config.init_noise_std, device=device,
                                   history_length=history_length)
    critic = setup_ppo_critic_module(obs_dim_dict=obs_dim_dict,
                                     module_config=replace(copy.deepcopy(config.module_dict.critic), input_dim=["critic_obs"]),
                                     device=device, history_length=history_length)
    if finger_mode == "learn":
        zero_finger_rows(actor)
    an = EmpiricalNormalization(HAND_ACTOR_OBS + HAND_TEAMMATE_OBS, device=device)
    cn = EmpiricalNormalization(HAND_CRITIC_OBS, device=device)
    an.train()
    cn.train()
    return Plan5ModelBundle(actor=actor, critic=critic, actor_obs_normalizer=an, critic_obs_normalizer=cn,
                            actor_optimizer=instantiate(config.actor_optimizer, params=actor.parameters(), lr=config.actor_learning_rate),
                            critic_optimizer=instantiate(config.critic_optimizer, params=critic.parameters(), lr=config.critic_learning_rate),
                            source_iteration=0, source_sha256=f"system1_t06_hand_fresh_{finger_mode}_v1")


class HandTrainDiag:
    every = 100

    def __init__(self, env):
        self.term = env.action_manager.get_term("joint_control")

    def summary(self, actor) -> dict:
        d = self.term.finger_diag
        n = int(d.get("steps", 0))
        out = {k: float(v) / max(n, 1) for k, v in d.items() if k != "steps"}
        std = actor.std.detach()
        out.update(policy_steps=n, finger_mode=self.term.finger_mode, body_action_std_mean=float(std[:BODY_DOFS].mean()))
        if std.numel() > BODY_DOFS:
            out["finger_action_std_mean"] = float(std[BODY_DOFS:].mean())
        self.term.finger_diag = {}
        return out


__all__ = ["HandTrainDiag", "initialize_hand_model_bundle", "patch_core4d_dims", "zero_finger_rows"]
