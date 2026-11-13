"""T06 / T07: turn a CORE4D pair config (D's Table / Bin training or evaluation config) into the G1 + Sharpa version.

with_sharpa_pair(config, finger_mode, reference_file) changes only what the hand needs (system1/hand/hand_env.py):
robot (system1/hand/robot_config.py), the paired motion reference (73-joint runtime reference from
system1/hand/sharpa_reference.py), the action term (finger mode), the body-DOF observation / reward terms plus the
two finger observation terms, and the object-hand contact bodies (all Sharpa links with a collider). Everything else -
rewards, weights, terminations, PPO, environments, iterations - stays D's.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_types.observation import ObsTermCfg
from system1.hand.robot_config import with_sharpa_hands
from system1.hand.sharpa import SIDES, SharpaHand

H = "system1.hand.hand_env"


def sharpa_contact_bodies() -> list[str]:
    """Every Sharpa link with a collision mesh, left hand then right hand (the object-hand contact filters)."""
    out = []
    for s in SIDES:
        seen = []
        for c in SharpaHand(s).collisions:
            if c.link not in seen:
                seen.append(c.link)
        out += seen
    return out


def _remap_command(command, reference_file: str):
    def remap(terms):
        out = dict(terms)
        t = out["paired_motion_command"]
        prm = dict(t.params)
        prm["motion_config"] = replace(prm["motion_config"], motion_file=reference_file)
        prm["paired_reference_file"] = reference_file
        out["paired_motion_command"] = replace(t, params=prm)
        return out
    return replace(command, setup_terms=remap(command.setup_terms), reset_terms=remap(command.reset_terms),
                   step_terms=remap(command.step_terms))


def _hand_observation(observation):
    groups = dict(observation.groups)
    actor = groups["actor_obs"]
    at = dict(actor.terms)
    for name, fn in (("motion_command", "hand_paired_motion_command"), ("dof_pos", "hand_paired_dof_pos"),
                     ("dof_vel", "hand_paired_dof_vel"), ("actions", "hand_paired_actions")):
        at[name] = replace(at[name], func=f"{H}:{fn}")
    at["sharpa_finger_pos"] = ObsTermCfg(func=f"{H}:sharpa_finger_pos", scale=1.0, noise=at["dof_pos"].noise)
    at["sharpa_finger_ref"] = ObsTermCfg(func=f"{H}:sharpa_finger_ref", scale=1.0, noise=at["motion_command"].noise)
    groups["actor_obs"] = replace(actor, terms=at)
    critic = groups["critic_obs"]
    ct = dict(critic.terms)
    for name, fn in (("agent_dof_pos", "hand_centralized_agent_dof_pos"), ("agent_dof_vel", "hand_centralized_agent_dof_vel"),
                     ("agent_actions", "hand_centralized_agent_actions"), ("shared_motion_command", "hand_centralized_shared_motion_command")):
        ct[name] = replace(ct[name], func=f"{H}:{fn}")
    ct["sharpa_agent_finger_pos"] = ObsTermCfg(func=f"{H}:sharpa_agent_finger_pos", scale=1.0, noise=0.0)
    ct["sharpa_agent_finger_ref"] = ObsTermCfg(func=f"{H}:sharpa_agent_finger_ref", scale=1.0, noise=0.0)
    groups["critic_obs"] = replace(critic, terms=ct)
    return replace(observation, groups=groups)


def _hand_reward(reward):
    terms = dict(reward.terms)
    terms["action_rate_l2"] = replace(terms["action_rate_l2"], func=f"{H}:hand_penalty_action_rate")
    terms["limits_dof_pos"] = replace(terms["limits_dof_pos"], func=f"{H}:hand_limits_dof_pos")
    return replace(reward, terms=terms)


def with_sharpa_pair(config, finger_mode: str, reference_file: str | Path):
    ref = str(Path(reference_file).resolve())
    action = ActionManagerCfg(terms={"joint_control": ActionTermCfg(func=f"{H}:SharpaDualJointPositionActionTerm",
                                                                     params={"finger_mode": finger_mode}, scale=1.0, clip=None)})
    sim = config.simulator
    sim = replace(sim, config=replace(sim.config, object_hand_contact_bodies=sharpa_contact_bodies()))
    return replace(config, robot=with_sharpa_hands(config.robot), command=_remap_command(config.command, ref), action=action,
                   observation=_hand_observation(config.observation), reward=_hand_reward(config.reward), simulator=sim)


# actor input = D's 154 actor terms + 88 finger terms + 4 teammate terms; critic = 527 + 2 x 88
HAND_ACTOR_OBS = 154 + 88
HAND_TEAMMATE_OBS = 4
HAND_CRITIC_OBS = 527 + 176


def policy_action_dim(finger_mode: str) -> int:
    return 29 + (44 if finger_mode == "learn" else 0)


__all__ = ["HAND_ACTOR_OBS", "HAND_CRITIC_OBS", "HAND_TEAMMATE_OBS", "policy_action_dim", "sharpa_contact_bodies", "with_sharpa_pair"]
