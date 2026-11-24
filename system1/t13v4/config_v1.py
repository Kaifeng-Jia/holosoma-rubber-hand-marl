"""T13 v4 section 2 switches applied to an experiment config (Table: CORE4D pair; Push: Plan5). Off = the config is returned
unchanged. Numbers: observation / noise in holosoma.managers.observation.terms.marl_v1 (NOISE_V1, NECK_POINT_TORSO_M,
clips); contact items in holosoma.managers.reward.terms.contact_v1 (CONTACT_V1)."""
from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_types.termination import TerminationTermCfg


def contact_weight(reward_cfg) -> tuple[float, str]:
    """w_c = (object position weight + object orientation weight) / 2 of this config (T13 v4 section 2, contact item 1)."""
    terms = reward_cfg.terms
    if "bucket_interaction" in terms:
        from holosoma.config_values.marl.g1.core4d_bucket_contract import bucket_block_weights
        t = terms["bucket_interaction"]
        pos, ori = bucket_block_weights(t.params["variant"])[:2]
        return t.weight * (pos + ori) / 2.0, f"bucket_interaction weight {t.weight} x (position {pos} + orientation {ori}) / 2"
    pos, ori = terms["object_global_ref_position_error_exp"].weight, terms["object_global_ref_orientation_error_exp"].weight
    return (pos + ori) / 2.0, f"(object_global_ref_position_error_exp {pos} + object_global_ref_orientation_error_exp {ori}) / 2"


def apply_t13v4(config, *, obs_v1: bool = False, critic_v1: bool = False, contact_labels: str | None = None,
                per_agent: bool = False, noise: bool = True):
    record = dict(obs_v1=obs_v1, critic_v1=critic_v1, contact_labels=contact_labels, per_agent=per_agent, noise=noise)
    if obs_v1 or critic_v1:
        from holosoma.config_values.marl.g1.observation_v1 import observation_v1
        config = replace(config, observation=observation_v1(obs_v1=obs_v1, critic_v1=critic_v1, noise=noise))
    if contact_labels is not None:
        w_c, why = contact_weight(config.reward)
        record.update(contact_weight=w_c, contact_weight_rule=why)
        terms = dict(config.reward.terms)
        terms["contact_v1"] = RewardTermCfg(func="holosoma.managers.reward.terms.contact_v1:EefContactExpV1",
                                            params={"labels_file": str(contact_labels), "per_agent": per_agent}, weight=w_c)
        tterms = dict(config.termination.terms)
        tterms["lost_contact_v1"] = TerminationTermCfg(func="holosoma.managers.reward.terms.contact_v1:LostContactV1",
                                                       params={"labels_file": str(contact_labels)})
        config = replace(config, reward=replace(config.reward, terms=terms), termination=replace(config.termination, terms=tterms),
                         simulator=replace(config.simulator, config=replace(config.simulator.config, enable_object_link_contact=True)))
    if per_agent:
        from system1.t13v4.per_agent_ppo import with_per_agent_config
        config = with_per_agent_config(config)
        record.update(env_class=config.env_class, per_agent_terms=[k for k, t in config.reward.terms.items() if "marl_per_agent" in t.func])
    return config, record


__all__ = ["apply_t13v4", "contact_weight"]
