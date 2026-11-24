"""System1 T13 v4: observation v1 presets (actor 154 + teammate 5 = 159; critic 585 with both robots' commands).

Switch: the training scripts' --obs-v1 / --critic-v1 select these presets; without them the legacy presets are used
unchanged. Term order inside a group is alphabetical (ObservationManager): teammate group = neck_partner_rel_pos_b (2),
object_rel_pos_b (3) -> "neck..." < "object..." keeps the T13 order (partner first, object second).
"""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_values.marl.g1.observation import plan5_actor_obs, plan5_centralized_critic_obs, plan5_teammate_obs
from holosoma.managers.observation.terms.marl_v1 import OBJECT_CLIP_M, PARTNER_CLIP_M

V1_ACTOR_OBS_DIM, V1_TEAMMATE_OBS_DIM, V1_CRITIC_OBS_DIM = 154, 5, 585
_M = "holosoma.managers.observation.terms.marl_v1"


def actor_obs_v1(noise: bool) -> ObsGroupCfg:
    terms = dict(plan5_actor_obs.terms)
    terms["motion_ref_ori_b"] = replace(terms["motion_ref_ori_b"], func=f"{_M}:MotionRefOriBV1", params={"noise": noise})
    return replace(plan5_actor_obs, terms=terms, enable_noise=noise)


def teammate_obs_v1(noise: bool) -> ObsGroupCfg:
    return ObsGroupCfg(
        concatenate=True, enable_noise=False, history_length=1,
        terms={
            "neck_partner_rel_pos_b": ObsTermCfg(func=f"{_M}:NeckPartnerRelPosBV1", params={"noise": noise}, scale=1.0, noise=0.0,
                                                 clip=(-PARTNER_CLIP_M, PARTNER_CLIP_M)),
            "object_rel_pos_b": ObsTermCfg(func=f"{_M}:ObjectRelPosBV1", params={"noise": noise}, scale=1.0, noise=0.0,
                                           clip=(-OBJECT_CLIP_M, OBJECT_CLIP_M)),
        },
    )


critic_obs_v1 = replace(
    plan5_centralized_critic_obs,
    terms={
        **{k: v for k, v in plan5_centralized_critic_obs.terms.items() if k != "shared_motion_command"},
        "shared_motion_command": ObsTermCfg(func=f"{_M}:centralized_both_motion_commands"),
    },
)


def observation_v1(*, obs_v1: bool, critic_v1: bool, noise: bool) -> ObservationManagerCfg:
    """Observation manager config: each switch replaces only its own group(s). noise = training noise of the actor group
    (legacy uniform term noise) and of the v1 internal noise (neck / object points, heading bias)."""
    actor = actor_obs_v1(noise) if obs_v1 else replace(plan5_actor_obs, enable_noise=noise)
    teammate = teammate_obs_v1(noise) if obs_v1 else plan5_teammate_obs
    critic = critic_obs_v1 if critic_v1 else plan5_centralized_critic_obs
    return ObservationManagerCfg(groups={"actor_obs": actor, "teammate_obs": teammate, "critic_obs": critic})


__all__ = ["V1_ACTOR_OBS_DIM", "V1_CRITIC_OBS_DIM", "V1_TEAMMATE_OBS_DIM", "actor_obs_v1", "critic_obs_v1", "observation_v1",
           "teammate_obs_v1"]
