"""Opt-in bucket A/B and ablation rewards; legacy presets remain untouched."""

from __future__ import annotations

import copy
from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_values.marl.g1.core4d_bucket_contract import bucket_block_weights
from holosoma.config_values.marl.g1.core4d_smalltable_reward import g1_29dof_core4d_smalltable_reward


def with_bucket_interaction_reward(
    base_reward: RewardManagerCfg, reference_file: str, variant: str,
    *, reference_sha256: str | None = None,
) -> RewardManagerCfg:
    """Replace only the two object tracking terms with one gated bucket block.

    Six robot-tracking terms and the existing three regularizers are unchanged.
    The new positive height reward replaces neither a hidden legacy penalty nor
    a second interaction term: these incompatible additions are rejected.
    """
    bucket_block_weights(variant)  # Validate against the shared fixed-variant contract.
    if not isinstance(reference_file, str) or not reference_file.strip():
        raise ValueError("Bucket reward requires an explicit reference_file")
    incompatible = {"interaction_mesh", "object_height_error_penalty", "bucket_interaction"} & base_reward.terms.keys()
    if incompatible:
        raise ValueError(f"Bucket base contains incompatible reward terms: {sorted(incompatible)}")
    replaced = {"object_global_ref_position_error_exp", "object_global_ref_orientation_error_exp"}
    if not replaced.issubset(base_reward.terms):
        raise ValueError("Bucket base must contain both original object tracking terms")
    params = {"reference_file": reference_file, "variant": variant}
    if reference_sha256 is not None:
        if not isinstance(reference_sha256, str) or len(reference_sha256) != 64 or any(c not in "0123456789abcdef" for c in reference_sha256):
            raise ValueError("Bucket reference_sha256 must be a lowercase SHA256 digest")
        params["reference_sha256"] = reference_sha256
    terms = {name: copy.deepcopy(term) for name, term in base_reward.terms.items() if name not in replaced}
    terms["bucket_interaction"] = RewardTermCfg(
        func="holosoma.managers.reward.terms.core4d_bucket:BucketInteractionReward",
        params=params, weight=1.0,
    )
    return replace(base_reward, terms=terms)


def make_core4d_bucket_reward(
    reference_file: str, variant: str, *, reference_sha256: str | None = None,
) -> RewardManagerCfg:
    return with_bucket_interaction_reward(
        g1_29dof_core4d_smalltable_reward, reference_file, variant,
        reference_sha256=reference_sha256,
    )


def with_bucket_reward(config, reference_file: str, variant: str):
    """Select the bucket environment and identical sensing for every variant."""
    return replace(
        config,
        env_class="holosoma.envs.marl.core4d_bucket_manager.Core4DBucketManager",
        reward=with_bucket_interaction_reward(config.reward, reference_file, variant),
        simulator=replace(
            config.simulator,
            config=replace(config.simulator.config, enable_object_hand_contact=True),
        ),
    )


__all__ = ["make_core4d_bucket_reward", "with_bucket_interaction_reward", "with_bucket_reward"]
