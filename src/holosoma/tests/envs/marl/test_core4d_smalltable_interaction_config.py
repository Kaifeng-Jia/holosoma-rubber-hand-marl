"""The optional relation term must not mutate the accepted baseline."""

import pytest

from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (
    g1_29dof_core4d_smalltable_baseline,
    g1_29dof_core4d_smalltable_smoke,
    with_interaction_mesh_reward,
)


@pytest.mark.parametrize("base", [g1_29dof_core4d_smalltable_baseline, g1_29dof_core4d_smalltable_smoke])
def test_interaction_variant_only_adds_one_reward(base):
    original_terms = dict(base.reward.terms)
    variant = with_interaction_mesh_reward(base, "/tmp/example_interaction.npz")
    assert base.reward.terms == original_terms
    assert "interaction_mesh" not in base.reward.terms
    assert list(variant.reward.terms) == [*original_terms, "interaction_mesh"]
    for name, term in original_terms.items():
        assert variant.reward.terms[name] is term
    for name in ("training", "algo", "robot", "simulator", "observation", "action", "termination", "randomization", "command", "curriculum"):
        assert getattr(variant, name) is getattr(base, name)
    added = variant.reward.terms["interaction_mesh"]
    assert added.weight == 1.0
    assert added.params == {"reference_file": "/tmp/example_interaction.npz", "sigma": 0.06}
    assert added.func == "holosoma.managers.reward.terms.interaction_mesh:InteractionMeshReward"


def test_interaction_variant_requires_explicit_artifact_and_cannot_be_doubled():
    base = g1_29dof_core4d_smalltable_baseline
    with pytest.raises(ValueError, match="explicit"):
        with_interaction_mesh_reward(base, " ")
    variant = with_interaction_mesh_reward(base, "/tmp/example_interaction.npz")
    with pytest.raises(ValueError, match="already enabled"):
        with_interaction_mesh_reward(variant, "/tmp/other.npz")
