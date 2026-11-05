"""Configuration contracts for the opt-in WBT teammate interface."""

from holosoma.config_values.experiment import DEFAULTS


def test_marl_compat_preset_is_registered_and_expands_only_actor() -> None:
    config = DEFAULTS["g1_29dof_wbt_w_object_marl_compat"]
    assert config.env_class.endswith(".GhostTeammateWholeBodyTrackingManager")
    assert config.algo.config.module_dict.actor.input_dim == ["actor_obs", "teammate_obs"]
    assert config.algo.config.module_dict.critic.input_dim == ["critic_obs"]
    assert config.algo.config.load_optimizer is False
    assert list(config.observation.groups) == ["actor_obs", "teammate_obs", "critic_obs"]


def test_standard_wbt_object_preset_remains_single_actor_group() -> None:
    config = DEFAULTS["g1_29dof_wbt_w_object"]
    assert config.env_class.endswith(".WholeBodyTrackingManager")
    assert config.algo.config.module_dict.actor.input_dim == ["actor_obs"]
    assert "teammate_obs" not in config.observation.groups
