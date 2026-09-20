"""CPU contracts for bucket geometry, missing-contact gate and composition."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_values.marl.g1.core4d_bucket_contract import BUCKET_VARIANTS, bucket_block_weights
from holosoma.config_values.marl.g1.core4d_smalltable_reward import g1_29dof_core4d_smalltable_reward


def _import_staged(name, variable, ordinary):
    path = os.environ.get(variable)
    if path:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return __import__(ordinary, fromlist=["*"])


bucket = _import_staged(
    "staged_bucket_reward", "CORE4D_BUCKET_TEST_MODULE", "holosoma.managers.reward.terms.core4d_bucket",
)
configs = _import_staged(
    "staged_bucket_config", "CORE4D_BUCKET_TEST_CONFIG_MODULE", "holosoma.config_values.marl.g1.core4d_bucket_reward",
)


def _fixture(tmp_path, variant="A"):
    generator = np.random.default_rng(981)
    source = tmp_path / "runtime.npz"
    np.savez(source, marker=np.array([81]))
    body = generator.normal(0, 0.2, (4, 2, 19, 3)).astype(np.float32)
    body[:, 1, :, 0] += 0.8
    points = generator.normal(0, 0.1, (7, 3)).astype(np.float32)
    object_positions = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.3, 0.1, 0.2], [0.4, 0.2, 0.3]], dtype=np.float32)
    reference_objects = points[None] + object_positions[:, None]
    offsets = np.zeros((19, 3), dtype=np.float32)
    offsets[15:] = np.array([[0.03, 0, 0], [0, 0.04, 0], [0.05, 0, 0], [0, 0.06, 0]])
    names = [f"body{i}" for i in range(19)]
    prior = np.ones(19, dtype=np.float32)
    prior[13:] = 1 / 3
    arrays = {
        "metadata_json": np.asarray(json.dumps({
            "version": bucket.ARTIFACT_VERSION,
            "runtime_reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "num_frames": 4, "num_agents": 2, "num_body_points": 19, "actual_object_points": 7, "fps": 50,
        })),
        "reference_body_points_world": body, "reference_object_points_world": reference_objects,
        "object_points": points, "point_offsets": offsets, "body_point_weights": prior,
        "reference_contact_weights": np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32),
        "point_body_names": np.array(names), "point_names": np.array([f"point{i}" for i in range(19)]),
        "fps": np.asarray(50.0),
    }
    artifact = tmp_path / "bucket.npz"
    np.savez(artifact, **arrays)
    phases = torch.tensor([0, 1, 2, 3])
    origins = torch.tensor([[100, -80, 0], [10, 30, 0], [-40, -20, 0], [3, -4, 0]], dtype=torch.float32)
    actual_body = torch.from_numpy(body.copy()) + origins[:, None, None]
    actual_body -= torch.from_numpy(offsets)[None, None]
    identity = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
    world_positions = torch.from_numpy(object_positions) + origins
    forces = torch.zeros(4, 1, 4, 3)
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(filter_prim_paths_expr=list(bucket.HAND_FILTER_PATHS)),
        data=SimpleNamespace(force_matrix_w=forces),
    )
    command = SimpleNamespace(
        reference=SimpleNamespace(num_frames=4, fps=50.0, motion_files=[str(source)]),
        time_steps=phases,
        object_pos_w=world_positions.clone(), simulator_object_pos_w=world_positions.clone(),
        object_quat_w=identity.expand(4, -1).clone(), simulator_object_quat_w=identity.expand(4, -1).clone(),
    )
    env = SimpleNamespace(
        num_envs=4, num_agents=2, device="cpu", dt=0.02,
        episode_length_buf=torch.ones(4, dtype=torch.long),
        simulator=SimpleNamespace(
            _body_list=names, agent_rigid_body_pos=actual_body,
            agent_rigid_body_rot=identity.expand(4, 2, 19, -1).clone(),
            object_hand_contact_sensor=sensor,
        ),
        actor_observation=torch.zeros(4, 2, 158), critic_observation=torch.zeros(4, 527),
    )
    cfg = RewardTermCfg(
        func="holosoma.managers.reward.terms.core4d_bucket:BucketInteractionReward",
        params={"variant": variant, "reference_file": str(artifact)}, weight=1,
    )
    return env, command, cfg, arrays


def _attach(env, command):
    env.command_manager = SimpleNamespace(get_state=lambda name: command if name == "paired_motion_command" else None)


def _rewrite(cfg, arrays, **changes):
    np.savez(cfg.params["reference_file"], **dict(arrays, **changes))


def test_lazy_bind_identity_origin_invariance_and_raw_not_dt(tmp_path):
    env, command, cfg, _ = _fixture(tmp_path)
    term = bucket.BucketInteractionReward(cfg, env)
    assert not hasattr(env, "command_manager")
    _attach(env, command)
    result = term(env)
    torch.testing.assert_close(result, torch.full((4,), 5.0), atol=1e-6, rtol=0)
    assert term.last_error_m2.max() < 1e-9
    assert term.last_components["contact_error"].tolist() == [1, 1, 1, 0]
    torch.testing.assert_close(term.last_gate, torch.ones(4))


def test_gate_literal_weighted_missing_contact_release_and_zero_alpha():
    alpha = torch.tensor([[1., 0.], [0., 1.], [1/3, 1.], [0., 0.]])
    contacts = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 0]], dtype=torch.bool)
    gate, error = bucket.contact_gate(alpha, contacts, "B")
    expected = torch.tensor([0., 1., .75, 0.])
    torch.testing.assert_close(error, expected)
    torch.testing.assert_close(gate, .5 + .5 * torch.exp(-2 * expected))
    assert gate[0] == gate[-1] == 1
    # A released agent has zero target weight; continuing contact is not penalized.
    extra, _ = bucket.contact_gate(alpha, torch.ones_like(contacts), "B")
    torch.testing.assert_close(extra, torch.ones(4))
    with pytest.raises(ValueError, match="variant"):
        bucket.contact_gate(alpha, contacts, "C")


def test_weak_confidence_fades_continuously_and_full_pair_is_mean():
    alpha = torch.tensor([[.01, 0.], [.5, 0.], [1., 0.], [1., 1.], [1., 1.], [0., 0.]])
    contacts = torch.tensor([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]], dtype=torch.bool)
    gate, error = bucket.contact_gate(alpha, contacts, "B")
    expected = torch.tensor([.01, .5, 1., .5, 1., 0.])
    torch.testing.assert_close(error, expected)
    torch.testing.assert_close(gate, .5 + .5 * torch.exp(-2 * expected))
    assert gate[0] > gate[1] > gate[2]
    assert gate[-1] == 1


def test_filtered_hand_any_threshold_and_no_vector_cancellation(tmp_path):
    env, command, cfg, _ = _fixture(tmp_path, "B")
    _attach(env, command)
    forces = env.simulator.object_hand_contact_sensor.data.force_matrix_w
    forces[0, 0, 0, 0] = 1.0  # Strictly greater than 1 N, not >=.
    forces[1, 0, 2, 0] = 1.01
    forces[2, 0, 0, 0] = 2
    forces[2, 0, 1, 0] = -2  # Must not sum the two vectors before taking norms.
    forces[2, 0, 3, 2] = 3
    term = bucket.BucketInteractionReward(cfg, env)
    reward = term(env)
    assert term.last_contacts.tolist() == [[False, False], [False, True], [True, True], [False, False]]
    expected = torch.tensor([.5 + .5 * math.exp(-2), 1, 1, 1])
    torch.testing.assert_close(reward, 5 * expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("variant", BUCKET_VARIANTS)
def test_reset_without_physics_does_not_count_stale_contact(tmp_path, variant):
    env, command, cfg, _ = _fixture(tmp_path, variant)
    _attach(env, command)
    env.simulator.object_hand_contact_sensor.data.force_matrix_w.fill_(10)
    env.episode_length_buf[0] = 0
    term = bucket.BucketInteractionReward(cfg, env)
    term(env)
    assert not term.last_contacts[0].any()
    assert term.last_contacts[1:].all()
    env.episode_length_buf[0] = 1
    term(env)
    assert term.last_contacts[0].all()


def test_position_height_rotation_components_match_literal_formula(tmp_path):
    env, command, cfg, _ = _fixture(tmp_path)
    _attach(env, command)
    delta = torch.tensor([.06, -.08, .10])
    command.simulator_object_pos_w += delta
    angle = 0.2
    command.simulator_object_quat_w[:] = torch.tensor([0, 0, math.sin(angle / 2), math.cos(angle / 2)])
    term = bucket.BucketInteractionReward(cfg, env)
    reward = term(env)
    values = term.last_components
    torch.testing.assert_close(values["position"], torch.full((4,), math.exp(-(.06**2 + .08**2 + 2*.10**2)/.3**2)), atol=1e-5, rtol=0)
    torch.testing.assert_close(values["height"], torch.full((4,), math.exp(-1)), atol=1e-6, rtol=0)
    torch.testing.assert_close(values["orientation"], torch.full((4,), math.exp(-angle**2/.4**2)), atol=1e-6, rtol=0)
    torch.testing.assert_close(values["relative"], torch.exp(-term.last_error_m2.mean(dim=1)/.04**2))
    torch.testing.assert_close(reward, values["position"] + values["orientation"] + values["height"] + 2*values["relative"])


def test_agent_order_body_mapping_offsets_and_current_frame(tmp_path):
    env, command, cfg, arrays = _fixture(tmp_path)
    permutation = torch.arange(18, -1, -1)
    env.simulator._body_list = list(reversed(env.simulator._body_list))
    env.simulator.agent_rigid_body_pos = env.simulator.agent_rigid_body_pos[:, :, permutation]
    env.simulator.agent_rigid_body_rot = env.simulator.agent_rigid_body_rot[:, :, permutation]
    _attach(env, command)
    term = bucket.BucketInteractionReward(cfg, env)
    torch.testing.assert_close(term(env), torch.full((4,), 5.), atol=1e-6, rtol=0)
    # The physical agent identity is not exchangeable, despite shared weights.
    env.simulator.agent_rigid_body_pos = env.simulator.agent_rigid_body_pos.flip(1)
    term(env)
    assert term.last_error_m2.min() > .05


def test_team_relation_exponential_is_after_agent_error_mean(tmp_path):
    env, command, cfg, _ = _fixture(tmp_path)
    _attach(env, command)
    env.simulator.agent_rigid_body_pos[:, 0, :, 0] += .05
    term = bucket.BucketInteractionReward(cfg, env)
    term(env)
    torch.testing.assert_close(term.last_error_m2[:, 0], torch.full((4,), .05**2), atol=5e-7, rtol=0)
    torch.testing.assert_close(term.last_error_m2[:, 1], torch.zeros(4), atol=1e-9, rtol=0)
    actual = term.last_components["relative"]
    torch.testing.assert_close(actual, torch.exp(-term.last_error_m2.mean(dim=1)/.04**2))
    assert not torch.allclose(actual, torch.exp(-term.last_error_m2/.04**2).mean(dim=1))


@pytest.mark.parametrize("variant", BUCKET_VARIANTS)
def test_reward_does_not_mutate_input_state_or_observations(tmp_path, variant):
    env, command, cfg, _ = _fixture(tmp_path, variant)
    _attach(env, command)
    tensors = [env.actor_observation, env.critic_observation, command.time_steps,
               command.object_pos_w, command.simulator_object_pos_w, command.simulator_object_quat_w,
               env.simulator.agent_rigid_body_pos, env.simulator.agent_rigid_body_rot,
               env.simulator.object_hand_contact_sensor.data.force_matrix_w]
    before = [tensor.clone() for tensor in tensors]
    bucket.BucketInteractionReward(cfg, env)(env)
    for tensor, original in zip(tensors, before):
        torch.testing.assert_close(tensor, original)


@pytest.mark.parametrize("variant", BUCKET_VARIANTS)
def test_iteration_sums_survive_partial_and_full_episode_resets(tmp_path, variant):
    env, command, cfg, _ = _fixture(tmp_path, variant)
    _attach(env, command)
    term = bucket.BucketInteractionReward(cfg, env)
    first = term(env).clone()
    env.simulator.object_hand_contact_sensor.data.force_matrix_w.fill_(2)
    second = term(env).clone()
    term.reset(torch.tensor([0, 2]))
    assert term.last_raw_reward[0] == term.last_raw_reward[2] == 0
    assert term.last_raw_reward[1] > 0
    report = term.get_iteration_diagnostics(reset=False)
    assert report["Bucket/sample_count"] == 8
    torch.testing.assert_close(report["Bucket/raw_reward_mean"], torch.cat((first, second)).mean())
    term.reset(None)
    drained = term.get_iteration_diagnostics()
    torch.testing.assert_close(drained["Bucket/raw_reward_mean"], report["Bucket/raw_reward_mean"])
    assert term.get_iteration_diagnostics() == {}
    assert not term.last_raw_reward.any()


@pytest.mark.parametrize("key, replacement", [
    ("reference_body_points_world", np.zeros((4, 2, 18, 3))),
    ("reference_object_points_world", np.zeros((4, 6, 3))),
    ("object_points", np.zeros((0, 3))),
    ("body_point_weights", -np.ones(19)),
    ("reference_contact_weights", -np.ones((4, 2))),
    ("reference_contact_weights", np.full((4, 2), 1.01)),
    ("point_offsets", np.full((19, 3), np.nan)),
    ("point_names", np.array(["duplicate"] * 19)),
    ("fps", np.asarray(0.0)),
])
def test_artifact_validation(tmp_path, key, replacement):
    env, _, cfg, arrays = _fixture(tmp_path)
    _rewrite(cfg, arrays, **{key: replacement})
    with pytest.raises(ValueError):
        bucket.BucketInteractionReward(cfg, env)


@pytest.mark.parametrize("mutation", ["source_hash", "fps", "filter_order", "force_shape", "missing_sensor", "missing_body"])
def test_binding_validation(tmp_path, mutation):
    env, command, cfg, _ = _fixture(tmp_path)
    _attach(env, command)
    term = bucket.BucketInteractionReward(cfg, env)
    if mutation == "source_hash":
        np.savez(command.reference.motion_files[0], altered=np.array([9]))
    elif mutation == "fps":
        command.reference.fps = 30
    elif mutation == "filter_order":
        env.simulator.object_hand_contact_sensor.cfg.filter_prim_paths_expr.reverse()
    elif mutation == "force_shape":
        env.simulator.object_hand_contact_sensor.data.force_matrix_w = torch.zeros(4, 4, 1, 3)
    elif mutation == "missing_sensor":
        env.simulator.object_hand_contact_sensor = None
    else:
        env.simulator._body_list[0] = "missing"
    with pytest.raises(ValueError):
        term(env)


def test_artifact_hash_pin_and_change_before_bind(tmp_path):
    env, command, cfg, arrays = _fixture(tmp_path)
    _attach(env, command)
    bad = replace(cfg, params=dict(cfg.params, reference_sha256="0" * 64))
    with pytest.raises(ValueError, match="SHA256"):
        bucket.BucketInteractionReward(bad, env)
    term = bucket.BucketInteractionReward(cfg, env)
    arrays["point_offsets"][0, 0] = .03
    _rewrite(cfg, arrays)
    with pytest.raises(ValueError, match="changed"):
        term(env)


def test_config_replaces_object_block_only_and_a_b_only_variant_differs():
    base = g1_29dof_core4d_smalltable_reward
    first = configs.with_bucket_interaction_reward(base, "fixture.npz", "A")
    second = configs.with_bucket_interaction_reward(base, "fixture.npz", "B")
    assert len(first.terms) == 10
    removed = {"object_global_ref_position_error_exp", "object_global_ref_orientation_error_exp"}
    assert removed <= base.terms.keys()
    assert not removed & first.terms.keys()
    assert set(first.terms) == (set(base.terms) - removed) | {"bucket_interaction"}
    for name in set(base.terms) - removed:
        assert first.terms[name] == second.terms[name] == base.terms[name]
    a = first.terms["bucket_interaction"]
    b = second.terms["bucket_interaction"]
    assert a.weight == b.weight == 1
    assert dict(a.params, variant="B") == b.params
    assert not first.only_positive_rewards
    with pytest.raises(ValueError):
        configs.with_bucket_interaction_reward(first, "fixture.npz", "A")
    with pytest.raises(ValueError):
        configs.with_bucket_interaction_reward(base, "fixture.npz", "C")


@pytest.mark.parametrize("variant", BUCKET_VARIANTS)
def test_full_config_wrapper_same_sensors_without_mutating_legacy(variant):
    from holosoma.config_values.marl.g1.core4d_smalltable_experiment import g1_29dof_core4d_smalltable_baseline
    base = g1_29dof_core4d_smalltable_baseline
    a = configs.with_bucket_reward(base, "fixture.npz", "A")
    b = configs.with_bucket_reward(base, "fixture.npz", variant)
    assert a.env_class == b.env_class == "holosoma.envs.marl.core4d_bucket_manager.Core4DBucketManager"
    assert a.simulator.config.enable_object_hand_contact
    assert not a.simulator.config.enable_object_contact_diagnostics
    assert a.simulator == b.simulator
    assert not base.simulator.config.enable_object_contact_diagnostics
    assert a.observation == b.observation == base.observation
    assert a.training == b.training == base.training
    for name in set(a.reward.terms) - {"bucket_interaction"}:
        assert a.reward.terms[name] == b.reward.terms[name] == base.reward.terms[name]
    assert b.reward.terms["bucket_interaction"].params == {"reference_file": "fixture.npz", "variant": variant}
    assert b.reward.terms["bucket_interaction"].weight == 1.0


@pytest.mark.parametrize("variant, expected", [
    ("A", (1., 1., 1., 2.)), ("B", (1., 1., 1., 2.)),
    ("A_no_rel", (1., 1., 1., 0.)), ("A_no_height", (1., 1., 0., 2.)),
    ("A_no_rel_no_height", (1., 1., 0., 0.)),
])
def test_fixed_variant_weights_and_only_b_has_contact_gate(variant, expected):
    assert bucket_block_weights(variant) == expected
    alpha = torch.tensor([[1., 1.], [.01, 0.], [0., 0.]])
    gate, error = bucket.contact_gate(alpha, torch.zeros_like(alpha, dtype=torch.bool), variant)
    torch.testing.assert_close(error, torch.tensor([1., .01, 0.]))
    expected_gate = .5 + .5 * torch.exp(-2 * error) if variant == "B" else torch.ones(3)
    assert torch.equal(gate, expected_gate)


@pytest.mark.parametrize("variant", ["A", "B"])
def test_legacy_a_b_arithmetic_is_bitwise_unchanged(tmp_path, variant):
    env, command, cfg, _ = _fixture(tmp_path, variant)
    _attach(env, command)
    command.simulator_object_pos_w += torch.tensor([.006, -.008, .01])
    env.simulator.agent_rigid_body_pos[:, 0, :, 1] += .015
    term = bucket.BucketInteractionReward(cfg, env)
    actual = term(env)
    values = term.last_components
    # The literal expression is the pre-ablation A/B implementation.
    old_ungated = values["position"] + values["orientation"] + values["height"] + 2.0 * values["relative"]
    old_gate = torch.ones(4) if variant == "A" else .5 + .5 * torch.exp(-2 * values["contact_error"])
    assert torch.equal(values["raw_ungated"], old_ungated)
    assert torch.equal(actual, old_ungated * old_gate)


@pytest.mark.parametrize("variant, removed_component, removed_weight", [
    ("A_no_rel", "relative", 2.), ("A_no_height", "height", 1.),
])
def test_ablations_remove_only_target_contribution_and_keep_diagnostics(
    tmp_path, variant, removed_component, removed_weight,
):
    env, command, cfg, _ = _fixture(tmp_path, variant)
    _attach(env, command)
    command.simulator_object_pos_w += torch.tensor([.006, -.008, .01])
    env.simulator.agent_rigid_body_pos[:, 0, :, 1] += .015
    baseline = bucket.BucketInteractionReward(replace(cfg, params=dict(cfg.params, variant="A")), env)
    ablation = bucket.BucketInteractionReward(cfg, env)
    baseline_reward = baseline(env).clone()
    ablation_reward = ablation(env).clone()
    torch.testing.assert_close(
        baseline_reward - ablation_reward,
        removed_weight * baseline.last_components[removed_component], atol=5e-7, rtol=0,
    )
    for name in set(baseline.last_components) - {"raw_ungated", "raw_reward"}:
        assert torch.equal(ablation.last_components[name], baseline.last_components[name])
    assert torch.equal(ablation.last_gate, torch.ones(4))
    assert ablation.last_components["height"].min() > 0
    assert ablation.last_components["relative"].min() > 0
    assert ablation.last_error_m2.min() > 0
    diagnostics = ablation.get_iteration_diagnostics()
    for name in ("height", "relative", "height_abs_error_m", "vector_error_agent0_m2"):
        assert diagnostics[f"Bucket/{name}_mean"] > 0
    # Sensing remains live, but contact cannot change either ablation's reward.
    env.simulator.object_hand_contact_sensor.data.force_matrix_w.fill_(2)
    assert torch.equal(ablation(env), ablation_reward)
    assert ablation.last_contacts.all()
    assert not ablation.last_components["contact_error"].any()


def test_remove_both_keeps_only_position_rotation_and_all_diagnostics(tmp_path):
    env, command, cfg, _ = _fixture(tmp_path, "A_no_rel_no_height")
    _attach(env, command)
    command.simulator_object_pos_w += torch.tensor([.006, -.008, .01])
    env.simulator.agent_rigid_body_pos[:, 0, :, 1] += .015
    baseline = bucket.BucketInteractionReward(replace(cfg, params=dict(cfg.params, variant="A")), env)
    ablation = bucket.BucketInteractionReward(cfg, env)
    original = baseline(env).clone()
    actual = ablation(env).clone()
    values = baseline.last_components
    torch.testing.assert_close(original - actual, values["height"] + 2 * values["relative"], atol=5e-7, rtol=0)
    torch.testing.assert_close(actual, values["position"] + values["orientation"])
    for name in set(values) - {"raw_ungated", "raw_reward"}:
        assert torch.equal(ablation.last_components[name], values[name])
    env.simulator.object_hand_contact_sensor.data.force_matrix_w.fill_(2)
    assert torch.equal(ablation(env), actual)
    assert torch.equal(ablation.last_gate, torch.ones(4))


@pytest.mark.parametrize("variant, constant", [("A_no_height", 3.), ("A_no_rel_no_height", 1.)])
def test_no_height_still_tracks_world_z_with_weight_two(tmp_path, variant, constant):
    env, command, cfg, _ = _fixture(tmp_path, variant)
    _attach(env, command)
    term = bucket.BucketInteractionReward(cfg, env)
    initial = term(env).clone()
    # Translate actual body and object together: relation stays matched, z does not.
    command.simulator_object_pos_w[:, 2] += .10
    env.simulator.agent_rigid_body_pos[..., 2] += .10
    moved = term(env)
    torch.testing.assert_close(initial, torch.full((4,), constant + 1), atol=1e-6, rtol=0)
    torch.testing.assert_close(moved, torch.full((4,), constant + math.exp(-2 * .10**2 / .3**2)), atol=1e-6, rtol=0)
    assert (moved < initial).all()


def test_no_rel_still_computes_changed_vector_diagnostics(tmp_path):
    env, command, cfg, _ = _fixture(tmp_path, "A_no_rel")
    _attach(env, command)
    term = bucket.BucketInteractionReward(cfg, env)
    initial = term(env).clone()
    env.simulator.agent_rigid_body_pos[..., 0] += .05
    assert torch.equal(term(env), initial)
    assert term.last_error_m2.min() > .002
    assert term.last_components["relative"].max() < .3


@pytest.mark.parametrize("variant", BUCKET_VARIANTS)
def test_reward_manager_applies_dt_once_for_each_variant(tmp_path, variant):
    from holosoma.managers.reward.manager import RewardManager
    env, command, cfg, _ = _fixture(tmp_path, variant)
    _attach(env, command)
    env.simulator.object_hand_contact_sensor.data.force_matrix_w.fill_(2)
    manager = RewardManager(RewardManagerCfg(terms={"bucket_interaction": cfg}), env, "cpu")
    maximum = sum(bucket_block_weights(variant))
    torch.testing.assert_close(manager.compute(dt=.02), torch.full((4,), maximum * .02), atol=1e-7, rtol=0)
    torch.testing.assert_close(manager.compute(dt=.10), torch.full((4,), maximum * .10), atol=1e-7, rtol=0)
