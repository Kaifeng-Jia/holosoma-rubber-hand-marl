"""CPU-only numerical contracts for the independent interaction-mesh reward."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from holosoma.config_types.reward import RewardTermCfg

# A staged implementation may be tested before the reviewed patch is installed.
# Normal repository tests always import the real package module.
if os.environ.get("INTERACTION_MESH_TEST_MODULE"):
    spec = importlib.util.spec_from_file_location("staged_interaction_mesh", os.environ["INTERACTION_MESH_TEST_MODULE"])
    interaction_mesh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interaction_mesh)
else:
    from holosoma.managers.reward.terms import interaction_mesh


def _rotation(quaternion):
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def _quaternion(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    return np.r_[axis / np.linalg.norm(axis) * np.sin(angle / 2), np.cos(angle / 2)]


def _laplacian_fixture():
    """Independent full-graph W/L construction, including object-row terms."""
    generator = np.random.default_rng(57)
    adjacency = np.zeros((104, 104), dtype=bool)
    for i in range(103):
        adjacency[i, i+1] = adjacency[i+1, i] = True
    cross = generator.random((19, 85)) < 0.1
    adjacency[:19, 19:] |= cross
    adjacency[19:, :19] |= cross.T
    laplacian = np.eye(104) - adjacency / adjacency.sum(axis=1)[:, None]
    weights = np.zeros(104)
    # 15 original groups; points 13/15/16 and 14/17/18 are the hands.
    weights[:13] = 0.5 / 15
    weights[[13, 14, 15, 16, 17, 18]] = 0.5 / 15 / 3
    active = adjacency[19:, :19].any(axis=1)
    weights[19:] = active * (0.5 / active.sum())
    q = laplacian[:, :19].T @ (weights[:, None] * laplacian[:, :19])
    return laplacian, weights, q


def _fixture(tmp_path):
    generator = np.random.default_rng(97)
    source = tmp_path / "source_motion.npz"
    np.savez(source, fixture=np.array([1, 2, 3]))
    _, _, q = _laplacian_fixture()
    points = generator.normal(size=(4, 2, 19, 3)) * 0.2
    points[:, 1, :, 0] += 0.8
    q_matrices = np.broadcast_to(q, (4, 2, 19, 19)).copy()
    q_matrices *= np.arange(1, 5)[:, None, None, None]
    offsets = np.zeros((19, 3), dtype=np.float32)
    offsets[15:] = [[0.1, 0, 0], [0, 0.12, 0], [0.1, 0, 0], [0, 0.12, 0]]
    body_names = [f"body_{i}" for i in range(19)]
    metadata = {
        "version": interaction_mesh.ARTIFACT_VERSION,
        "runtime_reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "num_frames": 4, "fps": 50.0, "num_agents": 2,
        "num_body_points": 19, "actual_object_points": 85,
    }
    artifact = tmp_path / "interaction_reference.npz"
    np.savez(
        artifact, reference_points_object=points.astype(np.float32),
        q_matrices=q_matrices.astype(np.float32), point_body_names=np.asarray(body_names),
        point_offsets=offsets, point_names=np.asarray([f"point_{i}" for i in range(19)]),
        object_points=np.zeros((85, 3), dtype=np.float32), fps=np.array(50.0),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    phases = np.array([3, 0, 2])
    object_quaternions = np.stack([
        _quaternion([1, 2, 3], 0.9), _quaternion([1, -1, 0], -0.7), _quaternion([0, 1, 1], 1.2),
    ])
    body_quaternions = np.broadcast_to(_quaternion([2, 1, 0], 0.4), (3, 2, 19, 4)).copy()
    object_positions = np.array([[10, 20, 1.2], [-6, 4, 0.8], [23, -19, 2.4]])
    body_positions = np.empty((3, 2, 19, 3))
    for n in range(3):
        body_positions[n] = points[phases[n]] @ _rotation(object_quaternions[n]).T + object_positions[n]
        for agent in range(2):
            for point in range(19):
                body_positions[n, agent, point] -= _rotation(body_quaternions[n, agent, point]) @ offsets[point]
    command = SimpleNamespace(
        reference=SimpleNamespace(num_frames=4, fps=50.0, motion_files=[str(source)]),
        time_steps=torch.tensor(phases),
        simulator_object_pos_w=torch.tensor(object_positions, dtype=torch.float32),
        simulator_object_quat_w=torch.tensor(object_quaternions, dtype=torch.float32),
    )
    env = SimpleNamespace(
        num_envs=3, num_agents=2, device="cpu", dt=0.02,
        actor_observation=torch.randn(3, 2, 158), critic_observation=torch.randn(3, 527),
        simulator=SimpleNamespace(
            _body_list=body_names,
            agent_rigid_body_pos=torch.tensor(body_positions, dtype=torch.float32),
            agent_rigid_body_rot=torch.tensor(body_quaternions, dtype=torch.float32),
        ),
    )
    cfg = RewardTermCfg(
        func="holosoma.managers.reward.terms.interaction_mesh:InteractionMeshReward",
        params={"reference_file": str(artifact), "sigma": 0.06}, weight=1.0,
    )
    return env, command, cfg, points, q_matrices


def _attach(env, command):
    env.command_manager = SimpleNamespace(get_state=lambda name: command if name == "paired_motion_command" else None)


def test_q_score_matches_full_grouped_laplacian_numpy():
    laplacian, weights, q = _laplacian_fixture()
    delta = np.random.default_rng(4).normal(size=(7, 2, 19, 3)) * 0.08
    full_delta = np.concatenate([delta, np.zeros((7, 2, 85, 3))], axis=2)
    residual = np.einsum("ij,najk->naik", laplacian, full_delta)
    expected = (np.square(residual).sum(axis=-1) * weights).sum(axis=-1)
    actual = interaction_mesh.grouped_quadratic_error(
        torch.tensor(delta), torch.zeros_like(torch.tensor(delta)), torch.tensor(q),
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-14)
    body_only = (np.square(residual[:, :, :19]).sum(axis=-1) * weights[:19]).sum(axis=-1)
    assert np.all(expected > body_only)


def test_lazy_binding_identity_phase_and_full_object_rotation(tmp_path):
    env, command, cfg, points, _ = _fixture(tmp_path)
    term = interaction_mesh.InteractionMeshReward(cfg, env)
    assert not hasattr(env, "command_manager")
    _attach(env, command)
    actual = term(env)
    torch.testing.assert_close(actual, torch.ones(3), atol=1e-6, rtol=0)
    np.testing.assert_allclose(term.points_object(env).numpy(), points[command.time_steps.numpy()], atol=3e-6)
    assert term.last_error_m2.max() < 1e-10


def test_team_exp_uses_mean_error_not_mean_individual_rewards(tmp_path):
    env, command, cfg, _, _ = _fixture(tmp_path)
    _attach(env, command)
    term = interaction_mesh.InteractionMeshReward(cfg, env)
    env.simulator.agent_rigid_body_pos[:, 0, :, 0] += 0.12
    reward = term(env)
    expected = torch.exp(-term.last_error_m2.mean(dim=1) / 0.06**2)
    torch.testing.assert_close(reward, expected)
    assert not torch.allclose(reward, torch.exp(-term.last_error_m2 / 0.06**2).mean(dim=1))
    assert term.last_error_m2[:, 0].min() > 1e-4
    assert term.last_error_m2[:, 1].max() < 1e-10
    # A raw reward is returned; dt and cfg.weight belong to RewardManager.
    assert reward.max() > (reward * env.dt).max()


def test_body_name_mapping_and_fixed_offsets_are_not_body_origins(tmp_path):
    env, command, cfg, points, _ = _fixture(tmp_path)
    permutation = np.random.default_rng(56).permutation(19)
    env.simulator._body_list = [env.simulator._body_list[i] for i in permutation]
    env.simulator.agent_rigid_body_pos = env.simulator.agent_rigid_body_pos[:, :, permutation]
    env.simulator.agent_rigid_body_rot = env.simulator.agent_rigid_body_rot[:, :, permutation]
    _attach(env, command)
    term = interaction_mesh.InteractionMeshReward(cfg, env)
    term(env)
    np.testing.assert_allclose(term.points_object(env).numpy(), points[command.time_steps.numpy()], atol=3e-6)
    points_before = term.landmarks_world(env).clone()
    hand_body = env.simulator._body_list.index("body_15")
    env.simulator.agent_rigid_body_rot[:, :, hand_body] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    points_after = term.landmarks_world(env)
    assert (points_after[:, :, 15] - points_before[:, :, 15]).norm(dim=-1).min() > 0.1
    torch.testing.assert_close(points_after[:, :, :15], points_before[:, :, :15])


def test_agent_reference_identity_is_preserved(tmp_path):
    env, command, cfg, _, _ = _fixture(tmp_path)
    _attach(env, command)
    term = interaction_mesh.InteractionMeshReward(cfg, env)
    env.simulator.agent_rigid_body_pos = env.simulator.agent_rigid_body_pos.flip(1)
    reward = term(env)
    assert reward.max() < 0.1
    assert term.last_error_m2.min() > 0.01


def test_read_only_state_and_observation_contract(tmp_path):
    env, command, cfg, _, _ = _fixture(tmp_path)
    _attach(env, command)
    tensors = [env.actor_observation, env.critic_observation, env.simulator.agent_rigid_body_pos,
               env.simulator.agent_rigid_body_rot, command.simulator_object_pos_w, command.simulator_object_quat_w,
               command.time_steps]
    before = [value.clone() for value in tensors]
    term = interaction_mesh.InteractionMeshReward(cfg, env)
    term(env)
    for actual, original in zip(tensors, before):
        torch.testing.assert_close(actual, original)
    assert env.actor_observation.shape[-1] == 158
    assert env.critic_observation.shape[-1] == 527


def test_iteration_diagnostics_cover_all_steps_and_survive_resets(tmp_path):
    env, command, cfg, _, _ = _fixture(tmp_path)
    _attach(env, command)
    term = interaction_mesh.InteractionMeshReward(cfg, env)
    first = term(env).clone()
    first_error = term.last_error_m2.clone()
    env.simulator.agent_rigid_body_pos[:, 1, :, 0] += 0.04
    second = term(env).clone()
    second_error = term.last_error_m2.clone()
    term.reset(torch.tensor([1]))
    assert term.last_raw_reward[1] == 0
    snapshot = term.get_iteration_diagnostics(reset=False)
    result = term.get_iteration_diagnostics()
    assert result["Interaction/sample_count"] == 6
    torch.testing.assert_close(result["Interaction/raw_reward_mean"], torch.cat([first, second]).mean())
    expected_error = torch.cat([first_error, second_error]).mean(dim=0)
    torch.testing.assert_close(result["Interaction/error_agent1_m2"], expected_error[1])
    torch.testing.assert_close(result["Interaction/rms_team_m"], expected_error.mean().sqrt())
    for name in snapshot:
        torch.testing.assert_close(snapshot[name], result[name])
    assert term.get_iteration_diagnostics() == {}
    term.reset()
    assert not term.last_error_m2.any()


def test_mismatched_source_or_fps_is_rejected_before_scoring(tmp_path):
    env, command, cfg, _, _ = _fixture(tmp_path)
    _attach(env, command)
    term = interaction_mesh.InteractionMeshReward(cfg, env)
    command.reference.fps = 30.0
    try:
        term(env)
    except ValueError as error:
        assert "frame count/fps" in str(error)
    else:
        raise AssertionError("mismatched reference FPS was accepted")
    command.reference.fps = 50.0
    np.savez(command.reference.motion_files[0], fixture=np.array([4, 5, 6]))
    try:
        term(env)
    except ValueError as error:
        assert "SHA256" in str(error)
    else:
        raise AssertionError("mismatched source reference was accepted")


if __name__ == "__main__":
    import tempfile

    passed = 0
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            with tempfile.TemporaryDirectory(prefix="interaction_reward_test_") as directory:
                if function.__code__.co_argcount:
                    function(Path(directory))
                else:
                    function()
            passed += 1
            print(f"PASS {name}")
    print(f"{passed} interaction reward tests passed")
