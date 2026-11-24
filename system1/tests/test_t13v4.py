"""T13 v4 section 2 unit checks on a small fake environment (exact; no simulator):
  1. observation v1 terms vs an independent numpy computation of the definitions (noise off: <= 1e-5; noise on: the obs equal
     the definition with the state's offsets added, offsets persist between steps and are redrawn only on reset, ranges);
  2. contact items vs the definitions, including the three synthetic cases: hand in the air, hand touching without force,
     leg pushing the object (plus the 25-step lost-contact counter).
    hssim_s1/bin/python -B system1/tests/test_t13v4.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/holosoma"))
from holosoma.managers.observation.terms import marl_v1 as ov  # noqa: E402
from holosoma.managers.reward.terms import contact_v1 as cv  # noqa: E402
from holosoma.config_types.observation import ObsTermCfg  # noqa: E402

BODIES = ["pelvis", "left_knee_link", "torso_link", "left_rubber_hand_link", "right_rubber_hand_link"]
LINKS = ["pelvis", "left_knee_link", "torso_link"]
rng = np.random.default_rng(0)


def rand_quat(n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)  # xyzw


def qmat(q):  # xyzw -> R
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


class FakeEnv:
    def __init__(self, E=64):
        self.num_envs, self.num_agents, self.num_dof, self.device = E, 2, 29, "cpu"
        B = len(BODIES)
        self.simulator = SimpleNamespace(
            _body_list=BODIES,
            agent_rigid_body_pos=torch.tensor(rng.normal(size=(E, 2, B, 3)), dtype=torch.float),
            agent_rigid_body_rot=torch.tensor(rand_quat(E * 2 * B).reshape(E, 2, B, 4), dtype=torch.float))
        filters = [f"/World/envs/env_.*/{r}/{b}" for r in ("Robot", "Robot_1") for b in BODIES]
        self.simulator.object_robot_contact_sensor = SimpleNamespace(cfg=SimpleNamespace(filter_prim_paths_expr=filters),
                                                                     data=SimpleNamespace(force_matrix_w=torch.zeros(E, 1, len(filters), 3)))
        self.command = SimpleNamespace(
            ref_body_index=BODIES.index("torso_link"), time_steps=torch.zeros(E, dtype=torch.long),
            agent_ref_pos_w=torch.tensor(rng.normal(size=(E, 2, 3)), dtype=torch.float),
            agent_ref_quat_w=torch.tensor(rand_quat(E * 2).reshape(E, 2, 4), dtype=torch.float),
            simulator_object_pos_w=torch.tensor(rng.normal(size=(E, 3)), dtype=torch.float),
            simulator_object_quat_w=torch.tensor(rand_quat(E), dtype=torch.float),
            command=torch.tensor(rng.normal(size=(E, 2, 58)), dtype=torch.float))
        self.command_manager = SimpleNamespace(get_state=lambda name: self.command)
        self.episode_length_buf = torch.ones(E, dtype=torch.long)


def np_v1(env, bias):
    """independent numpy: (ref_ori 6, partner 2, object 3) for given per-robot heading bias [E, 2] and noisy points."""
    sim, c = env.simulator, env.command
    ti = BODIES.index("torso_link")
    tp, tq = sim.agent_rigid_body_pos[:, :, ti].double().numpy(), sim.agent_rigid_body_rot[:, :, ti].double().numpy()
    Rt = qmat(tq)
    st = env._system1_obs_v1
    neck = tp + np.einsum("eaij,j->eai", Rt, np.array(ov.NECK_POINT_TORSO_M))
    if st.noise_enabled:
        neck = neck + (st.neck_offset + st.neck_step).double().numpy()
    obj = np.repeat(c.simulator_object_pos_w.double().numpy()[:, None], 2, 1)
    if st.noise_enabled:
        obj = obj + (st.object_offset + st.object_step).double().numpy()
    yaw = np.arctan2(Rt[..., 1, 0], Rt[..., 0, 0]) + bias
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ryaw = np.zeros(yaw.shape + (3, 3))
    Ryaw[..., 0, 0], Ryaw[..., 0, 1], Ryaw[..., 1, 0], Ryaw[..., 1, 1], Ryaw[..., 2, 2] = cy, -sy, sy, cy, 1.0
    partner = np.einsum("eaji,eaj->eai", Ryaw, neck[:, [1, 0]] - neck)[..., :2]
    objrel = np.einsum("eaji,eaj->eai", Ryaw, obj - neck)
    # reference orientation relative to the biased actual torso orientation
    Rb = np.zeros_like(Ryaw)
    Rb[..., 0, 0], Rb[..., 0, 1], Rb[..., 1, 0], Rb[..., 1, 1], Rb[..., 2, 2] = np.cos(bias), -np.sin(bias), np.sin(bias), np.cos(bias), 1.0
    Ract = np.einsum("eaij,eajk->eaik", Rb, Rt)
    Rref = qmat(c.agent_ref_quat_w.double().numpy())
    rel = np.einsum("eaji,eajk->eaik", Ract, Rref)
    ori = rel[..., :2].reshape(env.num_envs, 2, 6)
    return ori, partner, objrel


def obs_terms(env, noise):
    if hasattr(env, "_system1_obs_v1"):
        del env._system1_obs_v1
    cfgs = [ObsTermCfg(func="x", params={"noise": noise}) for _ in range(3)]
    return ov.MotionRefOriBV1(cfgs[0], env), ov.NeckPartnerRelPosBV1(cfgs[1], env), ov.ObjectRelPosBV1(cfgs[2], env)


def test_obs():
    env = FakeEnv()
    t_ori, t_par, t_obj = obs_terms(env, noise=False)
    o, p, b = t_ori(env), t_par(env), t_obj(env)
    eo, ep, eb = np_v1(env, np.zeros((env.num_envs, 2)))
    err_off = max(np.abs(o.double().numpy() - eo).max(), np.abs(p.double().numpy() - ep).max(), np.abs(b.double().numpy() - eb).max())
    assert err_off <= 1e-5, err_off
    # legacy ref-orientation term with the same inputs (noise off, bias 0) must agree
    from holosoma.managers.observation.terms.marl import paired_motion_ref_ori_b
    legacy = (paired_motion_ref_ori_b(env) - o).abs().max().item()
    assert legacy <= 1e-6, legacy
    crit = ov.centralized_both_motion_commands(env)
    assert crit.shape == (env.num_envs, 116) and torch.equal(crit[:, :58], env.command.command[:, 0]) and torch.equal(crit[:, 58:], env.command.command[:, 1])
    # noise on
    env = FakeEnv(E=4096)
    t_ori, t_par, t_obj = obs_terms(env, noise=True)
    o, p, b = t_ori(env), t_par(env), t_obj(env)
    st = env._system1_obs_v1
    eo, ep, eb = np_v1(env, st.heading_bias.double().numpy())
    err_on = max(np.abs(o.double().numpy() - eo).max(), np.abs(p.double().numpy() - ep).max(), np.abs(b.double().numpy() - eb).max())
    assert err_on <= 1e-5, err_on
    stats = {}
    for name, t, half in (("neck_offset", st.neck_offset, 0.03), ("neck_step", st.neck_step, 0.01), ("object_offset", st.object_offset, 0.02),
                          ("object_step", st.object_step, 0.01), ("heading_bias", st.heading_bias, math.radians(5.0))):
        x = t.double().numpy().ravel()
        stats[name] = dict(min=float(x.min()), max=float(x.max()), mean=float(x.mean()), std=float(x.std()), uniform_std=half / math.sqrt(3))
        assert x.min() >= -half and x.max() <= half and abs(x.std() - half / math.sqrt(3)) < 0.03 * half and abs(x.mean()) < 0.03 * half, (name, stats[name])
    off0, bias0, step0 = st.neck_offset.clone(), st.heading_bias.clone(), st.neck_step.clone()
    t_ori(env)  # next step
    assert torch.equal(off0, st.neck_offset) and torch.equal(bias0, st.heading_bias) and not torch.equal(step0, st.neck_step)
    ids = torch.tensor([0, 5])
    t_ori.reset(ids)
    t_par.reset(ids)
    changed = (st.neck_offset != off0).any(-1).any(-1)
    assert changed[ids].all() and not changed[torch.tensor([1, 2, 3])].any()
    # robots independent
    corr = np.corrcoef(st.heading_bias[:, 0].numpy(), st.heading_bias[:, 1].numpy())[0, 1]
    assert abs(corr) < 0.1, corr
    return dict(noise_off_max_abs_err=float(err_off), legacy_ref_ori_max_abs_diff=float(legacy), noise_on_max_abs_err=float(err_on),
                noise_samples=stats, heading_bias_robot_correlation=float(corr))


# ------------------------------------------------------------------------------------------------- contact
def make_labels(path, F=40):
    should = np.zeros((F, 2, 2), bool)
    should[10:30] = True
    point = rng.normal(scale=0.2, size=(F, 2, 2, 3)).astype(np.float32)
    far = np.zeros((F, 2, len(LINKS)), bool)
    far[:, :, LINKS.index("left_knee_link")] = True  # the person's shin is far from the object in every frame
    palm = np.array([[0.1, -0.03, 0.0], [0.1, 0.03, 0.0]], np.float32)
    np.savez(path, should_contact=should, contact_point=point, human_far=far, link_names=np.array(LINKS), palm_offset=palm)
    return should, point, far, palm


def np_contact(env, labels, force_norms):
    should, point, far, palm = labels
    c, sim = env.command, env.simulator
    f = c.time_steps.numpy().clip(0, len(should) - 1)
    Ro = qmat(c.simulator_object_quat_w.double().numpy())
    cp = c.simulator_object_pos_w.double().numpy()[:, None, None] + np.einsum("eij,erhj->erhi", Ro, point[f].astype(np.float64))
    hi = [BODIES.index(h) for h in cv.HANDS]
    hp = sim.agent_rigid_body_pos[:, :, hi].double().numpy()
    Rh = qmat(sim.agent_rigid_body_rot[:, :, hi].double().numpy())
    pw = hp + np.einsum("erhij,hj->erhi", Rh, palm.astype(np.float64))
    d = np.linalg.norm(pw - cp, axis=-1)
    nb = len(BODIES)
    fh = np.stack([force_norms[:, [r * nb + BODIES.index(h) for h in cv.HANDS]] for r in (0, 1)], 1)
    fl = np.stack([force_norms[:, [r * nb + BODIES.index(l) for l in LINKS]] for r in (0, 1)], 1)
    k = cv.CONTACT_V1
    r_h = np.where(should[f], k["gain"] * np.exp(-d / k["sigma_m"]) * np.exp(np.minimum(fh - k["force_ref_n"], 0) / k["force_scale_n"]), 1.0)
    count = ((fl > k["undesired_force_n"]) & far[f]).sum(-1)
    robot = r_h.mean(-1) * np.exp(-k["undesired_k"] * count)
    lost = should[f] & ((d > k["lost_distance_m"]) | (fh < k["lost_force_n"]))
    return dict(robot=robot, team=robot.mean(-1), lost=lost, d=d, count=count)


def set_forces(env, norms):
    E, NF = norms.shape
    dirs = rng.normal(size=(E, NF, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    env.simulator.object_robot_contact_sensor.data.force_matrix_w = torch.tensor((dirs * norms[..., None])[:, None], dtype=torch.float)


def test_contact():
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.npz"
        labels = make_labels(path)
        env = FakeEnv(E=32)
        rew = cv.EefContactExpV1(SimpleNamespace(), env)
        term = cv.LostContactV1(SimpleNamespace(), env)
        nb, E = len(BODIES), env.num_envs
        # random states / forces at a contact frame: general agreement
        env.command.time_steps[:] = 15
        norms = rng.uniform(0, 30, size=(E, 2 * nb))
        set_forces(env, norms)
        exp = np_contact(env, labels, norms)
        got = rew(env, labels_file=str(path))
        got_pa = rew(env, labels_file=str(path), per_agent=True)
        out["random_states_max_abs_err"] = float(max(np.abs(got.double().numpy() - exp["team"]).max(), np.abs(got_pa.double().numpy() - exp["robot"]).max()))
        assert out["random_states_max_abs_err"] <= 1e-5, out
        # synthetic case 1: hand in the air (contact frame, palms 1 m from the contact points, no force)
        ex = np.zeros((E, 2 * nb))
        set_forces(env, ex)
        hi = [BODIES.index(h) for h in cv.HANDS]
        Ro = qmat(env.command.simulator_object_quat_w.double().numpy())
        cp = env.command.simulator_object_pos_w.double().numpy()[:, None, None] + np.einsum("eij,rhj->erhi", Ro, labels[1][15].astype(np.float64))
        Rh = qmat(env.simulator.agent_rigid_body_rot[:, :, hi].double().numpy())
        palm_w_off = np.einsum("erhij,hj->erhi", Rh, labels[3].astype(np.float64))
        env.simulator.agent_rigid_body_pos[:, :, hi] = torch.tensor(cp - palm_w_off + np.array([0, 0, 1.0]), dtype=torch.float)
        r_air = rew(env, labels_file=str(path), per_agent=True)
        want = 5.0 * math.exp(-1.0 / 0.3) * math.exp(-10.0 / 40.0)
        out["case_hand_in_air"] = dict(reward=float(r_air.mean()), expected=want)
        assert np.allclose(r_air.numpy(), want, atol=1e-5)
        term.reset()
        flags = []
        for step in range(26):
            flags.append(bool(term(env, labels_file=str(path)).any()))
        out["case_hand_in_air"]["first_terminating_step"] = flags.index(True) + 1
        assert flags.index(True) + 1 == 25, flags
        # synthetic case 2: hands touching (palm on the contact point) without force: reward gain * exp(-10/40), lost (F < 1 N)
        env.simulator.agent_rigid_body_pos[:, :, hi] = torch.tensor(cp - palm_w_off, dtype=torch.float)
        r_touch = rew(env, labels_file=str(path), per_agent=True)
        out["case_touching_no_force"] = dict(reward=float(r_touch.mean()), expected=5.0 * math.exp(-10.0 / 40.0))
        assert np.allclose(r_touch.numpy(), 5.0 * math.exp(-10.0 / 40.0), atol=1e-5)
        term.reset()
        term(env, labels_file=str(path))
        assert term.last_lost.all()
        # ... and with 20 N on every hand: full reward 5, not lost
        f20 = np.zeros((E, 2 * nb))
        for r in (0, 1):
            for h in cv.HANDS:
                f20[:, r * nb + BODIES.index(h)] = 20.0
        set_forces(env, f20)
        r_press = rew(env, labels_file=str(path), per_agent=True)
        term(env, labels_file=str(path))
        out["case_touching_20N"] = dict(reward=float(r_press.mean()), expected=5.0, lost_any=bool(term.last_lost.any()))
        assert np.allclose(r_press.numpy(), 5.0, atol=1e-5) and not term.last_lost.any()
        # synthetic case 3: robot 0 pushes the object with its left knee (shin far in the human data): penalty exp(-5)
        f_leg = f20.copy()
        f_leg[:, BODIES.index("left_knee_link")] = 50.0
        set_forces(env, f_leg)
        r_leg = rew(env, labels_file=str(path), per_agent=True)
        out["case_leg_push"] = dict(robot0=float(r_leg[:, 0].mean()), robot1=float(r_leg[:, 1].mean()), expected_robot0=5.0 * math.exp(-5.0))
        assert np.allclose(r_leg[:, 0].numpy(), 5.0 * math.exp(-5.0), atol=1e-5) and np.allclose(r_leg[:, 1].numpy(), 5.0, atol=1e-5)
        # the same knee force in a frame whose human shin is near the object: no penalty; torso force (human trunk near): none
        f_torso = f20.copy()
        f_torso[:, BODIES.index("torso_link")] = 50.0
        set_forces(env, f_torso)
        assert np.allclose(rew(env, labels_file=str(path), per_agent=True).numpy(), 5.0, atol=1e-5)
        # non-contact frame: reward 1 per hand whatever the distance / force
        env.command.time_steps[:] = 2
        set_forces(env, ex)
        r_free = rew(env, labels_file=str(path), per_agent=True)
        out["non_contact_frame_reward"] = float(r_free.mean())
        assert np.allclose(r_free.numpy(), 1.0)
    return out


if __name__ == "__main__":
    import json
    res = dict(observation_v1=test_obs(), contact_v1=test_contact())
    print(json.dumps(res, indent=1))
    print("OK")
