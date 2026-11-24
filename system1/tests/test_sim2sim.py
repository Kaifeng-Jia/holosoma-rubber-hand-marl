"""Smoke test for the T10 sim2sim components (system1/sim2sim/): official PD formulas, MuJoCo scene from the Isaac export,
reset = Isaac initial state, observation = Isaac observation at a recorded step, controllers run. Needs the Isaac export in
logs_system1/t10/isaac (system1/sim2sim/run_isaac_dumps.py); skipped otherwise.

    PYTHONPATH=src/holosoma sim2sim_s1/bin/python -B -m pytest -q system1/tests/test_sim2sim.py   (the repo conftest imports holosoma)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DUMP = ROOT / "logs_system1/t10/isaac/d_table/dump"
CIN = ROOT / "logs_system1/t10/isaac/cin_table/dump"
pytestmark = pytest.mark.skipif(not (DUMP / "scene.json").exists(), reason="Isaac export missing")


def test_official_pd_matches_isaac_cin_gains():
    from system1.sim2sim.mj_eval import official_pd
    kp, kd, eff = official_pd()
    assert abs(kp["left_hip_pitch_joint"] - 99.0984) < 1e-3 and abs(kp["left_ankle_pitch_joint"] - 28.5012) < 1e-3
    c = json.loads((CIN / "scene.json").read_text())["control"]
    for i, n in enumerate(c["dof_names"]):
        assert abs(c["p_gains"][i] - kp[n]) < 1e-3 and abs(c["d_gains"][i] - kd[n]) < 1e-3
    assert eff["left_hip_roll_joint"] == 88.0 and eff["left_wrist_pitch_joint"] == 5.0


def test_scene_and_reset():
    import mujoco
    from system1.sim2sim.mj_env import PairEnv
    env = PairEnv(DUMP, "table")
    m = env.model
    assert m.nu == 58 and abs(m.opt.timestep - 0.005) < 1e-12
    s = json.loads((DUMP / "scene.json").read_text())
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r0_pelvis")
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r1_pelvis")
    assert abs(m.body_mass[a:b].sum() - sum(s["bodies"]["robot"]["masses"])) < 1e-6
    states = json.loads(Path(json.loads((ROOT / "logs_system1/t10/isaac/d_table/launch.json").read_text())["argv"][
        json.loads((ROOT / "logs_system1/t10/isaac/d_table/launch.json").read_text())["argv"].index("--states") + 1]).read_text())["states"]
    env.reset(states[0])
    st = env.state()
    isa = json.loads((ROOT / "logs_system1/t10/isaac/d_table/eval/nominal.initial.json").read_text())["actual_initial_state"]
    assert np.abs(st["dof_pos"] - np.array(isa["dof_pos"])).max() < 1e-6
    assert np.abs(st["root"][:, :3] - np.array(isa["root_pos"])).max() < 1e-6
    assert np.abs(st["obj_pos"] - np.array(isa["object_pos_w"])).max() < 1e-6


def test_observation_matches_isaac_record():
    from system1.sim2sim.mj_env import PairEnv
    z = np.load(DUMP / "nominal.sim2sim.npz")
    env = PairEnv(DUMP, "table")
    k = 100
    env.set_state(z["root_states"][k, 0].astype(float), z["dof_pos"][k, 0], z["dof_vel"][k, 0], np.zeros(3), np.array([0, 0, 0, 1.0]))
    env.t = k
    actor, mate = env.observation(z["actions"][k - 1, 0].reshape(2, 29))
    assert np.abs(actor - z["actor_obs"][k, 0]).max() < 1e-4
    assert np.abs(mate - z["teammate_obs"][k, 0]).max() < 1e-4


@pytest.mark.skipif(not (ROOT / "exports_system1/t10/onnx/cin_table_seed721.onnx").exists(), reason="ONNX export missing")
def test_controllers_step():
    from system1.sim2sim.mj_env import PairEnv
    from system1.sim2sim.sonic_mj import CInController
    env = PairEnv(CIN, "table")
    c = env.ss.control
    ctrl = CInController(env, ROOT / "exports_system1/t10/onnx/cin_table_seed721.onnx", c["action_scales"], (c["dof_pos_lower"], c["dof_pos_upper"]))
    states = json.loads(Path("/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003/table_states.json").read_text())["states"]
    env.reset(states[0])
    ctrl.reset()
    target, info = ctrl.step(env.state())
    assert target.shape == (2, 29) and np.isfinite(target).all() and info["sonic_tokenizer"].shape == (2, 1761)
    env.physics_step(target)
    assert np.isfinite(env.state()["root"]).all()


@pytest.mark.skipif(not (ROOT / "exports_system1/t10/onnx/cout_table_seed721.onnx").exists(), reason="C-out ONNX export missing")
def test_cout_controller_step():
    """T13 C1: C-out = SONIC on the unmodified reference + paper-D scale * clip(a) on SONIC's output target; last action = raw a.
    Runs on the C-out Isaac export when present, else on the C-in export (same scheme-C PD path)."""
    from system1.sim2sim.mj_env import PairEnv
    from system1.sim2sim.sonic_mj import COutController
    cout = ROOT / "logs_system1/t10/isaac/cout_table/dump"
    env = PairEnv(cout if (cout / "scene.json").exists() else CIN, "table")
    ctrl = COutController(env, ROOT / "exports_system1/t10/onnx/cout_table_seed721.onnx", env.ss.control["action_scales"])
    states = json.loads(Path("/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003/table_states.json").read_text())["states"]
    env.reset(states[0])
    ctrl.reset()
    st = env.state()
    target, info = ctrl.step(st)
    assert target.shape == (2, 29) and np.isfinite(target).all() and info["sonic_tokenizer"].shape == (2, 1761)
    # q* - SONIC target = scale * clip(a): recompute SONIC alone on the same inputs from a fresh history
    ctrl.sonic.reset()
    _, target_s, _ = ctrl.sonic.act(*ctrl.sonic_inputs_out(st, 0))
    a = info["actions"]
    a_c = a if env.clip_a is None else np.clip(a, -env.clip_a, env.clip_a)
    assert np.allclose(target - target_s[:, ctrl.sonic.s2h], a_c * np.asarray(env.ss.control["action_scales"]), atol=1e-6)
    assert np.array_equal(ctrl.last, a)  # the actor's last-action input is the raw a (as D)
    env.physics_step(target)
    assert np.isfinite(env.state()["root"]).all()
