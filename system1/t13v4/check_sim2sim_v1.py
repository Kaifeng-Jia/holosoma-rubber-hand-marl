#!/usr/bin/env python3
"""T13 v4 section 2 / 5 item 3: MuJoCo-side checks of observation v1 with the T10 thresholds, for one finished run (read only, CPU).

1. Initial states: every frozen +/-2 / +/-8 test state (nominal + 50) loaded into MuJoCo (mj_env.PairEnv.reset: reference frame 0
   + the evaluator's yaw_transform) versus the state Isaac had after its reset in this run's laptop-5070 finals
   (logs_system1/t13v4/finals/<run>_pm{2,8}/<state>.initial.json; state-file sha256 checked against the finals' launch record).
   Pass: joints 1e-4 rad, robot root and object positions 1e-4 m (T10 step 1).
2. Observation v1 (noise off): for every control step of this run's Isaac nominal episode with scene export
   (logs_system1/t13v4/isaac_dump/<run>: dump/nominal.sim2sim.npz holds the per-step observations and robot states;
   eval/nominal.trajectory.npz holds the object pose of the same frames), that state is written into MuJoCo and the 159-D actor
   input is built by mj_env (obs_v1 on) and compared per dimension with the Isaac observation. Pass: every dimension < 1e-4.
   Also: MuJoCo forward kinematics of all Isaac bodies, and the actor ONNX on the recorded observations vs the Isaac actions.

    sim2sim_s1/bin/python -B system1/t13v4/check_sim2sim_v1.py --run d_push --task push --onnx exports_system1/t13v4/onnx/d_push_t13v4.onnx \
        --out logs_system1/t13v4/sim2sim/checks_d_push
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sim2sim.check_observations import NAMES as NAMES158  # noqa: E402
from system1.sim2sim.mj_env import PairEnv  # noqa: E402

NAMES = NAMES158[:154] + ["neck_partner_rel_pos_b_x", "neck_partner_rel_pos_b_y", "object_rel_pos_b_x", "object_rel_pos_b_y", "object_rel_pos_b_z"]
TOL = 1e-4


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def qangle(a, b):
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return 2 * np.arccos(min(1.0, d))


def initial_states(env, run, rows, files):
    for lvl in (2, 8):
        idir = ROOT / f"logs_system1/t13v4/finals/{run}_pm{lvl}"
        launch = json.loads((idir.parent / f"{run}_pm{lvl}.launch.json").read_text())
        states_path = ROOT / launch["states"]
        assert sha(states_path) == launch["states_sha256"]
        files.append(dict(level=lvl, states=launch["states"], states_sha256=launch["states_sha256"], isaac_eval_dir=str(idir.relative_to(ROOT))))
        for init in json.loads(states_path.read_text())["states"]:
            f = idir / f"{init['state_id']}.initial.json"
            if not f.exists():
                continue
            env.reset(init)
            st = env.state()
            isa = json.loads(f.read_text())["actual_initial_state"]
            iv = np.array(isa["root_velocity"])
            r = dict(level=lvl, state_id=init["state_id"],
                     joint_pos_max_abs_rad=float(np.abs(st["dof_pos"] - np.array(isa["dof_pos"])).max()),
                     joint_vel_max_abs=float(np.abs(st["dof_vel"] - np.array(isa["dof_vel"])).max()),
                     root_pos_max_abs_m=float(np.abs(st["root"][:, :3] - np.array(isa["root_pos"])).max()),
                     root_ori_max_angle_rad=max(qangle(st["root"][i, 3:7], np.array(isa["root_quat_xyzw"])[i]) for i in range(2)),
                     root_lin_vel_max_abs=float(np.abs(st["root"][:, 7:10] - iv[:, :3]).max()),
                     root_ang_vel_max_abs=float(np.abs(st["root"][:, 10:13] - iv[:, 3:6]).max()),
                     object_pos_max_abs_m=float(np.abs(st["obj_pos"] - np.array(isa["object_pos_w"])).max()),
                     object_ori_angle_rad=qangle(st["obj_quat_xyzw"], np.array(isa["object_quat_xyzw"])))
            r["passed"] = r["joint_pos_max_abs_rad"] < 1e-4 and r["root_pos_max_abs_m"] < 1e-4 and r["object_pos_max_abs_m"] < 1e-4
            rows.append(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scheme", choices=("D", "C-out"), default="D",
                    help="round 2 C-out: also SONIC's inputs (unmodified reference, its own last action) and q* = SONIC target + D scale * "
                         "clip(a) step by step against the Isaac record (as T10c check_observations.py --runs cout_*); D = unchanged")
    ap.add_argument("--base", type=Path, default=None, help="Isaac export directory (default logs_system1/t13v4/isaac_dump/<run>)")
    ap.add_argument("--no-initial", action="store_true", help="test: skip the initial-state check (needs the run's Isaac finals)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    base = ROOT / "logs_system1/t13v4/isaac_dump" / a.run if a.base is None else a.base.resolve()
    env = PairEnv(base / "dump", a.task)
    env.obs_v1 = True
    # 1. initial states
    irows, files = [], []
    if not a.no_initial:
        initial_states(env, a.run, irows, files)
        with (a.out / "initial_states.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(irows[0]))
            w.writeheader()
            w.writerows(irows)
    imax = {k: max(x[k] for x in irows) for k in irows[0] if k.endswith(("_rad", "_m", "_abs"))} if irows else {}
    cout = sonic = None
    s_err = dict(actor_obs=0.0, tokenizer=0.0, action=0.0, target=0.0, q_rec=0.0, q_onnx=0.0)
    if a.scheme == "C-out":
        import torch
        from system1.sim2sim.sonic_mj import ActorOnnx as _A, COutController, SonicOnnx
        sonic = SonicOnnx(env.dof_names)
        cout = COutController.__new__(COutController)
        cout.env, cout.sonic, cout.scale_d = env, sonic, np.array(env.ss.control["action_scales"])
        act_c = _A(a.onnx)
    # 2. observation v1
    z = np.load(base / "dump/nominal.sim2sim.npz")
    tr = np.load(base / "eval/nominal.trajectory.npz")
    K = z["actor_obs"].shape[0]
    assert z["teammate_obs"].shape[-1] == 5, z["teammate_obs"].shape
    err = np.zeros((K, 2, 159))
    fk_pos = fk_ang = 0.0
    obj_frame_root_err = 0.0
    for k in range(K):
        root = z["root_states"][k, 0].astype(np.float64)
        # trajectory frame k = the state after k control steps (frame 0 = after the reset) = the state observation k is built from
        obj_p = tr["object_pos_w"][k].reshape(3).astype(np.float64)
        obj_q = tr["object_quat_xyzw"][k].reshape(4).astype(np.float64)
        obj_frame_root_err = max(obj_frame_root_err, float(np.abs(tr["root_pos"][k] - root[:, :3]).max()))
        env.set_state(root, z["dof_pos"][k, 0], z["dof_vel"][k, 0], obj_p, obj_q)
        st = env.state()
        fk_pos = max(fk_pos, float(np.abs(st["body_pos"] - z["body_pos_w"][k, 0]).max()))
        dq = np.abs(np.sum(st["body_quat_xyzw"] * z["body_quat_w_xyzw"][k, 0], axis=-1))
        fk_ang = max(fk_ang, float((2 * np.arccos(np.clip(dq, -1, 1))).max()))
        lk = "actions" if a.scheme == "D" else "policy_action"  # scheme C: the env's last-action input (as T10c check_observations.py)
        last = z[lk][k - 1, 0].reshape(2, 29) if k > 0 else np.zeros((2, 29))
        env.t = k
        actor, mate = env.observation(last, st=st)
        err[k] = np.abs(np.concatenate([actor, mate], axis=1) - np.concatenate([z["actor_obs"][k, 0], z["teammate_obs"][k, 0]], axis=1))
        if cout is not None:  # as system1/sim2sim/check_observations.py (T10c, C-out)
            if k == 0:
                sonic.reset()
            else:
                sonic.last_action = torch.as_tensor(np.clip(z["sonic_action"][k - 1], -sonic.clip, sonic.clip), dtype=torch.float32)
            act, target_s, sobs = sonic.act(*cout.sonic_inputs_out(st, k))
            s_err["actor_obs"] = max(s_err["actor_obs"], float(np.abs(sobs["sonic_actor_obs"] - z["sonic_actor_obs"][k]).max()))
            s_err["tokenizer"] = max(s_err["tokenizer"], float(np.abs(sobs["sonic_tokenizer"] - z["sonic_tokenizer"][k]).max()))
            s_err["action"] = max(s_err["action"], float(np.abs(act - z["sonic_action"][k]).max()))
            s_err["target"] = max(s_err["target"], float(np.abs(target_s - z["sonic_target_s"][k]).max()))
            th = target_s[:, sonic.s2h].astype(np.float64)
            pd_rec = z["pd_target"][k, 0].reshape(2, 29).astype(np.float64)
            s_err["q_rec"] = max(s_err["q_rec"], float(np.abs(th + cout.residual(z["actions"][k, 0].reshape(2, 29).astype(np.float64)) - pd_rec).max()))
            mine = np.concatenate([actor, mate], axis=1)
            s_err["q_onnx"] = max(s_err["q_onnx"], float(np.abs(th + cout.residual(act_c(mine)) - pd_rec).max()))
    from system1.sim2sim.sonic_mj import ActorOnnx
    obs_rec = np.concatenate([z["actor_obs"][:, 0], z["teammate_obs"][:, 0]], axis=-1).reshape(-1, 159)
    a_err = float(np.abs(ActorOnnx(a.onnx)(obs_rec) - z["actions"][:, 0].reshape(-1, 29)).max())
    mx = err.max(axis=(0, 1))
    with (a.out / "observations_per_dim.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dim", "name", "max_abs_diff", "passed"])
        w.writeheader()
        w.writerows([dict(dim=i, name=n, max_abs_diff=float(mx[i]), passed=bool(mx[i] < TOL)) for i, n in enumerate(NAMES)])
    summary = dict(run=a.run, task=a.task, rules=__doc__,
                   initial_states=dict(n_states=len(irows), all_passed=all(x["passed"] for x in irows), max=imax, thresholds=dict(joint_rad=1e-4, position_m=1e-4),
                                       state_files=files),
                   observations=dict(steps=int(K), obs_max_abs_diff=float(mx.max()), obs_worst_dims=[NAMES[i] for i in np.argsort(-mx)[:5]],
                                     teammate_v1_max_abs_diff=float(mx[154:].max()), obs_passed=bool(mx.max() < TOL), threshold=TOL,
                                     fk_body_pos_max_m=fk_pos, fk_body_ori_max_rad=fk_ang, trajectory_vs_dump_root_pos_max_m=obj_frame_root_err,
                                     actor_onnx_on_recorded_obs_vs_isaac_actions_max_abs=a_err),
                   inputs=dict(dump=str((base / "dump").relative_to(ROOT)), scene_json_sha256=sha(base / "dump/scene.json"),
                               record_sha256=sha(base / "dump/nominal.sim2sim.npz"), trajectory_sha256=sha(base / "eval/nominal.trajectory.npz"),
                               onnx=str(a.onnx), onnx_sha256=sha(a.onnx)))
    if cout is not None:
        summary["cout"] = dict(sonic_actor_obs_max_abs_diff=s_err["actor_obs"], sonic_tokenizer_max_abs_diff=s_err["tokenizer"],
                               sonic_inputs_passed=bool(max(s_err["actor_obs"], s_err["tokenizer"]) < TOL), sonic_action_onnx_vs_isaac_max_abs_diff=s_err["action"],
                               sonic_target_onnx_vs_isaac_max_abs_rad=s_err["target"], q_star_from_recorded_actions_vs_isaac_pd_target_max_abs_rad=s_err["q_rec"],
                               q_star_from_mujoco_obs_and_actor_onnx_vs_isaac_pd_target_max_abs_rad=s_err["q_onnx"], q_star_passed=bool(s_err["q_rec"] < TOL),
                               threshold=TOL)
    if a.no_initial:
        summary["initial_states"]["note"] = "skipped (--no-initial, test)"
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(dict(initial=summary["initial_states"]["all_passed"], n=len(irows), imax=imax, obs=summary["observations"]), indent=1))


if __name__ == "__main__":
    main()
