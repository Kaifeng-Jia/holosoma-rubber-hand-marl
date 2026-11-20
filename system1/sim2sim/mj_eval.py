#!/usr/bin/env python3
"""T10 step 3: closed-loop evaluation in MuJoCo with the protocol of the laptop-5070 finals (scripts/evaluate_paper_supplement.py,
results_system1/T03/d_final_5070): the same frozen state file, deterministic actions, the same per-step order (observation at
reference frame k -> action -> 4 physics substeps -> termination against reference frame k (JointBadTrackingZOnly: torso gravity-z
error > 0.8, object position error > 0.25 m, object orientation error > 0.8 rad) -> record state with reference frame min(k+1, N-1)
-> reference frame k+1), completed = all N steps without joint_bad_tracking, metrics = paper_evaluation_utils.metrics.

  --scheme D     paper D actor (ONNX), D's PD (Isaac export of the D run)
  --scheme C-in  C-in actor (ONNX) + official SONIC ONNX; --pd isaac: the Isaac C-in PD (our gains, SONIC hip_pitch);
                 --pd official: the official deployment kp / kd (gear_sonic_deploy policy_parameters.hpp, computed by its formulas)
                 and the official sim2sim motor effort limits (gear_sonic mujoco_sim g1_29dof_sonic_model12.yaml), by joint name.

    sim2sim_s1/bin/python -B system1/sim2sim/mj_eval.py --task table --scheme D --seed 721 --level 2 --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from paper_evaluation_utils import metrics  # noqa: E402

from system1.sim2sim.mj_env import PairEnv  # noqa: E402
from system1.sim2sim.sonic_mj import CInController, DController  # noqa: E402

DEPLOY = Path("/home/kevin/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp")
SIMCFG = Path("/home/kevin/third_party/GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml")
MUJOCO_ORDER = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
                "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
                "right_wrist_pitch_joint", "right_wrist_yaw_joint"]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def official_pd():
    """kp, kd from the deploy header's formulas (per motor type, MuJoCo joint order) and the sim2sim effort limits; by joint name."""
    arm = dict(A5020=0.003609725, A7520_14=0.010177520, A7520_22=0.025101925, A4010=0.00425)
    w = 10 * 2.0 * 3.1415926535
    st = {k: v * w * w for k, v in arm.items()}
    da = {k: 2.0 * 2 * v * w for k, v in arm.items()}
    motor = ["A7520_22", "A7520_22", "A7520_14", "A7520_22", "2A5020", "2A5020"] * 2 + ["A7520_14", "2A5020", "2A5020"] + \
            ["A5020", "A5020", "A5020", "A5020", "A5020", "A4010", "A4010"] * 2
    kp = [2 * st[m[1:]] if m.startswith("2") else st[m] for m in motor]
    kd = [2 * da[m[1:]] if m.startswith("2") else da[m] for m in motor]
    import re
    txt = SIMCFG.read_text()
    eff = [float(x) for x in re.search(r"motor_effort_limit_list:\s*\[([^\]]*)\]", txt).group(1).replace("\n", " ").split(",")]
    eff_body = eff[:22] + eff[29:36]  # 43 motors: 22 body, 7 left hand, 7 right body, 7 right hand
    return dict(zip(MUJOCO_ORDER, kp)), dict(zip(MUJOCO_ORDER, kd)), dict(zip(MUJOCO_ORDER, eff_body))


def checkpoint_for(scheme, task, seed):
    rows = list(csv.DictReader((ROOT / "results_system1/T03/c2_main/per_seed_summary.csv").open()))
    r = next(x for x in rows if x["variant"] == scheme and x["task"] == task and int(x["training_seed"]) == seed and x["perturbation_level_cm"] == "2")
    return r["checkpoint_sha256"]


def states_for(task, level):
    name = {("table", 2): "cin_table_m5kg_seed721_fresh12000_pm2", ("table", 8): "cin_table_m5kg_seed721_fresh12000_pm8",
            ("bucket", 2): "cin_bucket_m1kg_seed721_fresh12000_pm2", ("bucket", 8): "cin_bucket_m1kg_seed721_fresh12000_pm8"}[(task, level)]
    argv = json.loads((ROOT / "logs_system1/post_train/final" / (name + ".launch.json")).read_text())["argv"]
    return Path(argv[argv.index("--states") + 1])


def run_episode(env, ctrl, init, save=None):
    env.reset(init)
    ctrl.reset()
    n = env.n

    def snap(frame):
        st = env.state()
        s = env.sample(frame)
        return dict(root_pos=st["root"][:, :3].copy(), root_quat_xyzw=st["root"][:, 3:7].copy(), root_velocity=st["root"][:, 7:13].copy(),
                    dof_pos=st["dof_pos"].copy(), dof_vel=st["dof_vel"].copy(), object_pos_w=st["obj_pos"].copy(), object_quat_xyzw=st["obj_quat_xyzw"].copy(),
                    reference_object_pos_w=s["object_pos_w"].copy(), reference_object_quat_xyzw=s["object_quat_w"].copy(), reference_frame=np.asarray(frame))
    frames = [snap(0)]
    terms = {}
    step = 0
    for k in range(n):
        st = env.state()
        target, _ = ctrl.step(st)
        env.physics_step(target)
        step = k + 1
        tm = env.termination(t=k)
        bad = tm["bad_robot"] or tm["bad_object_pos"] or tm["bad_object_ori"]
        horizon = (k >= n - 1) and not bad
        terms = dict(reference_horizon=bool(horizon), joint_bad_tracking=bool(bad))
        frames.append(snap(min(step, n - 1)))
        if not np.isfinite(frames[-1]["root_pos"]).all():
            raise ValueError("non-finite state")
        env.t = k + 1
        if bad or horizon:
            break
    channels = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    completed = step == n and not terms.get("joint_bad_tracking", False)
    res = dict(state_id=init["state_id"], kind=init["kind"], steps=step, horizon_steps=n, completed_reference=bool(completed), progress=step / n,
               terminal_terms=terms, **metrics(channels), last_termination=dict(gravity_z_err=tm["gravity_z_err"], object_pos_err=tm["object_pos_err"],
                                                                                object_ori_err=tm["object_ori_err"]))
    if save is not None:
        np.savez_compressed(save, **channels)
        res["trajectory"] = str(save)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("table", "bucket"), required=True)
    ap.add_argument("--scheme", choices=("D", "C-in"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--level", type=int, choices=(2, 8), required=True)
    ap.add_argument("--pd", choices=("isaac", "official"), default="isaac")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    dump = ROOT / "logs_system1/t10/isaac" / (("d_" if a.scheme == "D" else "cin_") + a.task) / "dump"
    env = PairEnv(dump, a.task)
    name = f"{a.scheme.replace('-', '').lower()}_{a.task}_seed{a.seed}"
    onnx = ROOT / "exports_system1/t10/onnx" / f"{name}.onnx"
    pd_note = "isaac"
    if a.scheme == "D":
        if a.pd != "isaac":
            raise ValueError("D runs with its paper setting only")
        ctrl = DController(env, onnx)
    else:
        c = env.ss.control
        ctrl = CInController(env, onnx, c["action_scales"], (c["dof_pos_lower"], c["dof_pos_upper"]))
        if a.pd == "official":
            kp, kd, eff = official_pd()
            env.kp = np.array([kp[n] for n in env.dof_names])
            env.kd = np.array([kd[n] for n in env.dof_names])
            env.tau_lim = np.array([eff[n] for n in env.dof_names])
            for r in range(2):
                env.model.actuator_ctrlrange[env.act[r]] = np.stack([-env.tau_lim, env.tau_lim], axis=1)
            pd_note = "official deploy kp/kd (policy_parameters.hpp formulas) + official sim2sim motor effort limits"
    states_path = states_for(a.task, a.level)
    states = json.loads(states_path.read_text())["states"][: a.limit or None]
    manifest = dict(task=a.task, scheme=a.scheme, seed=a.seed, level=a.level, pd=a.pd, pd_note=pd_note, onnx=str(onnx), onnx_sha256=sha(onnx),
                    checkpoint_sha256=checkpoint_for(a.scheme, a.task, a.seed), states=str(states_path), states_sha256=sha(states_path),
                    isaac_dump=str(dump.relative_to(ROOT)), scene_json_sha256=sha(dump / "scene.json"), convexes_sha256=sha(dump / "collision_convexes.npz"),
                    kp=env.kp.tolist(), kd=env.kd.tolist(), torque_limits=env.tau_lim.tolist(), sim_dt=float(env.model.opt.timestep), decimation=4,
                    sonic_encoder_sha256=sha(ctrl.sonic.enc._model_path) if False else None, command=[sys.executable, *sys.argv])
    if a.scheme == "C-in":
        from system1.sim2sim.sonic_mj import HF
        manifest.update(sonic_encoder=str(HF / "model_encoder.onnx"), sonic_encoder_sha256=sha(HF / "model_encoder.onnx"),
                        sonic_decoder=str(HF / "model_decoder.onnx"), sonic_decoder_sha256=sha(HF / "model_decoder.onnx"))
    (a.out / "run_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    results = []
    t0 = time.time()
    for init in states:
        r = run_episode(env, ctrl, init, save=a.out / f"{init['state_id']}.trajectory.npz")
        results.append(r)
        (a.out / "episodes.json").write_text(json.dumps(results, indent=1) + "\n")
        print(json.dumps({k: r[k] for k in ("state_id", "steps", "completed_reference")}), flush=True)
    (a.out / "summary.json").write_text(json.dumps(dict(manifest, episodes=results, complete=len(results) == len(states), episode_count=len(results),
                                                        wall_s=time.time() - t0), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
