#!/usr/bin/env python3
"""T04 v4: physical-limit check per task and mass level (quasi-static, simulation parameters, CPU only).

Inputs (all read from the repository, nothing estimated by hand):
  * reference trajectories: the D training references of Pull (plan5_attempt08 mirrored pair), Table
    (core4d_smalltable5kg_A) and Bin (core4d_bucket003_20231020_071_a1): robot root and joint poses, object pose;
  * robot: the G1 rubber-hand training URDF (link masses, kinematics), joint torque limits of the holosoma robot config;
  * friction in the simulator: terrain plane 1.0 / 1.0 with friction_combine_mode "multiply" (isaacsim.py), object
    material 0.5 / 0.5 (startup randomization term, read back from PhysX in T04 A/object_physics_readback.csv), robot =
    the global material 1.0 / 1.0 with the default combine mode "average" (isaacsim.py SimulationCfg; to be read back on
    the GPU). PhysX uses the higher-priority combine mode of a pair (average < min < multiply < max), so
    mu(foot, ground) = 1.0 * 1.0, mu(object, ground) = 0.5 * 1.0, mu(hand, object) = (1.0 + 0.5) / 2.

Required force on the object, per frame (object acceleration from the reference, central differences of a 5-frame
moving average of the position):
  * Pull (dragged on its legs): horizontal F = mu_og m g + m a along the motion direction, shared by the 4 hands;
  * Table (carried by both robots): frames with the object >= 1 cm above its first-frame height; F = m (g z + a),
    shared by the 4 hands;
  * Bin: frames with the object >= 1 cm above its first-frame height; shared by the robots whose reference contact
    weight (interaction_vectors_v1.npz) is > 0.5 (2 hands each).
Two grasp cases for the lifting tasks: "support" (hands push the object up, no friction needed: the lower bound of
the arm load) and "squeeze" (each hand also presses horizontally toward the object centre with N = |F_up| / mu_ho so
friction carries the weight).
Arm torques at the reference posture, per arm: tau = J^T F + tau_gravity(arm links), J by finite differences of the
rubber-hand palm point (rubber_hand_link inertial origin) w.r.t. the 7 arm joints, gravity torque = dU/dq of the arm
links' potential energy. Utilization = |tau| / limit per joint; a level is "beyond the arm torque limit" when the
maximum utilization over the required frames is > 1 even in the support case. Ground: total horizontal force needed
vs mu_fg x the two robots' weight.

    hssim_s1/bin/python -B system1/t04/physics_check.py --out results_system1/T04/physics_check
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/holosoma"))
WBT = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
TASKS = dict(
    pull=dict(reference=WBT / "rubber_hand_largetable_v1/pull/plan5_attempt08_mirrored_pair_runtime.npz", kind="drag",
              orig=20.0, levels=(5.0, 10.0, 20.0, 40.0, 80.0)),
    table=dict(reference=WBT / "core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz", kind="carry", orig=5.0,
               levels=(2.5, 5.0, 10.0, 20.0, 40.0)),
    bucket=dict(reference=WBT / "core4d_bucket003_20231020_071_a1/core4d_pair_runtime_fps50.npz", kind="lift", orig=1.0,
                levels=(1.0, 2.0, 4.0, 4.0 * 2 ** 0.5, 8.0),
                contact=WBT / "core4d_bucket003_20231020_071_a1/interaction_vectors_v1.npz"),
)
PROPOSED = {("pull", 80.0): "add-level rule", ("table", 40.0): "add-level rule", ("bucket", round(4 * 2 ** 0.5, 6)): "midpoint rule"}
G = 9.81
MU_FG, MU_OG, MU_HO = 1.0 * 1.0, 0.5 * 1.0, (1.0 + 0.5) / 2
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def smooth_acc(x, fps):
    k = 5
    xs = np.copy(x)
    for i in range(len(x)):
        a, b = max(0, i - k // 2), min(len(x), i + k // 2 + 1)
        xs[i] = x[a:b].mean(0)
    v = np.gradient(xs, 1 / fps, axis=0)
    return v, np.gradient(v, 1 / fps, axis=0)


class Arm:
    def __init__(self):
        import yourdfpy
        self.u = yourdfpy.URDF.load(str(URDF), build_scene_graph=True, load_meshes=False, build_collision_scene_graph=False)
        self.names = list(self.u.actuated_joint_names)
        from holosoma.config_values import robot
        r = robot.g1_29dof_w_object
        self.limit = dict(zip(r.dof_names, r.dof_effort_limit_list))
        self.mass = {n: float(l.inertial.mass) for n, l in self.u.link_map.items() if l.inertial is not None}
        self.com = {n: l.inertial.origin[:3, 3] if l.inertial.origin is not None else np.zeros(3) for n, l in self.u.link_map.items() if l.inertial is not None}
        self.robot_mass = sum(self.mass.values())

    def links_distal(self, side):
        return [l for l in self.mass if l.startswith(f"{side}_") and any(k in l for k in ("shoulder", "elbow", "wrist", "rubber", "thumb", "pinky"))]

    def state(self, q, Tb, side, point="palm"):
        self.u.update_cfg(q)
        T = lambda l: Tb @ self.u.get_transform(l, self.u.base_link)  # noqa: E731
        if point == "palm":
            hand = T(f"{side}_rubber_hand_link")
            p = hand[:3, :3] @ self.com[f"{side}_rubber_hand_link"] + hand[:3, 3]
        else:  # wrist centre: origin of the wrist pitch link (on the wrist pitch axis)
            p = T(f"{side}_wrist_pitch_link")[:3, 3]
        U = sum(self.mass[l] * G * (T(l)[:3, :3] @ self.com[l] + T(l)[:3, 3])[2] for l in self.links_distal(side))
        return p, U

    def jac_grav(self, q, Tb, side, eps=1e-5, point="palm"):
        idx = [self.names.index(f"{side}_{j}_joint") for j in ARM]
        p0, U0 = self.state(q, Tb, side, point)
        J, g = np.zeros((3, 7)), np.zeros(7)
        for k, i in enumerate(idx):
            dq = q.copy()
            dq[i] += eps
            p1, U1 = self.state(dq, Tb, side, point)
            J[:, k] = (p1 - p0) / eps
            g[k] = (U1 - U0) / eps
        return p0, J, g, np.array([self.limit[f"{side}_{j}_joint"] for j in ARM])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    arm = Arm()
    rows, frames_out = [], []
    for task, cfg in TASKS.items():
        z = np.load(cfg["reference"], allow_pickle=True)
        fps = float(np.asarray(z["fps"]).reshape(-1)[0])
        jn = [str(x) for x in z["joint_names"]]
        bn = [str(x) for x in z["body_names"]]
        ip = bn.index("pelvis")
        op = np.asarray(z["object_pos_w"], float)
        v, acc = smooth_acc(op, fps)
        T = len(op)
        lifted = op[:, 2] >= op[0, 2] + 0.01
        if cfg["kind"] == "drag":
            moving = np.linalg.norm(v[:, :2], axis=1) > 0.02
            req = moving
        else:
            req = lifted
        cw = None
        if cfg.get("contact"):
            cw = np.load(cfg["contact"], allow_pickle=False)["reference_contact_weights"]
        travel = op[-1, :2] - op[0, :2]
        d_all = travel / max(np.linalg.norm(travel), 1e-9)
        idx_q = [jn.index(n) for n in arm.names]
        geo = {}
        for t in range(0, T, a.stride):
            if not req[t]:
                continue
            for ag in range(2):
                if cfg["kind"] == "lift":
                    if cw[t, ag] <= 0.5:
                        continue
                q = np.asarray(z["agent_joint_pos"][t, ag], float)[idx_q]
                pq = z["agent_body_quat_w"][t, ag, ip]
                Tb = np.eye(4)
                Tb[:3, :3] = Rotation.from_quat(np.asarray(pq)[[1, 2, 3, 0]]).as_matrix()
                Tb[:3, 3] = z["agent_body_pos_w"][t, ag, ip]
                for side in ("left", "right"):
                    geo[(t, ag, side)] = arm.jac_grav(q, Tb, side)
                    geo[(t, ag, side, "wrist")] = arm.jac_grav(q, Tb, side, point="wrist")
        for m in cfg["levels"]:
            util_sup, util_sq, ground, util_lb = [], [], [], []
            per_joint_lb = np.zeros(7)
            per_joint_sup = np.zeros(7)
            per_joint_sq = np.zeros(7)
            for t in range(0, T, a.stride):
                if not req[t]:
                    continue
                if cfg["kind"] == "drag":
                    sp = np.linalg.norm(v[t, :2])
                    d = v[t, :2] / sp if sp > 0.02 else d_all
                    Fh = MU_OG * m * G + m * max(0.0, float(acc[t, :2] @ d))
                    F_total = np.array([d[0] * Fh, d[1] * Fh, 0.0])
                    agents = [0, 1]
                else:
                    F_total = m * (np.array([0, 0, G]) + acc[t])
                    agents = [0, 1] if cfg["kind"] == "carry" else [ag for ag in range(2) if cw[t, ag] > 0.5]
                    if not agents:
                        continue
                ground.append(float(np.linalg.norm(F_total[:2])))
                share = F_total / (2 * len(agents))
                for ag in agents:
                    for side in ("left", "right"):
                        _, Jw, gw, lim = geo[(t, ag, side, "wrist")]
                        tl = np.abs(Jw.T @ share + gw) / lim  # lower bound: force through the wrist centre, support case
                        util_lb.append(tl.max())
                        per_joint_lb = np.maximum(per_joint_lb, tl)
                        p, J, g, lim = geo[(t, ag, side)]
                        ts = np.abs(J.T @ share + g) / lim
                        util_sup.append(ts.max())
                        per_joint_sup = np.maximum(per_joint_sup, ts)
                        if cfg["kind"] == "drag":
                            util_sq.append(ts.max())
                            per_joint_sq = np.maximum(per_joint_sq, ts)
                            continue
                        n = op[t] - p
                        n[2] = 0.0
                        n = n / max(np.linalg.norm(n), 1e-9)
                        F_sq = share + n * abs(share[2]) / MU_HO
                        tq = np.abs(J.T @ F_sq + g) / lim
                        util_sq.append(tq.max())
                        per_joint_sq = np.maximum(per_joint_sq, tq)
            us, uq, ul = np.array(util_sup), np.array(util_sq), np.array(util_lb)
            ground_cap = MU_FG * 2 * arm.robot_mass * G
            row = dict(task=task, mass_kg=round(m, 3), mass_ratio=round(m / cfg["orig"], 3), kind=cfg["kind"],
                       proposed=PROPOSED.get((task, round(m, 6)), ""), required_frames=int(len(us) // 2 // max(1, (2 if cfg["kind"] != "lift" else 1))),
                       arm_samples=len(us), util_support_max=float(us.max()), util_support_p95=float(np.percentile(us, 95)),
                       frac_support_over_1=float((us > 1).mean()), util_squeeze_max=float(uq.max()), frac_squeeze_over_1=float((uq > 1).mean()),
                       worst_joint_support=ARM[int(per_joint_sup.argmax())], worst_joint_squeeze=ARM[int(per_joint_sq.argmax())],
                       util_lowerbound_max=float(ul.max()), util_lowerbound_p95=float(np.percentile(ul, 95)),
                       frac_lowerbound_over_1=float((ul > 1).mean()), worst_joint_lowerbound=ARM[int(per_joint_lb.argmax())],
                       ground_force_needed_max_N=float(max(ground)), ground_friction_capacity_N=ground_cap,
                       ground_ratio=float(max(ground) / ground_cap))
            beyond = []
            if row["util_lowerbound_max"] > 1:
                beyond.append("arm torque lower bound > limit")
            if row["ground_ratio"] > 1:
                beyond.append("foot-ground friction")
            row["physically_impossible"] = bool(beyond)
            row["verdict"] = "beyond physical limit: " + ", ".join(beyond) if beyond else (
                "feasible by the lower bound; palm-contact model exceeds the wrist limit" if row["util_support_max"] > 1 else
                "feasible; squeeze grasp exceeds a limit" if row["util_squeeze_max"] > 1 else "feasible in all three models")
            rows.append(row)
            print(json.dumps(row), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "physics_check.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out / "inputs.json").write_text(json.dumps(dict(
        urdf=str(URDF.relative_to(ROOT)), urdf_sha256=sha(URDF), robot_mass_kg=arm.robot_mass,
        mu_foot_ground=MU_FG, mu_object_ground=MU_OG, mu_hand_object=MU_HO, g=G,
        arm_torque_limits_Nm={f"{s}_{j}_joint": arm.limit[f"{s}_{j}_joint"] for s in ("left",) for j in ARM},
        references={k: dict(path=str(v["reference"].relative_to(ROOT)), sha256=sha(v["reference"])) for k, v in TASKS.items()},
        bucket_contact=dict(path=str(TASKS["bucket"]["contact"].relative_to(ROOT)), sha256=sha(TASKS["bucket"]["contact"])),
        stride=a.stride), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
