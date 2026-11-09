#!/usr/bin/env python3
"""Build robot_model_diff.csv: SONIC's G1 versus our rubber-hand G1.

SONIC side: runtime values read back from SONIC's own Isaac Lab articulation
(``static/*`` arrays in the native dump), plus its URDF for link masses/colliders.
Our side: holosoma ``g1_29dof_w_object`` config (what the paper runs used) and the
rubber-hand URDF. Joint-wise rows are matched by joint name, so ordering
differences are reported separately and never hide value differences.

Run with hssim_s1 from the repository root:
    python system1/sonic/robot_model_diff.py --dump <native_dump.npz> --sonic-urdf <main.urdf> --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "holosoma"))
OUR_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
# gear_sonic/envs/manager_env/robots/g1.py G1_CYLINDER_MODEL_12_DEX_CFG.init_state.joint_pos (others 0)
SONIC_NOMINAL_DEFAULT = {
    "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669, "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6, "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2,
}


def urdf_links(path):
    r = ET.parse(path).getroot()
    out = {}
    for l in r.findall("link"):
        m = l.find("inertial/mass")
        out[l.get("name")] = dict(
            mass=float(m.get("value")) if m is not None else 0.0,
            collision=";".join(c.find("geometry")[0].tag for c in l.findall("collision")) or "none",
        )
    joints = {}
    for j in r.findall("joint"):
        joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"))
    return out, joints


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--sonic-urdf", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    from holosoma.config_values.robot import g1_29dof_w_object as R

    z = np.load(a.dump, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    s_names = meta["joint_names"]
    S = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith("static/")}
    o_names = list(R.dof_names)
    ctl = R.control

    def ours(name):
        kp = kd = None
        for k, v in ctl.stiffness.items():
            if k in name:
                kp, kd = v, ctl.damping[k]
        i = o_names.index(name)
        eff = R.dof_effort_limit_list[i]
        return dict(
            index=i,
            default=R.init_state.default_joint_angles[name],
            kp=kp,
            kd=kd,
            effort=eff,
            velocity=R.dof_vel_limit_list[i],
            armature=R.dof_armature_list[i],
            lower=R.dof_pos_lower_limit_list[i],
            upper=R.dof_pos_upper_limit_list[i],
            action_scale=ctl.action_scale * eff / kp,
        )

    rows = []

    def add(cat, item, sv, ov, note=""):
        if isinstance(sv, float) and isinstance(ov, float):
            same = abs(sv - ov) <= 1e-5 * max(1.0, abs(ov))
        else:
            same = sv == ov
        rows.append(dict(category=cat, item=item, sonic=sv, ours=ov, same=same, note=note))

    add("joint_order", "order", " ".join(s_names), " ".join(o_names),
        "SONIC = Isaac Lab articulation order; ours = holosoma dof_names (MuJoCo order); remapped by name")
    add("joint_order", "same_name_set", True, set(s_names) == set(o_names))
    for name in o_names:
        si = s_names.index(name)
        o = ours(name)
        lim = S["joint_pos_limits"].reshape(-1, 2)[si]
        add("default_pose_nominal", name, float(SONIC_NOMINAL_DEFAULT.get(name, 0.0)), float(o["default"]),
            "SONIC g1.py init_state (regex) vs holosoma default_joint_angles")
        add("default_pose_runtime_env0", name, float(S["default_joint_pos"][si]), float(o["default"]),
            "SONIC eval env0 after startup event add_joint_default_pos U(-0.01,0.01); informational")
        add("kp", name, float(S["joint_stiffness"][si]), float(o["kp"]))
        add("kd", name, float(S["joint_damping"][si]), float(o["kd"]))
        add("effort_limit", name, float(S["joint_effort_limits"][si]), float(o["effort"]))
        add("velocity_limit", name, float(S["joint_vel_limits"][si]), float(o["velocity"]))
        add("armature", name, float(S["joint_armature"][si]), float(o["armature"]))
        add("pos_limit_lower", name, float(lim[0]), float(o["lower"]))
        add("pos_limit_upper", name, float(lim[1]), float(o["upper"]))
        add("action_scale", name, float(S["action_scale"][si]), float(o["action_scale"]))
    add("actuator", "pd_implementation", "Isaac Lab ImplicitActuatorCfg (PhysX joint drive)",
        "IdealPDActuatorCfg(stiffness=0,damping=0); PD computed in Python each 200 Hz substep",
        "both 200 Hz physics, decimation 4")
    add("control", "physics_dt", meta.get("physics_dt"), 0.005)
    add("control", "policy_dt", meta.get("step_dt"), 0.02)

    s_links, s_joints = urdf_links(a.sonic_urdf)
    o_links, o_joints = urdf_links(OUR_URDF)
    for name in sorted(set(s_links) | set(o_links)):
        sl, ol = s_links.get(name), o_links.get(name)
        add("link_mass_kg", name, None if sl is None else sl["mass"], None if ol is None else ol["mass"])
        add("collision", name, None if sl is None else sl["collision"], None if ol is None else ol["collision"])
    hand = lambda L, keys: sum(v["mass"] for k, v in L.items() if any(t in k for t in keys))  # noqa: E731
    add("hand_mass_total_per_side_kg", "left",
        hand(s_links, ["left_hand_"]), hand(o_links, ["left_rubber_hand"]),
        "SONIC: Dex3 palm + 7 finger links (fixed joints); ours: rubber hand")
    add("robot_mass_total_kg", "urdf_sum", sum(v["mass"] for v in s_links.values()), sum(v["mass"] for v in o_links.values()))
    sim_masses = S["body_masses"]
    add("robot_mass_total_kg", "sonic_runtime_physx_sum", float(sim_masses.sum()), None,
        "PhysX read-back in SONIC eval env0 (startup mass randomization events active)")
    add("asset", "urdf", str(a.sonic_urdf), str(OUR_URDF))
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["category", "item", "sonic", "ours", "same", "note"])
        w.writeheader()
        w.writerows(rows)
    n_diff = sum(not r["same"] for r in rows)
    print(f"{len(rows)} rows, {n_diff} differ -> {a.out}")


if __name__ == "__main__":
    main()
