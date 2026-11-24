#!/usr/bin/env python3
"""T13 v4 section 0 item 4: test and validation initial states on the T14 round 6 C references (Kevin 2026-11-24).

Generators and seeds = the original ones:
  * +/-2 test: scripts/paper_evaluation_utils.make_states(case, seed=241003) (nominal + 50 states, independent uniform per robot
    world XY within +/-2 cm and yaw within +/-2 deg);
  * +/-8 test: the +/-2 draws scaled exactly by 4.0 (method of the V3 extended evaluation, recorded in its pm8 files);
  * validation: the T10b generator = make_states(case, seed=20261120); Table had one (system1/t10b/validation_states), Push had
    none and gets 50 +/-2 states here. Every validation state is checked against every +/-2 and +/-8 test state: no duplicate.
The states are perturbations relative to the reference's first frame (robot_xy_m, robot_yaw_rad; the evaluator applies them
with paper_evaluation_utils.yaw_transform to the nominal roots); so the regenerated values equal the old files' values. Each new
file carries a system1_t13v4 block naming the T14 reference it belongs to.

Also checked here (geometry only, no simulation): every state applied to frame 0 of the new reference (and, for comparison, of
the old one) - largest robot-object penetration per robot (simulator collision shapes, MuJoCo mj_geomDistance as T14) and the
smallest robot-robot distance.

    sonic_s1/bin/python -B system1/t13v4/make_states_t14.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
for p in ("system1/t14", "system1/t12v2", "src/holosoma_retargeting"):
    sys.path.insert(0, str(ROOT / p))
from paper_evaluation_utils import make_states, yaw_transform  # noqa: E402

B = Path("/home/kevin/holosoma-core4d-base/logs")
OLD = dict(table=dict(pm2=B / "PaperSupplement20260929/unified_evaluation_20261003/table_states.json",
                      pm8=B / "V3_20261102/cloud_backup_20261104/raw/v3_extended_eval_20261104_ops/states/table_pm8.json",
                      validation=ROOT / "system1/t10b/validation_states/table_validation_states.json"),
           push=dict(pm2=B / "PaperSupplement20260929/unified_evaluation_20261003/push_states.json",
                     pm8=B / "V3_20261102/cloud_backup_20261104/raw/v3_extended_eval_20261104_ops/states/push_pm8.json",
                     validation=None))
WBT = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
MANIFEST = dict(table=WBT / "system1_t14r6C/table/t14_reference.json", push=WBT / "system1_t14r6C/push/t14_reference.json")
RUNTIME_OLD = dict(table=WBT / "core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz",
                   push=WBT / "rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz")
OUT = ROOT / "system1/t13v4/states_t14r6C"
TEST_SEED, VAL_SEED = 241003, 20261120
PUSH_SPACING_M = 0.8780472


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(s):
    return json.dumps([s["robot_xy_m"], s["robot_yaw_rad"]])


def scaled(rec, f):
    out = json.loads(json.dumps(rec))
    out.update(position_bound_m=rec["position_bound_m"] * f, yaw_bound_deg=rec["yaw_bound_deg"] * f)
    for s in out["states"]:
        s["robot_xy_m"] = (np.asarray(s["robot_xy_m"]) * f).tolist()
        s["robot_yaw_rad"] = (np.asarray(s["robot_yaw_rad"]) * f).tolist()
    return out


def max_value_diff(a, b):
    sa, sb = {s["state_id"]: s for s in a["states"]}, {s["state_id"]: s for s in b["states"]}
    if set(sa) != set(sb):
        return None
    return max(max(float(np.abs(np.subtract(sa[i]["robot_xy_m"], sb[i]["robot_xy_m"])).max()),
                   float(np.abs(np.subtract(sa[i]["robot_yaw_rad"], sb[i]["robot_yaw_rad"])).max())) for i in sa)


def generate(case):
    man = json.loads(MANIFEST[case].read_text())
    block = dict(t14_reference=man["name"], t14_reference_manifest=str(MANIFEST[case].relative_to(ROOT)),
                 t14_reference_manifest_sha256=sha(MANIFEST[case]), decision="Kevin 2026-11-24: T13 v4 section 0 item 4",
                 generator="scripts/paper_evaluation_utils.make_states")
    pm2 = make_states(case, seed=TEST_SEED)
    pm8 = scaled(pm2, 4.0)
    val = make_states(case, seed=VAL_SEED)
    val["purpose"] = "T13 v4 validation states for checkpoint monitoring only (not the frozen test states)"
    test_keys = {key(s) for r in (pm2, pm8) for s in r["states"] if s["kind"] != "nominal"}
    vs = [s for s in val["states"] if s["kind"] != "nominal"]
    dup = [s["state_id"] for s in vs if key(s) in test_keys]
    if dup or len(vs) != 50:
        raise SystemExit(f"{case}: validation duplicates {dup} / count {len(vs)}")
    files, rows = {}, []
    for name, rec, extra in (("pm2", pm2, dict(seed=TEST_SEED)), ("pm8", pm8, dict(seed=TEST_SEED, scale_from_pm2=4.0,
                              method="exact scaling of the +/-2 uniform draws (as the V3 extended evaluation pm8 files)")),
                             ("validation", val, dict(seed=VAL_SEED, duplicates_with_test_states=0))):
        old = OLD[case][name]
        d = None if old is None else max_value_diff(rec, json.loads(Path(old).read_text()))
        rec["system1_t13v4"] = dict(block, **extra, old_file=None if old is None else str(old), old_file_sha256=None if old is None else sha(old),
                                    max_abs_value_difference_vs_old=d)
        p = OUT / f"{case}_{name}_states.json"
        p.write_text(json.dumps(rec, indent=1) + "\n")
        files[name] = dict(path=str(p.relative_to(ROOT)), sha256=sha(p), states=len(rec["states"]), max_abs_value_difference_vs_old=d)
    return files, dict(pm2=pm2, pm8=pm8, validation=val)


# ---------------------------------------------------------------------------------------------- frame-0 geometry check
def frame0(case, runtime):
    """nominal roots [2, 13] (xyz, quat xyzw, zero velocities), joints [2, 29] (runtime order), object [7] (xyz, quat wxyz)."""
    z = np.load(runtime, allow_pickle=False)
    if case == "table":
        bi = list(z["body_names"]).index("pelvis")
        pos, qw = z["agent_body_pos_w"][0, :, bi], z["agent_body_quat_w"][0, :, bi]
        dof, jn = z["agent_joint_pos"][0], [str(n) for n in z["joint_names"]]
        obj = np.concatenate([z["object_pos_w"][0], z["object_quat_w"][0]])
    else:
        from scipy.spatial.transform import Rotation
        jp = z["joint_pos"][0]
        oq = z["object_quat_w"][0]
        xl = Rotation.from_quat(oq[[1, 2, 3, 0]]).apply([1.0, 0.0, 0.0])
        pos = np.stack([jp[:3] - 0.5 * PUSH_SPACING_M * xl, jp[:3] + 0.5 * PUSH_SPACING_M * xl])
        qw = np.stack([jp[3:7], jp[3:7]])
        dof, jn = np.stack([jp[7:], jp[7:]]), [str(n) for n in z["joint_names"]]
        obj = np.concatenate([z["object_pos_w"][0], oq])
    root = np.zeros((2, 13))
    root[:, :3], root[:, 3:7] = pos, qw[:, [1, 2, 3, 0]]
    return root, dof, jn, obj


def check(case, recs):
    import contact_stats as cs
    import dual_stretch as ds
    import eval_core4d_hulls as eh
    robot = cs.Robot("physx")
    meshes = eh.dump_object_meshes("table", 1.0) if case == "table" else eh.urdf_box_meshes(ds.TABLE["push"][0])
    ev = eh.ObjectRobotDistance(meshes, robot, depth="exact")
    pc = ds.PairCollision(robot, depth="exact")
    res = {}
    new = ROOT / "src/holosoma" / json.loads(MANIFEST[case].read_text())["files"]["runtime_reference" if case == "table" else "motion"]["path"]
    for which, runtime in (("new", new),
                           ("old", RUNTIME_OLD[case])):
        root, dof, jn, obj = frame0(case, runtime)
        perm = [jn.index(n) for n in robot.dofs]
        for name, rec in recs.items():
            pairs = []
            for s in rec["states"]:
                r = yaw_transform(root, s["robot_xy_m"], s["robot_yaw_rad"])
                q = np.zeros((2, 36))
                q[:, :3], q[:, 3:7], q[:, 7:] = r[:, :3], r[:, [6, 3, 4, 5]], dof[:, perm]
                pairs.append(q)
            pairs = np.stack(pairs)
            objq = np.repeat(obj[None], len(pairs), 0)
            H, P = ev.frames(pairs, objq)
            rr = pc.min_distance(pairs)
            res[f"{which}_{name}"] = dict(
                states=len(pairs), runtime=str(Path(runtime).relative_to(ROOT)), runtime_sha256=sha(runtime),
                max_robot_object_penetration_cm={f"r{i}": round(100 * float(P[:, i].max()), 3) for i in (0, 1)},
                states_with_robot_object_penetration={f"r{i}": int((P[:, i] > 0).sum()) for i in (0, 1)},
                hand_signed_distance_max_cm={f"r{i}": round(100 * float(H[:, i].max()), 3) for i in (0, 1)},
                robot_robot_min_distance_cm=round(100 * float(rr.min()), 3), states_robot_robot_intersect=int((rr < 0).sum()),
                nominal=dict(penetration_cm=[round(100 * float(x), 3) for x in P[0]], robot_robot_cm=round(100 * float(rr[0]), 3)))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for case in ("table", "push"):
        files, recs = generate(case)
        summary[case] = dict(files=files, frame0_check=check(case, recs))
        print(case, json.dumps(summary[case], indent=1), flush=True)
    (OUT / "summary.json").write_text(json.dumps(dict(rules=__doc__, **summary), indent=1) + "\n")


if __name__ == "__main__":
    main()
