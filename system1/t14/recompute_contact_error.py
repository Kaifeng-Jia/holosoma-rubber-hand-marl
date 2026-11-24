#!/usr/bin/env python3
"""Recompute the contact-position error with the 3-D definition (contact_error3d.py) for every reference reported so far:
round 2, round 3, round 3 + step 2, round 4 A / B, the current references and OmniRetarget as is (1.0x).
Writes <out>/contact_error_3d.csv (per reference x robot x hand) and <out>/contact_error_3d_summary.csv (per reference:
means over the robots' hands).

    sonic_s1/bin/python -B system1/t14/recompute_contact_error.py --rounds round2 round3 round3+step2 comparisons round4A round4B --out <dir>
    [--tasks push pull]   (round 6: the OMOMO rows of every round again, with the person's target on the capture table)
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
sys.path.insert(0, str(ROOT / "system1/t14"))
import contact_error3d as ce  # noqa: E402

REF = ROOT / "exports_system1/T14_references"
WBT = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
PAIR = dict(table="20231030_001_desk001_move_{}_fps30.npz", bin="bin_{}_fps30.npz", chair="chair_{}_fps30.npz")
KS = ("0.8", "0.9", "1.0", "1.1", "1.2")
CUR = dict(table=WBT / "core4d_smalltable/20231030_001_desk001_move_compact_fps30.npz", bin=WBT / "core4d_bucket003_20231020_071_a1/core4d_pair_compact_fps30.npz",
           chair=WBT / "core4d_chair021_20231020_074_a1/core4d_pair_compact_fps30.npz",
           push=WBT / "rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz", pull=WBT / "rubber_hand_largetable_v1/pull/sub3_largetable_010_pull_a1_mj_fps50_w_obj.npz")


def refs(round_):
    """[(task, level, kind, path, size, fps50)]; kind core4d = pair npz, single = single-robot qpos."""
    out = []
    kk = lambda k: "k" + k.replace(".", "p")
    if round_ == "round2":
        out += [(t, k, "core4d", REF / f"{t}_{kk(k)}" / PAIR[t].format("new"), float(k), False) for t in PAIR for k in KS]
        for t, a in (("push", 0.70), ("pull", 0.55)):
            out += [(t, m, "single", ROOT / f"logs_system1/t14/single/{t}/{t}_new_qpos.npz", m, False) for m in (a, 1.0, 1.5, 2.0)]
    elif round_ in ("round3", "round3+step2"):
        tag = "" if round_ == "round3" else "_a10"
        out += [(t, k, "core4d", REF / f"r3/{t}_{kk(k)}{tag}" / PAIR[t].format("new"), float(k), False) for t in PAIR for k in KS]
        if round_ == "round3":
            a = json.loads((ROOT / "results_system1/T14_retargeting/step3/a_levels.json").read_text())["round3"]
            for t in ("push", "pull"):
                out += [(t, m, "single", ROOT / f"logs_system1/t14/r3/single_{t}/{t}_new_qpos.npz", m, False) for m in (a[t], 1.0, 1.5, 2.0)]
    elif round_ in ("round4A", "round4B", "round5C", "round6C", "round6Cp"):
        V, rn = round_[6:], round_[5]
        out += [(t, k, "core4d", REF / f"r{rn}/{t}_{kk(k)}_{V}" / PAIR[t].format("new"), float(k), False) for t in PAIR for k in KS]
        for t in ("push", "pull"):
            ev = ROOT / f"results_system1/T14_retargeting/step{rn}/{t}/eval_r{rn}_{V}.json"
            a = json.loads(ev.read_text())["a"] if ev.exists() else None
            out += [(t, m, "single", ROOT / f"logs_system1/t14/r{rn}/single_{t}_{V}/{t}_new_qpos.npz", m, False) for m in ([a] if a else []) + [1.0, 1.5, 2.0]]
    elif round_ == "comparisons":
        for t in PAIR:
            out += [(t, "current", "core4d", CUR[t], 1.0, False), (t, "omni_asis", "core4d", REF / f"r3/asis_{t}_k1p0" / PAIR[t].format("omni_asis"), 1.0, False)]
        for t in ("push", "pull"):
            out += [(t, "current", "single", CUR[t], 1.0, True), (t, "omni_asis", "single", ROOT / f"logs_system1/t14/r3/single_asis_{t}/{t}_omni_asis_qpos.npz", 1.0, False)]
    return [x for x in out if Path(x[3]).exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=None, help="only these tasks (default all); existing rows of other tasks are kept")
    a = ap.parse_args()
    tasks = set(a.tasks) if a.tasks else None
    import contact_gap as cg
    import dual_stretch as ds
    robot = cg.cs.Robot("physx")
    offs = cg.palm_offsets()
    human = {}
    rows, summ = [], []
    for rd in a.rounds:
        for task, level, kind, path, size, fps50 in refs(rd):
            if tasks is not None and task not in tasks:
                continue
            if task not in human:
                human[task] = ce.human_omomo(task) if task in ("push", "pull") else ce.human_core4d(task)
            if kind == "core4d":
                z = np.load(path)
                res = ce.core4d(task, robot, offs, human[task], z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64), size)
            else:
                q = cg.omomo_single(str(path), fps50)
                pair, obj_q = ds.make_pair(task, q, size)
                res = ce.omomo(task, robot, offs, human[task], pair, obj_q, size)
            sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            lv = level if isinstance(level, str) and not level[0].isdigit() else (f"k = {level}" if kind == "core4d" else f"m = {float(level):.2f}")
            for key, v in res.items():
                rows.append(dict(round=rd, task=task, level=lv, robot=key.split("_")[0], hand=key.split("_")[1], **v, path=str(Path(path).relative_to(ROOT)), sha256=sha))
            ok = [v for v in res.values() if v.get("frames")]
            mean = lambda k: round(float(np.mean([v[k] for v in ok])), 2) if ok else None
            summ.append(dict(round=rd, task=task, level=lv, dist_mean_cm=mean("dist_mean_cm"), tangential_mean_cm=mean("tangential_mean_cm"),
                             normal_signed_mean_cm=mean("normal_signed_mean_cm"), normal_abs_mean_cm=mean("normal_abs_mean_cm"),
                             dist_p95_max_cm=max(v["dist_p95_cm"] for v in ok) if ok else None, path=str(Path(path).relative_to(ROOT))))
            print(rd, task, lv, json.dumps(summ[-1]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("contact_error_3d.csv", rows), ("contact_error_3d_summary.csv", summ)):
        p = a.out / name
        old = list(csv.DictReader(open(p))) if p.exists() else []
        old = [r for r in old if not (r["round"] in a.rounds and (tasks is None or r["task"] in tasks))]  # replace the rounds x tasks recomputed now
        keys = []
        for d in old + data:
            keys += [k for k in d if k not in keys]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(old + data)
    (a.out / "contact_error_3d_rules.txt").write_text(ce.__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
