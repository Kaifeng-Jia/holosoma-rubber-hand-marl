#!/usr/bin/env python3
"""T14 step 8 rerun: old solves (step 8, person-side frame = own object) vs new solves (step 8 rerun, --human-frame target),
both judged with the round-6 support windows (step 9 re-judge rule). Per candidate: verdict, failed items, the numbers behind
them, the robot-person pelvis distance (mean, per role) and the frame flag of step8_frame_check. Per sequence: applicable
candidates and the existing selection rule among them (smallest joint weighted fit error, ties by ID; run_step8.py).

    python3 system1/t14/objects/step8b_compare.py --old results_system1/T14_retargeting/step9/step8_rejudge.csv \
        --new results_system1/T14_retargeting/step8b/step8b_rejudge.csv --out results_system1/T14_retargeting/step8b
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TAGS = ("smallboxb", "largeboxb", "suitcaseb", "smallboxc", "largeboxc", "suitcasec", "push", "pull")


def load(p):
    return {(r["sequence"], r["candidate"]): r for r in csv.DictReader(open(p))}


def ev(logdir, tag, c):
    f = ROOT / logdir / f"cands/{tag}/{c}/m1p00/eval.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    frame = {(r["sequence"], r["candidate"]): r for r in csv.DictReader(open(ROOT / "results_system1/T14_retargeting/step9c/step8_frame_check.csv"))}
    rows, summ = [], {}
    for tag in TAGS:
        cs = json.loads((ROOT / f"logs_system1/t14/step8/cands/{tag}/candidates.json").read_text())
        fit = {f"c{c['index']:02d}": c for c in cs["candidates"]}
        keys = sorted({k for k in list(old) + list(new) if k[0] == tag}, key=lambda k: k[1])
        for k in keys:
            o, n = old.get(k), new.get(k)
            row = dict(sequence=tag, candidate=k[1], id=fit[k[1]]["id"], fit_rmse_m=fit[k[1]]["fit_rmse_m"],
                       frames_rotated_in_old_solve=(frame.get(k) or {}).get("affected"),
                       old_applicable=None if o is None else o["new_applicable"], old_failed=None if o is None else o["new_failed"],
                       new_applicable=None if n is None else n["new_applicable"], new_failed="未求解完" if n is None else n["new_failed"])
            for lab, logdir in (("old", "logs_system1/t14/step8"), ("new", "logs_system1/t14/step8b")):
                e = ev(logdir, tag, k[1])
                src = o if lab == "old" else n
                if e is None:
                    continue
                v = e["V0"]
                row.update({f"{lab}_drift_r0_m": None if src is None else src["new_support_drift_r0_m"], f"{lab}_drift_r1_m": None if src is None else src["new_support_drift_r1_m"],
                            f"{lab}_nonhand_r0_m": v["nonhand_penetration_r0"]["value_m"], f"{lab}_nonhand_r1_m": v["nonhand_penetration_r1"]["value_m"],
                            f"{lab}_rr_proxy_m": v["robot_robot_clearance"]["value_m"], f"{lab}_rr_min_cm": e["robot_robot_min_distance_cm"],
                            f"{lab}_retention_r0": e["contact_retention"]["r0"], f"{lab}_retention_r1": e["contact_retention"]["r1"],
                            f"{lab}_max_pen_r1_cm": e["max_penetration_cm"]["r1"]})
                for r in (0, 1):
                    rec = json.loads((ROOT / logdir / f"cands/{tag}/{k[1]}/m1p00/r{r}/run_record.json").read_text())
                    z = np.load(ROOT / logdir / f"cands/{tag}/{k[1]}/m1p00/r{r}/{rec['tag']}_new_qpos.npz")
                    row[f"{lab}_robot_person_pelvis_r{r}_m"] = round(float(np.linalg.norm(z["qpos"][:, :2] - z["human_joints"][:, 0, :2], axis=1).mean()), 3)
            rows.append(row)

        def pick(lab):
            ok = [r for r in rows if r["sequence"] == tag and r[f"{lab}_applicable"] in ("True", True)]
            sel = sorted(ok, key=lambda r: (round(float(r["fit_rmse_m"]) / 1e-10), r["id"]))[0]["candidate"] if ok else None
            return [r["candidate"] for r in ok], sel
        oa, osel = pick("old")
        na, nsel = pick("new")
        summ[tag] = dict(solved_old=sum(1 for r in rows if r["sequence"] == tag and r["old_applicable"] is not None),
                         solved_new=sum(1 for r in rows if r["sequence"] == tag and r["new_applicable"] is not None),
                         applicable_old=oa, selected_old=osel, applicable_new=na, selected_new=nsel, selection_changed=osel != nsel)
    a.out.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out / "step8_old_vs_new.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (a.out / "step8_old_vs_new.json").write_text(json.dumps(dict(rules=__doc__, summary=summ), indent=1, ensure_ascii=False) + "\n")
    for t, s in summ.items():
        print(t, s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
