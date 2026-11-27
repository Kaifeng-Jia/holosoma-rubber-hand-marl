#!/usr/bin/env python3
"""T14 step 9g, second fix, item 1 (Kevin 2026-11-26, received 10:29): re-judge the two-robot clearance with the actual collision shapes,
as the reported "two-robot minimum distance" already is (shared_eval / step9_eval: dual_stretch.PairCollision depth="exact" =
the 23 PhysX link shapes of both robots, every cross-robot link pair, mj_geomDistance); the cross-robot OBB proxy
(reference_geometry_round2_checks.CollisionProxy, robot_robot_clearance_m) is no longer used. Limit unchanged: >= -0.01 m.
Only the verdict is recomputed: the exact minimum distance is the one stored in each eval.json (robot_robot_min_distance_cm,
computed on the same frames as the proxy); no solve and no geometry is re-run. Read only.

Three verdicts side by side per evaluated level / candidate:
  A  non-hand penetration on the per-link boxes, clearance on the OBB proxies (the verdict of the run itself; step8b: the
     round-6 re-judge it reported);
  B  non-hand penetration on the collision shapes (step 9g item 1, rejudge_nonhand_exact.py tables), clearance on the proxies;
  C  both on the collision shapes (this script): B with "V0 robot-robot clearance" decided by the exact minimum distance.
Every other item as it was. step8b: Push / Pull selection rule as step8b_compare.py (applicable candidates, smallest joint fit
error rounded to 1e-10, ties by ID), under A, B and C.

    python3 system1/t14/objects/rejudge_clearance_exact.py --set step9g --nonhand <rejudge_nonhand_exact csv> --logdir logs_system1/t14/step9g \
        --out <dir> [--name <stem>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CL = "V0 robot-robot clearance"
ORDER = ["V0 gravity", "V0 support drift r0", "V0 support drift r1", "V0 non-hand penetration r0", "V0 non-hand penetration r1",
         CL, "section-4 contact retention >= 0.99", "main motion"]
LIMIT_M = -0.01


def split(s):
    return [x for x in (s or "").split(" | ") if x]


def with_clearance(failed, passed):
    f = [x for x in failed if x != CL] + ([] if passed else [CL])
    return sorted(set(f), key=lambda x: ORDER.index(x) if x in ORDER else 99)


def eval_path(a, row):
    if a.set == "step8b":
        return a.logdir / "cands" / row["sequence"] / row["candidate"] / "m1p00/eval.json"
    return a.logdir / row["clip"] / ("m" + row["level"].replace(".", "p")) / "eval.json"


def table(a):
    rows = []
    for r in csv.DictReader(open(a.nonhand)):
        e = json.loads(eval_path(a, r).read_text())
        fa, fb = split(r["old_failed"]), split(r["new_failed"])
        px = e["V0"]["robot_robot_clearance"]
        assert (CL in fb) == (not px["passed"]) == (CL in fa), (r, px)  # B keeps A's clearance item = the eval's proxy verdict
        ex_m = e["robot_robot_min_distance_cm"] / 100.0
        ok = ex_m >= LIMIT_M - 1e-10
        fc = with_clearance(fb, ok)
        key = dict(sequence=r["sequence"], candidate=r["candidate"], id=r["id"], fit_rmse_m=r["fit_rmse_m"]) if a.set == "step8b" else \
            dict(clip=r["clip"], sequence=r["sequence"], level=r["level"], reported=r["reported"] == "True")
        rows.append(dict(**key, old_proxy_clearance_m=px["value_m"], old_proxy_clearance_passed=px["passed"],
                         new_exact_min_distance_m=round(ex_m, 5), new_exact_clearance_passed=ok, exact_intersect_frames=e["robot_robot_intersect_frames"],
                         clearance_verdict_changed=ok != px["passed"],
                         applicable_A=not fa, failed_A=" | ".join(fa), applicable_B=not fb, failed_B=" | ".join(fb), applicable_C=not fc, failed_C=" | ".join(fc),
                         eval=str(eval_path(a, r).relative_to(ROOT))))
    return rows


def summary(a, rows):
    out = {}
    if a.set == "step8b":
        for seq in dict.fromkeys(r["sequence"] for r in rows):
            rs = [r for r in rows if r["sequence"] == seq]

            def sel(v):
                ok = [r for r in rs if r[f"applicable_{v}"]]
                return sorted(ok, key=lambda r: (round(float(r["fit_rmse_m"]) / 1e-10), r["id"]))[0]["candidate"] if ok else None
            out[seq] = dict(evaluated=len(rs), **{f"applicable_{v}": [r["candidate"] for r in rs if r[f"applicable_{v}"]] for v in "ABC"},
                            **{f"selected_{v}": sel(v) for v in "ABC"})
            out[seq]["selection_changed_B_to_C"] = out[seq]["selected_B"] != out[seq]["selected_C"]
            out[seq]["selection_changed_A_to_C"] = out[seq]["selected_A"] != out[seq]["selected_C"]
        return out
    for clip in dict.fromkeys(r["clip"] for r in rows):
        rr = [r for r in rows if r["clip"] == clip and r["reported"]]
        out[clip] = dict(sequence=rr[0]["sequence"] if rr else None, evaluated_report_levels=[float(r["level"]) for r in rr],
                         **{f"applicable_{v}": [float(r["level"]) for r in rr if r[f"applicable_{v}"]] for v in "ABC"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=("step8b", "step9f", "step9g", "step9g2", "step9_final"), required=True)
    ap.add_argument("--nonhand", type=Path, required=True, help="rejudge_nonhand_exact.py csv of this set (verdicts A and B)")
    ap.add_argument("--logdir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    a.logdir = a.logdir or ROOT / f"logs_system1/t14/{a.set}"
    a.logdir = a.logdir if a.logdir.is_absolute() else ROOT / a.logdir
    rows = table(a)
    summ = summary(a, rows)
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.name or f"{a.set}_clearance_exact"
    with open(a.out / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out / f"{stem}.json").write_text(json.dumps(dict(rules=__doc__, command=" ".join(sys.argv), nonhand_table=str(a.nonhand), summary=summ),
                                                   indent=1, ensure_ascii=False) + "\n")
    n = len(rows)
    print(stem, n, "rows; clearance changed", sum(r["clearance_verdict_changed"] for r in rows),
          "; applicable A/B/C", [sum(r[f"applicable_{v}"] for r in rows) for v in "ABC"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
