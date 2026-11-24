#!/usr/bin/env python3
"""T14 round 3 package tables (results_system1/T14_retargeting/step3/):
  levels.csv                  one row per task x level x variant: round 2 as reported (step2_sizes/levels.csv), round 2
                              re-measured with the exact measure, round 3 (cause fixes), round 3 + step 2 (where run);
  comparisons.csv             current reference and OmniRetarget as is (1.0x), same measure;
  contact_position_error.csv  per reference x robot x hand: frames, mean, p95 (cm), report only.

    python3 system1/t14/compose_step3.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_system1/T14_retargeting/step3"
S = ROOT / "logs_system1/t14/r3"
TASKS = ("table", "bin", "chair", "push", "pull")
KS = ("0.8", "0.9", "1.0", "1.1", "1.2")


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def row(task, level, variant, r, success=None, note=""):
    if r is None:
        return dict(task=task, level=level, variant=variant, success=success, note=note)
    g = r.get("contact_position_error", {})
    d = dict(task=task, level=level, variant=variant, success=success if success is not None else r.get("finite"),
             retention_pooled=r["contact_retention"]["pooled"], retention_r0=r["contact_retention"]["r0"], retention_r1=r["contact_retention"]["r1"],
             max_penetration_cm=max(r["max_penetration_cm"].values()), max_penetration_r0_cm=r["max_penetration_cm"]["r0"], max_penetration_r1_cm=r["max_penetration_cm"]["r1"],
             frames_penetrating_r0=r["frames_penetrating"]["r0"], frames_penetrating_r1=r["frames_penetrating"]["r1"],
             robot_robot_min_distance_cm=r["robot_robot_min_distance_cm"], robot_robot_intersect_frames=r["robot_robot_intersect_frames"],
             contact_error_mean_r0_cm=(g.get("r0") or {}).get("mean_cm"), contact_error_mean_r1_cm=(g.get("r1") or {}).get("mean_cm"),
             contact_error_p95_r0_cm=(g.get("r0") or {}).get("p95_cm"), contact_error_p95_r1_cm=(g.get("r1") or {}).get("p95_cm"),
             pair=r.get("pair") or r.get("path"), pair_sha256=r.get("pair_sha256") or r.get("sha256"), note=note)
    return d


def solve_ok(task, kk, tag):
    rp = [S / f"{task}_{kk}{tag}/person{p}/run_record.json" for p in (1, 2)]
    if not all(x.exists() for x in rp):
        return False
    return all((lambda x: x["finite"] and x["object_locked_max_err"] == 0.0)(json.loads(x.read_text())) for x in rp)


def main():
    r2 = {(r["task"], r["size"]): r for r in csv.DictReader(open(ROOT / "results_system1/T14_retargeting/step2_sizes/levels.csv"))}
    rows, comp, gaps = [], [], []
    for t in TASKS:
        ev = load(OUT / t / "eval_r3.json")
        ev2 = load(OUT / t / "eval_r3_a10.json")
        if ev is None:
            continue
        refs = ev["refs"]
        if t in ("table", "bin", "chair"):
            for k in KS:
                kk = "k" + k.replace(".", "p")
                o = r2.get((t, k))
                rows.append(dict(task=t, level=f"k = {k}", variant="round2_as_reported", success=None if o is None else o["solve_success"],
                                 retention_pooled=None if o is None else o["contact_retention"], max_penetration_cm=None if o is None else o["max_penetration_convex_cm"],
                                 robot_robot_min_distance_cm=None if o is None else o["robot_robot_min_distance_cm"], note="" if o is None else (o["failed"] or "")))
                rows.append(row(t, f"k = {k}", "round2_exact_measure", refs.get(f"r2_k{k}"), note="" if f"r2_k{k}" in refs else "round 2 had no reference"))
                rows.append(row(t, f"k = {k}", "round3", refs.get(f"r3_k{k}"), success=solve_ok(t, kk, ""), note="" if f"r3_k{k}" in refs else "solve did not finish"))
                if ev2 and f"r3_a10_k{k}" in ev2["refs"]:
                    rows.append(row(t, f"k = {k}", "round3_step2", ev2["refs"][f"r3_a10_k{k}"], success=solve_ok(t, kk, "_a10")))
                elif (S / f"{t}_{kk}_a10").exists():
                    rows.append(row(t, f"k = {k}", "round3_step2", None, success=False, note="step-2 solve did not finish"))
        else:
            for name, r in refs.items():
                if name.startswith("r2_m"):
                    m = float(name[4:])
                    o = r2.get((t, f"{m:g}" if m != int(m) else f"{m:.1f}")) or r2.get((t, str(m)))
                    rows.append(dict(task=t, level=f"m = {m:.2f}", variant="round2_as_reported", success=None if o is None else o["solve_success"],
                                     retention_pooled=None if o is None else o["contact_retention"], max_penetration_cm=None if o is None else o["max_penetration_convex_cm"],
                                     robot_robot_min_distance_cm=None if o is None else o["robot_robot_min_distance_cm"]))
                    rows.append(row(t, f"m = {m:.2f}", "round2_exact_measure", r))
            for name, r in refs.items():
                if name.startswith("r3_m"):
                    rows.append(row(t, f"m = {float(name[4:]):.2f}", "round3", r, note=("a (round 3)" if float(name[4:]) == ev["a"] else "")))
            if ev2:
                for name, r in ev2["refs"].items():
                    if name.startswith("r3_a10_m"):
                        rows.append(row(t, f"m = {float(name[8:]):.2f}", "round3_step2", r, note=("a (round 3 + step 2)" if float(name[8:]) == ev2["a"] else "")))
        for name in ("current", "omni_asis"):
            if name in refs:
                comp.append(row(t, "1.0", name, refs[name]))
        for e in (ev, ev2):
            if e is None:
                continue
            for name, r in e["refs"].items():
                for key, v in r["contact_position_error"].items():
                    if "_" in key:
                        gaps.append(dict(task=t, ref=name, robot=key.split("_")[0], hand=key.split("_")[1], frames=v["frames"], mean_cm=v["mean_cm"], p95_cm=v["p95_cm"]))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("levels.csv", rows), ("comparisons.csv", comp), ("contact_position_error.csv", gaps)):
        keys = []
        for d in data:
            keys += [k for k in d if k not in keys]
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
        print(name, len(data))
    a = {t: (load(OUT / t / "eval_r3.json") or {}).get("a") for t in ("push", "pull")}
    a2 = {t: (load(OUT / t / "eval_r3_a10.json") or {}).get("a") for t in ("push", "pull")}
    (OUT / "a_levels.json").write_text(json.dumps(dict(round3=a, round3_step2=a2, round2=dict(push=0.70, pull=0.55)), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
