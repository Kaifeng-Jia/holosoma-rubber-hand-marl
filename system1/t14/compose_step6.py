#!/usr/bin/env python3
"""T14 round 6 package tables (results_system1/T14_retargeting/step6/): round 6 C and C' next to round 5 C, round 4 A / B and the
round-3 baseline (rows copied from step5/levels.csv and hand_metrics.csv, round 5 C relabelled "round5C"); layout as step 4:
  levels.csv         task x level x version (A, B, round3 = round 3 with step 2 where it was run): retention, max penetration,
                     robot-robot distance, success, contact-position error, and hand metrics averaged over the robots' hands
                     (mean of the per-hand means; the per-hand values are in hand_metrics.csv);
  hand_metrics.csv   task x level x version x robot x hand: palm-normal / finger / long-axis angle to the person, palm-surface
                     angle (mean, p95, frames), wrist near-limit fraction (any, per joint), wrist speed peak;
  comparisons.csv    current reference and OmniRetarget as is (1.0x), same columns as levels.csv;
  push_legs.csv      per link: robot min distance (r0, r1) and human part min distance, versions A / B / round3; flags in json.
    python3 system1/t14/compose_step4.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_system1/T14_retargeting/step6"
S = ROOT / "logs_system1/t14/r6"
STEP4 = ROOT / "results_system1/T14_retargeting/step5"
TASKS = ("table", "bin", "chair", "push", "pull")
HM = ("palm_normal_angle", "finger_angle", "long_axis_angle", "palm_surface_angle")


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def level_name(name):
    if "_k" in name:
        return "k = " + name.split("_k")[-1]
    if "_m" in name:
        return f"m = {float(name.split('_m')[-1]):.2f}"
    return "1.0"


def row(task, level, version, r, note=""):
    d = dict(task=task, level=level, version=version, note=note)
    if r is None:
        return d
    g = r.get("contact_position_error", {})
    d.update(success=r.get("finite"), retention_pooled=r["contact_retention"]["pooled"], retention_r0=r["contact_retention"]["r0"], retention_r1=r["contact_retention"]["r1"],
             max_penetration_cm=max(r["max_penetration_cm"].values()), robot_robot_min_distance_cm=r["robot_robot_min_distance_cm"],
             robot_robot_intersect_frames=r["robot_robot_intersect_frames"],
             contact_error_mean_r0_cm=(g.get("r0") or {}).get("mean_cm"), contact_error_mean_r1_cm=(g.get("r1") or {}).get("mean_cm"))
    h = r.get("hand")
    if h:
        for m in HM:
            v = [x[m]["mean_deg"] for x in h.values() if x[m]["mean_deg"] is not None]
            p = [x[m]["p95_deg"] for x in h.values() if x[m]["p95_deg"] is not None]
            d[f"{m}_mean_deg"] = round(float(np.mean(v)), 1) if v else None
            d[f"{m}_p95_max_deg"] = round(float(np.max(p)), 1) if p else None
        d["wrist_near_limit_any_max"] = max(x["near_limit_any"] for x in h.values())
        d["wrist_speed_peak_max_dps"] = max(x["speed_peak_dps"] for x in h.values())
    d.update(pair=r.get("pair") or r.get("path"), pair_sha256=r.get("pair_sha256") or r.get("sha256"))
    return d


def hand_rows(task, level, version, r):
    out = []
    for key, x in (r.get("hand") or {}).items():
        rr, side = key.split("_")
        d = dict(task=task, level=level, version=version, robot=rr, hand=side)
        for m in HM:
            d.update({f"{m}_mean_deg": x[m]["mean_deg"], f"{m}_p95_deg": x[m]["p95_deg"], f"{m}_frames": x[m]["frames"]})
        d.update(wrist_near_limit_any=x["near_limit_any"], **{f"wrist_near_limit_{k}": v for k, v in x["near_limit_per_joint"].items()},
                 wrist_speed_peak_dps=x["speed_peak_dps"], wrist_speed_peak_joint=x["speed_peak_joint"], wrist_speed_peak_frame=x["speed_peak_frame"])
        out.append(d)
    return out


def write(name, data):
    keys = []
    for d in data:
        keys += [k for k in d if k not in keys]
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(data)
    print(name, len(data))


def main():
    rows, hands, comp, a_levels = [], [], [], {}
    for t in TASKS:
        for v in ("C", "Cp"):
            ev = load(OUT / t / f"eval_r6_{v}.json")
            if ev is None:
                continue
            if t in ("push", "pull"):
                a_levels[f"{t}_{v}"] = ev["a"]
            names = sorted(ev["refs"], key=lambda n: float(n.split("_k" if "_k" in n else "_m")[-1]))
            for n in names:
                r = ev["refs"][n]
                rows.append(row(t, level_name(n), v, r, note=("a" if t in ("push", "pull") and float(n.split("_m")[-1]) == ev["a"] else "")))
                hands += hand_rows(t, level_name(n), v, r)
            if t in ("table", "bin", "chair"):  # levels without a reference (a solve did not finish)
                done = {level_name(n) for n in names}
                for k in ("0.8", "0.9", "1.0", "1.1", "1.2"):
                    if f"k = {k}" not in done:
                        rows.append(dict(task=t, level=f"k = {k}", version=v, success=False, note="a solve did not finish"))
    OUT.mkdir(parents=True, exist_ok=True)
    relabel = lambda r: dict(r, version="round5C") if r["version"] == "C" else r
    rows = [relabel(r) for r in csv.DictReader(open(STEP4 / "levels.csv"))] + [dict(r, version="round6" + r["version"]) for r in rows]
    hands = [relabel(r) for r in csv.DictReader(open(STEP4 / "hand_metrics.csv"))] + [dict(r, version="round6" + r["version"]) for r in hands]
    comp = [r for r in csv.DictReader(open(STEP4 / "comparisons.csv"))]
    a_levels = json.loads((STEP4 / "a_levels.json").read_text()) | {k.replace("_C", "_round6C"): v for k, v in a_levels.items()}
    write("levels.csv", rows)
    write("hand_metrics.csv", hands)
    write("comparisons.csv", comp)
    (OUT / "a_levels.json").write_text(json.dumps(a_levels, indent=1) + "\n")
    legs = []
    for v, src in (("A", ROOT / "results_system1/T14_retargeting/step4"), ("B", ROOT / "results_system1/T14_retargeting/step4"),
                   ("round3", ROOT / "results_system1/T14_retargeting/step4"), ("C", STEP4), ("r6C", OUT), ("r6Cp", OUT)):
        j = load(src / "push_legs" / f"push_legs_{v}.json")
        if j:
            for L in j["links"]:
                legs.append(dict(version={"C": "round5C", "r6C": "round6C", "r6Cp": "round6Cp"}.get(v, v), link=L["link"], human_part=L["human_part"], robot_min_r0_cm=L["robot_min_cm"].get("r0"), robot_min_r1_cm=L["robot_min_cm"].get("r1"),
                                 robot_min_frame_r0=L["robot_min_frame"].get("r0"), human_min_cm=L["human_min_cm"], human_min_frame=L["human_min_frame"]))
    if legs:
        write("push_legs.csv", legs)
    rec = OUT / "records"
    rec.mkdir(exist_ok=True)
    for rr in sorted(S.glob("*/person*/run_record.json")) + sorted(S.glob("single*/run_record.json")):
        shutil.copy(rr, rec / ("solve__" + "__".join(rr.relative_to(S).parts[:-1]) + ".json"))
    for cr in sorted((ROOT / "exports_system1/T14_references/r4").glob("*/combine_record.json")):
        shutil.copy(cr, rec / f"combine__{cr.parent.name}.json")
    for f in [S / "pipeline.json", S / "jobs.txt", S / "run.log"] + sorted((S / "levels").glob("*.json")):
        if f.exists():
            shutil.copy(f, rec / (f"levels__{f.name}" if f.parent.name == "levels" else f.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
