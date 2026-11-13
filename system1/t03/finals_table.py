#!/usr/bin/env python3
"""T03 final-checkpoint table on the laptop RTX 5070 (Kevin 2026-11-12 item 4): C v2 main runs next to the D paper main
result of the same task, +/-2 and +/-8, from existing pipeline outputs (CPU only, read-only).

Rows: every C v2 main / extra-seed run whose final/<run>_pm2 and _pm8 are complete, and the D paper finals (d_curve +/-2,
d_paper_final +/-8) of the same task. A missing or incomplete output is written as not completed, with the reason.

    hssim_s1/bin/python -B system1/t03/finals_table.py --out results_system1/T03/c2_finals
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PT = ROOT / "logs_system1/post_train"
C_RUNS = {"pull": [f"{k}_pull_m20kg_seed{s}_joint8000" for k, s in (("cin", 721), ("cout", 721), ("cin", 722), ("cin", 723))],
          "table": [f"{k}_table_m5kg_seed{s}_fresh12000" for k, s in (("cin", 721), ("cout", 721), ("cin", 722), ("cin", 723))],
          "bucket": [f"{k}_bucket_m1kg_seed{s}_fresh12000" for k, s in (("cin", 721), ("cout", 721), ("cin", 722), ("cin", 723))]}
D_RUNS = {"pull": ("D_pull_paper_seed{s}", 8050), "table": ("D_table_paper_T-07_seed{s}", 12000), "bucket": ("D_bucket_paper_B-RZ0_seed{s}", 12000)}
ORIG_MASS = dict(pull=20.0, table=5.0, bucket=1.0)
# T04 per-mass retrained runs (seed 721; Kevin 2026-11-13: finals.csv carries them too, with mass_kg)
MASS_RUNS = {"pull": [f"{k}_pull_m40kg_seed721_joint8000" for k in ("d", "cin")],
             "table": [f"{k}_table_m{m}kg_seed721_fresh12000" for k in ("d", "cin") for m in (10, 20)],
             "bucket": [f"{k}_bucket_m{m}kg_seed721_fresh12000" for k in ("d", "cin") for m in (2, 4, 8)]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summ(d):
    s = d / "summary.json"
    if not s.exists():
        return None, "no summary.json"
    j = json.loads(s.read_text())
    if not j.get("complete") or j.get("episode_count") != 51:
        return None, f"complete={j.get('complete')} episodes={j.get('episode_count')}"
    m = json.loads((d / "run_manifest.json").read_text())
    pr = [e for e in j["episodes"] if e["kind"] != "nominal"]
    nom = next(e for e in j["episodes"] if e["kind"] == "nominal")
    f = [e for e in pr if not e["completed_reference"]]
    terms = Counter(k for e in f for k, v in (e.get("terminal_terms") or {}).items() if v)
    return dict(completed=sum(e["completed_reference"] for e in pr), n=len(pr),
                progress_mean=sum(e["progress"] for e in pr) / len(pr),
                nominal_completed=nom["completed_reference"], nominal_steps=nom["steps"],
                failed_terms=";".join(f"{k}:{v}" for k, v in sorted(terms.items())),
                failed_steps_min=min((e["steps"] for e in f), default=""), failed_steps_max=max((e["steps"] for e in f), default=""),
                sonic_fell_in_failed=sum(bool(e.get("sonic_fell_any")) for e in f),
                gpu=m.get("gpu_name"), checkpoint_sha256=j["checkpoint_sha256"], states_sha256=j["states_sha256"],
                summary_sha256=sha(d / "summary.json"), dir=str(d.relative_to(ROOT))), ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in ("pull", "table", "bucket"):
        for run in C_RUNS[task]:
            for lvl in (2, 8):
                r, why = summ(PT / "final" / f"{run}_pm{lvl}")
                rows.append(dict(task=task, scheme=run.split("_")[0], run=run, seed=int(run.split("seed")[1][:3]), level=lvl,
                                 mass_kg=ORIG_MASS[task], retrained="original mass",
                                 status="完成" if r else "未完成", reason=why, **(r or {})))
        lab, it = D_RUNS[task]
        for s in (721, 722, 723):
            for lvl, sub in ((2, "d_curve"), (8, "d_paper_final")):
                run = lab.format(s=s)
                r, why = summ(PT / sub / run / f"it{it:05d}_pm{lvl}")
                rows.append(dict(task=task, scheme="D", run=run, seed=s, level=lvl, mass_kg=ORIG_MASS[task], retrained="original mass",
                                 status="完成" if r else "未完成", reason=why, **(r or {})))
        for run in MASS_RUNS[task]:
            for lvl in (2, 8):
                r, why = summ(PT / "final" / f"{run}_pm{lvl}")
                rows.append(dict(task=task, scheme={"d": "D", "cin": "cin"}[run.split("_")[0]], run=run, seed=721, level=lvl,
                                 mass_kg=float(run.split("_m")[1].split("kg")[0]), retrained="T04 retrained at this mass",
                                 status="完成" if r else "未完成", reason=why, **(r or {})))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (a.out / "finals.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["task"], r["scheme"], r["seed"], r["level"], r["status"], r.get("completed", ""), r.get("failed_terms", ""), r["reason"])


if __name__ == "__main__":
    raise SystemExit(main())
