#!/usr/bin/env python3
"""T03 addition to the cross-GPU table: the D paper main result (Pull it8050, Table it12000, Bin it12000, seeds 721-723,
final checkpoints) as evaluated for the V3 closeout and as re-evaluated on this laptop now, per state.

Two blocks, both read from existing outputs (CPU only):
  * +/-2, same GPU, different code: V3 = the 2026-10-04 unified evaluation (PaperSupplement20260929/unified_evaluation_20261003,
    this laptop's RTX 5070, env hssim, evaluator of /home/kevin/holosoma-core4d-base); now = the pipeline's d_curve finals
    (logs_system1/post_train/d_curve, RTX 5070, env hssim_s1, this branch's evaluator with every System1 switch off);
  * +/-8, different GPU and code: V3 = the 2026-11-04 cloud extended evaluation (evaluator sha256 56c6450c..., the GPU model is
    not in its logs; the same cloud backup's U5 render outputs record "NVIDIA GeForce RTX 4090"); now = d_paper_final
    (RTX 5070, this branch).
V3 per-state values come from the V3 closeout table (V3_GPU_CLOSEOUT_20261104/results/episodes.csv: completed_reference,
executed_steps), with checkpoint sha256 and states-file sha256 checked against the laptop summaries. Per pair over the 50
perturbed states: completed on each side, flips, McNemar exact two-sided p, identical executed steps; nominal on its own.

    hssim_s1/bin/python -B system1/t03/v3_vs_laptop_compare.py --out results_system1/T03/c2_checks/xgpu
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from system1.t03.xgpu_compare import mcnemar_exact  # noqa: E402

V3 = Path("/home/kevin/holosoma-core4d-base/paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/results/episodes.csv")
PT = ROOT / "logs_system1/post_train"
RUNS = [("pull", "Pull-baseline", "D_pull_paper", 8050), ("table", "T-07", "D_table_paper_T-07", 12000),
        ("bucket", "B-RZ0", "D_bucket_paper_B-RZ0", 12000)]
BLOCK = {2: ("同卡（5070）不同代码", "", "d_curve"), 8: ("跨卡（V3 4090 / 本机 5070）且不同代码", "_Full_pm8", "d_paper_final")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    v3 = {}
    for r in csv.DictReader(V3.open()):
        v3.setdefault(r["model_id"], {})[r["state_id"]] = r
    rows, summ = [], []
    for level, (block, suffix, sub) in BLOCK.items():
        for task, cfg, label, it in RUNS:
            for seed in (721, 722, 723):
                mid, run = f"{cfg}_seed{seed}{suffix}", f"{label}_seed{seed}"
                d = PT / sub / run / f"it{it:05d}_pm{level}"
                base = dict(level=level, block=block, task=task, seed=seed, v3_model_id=mid, laptop_dir=str(d.relative_to(ROOT)))
                s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
                if not s or not s.get("complete") or s.get("episode_count") != 51:
                    summ.append(dict(base, status="未完成", reason="本机结果未完成"))
                    continue
                vv = v3[mid]
                ck = {r["checkpoint_sha256"] for r in vv.values()}
                st = {r["state_file_sha256"] for r in vv.values()}
                assert ck == {s["checkpoint_sha256"]} and st == {s["states_sha256"]}, (mid, ck, st)
                assert {r["checkpoint_iteration"] for r in vv.values()} == {str(it)}
                el = {e["state_id"]: e for e in s["episodes"]}
                assert set(el) == set(vv), mid
                rr = []
                for k in sorted(el):
                    x, y = el[k], vv[k]
                    yc = y["completed_reference"] == "True"
                    rr.append(dict(base, state_id=k, nominal=y["is_nominal"] == "True", v3_completed=yc, laptop_completed=x["completed_reference"],
                                   flip=yc != x["completed_reference"], v3_steps=int(y["executed_steps"]), laptop_steps=x["steps"],
                                   same_steps=int(y["executed_steps"]) == x["steps"]))
                rows += rr
                pr = [r for r in rr if not r["nominal"]]
                nom = next(r for r in rr if r["nominal"])
                b = sum(r["v3_completed"] and not r["laptop_completed"] for r in pr)
                c = sum(r["laptop_completed"] and not r["v3_completed"] for r in pr)
                ev = {r["evaluation_code_sha256"] for r in vv.values()}
                summ.append(dict(base, status="完成", reason="", n=len(pr), checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"],
                                 v3_evaluation_code_sha256="|".join(sorted(x for x in ev if x)) or "未记录",
                                 laptop_gpu=json.loads((d / "run_manifest.json").read_text()).get("gpu_name"),
                                 laptop_summary_sha256=sha(d / "summary.json"),
                                 v3_completed=sum(r["v3_completed"] for r in pr), laptop_completed=sum(r["laptop_completed"] for r in pr),
                                 both_completed=sum(r["v3_completed"] and r["laptop_completed"] for r in pr), v3_only=b, laptop_only=c,
                                 neither=sum(not r["v3_completed"] and not r["laptop_completed"] for r in pr), flips=b + c,
                                 mcnemar_exact_p=mcnemar_exact(b, c), same_executed_steps=sum(r["same_steps"] for r in pr),
                                 nominal_v3_completed=nom["v3_completed"], nominal_laptop_completed=nom["laptop_completed"],
                                 nominal_v3_steps=nom["v3_steps"], nominal_laptop_steps=nom["laptop_steps"]))
    keys = []
    for s in summ:
        keys += [k for k in s if k not in keys]
    for name, rr, fields in (("v3_vs_laptop_per_state.csv", rows, None), ("v3_vs_laptop_summary.csv", summ, keys)):
        with (a.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields or list(rr[0]), restval="")
            w.writeheader()
            w.writerows(rr)
    for s in summ:
        print(s["level"], s["task"], s["seed"], s["status"], *(s.get(k) for k in ("v3_completed", "laptop_completed", "v3_only", "laptop_only",
                                                                                  "mcnemar_exact_p", "same_executed_steps")))


if __name__ == "__main__":
    raise SystemExit(main())
