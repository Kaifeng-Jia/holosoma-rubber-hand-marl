#!/usr/bin/env python3
"""T03 cross-GPU table (Kevin 2026-11-12): the same checkpoint and frozen states, the same evaluator, +/-2, on the laptop
RTX 5070 and on the cloud RTX 4090, per state, for C-in and D in one table.

Pairs (laptop dir, cloud dir):
  * C-in Pull it1050, C-in Bin it1000: laptop logs_system1/post_train/recheck (check 3a), cloud logs_system1/cloud_evals/c2_curve;
  * D paper main result seed 721, Pull it1050 and Bin final it12000: laptop logs_system1/post_train/d_curve, cloud
    logs_system1/cloud_evals/xgpu (copied from the cloud logs/evals/xgpu, launched first among the cloud evaluations).
Per pair over the 50 perturbed states (nominal reported on its own): completed on each GPU, both / laptop only / cloud only /
neither, flips = laptop only + cloud only, McNemar exact two-sided p (binomial on the flips, p = 0.5), states with identical
executed steps. A pair whose cloud or laptop output is missing or incomplete is written as not completed, with the reason.

    hssim_s1/bin/python -B system1/t03/xgpu_compare.py --out results_system1/T03/c2_checks/xgpu
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
L, C = ROOT / "logs_system1/post_train", ROOT / "logs_system1/cloud_evals"
PAIRS = [
    ("C-in", "pull", "cin_pull_m20kg_seed721_joint8000", 1050, L / "recheck", C / "c2_curve"),
    ("C-in", "bucket", "cin_bucket_m1kg_seed721_fresh12000", 1000, L / "recheck", C / "c2_curve"),
    ("D", "pull", "D_pull_paper_seed721", 1050, L / "d_curve", C / "xgpu"),
    ("D", "bucket", "D_bucket_paper_B-RZ0_seed721", 12000, L / "d_curve", C / "xgpu"),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def term_pos(d, e):
    with np.load(d / Path(e["trajectory"]).name, allow_pickle=False) as z:
        return float(np.linalg.norm(z["object_pos_w"][-1] - z["reference_object_pos_w"][-1]))


def evaluator_sha(head):
    """sha256 of scripts/evaluate_paper_supplement.py at a git commit of this branch."""
    return hashlib.sha256(subprocess.run(["git", "show", f"{head}:scripts/evaluate_paper_supplement.py"], cwd=ROOT,
                                         capture_output=True, check=True).stdout).hexdigest()


def flat(d, p=""):
    """run_manifest.json flattened to {"/a/b[0]": value}."""
    if isinstance(d, dict):
        return {k: v for kk, vv in d.items() for k, v in flat(vv, f"{p}/{kk}").items()}
    if isinstance(d, list):
        return {k: v for i, vv in enumerate(d) for k, v in flat(vv, f"{p}[{i}]").items()}
    return {p: d}


def load(d):
    s = d / "summary.json"
    if not s.exists():
        return None, f"{d.relative_to(ROOT)}: no summary.json"
    j = json.loads(s.read_text())
    if not j.get("complete") or j.get("episode_count") != 51:
        return None, f"{d.relative_to(ROOT)}: complete={j.get('complete')} episodes={j.get('episode_count')}"
    return j, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, summ, prov, mdiff = [], [], [], []
    for scheme, task, label, it, lroot, croot in PAIRS:
        dl, dc = lroot / label / f"it{it:05d}_pm2", croot / label / f"it{it:05d}_pm2"
        (sl, wl), (sc, wc) = load(dl), load(dc)
        base = dict(scheme=scheme, task=task, run=label, iteration=it)
        if sl is None or sc is None:
            summ.append(dict(base, status="未完成", reason="; ".join(x for x in (wl, wc) if x)))
            continue
        ml, mc = (json.loads((d / "run_manifest.json").read_text()) for d in (dl, dc))
        assert sl["checkpoint_sha256"] == sc["checkpoint_sha256"] and sl["states_sha256"] == sc["states_sha256"], label
        fl, fc = flat(ml), flat(mc)
        mdiff += [dict(base, field=k, laptop=json.dumps(fl.get(k)), cloud=json.dumps(fc.get(k)))
                  for k in sorted(set(fl) | set(fc)) if fl.get(k) != fc.get(k)]
        ll = json.loads((dl.parent / f"{dl.name}.launch.json").read_text())
        lc = json.loads((C / "ops2" / f"{dc.relative_to(C).as_posix().replace('/', '__')}.launch.json").read_text())
        prov.append(dict(base, checkpoint_sha256=sl["checkpoint_sha256"], states_sha256=sl["states_sha256"],
                         laptop_gpu=ml.get("gpu_name"), cloud_gpu=mc.get("gpu_name"),
                         laptop_git_head=ll["git_head"], cloud_git_head=lc["git_head"],
                         laptop_git_status_clean=not ll.get("git_status"), cloud_git_status_clean=not lc.get("git_status"),
                         laptop_evaluator_sha256=evaluator_sha(ll["git_head"]), cloud_evaluator_sha256=evaluator_sha(lc["git_head"]),
                         laptop_argv=" ".join(ll["argv"]), cloud_argv=" ".join(lc["argv"]),
                         laptop_summary_sha256=sha(dl / "summary.json"), cloud_summary_sha256=sha(dc / "summary.json"),
                         laptop_dir=str(dl.relative_to(ROOT)), cloud_dir=str(dc.relative_to(ROOT))))
        el = {e["state_id"]: e for e in sl["episodes"]}
        ec = {e["state_id"]: e for e in sc["episodes"]}
        assert set(el) == set(ec), label
        rr = []
        for k in sorted(ec):
            x, y = el[k], ec[k]
            rr.append(dict(base, state_id=k, kind=y["kind"],
                           laptop_completed=x["completed_reference"], cloud_completed=y["completed_reference"],
                           flip=x["completed_reference"] != y["completed_reference"],
                           laptop_steps=x["steps"], cloud_steps=y["steps"], same_steps=x["steps"] == y["steps"],
                           laptop_progress=x["progress"], cloud_progress=y["progress"],
                           laptop_object_rmse_3d_m=x["object_rmse_3d_m"], cloud_object_rmse_3d_m=y["object_rmse_3d_m"],
                           laptop_terminal_object_pos_error_m=term_pos(dl, x), cloud_terminal_object_pos_error_m=term_pos(dc, y)))
        rows += rr
        pr = [r for r in rr if r["kind"] != "nominal"]
        nom = next(r for r in rr if r["kind"] == "nominal")
        b = sum(r["laptop_completed"] and not r["cloud_completed"] for r in pr)
        c = sum(r["cloud_completed"] and not r["laptop_completed"] for r in pr)
        summ.append(dict(base, status="完成", reason="", n=len(pr),
                         laptop_gpu=ml.get("gpu_name"), cloud_gpu=mc.get("gpu_name"),
                         laptop_completed=sum(r["laptop_completed"] for r in pr), cloud_completed=sum(r["cloud_completed"] for r in pr),
                         both_completed=sum(r["laptop_completed"] and r["cloud_completed"] for r in pr), laptop_only=b, cloud_only=c,
                         neither=sum(not r["laptop_completed"] and not r["cloud_completed"] for r in pr),
                         flips=b + c, mcnemar_exact_p=mcnemar_exact(b, c),
                         same_executed_steps=sum(r["same_steps"] for r in pr),
                         laptop_progress_mean=float(np.mean([r["laptop_progress"] for r in pr])),
                         cloud_progress_mean=float(np.mean([r["cloud_progress"] for r in pr])),
                         nominal_laptop_completed=nom["laptop_completed"], nominal_cloud_completed=nom["cloud_completed"],
                         nominal_laptop_steps=nom["laptop_steps"], nominal_cloud_steps=nom["cloud_steps"]))
    keys = []
    for s in summ:
        keys += [k for k in s if k not in keys]
    for name, rr, fields in (("per_state.csv", rows, None), ("summary.csv", summ, keys), ("provenance.csv", prov, None),
                             ("manifest_diff.csv", mdiff, None)):
        if not rr:
            continue
        with (a.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields or list(rr[0]), restval="")
            w.writeheader()
            w.writerows(rr)
    for s in summ:
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
