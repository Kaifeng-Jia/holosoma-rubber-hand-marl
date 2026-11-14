#!/usr/bin/env python3
"""Kevin 2026-11-13 23:0x: final evaluation of the D paper main result on the laptop RTX 5070 for Push, then Kick and Chair
(seeds 721-723, +/-2 and +/-8), with the final-evaluation protocol of the other D finals here (system1/t03/post_train.py
d_curve / d_paper_final jobs): scripts/evaluate_paper_supplement.py --controller policy, final checkpoint, frozen states,
deterministic, TF32 off (evaluator default), --device cuda:0; Push also --push-placement-proof (as every Push evaluation).

Checkpoints and states are the ones of the V3 closeout (paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/
results/episodes.csv, variant Full): each checkpoint is found on this machine by its sha256 and every states file is
checked against the V3 state_file_sha256 before anything starts. Two evaluations at a time (taskset 16-23 / 24-31, the
pipeline's two slots; the pipeline has no job of its own now). Records per evaluation next to its output directory:
<dir>.launch.json (argv, sha256 of checkpoint / states / proof, git HEAD / status / diff, GPU) and <dir>.exit.json.

    hssim_s1/bin/python -B system1/t03/run_d_finals_local.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PY = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
OUT = ROOT / "logs_system1/post_train/d_final_5070"
V3 = Path("/home/kevin/holosoma-core4d-base/paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/results/episodes.csv")
S8 = "/home/kevin/holosoma-core4d-base/logs/V3_20261102/cloud_backup_20261104/raw/v3_extended_eval_20261104_ops/states"
STATES = {("push", 2): "/home/kevin/holosoma-core4d-base/logs/ExperimentsR2_evaluation_20261026_retry01/U5_push_registered/push_states.json",
          ("push", 8): f"{S8}/push_pm8.json",
          ("kick", 2): "/home/kevin/holosoma-core4d-base/logs/ExperimentsR2_evaluation_20261026_retry01/U6_kick/kick_states.json",
          ("kick", 8): f"{S8}/kick_pm8.json",
          ("chair", 2): "/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003/chair_states.json",
          ("chair", 8): f"{S8}/chair_pm8.json"}
CKPT = {("push", 721): "/home/kevin/holosoma-core4d-base/logs/ExperimentsR2_retry1/push_seed721_stage2_from8050_plus5000/model_13050.pt",
        ("push", 722): "/home/kevin/holosoma-core4d-base/logs/ExperimentsR2_retry1/push_seed722_stage2_from8050_plus5000/model_13050.pt",
        ("push", 723): "/home/kevin/holosoma-core4d-base/logs/ExperimentsR2_retry1/push_seed723_stage2_from8050_plus5000/model_13050.pt",
        ("kick", 721): "/home/kevin/holosoma-core4d-base/logs/ExperimentsR2/deployment_20261024/evaluation_20261026/cloud_ops/historical_kick721/model_08000.pt",
        ("kick", 722): "/home/kevin/holosoma-core4d-base/logs/ExperimentsR2_retry1/kick_seed722_stage0_from0_plus8000/model_08000.pt",
        ("kick", 723): "/home/kevin/holosoma-core4d-base/logs/ExperimentsR2_retry1/kick_seed723_stage0_from0_plus8000/model_08000.pt",
        ("chair", 721): "/home/kevin/holosoma-core4d-base/logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/model_12000.pt",
        ("chair", 722): "/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs/C01_seed722_fresh12000_concurrent_20260930/model_12000.pt",
        ("chair", 723): "/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs/C01_seed723_fresh12000_concurrent_20260930/model_12000.pt"}
PROOF = ROOT / "system1/evidence/preflight_push_registered.json"
SLOTS = ("16-23", "24-31")
NY = ZoneInfo("America/New_York")


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def v3_records():
    out = {}
    with V3.open() as f:
        for r in csv.DictReader(f):
            if r["task"] in ("push", "kick", "chair") and r["variant"] == "Full" and r["perturbation_level_cm"] in ("2", "8"):
                out[(r["task"], int(r["training_seed"]), int(r["perturbation_level_cm"]))] = (r["run_id"], r["checkpoint_sha256"], int(r["checkpoint_iteration"]),
                                                                                            r["state_file_sha256"])
    return out


def jobs():
    v3 = v3_records()
    out = []
    for task in ("push", "kick", "chair"):
        for seed in (721, 722, 723):
            for lvl in (2, 8):
                run_id, ck_sha, it, st_sha = v3[(task, seed, lvl)]
                ck, st = CKPT[(task, seed)], STATES[(task, lvl)]
                if sha(ck) != ck_sha:
                    raise SystemExit(f"{task} {seed}: checkpoint sha256 differs from V3 {run_id}")
                if sha(st) != st_sha:
                    raise SystemExit(f"{task} +/-{lvl}: states sha256 differs from V3")
                label = f"D_{task}_paper_seed{seed}"
                name = f"{label}/it{it:05d}_pm{lvl}"
                argv = [PY, "-B", "scripts/evaluate_paper_supplement.py", "--case", task, "--controller", "policy", "--checkpoint", ck,
                        "--device", "cuda:0", "--states", st, "--model-id", f"d_final_5070__{label}__it{it:05d}_pm{lvl}",
                        "--output-dir", str(OUT / name)]
                if task == "push":
                    argv += ["--push-placement-proof", str(PROOF)]
                out.append(dict(name=name, task=task, seed=seed, level=lvl, iteration=it, v3_run_id=run_id, checkpoint=ck, checkpoint_sha256=ck_sha,
                                states=st, states_sha256=st_sha, argv=argv))
    return out


def complete(d):
    try:
        return bool(json.loads((d / "summary.json").read_text()).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    todo = [j for j in jobs() if not complete(OUT / j["name"])]
    print(now(), f"{len(todo)} evaluations to run", flush=True)
    if a.dry_run:
        for j in todo:
            print(j["name"], " ".join(j["argv"][2:]))
        return 0
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    running = {}
    while todo or running:
        for slot in SLOTS:
            if slot in running or not todo:
                continue
            j = todo.pop(0)
            d = OUT / j["name"]
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists():
                k = 1
                while d.with_name(f"{d.name}.attempt{k}").exists():
                    k += 1
                d.rename(d.with_name(f"{d.name}.attempt{k}"))
            argv = ["taskset", "-c", slot, *j["argv"]]
            rec = dict({k: v for k, v in j.items() if k != "argv"}, argv=argv, started=now(), gpu_model=gpu,
                       push_placement_proof_sha256=sha(PROOF) if j["task"] == "push" else None, git_head=git("rev-parse", "HEAD").strip(),
                       git_status=git("status", "--porcelain"), git_diff=git("diff"), task_book="Kevin 2026-11-13 (D finals on the 5070 for Table 1)")
            lj = d.with_name(d.name + ".launch.json")
            lj.write_text(json.dumps(rec, indent=1))
            log = d.with_name(d.name + ".stdout.log").open("w")
            p = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            lj.write_text(json.dumps(dict(rec, pid=p.pid), indent=1))
            running[slot] = (j, p, log)
            print(now(), "start", j["name"], "slot", slot, "pid", p.pid, flush=True)
            time.sleep(20)
        for slot, (j, p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                d = OUT / j["name"]
                ok = complete(d)
                d.with_name(d.name + ".exit.json").write_text(json.dumps(dict(name=j["name"], returncode=p.returncode, completed=ok, ended=now()), indent=1))
                print(now(), "end", j["name"], "rc", p.returncode, "complete", ok, flush=True)
                del running[slot]
        time.sleep(30)
    print(now(), "all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
