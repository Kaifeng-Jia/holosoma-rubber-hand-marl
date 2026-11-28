#!/usr/bin/env python3
"""T09 short version (docs/system1/T09_ablation_5070_zh.md, last section, Kevin 2026-11-28): re-evaluate three existing
final-checkpoint groups on the laptop RTX 5070, +/-2 and +/-8, with the protocol of results_system1/T03/d_final_5070 (frozen
states, deterministic actor, TF32 off = evaluator default, final checkpoint, 50 perturbed states + 1 nominal per level, nominal
not counted): scripts/evaluate_paper_supplement.py --controller policy --device cuda:0. No training, nothing else changed.

Order (task book): 1 Push prescribed placement (the draft-4 ablation group = Push-livePD, seeds 721 / 722 / 723, iteration
13050; checkpoints and the +/-2 states as recorded by the unified evaluation logs/PaperSupplement20260929/
unified_evaluation_20261003 (models.json, Push-livePD_seed*.exit.json); evaluated as then WITHOUT --push-placement-proof =
the original loader, prescribed 0.80 m spacing; +/-8 = push_pm8.json a58b2fae..., as the task book), 2 Pull from-scratch
(V3-25 / 26 / 27), 3 Pull zero-teammate-obs (V3-19 / 20 / 21) (V3 closeout results/episodes.csv: checkpoint and state-file
sha256 per seed and level; the ablation contract is in the checkpoint (v3_ablation), the V3 command had no extra flag).
Every checkpoint and states file is checked against its record before anything starts (mismatch -> that group is not run).

At most 2 evaluations at a time (taskset 16-23 / 24-31; the two levels of a seed together), starts 90 s apart (URDF -> USD
cache race of simultaneous Isaac starts); a launch waits while MemAvailable < 15 GB or while a GPU compute process that is
not one of these evaluations runs (the cloud queue's local post-run evaluations come first). Records next to each output
directory: <dir>.launch.json (argv, sha256 of checkpoint / states, record source, git HEAD / status / diff, GPU) and
<dir>.exit.json; an evaluation counts only with summary.json complete and 51 episodes.

    hssim_s1/bin/python -B system1/t09/run_t09_short.py [--dry-run]
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
OUT = ROOT / "logs_system1/t09_short"
V3 = Path("/home/kevin/holosoma-core4d-base/paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/results/episodes.csv")
UE = Path("/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003")
S8 = "/home/kevin/holosoma-core4d-base/logs/V3_20261102/cloud_backup_20261104/raw/v3_extended_eval_20261104_ops/states"
PULL_CK = "/home/kevin/holosoma-core4d-base/logs/V3_20261102/cloud_backup_20261104/raw/v3_20261103_pull/logs/V3_20261103_pull_retry01"
STATES = {("push", 2): str(UE / "push_states.json"), ("push", 8): f"{S8}/push_pm8.json",
          ("pull", 2): str(UE / "pull_states.json"), ("pull", 8): f"{S8}/pull_pm8.json"}
PUSH_PM8_SHA256 = "a58b2fae20f4"  # task book prefix of push_pm8.json (the D Push +/-8 file of d_final_5070)
PULL_RUNS = {("from-scratch", 721): "V3-25", ("from-scratch", 722): "V3-26", ("from-scratch", 723): "V3-27",
             ("zero-teammate-obs", 721): "V3-19", ("zero-teammate-obs", 722): "V3-20", ("zero-teammate-obs", 723): "V3-21"}
PULL_DIR = {"from-scratch": "scratch", "zero-teammate-obs": "zero"}
SLOTS = ("16-23", "24-31")
MEM_GB = 15
NY = ZoneInfo("America/New_York")


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def avail_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) / 1048576
    return 0.0


def push_records():
    """Push-livePD (prescribed placement): unified evaluation models.json + the per-seed exit records (states file used)."""
    models = {m["model_id"]: m for m in json.loads((UE / "models.json").read_text()) if m.get("configuration") == "Push-livePD"}
    out = {}
    for seed in (721, 722, 723):
        m = models[f"Push-livePD_seed{seed}"]
        ex = json.loads((UE / f"Push-livePD_seed{seed}.exit.json").read_text())
        st = ex["argv"][ex["argv"].index("--states") + 1]
        assert ex["checkpoint_sha256"] == m["checkpoint_sha256"] and ex["argv"][ex["argv"].index("--checkpoint") + 1] == m["checkpoint"], seed
        assert "--push-placement-proof" not in ex["argv"], "the recorded prescribed evaluation had no placement proof"
        out[seed] = dict(checkpoint=m["checkpoint"], checkpoint_sha256=m["checkpoint_sha256"], iteration=int(Path(m["checkpoint"]).stem.split("_")[1]),
                         states_pm2_recorded=st, states_pm2_sha256=sha(st), record=dict(models_json=str(UE / "models.json"),
                                                                                         exit_json=str(UE / f"Push-livePD_seed{seed}.exit.json"),
                                                                                         models_json_sha256=sha(UE / "models.json")))
    return out


def v3_pull_records():
    out = {}
    with V3.open() as f:
        for r in csv.DictReader(f):
            if r["task"] == "pull" and r["variant"] in ("from-scratch", "zero-teammate-obs") and r["perturbation_level_cm"] in ("2", "8"):
                out[(r["variant"], int(r["training_seed"]), int(r["perturbation_level_cm"]))] = (r["run_id"], r["checkpoint_sha256"], int(r["checkpoint_iteration"]),
                                                                                                r["state_file_sha256"])
    return out


def jobs():
    """all jobs in task-book order; a group whose checkpoint or states do not match its record is reported and left out."""
    out, refused = [], []
    pr = push_records()
    for seed in (721, 722, 723):
        r = pr[seed]
        for lvl in (2, 8):
            ck, st = r["checkpoint"], STATES[("push", lvl)]
            ok_ck = sha(ck) == r["checkpoint_sha256"]
            st_sha = sha(st)
            ok_st = st_sha == r["states_pm2_sha256"] if lvl == 2 else st_sha.startswith(PUSH_PM8_SHA256)
            label = f"push_prescribed_seed{seed}"
            j = dict(group="push prescribed placement", task="push", variant="prescribed-placement", seed=seed, level=lvl, iteration=r["iteration"],
                     record_run_id=f"Push-livePD_seed{seed}", record_source="unified_evaluation_20261003 models.json / exit.json" + ("" if lvl == 2 else
                     "; +/-8 states: task book (push_pm8.json a58b2fae...)"), record=r["record"], checkpoint=ck, checkpoint_sha256=r["checkpoint_sha256"],
                     states=st, states_sha256=st_sha, name=f"{label}/it{r['iteration']:05d}_pm{lvl}", label=label)
            (out if ok_ck and ok_st else refused).append(dict(j, check=dict(checkpoint=ok_ck, states=ok_st)))
    v3 = v3_pull_records()
    for var in ("from-scratch", "zero-teammate-obs"):
        for seed in (721, 722, 723):
            run = PULL_RUNS[(var, seed)]
            ck = f"{PULL_CK}/{run}_pull_{PULL_DIR[var]}_seed{seed}_joint8000/model_08050.pt"
            for lvl in (2, 8):
                run_id, ck_sha, it, st_sha = v3[(var, seed, lvl)]
                st = STATES[("pull", lvl)]
                ok_ck, ok_st = sha(ck) == ck_sha, sha(st) == st_sha
                label = f"pull_{var.replace('-', '_')}_seed{seed}"
                j = dict(group=f"pull {var}", task="pull", variant=var, seed=seed, level=lvl, iteration=it, record_run_id=run_id,
                         record_source="V3 closeout results/episodes.csv", record=dict(episodes_csv=str(V3), episodes_csv_sha256=sha(V3)),
                         checkpoint=ck, checkpoint_sha256=ck_sha, states=st, states_sha256=st_sha, name=f"{label}/it{it:05d}_pm{lvl}", label=label)
                (out if ok_ck and ok_st else refused).append(dict(j, check=dict(checkpoint=ok_ck, states=ok_st)))
    for j in out:
        j["argv"] = [PY, "-B", "scripts/evaluate_paper_supplement.py", "--case", j["task"], "--controller", "policy", "--checkpoint", j["checkpoint"],
                     "--device", "cuda:0", "--states", j["states"], "--model-id", f"t09_short__{j['label']}__it{j['iteration']:05d}_pm{j['level']}",
                     "--output-dir", str(OUT / j["name"])]
    return out, refused


def complete(d):
    try:
        s = json.loads((d / "summary.json").read_text())
        return bool(s.get("complete")) and len(s.get("episodes", [])) == 51
    except (OSError, json.JSONDecodeError):
        return False


def foreign_gpu(mine):
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()
    kids = set()
    for p in mine:
        kids.add(p)
        try:  # the evaluator's own children
            kids.update(int(x) for x in subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split())
        except ValueError:
            pass
    return [int(x) for x in q if x.strip().isdigit() and int(x) not in kids]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    alljobs, refused = jobs()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(dict(made=now(), jobs=[{k: v for k, v in j.items()} for j in alljobs], refused=refused), indent=1))
    for j in refused:
        print(now(), "REFUSED (record mismatch)", j["name"], j["check"], flush=True)
    todo = [j for j in alljobs if not complete(OUT / j["name"])]
    print(now(), f"{len(todo)} evaluations to run, {len(refused)} refused", flush=True)
    if a.dry_run:
        for j in todo:
            print(j["name"], j["checkpoint_sha256"][:12], j["states_sha256"][:12], " ".join(j["argv"][2:]))
        return 0
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    running, last_start = {}, 0.0
    while todo or running:
        for slot in SLOTS:
            if slot in running or not todo or time.time() - last_start < 90:
                continue
            fg = foreign_gpu([p.pid for _, p, _ in running.values()])
            if fg or avail_gb() < MEM_GB:
                print(now(), "wait", dict(foreign_gpu_pids=fg, mem_available_gb=round(avail_gb(), 1)), flush=True)
                break
            j = todo.pop(0)
            d = OUT / j["name"]
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists():
                k = 1
                while d.with_name(f"{d.name}.attempt{k}").exists():
                    k += 1
                d.rename(d.with_name(f"{d.name}.attempt{k}"))
            argv = ["taskset", "-c", slot, *j["argv"]]
            rec = dict({k: v for k, v in j.items() if k != "argv"}, argv=argv, started=now(), gpu_model=gpu, mem_available_gb=round(avail_gb(), 1),
                       evaluator_sha256=sha(ROOT / "scripts/evaluate_paper_supplement.py"), git_head=git("rev-parse", "HEAD").strip(),
                       git_status=git("status", "--porcelain"), git_diff=git("diff"), task_book="docs/system1/T09_ablation_5070_zh.md short version (Kevin 2026-11-28)",
                       task_book_sha256=sha(ROOT / "docs/system1/T09_ablation_5070_zh.md"))
            lj = d.with_name(d.name + ".launch.json")
            log = d.with_name(d.name + ".stdout.log").open("w")
            p = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            lj.write_text(json.dumps(dict(rec, pid=p.pid), indent=1))
            running[slot] = (j, p, log)
            last_start = time.time()
            print(now(), "start", j["name"], "slot", slot, "pid", p.pid, flush=True)
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
