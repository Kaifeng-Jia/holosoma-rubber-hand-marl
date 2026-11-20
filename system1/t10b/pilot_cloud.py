#!/usr/bin/env python3
"""T10b B4: pilot retraining with domain randomisation on cloud GPUs (Kevin provides them), one run per GPU, with the
preregistered checkpoint rule. Run on the cloud machine after `source /root/autodl-tmp/system1/s1_env.sh` (S1_PY).

Runs (task book T10b B4):
  d_table_seed721_rand    paper D Table command + --system1-randomization, up to 20000 iterations, checkpoint every 1000;
  cin_bucket_seed721_rand T03 C-in Bin command + --system1-randomization, 12000 iterations, checkpoint every 1000.
Validation (checkpoint monitoring only): every checkpoint at a multiple of 2000 is evaluated in Isaac on the validation states
(system1/t10b/validation_states/<task>_validation_states.json, never the test states), deterministic actions, protocol of the
laptop-5070 finals (scripts/evaluate_paper_supplement.py), on the run's own GPU, while training continues.
Stopping rule (D Table only): after iteration 8000, if the validation completion count has not exceeded the best earlier count
for 4000 consecutive iterations (two validation points), the training process is stopped (by its recorded PID) and the last
saved checkpoint is final; otherwise training runs to 20000. C-in Bin always runs 12000. Test states and MuJoCo results are
never read here. Records: <out>/<run>/launch.json, validation/<it>/ (evaluator output), validation_curve.json, decision.json.

    $S1_PY -B system1/t10b/pilot_cloud.py --out /root/autodl-tmp/system1/logs/t10b_pilot --gpus 0,1 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")
COMMON = ["--reward-variant", "bucket_A_no_rel_no_height", "--num-envs", "2048", "--steps-per-env", "24", "--seed", "721", "--save-interval", "1000",
          "--system1-randomization"]
RUNS = dict(
    d_table_seed721_rand=dict(task="table", controller="policy", max_it=20000, stop_rule=True,
                              train=["--experiment", "smalltable5kg_A", "--iterations", "20000"]),
    cin_bucket_seed721_rand=dict(task="bucket", controller="sonic_residual", max_it=12000, stop_rule=False,
                                 train=["--experiment", "bucket003", "--iterations", "12000", "--sonic-residual", "--sonic-tf32",
                                        "--sonic-residual-kind", "in", "--c-train-diag"]),
)
VAL_EVERY, STOP_AFTER, PATIENCE = 2000, 8000, 4000


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def completed_count(summary):
    s = json.loads(summary.read_text())
    return sum(1 for e in s["episodes"] if e["kind"] != "nominal" and e["completed_reference"])


def stop_decision(curve):
    """curve: {iteration: completed}. Stop when, after STOP_AFTER, no validation point in the last PATIENCE iterations beat the
    best value seen before that window."""
    its = sorted(curve)
    last = its[-1]
    if last - PATIENCE < STOP_AFTER:  # the 4000-iteration window lies entirely after 8000: earliest stop at 12000 (conservative reading)
        return False
    before = [curve[i] for i in its if i <= last - PATIENCE]
    window = [curve[i] for i in its if last - PATIENCE < i <= last]
    return bool(before) and len(window) >= PATIENCE // VAL_EVERY and max(window) <= max(before)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--python", default=os.environ.get("S1_PY"))
    ap.add_argument("--poll", type=float, default=120.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.python or not Path(a.python).exists():
        raise SystemExit(f"--python / S1_PY not set or missing: {a.python!r} (source s1_env.sh)")
    gpus = a.gpus.split(",")
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    state = {}
    for (name, r), gpu in zip(RUNS.items(), gpus):
        d = a.out / name
        argv = [a.python, "-B", "scripts/train_core4d_smalltable.py", *r["train"], *COMMON, "--output-dir", str(d / "train")]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, LOCAL_RANK=str(400 + int(gpu)), WORLD_SIZE="1")
        if a.dry_run:
            print(name, "gpu", gpu, " ".join(argv))
            continue
        d.mkdir(parents=True, exist_ok=False)
        log = (d / "train.stdout.log").open("w")
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        rec = dict(name=name, argv=argv, gpu=gpu, pid=p.pid, started=now(), git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
                   git_diff=git("diff"), env=dict(CUDA_VISIBLE_DEVICES=gpu, LOCAL_RANK=env["LOCAL_RANK"]), rules=dict(val_every=VAL_EVERY, stop_after=STOP_AFTER,
                   patience=PATIENCE, stop_rule=r["stop_rule"], max_it=r["max_it"]))
        (d / "launch.json").write_text(json.dumps(rec, indent=1))
        state[name] = dict(r, dir=d, proc=p, log=log, env=env, curve={}, val_proc=None, val_it=None, stopped=None)
        print(now(), "start", name, "gpu", gpu, "pid", p.pid, flush=True)
        time.sleep(30)
    if a.dry_run:
        return 0
    def pending(s):  # 2000-multiple checkpoints written but not yet validated
        ready = [int(p.stem.split("_")[1]) for p in (s["dir"] / "train").glob("model_*.pt")]
        return [i for i in sorted(ready) if i % VAL_EVERY == 0 and i > 0 and i not in s["curve"]]

    while any(s["proc"].poll() is None or s["val_proc"] is not None or pending(s) for s in state.values()):
        for name, s in state.items():
            d = s["dir"]
            vp = s["val_proc"]
            if vp is not None and vp.poll() is not None:  # a validation finished
                summ = d / "validation" / f"it{s['val_it']:05d}" / "summary.json"
                s["curve"][s["val_it"]] = completed_count(summ) if summ.exists() else None
                (d / "validation_curve.json").write_text(json.dumps(dict(sorted(s["curve"].items())), indent=1))
                print(now(), name, "validation", s["val_it"], s["curve"][s["val_it"]], flush=True)
                s["val_proc"] = None
                curve = {k: v for k, v in s["curve"].items() if v is not None}
                if s["stop_rule"] and s["stopped"] is None and curve and stop_decision(curve) and s["proc"].poll() is None:
                    s["proc"].send_signal(signal.SIGINT)
                    s["stopped"] = dict(time=now(), at_validation_iteration=s["val_it"], curve=dict(sorted(curve.items())), rule="no improvement in 4000 its after 8000")
                    print(now(), name, "stop rule fired", s["stopped"], flush=True)
            if s["val_proc"] is None:
                todo = pending(s)
                if todo:
                    it = todo[0]
                    out = d / "validation" / f"it{it:05d}"
                    out.parent.mkdir(exist_ok=True)
                    argv = [a.python, "-B", "scripts/evaluate_paper_supplement.py", "--case", s["task"], "--controller", s["controller"],
                            "--checkpoint", str(d / "train" / f"model_{it:05d}.pt"), "--device", "cuda:0",
                            "--states", str(ROOT / "system1/t10b/validation_states" / f"{s['task']}_validation_states.json"),
                            "--model-id", f"t10b_validation__{name}__it{it:05d}", "--output-dir", str(out)]
                    venv = dict(s["env"], LOCAL_RANK=str(int(s["env"]["LOCAL_RANK"]) + 50))
                    vlog = (d / "validation" / f"it{it:05d}.stdout.log").open("w")
                    s["val_proc"] = subprocess.Popen(argv, cwd=ROOT, env=venv, stdout=vlog, stderr=subprocess.STDOUT)
                    s["val_it"] = it
                    (d / "validation" / f"it{it:05d}.launch.json").write_text(json.dumps(dict(argv=argv, pid=s["val_proc"].pid, started=now()), indent=1))
        time.sleep(a.poll)
    for name, s in state.items():
        s["log"].close()
        ck = sorted((s["dir"] / "train").glob("model_*.pt"))
        dec = dict(name=name, train_returncode=s["proc"].returncode, stopped_by_rule=s["stopped"], final_checkpoint=str(ck[-1]) if ck else None,
                   validation_curve=dict(sorted(s["curve"].items())), ended=now())
        (s["dir"] / "decision.json").write_text(json.dumps(dec, indent=1))
        print(now(), "final", json.dumps(dec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
