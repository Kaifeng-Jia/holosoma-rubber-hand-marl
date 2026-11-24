#!/usr/bin/env python3
"""T13 v4 sections 1 / 3: one training run per cloud GPU, with the in-training Isaac validation (observation v1, noise off)
and the stopping rule. Adapted from system1/t10b/pilot_cloud.py (same validation counting and stop decision).

Runs (common: seed 721, T14 round 6 C reference, observation v1 + noise v1 in training, T10b randomisation, critic fix, the
three contact items; everything else = the paper D recipe of the task):
  d_table     train_core4d_smalltable.py smalltable5kg_A bucket_A_no_rel_no_height
  d_push      train_plan5_push.py --skill push, paper D Push recipe: critic-only warm-up (50 its) from the single-robot prior
              (158 -> 159 inputs, new weights zero), then actor + critic with --resume (lr actor 1e-5, critic 1e-3)
  d_table_pa  d_table + --per-agent-reward (insurance; T13 v4 section 1, launched only when Kevin says so)

Protocol (section 3), 2048 environments: checkpoint every 1000 its; validation every 2000 its on the T13 v4 validation states
(system1/t13v4/states_t14r6C/<task>_validation_states.json; deterministic, randomisation off, noise off, the d_final_5070
protocol of scripts/evaluate_paper_supplement.py); after 8000 its, stop when the validation completion count has not exceeded
the earlier best for 4000 its (earliest after the 12000 validation), else train to 20000; the final checkpoint is the last one
saved when the run stops. With 4096 environments every iteration number is halved (500 / 1000 / 4000 / 2000 / 6000 / 10000,
and the Push warm-up 25), so each checkpoint corresponds to the same amount of data. Iterations of d_push are counted after the
warm-up (checkpoint names carry the warm-up offset). A crash (non-zero exit before the end, not stopped by the rule) is resumed
once from the latest checkpoint; a second crash stops the run (recorded in state.json). Test states are never read here.

    S1_PY=<hssim python> python3 system1/t13v4/cloud_runs.py --run d_table --num-envs 2048 --gpu 0 --out <dir> [--dry-run]
Records: <out>/<run>/launch.json, train*/ (training outputs), validation/itNNNNN/, validation_curve.json, state.json, decision.json.
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
W = "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/system1_t14r6C"
LAB = "system1/t13v4/labels_t14r6C"
STATES = "system1/t13v4/states_t14r6C"
PROOF = "system1/evidence/preflight_push_registered.json"
V1 = ["--obs-v1", "--critic-v1", "--system1-randomization"]
TABLE = ["scripts/train_core4d_smalltable.py", "--experiment", "smalltable5kg_A", "--reward-variant", "bucket_A_no_rel_no_height",
         "--steps-per-env", "24", "--seed", "721", *V1, "--contact-v1", f"{LAB}/table_t14r6C_k1p0_labels.npz", "--t14-reference", f"{W}/table/t14_reference.json"]
PUSH = ["scripts/train_plan5_push.py", "--skill", "push", "--steps-per-env", "24", "--seed", "721", "--actor-learning-rate", "0.00001",
        "--critic-learning-rate", "0.001", "--push-placement-proof", PROOF, *V1, "--contact-v1", f"{LAB}/push_t14r6C_m1p00_labels.npz",
        "--t14-reference", f"{W}/push/t14_reference.json"]
VAL_FLAGS = dict(table=["--obs-v1", "--critic-v1", "--t14-reference", f"{W}/table/t14_reference.json"],
                 push=["--obs-v1", "--critic-v1", "--t14-reference", f"{W}/push/t14_reference.json", "--push-placement-proof", PROOF])
RUNS = dict(d_table=dict(task="table", train=TABLE, warmup=0),
            d_push=dict(task="push", train=PUSH, warmup=50),
            d_table_pa=dict(task="table", train=[*TABLE, "--per-agent-reward"], warmup=0))
BASE = dict(save=1000, val=2000, stop_after=8000, patience=4000, max_it=20000)


def rules(num_envs):
    if num_envs not in (2048, 4096):
        raise SystemExit("--num-envs: 2048 or 4096 (T13 v4 section 4 B)")
    f = 1 if num_envs == 2048 else 2
    r = {k: v // f for k, v in BASE.items()}
    r["earliest_stop"] = r["stop_after"] + r["patience"]
    return r


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def completed_count(summary):
    s = json.loads(Path(summary).read_text())
    if s.get("complete") is not True:
        return None
    return sum(1 for e in s["episodes"] if e["kind"] != "nominal" and e["completed_reference"])


def stop_decision(curve, R):
    """curve {iteration: completed}: stop when, after stop_after, no validation point in the last `patience` iterations beat the
    best value seen before that window (earliest at stop_after + patience)."""
    its = sorted(curve)
    last = its[-1]
    if last - R["patience"] < R["stop_after"]:
        return False
    before = [curve[i] for i in its if i <= last - R["patience"]]
    window = [curve[i] for i in its if last - R["patience"] < i <= last]
    return bool(before) and len(window) >= R["patience"] // R["val"] and max(window) <= max(before)


def ckpts(d):
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("model_*.pt"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=tuple(RUNS), required=True)
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--python", default=os.environ.get("S1_PY"))
    ap.add_argument("--torch-threads", type=int, default=None, help="passed to the training script (section 4 B test result)")
    ap.add_argument("--poll", type=float, default=120.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.python or not Path(a.python).exists():
        raise SystemExit(f"--python / S1_PY not set or missing: {a.python!r}")
    R, run = rules(a.num_envs), RUNS[a.run]
    warm = run["warmup"] // (1 if a.num_envs == 2048 else 2)
    d = a.out / a.run
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, LOCAL_RANK=str(400 + int(a.gpu)), WORLD_SIZE="1")
    extra = ["--num-envs", str(a.num_envs), "--save-interval", str(R["save"])] + ([] if a.torch_threads is None else ["--torch-threads", str(a.torch_threads)])
    stages = []
    if warm:
        stages.append(dict(name="train_warmup", argv=[a.python, "-B", *run["train"], *extra, "--iterations", str(warm), "--critic-only",
                                                      "--output-dir", str(d / "train_warmup")]))
    main_dir = d / "train"
    main_argv = [a.python, "-B", *run["train"], *extra, "--iterations", str(R["max_it"]), "--output-dir", str(main_dir)]
    if warm:
        main_argv += ["--resume", str(d / "train_warmup" / f"model_{warm:05d}.pt")]
    stages.append(dict(name="train", argv=main_argv))
    counting = dict(decision="Kevin 2026-11-24: training iterations are counted after the critic-only warm-up; the same rule for all three runs",
                    warmup_iterations=warm, counted_iteration="checkpoint file number - warmup_iterations",
                    applies_to="validation points, stop rule, maximum iterations, final checkpoint")
    plan = dict(run=a.run, task=run["task"], num_envs=a.num_envs, gpu=a.gpu, rules=R, iteration_counting=counting, warmup_iterations=warm, stages=stages,
                validation_states=f"{STATES}/{run['task']}_validation_states.json", validation_flags=VAL_FLAGS[run["task"]])
    if a.dry_run:
        print(json.dumps(plan, indent=1))
        return 0
    d.mkdir(parents=True, exist_ok=False)
    (d / "launch.json").write_text(json.dumps(dict(plan, started=now(), git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
                                                   git_diff=git("diff"), python=a.python), indent=1))
    state = dict(curve={}, stopped=None, crashes=[], resumed=None, status="running")
    save = lambda: (d / "state.json").write_text(json.dumps(state, indent=1, default=str))  # noqa: E731
    if warm:  # warm-up: one short stage, no validation
        log = (d / "train_warmup.stdout.log").open("w")
        rc = subprocess.run(stages[0]["argv"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        log.close()
        state["warmup_returncode"] = rc
        if rc != 0 or not (d / "train_warmup" / f"model_{warm:05d}.pt").exists():
            state["status"] = "warmup failed"
            save()
            return 1
    log = (d / "train.stdout.log").open("w")
    proc = subprocess.Popen(main_argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    state.update(train_pid=proc.pid, train_started=now())
    save()
    vproc = vit = None

    def pending():
        return [i for i in ckpts(main_dir) if (i - warm) % R["val"] == 0 and i > warm and (i - warm) not in state["curve"]]

    while proc is not None and proc.poll() is None or vproc is not None or pending():
        if vproc is not None and vproc.poll() is not None:
            summ = d / "validation" / f"it{vit:05d}" / "summary.json"
            state["curve"][vit] = completed_count(summ) if summ.exists() else None
            (d / "validation_curve.json").write_text(json.dumps(dict(sorted(state["curve"].items())), indent=1))
            vproc = None
            curve = {k: v for k, v in state["curve"].items() if v is not None}
            if state["stopped"] is None and curve and stop_decision(curve, R) and proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                state["stopped"] = dict(time=now(), at_validation_iteration=vit, curve=dict(sorted(curve.items())))
            save()
        if vproc is None and pending():
            it = pending()[0]
            out = d / "validation" / f"it{it - warm:05d}"
            out.parent.mkdir(exist_ok=True)
            argv = [a.python, "-B", "scripts/evaluate_paper_supplement.py", "--case", run["task"], "--controller", "policy", *VAL_FLAGS[run["task"]],
                    "--checkpoint", str(main_dir / f"model_{it:05d}.pt"), "--device", "cuda:0", "--states", plan["validation_states"],
                    "--model-id", f"t13v4_validation__{a.run}__it{it - warm:05d}", "--output-dir", str(out)]
            vlog = (d / "validation" / f"it{it - warm:05d}.stdout.log").open("w")
            vproc = subprocess.Popen(argv, cwd=ROOT, env=dict(env, LOCAL_RANK=str(int(env["LOCAL_RANK"]) + 50)), stdout=vlog, stderr=subprocess.STDOUT)
            vit = it - warm
            (d / "validation" / f"it{vit:05d}.launch.json").write_text(json.dumps(dict(argv=argv, pid=vproc.pid, started=now()), indent=1))
        if proc is not None and proc.poll() is not None and state["stopped"] is None:
            done = ckpts(main_dir)
            finished = proc.returncode == 0 and done and done[-1] == warm + R["max_it"]
            if not finished:
                state["crashes"].append(dict(time=now(), returncode=proc.returncode, last_checkpoint=done[-1] if done else None))
                if len(state["crashes"]) == 1 and done:  # one resume from the latest checkpoint
                    rdir = d / "train_resume1"
                    argv = [x for x in main_argv]
                    if "--resume" in argv:
                        i = argv.index("--resume")
                        del argv[i:i + 2]
                    argv[argv.index("--iterations") + 1] = str(warm + R["max_it"] - done[-1])
                    argv[argv.index("--output-dir") + 1] = str(rdir)
                    argv += ["--resume", str(main_dir / f"model_{done[-1]:05d}.pt")]
                    log.close()
                    log = (d / "train_resume1.stdout.log").open("w")
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    main_dir = rdir  # later checkpoints (same iteration numbering) come from the resumed run
                    state["resumed"] = dict(time=now(), argv=argv, pid=proc.pid)
                else:
                    state["status"] = "crashed twice: stopped"
                    proc = None
            save()
        time.sleep(a.poll)
    log.close()
    allck = sorted(set(ckpts(d / "train")) | set(ckpts(d / "train_resume1") if (d / "train_resume1").exists() else set()))
    state["status"] = state["status"] if state["status"] != "running" else ("stopped by rule" if state["stopped"] else "finished")
    state["final_checkpoint_iteration"] = allck[-1] if allck else None
    state["final_checkpoint_counted_iteration"] = allck[-1] - warm if allck else None
    state["ended"] = now()
    save()
    (d / "decision.json").write_text(json.dumps(dict(run=a.run, status=state["status"], stopped_by_rule=state["stopped"],
                                                     final_checkpoint_iteration=state["final_checkpoint_iteration"],
                                                     final_checkpoint_counted_iteration=state["final_checkpoint_counted_iteration"], iteration_counting=counting,
                                                     validation_curve=dict(sorted(state["curve"].items())), ended=state["ended"]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
