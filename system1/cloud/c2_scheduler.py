#!/usr/bin/env python3
"""T03 C v2 scheduler and cloud slot filler (runs on the cloud).

Rules (T03 "提前停止、质量扫描（事先定）"; Kevin 2026-11-11: act without waiting; keep the 5 GPUs full):

* Iteration 1000, per kind (cin / cout): when all 3 runs of the kind have logged iteration 1000, a run counts as
  progressing if its mean over iterations 901-1000 of episode length OR of reward_mean is higher than its own mean over
  1-100. If none of the 3 progresses, the kind's 3 runs are stopped (system1/cloud/stop_runs.py; data kept).
  Metrics: reward_mean; Episode/mean_length_last100_steps (Pull) or Episode/reset_length_mean_steps (Table / Bin).
  Iteration numbers are the logged ones (Pull: 1-50 critic-only segment, 51.. joint segment).
* Iteration 2000: when every kind not stopped has all 3 runs at iteration 2000: per task the kind with the higher mean
  reward_mean over 1901-2000 wins; the kind winning more tasks does the C mass sweep (pull 40, table 10 / 20,
  bucket 2 / 4 / 8). One kind left: that one. None: no C mass sweep.
* Slot filling: every --poll seconds, a GPU with fewer than --per-gpu compute processes (nvidia-smi) and at least
  --min-free-mib free memory gets one job, in this order:
    1. C mass sweep runs (once chosen; longest expected first); while any is pending no evaluation is started;
    1b. C-in extra seeds 722 / 723 for Pull / Table / Bin (Kevin 2026-11-12: same configuration as seed 721, only the seed;
       after the mass sweep, before every evaluation, longest expected first; an evaluation this scheduler started (PID
       recorded) is stopped to free its slot when a seed run waits and no GPU has room: <evals>/ops2/<name>.preempted.json,
       its output directory renamed <dir>.preempted<k>, re-run later, not counted as a failure); a training needs
       --train-min-free-mib free memory;
    1c. cross-GPU control (Kevin 2026-11-12): D paper seed 721 Pull it1050 and Bin it12000, +/-2 on the 4090 (xgpu/...),
       first among the evaluations; Kevin's choice (2026-11-12 02:5x): once 2 extra-seed runs have started, the next free
       slot runs these two evaluations one after the other before the 3rd seed run; they are never preempted;
    2. evaluations of C v2 checkpoints (every cin_/cout_ run, main and mass, every saved checkpoint, +/-2; the final
       checkpoint of a completed run also +/-8), oldest checkpoint first;
    3. D "completion vs iteration" curve evaluations from $S1_INPUTS/d_curve/catalog.json (V3-25-27 it04050 first,
       then seed 721, then 722 / 723), shared with the laptop pipeline through claims (system1/cloud/claims.py).
  GPUs used by a running run_eval_jobs.py (the first 12 evaluations) are left to it until it exits.
  Kevin 2026-11-12 (cloud slots freed by finished trainings all go to the curve backlog, cleared before shutdown):
    --per-gpu-idle N: a GPU with no training process (and no training launched in the last 5 min) takes up to N processes
       instead of --per-gpu (each evaluation is one environment; --min-free-mib still applies); default = --per-gpu;
    --pm8-last: the cloud +/-8 of a final checkpoint (c2_final/..., also run on the laptop 5070, where the paper table
       comes from) goes after every curve evaluation (C +/-2 and D); off = old order.
  Kevin 2026-11-12 (later): after the last training ends the cloud runs evaluations for at most 2 more hours; what is not
  done by then goes to the laptop, then the cloud is shut down. --post-train-hours H:
    * when no training process or training queue is left and no mass / seed run is pending, the time is recorded
      (c2_decisions.json all_training_done) and the deadline is set H hours later;
    * before the deadline an evaluation is started only if its expected duration (median of the last 5 completed cloud
      evaluations of the same case on a GPU without training, else a fixed default) fits before the deadline;
    * at the deadline: evaluations still running are stopped by their recorded PID (output renamed
      <dir>.stopped_deadline<k>, record <evals>/ops2/<name>.stopped_deadline.json), every evaluation not completed and
      not owned by the laptop is written to <evals>/ops2/transfer_to_laptop.json (checkpoint and states paths with
      sha256, argv), a handover.json is added to the cloud's claim directories, and nothing more is launched.
    Off (default) = old behaviour.
  Kevin 2026-11-12 (later): every point of the training-speed curves, C and D, from the cloud 4090; the paper comparison
  table stays on the laptop 5070. --curves-on-4090:
    * D curve checkpoints (d_curve/...) are evaluated here even when the laptop has claimed (and evaluated) them, before
      every other curve evaluation (priority (0.5, ...)); the claim stays the laptop's, a cloud_rerun.json is added to it;
    * the +/-2 of each finished C v2 run's final checkpoint (c2_curve/<run>/it<final>_pm2, laptop-claimed for the paper
      table) is evaluated here too, as a curve point, at its normal curve priority.
    Off (default) = old behaviour.
  Checkpoints younger than 120 s are not used (the Pull trainer does not write them atomically).
* Status for the laptop watcher every loop: <train>/c2_slots.json (per GPU: processes, free slots; queue lengths).

Every decision with its numbers goes to <train>/c2_decisions.json and c2_scheduler.log; nothing is deleted.

    nohup $S1_PY -B system1/cloud/c2_scheduler.py --train $S1_ROOT/logs/train > $S1_ROOT/logs/train/c2_scheduler.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.cloud.claims import claim_local_fs, safe  # noqa: E402

NY = ZoneInfo("America/New_York")
KINDS = ("cin", "cout")
TASKS = {"pull": "pull:20", "table": "table:5", "bucket": "bucket:1"}
LABEL = {"pull": "{k}_pull_m20kg_seed721", "table": "{k}_table_m5kg_seed721_fresh12000", "bucket": "{k}_bucket_m1kg_seed721_fresh12000"}
LEN_KEY = {"pull": "Episode/mean_length_last100_steps", "table": "Episode/reset_length_mean_steps", "bucket": "Episode/reset_length_mean_steps"}
MASS = [("pull", "pull:40", 8050), ("table", "table:10", 12000), ("table", "table:20", 12000),
        ("bucket", "bucket:2", 12000), ("bucket", "bucket:4", 12000), ("bucket", "bucket:8", 12000)]
ORIG = dict(pull=20.0, table=5.0, bucket=1.0)
S1 = Path("/root/autodl-tmp/system1")
EVALS = S1 / "logs/evals"
STATES = {(c, 2): S1 / f"inputs/states/{c}_states.json" for c in ORIG} | {(c, 8): S1 / f"inputs/states/{c}_pm8.json" for c in ORIG}
D_CAT = S1 / "inputs/d_curve/catalog.json"
SEEDS = [(task, run, its, seed) for task, run, its in (("table", "table:5", 12000), ("bucket", "bucket:1", 12000), ("pull", "pull:20", 8050))
         for seed in (722, 723)]
XGPU = [("D_pull_paper_seed721", "1050"), ("D_bucket_paper_B-RZ0_seed721", "12000")]
XGPU_AFTER_SEEDS = 2


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def metrics(train, kind, task):
    base = LABEL[task].format(k=kind)
    files = [train / f"{base}_critic50" / "metrics.jsonl", train / f"{base}_joint8000" / "metrics.jsonl"] if task == "pull" \
        else [train / base / "metrics.jsonl"]
    rows = {}
    for f in files:
        if f.exists():
            for line in f.read_text().splitlines():
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial last line
                rows[int(j["iteration"])] = j
    return rows


def wmean(rows, lo, hi, key):
    v = [rows[i][key] for i in range(lo, hi + 1) if i in rows and isinstance(rows[i].get(key), (int, float))]
    return (sum(v) / len(v) if v else None), len(v)


def gpu_state():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    gpus, uuid = {}, {}
    for l in q.strip().splitlines():
        i, u, used, tot = [x.strip() for x in l.split(",")]
        gpus[int(i)] = dict(free_mib=int(tot) - int(used), procs=[])
        uuid[u] = int(i)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    for l in apps.strip().splitlines():
        if "," not in l:
            continue
        pid, u = [x.strip() for x in l.split(",")]
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            cmd = "?"
        m = re.search(r"--output-dir\s+(\S+)", cmd)
        what = ("train " if ("train_plan5" in cmd or "train_core4d" in cmd) else "eval " if "evaluate_paper" in cmd else "other ") + \
            (Path(m.group(1)).name if m else cmd[:60])
        gpus[uuid[u]]["procs"].append(dict(pid=int(pid), what=what))
    return gpus


def excluded_gpus():
    """GPU slots held by a running run_eval_jobs.py (the first evaluation list)."""
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    out = set()
    for l in ps.splitlines():
        if "run_eval_jobs.py" in l and "python" in l:
            m = re.search(r"--gpu-slots\s+(\S+)", l)
            if m:
                out |= {int(x) for x in m.group(1).split(",")}
    return out


def gpu_cap(procs, recent_train_launch, per_gpu, per_gpu_idle):
    """Process cap of one GPU: per_gpu_idle when no training runs on it (nor was launched on it in the last 5 min)."""
    training = recent_train_launch or any(x["what"].startswith("train ") for x in procs)
    return per_gpu if (training or per_gpu_idle is None) else max(per_gpu, per_gpu_idle)


DEFAULT_EVAL_MIN = {"table": 130.0, "bucket": 95.0, "pull": 60.0}  # +/-2 next to trainings, later checkpoints (11/11-12)


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def job_case(job):
    return job["argv"][job["argv"].index("--case") + 1]


def training_alive():
    """Any training process or training queue on this machine (ps; detection only, nothing is signalled)."""
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    for l in ps.splitlines():
        tok = l.split()
        if tok and "python" in os.path.basename(tok[0]) and any(
                k in t for t in tok[1:4] for k in ("run_train_queue.py", "train_plan5", "train_core4d")):
            return True
    return False


def eval_minutes(ops):
    """Median duration (minutes) of the last 5 completed cloud evaluations per case, launched on a GPU without training."""
    per = {}
    for lj in ops.glob("*.launch.json"):
        ej = lj.with_name(lj.name[: -len(".launch.json")] + ".exit.json")
        if not ej.exists():
            continue
        try:
            L, E = json.loads(lj.read_text()), json.loads(ej.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        argv = L.get("argv") or []
        if not E.get("completed") or "--case" not in argv or any(str(x).startswith("train ") for x in L.get("gpu_procs_before") or []):
            continue
        m = (datetime.fromisoformat(E["ended"]) - datetime.fromisoformat(L["started"])).total_seconds() / 60
        per.setdefault(argv[argv.index("--case") + 1], []).append((L["started"], m))
    return {c: statistics.median(m for _, m in sorted(v)[-5:]) for c, v in per.items()}


def est_minutes(job, measured):
    c = job_case(job)
    return measured.get(c, DEFAULT_EVAL_MIN.get(c, 130.0))


def claim_owner(name):
    f = EVALS / "claims" / safe(name) / "owner.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("owner")
    except (OSError, json.JSONDecodeError):
        return "?"


def stop_for_deadline(name, pid, gpu, ops, how_found):
    """Stop one evaluation by its recorded PID; keep its output renamed."""
    out = EVALS / name
    how = stop_pid(pid)
    k = 1
    while out.with_name(f"{out.name}.stopped_deadline{k}").exists():
        k += 1
    if out.exists():
        out.rename(out.with_name(f"{out.name}.stopped_deadline{k}"))
    (ops / f"{safe(name)}.stopped_deadline.json").write_text(json.dumps(dict(
        name=name, pid=pid, gpu=gpu, stopped=now(), how=how, pid_from=how_found,
        reason="Kevin 2026-11-12: cloud evaluations end 2 h after the last training; unfinished ones go to the laptop",
        output_dir_renamed_to=str(out.with_name(f"{out.name}.stopped_deadline{k}"))), indent=1))
    return how


def transfer_list(jobs, fails):
    """Evaluations not completed on the cloud and not owned by the laptop, with what the laptop needs to run them."""
    out = []
    for j in jobs:
        if summary_complete(EVALS / j["name"]):
            continue
        owner = claim_owner(j["name"])
        if owner not in (None, "cloud"):
            continue
        argv = j["argv"]
        ck, st = argv[argv.index("--checkpoint") + 1], argv[argv.index("--states") + 1]
        out.append(dict(name=j["name"], kind=j["meta"]["kind"], run=j["meta"]["run"], iteration=j["meta"]["iteration"],
                        level=j["meta"].get("level", 2), case=job_case(j), controller=argv[argv.index("--controller") + 1],
                        object_mass_kg=float(argv[argv.index("--object-mass-kg") + 1]) if "--object-mass-kg" in argv else None,
                        checkpoint=ck, checkpoint_sha256=j.get("sha256") or sha256_file(ck), states=st, states_sha256=sha256_file(st),
                        cloud_owner=owner, cloud_failures=fails.get(j["name"], 0), cloud_argv=argv))
    return out


def busy_eval_names():
    """Evaluations already running on this machine (also those started before a scheduler restart)."""
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    out = set()
    for l in ps.splitlines():
        if "evaluate_paper_supplement.py" in l:
            m = re.search(r"--output-dir\s+(\S+)", l)
            if m and m.group(1).startswith(str(EVALS) + "/"):
                out.add(m.group(1)[len(str(EVALS)) + 1:])
    return out


def summary_complete(out_dir):
    try:
        return bool(json.loads((out_dir / "summary.json").read_text()).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def c2_eval_jobs(train, pm8_last=False, final_pm2=False):
    jobs = []
    exits = {}
    for f in (train / "ops").glob("c*.exit.json"):
        try:
            exits[f.name[:-10]] = json.loads(f.read_text()).get("completed")
        except json.JSONDecodeError:
            pass
    for d in sorted(list(train.glob("cin_*")) + list(train.glob("cout_*"))):
        if not d.is_dir() or d.name.endswith("_critic50"):
            continue
        label = d.name
        case = label.split("_")[1]
        mass = float(re.search(r"_m([\d.p]+)kg", label).group(1).replace("p", "."))
        cks = sorted(p for p in d.glob("model_*.pt") if time.time() - p.stat().st_mtime > 120)
        for ck in cks:
            it = int(ck.stem.split("_")[1])
            levels = [2] + ([8] if exits.get(label) and ck == cks[-1] else [])
            for lvl in levels:
                name = f"c2_curve/{label}/it{it:05d}_pm{lvl}" if lvl == 2 else f"c2_final/{label}/it{it:05d}_pm8"
                argv = ["scripts/evaluate_paper_supplement.py", "--case", case, "--controller", "sonic_residual", "--checkpoint", str(ck),
                        "--states", str(STATES[(case, lvl)]), "--model-id", safe(name), "--output-dir", str(EVALS / name)]
                if mass != ORIG[case]:
                    argv += ["--object-mass-kg", f"{mass:g}"]
                prio = (5, it) if (pm8_last and lvl == 8) else (1, it)
                meta = dict(kind="c2", run=label, iteration=it, level=lvl)
                if final_pm2 and lvl == 2 and exits.get(label) and ck == cks[-1]:
                    meta["ignore_claims"] = True  # curve point on the 4090; the laptop's own +/-2 is for the paper table
                jobs.append(dict(prio=prio, name=name, argv=argv, meta=meta))
    return jobs


def d_curve_jobs(all_4090=False):
    if not D_CAT.exists():
        return []
    cat = json.loads(D_CAT.read_text())["runs"]
    jobs = []
    for label, r in cat.items():
        for it, x in r["checkpoints"].items():
            v3_4050 = "_scratch_V3-" in label and int(it) == 4050
            prio = (2, 0) if v3_4050 else (3, int(it)) if r["seed"] == 721 else (4, int(it))
            if all_4090:
                prio = (0.5, *prio)  # before every other curve evaluation
            argv = ["scripts/evaluate_paper_supplement.py", "--case", r["case"], "--controller", "policy", "--checkpoint", x["path"],
                    "--states", str(STATES[(r["case"], 2)]), "--model-id", safe(x["name"]), "--output-dir", str(EVALS / x["name"])]
            meta = dict(kind="d_curve", run=label, iteration=int(it))
            if all_4090:
                meta["ignore_claims"] = True
            jobs.append(dict(prio=prio, name=x["name"], argv=argv, sha256=x["sha256"], meta=meta))
    return jobs


def xgpu_jobs():
    """Kevin 2026-11-12: cross-GPU control, D paper seed 721 checkpoints already evaluated on the laptop 5070."""
    if not D_CAT.exists():
        return []
    cat = json.loads(D_CAT.read_text())["runs"]
    jobs = []
    for k, (label, it) in enumerate(XGPU):
        x, case = cat[label]["checkpoints"][it], cat[label]["case"]
        name = f"xgpu/{label}/it{int(it):05d}_pm2"
        argv = ["scripts/evaluate_paper_supplement.py", "--case", case, "--controller", "policy", "--checkpoint", x["path"],
                "--states", str(STATES[(case, 2)]), "--model-id", safe(name), "--output-dir", str(EVALS / name)]
        jobs.append(dict(prio=(0, k), name=name, argv=argv, sha256=x["sha256"], meta=dict(kind="xgpu", run=label, iteration=int(it))))
    return jobs


def stop_pid(pid, wait=60):
    import signal
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "gone"
    t0 = time.time()
    while time.time() - t0 < wait:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "sigterm"
        time.sleep(2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "sigterm"
    return "sigkill"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--per-gpu", type=int, default=3)
    p.add_argument("--per-gpu-idle", type=int, default=None, help="cap of a GPU with no training process (default: --per-gpu)")
    p.add_argument("--pm8-last", action="store_true", help="cloud +/-8 of final checkpoints after every curve evaluation")
    p.add_argument("--curves-on-4090", action="store_true",
                   help="D curve checkpoints and final C +/-2 curve points evaluated here even if the laptop claimed them")
    p.add_argument("--post-train-hours", type=float, default=None,
                   help="evaluations end this many hours after the last training; the rest is handed to the laptop")
    p.add_argument("--min-free-mib", type=int, default=5000)
    p.add_argument("--train-min-free-mib", type=int, default=7500, help="free memory a GPU needs to take a training run")
    p.add_argument("--python", default=os.environ.get("S1_PY"))
    p.add_argument("--poll", type=int, default=60)
    a = p.parse_args()
    dec_path = a.train / "c2_decisions.json"
    dec = json.loads(dec_path.read_text()) if dec_path.exists() else dict(early_stop={}, choice=None, mass_started={})
    dec.setdefault("seed_started", {})
    save = lambda: dec_path.write_text(json.dumps(dec, indent=1))  # noqa: E731
    ops = EVALS / "ops2"
    ops.mkdir(parents=True, exist_ok=True)
    fails_path = ops / "failures.json"
    fails = json.loads(fails_path.read_text()) if fails_path.exists() else {}
    running = {}  # name -> (job, proc, log, gpu)
    recent = []  # (time, gpu, pid or None): launches not yet visible in nvidia-smi count against their GPU for 5 min
    n_launch = 0
    (a.train / "c2_scheduler.pid.json").write_text(json.dumps(dict(pid=os.getpid(), started=now(), argv=sys.argv), indent=1))
    print(now(), "c2 scheduler / slot filler start; pid", os.getpid(), "state", json.dumps(dec)[:300], flush=True)
    while True:
        # iteration 1000
        for k in KINDS:
            if k in dec["early_stop"]:
                continue
            m = {t: metrics(a.train, k, t) for t in TASKS}
            if not all(max(r, default=0) >= 1000 for r in m.values()):
                continue
            per = {}
            for t, r in m.items():
                e_len, n1 = wmean(r, 1, 100, LEN_KEY[t])
                l_len, n2 = wmean(r, 901, 1000, LEN_KEY[t])
                e_rew, n3 = wmean(r, 1, 100, "reward_mean")
                l_rew, n4 = wmean(r, 901, 1000, "reward_mean")
                prog = (e_len is not None and l_len is not None and l_len > e_len) or (e_rew is not None and l_rew is not None and l_rew > e_rew)
                per[t] = dict(len_1_100=e_len, len_901_1000=l_len, reward_1_100=e_rew, reward_901_1000=l_rew,
                              n=[n1, n2, n3, n4], length_key=LEN_KEY[t], progressing=bool(prog))
            stop = not any(x["progressing"] for x in per.values())
            dec["early_stop"][k] = dict(decided=now(), runs=per, stop=stop)
            save()
            print(now(), "iteration-1000 check", k, json.dumps(dec["early_stop"][k]), flush=True)
            if stop:
                r = subprocess.run(["python3", str(ROOT / "system1/cloud/stop_runs.py"), "--reason",
                                    f"T03 C v2 iteration-1000 rule: no {k} run progressed", *[f"{k}_{v}" for v in TASKS.values()]],
                                   capture_output=True, text=True)
                dec["early_stop"][k]["stop_output"] = (r.stdout + r.stderr)[-3000:]
                save()
                print(now(), "stopped", k, r.stdout[-1500:], flush=True)
        # iteration 2000
        if dec["choice"] is None and all(k in dec["early_stop"] for k in KINDS):
            alive = [k for k in KINDS if not dec["early_stop"][k]["stop"]]
            ready = all(max(metrics(a.train, k, t), default=0) >= 2000 for k in alive for t in TASKS)
            if not alive:
                dec["choice"] = dict(decided=now(), kind=None, reason="both kinds stopped at iteration 1000: no C mass sweep")
                save()
                print(now(), "no C mass sweep", flush=True)
            elif ready:
                per, wins = {}, {k: 0 for k in alive}
                for t in TASKS:
                    v = {k: wmean(metrics(a.train, k, t), 1901, 2000, "reward_mean") for k in alive}
                    per[t] = {k: dict(reward_1901_2000=x[0], n=x[1]) for k, x in v.items()}
                    if len(alive) == 2:
                        w = max(alive, key=lambda k: v[k][0] if v[k][0] is not None else float("-inf"))
                        per[t]["winner"] = w
                        wins[w] += 1
                kind = alive[0] if len(alive) == 1 else max(alive, key=lambda k: wins[k])
                dec["choice"] = dict(decided=now(), kind=kind, alive=alive, per_task=per, wins=wins)
                save()
                print(now(), "iteration-2000 choice", json.dumps(dec["choice"]), flush=True)
        # finished evaluations
        for name, (job, proc, log, g) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                ok = summary_complete(EVALS / name)
                (ops / f"{safe(name)}.exit.json").write_text(json.dumps(dict(name=name, returncode=proc.returncode, completed=ok, ended=now()), indent=1))
                if not ok:
                    fails[name] = fails.get(name, 0) + 1
                    fails_path.write_text(json.dumps(fails, indent=1))
                print(now(), "end", name, "rc", proc.returncode, "complete", ok, flush=True)
                del running[name]
        # queue
        mass_todo = []
        if dec["choice"] and dec["choice"].get("kind"):
            kind = dec["choice"]["kind"]
            for task, run, its in MASS:
                rid = f"{kind}_{run}"
                if rid in dec["mass_started"]:
                    continue
                r = metrics(a.train, kind, task)
                last = sorted(r)[-200:]
                spi = statistics.median([r[i]["elapsed_seconds"] for i in last if "elapsed_seconds" in r[i]]) if last else 0.0
                mass_todo.append((its * spi, rid, its, spi))
            mass_todo.sort(reverse=True)
        seed_todo = []
        for task, run, its, seed in SEEDS:
            rid = f"cin_{run}@seed{seed}"
            if rid in dec["seed_started"]:
                continue
            r = metrics(a.train, "cin", task)
            last = sorted(r)[-200:]
            spi = statistics.median([r[i]["elapsed_seconds"] for i in last if "elapsed_seconds" in r[i]]) if last else 0.0
            seed_todo.append((its * spi, rid, f"cin_{run}", seed, its, spi))
        seed_todo.sort(reverse=True)
        busy = busy_eval_names() | set(running)
        evals = [j for j in xgpu_jobs() + c2_eval_jobs(a.train, a.pm8_last, a.curves_on_4090) + d_curve_jobs(a.curves_on_4090)
                 if j["name"] not in busy and fails.get(j["name"], 0) < 2 and not summary_complete(EVALS / j["name"])]
        claimed = {x.name for x in (EVALS / "claims").iterdir()} if (EVALS / "claims").exists() else set()
        mine = {x for x in claimed if (EVALS / "claims" / x / "owner.json").exists() and
                '"owner": "cloud"' in (EVALS / "claims" / x / "owner.json").read_text()}
        evals = [j for j in evals if safe(j["name"]) not in claimed or safe(j["name"]) in mine or j["meta"].get("ignore_claims")]
        evals.sort(key=lambda j: (j["prio"], j["name"]))
        gs = gpu_state()
        excl = excluded_gpus()
        recent = [x for x in recent if time.time() - x[0] < 300]
        seen_pids = {pr["pid"] for v in gs.values() for pr in v["procs"]}
        extra = {}
        for t0, g, pid in recent:  # launched but not yet holding a CUDA context
            if pid is None or pid not in seen_pids:
                extra[g] = extra.get(g, 0) + 1
        tlaunch = {g for t0, g, pid in recent if pid is None}  # training queues (their child appears later)
        cap = {g: gpu_cap(v["procs"], g in tlaunch, a.per_gpu, a.per_gpu_idle) for g, v in gs.items()}
        free = sorted(((len(v["procs"]) + extra.get(g, 0), g) for g, v in gs.items()
                       if len(v["procs"]) + extra.get(g, 0) < cap[g] and v["free_mib"] >= a.min_free_mib and g not in excl))
        (a.train / "c2_slots.json").write_text(json.dumps(dict(
            time=now(), gpus={g: dict(procs=[x["what"] for x in v["procs"]], free_slots=max(0, cap[g] - len(v["procs"])), cap=cap[g],
                                      free_mib=v["free_mib"], held_by_first_eval_list=g in excl) for g, v in gs.items()},
            queue=dict(mass_pending=[x[1] for x in mass_todo], seeds_pending=[x[1] for x in seed_todo], evals_pending=len(evals),
                       evals_pending_by_kind={k: sum(1 for j in evals if j["meta"]["kind"] == k) for k in ("xgpu", "c2", "d_curve")},
                       next=[j["name"] for j in evals[:5]]),
            running_evals=sorted(running), all_training_done=dec.get("all_training_done"),
            handover=dec.get("handover")), indent=1))
        deadline = None
        if a.post_train_hours is not None:
            if "all_training_done" not in dec and not mass_todo and not seed_todo and not tlaunch and not training_alive():
                t = time.time()
                dec["all_training_done"] = dict(time=now(), hours=a.post_train_hours, deadline_unix=t + a.post_train_hours * 3600,
                                                deadline=datetime.fromtimestamp(t + a.post_train_hours * 3600, NY).isoformat(timespec="seconds"))
                save()
                print(now(), "all training done; evaluation deadline", dec["all_training_done"]["deadline"], flush=True)
            deadline = (dec.get("all_training_done") or {}).get("deadline_unix")
            if deadline and time.time() >= deadline and "handover" not in dec:
                stopped = []
                for name, (job, proc, log, g) in list(running.items()):
                    how = stop_for_deadline(name, proc.pid, g, ops, "scheduler Popen")
                    proc.poll()
                    log.close()
                    del running[name]
                    stopped.append(dict(name=name, pid=proc.pid, how=how))
                for name in sorted(busy_eval_names() - {x["name"] for x in stopped}):  # started by an earlier scheduler instance
                    lj = ops / f"{safe(name)}.launch.json"
                    rec = json.loads(lj.read_text()) if lj.exists() else {}
                    pid = rec.get("pid")
                    try:
                        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace") if pid else ""
                    except OSError:
                        cmd = ""
                    if pid and str(EVALS / name) in cmd:
                        stopped.append(dict(name=name, pid=pid, how=stop_for_deadline(name, pid, rec.get("gpu"), ops, "ops2 launch record")))
                    else:
                        stopped.append(dict(name=name, pid=pid, how="not stopped: no matching recorded PID"))
                tl = transfer_list(xgpu_jobs() + c2_eval_jobs(a.train, a.pm8_last, a.curves_on_4090) + d_curve_jobs(a.curves_on_4090), fails)
                for r in tl:
                    if r["cloud_owner"] == "cloud":
                        (EVALS / "claims" / safe(r["name"]) / "handover.json").write_text(json.dumps(dict(to="laptop", time=now()), indent=1))
                tf = ops / "transfer_to_laptop.json"
                tf.write_text(json.dumps(dict(time=now(), deadline=dec["all_training_done"]["deadline"], stopped=stopped, jobs=tl), indent=1))
                dec["handover"] = dict(time=now(), n_jobs=len(tl), stopped=stopped, file=str(tf))
                save()
                print(now(), "deadline reached: stopped", len(stopped), "evaluations; handed", len(tl), "to the laptop:", tf, flush=True)
            if "handover" in dec:
                time.sleep(a.poll)
                continue
        measured = eval_minutes(ops) if deadline else {}
        launched = False
        xg_running = any(n.startswith("xgpu/") for n in busy)
        xg_pending = [j for j in evals if j["meta"]["kind"] == "xgpu"]
        xgpu_turn = bool(seed_todo) and bool(xg_pending) and not xg_running and len(dec["seed_started"]) >= XGPU_AFTER_SEEDS
        if free and mass_todo:
            est, rid, its, spi = mass_todo[0]
            g = free[0][1]
            cmd = [a.python, "-B", str(ROOT / "system1/cloud/run_train_queue.py"), rid, "--out", str(a.train), "--python", a.python,
                   "--gpu", str(g), "--local-rank", str(600 + len(dec["mass_started"]))]
            log = a.train / f"queue_{rid.replace(':', '_')}_seed721.log"
            qp = subprocess.Popen(["nohup", *cmd], stdout=log.open("w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  cwd=ROOT, start_new_session=True)
            pf = a.train / "c2_pids.json"  # recorded PID (nohup execs python in the same process)
            pids = json.loads(pf.read_text()) if pf.exists() else {}
            pids[rid] = dict(pid=qp.pid, started=now(), gpu=g, by="c2_scheduler")
            pf.write_text(json.dumps(pids, indent=1))
            recent.append((time.time(), g, None))  # the training child appears later; counted for 5 min
            dec["mass_started"][rid] = dict(time=now(), gpu=g, gpu_procs_before=[x["what"] for x in gs[g]["procs"]],
                                            expected_seconds=est, sec_per_it_main=spi, iterations=its, cmd=cmd)
            save()
            print(now(), "started mass run", rid, "on GPU", g, flush=True)
            launched = True
        elif seed_todo and not mass_todo and not xgpu_turn:
            tfree = [(n, g) for n, g in free if gs[g]["free_mib"] >= a.train_min_free_mib]
            if tfree:
                est, rid, run, seed, its, spi = seed_todo[0]
                g = tfree[0][1]
                cmd = [a.python, "-B", str(ROOT / "system1/cloud/run_train_queue.py"), run, "--seed", str(seed), "--out", str(a.train),
                       "--python", a.python, "--gpu", str(g), "--local-rank", str(650 + len(dec["seed_started"]))]
                log = a.train / f"queue_{run.replace(':', '_')}_seed{seed}.log"
                qp = subprocess.Popen(["nohup", *cmd], stdout=log.open("w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      cwd=ROOT, start_new_session=True)
                pf = a.train / "c2_pids.json"
                pids = json.loads(pf.read_text()) if pf.exists() else {}
                pids[rid] = dict(pid=qp.pid, started=now(), gpu=g, by="c2_scheduler", seed=seed)
                pf.write_text(json.dumps(pids, indent=1))
                recent.append((time.time(), g, None))
                dec["seed_started"][rid] = dict(time=now(), gpu=g, gpu_procs_before=[x["what"] for x in gs[g]["procs"]],
                                                expected_seconds=est, sec_per_it_seed721=spi, iterations=its, cmd=cmd, pid=qp.pid)
                save()
                print(now(), "started seed run", rid, "on GPU", g, "pid", qp.pid, flush=True)
                launched = True
            else:
                # free a slot: stop one evaluation this scheduler started (recorded PID) on a GPU with room for a training once it ends
                cand = sorted(((sum(1 for x in gs[g]["procs"] if x["what"].startswith("eval ")), g, name)
                               for name, (job, proc, log, g) in running.items()
                               if not name.startswith("xgpu/") and g not in excl and gs[g]["free_mib"] + 2000 >= a.train_min_free_mib), reverse=True)
                for _, g, name in cand:
                    job, proc, log, _g = running[name]
                    how = stop_pid(proc.pid)
                    proc.poll()
                    log.close()
                    out = EVALS / name
                    k = 1
                    while out.with_name(f"{out.name}.preempted{k}").exists():
                        k += 1
                    if out.exists():
                        out.rename(out.with_name(f"{out.name}.preempted{k}"))
                    (ops / f"{safe(name)}.preempted.json").write_text(json.dumps(dict(
                        name=name, pid=proc.pid, gpu=g, stopped=now(), how=how, reason="Kevin 2026-11-12: C-in extra seeds before every evaluation",
                        output_dir_renamed_to=str(out.with_name(f"{out.name}.preempted{k}"))), indent=1))
                    del running[name]
                    print(now(), "preempted", name, "on GPU", g, "pid", proc.pid, how, flush=True)
                    launched = True
                    break
        elif free and not mass_todo and (not seed_todo or xgpu_turn):
            for job in (xg_pending if seed_todo else evals):
                if deadline and est_minutes(job, measured) > (deadline - time.time()) / 60:
                    continue  # would not finish before the deadline: left for the laptop
                if not claim_local_fs(job["name"], "cloud", dict(gpu="NVIDIA GeForce RTX 4090")):
                    if not job["meta"].get("ignore_claims"):
                        continue
                    rr = EVALS / "claims" / safe(job["name"]) / "cloud_rerun.json"  # claim stays the laptop's
                    if not rr.exists():
                        rr.write_text(json.dumps(dict(time=now(), gpu="NVIDIA GeForce RTX 4090",
                                                      reason="Kevin 2026-11-12: curve points on the cloud 4090"), indent=1))
                g = free[0][1]
                if job.get("sha256"):
                    import hashlib
                    h = hashlib.sha256(Path(job["argv"][job["argv"].index("--checkpoint") + 1]).read_bytes()).hexdigest()
                    if h != job["sha256"]:
                        print(now(), "checkpoint sha256 mismatch, skip", job["name"], flush=True)
                        fails[job["name"]] = 99
                        fails_path.write_text(json.dumps(fails, indent=1))
                        continue
                out = EVALS / job["name"]
                if out.exists():  # earlier attempt: keep it, renamed
                    k = 1
                    while out.with_name(f"{out.name}.attempt{k}").exists():
                        k += 1
                    out.rename(out.with_name(f"{out.name}.attempt{k}"))
                n_launch += 1
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), LOCAL_RANK=str(800 + n_launch % 100), WORLD_SIZE="1")
                argv = [a.python, "-B", *job["argv"]]
                git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
                (ops / f"{safe(job['name'])}.launch.json").write_text(json.dumps(dict(
                    name=job["name"], argv=argv, gpu=g, started=now(), env={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "LOCAL_RANK")},
                    pid=None,
                    git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
                    gpu_procs_before=[x["what"] for x in gs[g]["procs"]], meta=job["meta"]), indent=1))
                log = (ops / f"{safe(job['name'])}.stdout.log").open("w")
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                lj = ops / f"{safe(job['name'])}.launch.json"
                lj.write_text(json.dumps(dict(json.loads(lj.read_text()), pid=proc.pid), indent=1))
                running[job["name"]] = (job, proc, log, g)
                recent.append((time.time(), g, proc.pid))
                print(now(), "start", job["name"], "gpu", g, flush=True)
                launched = True
                break
        if launched:
            time.sleep(30)  # stagger Isaac start-ups; the new process shows up in nvidia-smi before the next decision
            continue
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
