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
    2. evaluations of C v2 checkpoints (every cin_/cout_ run, main and mass, every saved checkpoint, +/-2; the final
       checkpoint of a completed run also +/-8), oldest checkpoint first;
    3. D "completion vs iteration" curve evaluations from $S1_INPUTS/d_curve/catalog.json (V3-25-27 it04050 first,
       then seed 721, then 722 / 723), shared with the laptop pipeline through claims (system1/cloud/claims.py).
  GPUs used by a running run_eval_jobs.py (the first 12 evaluations) are left to it until it exits.
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


def c2_eval_jobs(train):
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
                jobs.append(dict(prio=(1, it), name=name, argv=argv, meta=dict(kind="c2", run=label, iteration=it, level=lvl)))
    return jobs


def d_curve_jobs():
    if not D_CAT.exists():
        return []
    cat = json.loads(D_CAT.read_text())["runs"]
    jobs = []
    for label, r in cat.items():
        for it, x in r["checkpoints"].items():
            v3_4050 = "_scratch_V3-" in label and int(it) == 4050
            prio = (2, 0) if v3_4050 else (3, int(it)) if r["seed"] == 721 else (4, int(it))
            argv = ["scripts/evaluate_paper_supplement.py", "--case", r["case"], "--controller", "policy", "--checkpoint", x["path"],
                    "--states", str(STATES[(r["case"], 2)]), "--model-id", safe(x["name"]), "--output-dir", str(EVALS / x["name"])]
            jobs.append(dict(prio=prio, name=x["name"], argv=argv, sha256=x["sha256"], meta=dict(kind="d_curve", run=label, iteration=int(it))))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--per-gpu", type=int, default=3)
    p.add_argument("--min-free-mib", type=int, default=5000)
    p.add_argument("--python", default=os.environ.get("S1_PY"))
    p.add_argument("--poll", type=int, default=60)
    a = p.parse_args()
    dec_path = a.train / "c2_decisions.json"
    dec = json.loads(dec_path.read_text()) if dec_path.exists() else dict(early_stop={}, choice=None, mass_started={})
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
        busy = busy_eval_names() | set(running)
        evals = [j for j in c2_eval_jobs(a.train) + d_curve_jobs()
                 if j["name"] not in busy and fails.get(j["name"], 0) < 2 and not summary_complete(EVALS / j["name"])]
        claimed = {x.name for x in (EVALS / "claims").iterdir()} if (EVALS / "claims").exists() else set()
        mine = {x for x in claimed if (EVALS / "claims" / x / "owner.json").exists() and
                '"owner": "cloud"' in (EVALS / "claims" / x / "owner.json").read_text()}
        evals = [j for j in evals if safe(j["name"]) not in claimed or safe(j["name"]) in mine]
        evals.sort(key=lambda j: (j["prio"], j["name"]))
        gs = gpu_state()
        excl = excluded_gpus()
        recent = [x for x in recent if time.time() - x[0] < 300]
        seen_pids = {pr["pid"] for v in gs.values() for pr in v["procs"]}
        extra = {}
        for t0, g, pid in recent:  # launched but not yet holding a CUDA context
            if pid is None or pid not in seen_pids:
                extra[g] = extra.get(g, 0) + 1
        free = sorted(((len(v["procs"]) + extra.get(g, 0), g) for g, v in gs.items()
                       if len(v["procs"]) + extra.get(g, 0) < a.per_gpu and v["free_mib"] >= a.min_free_mib and g not in excl))
        (a.train / "c2_slots.json").write_text(json.dumps(dict(
            time=now(), gpus={g: dict(procs=[x["what"] for x in v["procs"]], free_slots=max(0, a.per_gpu - len(v["procs"])),
                                      free_mib=v["free_mib"], held_by_first_eval_list=g in excl) for g, v in gs.items()},
            queue=dict(mass_pending=[x[1] for x in mass_todo], evals_pending=len(evals),
                       evals_pending_by_kind={k: sum(1 for j in evals if j["meta"]["kind"] == k) for k in ("c2", "d_curve")},
                       next=[j["name"] for j in evals[:5]]),
            running_evals=sorted(running)), indent=1))
        launched = False
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
        elif free and not mass_todo:
            for job in evals:
                if not claim_local_fs(job["name"], "cloud", dict(gpu="NVIDIA GeForce RTX 4090")):
                    continue
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
