#!/usr/bin/env python3
"""Kevin 2026-11-27: one machine queue for T13 section 9 (continued runs to counted 10000) and the T14 front-back training task,
on the cloud machines 199 / 196 / 210 (opened by the cloud Claude; this script never starts a machine).

Queue, longest expected run first (Kevin): d_frontback -> d_push_to10k -> d_table_to10k -> c_out_push_to10k. A task is "ready"
when its local smoke passed (d_frontback: logs_system1/t14/frontback_train/smoke, smoke.json passed and the T10 gate summary
passed; *_to10k: continue_prep.py --smoke, smoke.json passed).

Every --poll seconds (default 600 = "每 10 分钟试连 SSH"), per machine:
  * not up yet: try ssh; reachable and idle (no process besides the platform's own, no GPU compute process; post_run.py's
    cloud_processes) -> free; reachable but busy with something else -> recorded, tried again next time;
  * free: start the first ready queued task on it (a thread per bring-up):
      1. code: git pull system1-sonic on the cloud, HEAD = local HEAD (as round2_launch.py);
      2. C-out tasks: SONIC weights present on the machine with the laptop's sha256 (nothing is copied; a machine without them
         is skipped for C-out tasks, recorded);
      3. GPU idle;
      4. continued tasks: the base checkpoint and continue_ref.json are copied to <S1>/logs/t13v4/continue/<task>/ (scp) and
         their sha256 checked on the cloud;
      5. d_frontback: cloud_smoke.py --run d_frontback --num-envs 4096 --iterations 10 must pass (as round2_launch.py);
         continued tasks: no cloud smoke (resumed weights; cloud_runs.py's section 9 resume check stops the run itself);
      6. launch: nohup cloud_runs.py --run TASK --num-envs 4096 --gpu 0 --out <S1>/logs/t13v4/runs (+ --continue-checkpoint /
         --continue-ref); hosts.json["runs"][TASK] = {machine, ssh, dir} so that status_hourly.py and post_run_watch.py follow it.
    a failed step puts the task back in the queue (failures recorded; after 2 failures of a task it is held for Kevin);
    if nothing is queued any more: shut the machine down by post_run.py's rule (c) (3 checks, 60 s apart), as T13 section 8;
    if only not-ready tasks are queued: wait, at most WAIT_MAX (1 h), then shut down by the same rule (the task stays queued);
  * busy: the task's post_run.py writes logs_system1/t13v4/post_run/<task>/queue_release.json after the verified copy-back
    (it does not shut the machine down for queue tasks) -> free again (next task or shutdown, above). A continued task stopped by
    the resume check (post_run_watch.py marker resume_check_failed.json): copy-back of its run directory (copyback.sh), the
    other continued tasks are held for Kevin, the machine is free.
State: logs_system1/t13v4/cloud/queue_state.json (local); the public table (task, machine, start, state; no ssh data) goes to
results_system1/T13_runs/queue_status.json and is committed / pushed on every change (flock .git/s1_status.lock).

    python3 system1/t13v4/queue_s9.py            # in tmux (session queue_s9)
    python3 system1/t13v4/queue_s9.py --once --dry-run
"""
from __future__ import annotations

import argparse
import fcntl
import json
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t13v4.post_run import Pipe  # noqa: E402  (cloud_processes: read-only process check)
from system1.t13v4.round2_launch import L, RUNS, S1, now, sha, ssh  # noqa: E402

HOSTS = ROOT / "logs_system1/t13v4/cloud/hosts.json"
STATE = ROOT / "logs_system1/t13v4/cloud/queue_state.json"
PUB = ROOT / "results_system1/T13_runs/queue_status.json"
QUEUE = ("d_frontback", "d_push_to10k", "d_table_to10k", "c_out_push_to10k")
MACHINES = ("199", "196", "210")
WAIT_MAX = 3600  # s a free machine waits for a queued task that is not ready yet (e.g. awaiting Kevin), then shutdown by the rule
ENV = f"source {S1}/s1_env.sh; [[ -f /etc/network_turbo ]] && source /etc/network_turbo; cd {S1}/duet-system1"
lock = threading.Lock()


def ready(task):
    if task == "d_frontback":
        sm = ROOT / "logs_system1/t14/frontback_train/smoke/train/d_frontback/smoke.json"
        g = ROOT / "logs_system1/t13v4/sim2sim/checks_d_frontback_smoke/summary.json"
        if not (sm.exists() and g.exists()):
            return False
        gg = json.loads(g.read_text())
        return bool(json.loads(sm.read_text()).get("passed")) and gg["initial_states"]["all_passed"] and gg["observations"]["obs_passed"]
    p = ROOT / "logs_system1/t13v4/continue" / task / "smoke/smoke.json"
    return p.exists() and bool(json.loads(p.read_text()).get("passed"))


def batch(spec):
    return [*spec[:-1], "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", spec[-1]]


class Queue:
    def __init__(self, dry):
        self.dry = dry
        h = json.loads(HOSTS.read_text())
        self.spec = {}
        for key in ("round1", "round2", "runs"):
            for v in (h.get(key) or {}).values():
                self.spec.setdefault(str(v["machine"]), v["ssh"])
        assert all(m in self.spec for m in MACHINES), sorted(self.spec)
        if STATE.exists():
            self.s = json.loads(STATE.read_text())
        else:
            self.s = dict(created=now(), queue=list(QUEUE), machines={m: dict(state="not up", events=[]) for m in MACHINES},
                          tasks={t: dict(state="queued", failures=[], steps=[]) for t in QUEUE}, events=[])
        self.local_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        src = (ROOT / "system1/sonic_controller.py").read_text()
        ck = Path(next(x for x in src.splitlines() if x.startswith("SONIC_ROOT")).split('"')[-2]) / "sonic_release/last.pt"
        self.sonic_sha = sha(ck)
        self.threads = {}

    # ------------------------------------------------------------------ records
    def event(self, what, **kw):
        with lock:
            self.s["events"].append(dict(time=now(), what=what, **kw))
        print(now(), what, json.dumps(kw, default=str)[:300], flush=True)

    def save(self, commit=False):
        with lock:
            if self.dry:
                return
            STATE.write_text(json.dumps(self.s, indent=1, default=str) + "\n")
            pub = dict(updated=now(), rule="Kevin 2026-11-27: one queue on 199 / 196 / 210, longest first; next task on the same machine after "
                                            "a verified copy-back; shutdown when the queue is empty", queue=list(QUEUE),
                       tasks={t: {k: v.get(k) for k in ("state", "machine", "started", "launched", "ended", "released", "note")} for t, v in self.s["tasks"].items()},
                       machines={m: {k: v.get(k) for k in ("state", "task", "since", "shutdown_time")} for m, v in self.s["machines"].items()})
            old = json.loads(PUB.read_text()) if PUB.exists() else {}
            changed = {k: v for k, v in old.items() if k != "updated"} != {k: v for k, v in pub.items() if k != "updated"}
            if changed:
                PUB.write_text(json.dumps(pub, indent=1, ensure_ascii=False) + "\n")
        if commit and changed:
            self.commit("queue status: " + "; ".join(f"{t} {v['state']}" + (f" on {v['machine']}" if v.get("machine") else "") for t, v in self.s["tasks"].items()))

    def commit(self, what):
        with open(ROOT / ".git/s1_status.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            g = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True)  # noqa: E731
            if g("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() != "system1-sonic":
                return
            rel = str(PUB.relative_to(ROOT))
            g("add", rel)
            c = g("commit", "-m", f"system1(T13 section 9 + T14 front-back queue): {what}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", rel)
            if c.returncode == 0:
                g("pull", "--no-edit", "--no-rebase")
                g("push", "origin", "system1-sonic")

    def mset(self, m, **kw):
        with lock:
            self.s["machines"][m].update(kw, since=now())
            self.s["machines"][m]["events"].append(dict(time=now(), **kw))

    def tstep(self, t, name, ok, **kw):
        with lock:
            self.s["tasks"][t]["steps"].append(dict(name=name, ok=bool(ok), time=now(), **kw))
        self.save()
        return ok

    # ------------------------------------------------------------------ machine checks
    def idle(self, m):
        c = Pipe.cloud_processes(batch(self.spec[m]))
        return c.get("error") is None and not c["others"] and not c["gpu_compute"], c

    def shutdown(self, m):
        ssh_ = batch(self.spec[m])
        checks = []
        for k in range(3):
            ok, c = self.idle(m)
            checks.append(c)
            if ok:
                break
            if k < 2:
                time.sleep(60)
        if not ok:
            self.mset(m, state="up, not shut down (busy)", checks=checks)
            self.event("shutdown refused", machine=m, last=checks[-1])
            return False
        if self.dry:
            self.event("dry: would shut down", machine=m)
            return True
        p = subprocess.run([*ssh_, "shutdown"], capture_output=True, text=True, timeout=120)
        time.sleep(60)
        q = subprocess.run([*ssh_, "true"], capture_output=True, text=True, timeout=60)
        self.mset(m, state="shut down" if q.returncode != 0 else "shutdown sent, still reachable", task=None, shutdown_time=now(), rc=p.returncode)
        self.event("shutdown", machine=m, rc=p.returncode, reachable_60s_after=q.returncode == 0)
        self.save(commit=True)
        return True

    # ------------------------------------------------------------------ bring-up
    def bring_up(self, m, t):
        spec = self.spec[m]
        try:
            ok = self._bring_up(m, t, spec)
        except Exception as ex:  # noqa: BLE001
            ok = self.tstep(t, "exception", False, error=repr(ex)[:300])
        with lock:
            T = self.s["tasks"][t]
            if ok:
                T.update(state="running", launched=now())
                self.s["machines"][m].update(state="busy", task=t, since=now())
            else:
                T["failures"].append(dict(machine=m, time=now(), step=T["steps"][-1]["name"] if T["steps"] else None))
                T.update(state="queued" if len(T["failures"]) < 2 else "held (2 failed bring-ups)", machine=None)
                self.s["machines"][m].update(state="free", task=None, since=now())
        self.event("launched" if ok else "bring-up failed", task=t, machine=m)
        self.save(commit=True)

    def _bring_up(self, m, t, spec):
        self.local_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if self.dry:  # dry run: nothing on the cloud is changed
            return self.tstep(t, "dry: would bring up", True, machine=m)
        pull = "git fetch -q origin && git checkout -q system1-sonic && git pull -q --ff-only origin system1-sonic"
        rc, out, err = ssh(spec, f"{ENV}; ({pull}) || (sleep 10; git -c http.version=HTTP/1.1 fetch -q origin && git checkout -q system1-sonic && "
                                 f"git -c http.version=HTTP/1.1 pull -q --ff-only origin system1-sonic); git rev-parse HEAD")
        head = out.splitlines()[-1] if out else ""
        if not self.tstep(t, "code", rc == 0 and head == self.local_head, machine=m, cloud_head=head, local_head=self.local_head, rc=rc, err=err):
            return False
        if t.startswith("c_out_"):
            py = ("import hashlib,sys;sys.path.insert(0,'.');from system1 import sonic_controller as sc;p=sc.DEFAULT_CHECKPOINT;"
                  "print(p);print(hashlib.sha256(open(p,'rb').read()).hexdigest() if p.exists() else 'MISSING')")
            rc, out, err = ssh(spec, f"{ENV}; $S1_PY -c {shlex.quote(py)}")
            lines = out.splitlines()
            if not self.tstep(t, "sonic_weights", rc == 0 and len(lines) >= 2 and lines[-1] == self.sonic_sha, machine=m,
                              cloud_sha256=lines[-1] if lines else None, rc=rc):
                with lock:
                    self.s["machines"][m]["no_sonic"] = True
                return False
        rc, out, err = ssh(spec, "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader")
        lines = out.splitlines()
        if not self.tstep(t, "gpu_idle", rc == 0 and lines and lines[0].strip() == "0", machine=m, nvidia=lines[1:], rc=rc):
            return False
        extra = ""
        if t != "d_frontback":
            ref = json.loads((ROOT / "logs_system1/t13v4/continue" / t / "continue_ref.json").read_text())
            ck = ROOT / ref["checkpoint_local"]
            assert sha(ck) == ref["checkpoint_sha256"]
            rd = f"{S1}/logs/t13v4/continue/{t}"
            ssh(spec, f"mkdir -p {rd}")
            i = spec.index("-p")
            scp = ["scp", *spec[1:i], "-o", "BatchMode=yes", "-P", spec[i + 1]]
            files = [ck, ROOT / "logs_system1/t13v4/continue" / t / "continue_ref.json"]
            p = subprocess.run([*scp, *map(str, files), f"{spec[-1]}:{rd}/"], capture_output=True, text=True, timeout=900)
            rc, out, err = ssh(spec, f"cd {rd} && sha256sum {ck.name} continue_ref.json")
            got = {ln.split()[1]: ln.split()[0] for ln in out.splitlines() if len(ln.split()) == 2}
            if not self.tstep(t, "upload_checkpoint", p.returncode == 0 and got.get(ck.name) == ref["checkpoint_sha256"]
                              and got.get("continue_ref.json") == sha(files[1]), machine=m, scp_rc=p.returncode, cloud_sha256=got, err=p.stderr[-300:]):
                return False
            extra = f" --continue-checkpoint {rd}/{ck.name} --continue-ref {rd}/continue_ref.json"
        else:
            sm_dir = f"{L}/smoke_cloud_queue"
            rc, out, err = ssh(spec, f"{ENV}; mkdir -p {sm_dir}; nohup python3 system1/t13v4/cloud_smoke.py --run {t} --num-envs 4096 --iterations 10 "
                                     f"--out {sm_dir} > {sm_dir}/{t}.log 2>&1 < /dev/null & echo $! > {sm_dir}/{t}.pid; cat {sm_dir}/{t}.pid")
            if not self.tstep(t, "smoke_started", rc == 0, machine=m, pid=out, err=err):
                return False
            sm = None
            for _ in range(240):  # up to 2 h
                time.sleep(30)
                rc, out, _ = ssh(spec, f"cat {sm_dir}/{t}/smoke.json 2>/dev/null; kill -0 $(cat {sm_dir}/{t}.pid) 2>/dev/null && echo ALIVE")
                if out.startswith("{"):
                    sm = json.loads(out[:out.rindex("}") + 1])
                    break
                if "ALIVE" not in out:
                    break
            if sm is None:
                _, tail, _ = ssh(spec, f"tail -20 {sm_dir}/{t}.log")
                return self.tstep(t, "smoke", False, machine=m, note="no smoke.json", log_tail=tail)
            st = [dict(name=s["name"], rc=s["returncode"], seconds=s["seconds"], s_per_it=s["s_per_it_median_after_3"], nonfinite=s["nonfinite_keys"])
                  for s in sm["stages"]]
            if not self.tstep(t, "smoke", sm.get("passed"), machine=m, stages=st,
                              validation={k: v for k, v in (sm.get("validation") or {}).items() if k != "argv"}):
                return False
        if self.dry:
            return self.tstep(t, "launch", True, machine=m, dry=True)
        cmd = (f"{ENV}; mkdir -p {RUNS}; test ! -e {RUNS}/{t} && nohup python3 system1/t13v4/cloud_runs.py --run {t} --num-envs 4096 --gpu 0 --out {RUNS}{extra} "
               f"> {RUNS}/{t}.runner.log 2>&1 < /dev/null &\nP=$!\necho $P > {RUNS}/{t}.runner.pid\nsleep 5; kill -0 $P && echo RUNNER_ALIVE $P")
        rc, out, err = ssh(spec, cmd)
        ok = self.tstep(t, "launch", rc == 0 and "RUNNER_ALIVE" in out, machine=m, output=out, err=err, command=cmd.replace(ENV, "<env>"))
        if ok:
            with lock:
                h = json.loads(HOSTS.read_text())
                h.setdefault("runs", {})[t] = dict(machine=m, ssh=spec, dir=f"{RUNS}/{t}")
                tmp = HOSTS.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(h, indent=1) + "\n")
                tmp.replace(HOSTS)
        return ok

    # ------------------------------------------------------------------ one pass
    def tick(self):
        for m in MACHINES:
            if self.threads.get(m) is not None and self.threads[m].is_alive():
                continue
            M = self.s["machines"][m]
            if M["state"] == "busy":
                self.check_busy(m, M["task"])
                M = self.s["machines"][m]
                if M["state"] == "busy" or M["state"].startswith("held"):
                    continue
            if M["state"] in ("not up", "up, other processes", "up, not shut down (busy)", "free", "shutdown sent, still reachable", "shut down"):
                try:
                    rc = subprocess.run([*batch(self.spec[m]), "true"], capture_output=True, text=True, timeout=60).returncode
                except subprocess.TimeoutExpired:
                    rc = 255
                if rc != 0:
                    if M["state"] not in ("not up", "shut down"):
                        self.mset(m, state="not up")
                    continue
                ok, c = self.idle(m)
                if not ok:
                    if M["state"] != "up, other processes":
                        self.mset(m, state="up, other processes", check=c)
                        self.event("machine up but not idle", machine=m, check=c)
                    continue
                if M["state"] != "free":
                    with lock:
                        M.pop("waiting", None)
                        M.pop("waiting_since", None)
                    self.mset(m, state="free", task=None)
                    self.event("machine free", machine=m)
                self.assign(m)
        self.save(commit=True)

    def assign(self, m):
        M = self.s["machines"][m]
        T = self.s["tasks"]
        queued = [t for t in QUEUE if T[t]["state"] == "queued"]
        cand = [t for t in queued if ready(t) and not (t.startswith("c_out_") and M.get("no_sonic"))]
        if cand:
            t = cand[0]
            with lock:
                T[t].update(state="starting", machine=m, started=now())
                M.update(state="starting", task=t, since=now())
            self.event("start", task=t, machine=m)
            self.save(commit=True)
            th = threading.Thread(target=self.bring_up, args=(m, t), daemon=True)
            self.threads[m] = th
            th.start()
        elif queued:
            if M.get("waiting") != queued:
                M.update(waiting=queued, waiting_since=time.time())
                self.event("machine free, queued tasks not ready yet", machine=m, queued=queued)
            elif time.time() - M.get("waiting_since", time.time()) > WAIT_MAX:  # do not keep a paid card idle: shut down by the rule
                self.event("queued tasks still not ready after the wait limit: shutting down", machine=m, queued=queued, wait_s=WAIT_MAX)
                self.shutdown(m)
        else:
            self.event("queue empty: shutting down", machine=m)
            self.shutdown(m)

    def check_busy(self, m, t):
        d = ROOT / "logs_system1/t13v4/post_run" / t
        T = self.s["tasks"][t]
        if (d / "queue_release.json").exists():
            with lock:
                T.update(state="copied back, evaluating locally", released=now())
                self.s["machines"][m].update(state="free", task=None, since=now())
            self.event("released", task=t, machine=m)
        elif (d / "resume_check_failed.json").exists():
            rc = subprocess.run(["bash", "system1/t13v4/copyback.sh", t], cwd=ROOT, capture_output=True, text=True).returncode
            txt = (ROOT / "logs_system1/t13v4/copyback" / t / "copyback.txt").read_text() if (ROOT / "logs_system1/t13v4/copyback" / t / "copyback.txt").exists() else ""
            with lock:
                T.update(state="stopped by the resume check (held for Kevin)", ended=now(), note=f"copy-back rc {rc}: {txt.strip().splitlines()[-1] if txt else ''}")
                for o in QUEUE:
                    if o != "d_frontback" and self.s["tasks"][o]["state"] == "queued":
                        self.s["tasks"][o].update(state="held (resume check failed on another task)")
                self.s["machines"][m].update(state="free" if "sha256 match" in txt else "held (copy-back not verified)", task=None, since=now())
            self.event("resume check failed", task=t, machine=m, copyback=txt[-200:])
        else:
            st = ROOT / "results_system1/T13_runs" / t / "post_run_status.json"
            if st.exists() and json.loads(st.read_text()).get("state") == "stopped at copyback":
                with lock:
                    T.update(state="copy-back failed (held for Kevin)")
                    self.s["machines"][m].update(state="held (copy-back failed)", since=now())
                self.event("copy-back failed", task=t, machine=m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poll", type=float, default=600.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no shutdown, no launch, no state file / commit")
    a = ap.parse_args()
    q = Queue(a.dry_run)
    q.event("queue started", ready={t: ready(t) for t in QUEUE}, local_head=q.local_head)
    q.save(commit=True)
    while True:
        q.tick()
        if a.once:
            for th in q.threads.values():
                th.join()
            break
        if all(v["state"] not in ("queued", "starting", "running") for v in q.s["tasks"].values()) and \
                all(v["state"] in ("shut down", "not up") or v["state"].startswith("held") for v in q.s["machines"].values()):
            q.event("queue done")
            q.save(commit=True)
            break
        time.sleep(a.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
