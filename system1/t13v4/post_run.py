#!/usr/bin/env python3
"""T13 v4 section 6: everything after one run has ended, unattended, in the order used for d_push (2026-11-25):

  1 copyback   rsync the run back, per-file sha256 against the cloud (system1/t13v4/copyback.sh); pass = "sha256 match"
  2 finals_off Isaac d_final_5070 finals, noise off, +/-2 / +/-8 (run_finals_isaac.py); pass = both exit records complete
  3 dump       Isaac scene export of the nominal episode (run_isaac_dump.py); pass = rc 0 and the four dump files
  4 finals_on  Isaac finals, noise on (runs on the GPU while 5-8 run on the CPU); pass = both exit records complete
  5 onnx       actor ONNX export and check (onnx_check.py, 159 inputs); pass = float64 graph vs PyTorch <= 1e-10
  6 checks     MuJoCo initial states / observation v1 / FK / ONNX (check_sim2sim_v1.py); pass = all initial states and obs < 1e-4
  7 mujoco     MuJoCo closed loop, T10 setting, +/-2 / +/-8 (mj_eval.py --obs-v1); pass = both summaries complete
  8 videos     6 side-by-side videos (run_videos.sh, 2 at a time); pass = 6 x rc 0
  9 contact    geometric contact fractions (contact_stats_t13.py; after 4)
 10 package    results_system1/T13_runs/<run>/ tables (package_run.py), videos to Drive (T13_runs/<run>), keyframes, SHA256SUMS,
              commit and push (flock .git/s1_status.lock, as status_hourly.py)

C-out q* (step 6; Kevin 2026-11-26 after c_out_push stopped at step 294, a quantisation flip of one SONIC encoder token):
check_sim2sim_v1.py --qstar-inputs isaac = q* from SONIC ONNX on the Isaac-recorded SONIC inputs (< 1e-4) and the SONIC inputs
themselves (< 1e-4), checked separately, both must pass. --resume-from checks: continue a run stopped at step 6 from step 6
(steps 1-5 are not re-run; their outputs are verified again; finals_on is re-run only if its exit records are incomplete).
Any failed check stops the run's pipeline at that step (no retry, no protocol change); the state is written to
results_system1/T13_runs/<run>/post_run_status.json (committed) and shown by status_hourly.py. The Isaac steps of different runs
never overlap (flock logs_system1/t13v4/post_run/gpu.lock). Log: logs_system1/t13v4/post_run/<run>/.

    python3 system1/t13v4/post_run.py --run d_table
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")
CASE = {"d_push": "push", "d_table": "table", "d_table_pa": "table", "c_out_table": "table", "c_out_push": "push"}  # round 2: c_out_*
CASE.update(d_frontback="table", d_frontback_smoke="table")  # T14 front-back training task (Kevin 2026-11-26); _smoke: pre-training T10 gate
STATES_DIR = dict(d_frontback="system1/t14/frontback_train/states", d_frontback_smoke="system1/t14/frontback_train/states")
SO = "/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python"
SIM = "/home/kevin/.holosoma_deps/miniconda3/envs/sim2sim_s1/bin/python"
HS = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
CPU = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MUJOCO_GL="egl")


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


class Pipe:
    def __init__(self, run, dry=False, resume=None):
        import os
        self.resume = resume
        self.dry = dry  # --verify-existing: run no command, evaluate every check on the existing outputs (smoke test)
        self.run, self.task = run, CASE[run]
        self.scheme = "C-out" if run.startswith("c_out_") else "D"  # round 2 (task book section 8 item 2)
        self.L = ROOT / "logs_system1/t13v4/post_run" / run
        self.L.mkdir(parents=True, exist_ok=True)
        self.out = ROOT / "results_system1/T13_runs" / run
        self.out.mkdir(parents=True, exist_ok=True)
        self.status = dict(run=run, started=now(), steps=[], state="running")
        if resume:  # keep the earlier records (history); the steps from `resume` on are appended again
            old = json.loads((self.out / "post_run_status.json").read_text())
            assert old["state"] == f"stopped at {resume}", old["state"]
            self.status = old
            self.status["steps"].append(dict(step="resume", time=now(), from_step=resume, previous_state=old["state"], previous_stopped=old.get("stopped"),
                                             reason="Kevin 2026-11-26: C-out q* on the Isaac-recorded SONIC inputs (< 1e-4) and the SONIC inputs "
                                                    "(< 1e-4), separately, both must pass (check_sim2sim_v1.py --qstar-inputs isaac)"))
            self.status.update(state="running", resumed=now())
            self.status.pop("stopped", None)
        self.env = dict(os.environ)
        self.lock = threading.Lock()
        self.save()

    def save(self):
        if self.dry:
            return
        with self.lock:
            (self.out / "post_run_status.json").write_text(json.dumps(self.status, indent=1, ensure_ascii=False) + "\n")

    def sh(self, name, cmd, env=None, gpu=False):
        rec = dict(step=name, cmd=" ".join(map(str, cmd)), started=now())
        with self.lock:
            self.status["steps"].append(rec)
        self.save()
        e = dict(self.env, **(env or {}))
        if self.dry:
            rec.update(rc=0, ended=now(), dry=True)
            return rec
        lk = open(ROOT / "logs_system1/t13v4/post_run/gpu.lock", "w") if gpu else None
        if lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            rec["gpu_lock_acquired"] = now()
        try:
            with open(self.L / f"{name}.log", "w") as f:
                rc = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=e).returncode
        finally:
            if lk:
                fcntl.flock(lk, fcntl.LOCK_UN)
                lk.close()
        rec.update(rc=rc, ended=now())
        self.save()
        return rec

    def check(self, rec, ok, detail):
        rec.update(passed=bool(ok), detail=detail)
        self.save()
        if self.dry:
            print(("PASS " if ok else "FAIL ") + rec["step"], json.dumps(detail, default=str)[:300], flush=True)
            return
        if not ok:
            with self.lock:
                self.status.update(state=f"stopped at {rec['step']}", stopped=now())
            self.save()
            self.commit(f"stopped at {rec['step']}")
            raise SystemExit(1)

    def finals_ok(self, noise):
        sfx = "_noise_on" if noise == "on" else ""
        ex = [ROOT / "logs_system1/t13v4/finals" / f"{self.run}_pm{lv}{sfx}.exit.json" for lv in (2, 8)]
        d = [json.loads(p.read_text()) if p.exists() else {} for p in ex]
        return all(x.get("complete") for x in d), d

    def commit(self, what):
        with open(ROOT / ".git/s1_status.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            g = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True)  # noqa: E731
            if g("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() != "system1-sonic":
                return
            rel = str(self.out.relative_to(ROOT))
            g("add", rel)
            c = g("commit", "-m", f"system1(T13 runs): {self.run} post-run pipeline: {what}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", rel)
            if c.returncode == 0:
                g("pull", "--no-edit", "--no-rebase")
                g("push", "origin", "system1-sonic")

    def cloud_shutdown(self, rd, ck, copyback_txt, check_only=False):
        """T13 section 8 addendum (2026-11-26 03:20): after the verified copy-back, `shutdown` on the run's cloud machine (power
        off only, never a release), when all of these hold:
          (a) copyback.txt says "sha256 match" (every file of the cloud run directory, sha256 equal);
          (b) present locally: the final checkpoint, decision.json, launch.json, state.json, the logs (train.stdout.log,
              train/metrics.jsonl) and the run config (train/run_config.json);
          (c) the machine runs no other process: nothing but the platform's own (PID 1, supervisord and its direct children =
              sshd listener, jupyter-lab, tensorboard, proxy, autopanel; tensorboard's data server) and this check's own ssh
              session, and no GPU compute process; checked up to 3 times, 60 s apart.
        Otherwise no shutdown; the reason is written to post_run_status.json (cloud_shutdown) and committed. Either way the local
        evaluation goes on (it only uses the local copy). The shutdown time is written to post_run_status.json."""
        import time
        v = json.loads((ROOT / "logs_system1/t13v4/cloud/hosts.json").read_text())["runs"][self.run]
        ssh = [*v["ssh"][:-1], "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", v["ssh"][-1]]
        need = [ck, rd / "decision.json", rd / "launch.json", rd / "state.json", rd / "train.stdout.log", rd / "train/metrics.jsonl", rd / "train/run_config.json"]
        missing = [str(p.relative_to(rd)) for p in need if not p.exists()]
        rec = dict(step="cloud_shutdown", machine=v.get("machine"), started=now(), copyback_sha256_match="sha256 match" in copyback_txt, missing_local=missing, checks=[])
        if not check_only:
            with self.lock:
                self.status["steps"].append(rec)
                self.status["cloud_shutdown"] = rec
            self.save()
        ok = rec["copyback_sha256_match"] and not missing
        tries = 1 if (check_only or self.dry) else (3 if ok else 0)  # check-only / dry: one read-only look even when (a) or (b) fails
        for k in range(tries):
            c = self.cloud_processes(ssh)
            rec["checks"].append(c)
            if c.get("error") is None and not c["others"] and not c["gpu_compute"]:
                break
            if k < tries - 1:
                time.sleep(60)
        last = rec["checks"][-1] if rec["checks"] else {}
        ok = ok and last.get("error") is None and not last.get("others") and not last.get("gpu_compute")
        rec["gate_passed"] = bool(ok)
        if not ok or check_only or self.dry:
            rec.update(done=False, ended=now(), reason=None if ok else ("copy-back not verified" if not rec["copyback_sha256_match"] else
                       f"missing locally: {missing}" if missing else f"other processes / GPU use / check error on the machine: {last}"))
            if check_only or self.dry:
                print(json.dumps(rec, indent=1, ensure_ascii=False), flush=True)
                return rec
            self.save()
            self.commit(f"cloud machine NOT shut down ({rec['reason'][:80]})")
            return rec
        p = subprocess.run([*ssh, "shutdown"], capture_output=True, text=True, timeout=120)
        rec.update(command="shutdown", shutdown_sent=True, shutdown_time=now(), rc=p.returncode, stdout=p.stdout[-300:], stderr=p.stderr[-300:])
        time.sleep(60)
        q = subprocess.run([*ssh, "true"], capture_output=True, text=True, timeout=60)
        rec.update(reachable_60s_after=q.returncode == 0, done=q.returncode != 0, ended=now())
        self.save()
        self.commit(f"cloud machine {v.get('machine')} shut down {rec['shutdown_time']}" if rec["done"] else "cloud machine shutdown sent, still reachable")
        return rec

    @staticmethod
    def cloud_processes(ssh):
        """read-only: the processes and GPU compute processes on the machine, minus the platform's own and this session's."""
        cmd = "echo SELF $$ $PPID; ps -eo pid=,ppid=,args=; echo __GPU__; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1; echo __END__"
        try:
            p = subprocess.run([*ssh, cmd], capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return dict(time=now(), error="ssh timeout")
        if p.returncode != 0 or "__END__" not in p.stdout:
            return dict(time=now(), error=f"ssh rc {p.returncode}: {p.stderr.strip()[-200:]}")
        lines = p.stdout.splitlines()
        self_pid, sess = (int(x) for x in lines[0].split()[1:3])
        gi = lines.index("__GPU__")
        procs = []
        for ln in lines[1:gi]:
            f = ln.split(None, 2)
            if len(f) >= 2 and f[0].isdigit():
                procs.append((int(f[0]), int(f[1]), f[2] if len(f) > 2 else ""))
        sup = {pid for pid, pp, a in procs if pp == 1 and "supervisord" in a}
        base = {1} | sup | {pid for pid, pp, a in procs if pp in sup}
        base |= {pid for pid, pp, a in procs if pp in base and "tensorboard_data_server" in a}
        mine = {self_pid, sess} | {pid for pid, pp, a in procs if pp == self_pid}
        others = [dict(pid=pid, ppid=pp, args=a[:160]) for pid, pp, a in procs if pid not in base and pid not in mine]
        gpu = [x.strip() for x in lines[gi + 1:lines.index("__END__")] if x.strip()]
        if any("fail" in x.lower() or "error" in x.lower() for x in gpu):
            return dict(time=now(), error=f"nvidia-smi: {gpu[:2]}")
        return dict(time=now(), others=others, gpu_compute=gpu, platform_processes=len(base))

    def go(self):
        r, t = self.run, self.task
        rd = ROOT / "logs_system1/t13v4/runs" / r
        if self.resume == "checks":
            return self.go_from_checks(rd)
        # 1 copyback
        rec = self.sh("copyback", ["bash", "system1/t13v4/copyback.sh", r])
        txt = (ROOT / "logs_system1/t13v4/copyback" / r / "copyback.txt").read_text()
        dec = json.loads((rd / "decision.json").read_text()) if (rd / "decision.json").exists() else {}
        ck = rd / "train" / f"model_{dec.get('final_checkpoint_iteration', -1):05d}.pt"
        self.check(rec, "sha256 match" in txt and ck.exists(), dict(copyback=txt.strip().splitlines()[-1], final_checkpoint=ck.name, exists=ck.exists()))
        # 1b shut the cloud machine down (task book T13 section 8, 2026-11-26 03:20); never stops the local evaluation
        if r.startswith("c_out_") or r == "d_frontback":  # d_frontback: T14 front-back task, "按 T13 的规则拷回、核对、关机"
            try:
                self.cloud_shutdown(rd, ck, txt)
            except Exception as ex:  # noqa: BLE001  (no shutdown; reported; the local evaluation goes on)
                with self.lock:
                    self.status["cloud_shutdown"] = dict(step="cloud_shutdown", done=False, ended=now(), reason=f"error in the shutdown step: {ex!r}"[:300])
                self.save()
                self.commit("cloud machine NOT shut down (error in the shutdown step)")
        a = json.loads((rd / "launch.json").read_text())["stages"][-1]["argv"]  # cloud_launch.py: stages = warm-up, training
        seed = int(a[a.index("--seed") + 1])
        # 2 finals noise off
        rec = self.sh("finals_off", ["python3", "system1/t13v4/run_finals_isaac.py", "--run", r], gpu=True)
        ok, d = self.finals_ok("off")
        self.check(rec, ok, d)
        # 3 scene export
        rec = self.sh("dump", ["python3", "system1/t13v4/run_isaac_dump.py", r], gpu=True)
        dd = ROOT / "logs_system1/t13v4/isaac_dump" / r / "dump"
        files = ["scene.json", "collision_convexes.npz", "runtime_reference.npz", "nominal.sim2sim.npz"]
        ex = json.loads((ROOT / "logs_system1/t13v4/isaac_dump" / r / "exit.json").read_text())
        self.check(rec, ex["rc"] == 0 and all((dd / f).exists() for f in files), dict(rc=ex["rc"], files=[f for f in files if (dd / f).exists()]))
        # 4 noise on, in parallel with the CPU steps
        on = {}
        th = threading.Thread(target=lambda: on.update(rec=self.sh("finals_on", ["python3", "system1/t13v4/run_finals_isaac.py", "--run", r, "--noise", "on"], gpu=True)))
        th.start()
        # 5 onnx
        onnx = ROOT / "exports_system1/t13v4/onnx" / f"{r}_t13v4.onnx"
        rec = self.sh("onnx", [SIM, "-B", "system1/sim2sim/onnx_check.py", "--out", f"logs_system1/t13v4/sim2sim/onnx_{r}", "--onnx-dir", "exports_system1/t13v4/onnx",
                               "--extra-checkpoint", f"{self.scheme},{t},{seed},{r}_t13v4,{ck.relative_to(ROOT)},v1"], env=CPU)
        rows = list(csv.DictReader(open(ROOT / f"logs_system1/t13v4/sim2sim/onnx_{r}/onnx_check.csv")))
        f64 = max(float(x["float64_onnx_vs_torch"]) for x in rows)
        self.check(rec, rec["rc"] == 0 and onnx.exists() and f64 <= 1e-10, dict(float64_graph_vs_torch=f64, float32=max(float(x["max_abs_diff"]) for x in rows)))
        self.go_rest(ck, seed, dd, onnx, th, on)

    def go_from_checks(self, rd):
        """--resume-from checks: verify the outputs of steps 1-5 again (nothing re-run), re-run finals_on only if incomplete,
        then steps 6-10 exactly as go()."""
        r = self.run
        txt = (ROOT / "logs_system1/t13v4/copyback" / r / "copyback.txt").read_text()
        dec = json.loads((rd / "decision.json").read_text())
        ck = rd / "train" / f"model_{dec.get('final_checkpoint_iteration', -1):05d}.pt"
        a = json.loads((rd / "launch.json").read_text())["stages"][-1]["argv"]
        seed = int(a[a.index("--seed") + 1])
        dd = ROOT / "logs_system1/t13v4/isaac_dump" / r / "dump"
        ex = json.loads((ROOT / "logs_system1/t13v4/isaac_dump" / r / "exit.json").read_text())
        onnx = ROOT / "exports_system1/t13v4/onnx" / f"{r}_t13v4.onnx"
        rows = list(csv.DictReader(open(ROOT / f"logs_system1/t13v4/sim2sim/onnx_{r}/onnx_check.csv")))
        f64 = max(float(x["float64_onnx_vs_torch"]) for x in rows)
        off_ok, _ = self.finals_ok("off")
        cs = self.status.get("cloud_shutdown") or {}
        files = ["scene.json", "collision_convexes.npz", "runtime_reference.npz", "nominal.sim2sim.npz"]
        d = dict(copyback_sha256_match="sha256 match" in txt, final_checkpoint=ck.name, checkpoint_exists=ck.exists(),
                 cloud_shutdown_done=cs.get("done"), finals_off_complete=off_ok, dump_rc=ex["rc"], dump_files=all((dd / f).exists() for f in files),
                 onnx_exists=onnx.exists(), onnx_float64=f64)
        rec = dict(step="resume_verify", started=now())
        with self.lock:
            self.status["steps"].append(rec)
        self.check(rec, d["copyback_sha256_match"] and d["checkpoint_exists"] and off_ok and ex["rc"] == 0 and d["dump_files"] and onnx.exists() and f64 <= 1e-10, d)
        on_ok, _ = self.finals_ok("on")
        on, th = {}, None
        if on_ok:
            on["rec"] = next(x for x in reversed(self.status["steps"]) if x.get("step") == "finals_on")
        else:
            th = threading.Thread(target=lambda: on.update(rec=self.sh("finals_on", ["python3", "system1/t13v4/run_finals_isaac.py", "--run", r, "--noise", "on"], gpu=True)))
            th.start()
        self.go_rest(ck, seed, dd, onnx, th, on)

    def go_rest(self, ck, seed, dd, onnx, th, on):
        r, t = self.run, self.task
        # 6 MuJoCo checks
        rec = self.sh("checks", [SIM, "-B", "system1/t13v4/check_sim2sim_v1.py", "--run", r, "--task", t, "--onnx", onnx, "--out", f"logs_system1/t13v4/sim2sim/checks_{r}",
                                 *(["--scheme", "C-out", "--qstar-inputs", "isaac"] if self.scheme == "C-out" else [])], env=CPU)
        s = json.loads((ROOT / f"logs_system1/t13v4/sim2sim/checks_{r}/summary.json").read_text())
        cout_ok = self.scheme != "C-out" or (s["cout"]["sonic_inputs_passed"] and s["cout"]["q_star_passed"])  # round 2: SONIC inputs and q* (T10c thresholds)
        self.check(rec, s["initial_states"]["all_passed"] and s["observations"]["obs_passed"] and s["initial_states"]["n_states"] == 102 and cout_ok,
                   dict(initial_n=s["initial_states"]["n_states"], initial_max=s["initial_states"]["max"], obs_max=s["observations"]["obs_max_abs_diff"],
                        **({"cout": s["cout"]} if self.scheme == "C-out" else {})))
        # 7 MuJoCo closed loop
        mj = ROOT / "logs_system1/t13v4/mj_finals"
        recs = []
        ts = []
        for lv, cores in ((2, "0-3"), (8, "4-7")):
            cmd = ["taskset", "-c", cores, SIM, "-B", "system1/sim2sim/mj_eval.py", "--task", t, "--scheme", self.scheme, "--seed", seed, "--level", lv,
                   "--dump", dd.relative_to(ROOT), "--states", f"{STATES_DIR.get(r, 'system1/t13v4/states_t14r6C')}/{t}_pm{lv}_states.json", "--obs-v1", "--onnx", onnx,
                   "--checkpoint", ck.relative_to(ROOT), "--out", mj / f"{r}_pm{lv}"]
            ts.append(threading.Thread(target=lambda c=cmd, lv=lv: recs.append(self.sh(f"mujoco_pm{lv}", c, env=dict(CPU, OMP_NUM_THREADS="1")))))
        for x in ts:
            x.start()
        for x in ts:
            x.join()
        comp = [json.loads((mj / f"{r}_pm{lv}" / "summary.json").read_text()).get("complete") if (mj / f"{r}_pm{lv}" / "summary.json").exists() else False for lv in (2, 8)]
        self.check(recs[-1], all(comp), dict(complete=comp))
        # 8 videos
        V = ROOT / "exports_system1/T13_videos" / r
        rec = self.sh("videos", ["bash", "system1/t13v4/run_videos.sh", r, V.relative_to(ROOT), "2"])
        rc = (V / "logs/rc.txt").read_text().splitlines() if (V / "logs/rc.txt").exists() else []
        self.check(rec, len(rc) == 6 and all(" rc 0 " in x for x in rc), rc)
        # 4 (join) noise on
        if th is not None:
            th.join()
        ok, d = self.finals_ok("on")
        self.check(on["rec"], ok, d)
        # 9 contact
        rec = self.sh("contact", ["taskset", "-c", "0-15", HS, "-B", "system1/t13v4/contact_stats_t13.py", "--run", r, "--workers", "6"], env=dict(CPU, OMP_NUM_THREADS="1"))
        self.check(rec, rec["rc"] == 0 and (self.out / "contact_summary.csv").exists(), "contact_summary.csv")
        # 10 package
        rec = self.sh("package", [SO, "-B", "system1/t13v4/package_run.py", "--run", r])
        summ = list(csv.DictReader(open(self.out / "summary.csv")))
        self.check(rec, rec["rc"] == 0 and len(summ) == 6 and all(x["status"] == "complete" for x in summ),
                   {x["group"]: f"{x.get('n_completed')}/{x.get('n_perturbed')}" for x in summ})
        mp4 = sorted(V.glob("*.mp4"))
        if self.dry:
            print("videos", len(mp4))
        rec = self.sh("drive", ["python3", "system1/transfer/drive_upload.py", "--task", f"T13_runs/{r}", *[p.relative_to(ROOT) for p in mp4]])
        self.check(rec, rec["rc"] == 0 and (self.out / "drive_manifest.csv").exists(), f"{len(mp4)} videos")
        (self.out / "videos").mkdir(exist_ok=True)
        for p in list(V.glob("*.keyframes.jpg")) + list(V.glob("*.json")):
            (self.out / "videos" / p.name).write_bytes(p.read_bytes())
        if self.dry:
            print("VERIFY_DONE", r)
            return
        with self.lock:
            self.status.update(state="done (README_CN.md by hand; numbers in summary.csv)", ended=now(),
                               results={x["group"]: f"{x.get('n_completed')}/{x.get('n_perturbed')}" for x in summ})
        self.save()
        subprocess.run(f"cd {self.out} && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS", shell=True)
        self.commit("evaluation done: " + ", ".join(f"{k} {v}" for k, v in self.status["results"].items()))
        print("POST_RUN_DONE", r, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=sorted(CASE))
    ap.add_argument("--verify-existing", action="store_true", help="smoke test: no command is run; every check on the existing outputs")
    ap.add_argument("--resume-from", choices=("checks",), default=None, help="continue a run stopped at step 6 (checks) from step 6 "
                    "(Kevin 2026-11-26, C-out q* rule); steps 1-5 verified, not re-run")
    ap.add_argument("--shutdown-check-only", action="store_true", help="smoke test: the cloud-shutdown gate only (read-only ssh, never a shutdown)")
    a = ap.parse_args()
    if a.shutdown_check_only:
        p = Pipe(a.run, dry=True)
        rd = ROOT / "logs_system1/t13v4/runs" / a.run
        cb = ROOT / "logs_system1/t13v4/copyback" / a.run / "copyback.txt"
        dec = json.loads((rd / "decision.json").read_text()) if (rd / "decision.json").exists() else {}
        p.cloud_shutdown(rd, rd / "train" / f"model_{dec.get('final_checkpoint_iteration', -1):05d}.pt", cb.read_text() if cb.exists() else "", check_only=True)
        return 0
    Pipe(a.run, dry=a.verify_existing, resume=a.resume_from).go()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
