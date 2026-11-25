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
CASE = {"d_push": "push", "d_table": "table", "d_table_pa": "table"}
SO = "/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python"
SIM = "/home/kevin/.holosoma_deps/miniconda3/envs/sim2sim_s1/bin/python"
HS = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
CPU = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MUJOCO_GL="egl")


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


class Pipe:
    def __init__(self, run, dry=False):
        import os
        self.dry = dry  # --verify-existing: run no command, evaluate every check on the existing outputs (smoke test)
        self.run, self.task = run, CASE[run]
        self.L = ROOT / "logs_system1/t13v4/post_run" / run
        self.L.mkdir(parents=True, exist_ok=True)
        self.out = ROOT / "results_system1/T13_runs" / run
        self.out.mkdir(parents=True, exist_ok=True)
        self.status = dict(run=run, started=now(), steps=[], state="running")
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

    def go(self):
        r, t = self.run, self.task
        rd = ROOT / "logs_system1/t13v4/runs" / r
        # 1 copyback
        rec = self.sh("copyback", ["bash", "system1/t13v4/copyback.sh", r])
        txt = (ROOT / "logs_system1/t13v4/copyback" / r / "copyback.txt").read_text()
        dec = json.loads((rd / "decision.json").read_text()) if (rd / "decision.json").exists() else {}
        ck = rd / "train" / f"model_{dec.get('final_checkpoint_iteration', -1):05d}.pt"
        self.check(rec, "sha256 match" in txt and ck.exists(), dict(copyback=txt.strip().splitlines()[-1], final_checkpoint=ck.name, exists=ck.exists()))
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
                               "--extra-checkpoint", f"D,{t},{seed},{r}_t13v4,{ck.relative_to(ROOT)},v1"], env=CPU)
        rows = list(csv.DictReader(open(ROOT / f"logs_system1/t13v4/sim2sim/onnx_{r}/onnx_check.csv")))
        f64 = max(float(x["float64_onnx_vs_torch"]) for x in rows)
        self.check(rec, rec["rc"] == 0 and onnx.exists() and f64 <= 1e-10, dict(float64_graph_vs_torch=f64, float32=max(float(x["max_abs_diff"]) for x in rows)))
        # 6 MuJoCo checks
        rec = self.sh("checks", [SIM, "-B", "system1/t13v4/check_sim2sim_v1.py", "--run", r, "--task", t, "--onnx", onnx, "--out", f"logs_system1/t13v4/sim2sim/checks_{r}"], env=CPU)
        s = json.loads((ROOT / f"logs_system1/t13v4/sim2sim/checks_{r}/summary.json").read_text())
        self.check(rec, s["initial_states"]["all_passed"] and s["observations"]["obs_passed"] and s["initial_states"]["n_states"] == 102,
                   dict(initial_n=s["initial_states"]["n_states"], initial_max=s["initial_states"]["max"], obs_max=s["observations"]["obs_max_abs_diff"]))
        # 7 MuJoCo closed loop
        mj = ROOT / "logs_system1/t13v4/mj_finals"
        recs = []
        ts = []
        for lv, cores in ((2, "0-3"), (8, "4-7")):
            cmd = ["taskset", "-c", cores, SIM, "-B", "system1/sim2sim/mj_eval.py", "--task", t, "--scheme", "D", "--seed", seed, "--level", lv,
                   "--dump", dd.relative_to(ROOT), "--states", f"system1/t13v4/states_t14r6C/{t}_pm{lv}_states.json", "--obs-v1", "--onnx", onnx,
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
    a = ap.parse_args()
    Pipe(a.run, dry=a.verify_existing).go()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
