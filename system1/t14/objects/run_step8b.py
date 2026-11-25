#!/usr/bin/env python3
"""T14 step 8 rerun (Kevin 2026-11-25, item 1): the 57 solved step-8 candidates re-solved at level 1.0 with the person-side frame
fix (role_solve.py --human-frame target), evaluated as step 8 (shared_eval.py, same gates file) and re-judged with the round-6
support windows (step8_rejudge.py --logdir). Same candidates (cand.json), gates (gates_m1p00.json) and source solves as step 8;
only the role solves are new. c22 (push), a failed step-8 solve, is not among the 57 and is not retried.
Tmux, CPU only, at most --parallel worker processes, each started only when MemAvailable - need >= 8 GB.
Record: logs_system1/t14/step8b/driver.json.

    sonic_s1/bin/python -B system1/t14/objects/run_step8b.py --parallel 6
"""
import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SO = "/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python"
HS = "/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python"
OLD = ROOT / "logs_system1/t14/step8"
L = ROOT / "logs_system1/t14/step8b"
EXP = ROOT / "exports_system1/T14_references/step8b"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
           __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
TAGS = ("smallboxb", "largeboxb", "suitcaseb", "smallboxc", "largeboxc", "suitcasec", "push", "pull")
BOX = dict(smallboxb=("smallbox", "sub10_smallbox_007"), largeboxb=("largebox", "sub3_largebox_024"), suitcaseb=("suitcase", "sub1_suitcase_065"),
           smallboxc=("smallbox", "sub7_smallbox_027"), largeboxc=("largebox", "sub7_largebox_027"), suitcasec=("suitcase", "sub14_suitcase_016"))
WBT = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
TABLE = {"push": WBT / "objects_widetable_plan5_training.urdf", "pull": WBT / "objects_widetable_plan5_pull_training.urdf"}
log = dict(steps=[])
lock = threading.Lock()
slots = None


def avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2


def save():
    with lock:
        (L / "driver.json").write_text(json.dumps(log, indent=1) + "\n")


def run(cmd, name, need, logf):
    slots.acquire()
    try:
        while True:
            with lock:
                ok = avail() - need >= 8.0
                if ok:
                    rec = dict(name=name, cmd=" ".join(map(str, cmd)), started=time.strftime("%T"), avail_gb=round(avail(), 1))
                    log["steps"].append(rec)
                    logf.parent.mkdir(parents=True, exist_ok=True)
                    f = open(logf, "w")
                    p = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=ENV)
                    time.sleep(3)
            if ok:
                break
            time.sleep(10)
        save()
        rc = p.wait()
        f.close()
        rec.update(rc=rc, done=time.strftime("%T"))
        save()
        return rc
    finally:
        slots.release()


def candidate(tag, cdir_old):
    cs = json.loads((OLD / f"cands/{tag}/candidates.json").read_text())
    cd = L / f"cands/{tag}/{cdir_old.name}"
    cd.mkdir(parents=True, exist_ok=True)
    for f in ("cand.json", "gates_m1p00.json"):
        if not (cd / f).exists():
            shutil.copy2(cdir_old / f, cd / f)
    d = cd / "m1p00"
    jobs = []
    for r in (0, 1):
        if (d / f"r{r}/run_record.json").exists():
            continue
        cmd = [HS, "-B", "system1/t14/objects/role_solve.py", "--family", "box" if tag in BOX else "table", "--mode", "role", "--tag", tag,
               "--cand", cd / "cand.json", "--role", r, "--m", 1.0, "--out", d / f"r{r}", "--human-frame", "target"]
        if tag in BOX:
            cmd += ["--object", BOX[tag][0], "--sequence", BOX[tag][1], "--source", cs["source"]]
        else:
            cmd += ["--task", tag, "--target-urdf", TABLE[tag]]
        jobs.append((cmd, f"solve_{tag}_{cd.name}_r{r}", 2.0, d.parent / f"m1p00_r{r}.log"))
    out = [None] * len(jobs)
    ts = [threading.Thread(target=lambda i=i, j=j: out.__setitem__(i, run(*j))) for i, j in enumerate(jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if all((d / f"r{r}/run_record.json").exists() for r in (0, 1)) and not (d / "eval.json").exists():
        run([SO, "-B", "system1/t14/objects/shared_eval.py", "--cand-dir", d, "--source", cs["source"], "--gates", cd / "gates_m1p00.json",
             "--out", d / "eval.json", "--pair-out", EXP / f"{tag}_{cd.name}_m1p00/pair_fps30.npz"], f"eval_{tag}_{cd.name}", 2.5, cd / "eval_m1p00.log")


def main():
    global slots
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", type=int, default=6)
    a = ap.parse_args()
    slots = threading.BoundedSemaphore(a.parallel)
    L.mkdir(parents=True, exist_ok=True)
    log.update(started=time.strftime("%F %T"), parallel=a.parallel, rule=__doc__)
    todo = [(tag, c.parent.parent) for tag in TAGS for c in sorted((OLD / f"cands/{tag}").glob("c*/m1p00/eval.json"))]
    log["candidates"] = [f"{t}/{c.name}" for t, c in todo]
    assert len(todo) == 57, len(todo)
    save()
    ts = [threading.Thread(target=candidate, args=(t, c)) for t, c in todo]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    log["done"] = time.strftime("%F %T")
    save()
    print("STEP8B_DONE", flush=True)


if __name__ == "__main__":
    main()
