#!/usr/bin/env python3
"""T14 step 7 layer 1: run single_object.py for every layer1_choice.json entry (classes b, c), at most --parallel at a time,
starting one only when MemAvailable - 2.0 GB (measured peak 1.94 GB on the smoke) stays >= 8 GB (Kevin's rule). Run in tmux.
Return codes in <out>/solves.json."""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HS = "/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python"


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--choice", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=2)
    a = ap.parse_args()
    ch = json.loads(a.choice.read_text())["choice"]
    jobs = [(e["object"], e["sequence"], f"{e['object']}{cls}") for cls in "bc" for e in ch[cls]]
    a.out.mkdir(parents=True, exist_ok=True)
    log = dict(started=time.strftime("%F %T"), jobs={})
    running = {}
    env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    while jobs or running:
        for tag, p in list(running.items()):
            if p.poll() is not None:
                log["jobs"][tag].update(rc=p.returncode, done=time.strftime("%T"))
                del running[tag]
        if jobs and len(running) < a.parallel and avail_gb() - 2.0 >= 8.0:
            obj, seq, tag = jobs.pop(0)
            cmd = [HS, "-B", "system1/t14/objects/single_object.py", "--object", obj, "--sequence", seq, "--tag", tag, "--out", str(a.out / f"single_{tag}_C")]
            f = open(a.out / f"single_{tag}_C.log", "w")
            running[tag] = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
            log["jobs"][tag] = dict(cmd=" ".join(cmd), started=time.strftime("%T"), avail_gb_at_start=round(avail_gb(), 2), pid=running[tag].pid)
        (a.out / "solves.json").write_text(json.dumps(log, indent=1) + "\n")
        time.sleep(10)
    log["finished"] = time.strftime("%F %T")
    (a.out / "solves.json").write_text(json.dumps(log, indent=1) + "\n")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
