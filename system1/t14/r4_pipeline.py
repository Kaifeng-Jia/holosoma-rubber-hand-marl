#!/usr/bin/env python3
"""T14 round 4 orchestration (run inside tmux after run_r4.sh): wait for the solves; evaluate + render every task x version
(2 at a time); baseline metrics for round 3 / current / OmniRetarget as is; Push leg check (A, B, round 3, current at 1.0);
decisions and return codes in logs_system1/t14/r4/pipeline.json."""
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "logs_system1/t14/r4"
SO = "/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python"
log = dict(started=time.strftime("%F %T"), steps=[])


def save():
    (S / "pipeline.json").write_text(json.dumps(log, indent=1) + "\n")


def available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return 0.0


GATE = __import__("threading").Lock()


def sh(cmd, name, need_gb=4.5):
    # Kevin's rule: keep >= 8 GB available; start a job only when available >= 8 + its expected peak (measured: render ~4 GB)
    with GATE:
        waited = 0
        while available_gb() < 8.0 + need_gb:
            time.sleep(20)
            waited += 20
        time.sleep(30)  # let the job just started by the other worker reach its working size before the next check
    log.setdefault("waits_s", {})[name] = waited
    with open(S / f"{name}.log", "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    log["steps"].append(dict(name=name, rc=rc, at=time.strftime("%T")))
    save()
    return rc


def main():
    while "ALL_DONE" not in (S / "run.log").read_text():
        time.sleep(15)
    log["solves_done"] = time.strftime("%T")
    save()
    jobs = [(t, v) for v in ("A", "B") for t in ("table", "push", "bin", "chair", "pull")]
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda tv: sh(["bash", "system1/t14/finish_r4.sh", tv[0], tv[1]], f"fin_{tv[0]}_{tv[1]}"), jobs))
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda t: sh(["bash", "system1/t14/finish_r4_baseline.sh", t], f"baseline_{t}"), ("table", "bin", "chair", "push", "pull")))
    out = ROOT / "results_system1/T14_retargeting/step4/push_legs"
    W = "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/a1"
    for name, single in (("A", S / "single_push_A/push_new_qpos.npz"), ("B", S / "single_push_B/push_new_qpos.npz"),
                         ("round3", ROOT / "logs_system1/t14/r3/single_push/push_new_qpos.npz")):
        sh([SO, "-B", "system1/t14/push_legs.py", "--single", str(single), "--m", "1.0", "--out", str(out / f"push_legs_{name}.json")], f"push_legs_{name}")
    log["done"] = time.strftime("%T")
    save()
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
