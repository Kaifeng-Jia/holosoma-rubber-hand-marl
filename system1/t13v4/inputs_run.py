#!/usr/bin/env python3
"""T13 v4 section 5: inputs_sha256.csv of one finished run's return package (read only): final checkpoint, actor ONNX, test
states (+/-2, +/-8), T14 reference manifest, robot / object URDF, placement proof (Push), the Isaac scene export used by MuJoCo;
paths from the Isaac finals' launch argv and run_manifest. Also copies the copy-back sha256 lists and the evaluator switch diff
(commit of the switch vs its parent) into records/.

    python3 system1/t13v4/inputs_run.py --run d_table
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src/holosoma"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    out = ROOT / "results_system1/T13_runs" / a.run
    F = ROOT / "logs_system1/t13v4/finals"
    man = json.loads((F / f"{a.run}_pm2/run_manifest.json").read_text())
    dec = json.loads((ROOT / "logs_system1/t13v4/runs" / a.run / "decision.json").read_text())
    rows = []

    def add(kind, p):
        p = Path(p)
        rows.append(dict(kind=kind, path=rel(p), sha256=sha(p)))
    add(f"checkpoint (final, file iteration {dec['final_checkpoint_iteration']})", man["checkpoint"])
    assert sha(man["checkpoint"]) == man["checkpoint_sha256"]
    add("actor ONNX", ROOT / f"exports_system1/t13v4/onnx/{a.run}_t13v4.onnx")
    for lv in (2, 8):
        argv = json.loads((F / f"{a.run}_pm{lv}.launch.json").read_text())["argv"]
        add(f"test states +/-{lv}", argv[argv.index("--states") + 1])
        if "--push-placement-proof" in argv and lv == 2:
            pp = argv[argv.index("--push-placement-proof") + 1]
            add("placement proof", pp if Path(pp).is_absolute() else ROOT / pp)
    add("T14 reference manifest", ROOT / man["system1_t14_reference"]["manifest"])
    add("robot URDF", SRC / "holosoma/data/robots" / man["robot_urdf"])
    add("object URDF", SRC / man["object_urdf"])
    for n, k in (("scene.json", "Isaac scene export"), ("collision_convexes.npz", "Isaac scene convexes"),
                 ("runtime_reference.npz", "runtime reference (as sampled by Isaac)"), ("nominal.sim2sim.npz", "Isaac nominal record")):
        add(k, ROOT / "logs_system1/t13v4/isaac_dump" / a.run / "dump" / n)
    with open(out / "inputs_sha256.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "path", "sha256"], quoting=csv.QUOTE_NONNUMERIC)
        w.writeheader()
        w.writerows(rows)
    cb = ROOT / "logs_system1/t13v4/copyback" / a.run
    for n in ("sha256_local.txt", "sha256_cloud.txt"):
        if (cb / n).exists():
            shutil.copy2(cb / n, out / "records" / f"copyback_{n}")
    # the evaluator switch (--obs-v1-noise, default off) as committed
    c = subprocess.run(["git", "log", "--format=%H", "-1", "-S", "--obs-v1-noise", "--", "scripts/evaluate_paper_supplement.py"], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout.strip()
    d = subprocess.run(["git", "diff", f"{c}~1", c, "--", "scripts/evaluate_paper_supplement.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    (out / "records" / "code_switches.diff").write_text(f"# commit {c}\n" + d)
    for r in rows:
        print(r["kind"], r["path"], r["sha256"][:12])


if __name__ == "__main__":
    main()
