#!/usr/bin/env python3
"""T10b B step 5, Isaac side: final evaluation of a pilot's last checkpoint on the laptop 5070 with the d_final_5070 protocol,
+/-2 and +/-8: exactly the argv of the original policy's laptop-5070 final evaluation at that level, with only --checkpoint,
--model-id and --output-dir replaced. Plus one scene export (nominal episode, --limit 1 --sim2sim-dump, as
system1/sim2sim/run_isaac_dumps.py) to check that the evaluation scene of the pilot checkpoint equals the original's.
Two processes at a time (taskset 16-23 / 24-31 as the finals). Records <name>.launch.json / .exit.json; an existing output
directory is never reused.

    python3 system1/t10b/run_finals.py --run cin_bucket_seed721_rand --checkpoint exports_system1/t10b_pilot/cin_bucket_seed721_rand/model_12000.pt \
        --out logs_system1/t10b/finals
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")
SLOTS = ("16-23", "24-31")
# the original policy's laptop-5070 final evaluation per pilot run and level (T10 / T10b A use the same records)
ORIGINAL = {
    "cin_bucket_seed721_rand": {
        2: ROOT / "logs_system1/post_train/final/cin_bucket_m1kg_seed721_fresh12000_pm2.launch.json",
        8: ROOT / "logs_system1/post_train/final/cin_bucket_m1kg_seed721_fresh12000_pm8.launch.json",
    },
    "d_table_seed721_rand": {
        2: ROOT / "logs_system1/post_train/d_curve/D_table_paper_T-07_seed721/it12000_pm2.launch.json",
        8: ROOT / "logs_system1/post_train/d_paper_final/D_table_paper_T-07_seed721/it12000_pm8.launch.json",
    },
}


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def job_argv(src: Path, checkpoint: Path, name: str, out: Path, dump: bool):
    L = json.loads(src.read_text())
    argv = list(L["argv"])
    if argv[0] == "taskset":
        argv = argv[3:]
    argv[argv.index("--checkpoint") + 1] = str(checkpoint.resolve())
    argv[argv.index("--model-id") + 1] = "t10b_final__" + name
    if dump:
        argv[argv.index("--output-dir") + 1] = str(out / name / "eval")
        argv += ["--limit", "1", "--sim2sim-dump", str(out / name / "dump")]
    else:
        argv[argv.index("--output-dir") + 1] = str(out / name)
    return argv, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=tuple(ORIGINAL), required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--levels", default="2,8")
    ap.add_argument("--no-dump", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout  # noqa: E731
    ck_sha = sha(a.checkpoint)
    todo = []
    for lvl in (int(x) for x in a.levels.split(",")):
        src = ORIGINAL[a.run][lvl]
        if src is None:
            raise ValueError(f"no original +/-{lvl} final record for {a.run}")
        todo.append((f"{a.run}_pm{lvl}", src, False))
    if not a.no_dump:
        todo.insert(1, (f"{a.run}_dump", ORIGINAL[a.run][2], True))  # short; runs next to the first long job
    for name, _, _ in todo:
        if (a.out / name).exists():
            raise FileExistsError(a.out / name)
    running = {}
    while todo or running:
        for slot in SLOTS:
            if slot in running or not todo:
                continue
            name, src, dump = todo.pop(0)
            argv, L = job_argv(src, a.checkpoint, name, a.out, dump)
            st = argv[argv.index("--states") + 1]
            rec = dict(name=name, run=a.run, kind="scene_dump" if dump else "final", source_final_launch=str(src.relative_to(ROOT)),
                       source_argv=L["argv"], argv=["taskset", "-c", slot, *argv], started=now(), checkpoint=str(a.checkpoint.resolve()),
                       checkpoint_sha256=ck_sha, states=st, states_sha256=sha(st), source_states_sha256=L.get("states_sha256"),
                       git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"), git_diff=git("diff"))
            if rec["source_states_sha256"] and rec["source_states_sha256"] != rec["states_sha256"]:
                raise ValueError(f"state file changed since the original final: {st}")
            (a.out / name).mkdir(parents=True) if dump else None
            log = (a.out / f"{name}.stdout.log").open("w")
            p = subprocess.Popen(rec["argv"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            rec["pid"] = p.pid
            (a.out / f"{name}.launch.json").write_text(json.dumps(rec, indent=1))
            running[slot] = (name, p, log, dump)
            print(now(), "start", name, "slot", slot, "pid", p.pid, flush=True)
            time.sleep(20)
        for slot, (name, p, log, dump) in list(running.items()):
            if p.poll() is not None:
                log.close()
                s = a.out / name / ("eval" if dump else "") / "summary.json"
                ok = s.exists() and json.loads(s.read_text()).get("complete", False)
                (a.out / f"{name}.exit.json").write_text(json.dumps(dict(name=name, returncode=p.returncode, complete=ok, ended=now()), indent=1))
                print(now(), "end", name, "rc", p.returncode, "complete", ok, flush=True)
                del running[slot]
        time.sleep(15)
    print(now(), "all done", flush=True)


if __name__ == "__main__":
    main()
