#!/usr/bin/env python3
"""T13 section 9 (Kevin 2026-11-26, fixed training budget): prepare the continued runs d_push_to10k / d_table_to10k /
c_out_push_to10k (cloud_runs.py CONT) and smoke-test the resume on the laptop.

Per run (local, read only on the base run's copied-back directory logs_system1/t13v4/runs/<base>/):
  * continue_ref.json: the base checkpoint (file number = CONT from_iteration) and its sha256, and the pre-stop window = the 10
    metrics rows up to that iteration (cloud_runs.window_stats: reward_mean, value_loss, |surrogate_loss|, entropy, resets per
    environment, last actor / critic learning rate) - the reference of the section 9 resume check (cloud_runs.continue_check);
  * plan_diff.json: cloud_runs.py --dry-run plans (4096 environments) of the base run and the continued run; the training argv
    must differ only in --iterations, --output-dir and --resume, and the warm-up stage is absent (checked here);
  * --smoke: the continued run's training argv on the laptop 5070 (hssim_s1, --num-envs N, --iterations 12, local output dir,
    --resume = the local copy of the checkpoint), then continue_check on its first 10 rows -> smoke.json. Every launch waits
    until MemAvailable >= 17 GB. Pass = rc 0 and the check passed.
Output: logs_system1/t13v4/continue/<run>/.

    python3 system1/t13v4/continue_prep.py                    # refs + plan diffs
    python3 system1/t13v4/continue_prep.py --smoke            # + local resume smoke (one run after the other)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t13v4.cloud_runs import CONT, continue_check, now, window_stats  # noqa: E402

OUT = ROOT / "logs_system1/t13v4/continue"
HS = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
SMOKE_ENVS = dict(d_push_to10k=2048, d_table_to10k=2048, c_out_push_to10k=512)  # C-out: as round2_smoke_local.py (SONIC on the 8 GB card)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def avail_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) // 1048576
    return 0


def dry(run, **kw):
    argv = [sys.executable, str(ROOT / "system1/t13v4/cloud_runs.py"), "--run", run, "--num-envs", "4096", "--out", "/cloud", "--python", sys.executable, "--dry-run"]
    for k, v in kw.items():
        argv += [k, str(v)]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        raise SystemExit(p.stderr)
    return json.loads(p.stdout)


def prep(run):
    c = CONT[run]
    base = ROOT / "logs_system1/t13v4/runs" / c["base"]
    it0 = c["from_iteration"]
    ck = base / "train" / f"model_{it0:05d}.pt"
    rows = [json.loads(x) for x in (base / "train/metrics.jsonl").read_text().splitlines() if x.strip()]
    pre = [r for r in rows if r["iteration"] <= it0][-10:]
    assert [r["iteration"] for r in pre] == list(range(it0 - 9, it0 + 1)), [r["iteration"] for r in pre]
    cfg = json.loads((base / "train/run_config.json").read_text())
    d = OUT / run
    d.mkdir(parents=True, exist_ok=True)
    ref = dict(run=run, base_run=c["base"], from_iteration=it0, checkpoint_local=str(ck.relative_to(ROOT)), checkpoint_sha256=sha(ck),
               metrics=str((base / "train/metrics.jsonl").relative_to(ROOT)), metrics_sha256=sha(base / "train/metrics.jsonl"),
               base_num_envs=cfg["num_envs"], before=window_stats(pre, cfg["num_envs"]), made=now())
    (d / "continue_ref.json").write_text(json.dumps(ref, indent=1) + "\n")
    pb = dry(c["base"])
    pc = dry(run, **{"--continue-checkpoint": ck, "--continue-ref": d / "continue_ref.json"})  # local copy (cloud_runs checks its sha256)
    ab, ac = pb["stages"][-1]["argv"], pc["stages"][-1]["argv"]

    def strip(a):
        out, i = [], 0
        while i < len(a):
            if a[i] in ("--iterations", "--output-dir", "--resume"):
                i += 2
                continue
            out.append(a[i])
            i += 1
        return out
    diff = dict(base_stage_names=[s["name"] for s in pb["stages"]], continued_stage_names=[s["name"] for s in pc["stages"]],
                argv_equal_except_iterations_output_resume=strip(ab) == strip(ac),
                base=dict(iterations=ab[ab.index("--iterations") + 1], resume=ab[ab.index("--resume") + 1] if "--resume" in ab else None),
                continued=dict(iterations=ac[ac.index("--iterations") + 1], resume=ac[ac.index("--resume") + 1]),
                continued_plan=pc)
    assert diff["argv_equal_except_iterations_output_resume"] and diff["continued_stage_names"] == ["train"], diff
    assert int(diff["continued"]["iterations"]) == pc["warmup_iterations"] + pc["rules"]["max_it"] - it0
    (d / "plan_diff.json").write_text(json.dumps(diff, indent=1) + "\n")
    print(run, "ref", json.dumps(ref["before"]), "iterations to run", diff["continued"]["iterations"], flush=True)
    return ref, pc


def smoke(run, ref, plan, n_it):
    d = OUT / run / "smoke"
    assert not d.exists(), f"exists: {d}"
    d.mkdir(parents=True)
    argv = list(plan["stages"][-1]["argv"])
    argv[0] = HS
    argv[argv.index("--num-envs") + 1] = str(SMOKE_ENVS[run])
    argv[argv.index("--iterations") + 1] = str(n_it)
    argv[argv.index("--output-dir") + 1] = str(d / "train")
    argv[argv.index("--resume") + 1] = str(ROOT / ref["checkpoint_local"])
    while avail_gb() < 17:
        time.sleep(30)
    rec = dict(run=run, argv=argv, started=now(), mem_available_gb=avail_gb())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    with open(d / "train.stdout.log", "w") as f:
        rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    rec.update(rc=rc, ended=now())
    m = d / "train/metrics.jsonl"
    rows = [json.loads(x) for x in m.read_text().splitlines()[:10] if x.strip()] if m.exists() else []
    if len(rows) == 10:
        rec["check"] = continue_check(ref, rows, json.loads((d / "train/run_config.json").read_text()), ref["from_iteration"], sha(ROOT / ref["checkpoint_local"]))
    rec["passed"] = rc == 0 and bool(rec.get("check", {}).get("passed"))
    (d / "smoke.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(run, "smoke rc", rc, "passed", rec["passed"], json.dumps(rec.get("check", {}).get("ratios")), flush=True)
    return rec


def recheck(run, ref, note):
    """re-run continue_check (current code) on an existing smoke's outputs; the first record is kept as smoke_first_check.json."""
    d = OUT / run / "smoke"
    rec = json.loads((d / "smoke.json").read_text())
    first = d / "smoke_first_check.json"
    if not first.exists():
        first.write_text(json.dumps(rec, indent=1) + "\n")
    rows = [json.loads(x) for x in (d / "train/metrics.jsonl").read_text().splitlines()[:10] if x.strip()]
    rec["check"] = continue_check(ref, rows, json.loads((d / "train/run_config.json").read_text()), ref["from_iteration"], sha(ROOT / ref["checkpoint_local"]))
    rec["passed"] = rec["rc"] == 0 and bool(rec["check"]["passed"])
    rec.setdefault("rechecks", []).append(dict(time=now(), note=note, passed=rec["passed"], items=rec["check"]["items"]))
    (d / "smoke.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(run, "recheck passed", rec["passed"], json.dumps({k: v for k, v in rec["check"]["items"].items() if not v}), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--recheck", default=None, help="re-run the check on the existing smoke outputs with the current code; the text is the reason")
    ap.add_argument("--runs", nargs="+", default=list(CONT))
    ap.add_argument("--iterations", type=int, default=12)
    a = ap.parse_args()
    res = {}
    for run in a.runs:
        ref, plan = prep(run)
        res[run] = dict(prepared=True)
        if a.smoke:
            res[run]["smoke_passed"] = smoke(run, ref, plan, a.iterations)["passed"]
        elif a.recheck:
            res[run]["smoke_passed"] = recheck(run, ref, a.recheck)["passed"]
    print(json.dumps(res))
    return 0 if all(v.get("smoke_passed", True) for v in res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
