#!/usr/bin/env python3
"""T10b B smoke return pack (results_system1/T10b_friction_margin/retrain/, before the cloud pilot): per configuration (D Table,
C-in Bin, seed 721) the smoke run_config, the original final training run_config (gzip), their key-by-key diff, the smoke
launch / exit records, randomization_samples.json and the 50-iteration metrics (gzip); validation-state sha256; the pilot
commands (system1/t10b/pilot_cloud.py --dry-run). SHA256SUMS written separately. Read only on the inputs.

    python3 system1/t10b/package_retrain_smoke.py --out results_system1/T10b_friction_margin/retrain [--smoke logs_system1/t10b/smoke_v2] [--only d_table_seed721]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "logs_system1/t10b/smoke_v2"  # recipe v2 (docs/system1/T10b_dr_ranges_zh.md); v1 smoke: logs_system1/t10b/smoke
ORIG = dict(
    d_table_seed721=dict(run_config="/home/kevin/holosoma-core4d-base/logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/run_config.json",
                         checkpoint="/home/kevin/holosoma-core4d-base/logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/model_12000.pt"),
    cin_bucket_seed721=dict(run_config=str(ROOT / "exports_system1/train/cin_bucket_m1kg_seed721_fresh12000/run_config.json"),
                            checkpoint=str(ROOT / "exports_system1/train/cin_bucket_m1kg_seed721_fresh12000/model_12000.pt")),
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
    else:
        out[p] = d
    return out


def short(v, n=200):
    s = json.dumps(v)
    return s if len(s) <= n else s[:n] + f"... ({len(s)} chars)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", type=Path, default=SMOKE)
    ap.add_argument("--only", default=",".join(ORIG), help="configurations to package (default both)")
    a = ap.parse_args()
    out = a.out.resolve()
    smoke = a.smoke.resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "smoke").mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, o in ((n, ORIG[n]) for n in a.only.split(",")):
        d = smoke / name
        ex = json.loads((smoke / f"{name}.exit.json").read_text())
        A = flat(json.loads(Path(o["run_config"]).read_text()))
        B = flat(json.loads((d / "run_config.json").read_text()))
        lines = [f"# {name}: original final training run_config ({o['run_config']}, sha256 {sha(o['run_config'])})",
                 f"#   vs smoke run_config ({(d / 'run_config.json').relative_to(ROOT)}, sha256 {sha(d / 'run_config.json')})",
                 "# key | original | smoke   (every key whose value differs; nested keys joined with '.')"]
        for k in sorted(set(A) | set(B)):
            va, vb = A.get(k, "<missing>"), B.get(k, "<missing>")
            if va != vb:
                lines.append(f"{k} | {short(va)} | {short(vb)}")
        (out / "configs" / f"{name}_original_vs_smoke.diff.txt").write_text("\n".join(lines) + "\n")
        shutil.copyfile(d / "run_config.json", out / "configs" / f"{name}_smoke.run_config.json")
        (out / "configs" / f"{name}_original.run_config.json.gz").write_bytes(gzip.compress(Path(o["run_config"]).read_bytes(), mtime=0))
        for f in (smoke / f"{name}.launch.json", smoke / f"{name}.exit.json", d / "randomization_samples.json"):
            shutil.copyfile(f, out / "smoke" / f"{name}__{f.name.replace(name + '.', '')}")
        (out / "smoke" / f"{name}__metrics.jsonl.gz").write_bytes(gzip.compress((d / "metrics.jsonl").read_bytes(), mtime=0))
        m = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines() if x.strip()]
        nonfinite = sorted({k for r in m for k, v in r.items() if isinstance(v, float) and not math.isfinite(v)})
        summary[name] = dict(passed=ex["passed"], returncode=ex["returncode"], iterations_logged=len(m), nonfinite_metric_keys=nonfinite,
                             reward_mean_first=m[0].get("reward_mean"), reward_mean_last=m[-1].get("reward_mean"),
                             smoke_checkpoint_sha256=sha(d / "model_00050.pt"), original_checkpoint=o["checkpoint"], original_checkpoint_sha256=sha(o["checkpoint"]),
                             randomization_module_sha256=B.get("system1_randomization.module_sha256"))
    vs = {p.name: sha(p) for p in sorted((ROOT / "system1/t10b/validation_states").glob("*.json"))}
    (out / "smoke_summary.json").write_text(json.dumps(dict(runs=summary, validation_states_sha256=vs), indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
