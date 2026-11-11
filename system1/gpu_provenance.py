"""GPU model of every evaluation row (Kevin 2026-11-10: episodes.csv records the GPU model; cross-scheme tables say when
the two sides ran on different models).

  * our evaluations: run_manifest.json gpu_name (evaluator, since commit 801f755), else the Isaac Sim GPU table in the
    run's stdout log;
  * V3 closeout rows (results/episodes.csv has no GPU column), by the source of each row:
      - Full +/-2 Pull / Table / Bin / Chair: PaperSupplement20260929/unified_evaluation_20261003 on the Ubuntu laptop
        (GPU named in each model's stdout log there);
      - Full +/-2 Push / Kick: ExperimentsR2_evaluation_20261026_retry01, AutoDL R2 machine
        (r2_20261024_ops/torch_cuda_check.log, 2026-10-25: 2 x NVIDIA GeForce RTX 4090);
      - every other V3 row (cloud_backup_20261104: Full +/-4/6/8, V3 groups): AutoDL V3 machine
        (v3_20261102_rev14_ops/deployment_receipt.json, 2026-11-02: 5 x NVIDIA GeForce RTX 4090).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

LAPTOP = "NVIDIA GeForce RTX 5070 Laptop GPU"
RTX4090 = "NVIDIA GeForce RTX 4090"
_PAT = re.compile(r"NVIDIA GeForce RTX (\d{4})( Laptop)?")
UNIFIED = Path("/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003")
PM2_BASELINES = Path("/home/kevin/holosoma-core4d-base/logs/V3_20261102/cloud_backup_20261104/raw/v3_pm2_baselines.json")
EVIDENCE = {
    "r2": "AutoDL R2 machine; r2_20261024_ops/torch_cuda_check.log (2026-10-25): 2 x NVIDIA GeForce RTX 4090",
    "v3": "AutoDL V3 machine; v3_20261102_rev14_ops/deployment_receipt.json (2026-11-02): 5 x NVIDIA GeForce RTX 4090",
}


def normalize(text):
    m = _PAT.search(text or "")
    if not m:
        return None
    if m.group(1) == "5070" and m.group(2):
        return LAPTOP
    if m.group(1) == "4090":
        return RTX4090
    return m.group(0)


@lru_cache(maxsize=None)
def from_log(path):
    p = Path(path)
    if not p.exists():
        return None
    with p.open(errors="replace") as f:
        for line in f:
            g = normalize(line)
            if g:
                return g
    return None


def ours(run_dir, stdout_log=None):
    """(gpu_model, source) for one of our evaluation directories."""
    man = Path(run_dir) / "run_manifest.json"
    if man.exists():
        g = json.loads(man.read_text()).get("gpu_name")
        if g:
            return normalize(g) or g, "run_manifest.gpu_name"
    if stdout_log is not None:
        g = from_log(stdout_log)
        if g:
            return g, "isaac_stdout_log"
    return None, "unknown"


@lru_cache(maxsize=None)
def _pm2_sources():
    return {(e["case"], int(e["seed"])): e["summary_source"] for e in json.loads(PM2_BASELINES.read_text())}


def v3_row(r):
    """(gpu_model, source) for one row of the V3 closeout results/episodes.csv."""
    p = r["exit_record_path"]
    if p.endswith("v3_pm2_baselines.json"):
        src = Path(_pm2_sources()[(r["task"], int(r["training_seed"]))])
        log = src.parents[2] / f"{src.parent.name}.stdout.log"  # unified_evaluation_20261003/<model>.stdout.log
        g = from_log(log)
        if g is None or UNIFIED not in log.parents:
            raise RuntimeError(f"no GPU for V3 pm2 baseline {r['task']} {r['training_seed']}: {log}")
        return g, f"stdout log {log}"
    if "ExperimentsR2_evaluation_20261026" in p:
        return RTX4090, EVIDENCE["r2"]
    if "cloud_backup_20261104" in p:
        return RTX4090, EVIDENCE["v3"]
    raise RuntimeError(f"unknown V3 row source: {p}")
