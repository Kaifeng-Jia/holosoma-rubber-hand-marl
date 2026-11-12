#!/usr/bin/env python3
"""Smoke test for the c2_scheduler switches of Kevin 2026-11-12 (cloud curve backlog): --per-gpu-idle and --pm8-last.

    hssim_s1/bin/python -B system1/tests/test_c2_scheduler_caps.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.cloud.c2_scheduler import c2_eval_jobs, gpu_cap  # noqa: E402


def main():
    tr = [dict(pid=1, what="train cin_table_m5kg_seed722_fresh12000")]
    ev = [dict(pid=2, what="eval it01000_pm2")]
    # switch off: old cap everywhere
    assert gpu_cap([], False, 3, None) == 3 and gpu_cap(ev, False, 3, None) == 3 and gpu_cap(tr, False, 3, None) == 3
    # switch on: only a GPU with no training (and no training launched in the last 5 min) gets the larger cap
    assert gpu_cap([], False, 3, 6) == 6 and gpu_cap(ev * 4, False, 3, 6) == 6
    assert gpu_cap(tr + ev, False, 3, 6) == 3 and gpu_cap(ev, True, 3, 6) == 3
    assert gpu_cap([], False, 3, 2) == 3  # never below --per-gpu

    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        run = t / "cin_table_m5kg_seed721_fresh12000"
        run.mkdir()
        old = time.time() - 600
        for it in (0, 11000, 12000):
            f = run / f"model_{it}.pt"
            f.write_bytes(b"x")
            os.utime(f, (old, old))
        young = run / "model_13000.pt"  # younger than 120 s: not used
        young.write_bytes(b"x")
        (t / "ops").mkdir()
        (t / "ops" / f"{run.name}.exit.json").write_text(json.dumps(dict(completed=True)))
        off = {j["name"]: j["prio"] for j in c2_eval_jobs(t)}
        on = {j["name"]: j["prio"] for j in c2_eval_jobs(t, pm8_last=True)}
        pm8 = f"c2_final/{run.name}/it12000_pm8"
        assert set(off) == set(on) == {f"c2_curve/{run.name}/it{i:05d}_pm2" for i in (0, 11000, 12000)} | {pm8}, sorted(off)
        assert off[pm8] == (1, 12000) and on[pm8] == (5, 12000)
        assert all(off[k] == on[k] == (1, int(k.split("/it")[1][:5])) for k in off if k != pm8)
        # d_curve priorities are (2|3|4, it): with the switch on, +/-8 sorts after them
        assert max((2, 0), (3, 12000), (4, 12000)) < on[pm8]
    print("ok")


if __name__ == "__main__":
    raise SystemExit(main())
