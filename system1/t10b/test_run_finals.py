#!/usr/bin/env python3
"""Offline smoke test of run_finals.job_argv (no simulator): the pilot final argv equals the original final argv except
--checkpoint, --model-id and --output-dir (plus --limit 1 --sim2sim-dump for the scene export).

    hssim_s1/bin/python -B system1/t10b/test_run_finals.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system1.t10b.run_finals import ORIGINAL, job_argv  # noqa: E402


def main():
    ck = Path("exports_system1/t10b_pilot/cin_bucket_seed721_rand/model_12000.pt")
    for lvl in (2, 8):
        src = ORIGINAL["cin_bucket_seed721_rand"][lvl]
        orig = json.loads(src.read_text())["argv"][3:]
        for dump in (False, True):
            argv, _ = job_argv(src, ck, f"t_pm{lvl}", Path("/tmp/x"), dump)
            base = argv[:len(orig)]
            diff = {orig[i - 1] for i in range(len(orig)) if base[i] != orig[i]}
            assert diff == {"--checkpoint", "--model-id", "--output-dir"}, diff
            assert argv[len(orig):] == (["--limit", "1", "--sim2sim-dump", "/tmp/x/t_pm2/dump".replace("pm2", f"pm{lvl}")] if dump else []), argv[len(orig):]
            assert base[base.index("--checkpoint") + 1] == str(ck.resolve())
    print("ok run_finals argv")


if __name__ == "__main__":
    main()
