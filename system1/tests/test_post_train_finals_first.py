#!/usr/bin/env python3
"""Smoke test for post_train.fetch_final (--finals-first): the remote file names it asks for match the cloud's
zero-padded checkpoint names (model_08050.pt for Pull joint8000, model_12000.pt for Table/Bin fresh12000), and a
fetched run is turned into its final jobs. No network: ssh is replaced and the files are already present locally.

    hssim_s1/bin/python -B system1/tests/test_post_train_finals_first.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t03 import post_train as p  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as d:
        p.EXPORT = Path(d)
        asked = []
        for label, ck in (("cin_pull_m20kg_seed723_joint8000", "model_08050.pt"),
                          ("cin_table_m5kg_seed723_fresh12000", "model_12000.pt")):
            dst = Path(d) / "train" / label
            dst.mkdir(parents=True)
            (dst / ck).write_bytes(b"ck")
            (dst / "run_config.json").write_text("{}")

            def fake_ssh(cmd, timeout=None, dst=dst):
                names = cmd.split("sha256sum ", 1)[1].split()
                asked.append(names)
                return 0, "".join(f"{p.sha(dst / n)}  {n}\n" for n in names if (dst / n).exists())

            p.ssh = fake_ssh
            out = p.fetch_final(label)
            assert asked[-1] == [ck, "run_config.json"], asked[-1]
            assert (out / "fetched_final.json").exists()
    print("ok")


if __name__ == "__main__":
    raise SystemExit(main())
