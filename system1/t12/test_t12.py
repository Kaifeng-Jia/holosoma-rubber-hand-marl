#!/usr/bin/env python3
"""Offline smoke test of the T12 helpers (no rendering, no meshes): slip_onset run detection.

    hssim_s1/bin/python -B system1/t12/test_t12.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system1.t12.slip_onset import first, first_run  # noqa: E402


def main():
    m = np.zeros(400, bool)
    m[195:198] = True  # too short
    m[250:256] = True
    assert first_run(m) == 250
    m[100:110] = True  # before the start row 190: ignored
    assert first_run(m) == 250
    assert first_run(np.zeros(300, bool)) is None
    assert first(np.array([0, 0, 1, 1], bool)) == 2 and first(np.zeros(3, bool)) is None
    print("ok t12 helpers")


if __name__ == "__main__":
    main()
