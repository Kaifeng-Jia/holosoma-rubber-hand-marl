"""Smoke tests for system1/t02/pull_arm_diagnosis.py and system1/t03/early_progress.py (synthetic data, no files).

    hssim_s1/bin/python -B system1/tests/test_pull_arm_and_progress.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system1.t02 import pull_arm_diagnosis as pa  # noqa: E402
from system1.t03 import early_progress as ep  # noqa: E402


def main():
    # separation: first step after t0 with gap > threshold held for SEP_HOLD steps (or to the end)
    g = np.array([0.1] * 5 + [0.02] * 10 + [0.08] * 3 + [0.02] * 5 + [0.09] * 12)
    assert pa.separation(g, 5) == 23, pa.separation(g, 5)
    assert pa.separation(np.array([0.02] * 20 + [0.2] * 4), 0) == 20          # held to the end
    assert pa.separation(np.full(30, 0.01), 0) is None
    assert pa.first(np.array([False, False, True, True]), 1) == 2 and pa.first(np.zeros(4, bool), 0) is None
    # window mean needs a full window
    rows = {i: dict(reward_mean=float(i)) for i in range(1, 301)}
    assert ep.window(rows, 200, "reward_mean") == np.mean(np.arange(101, 201))
    assert ep.window(rows, 50, "reward_mean") is None
    assert abs(sum(ep.W_TRACK.values()) - 7.0) < 1e-12 and len(ep.W_TRACK) == 8 and len(ep.W_REG) == 3
    print("test_pull_arm_and_progress OK")


if __name__ == "__main__":
    main()
