#!/usr/bin/env python3
"""Offline smoke test of combine_mode_test.py helpers (no simulator): predictions, the velocity fit and the closest mode.

    hssim_s1/bin/python -B system1/t10b/test_combine_mode_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system1.t10b.combine_mode_test import DT, G, closest_mode, conditions, fit_mu, predictions  # noqa: E402


def main():
    p = predictions(1.0, 0.5)
    assert p == dict(average=0.75, min=0.5, multiply=0.5, max=1.0), p
    for mu, v0 in ((0.75, 2.0), (0.125, 1.0), (1.0, 1.0)):
        k = np.arange(400)
        v = np.maximum(v0 - mu * G * k * DT, 0.0)
        f = fit_mu(v, v0)
        assert abs(f["mu"] - mu) < 1e-9, (mu, f)
        assert f["n"] >= 5, f
    assert closest_mode(0.62, 0.75, 0.5)[0] == "average"
    assert closest_mode(0.38, 0.75, 0.5)[0] == "multiply"
    assert closest_mode(0.5, 1.0, 0.5)[0] == "min+multiply"  # tie reported, not hidden
    assert all(x["hand_friction"] == 0.75 for x in conditions([1.0], 0.5, [1.0])[1:])
    c = conditions([1.0, 0.5], 0.5, [1.0, 2.0])
    assert len(c) == 4 + 8 and sum(x["kind"] == "tensor_api" for x in c) == 4
    print("ok combine_mode_test helpers")


if __name__ == "__main__":
    main()
