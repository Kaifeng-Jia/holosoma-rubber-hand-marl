"""Smoke test for diag_table.py helpers (CPU, no data): percentiles and the tabletop-face summary.

    OMP_NUM_THREADS=1 hssim_s1/bin/python -B -m pytest -q -p no:cacheprovider system1/t12diag/test_diag_table.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_table as dt  # noqa: E402


def test_pct():
    assert dt.pct(np.arange(101))["p50"] == 50.0


def test_summarize_faces():
    n = 4
    face = np.zeros((n, 2, 2), int)
    face[:, 1, 0] = 2  # robot 1 left hand on the side face
    hgt = np.full((n, 2, 2), 0.02)
    over = np.ones((n, 2, 2), bool)
    sd = np.full((n, 2, 2), -0.5)
    sd[:2, 0, 0] = 0.0  # robot 0 left hand within 1.5 cm in rows 0, 1
    sd[:, 1, 0] = -0.01  # robot 1 left hand within 1.5 cm in all rows
    out = dt.summarize_faces(face, hgt, over, sd, np.arange(n))
    assert out["robot0"]["left"]["rows_within_1p5cm"] == 2 and out["robot0"]["left"]["face_share_top"] == 1.0
    assert out["robot0"]["right"]["rows_within_1p5cm"] == 0 and out["robot0"]["right"]["face_share_top"] is None
    assert out["robot1"]["left"]["face_share_side"] == 1.0
    assert out["robot1"]["left"]["median_height_above_top_cm_all_rows"] == 2.0
