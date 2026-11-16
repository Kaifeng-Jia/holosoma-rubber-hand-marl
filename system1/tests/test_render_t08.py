"""Smoke test for system1/render/u5/render_t08.py (T08 D7 / D8): playback plan rules and the camera fit (no rendering).

    sonic_s1/bin/python -B -m pytest -q system1/tests/test_render_t08.py
"""
import importlib.util
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1] / "render/u5/render_t08.py"
spec = importlib.util.spec_from_file_location("render_t08", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_short_episode_held_to_4s():
    p = m.playback_plan(93)  # 1.86 s at 50 Hz
    assert p["speed"] == 1 and p["rows"][0] == 0 and p["rows"][-1] == 92
    assert p["gif_frames"] == 80 and p["gif_seconds"] == 4.0 and p["hold_frames"] == 80 - len(p["rows"])
    assert all(b > a for a, b in zip(p["rows"], p["rows"][1:]))


def test_mid_episode_1x_no_hold():
    p = m.playback_plan(308)  # 6.16 s
    assert p["speed"] == 1 and p["hold_frames"] == 0 and p["rows"][-1] == 307 and 4 <= p["gif_seconds"] <= 8
    assert p["rows"][:5] == [0, 2, 5, 7, 10]


def test_long_episode_2x():
    p = m.playback_plan(600)  # 12 s
    assert p["speed"] == 2 and p["hold_frames"] == 0 and p["rows"][-1] == 599 and 4 <= p["gif_seconds"] <= 8
    assert p["rows"][:4] == [0, 5, 10, 15]


def test_camera_for_center_equals_u5_rule_at_bbox_centre():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3)) * [1.0, 0.5, 0.4] + [0.3, -0.2, 0.6]
    c = (pts.min(0) + pts.max(0)) / 2
    a = m.camera_for_center(pts, c, 1920, 1080, 135, -22)
    # the U5 helper's formula (previous_r0_helpers.camera_for), written out
    az, el = np.deg2rad([135, -22])
    back = np.array([-np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el), -np.sin(el)])
    right = np.cross(-back, [0, 0, 1]); right /= np.linalg.norm(right); up = np.cross(right, -back)
    q = pts - c; tan = np.tan(np.deg2rad(22.5))
    d = max(np.max(q @ back + np.abs(q @ right) / (tan * 1920 / 1080 * .86)), np.max(q @ back + np.abs(q @ up) / (tan * .86)))
    assert abs(a["distance"] - d) < 1e-12 and a["lookat"] == c.tolist()


def test_pick_rule():
    att = [dict(attempt=i, completed=c) for i, c in enumerate([False, True, False, False])]
    a, done = m.pick(att)
    assert done == 1 and a["attempt"] == 0
    att = [dict(attempt=i, completed=c) for i, c in enumerate([False, True, True, True])]
    a, done = m.pick(att)
    assert done == 3 and a["attempt"] == 1
