#!/usr/bin/env python3
"""Smoke test of system1/t10b/pilot_report.py helpers on synthetic records (no files, CPU):
window sums / weighted episode lengths, missing-window handling, reference holder labels."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t10b.pilot_report import BUCKET, holder, window_stats  # noqa: E402


def rec(it, resets, timeouts, length, flen):
    r = {"iteration": it, "reset_count": resets, "timeout_count": timeouts, "physical_failure_count": resets - timeouts,
         "Episode/reset_count": resets, "Episode/failure_count": resets - timeouts, "Episode/reset_length_mean_steps": length,
         "Episode/failure_length_mean_steps": flen, "reward_mean": 0.1 * it, "Reward/env_raw_mean": 0.01 * it}
    r.update({f"Bucket/{k}_mean": 1.0 for k in BUCKET})
    return r


def main():
    m = {1: rec(1, 10, 6, 100.0, 50.0), 2: rec(2, 30, 6, 200.0, 150.0)}
    s = window_stats(m, 2, 2)
    assert s["window"] == "1-2" and s["resets"] == 40 and s["reference_horizon"] == 12 and s["joint_bad_tracking"] == 28, s
    assert abs(s["share_reference_horizon"] - 0.3) < 1e-12 and abs(s["share_joint_bad_tracking"] - 0.7) < 1e-12, s
    assert abs(s["episode_length_mean_steps"] - (10 * 100 + 30 * 200) / 40) < 1e-9, s
    assert abs(s["failed_episode_length_mean_steps"] - (4 * 50 + 24 * 150) / 28) < 1e-9, s
    assert abs(s["reward_mean"] - 0.15) < 1e-12 and s["bucket_raw_reward_mean"] == 1.0, s
    assert window_stats(m, 3, 2) is None  # iteration 3 not logged
    assert holder([0.0, 1.0]) == "robot1" and holder([1.0, 1.0]) == "robot0+robot1" and holder([0.2, 0.0]) == "none"
    print("ok pilot_report helpers")


if __name__ == "__main__":
    main()
