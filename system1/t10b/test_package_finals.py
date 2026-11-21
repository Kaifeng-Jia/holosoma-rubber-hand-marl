#!/usr/bin/env python3
"""Offline smoke test of package_finals helpers (no simulator): per-frame labels from segments and the scene diff.

    hssim_s1/bin/python -B system1/t10b/test_package_finals.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system1.t10b.package_finals import per_frame, scene_diff  # noqa: E402


def main():
    assert per_frame([["still", "静止", 0, 1], ["lift", "抬起", 2, 3]], 4) == ["still", "still", "lift", "lift"]
    assert per_frame([["none", 0, 0], ["robot1", 1, 2]], 3, idx_first=1) == ["none", "robot1", "robot1"]
    d = scene_diff(dict(a=1, b=[1, 2], c=dict(x=1), new=0), dict(a=1, b=[1, 3], c=dict(x=1)))
    assert [x["path"] for x in d] == ["/b[1]", "/new"], d
    assert d[1]["diff"] == "only in the pilot export"
    print("ok package_finals helpers")


if __name__ == "__main__":
    main()
