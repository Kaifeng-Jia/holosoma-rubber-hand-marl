"""Smoke test for system1/t02/package_scheme_b.py (unit list) and system1/render/u5/render_nominal_video.py (helper patching).

    hssim_s1/bin/python -B -m pytest -q system1/tests/test_package_scheme_b.py
"""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_units():
    from system1.t02.package_scheme_b import DELTA_CM, FEEDBACK, OBJECT_FEEDBACK, units
    us = units()
    assert len(us) == 81
    c = Counter(u["group"] for u in us)
    assert c == {"main": 40, "B-open": 5, "T04": 36}
    assert len({u["name"] for u in us}) == 81
    assert all(u["level"] == 2 for u in us if u["group"] == "B-open")
    assert FEEDBACK["B-open"] is False and all(FEEDBACK[v] for v in ("B0", "B1-0", "B1-2", "B1-4"))
    assert DELTA_CM["B0"] is None and DELTA_CM["B1-4"] == 4.0 and OBJECT_FEEDBACK["B1-0"] and not OBJECT_FEEDBACK["B0"]


def test_render_helper_patch_targets_present():
    """Every string the renderer replaces in the snapshot of figround_common.py occurs exactly once."""
    spec = importlib.util.spec_from_file_location("rnv", ROOT / "system1/render/u5/render_nominal_video.py")
    src = (ROOT / "system1/render/u5/sources_snapshot/figround_common.py").read_text()
    code = (ROOT / "system1/render/u5/render_nominal_video.py").read_text()
    for needle in ("ROOT=Path('/home/kevin/holosoma-core4d-base')", "cfg=read(R0/'R0/material_lighting.json')",
                   "tree=ast.parse((R0/'sources/previous_r0_helpers.py').read_text())"):
        assert src.count(needle) == 1 and needle in code
    assert spec is not None
