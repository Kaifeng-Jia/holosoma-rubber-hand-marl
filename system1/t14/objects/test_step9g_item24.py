#!/usr/bin/env python3
"""Smoke tests, T14 step 9g items 2 and 4: step9g_render_both.verdict labels, render_box's optional verdict_lines field (absent =
the old header column), and clips_both.csv against the re-judge table it is built from.

    sonic_s1/bin/python -m pytest -q system1/t14/objects/test_step9g_item24.py
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
R = ROOT / "results_system1/T14_retargeting/step9g"


def test_verdict_labels():
    import step9g_render_both as rb
    assert rb.verdict(True, []) == "适用"
    assert rb.verdict(False, ["V0 non-hand penetration r1", "V0 robot-robot clearance"]) == "不适用（非手 r1、两机间隙）"
    assert rb.tagm(1.25) == "m1p25"


def test_render_box_verdict_lines_optional():
    src = (HERE / "render_box.py").read_text()
    assert 'x.get("verdict_lines") or []' in src  # a levels json without the field adds no line


def test_clips_both_matches_rejudge():
    summ = json.loads((R / "nonhand_exact/step9g_all22_nonhand_exact.json").read_text())["summary"]
    rows = {r["clip"]: r for r in csv.DictReader(open(R / "clips_both.csv"))}
    assert set(rows) == set(summ) and len(rows) == 22
    for clip, s in summ.items():
        for key, col in (("applicable_old", "applicable_old_measure"), ("applicable_new", "applicable_new_measure")):
            want = " ".join(f"{v:.2f}" for v in s[key]) or "无"
            assert rows[clip][col] == want, (clip, key)
        assert set(s["applicable_old"]) <= set(s["applicable_new"]), clip  # the new measure is never larger than the old one here
