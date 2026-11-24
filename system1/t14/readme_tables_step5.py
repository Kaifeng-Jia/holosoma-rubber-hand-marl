#!/usr/bin/env python3
"""Markdown tables for the step-4 README from levels.csv, comparisons.csv and contact_error_3d_summary.csv (no hand copying)."""
import csv
from pathlib import Path

O = Path(__file__).resolve().parents[2] / "results_system1/T14_retargeting/step5"
lv = list(csv.DictReader(open(O / "levels.csv")))
cp = list(csv.DictReader(open(O / "comparisons.csv")))
ce = {(r["round"], r["task"], r["level"]): r for r in csv.DictReader(open(O / "contact_error_3d/contact_error_3d_summary.csv"))}
ROUND = {"A": "round4A", "B": "round4B", "C": "round5C", "round3": "round3", "round3+step2": "round3+step2", "current": "comparisons", "omni_asis": "comparisons"}


def f(x, n=1):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "—"


def cell(r, task, level, ver):
    if not r.get("retention_pooled"):
        return "求解未完成"
    c = ce.get((ROUND[ver], task, level if ver not in ("current", "omni_asis") else ver), {})
    return (f"{f(r['retention_pooled'], 3)} / {f(r['max_penetration_cm'], 2)} / {f(c.get('dist_mean_cm'))} ({f(c.get('normal_signed_mean_cm'))}) / "
            f"{f(r.get('palm_normal_angle_mean_deg'), 0)}°, {f(r.get('finger_angle_mean_deg'), 0)}° / {f(r.get('palm_surface_angle_mean_deg'), 0)}° / "
            f"{f(r.get('wrist_near_limit_any_max'), 2)}, {f(r.get('wrist_speed_peak_max_dps'), 0)}")


for task in ("table", "bin", "chair", "push", "pull"):
    rows = [r for r in lv if r["task"] == task]
    levels = sorted({r["level"] for r in rows}, key=lambda s: float(s.split("= ")[1]))
    print(f"\n**{task}**\n\n| 档 | 第 3 轮（含第 2 步的档用第 2 步） | 第 4 轮 A | 第 4 轮 B | 第 5 轮 C |\n|---|---|---|---|---|")
    for L in levels:
        base = next((r for r in rows if r["level"] == L and r["version"] == "round3+step2"), None) or next((r for r in rows if r["level"] == L and r["version"] == "round3"), None)
        bver = base["version"] if base else None
        get = lambda v: next((r for r in rows if r["level"] == L and r["version"] == v), None)
        cells = [cell(base, task, L, bver) if base else "—"] + [cell(get(v), task, L, v) if get(v) else "—" for v in ("A", "B", "C")]
        note = [r["note"] for r in rows if r["level"] == L and r["version"] in ("A", "B", "C") and r.get("note")]
        print(f"| {L}{'（' + '、'.join(sorted(set(n for n in note if n))) + '）' if any(note) else ''} | " + " | ".join(cells) + " |")
print("\n**对照（1.0 倍）**\n\n| 任务 | 现有参考 | OmniRetarget 原样 |\n|---|---|---|")
for task in ("table", "bin", "chair", "push", "pull"):
    c = {r["version"]: r for r in cp if r["task"] == task}
    print(f"| {task} | " + " | ".join(cell(c[v], task, v, v) if v in c else "—" for v in ("current", "omni_asis")) + " |")

print("\n**接触位置误差（三维），各轮重算；均值 cm，括号内为法向分量（+ = 在物体外、悬空，- = 在人手接触面以内）**\n")
rounds = ("round2", "round3", "round3+step2", "round4A", "round4B", "round5C")
print("| 任务 | 档 | " + " | ".join(("第 2 轮", "第 3 轮", "第 3 轮 + 第 2 步", "第 4 轮 A", "第 4 轮 B", "第 5 轮 C")) + " | 现有参考 | OmniRetarget 原样 |")
print("|---" * 10 + "|")
for task in ("table", "bin", "chair", "push", "pull"):
    lvls = sorted({k[2] for k in ce if k[1] == task and k[0] != "comparisons"}, key=lambda s: float(s.split("= ")[1]))
    for L in lvls:
        cells = [f"{f(ce[(r, task, L)]['dist_mean_cm'])} ({f(ce[(r, task, L)]['normal_signed_mean_cm'])})" if (r, task, L) in ce else "—" for r in rounds]
        cmp_ = [f"{f(ce[('comparisons', task, v)]['dist_mean_cm'])} ({f(ce[('comparisons', task, v)]['normal_signed_mean_cm'])})" if L in ("k = 1.0", "m = 1.00") and ("comparisons", task, v) in ce else "" for v in ("current", "omni_asis")]
        print(f"| {task} | {L} | " + " | ".join(cells + cmp_) + " |")
