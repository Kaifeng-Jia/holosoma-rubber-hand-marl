#!/usr/bin/env python3
"""T14 step 9e: tables of the frozen-construction run on the 22 straight clips (run_step9.py --all-main, logs_system1/t14/step9e).
Read only. Writes <out>/levels.csv (one row per solved or attempted level, with the step-9e phase diagnosis where present),
<out>/clips.csv (one row per clip: reported levels a / 1.0 / 1.25 / 1.5 / first free above 1.5, applicable levels, and per
reported level the failed items or the failed solve with its frame and phase) and prints markdown.

Applicable clip = applicable at one of its reported levels (Kevin 2026-11-25 15:10: "任何一档通过即算适用"). Levels evaluated
only as search steps (between 1.0 and a, or between 1.5 and the first free level) are listed in levels.csv, not counted.

    python3 system1/t14/objects/step9e_tables.py --out results_system1/T14_retargeting/step9e
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import step9_tables as st9  # noqa: E402

ROOT = HERE.parents[2]
L = ROOT / "logs_system1/t14/step9e"
PH_ZH = {"lift": "抱起", "walk": "走路", "put_down": "放下"}
LINK_ZH = [("knee", "膝"), ("hip", "髋"), ("ankle", "踝"), ("torso", "躯干"), ("pelvis", "骨盆"), ("waist", "腰"), ("shoulder", "肩"),
           ("elbow", "肘"), ("wrist", "腕"), ("head", "头")]


def tagm(m):
    return f"m{m:.2f}".replace(".", "p")


def link_zh(n):
    side = "左" if n.startswith("left") else ("右" if n.startswith("right") else "")
    for k, z in LINK_ZH:
        if k in n:
            return side + z
    return n


def zh_counts(pairs, k=3):
    """[(link, n)] -> '躯干 12、左髋 5' with links of the same Chinese name merged (hip pitch / roll / yaw are all 髋)."""
    from collections import Counter
    c = Counter()
    for x, m in pairs:
        c[link_zh(x)] += m
    return "、".join(f"{z} {m}" for z, m in c.most_common(k))


def phase_of(t, w0, w1):
    return "lift" if t < w0 else ("walk" if t < w1 else "put_down")


def solve_fail(d, w0, w1):
    """failed solves of a level: [(robot, stage, frame, phase)]"""
    out = []
    for r in (0, 1):
        if (d / f"r{r}" / "run_record.json").exists() or not (d / f"r{r}.log").exists():
            continue
        txt = (d / f"r{r}.log").read_text(errors="replace")
        prog = re.findall(r"\|\s*(\d+)/(\d+)\s*\[", txt)
        stage = "第一阶段" if txt.count("Starting motion retargeting") <= 1 else "第二阶段"
        f = int(prog[-1][0]) if prog else None
        out.append(dict(robot=f"r{r}", stage=stage, frame=f, phase=None if f is None else phase_of(f, w0, w1),
                        error=([x.strip() for x in txt.splitlines() if "Error" in x] or [""])[-1][:80]))
    return out


def phase_text(diag, item):
    """'抱起 12 帧（右膝 10）、放下 3 帧' for one failed item of one level from diag9e.json."""
    if diag is None:
        return ""
    if item == "V0 robot-robot clearance":
        ph = diag["robot_robot"]["phases"]
        return "；".join(f"{PH_ZH[p]} {v['frames']} 帧" + (f"（{', '.join(f'{x} {n}' for x, n in v['link_pairs'][:2])}）" if v["link_pairs"] else "")
                        for p, v in ph.items() if v["frames"])
    m = re.match(r"V0 non-hand penetration (r\d)", item)
    if m:
        ph = diag["robots"][m.group(1)]["nonhand"]["phases"]
        return "；".join(f"{PH_ZH[p]} {v['frames']} 帧，最深 {100 * v['deepest_m']:.1f} cm" + (f"（{zh_counts(v['links'])}）" if v["links"] else "")
                        for p, v in ph.items() if v["frames"])
    if item.startswith("section-4 contact retention"):
        parts = []
        for r, v in diag["robots"].items():
            for side, h in v["retention"]["hands"].items():
                ph = [f"{PH_ZH[p]} {x['frames']}" for p, x in h["phases"].items() if x["frames"]]
                if ph:
                    parts.append(f"{r} {'左' if side == 'left' else '右'}手丢 {h['lost_frames']}/{h['condition_frames']} 帧（{'、'.join(ph)}）")
        return "；".join(parts)
    m = re.match(r"V0 support drift (r\d)", item)
    if m:
        w = diag["robots"][m.group(1)]["support_drift"]["windows_over_limit"]
        return "；".join(f"{'左' if side == 'left' else '右'}脚 {x['frames'][0]}–{x['frames'][1] - 1} 帧 {100 * x['value_m']:.1f} cm（{'、'.join(PH_ZH[p] for p in x['phase'])}）"
                        for side, ws in w.items() for x in ws)
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    st9.L = L
    drv = json.loads((L / "driver.json").read_text())
    lv_rows, clip_rows = [], []
    for clip, sm in drv["summary"].items():
        g1 = json.loads((L / clip / "gates_m1p00.json").read_text())
        w0, w1 = int(g1["diag"]["walk"]["w0"]), int(g1["diag"]["walk"]["w1"])
        T = int(sm["frames"].split(":")[1]) - int(sm["frames"].split(":")[0])
        tried = sorted({float(p.name[1:].replace("p", ".")) for p in (L / clip).glob("m*p*") if p.is_dir()})
        rep = sm.get("report_levels") or [1.0, 1.25, 1.5]
        per = {}
        for m in tried:
            row = st9.level_row(clip, sm, m)
            d = L / clip / tagm(m)
            fails = solve_fail(d, w0, w1)
            diag = json.loads((d / "diag9e.json").read_text()) if (d / "diag9e.json").exists() else None
            row.update(reported=m in rep, T=T, w0=w0, w1=w1,
                       solve_fail="；".join(f"{x['robot']} {x['stage']}停在第 {x['frame']} 帧（{PH_ZH.get(x['phase'], '?')}"
                                           + ("段第 1 帧" if x["frame"] in (w0, w1) else "") + f"）{x['error']}" for x in fails),
                       solve_fail_frames=" ".join(f"{x['robot']}:{x['frame']}" for x in fails))
            if row.get("evaluated") and not row["applicable"]:
                ev = json.loads((d / "eval.json").read_text())
                row["phase_diag"] = " | ".join(f"{st9.ZH.get(it, it)}：{phase_text(diag, it) or '—'}" for it in ev["failed"])
            lv_rows.append(row)
            per[m] = row
        appl = [m for m in rep if (per.get(m) or {}).get("applicable") is True]
        desc = []
        for m in rep:
            r = per.get(m)
            if r is None:
                desc.append(f"{m:.2f}：未求解")
            elif r.get("evaluated"):
                desc.append(f"{m:.2f}：" + ("适用" if r["applicable"] else "不适用（" + (r.get("phase_diag") or r["failed_zh"]) + "）"))
            else:
                desc.append(f"{m:.2f}：未评测（{r['solve_fail'] or r['solve_status']}）")
        clip_rows.append(dict(clip=clip, sequence=sm["sequence"], object=sm["object"], frames=sm["frames"], debug_clip=sm["main"], T=T, walk=f"{w0}:{w1}",
                              a=sm.get("a"), first_free_above_1p5=sm.get("first_free_above_1p5"), report_levels=" ".join(f"{m:.2f}" for m in rep),
                              levels_tried=" ".join(f"{m:.2f}" for m in tried), applicable=bool(appl), applicable_levels=" ".join(f"{m:.2f}" for m in appl),
                              per_level="  ‖  ".join(desc), note=sm.get("error") or (sm.get("exception") or "")[-200:]))
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("levels.csv", lv_rows), ("clips.csv", clip_rows)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(a.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    n = sum(r["applicable"] for r in clip_rows)
    print(f"applicable clips: {n} / {len(clip_rows)}")
    print("| 片段 | 物体 | 走路段 | 评测的档 | 适用的档 | 各档结果 |")
    print("|---|---|---|---|---|---|")
    for r in clip_rows:
        print(f"| {r['clip']} | {r['object']} | {r['walk']}（共 {r['T']} 帧） | {r['report_levels']} | {r['applicable_levels'] or '无'} | {r['per_level']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
