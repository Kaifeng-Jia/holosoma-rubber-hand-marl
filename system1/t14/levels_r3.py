#!/usr/bin/env python3
"""T14 round 3: compose the levels.json for render_sizes.py from eval_r3.py output (contact moments + one block of numbers
per level). Numbers: contact retention (pooled), max penetration (exact, either robot), solve success, smallest robot-robot
distance (exact), contact-position error (mean per robot, report only). A level whose solve did not finish gets a
placeholder row with the reason read from its log.

    sonic_s1/bin/python -B system1/t14/levels_r3.py --task bin --eval <eval json> --prefix r3 --solves logs_system1/t14/r3 \
        --refs-dir exports_system1/T14_references/r3 --out <levels json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
from size_levels import moments  # noqa: E402

KS = ("0.8", "0.9", "1.0", "1.1", "1.2")


def numbers(r, ok):
    rr = r["robot_robot_min_distance_cm"]
    rrs = "> 50 cm" if not np.isfinite(rr) else (f"{10 * rr:.2f} mm（不相交）" if 0 <= rr < 0.1 else (f"{rr:.1f} cm" if rr >= 0 else f"{rr:.2f} cm（相交）"))
    g = r["contact_position_error"]
    gs = " / ".join(f"{g[f'r{i}']['mean_cm']:.1f}" if g.get(f"r{i}") else "—" for i in (0, 1))
    return [f"接触保持率 {r['contact_retention']['pooled']:.3f}", f"最大穿透 {max(r['max_penetration_cm'].values()):.2f} cm",
            f"求解成功：{'是' if ok else '否'}", f"机器人最小间隙 {rrs}", f"接触位置误差 {gs} cm"]


def failure_reason(log):
    txt = Path(log).read_text(errors="replace") if Path(log).exists() else ""
    prog = re.findall(r"(\d+)/(\d+) \[", txt)
    err = "Clarabel 求解失败" if "SolverError" in txt else "未完成"
    return err + (f"（第 {prog[-1][0]}/{prog[-1][1]} 帧）" if prog else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("table", "bin", "chair", "push", "pull"))
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--prefix", default="r3", help="name prefix of the level refs in the eval json")
    ap.add_argument("--solves", type=Path, required=True, help="solve directory (run records, logs)")
    ap.add_argument("--solve-tag", default="", help="CORE4D solve sub-directory suffix, e.g. _a10")
    ap.add_argument("--refs-dir", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="CORE4D: only these k (step-2 levels)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ev = json.loads(a.eval.read_text())
    refs = ev["refs"]
    levels = []
    if a.task in ("push", "pull"):
        import dual_stretch as ds
        h, _ = ds.human_hand_sd(ds.SEQ[a.task])
        frames = moments((h >= -0.015).any(axis=1))
        rec = json.loads((a.solves / f"single_{a.task}{a.solve_tag}/run_record.json").read_text())
        ok = rec["finite"] and rec["object_locked_max_err"] == 0.0
        lv = ([ev["a"]] if ev["a"] is not None else []) + [1.0, 1.5, 2.0]
        for m in lv:
            r = refs[f"{a.prefix}_m{m:.2f}"]
            name = f"a = {m:.2f}" if m == ev["a"] else f"m = {m:.1f}"
            levels.append(dict(label=f"{name}（桌长 {1.4 * m:.2f} m）", size=float(m), pair=str(Path(r["pair"]).resolve().relative_to(ROOT)), urdf=str(Path(r["urdf"]).resolve().relative_to(ROOT)),
                               numbers=numbers(r, ok), values=r))
    else:
        import eval_core4d_hulls as eh
        h, _ = eh.human_hand_sd(a.task)
        frames = [2 * x for x in moments((h >= -0.015).any(axis=(1, 2)))]
        for k in (a.only or KS):
            kk = "k" + k.replace(".", "p")
            sd = a.solves / f"{a.task}_{kk}{a.solve_tag}"
            rp = [sd / f"person{p}/run_record.json" for p in (1, 2)]
            label = f"k = {k}" + ("（原尺寸桶）" if (a.task == "bin" and k == "1.0") else "")
            if not all(x.exists() for x in rp):
                why = [f"机器人 {p - 1}：{failure_reason(a.solves / f'{a.task}_{kk}{a.solve_tag}_p{p}.log')}" for p, x in zip((1, 2), rp) if not x.exists()]
                levels.append(dict(label=label, size=float(k), failed="；".join(why), numbers=["求解成功：否"] + why, values=dict(success=False)))
                continue
            recs = [json.loads(x.read_text()) for x in rp]
            ok = all(x["finite"] and x["object_locked_max_err"] == 0.0 for x in recs)
            r = refs[f"{a.prefix}_k{k}"]
            out = (a.refs_dir / f"{a.task}_{kk}{a.solve_tag}").resolve()
            levels.append(dict(label=label, size=float(k), pair=str(Path(r["path"]).resolve().relative_to(ROOT)),
                               urdf=str((out / f"{a.task}_{kk}.urdf").relative_to(ROOT)), numbers=numbers(r, ok), values=dict(r, success=ok)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(task=a.task, contact_frames_30hz=frames, levels=levels), indent=1, ensure_ascii=False) + "\n")
    print(a.task, frames, [(L["label"], L["numbers"]) for L in levels])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
