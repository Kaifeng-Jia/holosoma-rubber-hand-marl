#!/usr/bin/env python3
"""T12-v2 return pack (results_system1/T12_videos_v2/), after contact_stats.py and the renders:

1. contact_episodes.csv: adds termination_causes next to the evaluator's term flags (the evaluator records every failure as
   joint_bad_tracking; the cause is derived as T10 / T10b: MuJoCo from the saved last_termination record with
   system1/sim2sim/failure_t10s.causes, Isaac from the last row with system1/t10b/phases.termination_errors / termination_reason).
2. contact_tables.md: the summary as Markdown tables (task x simulator x level; completed / failed / all), numbers from
   contact_summary.csv only.
3. keyframes/: the key-frame sheets of the videos (copied from exports_system1/T12_videos_v2, sha256 checked); manifest.json: the
   per-video records of the renders.

    hssim_s1/bin/python -B system1/t12v2/package_t12v2.py --videos exports_system1/T12_videos_v2 --out results_system1/T12_videos_v2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sim2sim.failure_t10s import causes as mj_causes  # noqa: E402
from system1.t10b.phases import termination_errors, termination_reason  # noqa: E402

TASKS = ("push", "pull", "bucket", "chair", "table")
TASK_ZH = dict(push="Table Pushing", pull="Table Pulling", bucket="Bin Transport", chair="Chair Righting", table="Table Transport")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def add_causes(out: Path):
    f = out / "contact_episodes.csv"
    rows = list(csv.DictReader(f.open()))
    if "termination_causes" in rows[0]:
        return sum(r["completed"] != "True" for r in rows)
    prov = json.loads((out / "contact_provenance.json").read_text())
    summaries = {}
    for r in prov["runs"]:
        s = json.loads((ROOT / r["eval_dir"] / "summary.json").read_text())
        assert hashlib.sha256((ROOT / r["eval_dir"] / "summary.json").read_bytes()).hexdigest() == r["summary_sha256"]
        summaries[(r["task"], r["simulator"], int(r["level_cm"]), int(r["seed"]))] = {e["state_id"]: e for e in s["episodes"]}
    n = 0
    for r in rows:
        if r["completed"] == "True":
            r["termination_causes"] = ""
            continue
        e = summaries[(r["task"], r["simulator"], int(r["level_cm"]), int(r["seed"]))][r["state_id"]]
        if r["simulator"] == "mujoco":
            c = mj_causes(e["last_termination"])
        else:
            with np.load(r["trajectory"]) as z:
                pe, oe, _ = termination_errors({k: z[k] for k in z.files})
            c = [termination_reason(pe, oe)]
        r["termination_causes"] = "+".join(c) or "none_exceeded"
        n += 1
    keys = list(rows[0])
    keys.insert(keys.index("termination") + 1, keys.pop(keys.index("termination_causes")))
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return n


def pct(x):
    return "" if x in ("", None) else f"{100 * float(x):.0f}%"


def tables(out: Path, title="# T12-v2 接触统计汇总表（URDF 碰撞几何，任务书口径）"):
    S = list(csv.DictReader((out / "contact_summary.csv").open()))
    R = {r["task"]: r for r in csv.DictReader((out / "reference_contact.csv").open())}
    lines = [title, "",
             "数字全部来自 `contact_summary.csv`（每组 3 个种子 × 50 个扰动状态，均值）和 `reference_contact.csv`。",
             "比例的分母是统计时段内的行数；统计时段从参考第一次手接触物体开始。", ""]
    head = ("| 仿真器 | 档位 | 回合 | n | 物体最大升高 cm（参考） | 机器人 0 手 | 机器人 1 手 | 两台同时 | 机器人 0 非手部 | 机器人 1 非手部 "
            "| 0 非手部（除手腕） | 1 非手部（除手腕） |")
    sep = "|" + "---|" * 12
    grp_zh = dict(completed="完成", failed="失败", all="合计")
    for t in TASKS:
        ref = R[t]
        lines += [f"## {TASK_ZH[t]}", "",
                  f"参考：{ref['frames']} 帧，第一次手接触在第 {ref['first_contact_frame']} 帧（{ref['first_contact_s']} s），最大升高 {float(ref['reference_max_rise_cm']):.1f} cm；"
                  f"参考自身：机器人 0 手 {pct(ref['frac_r0_hand'])}，机器人 1 手 {pct(ref['frac_r1_hand'])}，两台同时 {pct(ref['frac_both_hands'])}，"
                  f"机器人 0 非手部 {pct(ref['frac_r0_nonhand'])}（除手腕 {pct(ref['frac_r0_nonhand_excl_wrist'])}），"
                  f"机器人 1 非手部 {pct(ref['frac_r1_nonhand'])}（除手腕 {pct(ref['frac_r1_nonhand_excl_wrist'])}）。", "", head, sep]
        for r in S:
            if r["task"] != t:
                continue
            rise = "" if r["object_max_rise_cm_mean"] == "" else f"{float(r['object_max_rise_cm_mean']):.1f}（{float(r['reference_max_rise_executed_cm_mean']):.1f}）"
            lines.append(f"| {'Isaac' if r['simulator'] == 'isaac' else 'MuJoCo'} | ±{r['level_cm']} | {grp_zh[r['episodes']]} | {r['n']} | {rise} | "
                         f"{pct(r['frac_r0_hand_mean'])} | {pct(r['frac_r1_hand_mean'])} | {pct(r['frac_both_hands_mean'])} | {pct(r['frac_r0_nonhand_mean'])} | "
                         f"{pct(r['frac_r1_nonhand_mean'])} | {pct(r['frac_r0_nonhand_excl_wrist_mean'])} | {pct(r['frac_r1_nonhand_excl_wrist_mean'])} |")
        lines.append("")
    lines += ["物体最大升高括号里是参考在同样执行行内的最大升高（失败回合只算到终止）。n = 0 的组没有回合，留空。", ""]
    (out / "contact_tables.md").write_text("\n".join(lines))


def videos(vdir: Path, out: Path):
    recs = []
    kd = out / "keyframes"
    kd.mkdir(parents=True, exist_ok=True)
    for j in sorted(vdir.glob("d_*_seed721_pm2_perturbed_000_*.json")):
        r = json.loads(j.read_text())
        assert sha(ROOT / r["video"]) == r["video_sha256"] and sha(ROOT / r["keyframes"]) == r["keyframes_sha256"]
        dst = kd / Path(r["keyframes"]).name
        shutil.copyfile(ROOT / r["keyframes"], dst)
        assert sha(dst) == r["keyframes_sha256"] and dst.stat().st_size <= 1_000_000
        r["keyframes_in_pack"] = str(dst.resolve().relative_to(ROOT))
        recs.append(r)
    assert len(recs) == 10, len(recs)
    (out / "manifest.json").write_text(json.dumps(dict(task="T12-v2", note="mp4 local only (exports_system1/T12_videos_v2/) and on Google Drive "
                                                       "(drive_manifest.csv); key-frame sheets in keyframes/", videos=recs), ensure_ascii=False, indent=1) + "\n")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    n = add_causes(a.out)
    tables(a.out)
    px = a.out / "physx_geometry"
    if (px / "contact_episodes.csv").exists():
        n_px = add_causes(px)
        tables(px, "# T12-v2 接触统计汇总表（对照：两个仿真器实际用的 PhysX 碰撞形状）")
        assert n_px == n
    recs = videos(a.videos, a.out)
    (a.out / "package_provenance.json").write_text(json.dumps(dict(script=str(Path(__file__).relative_to(ROOT)), script_sha256=sha(__file__),
                                                                    termination_causes_added=n, videos=len(recs)), indent=1) + "\n")
    print("causes added", n, "videos", len(recs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
