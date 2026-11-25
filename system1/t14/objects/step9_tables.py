#!/usr/bin/env python3
"""T14 step 9b: tables of the front-back run (run_step9.py) from logs_system1/t14/step9b (driver.json, gates, run records,
evaluations). Writes <out>/levels.csv (one row per solved or attempted level), <out>/clips.csv (one row per straight clip:
1.0 and the smallest level without robot-robot intersection), <out>/diag.csv (pre-solve gates and pelvis mismatch at 1.0,
1.25, 1.5 for every clip with gates) and prints markdown tables.

    python3 system1/t14/objects/step9_tables.py --out results_system1/T14_retargeting/step9b
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
L = ROOT / "logs_system1/t14/step9b"
ZH = {"V0 gravity": "V0 重力", "V0 support drift r0": "支撑脚漂移 r0", "V0 support drift r1": "支撑脚漂移 r1",
      "V0 non-hand penetration r0": "非手部位穿透 r0", "V0 non-hand penetration r1": "非手部位穿透 r1", "V0 robot-robot clearance": "两机间隙",
      "section-4 contact retention >= 0.99": "接触保持率", "main motion": "主运动"}


def tagm(m):
    return f"m{m:.2f}".replace(".", "p")


def solve_status(d):
    out = []
    for r in (0, 1):
        if (d / f"r{r}" / "run_record.json").exists():
            continue
        lg = d / f"r{r}.log"
        if not lg.exists():
            out.append(f"r{r} 未开始")
            continue
        lines = [x for x in lg.read_text(errors="replace").splitlines() if x.strip()]
        errs = [x.strip() for x in lines if "Error" in x]
        if errs:
            last = [x for x in lines if "/1" in x and "cost=" in x]
            at = last[-1].split("|")[-1].split("[")[0].strip() if last else "?"
            stage = "第一阶段" if sum("Starting motion retargeting" in x for x in lines) <= 1 else "第二阶段"
            out.append(f"r{r} 求解失败（{stage} {errs[-1][:60]}，停在第 {at} 帧）")
        else:
            out.append(f"r{r} 未完成")
    return "；".join(out)


def level_row(clip, sm, m):
    d = L / clip / tagm(m)
    g = json.loads((L / clip / f"gates_{tagm(m)}.json").read_text())
    lb = g["diag"].get("leg_box") or {}
    row = dict(clip=clip, sequence=sm["sequence"], object=sm["object"], frames=sm["frames"], level=f"{m:.2f}", L_m=g["diag"]["L_m"],
               walk=None if not g["diag"].get("walk") else f"{g['diag']['walk']['w0']}:{g['diag']['walk']['w1']}",
               partner_legs_to_box_min_m=(lb.get("partner") or {}).get("min_m"), partner_legs_in_box_frames=(lb.get("partner") or {}).get("frames_inside"),
               gravity_deg=g["gravity_deg"], persons_pelvis_net_m=f"{g['main_motion']['person_r0_m']} / {g['main_motion']['person_r1_m']}",
               pelvis_mismatch_max_m=g["diag"]["pelvis_mismatch_max_m"], pelvis_mismatch_mean_m=g["diag"]["pelvis_mismatch_mean_m"],
               pelvis_mismatch_max_person_m=g["diag"]["pelvis_mismatch_max_person_scale_m"])
    ef = d / "eval.json"
    if not ef.exists():
        row.update(evaluated=False, applicable="", failed="", solve_status=solve_status(d) or "未评测")
        return row
    e = json.loads(ef.read_text())
    v0 = e["V0"]
    dr = e["support_drift_round6"]
    ce = e["contact_position_error_3d"]
    row.update(evaluated=True, applicable=e["applicable"], failed=" | ".join(e["failed"]), failed_zh="、".join(ZH.get(x, x) for x in e["failed"]),
               support_drift_r0_m=dr["r0"]["value_m"], support_drift_r1_m=dr["r1"]["value_m"],
               old_window_drift_r0_m=dr["r0"]["old_window_value_m"], old_window_drift_r1_m=dr["r1"]["old_window_value_m"],
               nonhand_pen_r0_m=v0["nonhand_penetration_r0"]["value_m"], nonhand_pen_r1_m=v0["nonhand_penetration_r1"]["value_m"],
               rr_proxy_clearance_m=v0["robot_robot_clearance"]["value_m"], rr_min_distance_cm=e["robot_robot_min_distance_cm"],
               rr_intersect_frames=e["robot_robot_intersect_frames"], retention_r0=e["contact_retention"]["r0"], retention_r1=e["contact_retention"]["r1"],
               max_pen_r0_cm=e["max_penetration_cm"]["r0"], max_pen_r1_cm=e["max_penetration_cm"]["r1"],
               contact_err_r0_cm=(ce["r0"]["distance"] or {}).get("mean_cm"), contact_err_r1_cm=(ce["r1"]["distance"] or {}).get("mean_cm"),
               robots_pelvis_net_m=f"{e['main_motion']['robot_r0_m']} / {e['main_motion']['robot_r1_m']}",
               robots_pelvis_net_solver_m=f"{e['main_motion'].get('robot_r0_solver_m')} / {e['main_motion'].get('robot_r1_solver_m')}",
               robot_limit_solver_m=e["main_motion"].get("robot_limit_solver_m"),
               toe_drift_max_r0_cm=max((e["foot_slide"]["r0"][f"foot_slide_{s}"]["drift_max_cm"] or 0) for s in ("left", "right")),
               toe_drift_max_r1_cm=max((e["foot_slide"]["r1"][f"foot_slide_{s}"]["drift_max_cm"] or 0) for s in ("left", "right")),
               torso_tilt_r0_deg=e["posture"]["r0"]["torso_tilt_deg"], torso_tilt_r1_deg=e["posture"]["r1"]["torso_tilt_deg"],
               solve_status="", pair_sha256=e.get("pair_sha256"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--logdir", type=Path, default=None, help="default logs_system1/t14/step9b")
    a = ap.parse_args()
    global L
    if a.logdir is not None:
        L = a.logdir if a.logdir.is_absolute() else ROOT / a.logdir
    drv = json.loads((L / "driver.json").read_text())
    summ = drv["summary"]
    lv_rows, clip_rows, diag_rows = [], [], []
    for clip, sm in summ.items():
        if not (L / clip / "gates_m1p00.json").exists():
            clip_rows.append(dict(clip=clip, sequence=sm["sequence"], object=sm["object"], frames=sm["frames"], main=sm["main"], note=sm.get("error") or sm.get("exception", "")[-200:] or "未开始"))
            continue
        for m in (1.0, 1.25, 1.5):
            g = json.loads((L / clip / f"gates_{tagm(m)}.json").read_text())
            diag_rows.append(dict(clip=clip, sequence=sm["sequence"], level=f"{m:.2f}", gravity_deg=g["gravity_deg"], gravity_passed=g["gravity_passed"],
                                  persons_main_motion_passed=g["main_motion"]["passed"], **g["diag"]))
        levels = sorted(float(k) for k in (sm.get("levels") or {}))
        tried = sorted(set(levels) | {float(p.name[1:].replace("p", ".")) for p in (L / clip).glob("m*p*") if p.is_dir()})
        rows = {m: level_row(clip, sm, m) for m in tried}
        lv_rows += list(rows.values())
        free = [m for m in tried if rows[m].get("evaluated") and rows[m]["rr_intersect_frames"] == 0]
        mf = sm.get("min_free") if sm.get("min_free") is not None else (min(free) if free else None)
        r1 = rows.get(1.0, {})
        rf = rows.get(mf, {}) if mf is not None else {}
        clip_rows.append(dict(clip=clip, sequence=sm["sequence"], object=sm["object"], frames=sm["frames"], main=sm["main"],
                              gravity_deg=rows[tried[0]]["gravity_deg"] if tried else None,
                              pelvis_mismatch_max_m_1p0=json.loads((L / clip / "gates_m1p00.json").read_text())["diag"]["pelvis_mismatch_max_m"],
                              applicable_1p0=r1.get("applicable", ""), failed_1p0=r1.get("failed_zh") or r1.get("solve_status", ""),
                              rr_intersect_1p0=r1.get("rr_intersect_frames", ""), min_free_level=mf if mf is None else f"{mf:.2f}",
                              applicable_min_free=rf.get("applicable", ""), failed_min_free=rf.get("failed_zh") or rf.get("solve_status", ""),
                              a=sm.get("a"), first_free_above_1p5=sm.get("first_free_above_1p5"), levels_tried=" ".join(f"{m:.2f}" for m in tried),
                              note=sm.get("error") or (sm.get("exception") or "")[-200:]))
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("levels.csv", lv_rows), ("clips.csv", clip_rows), ("diag.csv", diag_rows)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(a.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print("| 片段 | 档 | 判定 | 不过的项 | 重力 ° | 脚漂移 r0 / r1 m | 非手穿透 r0 / r1 m | 两机最小距离 cm | 相交帧 | 保持率 r0 / r1 | 机器人骨盆净位移 m |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in lv_rows:
        if not r.get("evaluated"):
            print(f"| {r['clip']} | {r['level']} | 未评测 | {r['solve_status']} | {r['gravity_deg']} | | | | | | |")
            continue
        print(f"| {r['clip']} | {r['level']} | {'适用' if r['applicable'] else '不适用'} | {r['failed_zh'] or '无'} | {r['gravity_deg']} | {r['support_drift_r0_m']} / {r['support_drift_r1_m']} | "
              f"{r['nonhand_pen_r0_m']} / {r['nonhand_pen_r1_m']} | {r['rr_min_distance_cm']} | {r['rr_intersect_frames']} | {r['retention_r0']} / {r['retention_r1']} | {r['robots_pelvis_net_m']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
