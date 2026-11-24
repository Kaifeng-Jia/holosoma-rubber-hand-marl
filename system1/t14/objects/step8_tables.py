#!/usr/bin/env python3
"""T14 step 8: candidates.csv (every candidate of every sequence), levels.csv (applicable sequences), push_pull_m1p25.csv and the
markdown tables of README_CN.md (stdout)."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
L = ROOT / "logs_system1/t14/step8"
O = ROOT / "results_system1/T14_retargeting/step8"
TAGS = ("smallboxb", "largeboxb", "suitcaseb", "smallboxc", "largeboxc", "suitcasec", "push", "pull")
NAME = dict(smallboxb="smallbox b · sub10_smallbox_007", largeboxb="largebox b · sub3_largebox_024", suitcaseb="suitcase b · sub1_suitcase_065",
            smallboxc="smallbox c · sub7_smallbox_027", largeboxc="largebox c · sub7_largebox_027", suitcasec="suitcase c · sub14_suitcase_016",
            push="Push · sub6_largetable_033", pull="Pull · sub3_largetable_010")
SHORT = {"V0 gravity": "重力", "V0 support drift r0": "支撑脚漂移 r0", "V0 support drift r1": "支撑脚漂移 r1", "V0 non-hand penetration r0": "非手部位穿透 r0",
         "V0 non-hand penetration r1": "非手部位穿透 r1", "V0 robot-robot clearance": "两机间隙", "section-4 contact retention >= 0.99": "接触保持率", "main motion": "主运动",
         "solve failed": "求解失败"}
f = lambda v, n=1: "—" if v is None else f"{float(v):.{n}f}"


def main():
    summ = {}
    for f in sorted(L.glob("driver*.json"), key=lambda x: x.stat().st_mtime):  # re-runs (driver_rerun*.json) override the main run
        summ.update(json.loads(f.read_text()).get("summary", {}))
    rows, md_seq, md_cand = [], [], []
    for tag in TAGS:
        cs = json.loads((L / f"cands/{tag}/candidates.json").read_text())
        n = len(cs["candidates"])
        g_fail = sum(not c["gravity_passed"] for c in cs["candidates"])
        m_fail = sum(c["gravity_passed"] and not c["main_motion"]["passed"] for c in cs["candidates"])
        solved = [c for c in cs["candidates"] if c["passed"]]
        fails = {}
        app = []
        for c in cs["candidates"]:
            e = None
            ef = L / f"cands/{tag}/c{c['index']:02d}/m1p00/eval.json"
            if c["passed"] and ef.exists():
                e = json.loads(ef.read_text())
                for x in e["failed"]:
                    fails[x] = fails.get(x, 0) + 1
                if e["applicable"]:
                    app.append(c["index"])
            mm = c["main_motion"]
            row = dict(sequence=tag, candidate=f"c{c['index']:02d}", id=c["id"], axis=c["axis"], parities=c["parities"], fit_rmse_m=c["fit_rmse_m"],
                       S_translation_horizontal_m=c["S_translation_horizontal_m"], improper=c["improper"], gravity_deg=c["gravity_deg"], gravity_passed=c["gravity_passed"],
                       main_motion_rule=mm["cls"], main_travel_m=mm["travel_m"], main_lift_max_m=mm["lift_max_m"], main_longest_run_above_0p10=mm["longest_run_above_0p10"],
                       main_motion_passed=mm["passed"], solved=e is not None)
            if c["passed"] and e is None:
                cdir = L / f"cands/{tag}/c{c['index']:02d}/m1p00"
                miss = [f"r{r}" for r in (0, 1) if not (cdir / f"r{r}" / "run_record.json").exists()]
                row["solve_status"] = "solve failed: " + ",".join(miss) if miss else "not evaluated"
                fails["solve failed"] = fails.get("solve failed", 0) + 1
            if e is not None:
                v = e["V0"]
                row.update(support_drift_r0_m=v["support_drift_r0"]["value_m"], support_drift_r1_m=v["support_drift_r1"]["value_m"],
                           nonhand_pen_added_r0_m=v["nonhand_penetration_r0"]["value_m"], nonhand_pen_added_r1_m=v["nonhand_penetration_r1"]["value_m"],
                           robot_robot_proxy_clearance_m=v["robot_robot_clearance"]["value_m"], retention_r0=e["contact_retention"]["r0"], retention_r1=e["contact_retention"]["r1"],
                           max_penetration_cm_r0=e["max_penetration_cm"]["r0"], max_penetration_cm_r1=e["max_penetration_cm"]["r1"],
                           robot_robot_min_distance_cm=e["robot_robot_min_distance_cm"], robot_robot_intersect_frames=e["robot_robot_intersect_frames"],
                           applicable=e["applicable"], failed=" | ".join(e["failed"]))
            rows.append(row)
        s = summ.get(tag, {})
        md_seq.append(f"| {NAME[tag]} | {n} | {g_fail} | {m_fail} | {len(solved)} | " + ("、".join(f"{SHORT[k]} {v}" for k, v in sorted(fails.items(), key=lambda x: -x[1])) or "—")
                      + f" | {len(app)} | {'适用（c%02d）' % s['selected'] if s.get('selected') is not None else '不适用'} |")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(O / "candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\n**每条序列：候选和判定**\n")
    print("| 序列 | 候选对 | 重力不过 | 主运动不过（重力已过） | 求解的候选 | 求解后各项不通过的候选数 | 全部通过 | 结论 |")
    print("|---" * 8 + "|")
    print("\n".join(md_seq))
    # levels of applicable sequences
    lrows = []
    print("\n**适用序列的各档**\n")
    print("| 序列 | 档 | 接触保持率 r0 / r1 | 最大穿透 cm r0 / r1 | 两机最小距离 cm（相交帧） | 三维接触误差 cm r0 / r1（法向） | 脚尖段内漂移最大 cm r0 / r1 | 前倾角 ° r0 / 人 | 骨盆高 cm r0 / 人 | 肘角 左 / 右 ° r0（人） |")
    print("|---" * 10 + "|")
    for tag in TAGS:
        s = summ.get(tag, {})
        if s.get("selected") is None:
            continue
        cdir = L / f"cands/{tag}/c{s['selected']:02d}"
        levels = sorted({1.0, 1.25, 1.5} | ({s["a"]} if s.get("a") else set()))
        for m in levels:
            ef = cdir / f"m{m:.2f}".replace(".", "p") / "eval.json"
            if not ef.exists():
                print(f"| {NAME[tag]} | {m:.2f} | 未完成 | | | | | | | |")
                continue
            e = json.loads(ef.read_text())
            fs = e["foot_slide"]
            d = [max((fs[f"r{r}"][f"foot_slide_{sd}"]["drift_max_cm"] or 0) for sd in ("left", "right")) for r in (0, 1)]
            ce = e["contact_position_error_3d"]
            p = e["posture"]["r0"]
            g = lambda k: (p[k] or {})
            lab = f"{m:.2f}" + ("（a）" if m == s.get("a") else "")
            print(f"| {NAME[tag]} | {lab} | {e['contact_retention']['r0']} / {e['contact_retention']['r1']} | {e['max_penetration_cm']['r0']:.2f} / {e['max_penetration_cm']['r1']:.2f} | "
                  f"{e['robot_robot_min_distance_cm']:.1f}（{e['robot_robot_intersect_frames']}） | {f(ce['r0']['distance']['mean_cm'] if ce['r0']['distance'] else None)} / "
                  f"{f(ce['r1']['distance']['mean_cm'] if ce['r1']['distance'] else None)}（{f(ce['r0']['normal_signed_mean_cm'])}） | {d[0]:.2f} / {d[1]:.2f} | "
                  f"{f(g('torso_tilt_deg').get('robot_mean'))} / {f(g('torso_tilt_deg').get('person_mean'))} | {f(g('pelvis_height_cm').get('robot_mean'))} / {f(g('pelvis_height_cm').get('person_mean'))} | "
                  f"{f(g('elbow_angle_left_deg').get('robot_mean'))} / {f(g('elbow_angle_right_deg').get('robot_mean'))}（{f(g('elbow_angle_left_deg').get('person_mean'))} / {f(g('elbow_angle_right_deg').get('person_mean'))}） |")
            lrows.append(dict(sequence=tag, candidate=f"c{s['selected']:02d}", level=m, a=s.get("a"), applicable_at_level=e["applicable"], failed_at_level=" | ".join(e["failed"]),
                              retention_r0=e["contact_retention"]["r0"], retention_r1=e["contact_retention"]["r1"], max_penetration_cm_r0=e["max_penetration_cm"]["r0"],
                              max_penetration_cm_r1=e["max_penetration_cm"]["r1"], robot_robot_min_distance_cm=e["robot_robot_min_distance_cm"],
                              robot_robot_intersect_frames=e["robot_robot_intersect_frames"], contact_error_mean_cm_r0=ce["r0"]["distance"]["mean_cm"] if ce["r0"]["distance"] else None,
                              contact_error_mean_cm_r1=ce["r1"]["distance"]["mean_cm"] if ce["r1"]["distance"] else None, foot_drift_max_cm_r0=d[0], foot_drift_max_cm_r1=d[1],
                              tilt_r0=g("torso_tilt_deg").get("robot_mean"), tilt_person=g("torso_tilt_deg").get("person_mean"), pair=e.get("pair"), pair_sha256=e.get("pair_sha256")))
    if lrows:
        with open(O / "levels.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lrows[0]))
            w.writeheader()
            w.writerows(lrows)
    # Push / Pull 1.25
    prow = []
    print("\n**Push、Pull 第 6 轮 C 补 1.25 档（单台求解的角色等距，同第 6 步口径）**\n")
    print("| 任务 | 档 | 接触保持率 r0 / r1 | 最大穿透 cm r0 / r1 | 两机最小距离 cm（相交帧） | 三维接触误差 cm 左 / 右（法向） | 手心法向与人 ° | 手指方向与人 ° | 手心与表面法向 ° |")
    print("|---" * 9 + "|")
    for t in ("push", "pull"):
        d = json.loads((O / f"push_pull_m1p25/eval_r6C_{t}_m1p25.json").read_text())
        v = d["refs"]["r6C_m1.25"]
        c = json.loads((O / f"push_pull_m1p25/contact_error_3d_{t}_m1p25.json").read_text())["result"]
        h = v["hand"]
        av = lambda k: sum(h[x][k]["mean_deg"] for x in h) / len(h)
        rr = v["robot_robot_min_distance_cm"]
        rrs = "≥ 50（测距上限）" if rr >= 50 else f"{rr:.1f}"
        print(f"| {t} | 1.25 | {v['contact_retention']['r0']} / {v['contact_retention']['r1']} | {v['max_penetration_cm']['r0']:.2f} / {v['max_penetration_cm']['r1']:.2f} | {rrs}（{v['robot_robot_intersect_frames']}） | "
              f"{c['r0_left']['dist_mean_cm']} / {c['r0_right']['dist_mean_cm']}（{c['r0_left']['normal_signed_mean_cm']} / {c['r0_right']['normal_signed_mean_cm']}） | {av('palm_normal_angle'):.1f} | {av('finger_angle'):.1f} | {av('palm_surface_angle'):.1f} |")
        prow.append(dict(task=t, level=1.25, retention_r0=v["contact_retention"]["r0"], retention_r1=v["contact_retention"]["r1"], max_penetration_cm_r0=v["max_penetration_cm"]["r0"],
                         max_penetration_cm_r1=v["max_penetration_cm"]["r1"], robot_robot_min_distance_cm=rr, robot_robot_intersect_frames=v["robot_robot_intersect_frames"],
                         contact_error_left_cm=c["r0_left"]["dist_mean_cm"], contact_error_right_cm=c["r0_right"]["dist_mean_cm"], pair=v.get("pair"), pair_sha256=v.get("pair_sha256"),
                         urdf_sha256=v.get("urdf_sha256")))
    with open(O / "push_pull_m1p25.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)


if __name__ == "__main__":
    main()
