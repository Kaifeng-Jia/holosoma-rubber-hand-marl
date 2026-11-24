#!/usr/bin/env python3
"""T14 step 7: markdown tables for README_CN.md and levels.csv (from screen.json, layer1_choice.json, role / levels / posture)."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
O = ROOT / "results_system1/T14_retargeting/step7"
S = ROOT / "logs_system1/t14/step7"
OBJS = ("largebox", "smallbox", "plasticbox", "trashcan", "suitcase", "smalltable")
f = lambda v, n=1: "—" if v is None else f"{float(v):.{n}f}"


def main():
    sc = json.loads((O / "screen/screen.json").read_text())
    print("\n**各物体、各类的序列数（括号内为片段数）**\n")
    print("| 物体 | 序列 | a 抱起往前走 | b 推/拉/拖 | c 原地抬起放下 | a 两侧夹住 / 前面或底部托住 / 其他 | b 两手同侧 / 不同侧 | b 推 / 拉 / 侧向 |")
    print("|---" * 8 + "|")
    tot = dict(seq=0, a=0, b=0, c=0)
    for o in OBJS:
        s = sc["summary"][o]
        tot["seq"] += s["sequences"]
        for k in "abc":
            tot[k] += s[f"class_{k}_sequences"]
        print(f"| {o} | {s['sequences']} | {s['class_a_sequences']}（{s['class_a_episodes']}） | {s['class_b_sequences']}（{s['class_b_episodes']}） | "
              f"{s['class_c_sequences']}（{s['class_c_episodes']}） | {s['a_grip']['两侧夹住']} / {s['a_grip']['前面或底部托住']} / {s['a_grip']['其他']} | "
              f"{s['b_same_side']['same']} / {s['b_same_side']['different']} | {s['b_push_pull']['推']} / {s['b_push_pull']['拉']} / {s['b_push_pull']['侧向']} |")
    print(f"| 合计 | {tot['seq']} | {tot['a']} | {tot['b']} | {tot['c']} | | | |")
    print("\n**每类 3 条典型（每个物体最多 1 条）**\n")
    print("| 类 | 序列 | 片段帧 | 要点 |")
    print("|---|---|---|---|")
    for k in "abc":
        for p in sc["picks"][k]:
            if k == "a":
                x = f"抓握 {p['grip']}，{p['steps']} 步，骨盆净位移 {p['pelvis_net_m']:.2f} m，与朝向夹角 {p['travel_vs_heading_deg']}°，抬高峰值 {100 * p['peak_lift_m']:.0f} cm"
            elif k == "b":
                x = f"物体移动 {p['travel_m']:.2f} m，{p['push_pull']}，手 {p['left_face']} / {p['right_face']}（同侧 {'是' if p['same_side'] else '否'}），抬高最大 {100 * p['max_lift_m']:.1f} cm，物体倾斜最大 {p['max_tilt_deg']}°"
            else:
                x = f"抓握 {p['grip']}，抬高峰值 {100 * p['peak_lift_m']:.0f} cm，骨盆离起点最远 {p['pelvis_max_from_start_m']:.2f} m"
            print(f"| {k} | {p['sequence']} | {p['start']}–{p['stop'] - 1} | {x} |")
    ch = json.loads((O / "layer1_choice.json").read_text())
    print("\n**盒子近似（每个物体用其 c 类典型序列的尺度；按盒面覆盖率排序）**\n")
    print("| 物体 | 有向包围盒 m | 凸包 / 盒体积 | 盒面 2 cm 内有网格的比例 | 盒面离网格 p95 cm |")
    print("|---|---|---|---|---|")
    for o in ch["object_order"]:
        q = ch["box_quality"][o]
        print(f"| {o} | {' × '.join(f'{x:.3f}' for x in q['extents_m'])} | {q['hull_over_box_volume']:.2f} | {q['box_surface_within_2cm_share']:.2f} | {100 * q['box_surface_gap_p95_m']:.1f} |")
    rows = []
    print("\n**第 1 层：角色等距和各档**\n")
    print("| 类 | 物体 · 序列 | 角色检查：候选对 / 通过 | 选中（方向，偏移 m） | a | 档 | 盒长 m | 接触保持率 r0 / r1 | 最大穿透 cm r0 / r1 | 两机最小距离 cm（相交帧） | 三维接触误差 cm r0（法向） | 脚尖段内漂移最大 cm r0 / r1 |")
    print("|---" * 12 + "|")
    for cls in "bc":
        for e in ch["choice"][cls]:
            tag = f"{e['object']}{cls}"
            rp, lp = S / f"role/{tag}/role_check.json", S / f"levels/{tag}/levels.json"
            if not lp.exists():
                print(f"| {cls} | {e['object']} · {e['sequence']} | 未完成 | | | | | | | | | |")
                continue
            r, lv = json.loads(rp.read_text()), json.loads(lp.read_text())
            sel = r["selected_roles"]
            kind = " / ".join("平移" if abs(s_["Q"][0][0] - 1) < 1e-6 else "绕竖直轴转 180°" for s_ in sel)
            show = lv["report_levels"] + ([lv["first_free_level_above_1"]] if lv.get("first_free_level_above_1") else [])
            for i, m in enumerate(show):
                x = lv["levels"][f"{m:.2f}"]
                ce = x["contact_position_error_3d"]["r0"]
                fs = x["foot_slide"]
                d0 = max((v["drift_max_cm"] or 0) for k, v in fs.items() if k.startswith("r0"))
                d1 = max((v["drift_max_cm"] or 0) for k, v in fs.items() if k.startswith("r1"))
                lab = f"{m:.2f}" + ("（a）" if m == lv["a"] else "") + ("（1.0 以上第一个不相交档，补充）" if m == lv.get("first_free_level_above_1") else "")
                offs = ", ".join("%+.3f" % s_["c"][lv["axis"]] for s_ in sel)
                a_txt = lv["a"] if lv["a"] is not None else "无（1.0 已相交）"
                head = (f"| {cls} | {e['object']} · {e['sequence']} | {r['pairs_total']} / {r['passing_pairs_total']}（{r['status']}） | 盒子轴 {lv['axis']}，{kind}，"
                        f"{offs} | {a_txt} |") if i == 0 else "| | | | | |"
                print(f"{head} {lab} | {x['length_along_axis_m']:.3f} | {x['contact_retention']['r0']} / {x['contact_retention']['r1']} | "
                      f"{x['max_penetration_cm']['r0']:.2f} / {x['max_penetration_cm']['r1']:.2f} | {x['robot_robot_min_distance_cm']:.1f}（{x['robot_robot_intersect_frames']}） | "
                      f"{f(ce['distance']['mean_cm'] if ce['distance'] else None)}（{f(ce['normal_signed_mean_cm'])}） | {d0:.2f} / {d1:.2f} |")
                rows.append(dict(cls=cls, object=e["object"], sequence=e["sequence"], level=m, a=lv["a"], role_pairs=r["pairs_total"], role_passing=r["passing_pairs_total"],
                                 role_status=r["status"], axis=lv["axis"], box_length_m=x["length_along_axis_m"], box_size_m=x["box_size_m"],
                                 retention_r0=x["contact_retention"]["r0"], retention_r1=x["contact_retention"]["r1"], retention_pooled=x["contact_retention"]["pooled"],
                                 condition_frames=x["condition_frames"], max_penetration_cm_r0=x["max_penetration_cm"]["r0"], max_penetration_cm_r1=x["max_penetration_cm"]["r1"],
                                 hand_max_penetration_cm_r0=x["hand_max_penetration_cm"]["r0"], robot_robot_min_distance_cm=x["robot_robot_min_distance_cm"],
                                 robot_robot_intersect_frames=x["robot_robot_intersect_frames"], contact_error_mean_cm_r0=ce["distance"]["mean_cm"] if ce["distance"] else None,
                                 contact_error_normal_signed_mean_cm_r0=ce["normal_signed_mean_cm"], foot_drift_max_cm_r0=d0, foot_drift_max_cm_r1=d1,
                                 palm_normal_angle_mean_deg_r0=round(sum(x["hand"][f"r0_{s}"]["palm_normal_angle"]["mean_deg"] or 0 for s in ("left", "right")) / 2, 1),
                                 finger_angle_mean_deg_r0=round(sum(x["hand"][f"r0_{s}"]["finger_angle"]["mean_deg"] or 0 for s in ("left", "right")) / 2, 1),
                                 palm_surface_angle_mean_deg_r0=round(sum(x["hand"][f"r0_{s}"]["palm_surface_angle"]["mean_deg"] or 0 for s in ("left", "right")) / 2, 1),
                                 pair=x["pair"], pair_sha256=x["pair_sha256"], urdf_sha256=x["urdf_sha256"]))
    if rows:
        with open(O / "levels.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    pp = O / "posture/posture.json"
    if pp.exists():
        P = json.loads(pp.read_text())["refs"]
        print("\n**姿势和人比（单台求解，接触帧均值；机器人 / 按 s 缩放的人）**\n")
        print("| 求解 | 前倾角 ° | 骨盆高 cm | 肘角 左 ° | 肘角 右 ° | 手肘位置误差 左 / 右 cm | 支撑段内脚尖漂移最大 左 / 右 cm | 支撑帧速度 > 0.01 m/s 占比 左 / 右 |")
        print("|---" * 8 + "|")
        for k, v in P.items():
            x = v["r0"]
            g = lambda key: f"{f(x[key]['robot_mean'])} / {f(x[key]['person_mean'])}" if x[key] else "—"
            print(f"| {k} | {g('torso_tilt_deg')} | {g('pelvis_height_cm')} | {g('elbow_angle_left_deg')} | {g('elbow_angle_right_deg')} | "
                  f"{f((x['elbow_position_error_left_cm'] or {}).get('mean'))} / {f((x['elbow_position_error_right_cm'] or {}).get('mean'))} | "
                  f"{f(x['foot_slide_left']['drift_max_cm'], 2)} / {f(x['foot_slide_right']['drift_max_cm'], 2)} | "
                  f"{f(x['foot_slide_left']['share_above_0p01mps'], 2)} / {f(x['foot_slide_right']['share_above_0p01mps'], 2)} |")


if __name__ == "__main__":
    main()
