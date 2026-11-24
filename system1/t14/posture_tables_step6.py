#!/usr/bin/env python3
"""T14 round 6: posture.csv (every solve: torso tilt, pelvis height, elbow angles vs the scaled person, elbow position error,
foot slide) and trend.csv (Table k = 0.8-1.2 per robot; Push / Pull m levels share one single-robot solve, so their posture is
identical at every m) from step6/posture/posture.json and step6/r_ratios.json; markdown tables on stdout."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "results_system1/T14_retargeting/step6"
P = json.loads((O / "posture/posture.json").read_text())["refs"]
R = json.loads((O / "r_ratios.json").read_text())["tasks"]
VER = {"r5C": "第 5 轮 C", "r6C": "第 6 轮 C", "r6Cp": "第 6 轮 C′"}


def split(name):
    ver, rest = name.split("_", 1)
    task, _, k = rest.partition("_k")
    return ver, task, (float(k) if k else None)


rows = []
for name, rob in P.items():
    ver, task, k = split(name)
    for rr, x in rob.items():
        d = dict(version=ver, task=task, level=k if k is not None else "single", robot=rr, contact_frames=x["contact_frames"])
        for key in ("torso_tilt_deg", "pelvis_height_cm", "elbow_angle_left_deg", "elbow_angle_right_deg"):
            for f in ("robot_mean", "person_mean", "max_abs_diff"):
                d[f"{key}_{f}"] = (x[key] or {}).get(f)
        for side in ("left", "right"):
            d[f"elbow_pos_err_{side}_mean_cm"] = (x[f"elbow_position_error_{side}_cm"] or {}).get("mean")
            d[f"elbow_pos_err_{side}_p95_cm"] = (x[f"elbow_position_error_{side}_cm"] or {}).get("p95")
            fs = x[f"foot_slide_{side}"]
            d[f"foot_{side}_support_frames"] = fs["support_frames"]
            d[f"foot_{side}_segments"] = fs["segments"]
            d[f"foot_{side}_drift_max_cm"] = fs["drift_max_cm"]
            d[f"foot_{side}_robot_toe_speed_p95_mps"] = fs["robot_toe_speed_p95_mps"]
            d[f"foot_{side}_share_above_0p01mps"] = fs["share_above_0p01mps"]
        d["solver_foot_xy_deviation_max_cm"] = x.get("solver_foot_xy_deviation_max_cm")
        rows.append(d)
keys = list(rows[0])
with open(O / "posture/posture.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

trend = []
for d in rows:
    t = d["task"]
    if t == "table" and d["level"] != "single":
        r_h = R["table"]["persons"][f"person{int(d['robot'][1]) + 1}"]["r"]["height"] * d["level"]
    elif t in ("push", "pull"):
        r_h = R[t]["persons"]["person"]["r_vs_intermimic"]["height"]
    else:
        continue
    trend.append(dict(version=d["version"], task=t, level=d["level"], robot=d["robot"], r_height=round(r_h, 3),
                      tilt_robot=d["torso_tilt_deg_robot_mean"], tilt_person=d["torso_tilt_deg_person_mean"],
                      pelvis_robot_cm=d["pelvis_height_cm_robot_mean"], pelvis_person_cm=d["pelvis_height_cm_person_mean"],
                      elbow_robot=round((d["elbow_angle_left_deg_robot_mean"] + d["elbow_angle_right_deg_robot_mean"]) / 2, 1),
                      elbow_person=round((d["elbow_angle_left_deg_person_mean"] + d["elbow_angle_right_deg_person_mean"]) / 2, 1)))
trend.sort(key=lambda x: (x["task"], x["version"], x["robot"], x["level"] if x["level"] != "single" else 0))
with open(O / "posture/trend.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(trend[0]))
    w.writeheader()
    w.writerows(trend)
# monotonicity of the Table trend (k increasing = r increasing: tilt should not increase, pelvis should not decrease)
mono = {}
for ver in VER:
    for rr in ("r0", "r1"):
        seq = [x for x in trend if x["task"] == "table" and x["version"] == ver and x["robot"] == rr]
        seq.sort(key=lambda x: x["level"])
        if len(seq) >= 2:
            tilts, pels = [x["tilt_robot"] for x in seq], [x["pelvis_robot_cm"] for x in seq]
            mono[f"{ver}_{rr}"] = dict(levels=[x["level"] for x in seq], tilt_nonincreasing=all(b <= a for a, b in zip(tilts, tilts[1:])),
                                       pelvis_nondecreasing=all(b >= a for a, b in zip(pels, pels[1:])))
(O / "posture/trend_monotonic.json").write_text(json.dumps(mono, indent=1) + "\n")

f = lambda v, n=1: "—" if v is None else f"{float(v):.{n}f}"
print("\n**姿势和人比（接触帧均值，机器人 / 人；最大差）**\n")
print("| 版本 | 任务 | 档 | 机器人 | 前倾角 ° | 骨盆高 cm | 肘角 左 ° | 肘角 右 ° | 手肘位置误差 左 / 右 cm（均值） | 支撑段内脚尖漂移最大 左 / 右 cm | 支撑帧脚尖速度 p95 左 / 右 m/s |")
print("|---" * 11 + "|")
for d in sorted(rows, key=lambda x: (x["task"], str(x["level"]), x["robot"], x["version"])):
    if d["task"] in ("table",) and d["level"] not in (0.8, 1.0, 1.2):
        continue
    print(f"| {VER[d['version']]} | {d['task']} | {d['level']} | {d['robot']} | {f(d['torso_tilt_deg_robot_mean'])} / {f(d['torso_tilt_deg_person_mean'])}；{f(d['torso_tilt_deg_max_abs_diff'])} | "
          f"{f(d['pelvis_height_cm_robot_mean'])} / {f(d['pelvis_height_cm_person_mean'])}；{f(d['pelvis_height_cm_max_abs_diff'])} | "
          f"{f(d['elbow_angle_left_deg_robot_mean'])} / {f(d['elbow_angle_left_deg_person_mean'])} | {f(d['elbow_angle_right_deg_robot_mean'])} / {f(d['elbow_angle_right_deg_person_mean'])} | "
          f"{f(d['elbow_pos_err_left_mean_cm'])} / {f(d['elbow_pos_err_right_mean_cm'])} | "
          f"{f(d['foot_left_drift_max_cm'], 2)} / {f(d['foot_right_drift_max_cm'], 2)} | {f(d['foot_left_robot_toe_speed_p95_mps'], 3)} / {f(d['foot_right_robot_toe_speed_p95_mps'], 3)} |")
print("\n**趋势（Table 各档；Push、Pull 各档共用一次单台求解，姿势相同）**\n")
print("| 版本 | 任务 | 机器人 | 档 | r（高） | 前倾角 机器人 / 人 ° | 骨盆高 机器人 / 人 cm | 肘角 机器人 / 人 ° |")
print("|---" * 8 + "|")
for x in trend:
    print(f"| {VER[x['version']]} | {x['task']} | {x['robot']} | {x['level']} | {x['r_height']} | {f(x['tilt_robot'])} / {f(x['tilt_person'])} | {f(x['pelvis_robot_cm'])} / {f(x['pelvis_person_cm'])} | {f(x['elbow_robot'])} / {f(x['elbow_person'])} |")
print("\n**Table 单调性（k 增大即 r 增大：前倾角不增、骨盆不降）**：" + "；".join(f"{VER[k.split('_')[0]]} {k.split('_')[1]}：前倾{'是' if v['tilt_nonincreasing'] else '否'}、骨盆{'是' if v['pelvis_nondecreasing'] else '否'}" for k, v in mono.items()))
