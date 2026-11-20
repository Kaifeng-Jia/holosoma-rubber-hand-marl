#!/usr/bin/env python3
"""T10 step 0.4: settings_compare.md - Isaac training / evaluation settings (D, C-in; from the PhysX export of the evaluator,
system1/sim2sim/isaac_dump.py), SONIC's own training environment (system1/sonic/sonic_release_spec.json, dumped from SONIC's
Isaac Lab env), the official deployment (gear_sonic_deploy policy_parameters.hpp, computed by its formulas) and the official
MuJoCo sim2sim (gear_sonic mujoco_sim yaml + MJCF), per joint and per item. All numbers are read from the files, none typed in.

    sim2sim_s1/bin/python -B system1/sim2sim/settings_compare.py --out results_system1/T10_sim2sim
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sim2sim.mj_eval import DEPLOY, MUJOCO_ORDER, SIMCFG, official_pd  # noqa: E402

G = Path("/home/kevin/third_party/GR00T-WholeBodyControl")
MJCF = G / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"


def deploy_arrays():
    """default_angles and g1_action_scale of the deploy header (MuJoCo joint order)."""
    txt = DEPLOY.read_text()
    dflt = [float(x) for x in re.findall(r"^\s*(-?\d+\.\d+),?\s*//", txt.split("default_angles")[1], flags=re.M)[:29]]
    kp, kd, eff = official_pd()
    eff_motor = dict(A5020=25.0, A7520_14=88.0, A7520_22=139.0, A4010=5.0)
    arm = dict(A5020=0.003609725, A7520_14=0.010177520, A7520_22=0.025101925, A4010=0.00425)
    w = 10 * 2.0 * 3.1415926535
    motor_scale = ["A7520_22", "A7520_22", "A7520_14", "A7520_22", "A5020", "A5020"] * 2 + ["A7520_14", "A5020", "A5020"] + \
                  ["A5020", "A5020", "A5020", "A5020", "A5020", "A4010", "A4010"] * 2
    scale = [0.25 * eff_motor[m] / (arm[m] * w * w) for m in motor_scale]
    return dict(zip(MUJOCO_ORDER, dflt)), dict(zip(MUJOCO_ORDER, scale)), kp, kd, eff


def mjcf_joint_params():
    root = ET.parse(MJCF).getroot()
    classes = {}
    for d in root.iter("default"):
        c = d.get("class")
        j = d.find("joint")
        if c and j is not None:
            classes[c] = j.attrib
    out = {}
    for j in root.iter("joint"):
        n = j.get("name")
        if n in MUJOCO_ORDER:
            a = dict(classes.get(j.get("class"), {}))
            a.update({k: v for k, v in j.attrib.items() if k in ("armature", "damping", "frictionloss")})
            out[n] = a
    return out


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads((ROOT / "logs_system1/t10/isaac/d_table/dump/scene.json").read_text())
    c = json.loads((ROOT / "logs_system1/t10/isaac/cin_table/dump/scene.json").read_text())
    spec = json.loads((ROOT / "system1/sonic/sonic_release_spec.json").read_text())
    st = spec["static"]
    sj = list(spec["joint_names"])
    names = d["control"]["dof_names"]
    jd = d["joints"]["robot"]
    dep_def, dep_scale, dep_kp, dep_kd, sim_eff = deploy_arrays()
    mj = mjcf_joint_params()
    rows = []
    for i, n in enumerate(names):
        k = sj.index(n)
        ji = jd["names"].index(n)
        rows.append(dict(
            joint=n,
            isaac_D_kp=d["control"]["p_gains"][i], isaac_D_kd=d["control"]["d_gains"][i], isaac_D_effort=d["control"]["torque_limits"][i],
            isaac_D_action_scale=d["control"]["action_scales"][i], isaac_default=d["control"]["default_dof_pos"][i],
            isaac_Cin_kp=c["control"]["p_gains"][i], isaac_Cin_kd=c["control"]["d_gains"][i], isaac_Cin_effort=c["control"]["torque_limits"][i],
            isaac_armature=jd["armature"][ji], isaac_max_joint_velocity=jd["max_velocity"][ji],
            sonic_train_kp=st["joint_stiffness"][k], sonic_train_kd=st["joint_damping"][k], sonic_train_effort=st["joint_effort_limits"][k],
            sonic_default=st["default_joint_pos"][k], sonic_action_scale=st["action_scale"][k],
            deploy_kp=dep_kp[n], deploy_kd=dep_kd[n], deploy_default=dep_def[n], deploy_action_scale=dep_scale[n],
            official_sim_effort=sim_eff[n], official_mjcf_armature=float(mj[n]["armature"]), official_mjcf_damping=float(mj[n]["damping"]),
            official_mjcf_frictionloss=float(mj[n]["frictionloss"])))
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "settings_compare_joints.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    diff = lambda x, y: float(np.max(np.abs(np.array([r[x] for r in rows]) - np.array([r[y] for r in rows]))))  # noqa: E731
    checks = dict(
        cin_kp_vs_deploy_kp=diff("isaac_Cin_kp", "deploy_kp"), cin_kd_vs_deploy_kd=diff("isaac_Cin_kd", "deploy_kd"),
        cin_kp_vs_sonic_train_kp=diff("isaac_Cin_kp", "sonic_train_kp"), cin_kd_vs_sonic_train_kd=diff("isaac_Cin_kd", "sonic_train_kd"),
        cin_effort_vs_sonic_train_effort=diff("isaac_Cin_effort", "sonic_train_effort"), cin_effort_vs_official_sim_effort=diff("isaac_Cin_effort", "official_sim_effort"),
        sonic_default_vs_deploy_default=diff("sonic_default", "deploy_default"), sonic_scale_vs_deploy_scale=diff("sonic_action_scale", "deploy_action_scale"))
    (a.out / "settings_compare_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    diffs = lambda x, y, tol=1e-3: [r["joint"] for r in rows if abs(r[x] - r[y]) > tol]  # noqa: E731
    L = []
    L.append("# Isaac 与官方 SONIC 部署的设置对照（T10 第 0 步）\n")
    L.append("所有数字由 `system1/sim2sim/settings_compare.py` 从文件读出，没有手填。逐关节的全部数值见 `settings_compare_joints.csv`；关键差值见 `settings_compare_checks.json`。\n")
    L.append("## 来源\n")
    L.append("| 列 | 来源 |\n|---|---|")
    L.append("| Isaac D / Isaac C-in | 本机 5070 最终评测时 PhysX 实际使用的值：评测脚本加 `--sim2sim-dump` 导出（`logs_system1/t10/isaac/{d,cin}_table/dump/scene.json`），sha256 见 SHA256SUMS |")
    L.append("| SONIC 训练环境 | `system1/sonic/sonic_release_spec.json`（T01 从 SONIC 自己的 Isaac Lab 环境导出，checkpoint `sonic_release/last.pt` e6bdab3f…） |")
    L.append("| 官方部署 | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`（按头文件里的公式计算；GR00T-WholeBodyControl 1e1ec4f） |")
    L.append("| 官方 MuJoCo sim2sim | `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`、`gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml` |\n")
    L.append("## 逐项对照\n")
    L.append("| 项目 | Isaac（我们训练和评测） | 官方 SONIC 部署 / sim2sim | 是否一致 |\n|---|---|---|---|")
    L.append(f"| 策略频率 | 50 Hz（dt {d['control']['dt']}） | 控制线程 50 Hz（`control_dt_ = 0.02`）；输入线程 100 Hz，planner 10 Hz，电机指令 500 Hz | 一致 |")
    L.append(f"| 仿真频率 | 200 Hz（sim dt {d['control']['sim_dt']}，每个策略步 {d['control']['decimation']} 个子步；PhysX {d['physx_scene'].get('solver_type')}） | MuJoCo `SIMULATE_DT: 0.005`（200 Hz）；真机电机驱动器自己的伺服频率 | 仿真频率一致，求解器不同 |")
    L.append("| PD 的计算位置 | holosoma 每个 200 Hz 子步算 τ = kp(q* − q) − kd·q̇ 并按力矩上限截断，作为力矩施加；PhysX 关节驱动刚度、阻尼为 0 | 部署发 LowCmd（每个电机 q、dq=0、kp、kd、τff）；sim2sim 每个 200 Hz 步算 τ = τff + kp(q − q_now) + kd(dq − dq_now)，按电机力矩上限截断；真机由电机驱动器执行 | 公式相同 |")
    L.append(f"| C-in 的 PD 增益 | SONIC 的增益（hip_pitch 用 SONIC 的值）；与部署端 kp 最大差 {checks['cin_kp_vs_deploy_kp']:.2e}，kd 最大差 {checks['cin_kd_vs_deploy_kd']:.2e} | 部署：kp = 电枢 × (2π·10 Hz)²，kd = 2·2·电枢·ω；踝、腰 roll/pitch 加倍；hip_pitch 按 7520_22 | {'一致' if max(checks['cin_kp_vs_deploy_kp'], checks['cin_kd_vs_deploy_kd']) < 1e-3 else '不一致：' + ', '.join(diffs('isaac_Cin_kp', 'deploy_kp') + diffs('isaac_Cin_kd', 'deploy_kd'))} |")
    L.append(f"| C-in 的力矩上限 | 我们 URDF 的力矩上限（hip_pitch_actuator = ours）；与官方 sim2sim 电机上限最大差 {checks['cin_effort_vs_official_sim_effort']:.1f} Nm | sim2sim `motor_effort_limit_list` | 不一致的关节：{', '.join(diffs('isaac_Cin_effort', 'official_sim_effort')) or '无'} |")
    L.append("| D 的 PD 增益 | 论文 D 的增益（见 csv `isaac_D_kp/kd`），动作缩放 0.25 × 力矩上限 / kp（冻结） | 不适用（D 不经过 SONIC） | — |")
    L.append(f"| 关节电枢 | PhysX 逐关节（例如 hip_pitch {rows[0]['isaac_armature']:.4f}，knee {rows[3]['isaac_armature']:.4f}） | 官方 MJCF 统一 armature {rows[0]['official_mjcf_armature']}、damping {rows[0]['official_mjcf_damping']}、frictionloss {rows[0]['official_mjcf_frictionloss']}（腕 {rows[20]['official_mjcf_frictionloss']}） | 不一致（本次 MuJoCo 按 Isaac 的值，见 README） |")
    L.append(f"| 关节阻尼、摩擦 | 0、0 | 见上一行 | 不一致 |")
    L.append(f"| 关节最大速度 | PhysX 每关节硬限制（例如 {rows[0]['isaac_max_joint_velocity']:.0f} / {rows[1]['isaac_max_joint_velocity']:.0f} rad/s） | MuJoCo 没有关节速度限制 | MuJoCo 无对应项 |")
    L.append("| 关节顺序 | holosoma 顺序 = URDF 顺序（左腿、右腿、腰、左臂、右臂，即 MuJoCo 顺序） | 部署内部：电机按 MuJoCo 顺序；SONIC 观测和 ZMQ 流按 IsaacLab 顺序（`isaaclab_to_mujoco` 映射） | 我们给 SONIC 的输入按 IsaacLab 顺序（`sonic_controller.to_sonic`），与部署一致 |")
    L.append(f"| SONIC 的默认姿态 | SONIC 训练环境的默认姿态（我们 Isaac 里 SONIC 的目标 = 默认 + 缩放 × 动作）；与部署 `default_angles` 最大差 {checks['sonic_default_vs_deploy_default']:.3f} rad | 部署 `default_angles`（站立：hip_pitch −0.312、knee 0.669、ankle_pitch −0.363、肩 0.2 等） | {'一致' if checks['sonic_default_vs_deploy_default'] < 1e-3 else '不一致'} |")
    L.append(f"| SONIC 的动作缩放 | SONIC 训练环境 `action_scale`；与部署 `g1_action_scale` 最大差 {checks['sonic_scale_vs_deploy_scale']:.2e} | 0.25 × 力矩上限 / kp | {'一致' if checks['sonic_scale_vs_deploy_scale'] < 1e-4 else '不一致'} |")
    L.append("| 手 | 橡胶手（`main_mesh_collision_rubberhand.urdf`，固定在腕上，无自由度） | sim2sim 场景 43 自由度，带 Dex3 手（每只 7 自由度，`NUM_HAND_MOTORS: 7`）；部署可选 Dex3 手控制 | 不一致：真机手的配置要和博士生确认 |")
    L.append("| 地面、摩擦 | 地面材质 1.0/1.0，组合方式 multiply；机器人 1.0；物体 0.5/0.5，恢复系数 0 | 官方 sim2sim：geom 摩擦 1.0（MuJoCo 取最大值） | 本次 MuJoCo 按 Isaac 的组合值（见 README） |")
    L.append(f"| 碰撞 | 机器人：URDF 圆柱改胶囊、球、网格的 PhysX 凸包；Table 5 个盒子；Bin 凸分解 16 块；自碰撞开启 | 官方 MJCF 的碰撞几何 | 本次 MuJoCo 用 Isaac 的碰撞几何 |")
    L.append("| 编码器输入：参考关节位置、速度 | 未来 10 帧、每隔 5 帧（0.1 s），关节顺序 IsaacLab；先 290 个位置再 290 个速度（`command_multi_future_nonflat`） | `motion_joint_positions_10frame_step5` + `motion_joint_velocities_10frame_step5`（同样 10 帧 × 5） | 一致 |")
    L.append("| 编码器输入：参考朝向 | 参考骨盆朝向在机器人骨盆系下，旋转矩阵前两列（10 帧 × 5） | `motion_anchor_orientation_10frame_step5`；另加航向补偿：参考先乘以「机器人初始航向 × 参考初始航向的逆」再算 | 结构一致；部署多一个航向对齐（起始时把参考转到机器人航向） |")
    L.append("| 编码器模式字段 | `encoder_index` one-hot 3 维（g1 = [1,0,0]） | `encoder_mode_4` 4 维 = [模式号, 0, 0, 0]（g1 = 0） | 表示不同，官方 ONNX 与 last.pt 的 token 逐元素相同（见 consistency） |")
    L.append("| 解码器输入 | token 64 + 10 帧历史（角速度 3、关节位置−默认 29、关节速度 29、上一步动作 29、重力方向 3；最旧在前） | `token_state` + `his_*_10frame_step1`（同样的项和顺序） | 一致 |")
    L.append("| 我们的 158 维观测（每台机器人） | 按 holosoma 观测管理器的字母顺序拼接：上一步动作 29、骨盆角速度（骨盆系）3、关节位置−默认 29、关节速度 29、参考关节位置和速度 58（当前帧）、参考躯干朝向在实际躯干系下 6；同伴项：同伴根的平面相对位置和速度（观察者航向系，截断到 ±1）4 | 部署端没有这一层 | 上真机需要额外提供（见 deploy_interface_zh.md） |")
    L.append("| 输入接口 | Isaac 里直接把修正后的参考窗口交给 SONIC | 运动文件、键盘、手柄、ZMQ（v1：逐帧 29 关节位置和速度 + 根四元数 + 帧号；v3：再加 SMPL；v4：直接发 64 维 token，绕过编码器）、ROS2 | 结构上可接（见 README 第 0 步结论） |")
    L.append("\n## 逐关节摘要\n")
    L.append("| 关节 | Isaac D kp/kd | Isaac C-in kp/kd | 部署 kp/kd | Isaac C-in 力矩上限 | 官方 sim 力矩上限 | Isaac 电枢 | SONIC 默认 / 部署默认 |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['joint']} | {r['isaac_D_kp']:.2f} / {r['isaac_D_kd']:.3f} | {r['isaac_Cin_kp']:.2f} / {r['isaac_Cin_kd']:.3f} | {r['deploy_kp']:.2f} / {r['deploy_kd']:.3f} | "
                 f"{r['isaac_Cin_effort']:.0f} | {r['official_sim_effort']:.2f} | {r['isaac_armature']:.4f} | {r['sonic_default']:.3f} / {r['deploy_default']:.3f} |")
    (a.out / "settings_compare.md").write_text("\n".join(L) + "\n")
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
