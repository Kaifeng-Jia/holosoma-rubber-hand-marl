#!/usr/bin/env python3
"""Compare 64/100-point Omni-style meshes against an existing CORE4D replay.

Read-only with respect to inputs. Writes only to a new explicit audit directory.
No optimization, training, physics rollout, reward/config mutation or checkpoint
selection is performed. The chosen replay is a development example, not a test
of unseen-seed generalization or contact-force causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.interaction_mesh_audit import (  # noqa: E402
    analyze_budget,
    load_pair_states,
    points_in_object_frame,
    rotation_matrices,
)
from holosoma_retargeting.interaction_mesh_geometry import (  # noqa: E402
    build_robot_landmarks,
    coverage_report,
    load_box_object_mesh,
    sample_object_points,
)

MOTION_DIR = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
DEFAULT_REFERENCE = MOTION_DIR / "core4d_smalltable/core4d_pair_runtime_fps50.npz"
DEFAULT_XML = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
DEFAULT_TABLE = MOTION_DIR / "objects_core4d_desk001_small_training.urdf"
DEFAULT_ROLLOUT = ROOT / (
    "logs/Core4DSmallTable/paired_reference_fresh12000_save2000_actor158_seed721_env2048/"
    "evaluation/model12000_seed721/representative_episode.npz"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def invariance_check(states: dict) -> float:
    rotation = Rotation.from_euler("xyz", [0.2, -0.1, 0.7]).as_matrix()
    translation = np.array([1.2, -2.3, 0.4])
    maximum = 0.0
    for name in ("reference", "actual"):
        points = states[f"{name}_landmarks_w"]
        pose = states[f"{name}_object_qpos"]
        transformed_points = points @ rotation.T + translation
        # Recorded poses are float32; keep this numerical invariance test in
        # float64 instead of rounding the synthetic transform back to float32.
        transformed_pose = np.array(pose, dtype=np.float64, copy=True)
        transformed_pose[:, :3] = pose[:, :3] @ rotation.T + translation
        quat = Rotation.from_matrix(rotation @ rotation_matrices(pose[:, 3:])).as_quat()
        transformed_pose[:, 3:] = quat[:, [3, 0, 1, 2]]
        difference = points_in_object_frame(transformed_points, transformed_pose) - states[f"{name}_local"]
        maximum = max(maximum, float(np.max(np.abs(difference))))
    return maximum


def plot_results(output: Path, states: dict, arrays: dict, specs: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.arange(len(states["reference_local"])) / states["fps"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for count, color in ((64, "tab:blue"), (100, "tab:orange")):
        curve = 100 * np.sqrt(arrays[f"grouped_mse_m2_{count}"])
        for agent in range(2):
            axes[0, agent].plot(time, curve[:, agent], label=f"budget {count}", color=color, alpha=0.85)
            axes[0, agent].set(title=f"Agent {agent + 1}: interaction geometry", ylabel="Grouped RMS (cm)", xlabel="Time (s)")
            axes[0, agent].legend()
            axes[0, agent].grid(alpha=0.2)
    independent_error = np.linalg.norm(states["actual_local"] - states["reference_local"], axis=-1) * 100
    max_error = np.percentile(independent_error, 98)
    for agent in range(2):
        artist = axes[1, agent].imshow(
            independent_error[:, agent].T, origin="lower", aspect="auto", vmin=0, vmax=max_error,
            extent=[0, time[-1], -0.5, len(specs)-0.5], cmap="magma",
        )
        axes[1, agent].set_yticks(np.arange(len(specs)), [s["name"] for s in specs], fontsize=6)
        axes[1, agent].set(title=f"Agent {agent + 1}: object-relative point error", xlabel="Time (s)")
        figure.colorbar(artist, ax=axes[1, agent], label="cm (not force/contact)")
    figure.savefig(output / "interaction_errors.png", dpi=160)
    plt.close(figure)

    figure = plt.figure(figsize=(12, 5), constrained_layout=True)
    for panel, count in enumerate((64, 100), start=1):
        axis = figure.add_subplot(1, 2, panel, projection="3d")
        points = arrays[f"object_points_{count}"]
        axis.scatter(*points.T, s=15)
        axis.set(title=f"Budget {count}, returned {len(points)} points", xlabel="Object X (m)", ylabel="Object Y (m)", zlabel="Object Z (m)")
        axis.set_box_aspect(np.ptp(points, axis=0))
    figure.savefig(output / "object_samples.png", dpi=160)
    plt.close(figure)


def write_report(output: Path, report: dict, states: dict, specs: list[dict]) -> None:
    lines = [
        "# CORE4D 小桌交互网格离线验证", "",
        "本次没有训练、没有重新重定向、没有改变现有奖励/Actor/物体资产。",
        "以下是几何关系诊断，不是接触力、协作承载或动力学可行性的证明。", "",
        "## 输入和选点", "",
        "- 使用已确认的 50 Hz 双机器人小桌参考与 model_12000 的已保存实际回放。",
        "- Actual 是回放 root、29 关节角经参考/可视化 MuJoCo 模型 FK 重建的点位，不是直接记录的 PhysX 身体状态；不执行新物理仿真。",
        "- 训练 URDF 与参考模型可能存在安装偏移；这项差异须单独核对。未来在线奖励应直接读取仿真身体状态，而不是沿用此回放近似。",
        "- 身体为 Omni 原 15 点＋每只橡胶手两个固定网格表面点＝每台 19 点；不增加头点。",
        "- 手部三个点共用一个身体组的权重；附加点表达刚体朝向，不等于准确的掌面接触/法向标注。",
        "- 表面点只在当前小桌训练碰撞几何外表面采样，预算 64/100、seed 42；不人为补齐采样数量。",
        "- 原采样算法在两种预算下不是嵌套子集，结果同时包含采样布局变化，不声称是严格纯点数消融。",
        "- 每帧用参考建图并冻结，实际回放不会改变邻接和评分权重。", "",
        "## 数学与计分", "",
        "Laplacian 使用现有 Omni 的 Delaunay、邻接和均匀邻居权重函数。",
        "记录原始全节点平方和、全节点均值，以及诊断用的分组误差：",
        "`E_grouped = 0.5 × 身体组均方误差 + 0.5 × 与身体相邻的物体节点均方误差`。",
        "纯物体邻接节点在物体坐标系内误差为零，不纳入第二组均值；但与人体相连的物体节点保留。",
        "50/50 是本次离线比较的固定约定，不是已批准的最终训练奖励权重。", "",
        "## 一致性检查", "",
        f"- 参考 FK 最大位置差：{report['fk_max_position_error_m']:.3g} m。",
        f"- 整体刚体变换后的物体系坐标最大差：{report['rigid_transform_max_error_m']:.3g} m。",
        "- 参考与自身的误差为零；单独位移、手绕原点转向是人工构造的度量单测，不是训练数据。", "",
        "## 点预算与现有回放", "",
        "| 请求预算 | 实际物体点数 | 机器人1 RMS | 机器人2 RMS | 构图 CPU ms/机器人帧 | 计分 CPU ms/机器人帧 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for count in (64, 100):
        item = report["budgets"][str(count)]
        errors = item["actual_grouped_rms_cm_per_agent"]
        lines.append(f"| {count} | {item['actual_object_points']} | {errors[0]:.2f} cm | {errors[1]:.2f} cm | {item['cpu_graph_build_ms_per_agent_frame']:.3f} | {item['cpu_score_ms_per_agent_frame']:.3f} |")
    lines += ["", "此表不是训练效果的比较：两行用的是同一个实际回放。分数变化不意味着某预算对应更好的策略。",
              "CPU 离线计时不代表 GPU/2048 环境训练开销。详细覆盖、敏感性、拓扑变化见 summary.json。", "",
              "## 人工扰动敏感性（平均 MSE，单位 m²）", "",
              "| 扰动 | 64预算 | 100预算 |", "|---|---:|---:|"]
    for key in report["budgets"]["64"]["sensitivity_mean_mse_m2"]:
        first = report["budgets"]["64"]["sensitivity_mean_mse_m2"][key]
        second = report["budgets"]["100"]["sensitivity_mean_mse_m2"][key]
        lines.append(f"| {key} | {first:.7g} | {second:.7g} |")
    lines += ["", "## 点定义", "", "| 名称 | 所属刚体 | 刚体局部坐标（m） |", "|---|---|---|"]
    for spec in specs:
        lines.append(f"| {spec['name']} | {spec['body']} | {np.array2string(np.asarray(spec['local_xyz']), precision=5)} |")
    lines += ["", "## 图像", "", "![关系误差](interaction_errors.png)", "", "![物体采样](object_samples.png)", "",
              "## 解释边界", "",
              "- 这是单条已用于开发的参考和一次代表回放，不是独立测试集或多训练 seed 的结论。",
              "- 关系误差低不等于两人都接触/施力；现有回放不含可区分机器人—桌子作用的接触力。",
              "- 手部点更密会改变 Delaunay 拓扑，按身体组归一化不能完全消除这种结构变化。",
              "- 只比较整段不够：末段打招呼/松手属于参考内容，不应当一律要求持续搬运。",
              "- 未选定正式训练采样预算或奖励权重；先看选点及误差定位，再与用户确认。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--object-urdf", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output directory exists; use a new directory to preserve earlier results")
    paths = {key: getattr(args, key).resolve() for key in ("reference", "rollout", "robot_xml", "object_urdf")}
    before_hashes = {key: digest(path) for key, path in paths.items()}
    specs, geometry_meta = build_robot_landmarks(paths["robot_xml"])
    states = load_pair_states(paths["reference"], paths["rollout"], paths["robot_xml"], specs)
    mesh, object_meta = load_box_object_mesh(paths["object_urdf"])
    report = {
        "scope": "offline_geometry_only_no_training_or_contact_force_claim",
        "input_paths": paths, "input_sha256": before_hashes,
        "robot_geometry": geometry_meta, "object_geometry": object_meta,
        "landmarks": specs, "rollout_metadata": states["metadata"],
        "fk_max_position_error_m": states["fk_max_position_error_m"],
        "fk_max_quaternion_dot_error": states["fk_max_quaternion_dot_error"],
        "rigid_transform_max_error_m": invariance_check(states), "budgets": {},
    }
    if report["rigid_transform_max_error_m"] > 1e-10:
        raise ValueError("Object-frame invariance check failed")
    arrays = {key: states[key] for key in (
        "reference_robot_qpos", "actual_robot_qpos", "reference_object_qpos", "actual_object_qpos",
        "reference_landmarks_w", "actual_landmarks_w", "fps",
    )}
    arrays["node_names"] = np.asarray([spec["name"] for spec in specs])
    arrays["object_relative_point_error_m"] = np.linalg.norm(states["actual_local"] - states["reference_local"], axis=-1)
    for count in (64, 100):
        points, sampling_meta = sample_object_points(mesh, object_meta, count, seed=42)
        summary, budget_arrays = analyze_budget(states, points, specs)
        summary["sampling"] = sampling_meta
        summary["coverage"] = coverage_report(mesh, object_meta, points, seed=2026)
        report["budgets"][str(count)] = summary
        arrays[f"object_points_{count}"] = points
        arrays.update({f"{key}_{count}": value for key, value in budget_arrays.items()})
        print(json.dumps({"budget": count, "actual_points": len(points), "rms_cm": summary["actual_grouped_rms_cm_per_agent"]}), flush=True)
    after_hashes = {key: digest(path) for key, path in paths.items()}
    if after_hashes != before_hashes:
        raise ValueError("Input changed during audit")
    report["inputs_unchanged"] = True
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "audit_arrays.npz", **arrays)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    plot_results(args.output_dir, states, arrays, specs)
    write_report(args.output_dir, report, states, specs)
    print(f"Saved offline audit: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
