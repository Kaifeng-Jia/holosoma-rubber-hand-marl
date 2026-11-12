#!/usr/bin/env python3
"""T03 C v2 check 3c (Kevin 2026-11-12): does the object really reach its target in the C-in evaluations that count as
completed? CPU only; reads existing evaluator outputs.

Definitions (written before computing):
  * Sets, +/-2, the 50 perturbed states (nominal reported on its own):
      C-in it1000 (Pull: it1050, the first saved checkpoint after 1000), cloud RTX 4090 (logs_system1/cloud_evals/c2_curve);
      D, same evaluator, seed 721: the paper main result at its final checkpoint (Pull it8050, Table it12000, Bin it12000) and
      at the nearest evaluated iteration to C's (Pull it1050, Table it2000 (cloud), Bin it2000); for Pull also the D run
      trained from scratch like C Pull (V3-25, seed 721) at it1050 and it8050.
  * Per episode (completed or not; the summaries split completed / all):
      - push ratio / push distance: T02 preregistration 9.1 (system1/t02/object_analysis.push_at) at the last executed step;
      - object position RMSE: summary.json object_rmse_3d_m and object_rmse_xy_m (the evaluator's, over the executed steps);
      - terminal object error: last saved state, |object_pos - reference_object_pos| (m) and the rotation angle between the
        object and reference quaternions 2 acos |<q, q_ref>| (rad) (system1/package_scheme_a.py, the scheme A pack);
      - completed = the evaluator's completed_reference (all reference frames executed and joint_bad_tracking not raised; its
        object thresholds are 0.25 m and 0.8 rad).
  * Figures (nominal episode of each task): top view (x-y, world) of the object centre and of both pelvises, executed vs
    reference, C-in it1000 and D paper final side by side, fixed scheme colours (system1/render/scheme_colors.py); keyframe
    montage of the C-in nominal: system1/render/u5/render_keyframes.py (U5 pipeline, Unitree colours, F3 camera).

    hssim_s1/bin/python -B system1/t03/c2_object_check.py --out results_system1/T03/c2_checks/c_object
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t02.object_analysis import push_at  # noqa: E402

CL = ROOT / "logs_system1/cloud_evals"
PT = ROOT / "logs_system1/post_train"
SETS = {
    "pull": [("C-in it1050", "C", CL / "c2_curve/cin_pull_m20kg_seed721_joint8000/it01050_pm2"),
             ("D paper it1050", "D", PT / "d_curve/D_pull_paper_seed721/it01050_pm2"),
             ("D paper it8050 (final)", "D", PT / "d_curve/D_pull_paper_seed721/it08050_pm2"),
             ("D from scratch V3-25 it1050", "D", PT / "d_curve/D_pull_scratch_V3-25_seed721/it01050_pm2"),
             ("D from scratch V3-25 it8050 (final)", "D", PT / "d_curve/D_pull_scratch_V3-25_seed721/it08050_pm2")],
    "table": [("C-in it1000", "C", CL / "c2_curve/cin_table_m5kg_seed721_fresh12000/it01000_pm2"),
              ("D paper it2000", "D", CL / "d_curve/D_table_paper_T-07_seed721/it02000_pm2"),
              ("D paper it12000 (final)", "D", PT / "d_curve/D_table_paper_T-07_seed721/it12000_pm2")],
    "bucket": [("C-in it1000", "C", CL / "c2_curve/cin_bucket_m1kg_seed721_fresh12000/it01000_pm2"),
               ("D paper it2000", "D", PT / "d_curve/D_bucket_paper_B-RZ0_seed721/it02000_pm2"),
               ("D paper it12000 (final)", "D", PT / "d_curve/D_bucket_paper_B-RZ0_seed721/it12000_pm2")],
}
TASK_NAME = {"pull": "Table Pulling", "table": "Table Transport", "bucket": "Bin Transport"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return dict(n=0, median=None, mean=None, p90=None, max=None)
    a = np.asarray(v, float)
    return dict(n=len(v), median=float(np.median(a)), mean=float(a.mean()), p90=float(np.percentile(a, 90)), max=float(a.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, summ, prov = [], [], []
    nominal = {}
    for task, sets in SETS.items():
        for tag, scheme, d in sets:
            s = json.loads((d / "summary.json").read_text())
            m = json.loads((d / "run_manifest.json").read_text())
            assert s["complete"] and s["episode_count"] == 51, d
            prov.append(dict(task=task, set=tag, dir=str(d.relative_to(ROOT)), model_id=s["model_id"], checkpoint_sha256=s["checkpoint_sha256"],
                             checkpoint_iteration=s.get("checkpoint_iteration"), states_sha256=s["states_sha256"], gpu=m.get("gpu_name"),
                             summary_sha256=sha(d / "summary.json")))
            for e in s["episodes"]:
                tp = d / Path(e["trajectory"]).name
                with np.load(tp, allow_pickle=False) as z:
                    obj, ref = z["object_pos_w"].astype(np.float64), z["reference_object_pos_w"].astype(np.float64)
                    q, qr = z["object_quat_xyzw"].astype(np.float64), z["reference_object_quat_xyzw"].astype(np.float64)
                    if e["kind"] == "nominal":
                        nominal[(task, tag)] = dict(obj=obj, ref=ref, root=z["root_pos"].astype(np.float64), completed=e["completed_reference"],
                                                    steps=e["steps"], traj=tp, summary=d / "summary.json")
                pu = push_at(obj, ref, len(obj) - 1)
                rows.append(dict(task=task, set=tag, scheme=scheme, state_id=e["state_id"], kind=e["kind"], completed_reference=e["completed_reference"],
                                 executed_steps=e["steps"], horizon_steps=e["horizon_steps"], push_distance_m=pu["push_distance_m"],
                                 push_ratio=pu["push_ratio"], ref_disp_m=pu["ref_disp_m"], obj_disp_m=pu["obj_disp_m"],
                                 object_rmse_3d_m=e["object_rmse_3d_m"], object_rmse_xy_m=e["object_rmse_xy_m"],
                                 terminal_object_pos_error_m=float(np.linalg.norm(obj[-1] - ref[-1])),
                                 terminal_object_ori_error_rad=float(2 * np.arccos(min(1.0, abs(float(np.dot(q[-1], qr[-1])))))),
                                 trajectory_sha256=sha(tp)))
            for sel, filt in (("completed", lambda r: r["completed_reference"]), ("all", lambda r: True)):
                rr = [r for r in rows if r["task"] == task and r["set"] == tag and r["kind"] != "nominal" and filt(r)]
                x = dict(task=task, set=tag, scheme=scheme, episodes=sel, n=len(rr))
                for k in ("push_ratio", "push_distance_m", "object_rmse_3d_m", "object_rmse_xy_m", "terminal_object_pos_error_m",
                          "terminal_object_ori_error_rad"):
                    st = stats([r[k] for r in rr])
                    x.update({f"{k}_median": st["median"], f"{k}_mean": st["mean"], f"{k}_p90": st["p90"], f"{k}_max": st["max"]})
                x["n_terminal_pos_gt_0p25m"] = sum(r["terminal_object_pos_error_m"] > 0.25 for r in rr)
                x["n_terminal_ori_gt_0p8rad"] = sum(r["terminal_object_ori_error_rad"] > 0.8 for r in rr)
                summ.append(x)
    for name, rr in (("episodes.csv", rows), ("summary.csv", summ), ("provenance.csv", prov)):
        with (a.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    # top view of the nominal episodes
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from system1.render.scheme_colors import SCHEME_COLOR, SCHEME_NAME, use_cjk_font
    use_cjk_font()
    figs = []
    for task, sets in SETS.items():
        pick = [sets[0], next(s for s in sets if s[0].startswith("D paper") and "final" in s[0])]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
        for ax, (tag, scheme, d) in zip(axes, pick):
            n = nominal[(task, tag)]
            col = SCHEME_COLOR[scheme]
            ax.plot(n["ref"][:, 0], n["ref"][:, 1], color="#888888", lw=2, ls="--", label="参考：物体中心")
            ax.plot(n["obj"][:, 0], n["obj"][:, 1], color=col, lw=2, label="执行：物体中心")
            for r, mk in ((0, "o"), (1, "s")):
                ax.plot(n["root"][:, r, 0], n["root"][:, r, 1], color=col, lw=1, alpha=0.6, ls="-" if r == 0 else ":",
                        label=f"执行：机器人 {r} 骨盆（起点{'圆' if r == 0 else '方'}框）")
                ax.plot(n["root"][0, r, 0], n["root"][0, r, 1], mk, color=col, ms=8, mfc="white")
            ax.plot(n["obj"][0, 0], n["obj"][0, 1], "o", color=col, ms=8)
            ax.plot(n["obj"][-1, 0], n["obj"][-1, 1], "X", color=col, ms=9)
            ax.plot(n["ref"][-1, 0], n["ref"][-1, 1], "X", color="#888888", ms=9)
            ax.set_title(f"{SCHEME_NAME[scheme]}：{tag}\n{'完成' if n['completed'] else '未完成'}，执行 {n['steps']} 步", fontsize=10)
            ax.set_aspect("equal")
            ax.grid(color="#dddddd", lw=0.6)
            ax.set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        axes[1].legend(loc="best", fontsize=8, frameon=False)
        fig.suptitle(f"{TASK_NAME[task]} nominal 回合俯视轨迹（圆点 = 起点，叉 = 终点；灰色虚线 = 参考）", fontsize=11)
        fig.tight_layout()
        f = a.out / f"topview_nominal_{task}.png"
        fig.savefig(f, dpi=150)
        plt.close(fig)
        figs.append(dict(task=task, file=f.name, sets=[s[0] for s in pick], trajectories=[str(nominal[(task, s[0])]["traj"].relative_to(ROOT)) for s in pick]))
    (a.out / "figures.json").write_text(json.dumps(figs, indent=1))
    (a.out / "nominal_for_montage.json").write_text(json.dumps(
        {task: dict(trajectory=str(nominal[(task, sets[0][0])]["traj"]), summary=str(nominal[(task, sets[0][0])]["summary"]))
         for task, sets in SETS.items()}, indent=1))
    for x in summ:
        if x["episodes"] == "completed":
            print(x["task"], x["set"], "n", x["n"], "push_ratio med %.3f" % (x["push_ratio_median"] or float("nan")),
                  "rmse3d med %.3f" % (x["object_rmse_3d_m_median"] or float("nan")),
                  "term pos med/max %.3f/%.3f" % (x["terminal_object_pos_error_m_median"] or float("nan"), x["terminal_object_pos_error_m_max"] or float("nan")),
                  "term ori med/max %.3f/%.3f" % (x["terminal_object_ori_error_rad_median"] or float("nan"), x["terminal_object_ori_error_rad_max"] or float("nan")))


if __name__ == "__main__":
    raise SystemExit(main())
