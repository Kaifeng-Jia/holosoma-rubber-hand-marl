#!/usr/bin/env python3
"""T01b figures (Kevin 2026-11-10), existing episodes only, nothing selected:

(a) keyframes_<task>.png: top row scheme A nominal (T01b main, +/-2 directory), bottom row D seed 721 nominal
    (V3 closeout Full +/-2, existing trajectory; no re-run), frames at 0/25/50/75/100 % of the reference duration.
    A tile after A's termination shows A's terminal state, marked as such.
(b) topdown_<task>.png: reference pelvis routes of both robots and the reference object-centre route, with the
    actual routes of A and D (nominal thick; all 50 +/-2 episodes thin), A termination points marked.
Rendering: MuJoCo kinematic replay of recorded states (system1/render/render_traj.build_model), fixed camera per task.
Run with sonic_s1 (mujoco + matplotlib), MUJOCO_GL=egl.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.render.render_traj import ROBOT_URDF, build_model, urdf_joint_order  # noqa: E402

TASKS = ("push", "pull", "kick", "chair", "bucket", "table")
A_DIR = ROOT / "logs_system1/T01b/schemeA_main"
V3 = Path("/home/kevin/holosoma-core4d-base/paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/results/episodes.csv")
OUT = ROOT / "results_system1/T01b/figures"
FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)


def d_paths():
    nom, eps = {}, {}
    with V3.open() as f:
        for r in csv.DictReader(f):
            if r["variant"] == "Full" and r["training_seed"] == "721" and r["perturbation_level_cm"] == "2":
                if r["is_nominal"] == "True":
                    nom[r["task"]] = r["trajectory_path"]
                else:
                    eps.setdefault(r["task"], []).append(r["trajectory_path"])
    return nom, eps


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def ref_offset(z, obj_ref):
    """env-origin offset: reference_object_pos_w - reference object position at that frame (constant)."""
    f = z["reference_frame"].astype(int)
    off = z["reference_object_pos_w"] - obj_ref[f]
    assert np.max(np.abs(off - off[0])) < 3e-6
    return off[0]


def render_tiles(task, trajs, frames, obj_urdf, size=(480, 320)):
    """trajs: list of trajectory dicts (same task); frames: list of row index lists. Returns images[row][col]."""
    import mujoco

    m = build_model(2, obj_urdf)
    d = mujoco.MjData(m)
    names = urdf_joint_order(ROBOT_URDF)
    adr = [[m.joint(f"r{r}_{n}").qposadr[0] for n in names] for r in range(2)]
    free = [m.joint(m.body(f"r{r}_pelvis").jntadr[0]).qposadr[0] for r in range(2)]
    ob = [i for i in range(m.nbody) if m.body(i).name.startswith("obj_")][0]
    ofree = m.joint(m.body(ob).jntadr[0]).qposadr[0]
    rnd = mujoco.Renderer(m, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    allp = np.concatenate([t["root_pos"].reshape(-1, 3) for t in trajs] + [t["object_pos_w"] for t in trajs])
    lo, hi = allp.min(0), allp.max(0)
    cam.lookat[:] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.6]
    cam.distance = float(max(3.2, 1.6 * max(hi[0] - lo[0], hi[1] - lo[1]) + 2.0))
    cam.elevation, cam.azimuth = -28.0, 135.0
    imgs = []
    for t, fr in zip(trajs, frames):
        row = []
        quat = t["root_quat_xyzw"][..., [3, 0, 1, 2]]
        oq = t["object_quat_xyzw"][..., [3, 0, 1, 2]]
        for k in fr:
            for r in range(2):
                d.qpos[free[r]:free[r] + 3] = t["root_pos"][k, r]
                d.qpos[free[r] + 3:free[r] + 7] = quat[k, r]
                d.qpos[adr[r]] = t["dof_pos"][k, r]
            d.qpos[ofree:ofree + 3] = t["object_pos_w"][k]
            d.qpos[ofree + 3:ofree + 7] = oq[k]
            mujoco.mj_forward(m, d)
            rnd.update_scene(d, camera=cam)
            row.append(np.ascontiguousarray(rnd.render()))
        imgs.append(row)
    return imgs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
               "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    out = OUT if len(sys.argv) < 3 else Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tasks = TASKS if len(sys.argv) < 2 or sys.argv[1] == "all" else tuple(sys.argv[1].split(","))
    dnom, deps = d_paths()
    prov = {}
    for task in tasks:
        sa = json.loads((A_DIR / f"{task}_pm2/summary.json").read_text())
        N = int(sa["reference_frames"])
        a_nom = load(A_DIR / f"{task}_pm2/nominal.trajectory.npz")
        d_nom = load(dnom[task])
        obj_urdf = ROOT / "src/holosoma" / sa["object_urdf"]
        want = [int(round(p * (N - 1))) for p in FRACS]
        rows, notes = [], []
        for t in (a_nom, d_nom):
            rf = t["reference_frame"].astype(int)
            idx, nt = [], []
            for w in want:
                hit = np.nonzero(rf >= w)[0]
                if len(hit):
                    idx.append(int(hit[0])); nt.append("")
                else:
                    idx.append(len(rf) - 1); nt.append(f"已终止（第 {len(rf) - 1} 步）")
            rows.append(idx); notes.append(nt)
        imgs = render_tiles(task, [a_nom, d_nom], rows, obj_urdf)
        fig, ax = plt.subplots(2, 5, figsize=(16, 6.4))
        for r, lab in enumerate(("A：SONIC 静态目标（nominal）", "D：端到端 MARL，seed 721（nominal）")):
            for c in range(5):
                ax[r, c].imshow(imgs[r][c]); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
                if r == 0:
                    ax[r, c].set_title(f"参考时长 {int(FRACS[c] * 100)}%（第 {want[c]} 帧）", fontsize=11)
                if notes[r][c]:
                    ax[r, c].imshow(np.full_like(imgs[r][c], 255), alpha=0.35)
                    ax[r, c].text(0.5, 0.06, notes[r][c], transform=ax[r, c].transAxes, ha="center", fontsize=10,
                                  color="#b00020", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
            ax[r, 0].set_ylabel(lab, fontsize=11)
        fig.suptitle(f"{task}：关键帧（运动学回放已有回合，未挑选）", fontsize=13)
        fig.tight_layout()
        fa = out / f"keyframes_{task}.png"
        fig.savefig(fa, dpi=110); plt.close(fig)
        # (b) top-down routes in the reference frame
        ref = np.load(ROOT / "logs_system1/T01b/figures_cache" / f"{task}_ref.npz")
        pel, obj = ref["pelvis"], ref["object"]
        fig, axb = plt.subplots(figsize=(7.5, 7.5))
        for r, cr in enumerate(("#222222", "#666666")):
            axb.plot(pel[:, r, 0], pel[:, r, 1], color=cr, lw=2.4, label=f"参考骨盆 机器人{r}")
        axb.plot(obj[:, 0], obj[:, 1], color="#2e7d32", lw=2.4, ls="--", label="参考物体中心")
        a_eps = [A_DIR / f"{task}_pm2/{e['state_id']}.trajectory.npz" for e in sa["episodes"] if e["kind"] == "perturbed"]
        for paths, col, nm in ((a_eps, "#1f77b4", "A"), (deps[task], "#ff7f0e", "D")):
            for p in paths:
                t = load(p); o = ref_offset(t, obj)
                for r in range(2):
                    axb.plot(t["root_pos"][:, r, 0] - o[0], t["root_pos"][:, r, 1] - o[1], color=col, lw=0.5, alpha=0.25)
                if nm == "A":
                    axb.plot(*(t["root_pos"][-1, :, :2] - o[:2]).T, "x", color=col, ms=4, alpha=0.5)
        for t, col, nm in ((a_nom, "#1f77b4", "A nominal"), (d_nom, "#ff7f0e", "D nominal")):
            o = ref_offset(t, obj)
            for r in range(2):
                axb.plot(t["root_pos"][:, r, 0] - o[0], t["root_pos"][:, r, 1] - o[1], color=col, lw=2.0, label=nm if r == 0 else None)
            axb.plot(t["object_pos_w"][:, 0] - o[0], t["object_pos_w"][:, 1] - o[1], color=col, lw=1.6, ls=":",
                     label=f"{nm} 物体中心")
        o = ref_offset(a_nom, obj)
        axb.plot(*(a_nom["root_pos"][-1, :, :2] - o[:2]).T, "X", color="#d62728", ms=11, label="A nominal 终止点")
        axb.plot([], [], "x", color="#1f77b4", label="A ±2 各回合终止点")
        axb.plot([], [], color="#1f77b4", lw=0.8, alpha=0.5, label="A ±2 全部 50 回合")
        axb.plot([], [], color="#ff7f0e", lw=0.8, alpha=0.5, label="D seed721 ±2 全部 50 回合")
        axb.set_aspect("equal"); axb.grid(alpha=0.3); axb.set_xlabel("x (m)"); axb.set_ylabel("y (m)")
        axb.set_title(f"{task}：俯视路线（参考坐标系）")
        axb.legend(fontsize=7.5, loc="best")
        fig.tight_layout()
        fb = out / f"topdown_{task}.png"
        fig.savefig(fb, dpi=120); plt.close(fig)
        prov[task] = dict(a_nominal=str(A_DIR / f"{task}_pm2/nominal.trajectory.npz"), d_nominal=dnom[task],
                          a_episodes=len(a_eps), d_episodes=len(deps[task]), keyframe_reference_frames=want,
                          keyframe_rows=rows, files=[fa.name, fb.name])
        print(task, "done", flush=True)
    (out / "figures_provenance.json").write_text(json.dumps(dict(
        note="existing episodes only; A = T01b scheme A main +/-2 (nominal + all 50); D = V3 Full seed 721 +/-2 "
             "(nominal + all 50); kinematic MuJoCo replay of recorded states; keyframe = first saved row whose "
             "reference_frame >= target frame, else the terminal row (marked)", tasks=prov), indent=1) + "\n")


if __name__ == "__main__":
    main()
