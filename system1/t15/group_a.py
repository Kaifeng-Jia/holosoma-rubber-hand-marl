#!/usr/bin/env python3
"""T15 group A (paper Fig. 3b): palm-level contact close-ups. Person, OmniRetarget as is, ours, same frame and view.

References (1.0x): ours = T14 round 6 C (Table / Chair: exports_system1/T14_references/r6/<task>_k1p0_C pair; Push: the round 6
C single solve logs_system1/t14/r6/single_push_C, made a pair by dual_stretch.make_pair(m = 1.0) as every Push / Pull metric
of round 6, checked against the exported r6/push_sizes_C/m1p00 pair); OmniRetarget as is = the comparisons.csv omni_asis rows
(step6), sha256 checked. Object: the 1.0x URDF of round 6 C (Table k1p0, Chair k1p0, Push table m1p00); the as-is object track
must equal ours (checked).
Frame (task book): the middle frame of the person's hand-contact segment of robot 0's RIGHT hand (round-6 rule
contact_error3d: any SMPL-X hand vertex within 1.5 cm of the captured mesh; CORE4D person 1 <-> robot 0, 15 Hz source frame
t <-> reference frame 2t; Push: the single person, 30 Hz). Segment = the longest run of consecutive contact frames of that
hand (all runs of both hands recorded); middle = (first + last) // 2 (source frames).
Views:
  close-up (human_render.py / render_sizes.py round-4 close-up): robot 0 in profile, azimuth = robot 0 root yaw + 90 deg
    (ours, that frame), elevation -10 deg, fitted (U5 camera_for) to robot 0's two rubber-hand link origins +- 0.15 m of ours
    AND as-is (one camera for both robot images); robot 1 hidden. Person: same azimuth / elevation, look-at = the person's
    wrist midpoint, distance = robot distance / human-to-robot scale; the other person not drawn; floor at the person's
    lowest vertex;
  full body: azimuth + 30 deg, elevation -15 deg, fitted to robot 0 + object mesh bounds of ours and as-is; person: same
    azimuth / elevation, look-at = centre of the person + captured-object bounds, distance / scale.
Per image (t15_scene): PNG 2560 x 1440, .seg.png, .json (camera, frame / time, input sha256, key-point pixels with occlusion,
subject extent, and for robot images the frame's 3-D contact-position error and palm-normal angle of robot 0's hands with the
round-6 code: contact_error3d / hand_metrics per-frame values; the clip means of the same code are checked against
step6/contact_error_3d.csv and step6/hand_metrics.csv). The person and "ours" close-up / full images also get
<stem>.traj.json: the object-centre pixel track of the whole reference in that camera (U5 F3_traj format).

    MUJOCO_GL=egl sonic_s1/bin/python -B system1/t15/group_a.py --out exports_system1/T15_figures/A
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "system1/t14", ROOT / "system1/t12v2", ROOT / "src/holosoma_retargeting"):
    sys.path.insert(0, str(p))
import t15_scene as ts  # noqa: E402

R6, R3 = ROOT / "exports_system1/T14_references/r6", ROOT / "exports_system1/T14_references/r3"
STEP6 = ROOT / "results_system1/T14_retargeting/step6"
TASKS = dict(
    table=dict(kind="core4d", ours=R6 / "table_k1p0_C/20231030_001_desk001_move_new_fps30.npz", omni=R3 / "asis_table_k1p0/20231030_001_desk001_move_omni_asis_fps30.npz",
               urdf=R6 / "table_k1p0_C/table_k1p0.urdf", name="Table Transport"),
    chair=dict(kind="core4d", ours=R6 / "chair_k1p0_C/chair_new_fps30.npz", omni=R3 / "asis_chair_k1p0/chair_omni_asis_fps30.npz",
               urdf=R6 / "chair_k1p0_C/chair_k1p0.urdf", name="Chair Righting"),
    push=dict(kind="single", ours=ROOT / "logs_system1/t14/r6/single_push_C/push_new_qpos.npz", omni=ROOT / "logs_system1/t14/r3/single_asis_push/push_omni_asis_qpos.npz",
              urdf=R6 / "push_sizes_C/m1p00/table_m1p00.urdf", export=R6 / "push_sizes_C/m1p00/push_m1p00_pair_fps30.npz", name="Table Pushing"))
CU_EL, CU_MARGIN, FULL_DAZ, FULL_EL = -10.0, 0.15, 30.0, -15.0


def runs(flag):
    out, i = [], 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j + 1 < len(flag) and flag[j + 1]:
                j += 1
            out.append([i, j])
            i = j + 1
        else:
            i += 1
    return out


def yaw_az(q):
    w, x, y, z = q[3:7]
    fwd = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w)])
    return float(np.degrees(np.arctan2(fwd[1], fwd[0]))) + 90.0


def load_pair(task, path):
    import contact_gap as cg
    import dual_stretch as ds
    if TASKS[task]["kind"] == "core4d":
        z = np.load(path)
        return z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64)
    q = cg.omomo_single(str(path), False)
    return ds.make_pair(task, q, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    a = ap.parse_args()
    assert os.environ.get("MUJOCO_GL") == "egl"
    import contact_error3d as ce
    import contact_gap as cg
    import dual_stretch as ds
    import hand_metrics as hm
    import human_render as hr
    from table_shared import palm_vertex_ids  # noqa: F401  (HumanScene.axes)
    robot = cg.cs.Robot("physx")
    offs = cg.palm_offsets()
    rh = hm.RobotHands()
    comp = {(r["task"], r["version"]): r for r in csv.DictReader(open(STEP6 / "comparisons.csv"))}
    ce_rows = list(csv.DictReader(open(STEP6 / "contact_error_3d/contact_error_3d.csv")))
    hm_rows = list(csv.DictReader(open(STEP6 / "hand_metrics.csv")))
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "system1/t15"], capture_output=True, text=True).stdout.strip()
    summary = dict(group="A", git_head=head, git_status_system1_t15=dirty, versions=ts.versions(), tasks={})
    for task in a.tasks:
        T = TASKS[task]
        checks = {}
        # ---------------------------------------------------------------- inputs and checks
        checks["omni_sha256_equals_comparisons_csv"] = ts.sha(T["omni"]) == comp[(task, "omni_asis")]["pair_sha256"]
        assert checks["omni_sha256_equals_comparisons_csv"], (task, "omni_asis pair differs from comparisons.csv")
        q_o, o_o = load_pair(task, T["ours"])
        q_a, o_a = load_pair(task, T["omni"])
        n = min(len(q_o), len(q_a))
        checks["frames_ours_asis"] = [len(q_o), len(q_a)]
        checks["object_track_ours_vs_asis_max"] = float(np.abs(o_o[:n] - o_a[:n]).max())
        if task == "push":
            ez = np.load(T["export"])
            checks["push_pair_vs_exported_m1p00_max"] = float(max(np.abs(ez["robot_qpos"] - q_o).max(), np.abs(ez["object_qpos"] - o_o).max()))
        # person contact flags (round-6 rule) and the frame
        if T["kind"] == "core4d":
            hum = ce.human_core4d(task)
            fl = hum["flags"][:, 0]  # [T15, 2 hands] person 1 = robot 0
            src_fps, to30 = 15, 2
        else:
            hum = ce.human_omomo(task)
            fl = hum["flags"]  # [T30, 2 hands]
            src_fps, to30 = 30, 1
        nsrc = min(len(fl), (n - 1) // to30 + 1)
        seg = {side: runs(fl[:nsrc, h]) for h, side in enumerate(("left", "right"))}
        longest = max(seg["right"], key=lambda r: (r[1] - r[0], -r[0]))
        t_src = (longest[0] + longest[1]) // 2
        t30 = t_src * to30
        sel = dict(rule="middle of the longest run of person contact frames of robot 0's right hand (round-6 contact rule)", runs_source_frames=seg,
                   right_hand_segment=longest, same_runs_both_hands=seg["left"] == seg["right"], source_frame=int(t_src), source_fps=src_fps,
                   reference_frame_30hz=int(t30), time_s=round(t30 / 30, 4))
        # per-frame metrics (round-6 code) at that frame, ours and as is, robot 0 both hands; clip means checked against step6
        metrics, mean_check = {}, {}
        if T["kind"] == "core4d":
            hflags, H = hm.human_core4d(task)
        else:
            hflags, H = hm.human_omomo(task)
        for ver, (q, o) in (("ours", (q_o, o_o)), ("omni_asis", (q_a, o_a))):
            P = cg.robot_palms_local(robot, q[[t30]], o[[t30]], offs)[0]  # [2 robots, 2 hands, 3]
            rd = rh.directions_local(q[[t30]], o[[t30]])
            mm = {}
            for h, side in enumerate(("left", "right")):
                if T["kind"] == "core4d":
                    tg = ce.S_BAR[task] * hum["target"][t_src, 0, h]
                    nm = hum["normal"][t_src, 0, h]
                    hpn = H["palm_normal"][t_src, 0, h]
                    flag = bool(hum["flags"][t_src, 0, h])
                else:
                    c = ds.offset_c(1.0)
                    tg = hum["target"][t_src, h] - c * np.eye(3)[ds.TABLE[task][1]]
                    nm = hum["normal"][t_src, h]
                    hpn = H["palm_normal"][t_src, h]
                    flag = bool(hum["flags"][t_src, h])
                e = P[0, h] - tg
                nc = float(e @ nm)
                mm[f"r0_{side}"] = dict(person_hand_in_contact=flag, contact_error_3d_cm=round(100 * float(np.linalg.norm(e)), 2),
                                        contact_error_normal_signed_cm=round(100 * nc, 2), contact_error_tangential_cm=round(100 * float(np.linalg.norm(e - nc * nm)), 2),
                                        palm_normal_angle_deg=round(float(hm._ang(rd["palm_normal"][0, 0, h], hpn)), 2))
            metrics[ver] = mm
            # clip means with the same functions vs the step-6 tables (robot 0 right hand)
            if T["kind"] == "core4d":
                res = ce.core4d(task, robot, offs, hum, q, o, 1.0)
                hres = hm.core4d(task, rh, (hflags, H), q, o, 1.0)
            else:
                res = ce.omomo(task, robot, offs, hum, q, o, 1.0)
                hres = hm.omomo(task, rh, (hflags, H), q, o, T["urdf"] if ver == "ours" else T["urdf"])
            lv = "k = 1.0" if T["kind"] == "core4d" else "m = 1.00"
            want_ce = [r for r in ce_rows if r["task"] == task and r["robot"] == "r0" and r["hand"] == "right" and
                       ((ver == "ours" and r["round"] == "round6C" and r["level"] == lv) or (ver == "omni_asis" and r["round"] == "comparisons" and r["level"] == "omni_asis"))]
            want_hm = [r for r in hm_rows if r["task"] == task and r["robot"] == "r0" and r["hand"] == "right" and
                       ((ver == "ours" and r["version"] == "round6C" and r["level"] == lv) or (ver == "omni_asis" and r["version"] == "omni_asis"))]
            mean_check[ver] = dict(contact_error_3d_mean_cm=res["r0_right"].get("dist_mean_cm"), step6_contact_error_3d_csv=[w["dist_mean_cm"] for w in want_ce],
                                   palm_normal_angle_mean_deg=hres["r0_right"]["palm_normal_angle"]["mean_deg"],
                                   step6_hand_metrics_csv=[w["palm_normal_angle_mean_deg"] for w in want_hm])
        checks["clip_means_vs_step6"] = mean_check
        # ---------------------------------------------------------------- scenes
        hs = hr.HumanScene(task, ROOT / "logs_system1/t15/human_meshes")
        tH = min(hs.frame(t30), len(hs.V[0]) - 1)
        Vp, Fp = hs.V[0][tH], hs.faces
        Ro, xo = hs.obj_R[tH], hs.obj_x[tH]
        Tob = np.eye(4)
        Tob[:3, :3], Tob[:3, 3] = Ro, xo
        if T["kind"] == "core4d":
            scale = float(np.load(T["ours"])["human_to_robot_scales"][0])
        else:
            scale = float(np.load(T["ours"])["smpl_scale"])
        palms_h, _ = hs.axes(0, t30)
        import trimesh
        cap = trimesh.load(str(hs.mesh), force="mesh")
        cap_c_local = (cap.bounds[0] + cap.bounds[1]) / 2
        out_dir = a.out / task
        out_dir.mkdir(parents=True, exist_ok=True)
        rs = ts.T15Scene(robots=2, object_urdf=T["urdf"])
        hsn = ts.T15Scene(robots=0, object_mesh=hs.mesh, persons=[(Vp, Fp)])
        hsn.set_ground(float(Vp[:, 2].min()))
        hsn.set_pose(obj_mesh_pose=Tob)
        # cameras
        az = yaw_az(q_o[t30, 0])
        box = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * CU_MARGIN
        pts, full_pts = [], []
        for q, o in ((q_o, o_o), (q_a, o_a)):
            rs.set_pose(q[t30], o[t30])
            Hh = np.array([rs.link(0, f"{s}_rubber_hand_link")[:3, 3] for s in ("left", "right")])
            pts.append((Hh[:, None] + box[None]).reshape(-1, 3))
            full_pts.append(rs.mesh_bounds(("robot0", "object")))
        cam_cu = ts.camera_for(np.concatenate(pts), ts.W, ts.H, az, CU_EL)
        cam_full = ts.camera_for(np.concatenate(full_pts), ts.W, ts.H, az + FULL_DAZ, FULL_EL)
        wm = hs.wrist_mid(0, t30)
        cam_cu_h = dict(lookat=wm.tolist(), distance=cam_cu["distance"] / scale, azimuth=cam_cu["azimuth"], elevation=cam_cu["elevation"])
        capw = np.asarray(cap.vertices) @ Ro.T + xo
        hb = np.concatenate([Vp, capw])
        cam_full_h = dict(lookat=((hb.min(0) + hb.max(0)) / 2).tolist(), distance=cam_full["distance"] / scale, azimuth=cam_full["azimuth"], elevation=cam_full["elevation"])
        inputs = {str(Path(p).relative_to(ROOT)): ts.sha(p) for p in (T["ours"], T["omni"], T["urdf"], ts.ROBOT_URDF)}
        inputs["captured_object_mesh"] = dict(path=str(hs.mesh), sha256=ts.sha(hs.mesh))
        base = dict(group="A", task=task, task_name=T["name"], selection=sel, human_to_robot_scale=scale, checks=checks, inputs_sha256=inputs,
                    code_sha256={str(Path(p).relative_to(ROOT)): ts.sha(p) for p in (Path(__file__), HERE / "t15_scene.py", ROOT / "system1/t14/human_render.py",
                                                                                     ROOT / "system1/t14/contact_error3d.py", ROOT / "system1/t14/hand_metrics.py")},
                    segmentation_classes=ts.CLASSES, git_head=head, command=" ".join([sys.executable, "-B", *sys.argv]))
        recs = {}
        for view, cam_r, cam_h in (("closeup", cam_cu, cam_cu_h), ("full", cam_full, cam_full_h)):
            # person
            stem = out_dir / f"A_{task}_{view}_human"
            out = hsn.render(cam_h, {"object", "person1"}, stem)
            cw, ow = Tob[:3, :3] @ cap_c_local + Tob[:3, 3], Tob[:3, 3]
            objtol = 0.5 * float(np.max(cap.extents)) + 0.05
            kp = dict(zip(("object_center", "object_origin", "person_left_palm", "person_right_palm"),
                          ts.T15Scene.project(out, [cw, ow, palms_h[0], palms_h[1]], ["object", "object", "person1", "person1"], [objtol, objtol, 0.04, 0.04])))
            rec = dict(base, asset_id=stem.name, image=stem.name + ".png", kind="person (SMPL-X, grey) + captured object", view=view, frame=dict(
                reference_frame_30hz=int(t30), time_s=round(t30 / 30, 4), source_frame=int(tH), source_fps=hs.fps), camera=out["camera"], keypoints=kp,
                subject_extent=out["subject_extent"], ground_z_m=hsn.ground_z, person="person 1 (robot 0's source)" if T["kind"] == "core4d" else "the demonstrator",
                hidden="the other person" if T["kind"] == "core4d" else None, gl=ts.gl_info())
            ts.write_json(stem.with_suffix(".json"), rec)
            recs[stem.name] = rec
            # person trajectory (captured object centre, whole clip)
            Pw = np.array([hs.obj_R[i] @ cap_c_local + hs.obj_x[i] for i in range(len(hs.obj_x))])
            traj_h = ts.T15Scene.project(out, Pw, ["object"] * len(Pw), [objtol] * len(Pw))
            ts.write_json(stem.with_name(stem.name + ".traj.json"), dict(asset_id=stem.name + ".traj", image=stem.name + ".png", camera=out["camera"],
                                                                         point="captured-object mesh bounding-box centre", source_fps=hs.fps,
                                                                         source_frame=list(range(len(Pw))), points=traj_h,
                                                                         visibility="each point against THIS image's depth / segmentation (the final frame), tolerance as key points"))
            # robots
            for ver, (q, o) in (("omni_asis", (q_a, o_a)), ("ours", (q_o, o_o))):
                stem = out_dir / f"A_{task}_{view}_{ver}"
                rs.set_pose(q[t30], o[t30])
                out = rs.render(cam_r, {"robot0", "object"}, stem)
                oc, oo = rs.object_center()
                ext = rs.mesh_bounds(("object",))
                objtol = 0.5 * float(np.max(ext.max(0) - ext.min(0))) + 0.05
                names, pts_, own, tol = [], [], [], []
                for r in (0, 1):
                    for side in ("left", "right"):
                        M = rs.link(r, f"{side}_rubber_hand_link")
                        names.append(f"robot{r}_{side}_palm")
                        pts_.append(M[:3, :3] @ offs[side] + M[:3, 3])
                        own.append(f"robot{r}")
                        tol.append(0.06)
                    names.append(f"robot{r}_pelvis")
                    pts_.append(rs.link(r, "pelvis")[:3, 3])
                    own.append(f"robot{r}")
                    tol.append(0.15)
                names += ["object_center", "object_origin"]
                pts_ += [oc, oo]
                own += ["object", "object"]
                tol += [objtol, objtol]
                kp = dict(zip(names, ts.T15Scene.project(out, pts_, own, tol)))
                for k in kp:
                    if k.startswith("robot1"):
                        kp[k]["drawn"] = False
                        kp[k]["occluded"] = True
                        kp[k]["occlusion_reason"] = "robot 1 not drawn (hidden in group A)"
                # FK check: scene (yourdfpy) palms vs the metric code (PhysX-URDF FK, object frame)
                Pl = cg.robot_palms_local(robot, q[[t30]], o[[t30]], offs)[0]
                Ob = rs.bases["object"]
                Pw_ = np.einsum("ij,rhj->rhi", Ob[:3, :3], Pl) + Ob[:3, 3]
                fk_diff = float(max(np.linalg.norm(np.array(kp[f"robot{r}_{s}_palm"]["world_xyz_m"]) - Pw_[r, h]) for r in (0, 1) for h, s in enumerate(("left", "right"))))
                rec = dict(base, asset_id=stem.name, image=stem.name + ".png", kind=dict(ours="T14 round 6 C, 1.0x", omni_asis="OmniRetarget as is, 1.0x")[ver], view=view,
                           frame=dict(reference_frame_30hz=int(t30), time_s=round(t30 / 30, 4)), camera=out["camera"], keypoints=kp, subject_extent=out["subject_extent"],
                           hidden="robot 1", metrics_at_frame=metrics[ver], metrics_definition="contact_error3d.py / hand_metrics.py per-frame values (round 6 rule) at this frame; "
                           "contact error = robot palm point - the person's palm target on the object (1.0x); palm-normal angle = robot vs person palm normal (object frame)",
                           palm_fk_scene_vs_metric_code_max_m=fk_diff, gl=ts.gl_info())
                ts.write_json(stem.with_suffix(".json"), rec)
                recs[stem.name] = rec
                if ver == "ours":
                    Pw = []
                    for i in range(len(o)):
                        rs.bases["object"] = ts.pose(o[i, :3], o[i, [4, 5, 6, 3]])
                        Pw.append(rs.object_center()[0])
                    rs.set_pose(q[t30], o[t30])
                    traj = ts.T15Scene.project(out, np.array(Pw), ["object"] * len(Pw), [objtol] * len(Pw))
                    ts.write_json(stem.with_name(stem.name + ".traj.json"), dict(asset_id=stem.name + ".traj", image=stem.name + ".png", camera=out["camera"],
                                                                                 point="object visual-mesh bounding-box centre (object URDF)", fps=30,
                                                                                 reference_frame=list(range(len(o))), time_s=(np.arange(len(o)) / 30).round(4).tolist(),
                                                                                 points=traj, same_track_as="omni_asis (object tracks equal, checks)",
                                                                                 visibility="each point against THIS image's depth / segmentation (the final frame), tolerance as key points"))
                print(task, view, ver, out["subject_extent"], flush=True)
        rs.close()
        hsn.close()
        hs.close()
        summary["tasks"][task] = dict(selection=sel, checks=checks, images=list(recs), cameras=dict(closeup=cam_cu, full=cam_full, closeup_person=cam_cu_h, full_person=cam_full_h),
                                      metrics_at_frame=metrics)
        print(task, json.dumps(sel), json.dumps(checks)[:600], flush=True)
    ts.write_json(a.out / "group_a_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
