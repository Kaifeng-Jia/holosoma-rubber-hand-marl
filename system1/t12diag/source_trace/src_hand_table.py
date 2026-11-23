#!/usr/bin/env python3
"""Read-only: hand-to-table distances for each CORE4D person (source) and each pipeline stage.

Signed distance convention: positive = penetration into the object, negative = gap (m), as T12.
Object geometry: (a) desk001_m.obj real mesh (watertight, trimesh signed distance), (b) the 5-box proxy (exact union-of-boxes SDF).
"""
import json, sys
from pathlib import Path
import numpy as np
import trimesh

OUT = Path(__file__).resolve().parent
SEQ = Path('/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/human_object_motions/20231030/001')
MESH = Path('/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/object_models/desk/desk001_m.obj')
SMPLX = Path('/home/kevin/CS6758Project/data/smplx_models/smplx/SMPLX_NEUTRAL.npz')
RUN = Path('/home/kevin/datasets/CORE4D-V1-full/retarget_results/method_comparison_20260904/20231030_001_desk001_move')
RUNTIME = Path('/home/kevin/duet-system1/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz')
PERFRAME = Path('/home/kevin/duet-system1/results_system1/T12_diag/reference_hand_table_distance_per_frame.csv')
CANON = Path('/home/kevin/datasets/CORE4D-V1-full/derived/20231030_001_desk001_v1_canonical.npz')

BOX_C = np.array([[0, .3415, 0], [0, -.005, -.29], [0, -.005, .29], [0, -.34, -.29], [0, -.34, .29]])
BOX_H = np.array([[.5986, .047, .799], [.06, .655, .075], [.06, .655, .075], [.55, .05, .1], [.55, .05, .1]]) / 2


def box_sdf(p, scale=1.0):
    c, h = BOX_C * scale, BOX_H * scale
    d = np.abs(p[..., None, :] - c) - h
    out = np.linalg.norm(np.maximum(d, 0), axis=-1)
    ins = np.minimum(d.max(axis=-1), 0)
    return (-(out + ins)).max(axis=-1)


def stats(x, mask=None):
    x = np.asarray(x, float)
    if mask is not None:
        x = x[mask]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), p5_cm=round(100 * np.percentile(x, 5), 2), p25_cm=round(100 * np.percentile(x, 25), 2),
                p50_cm=round(100 * np.percentile(x, 50), 2), p75_cm=round(100 * np.percentile(x, 75), 2),
                p95_cm=round(100 * np.percentile(x, 95), 2), within_1p5cm=round(float((x >= -0.015).mean()), 4),
                within_0p5cm=round(float((x >= -0.005).mean()), 4), penetrating=round(float((x > 0).mean()), 4))


# ------------------------------------------------------------------ phases from the runtime (50 Hz) labels used in T12_diag
import csv
rows = list(csv.DictReader(PERFRAME.open()))
phase50 = np.array([r['phase'] for r in rows])
ref50 = {(rb, s): np.array([float(r[f'urdf_r{rb}_{s}_cm']) for r in rows]) / 100 for rb in (0, 1) for s in ('left', 'right')}
N50 = len(rows)
contact50 = np.isin(phase50, ['lift', 'push', 'carry', 'lower'])
print('runtime contact-phase frames', np.flatnonzero(contact50)[[0, -1]], contact50.sum())


def phase_mask_at_times(t):
    k = np.clip(np.rint(t * 50).astype(int), 0, N50 - 1)
    return contact50[k], k


# ------------------------------------------------------------------ source (raw CORE4D, 15 Hz, Y-up world)
W = np.load(SMPLX, allow_pickle=False)['weights']
lab = W.argmax(1)
HAND_SETS = dict(left=[20] + list(range(25, 40)), right=[21] + list(range(40, 55)))
hand_idx = {s: np.flatnonzero(np.isin(lab, v)) for s, v in HAND_SETS.items()}
print('hand vertices', {s: len(v) for s, v in hand_idx.items()})
TIP = dict(left=list(range(66, 71)), right=list(range(71, 76)))
WRIST = dict(left=20, right=21)

mesh = trimesh.load(MESH, force='mesh')
assert mesh.is_watertight
pq = trimesh.proximity.ProximityQuery(mesh)
To = np.load(SEQ / 'smooth_objposes.npy')
T = len(To)
people = [np.load(SEQ / f'person{i}_poses.npz', allow_pickle=True)['arr_0'].item() for i in (1, 2)]
canon = np.load(CANON, allow_pickle=False)
ground = []
for i in range(2):
    g = float(canon['human_joints'][:, i, [10, 11], 2].min())
    ground.append(g - 0.1 if g >= 0.1 else g)  # core4d_retarget.py:172-174
heights = canon['human_heights']
scales = 1.32 / heights
print('ground offsets', ground, 'scales', scales)


def to_local(p, t):
    R, x = To[t, :3, :3], To[t, :3, 3]
    return (p - x) @ R


def mesh_sd(pl, prefilter=0.06):
    """trimesh signed distance (positive inside) for points within prefilter of the box hull; others = box sdf (far)."""
    sd = box_sdf(pl)
    near = sd > -prefilter
    if near.any():
        sd = sd.copy()
        sd[near] = pq.signed_distance(pl[near])
    return sd


src_t = np.arange(T) / 15.0
cmask, k_of_src = phase_mask_at_times(src_t)
res = dict(source=dict(), stage1_input=dict())
per_frame = []
for i, P in enumerate(people):
    V, J = P['vertices'], P['joints']
    out = {}
    out_in = {}
    for s in ('left', 'right'):
        box_d = np.empty(T); mesh_d = np.empty(T); wr = np.empty(T); tip = np.empty(T)
        box_d_s1 = np.empty(T); mesh_d_s1 = np.empty(T)
        loc_y = np.empty(T)
        for t in range(T):
            pl = to_local(V[t, hand_idx[s]], t)
            bd = box_sdf(pl)
            box_d[t] = bd.max()
            md = mesh_sd(pl)
            mesh_d[t] = md.max()
            loc_y[t] = pl[md.argmax(), 1]
            wr[t] = box_sdf(to_local(J[t, WRIST[s]][None], t)).max()
            tip[t] = box_sdf(to_local(J[t, TIP[s]], t)).max()
            # Stage-1 input: human shifted by -ground (Z-up z == CORE4D +y), scaled about the anchor with the person's
            # scale; nominal object scaled about the same anchor and NOT shifted (core4d_retarget.py:172-191).
            # In the scaled object's local frame this equals scale * sdf(R^T (p - g*e_y - x)).
            ps = V[t, hand_idx[s]] - np.array([0, ground[i], 0])
            pls = to_local(ps, t)
            box_d_s1[t] = scales[i] * box_sdf(pls).max()
            mesh_d_s1[t] = scales[i] * mesh_sd(pls).max()
        out[s] = dict(box=box_d, mesh=mesh_d, wrist_joint_box=wr, fingertip_joint_box=tip, closest_local_y=loc_y)
        out_in[s] = dict(box=box_d_s1, mesh=mesh_d_s1)
    res['source'][i] = out
    res['stage1_input'][i] = out_in

# ------------------------------------------------------------------ robot stages
sys.path.insert(0, '/home/kevin/duet-system1')
from system1.t12v2 import contact_stats as cs  # read-only import (python -B)
robot = cs.Robot('urdf')
rt = np.load(RUNTIME, allow_pickle=False)
assert list(robot.dofs) == list(rt['joint_names']), 'dof order'
HL = {'left': 'left_rubber_hand_link', 'right': 'right_rubber_hand_link'}


def robot_hand_sd(root_pos, root_quat_wxyz, dof, obj_pos, obj_quat_wxyz, scale):
    """(F, 2 sides) max signed distance of the rubber-hand collision vertices to the scaled 5-box proxy."""
    Tl = robot.fk(root_pos, root_quat_wxyz[..., [1, 2, 3, 0]], dof)
    To_inv = np.linalg.inv(cs.pose_mat(obj_pos, obj_quat_wxyz[..., [1, 2, 3, 0]]))
    out = []
    for s in ('left', 'right'):
        M = To_inv @ Tl[HL[s]]
        Pp = robot.points[HL[s]]
        pts = np.einsum('fij,pj->fpi', M[:, :3, :3], Pp) + M[:, None, :3, 3]
        out.append(box_sdf(pts, scale).max(axis=1))
    return np.stack(out, 1)


rob = {}
# Stage 1 nominal (per person, own scale, own nominal object), 30 Hz
for i in range(2):
    q = np.load(RUN / f'two_stage_fullnominal/person{i+1}/person{i+1}_nominal_scaled.npz', allow_pickle=False)['qpos']
    rob[('stage1_nominal', i)] = robot_hand_sd(q[:, :3], q[:, 3:7], q[:, 7:36], q[:, 36:39], q[:, 39:43], scales[i])
    q2 = np.load(RUN / f'two_stage_fullnominal/person{i+1}/person{i+1}_fixed_object.npz', allow_pickle=False)['qpos']
    rob[('stage2_physical_unused', i)] = robot_hand_sd(q2[:, :3], q2[:, 3:7], q2[:, 7:36], q2[:, 36:39], q2[:, 39:43], 1.0)
pre = np.load(RUN / 'small_table_nominal_preview/core4d_pair_reference.npz', allow_pickle=False)
sh = float(pre['shared_object_scale'])
oq = pre['object_qpos']
for i in range(2):
    q = pre['robot_qpos'][:, i]
    rob[('shared_preview', i)] = robot_hand_sd(q[:, :3], q[:, 3:7], q[:, 7:36], oq[:, :3], oq[:, 3:7], sh)
    # same robot pose, but the person's own nominal object (to isolate the merge step)
# runtime 50 Hz
bn = list(rt['body_names']); pel = bn.index('pelvis')
for i in range(2):
    rob[('runtime50', i)] = robot_hand_sd(rt['agent_body_pos_w'][:, i, pel], rt['agent_body_quat_w'][:, i, pel], rt['agent_joint_pos'][:, i],
                                          rt['object_pos_w'], rt['object_quat_w'], sh)
# check vs T12_diag per-frame
for i in range(2):
    for j, s in enumerate(('left', 'right')):
        a = rob[('runtime50', i)][:, j]; b = ref50[(i, s)]
        ok = b > -0.99
        print('runtime check r', i, s, 'max abs diff (non-culled)', np.abs(a[ok] - b[ok]).max() if ok.any() else None,
              'within1.5 mine', (a[1:] >= -0.015).mean(), 'csv', (b[1:] >= -0.015).mean())

# merge-step displacement per person: preview object vs person nominal object
for i in range(2):
    qn = np.load(RUN / f'two_stage_fullnominal/person{i+1}/person{i+1}_nominal_scaled.npz', allow_pickle=False)['qpos']
    d = oq[:, :3] - qn[:, 36:39]
    print(f'merge shift person{i+1}: mean xyz cm', np.round(100 * d.mean(0), 2), 'max |d| cm', round(100 * np.linalg.norm(d, axis=1).max(), 2),
          'robot qpos identical', np.array_equal(pre['robot_qpos'][:, i], qn[:, :36]))

# ------------------------------------------------------------------ summarise
summary = dict(meta=dict(hand_vertices={s: int(len(v)) for s, v in hand_idx.items()}, ground_offsets_m=ground, scales=scales.tolist(),
                         shared_scale=sh, source_frames=T, contact_phase_runtime_frames=[int(np.flatnonzero(contact50)[0]), int(np.flatnonzero(contact50)[-1])],
                         contact_phase_source_frames=[int(np.flatnonzero(cmask)[0]), int(np.flatnonzero(cmask)[-1])]))


def closest(dct, keyname):
    return np.maximum(dct['left'][keyname], dct['right'][keyname])


for i in range(2):
    S = res['source'][i]; S1 = res['stage1_input'][i]
    blk = {}
    for name, arr in [('source_mesh_closest_hand', closest(S, 'mesh')), ('source_box_closest_hand', closest(S, 'box')),
                      ('source_mesh_left', S['left']['mesh']), ('source_mesh_right', S['right']['mesh']),
                      ('source_box_left', S['left']['box']), ('source_box_right', S['right']['box']),
                      ('source_wrist_joint_box_left', S['left']['wrist_joint_box']), ('source_wrist_joint_box_right', S['right']['wrist_joint_box']),
                      ('source_fingertip_box_left', S['left']['fingertip_joint_box']), ('source_fingertip_box_right', S['right']['fingertip_joint_box']),
                      ('stage1input_mesh_closest_hand', closest(S1, 'mesh')), ('stage1input_box_closest_hand', closest(S1, 'box')),
                      ('stage1input_mesh_left', S1['left']['mesh']), ('stage1input_mesh_right', S1['right']['mesh'])]:
        blk[name] = dict(all=stats(arr), contact_phase=stats(arr, cmask), still=stats(arr, ~cmask))
    # where the closest hand vertex sits relative to the tabletop (local y; tabletop spans 0.318..0.365)
    for s in ('left', 'right'):
        y = S[s]['closest_local_y'][cmask]
        blk[f'closest_point_local_y_{s}_contact_phase'] = dict(p50=round(float(np.median(y)), 4),
                                                                below_top_underside=round(float((y < 0.318).mean()), 3),
                                                                within_top_thickness=round(float(((y >= 0.318) & (y <= 0.365)).mean()), 3),
                                                                above_top=round(float((y > 0.365).mean()), 3))
    summary[f'person{i+1}'] = blk

t30 = np.arange(413) / 30.0
c30, _ = phase_mask_at_times(t30)
for (stage, i), arr in rob.items():
    m = c30 if arr.shape[0] == 413 else contact50
    cl = arr.max(1)
    summary.setdefault(f'robot_{stage}', {})[f'robot{i}'] = dict(
        closest_hand=dict(all=stats(cl[1:] if stage == 'runtime50' else cl), contact_phase=stats(cl, m), still=stats(cl, ~m)),
        left=dict(all=stats(arr[:, 0]), contact_phase=stats(arr[:, 0], m)), right=dict(all=stats(arr[:, 1]), contact_phase=stats(arr[:, 1], m)))

(OUT / 'hand_table_summary.json').write_text(json.dumps(summary, indent=1))
# per-frame CSV on the 15 Hz source grid
with (OUT / 'hand_table_per_source_frame.csv').open('w') as f:
    w = csv.writer(f)
    hdr = ['src_frame', 't_s', 'runtime_frame', 'runtime_phase']
    for i in range(2):
        for s in ('left', 'right'):
            hdr += [f'p{i+1}_{s}_mesh_cm', f'p{i+1}_{s}_box_cm', f'p{i+1}_{s}_stage1input_mesh_cm']
    for i in range(2):
        for s in ('left', 'right'):
            hdr += [f'r{i}_{s}_runtime_cm']
    w.writerow(hdr)
    for t in range(T):
        k = int(k_of_src[t])
        row = [t, round(t / 15, 4), k, phase50[k]]
        for i in range(2):
            for s in ('left', 'right'):
                row += [round(100 * res['source'][i][s]['mesh'][t], 2), round(100 * res['source'][i][s]['box'][t], 2),
                        round(100 * res['stage1_input'][i][s]['mesh'][t], 2)]
        for i in range(2):
            for j, s in enumerate(('left', 'right')):
                row += [round(100 * rob[('runtime50', i)][k, j], 2)]
        w.writerow(row)
print(json.dumps(summary, indent=1))
