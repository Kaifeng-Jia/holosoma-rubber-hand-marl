import sys, csv, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv=['x']
RUN = Path('/home/kevin/datasets/CORE4D-V1-full/retarget_results/method_comparison_20260904/20231030_001_desk001_move')
sys.path.insert(0, '/home/kevin/duet-system1')
from system1.t12v2 import contact_stats as cs
BOX_C = np.array([[0, .3415, 0], [0, -.005, -.29], [0, -.005, .29], [0, -.34, -.29], [0, -.34, .29]])
BOX_H = np.array([[.5986, .047, .799], [.06, .655, .075], [.06, .655, .075], [.55, .05, .1], [.55, .05, .1]]) / 2
def box_sdf(p, scale):
    c, h = BOX_C * scale, BOX_H * scale
    d = np.abs(p[..., None, :] - c) - h
    return (-(np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(axis=-1), 0))).max(axis=-1)
robot = cs.Robot('urdf')
HL = ('left_rubber_hand_link', 'right_rubber_hand_link')
def hsd(q, op, oq, scale):
    Tl = robot.fk(q[:, :3], q[:, 3:7][:, [1, 2, 3, 0]], q[:, 7:36])
    Ti = np.linalg.inv(cs.pose_mat(op, oq[:, [1, 2, 3, 0]]))
    res = []
    for h in HL:
        M = Ti @ Tl[h]; P = robot.points[h]
        res.append(box_sdf(np.einsum('fij,pj->fpi', M[:, :3, :3], P) + M[:, None, :3, 3], scale).max(1))
    return np.stack(res, 1)
rows = list(csv.DictReader(open('/home/kevin/duet-system1/results_system1/T12_diag/reference_hand_table_distance_per_frame.csv')))
ph = np.array([r['phase'] for r in rows]); c50 = np.isin(ph, ['lift', 'push', 'carry', 'lower'])
k = np.clip(np.rint(np.arange(413) / 30 * 50).astype(int), 0, len(rows) - 1); m = c50[k]
pre = np.load(RUN / 'small_table_nominal_preview/core4d_pair_reference.npz')
sh = float(pre['shared_object_scale']); sc = pre['human_to_robot_scales']; oq = pre['object_qpos']
for i in range(2):
    qn = np.load(RUN / f'two_stage_fullnominal/person{i+1}/person{i+1}_nominal_scaled.npz')['qpos']
    q = qn[:, :36]
    for name, op, oo, s in [('own pose, own scale (Stage 1)', qn[:, 36:39], qn[:, 39:43], sc[i]),
                            ('own pose, shared scale', qn[:, 36:39], qn[:, 39:43], sh),
                            ('shared pose, own scale', oq[:, :3], oq[:, 3:7], sc[i]),
                            ('shared pose, shared scale (preview)', oq[:, :3], oq[:, 3:7], sh)]:
        d = hsd(q, op, oo, s)[m]
        cl = d.max(1)
        print(f'robot{i} {name:38s} contact-phase median cm: closest {100*np.median(cl):6.2f} left {100*np.median(d[:,0]):6.2f} right {100*np.median(d[:,1]):6.2f} | within1.5 closest {np.mean(cl>=-.015):.3f} left {np.mean(d[:,0]>=-.015):.3f} | pen closest {np.mean(cl>0):.3f}')
    dd = oq[:, :3] - qn[:, 36:39]
    print(f'   object shift preview-own (contact phase) mean cm', np.round(100 * dd[m].mean(0), 2), '; own quat == preview quat:', np.allclose(qn[:, 39:43], oq[:, 3:7]))
