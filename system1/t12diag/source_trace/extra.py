import numpy as np, json
from pathlib import Path
SEQ = Path('/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/human_object_motions/20231030/001')
RUN = Path('/home/kevin/datasets/CORE4D-V1-full/retarget_results/method_comparison_20260904/20231030_001_desk001_move')
RT = Path('/home/kevin/duet-system1/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz')
BOX_C = np.array([[0, .3415, 0], [0, -.005, -.29], [0, -.005, .29], [0, -.34, -.29], [0, -.34, .29]])
BOX_H = np.array([[.5986, .047, .799], [.06, .655, .075], [.06, .655, .075], [.55, .05, .1], [.55, .05, .1]]) / 2
def box_sdf(p):
    d = np.abs(p[..., None, :] - BOX_C) - BOX_H
    return (-(np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(axis=-1), 0))).max(axis=-1)
To = np.load(SEQ / 'smooth_objposes.npy')
P = [np.load(SEQ / f'person{i}_poses.npz', allow_pickle=True)['arr_0'].item() for i in (1, 2)]
c = np.load('/home/kevin/datasets/CORE4D-V1-full/derived/20231030_001_desk001_v1_canonical.npz')
g = [float(c['human_joints'][:, i, [10, 11], 2].min()) for i in range(2)]
sc = 1.32 / c['human_heights']
cm = np.zeros(207, bool); cm[57:152] = True
loc = lambda p, t: (p - To[t, :3, 3]) @ To[t, :3, :3]
for i in range(2):
    J = P[i]['joints']
    pel = np.array([loc(J[t, 0], t) for t in range(207)])
    print(f'person{i+1} pelvis in object frame (x, y_up, z) contact-phase mean m', np.round(pel[cm].mean(0), 3))
    for s, w in (('left', 20), ('right', 21)):
        wl = np.array([loc(J[t, w], t) for t in range(207)])
        ws = np.array([loc(J[t, w] - np.array([0, g[i], 0]), t) for t in range(207)])
        d0 = box_sdf(wl); d1 = sc[i] * box_sdf(ws)
        print(f'   {s} wrist joint: local mean (contact) {np.round(wl[cm].mean(0),3)}; box sdf median cm source {100*np.median(d0[cm]):.2f} -> stage1 input (shift+scale) {100*np.median(d1[cm]):.2f}')
# chain check: runtime agent pelvis vs preview robot_qpos at t=0 and t=1.0 s
rt = np.load(RT); pre = np.load(RUN / 'small_table_nominal_preview/core4d_pair_reference.npz')
pel = list(rt['body_names']).index('pelvis')
for k30, k50 in ((0, 0), (30, 50), (300, 500)):
    for a in range(2):
        print(f'frame30 {k30}/frame50 {k50} agent{a}: |runtime pelvis - preview robot_qpos[:, {a}, :3]| = {np.linalg.norm(rt["agent_body_pos_w"][k50, a, pel] - pre["robot_qpos"][k30, a, :3]):.2e} m; joints max diff {np.abs(rt["agent_joint_pos"][k50, a] - pre["robot_qpos"][k30, a, 7:]).max():.2e}')
q1 = np.load(RUN / 'two_stage_fullnominal/person1/person1_nominal_scaled.npz')['qpos']
print('person1 nominal qpos == preview robot_qpos[:,0]:', np.array_equal(q1[:, :36], pre['robot_qpos'][:, 0]))
print('pair human_heights', pre['human_heights'], 'scales', pre['human_to_robot_scales'], 'shared', pre['shared_object_scale'])
print('provenance', str(pre['provenance_json']))
