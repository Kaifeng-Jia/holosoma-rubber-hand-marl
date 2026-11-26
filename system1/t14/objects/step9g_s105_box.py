#!/usr/bin/env python3
"""T14 step 9g item 3 context (report only): the partner's own box vs the partner person's pelvis around frame 105 of smallboxa
(Stage-1 inputs of the failing solve: own1_real = the mirrored real box, human1 = the partner person; solver scale, m = 1.0).
Rebuilt as step9_gates.py does. Writes logs_system1/t14/step9g_s105/box_vs_person_96_110.json. Read only.

    hsretargeting_s1/bin/python -B system1/t14/objects/step9g_s105_box.py
"""
import sys, json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
import os
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "system1/t14/objects")); sys.path.insert(0, str(ROOT / "system1/t14"))
import frontback as fb, omomo_motion as om, partner as pt
from step9_diag import load_role0
obj, seq, fr = 'smallbox', 'sub4_smallbox_047', (36, 172)
h0, names, nom, size, s = load_role0(obj, seq, fr)
walk = fb.walking_segment(om.load_sequences({obj})[seq], *fr)
C = fb.build(h0, names, pt.mj_to_mat(nom), size, 1.0, turn='world', legs='mirror', walk=walk)
B = C['own1_real']; H = C['human1']; ip = names.index('Pelvis')
ia = {n: names.index(n) for n in ('L_Hip', 'R_Hip', 'L_Knee', 'R_Knee')}
tau = C['tau']
out = []
for t in range(96, 111):
    c = B[t, :3, 3]; p = H[t, ip]
    row = dict(frame=t, tau=round(float(tau[t]), 2), box_step_m=round(float(np.linalg.norm(B[t, :3, 3] - B[t-1, :3, 3])), 4),
               pelvis_step_m=round(float(np.linalg.norm(H[t, ip] - H[t-1, ip])), 4),
               box_minus_pelvis_step_m=round(float(np.linalg.norm((B[t, :3, 3] - H[t, ip]) - (B[t-1, :3, 3] - H[t-1, ip]))), 4),
               box_rot_step_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(B[t, :3, :3] @ B[t-1, :3, :3].T) - 1) / 2, -1, 1)))), 3),
               box_center_minus_pelvis=np.round(c - p, 3).tolist(),
               R_hip_to_box_center_m=round(float(np.linalg.norm(H[t, ia['R_Hip']] - c)), 3))
    out.append(row); print(json.dumps(row))
json.dump(dict(rows=out, walk=walk, smpl_scale=s, box_size=np.round(size, 4).tolist()), open('logs_system1/t14/step9g_s105/box_vs_person_96_110.json', 'w'), indent=1)
