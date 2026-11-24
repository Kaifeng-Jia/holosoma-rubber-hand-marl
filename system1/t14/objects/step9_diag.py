#!/usr/bin/env python3
"""T14 step 9: build the front-back construction (frontback.py) for a class-a clip without solving and report its
diagnostics: walking axis, angle to the box axis, reversed-leg pelvis vs object-attached pelvis mismatch, partner pelvis net
displacement, shared object rotation vs the demonstration's.

    hsretargeting_s1/bin/python -B system1/t14/objects/step9_diag.py --object smallbox --sequence sub17_smallbox_041 --frames 29:151 --out <json>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RT = ROOT / "src/holosoma_retargeting/holosoma_retargeting"
for p in (ROOT / "src/holosoma_retargeting", ROOT / "scripts", HERE.parent, HERE):
    sys.path.insert(0, str(p))


def load_role0(name, seq, frames):
    """solver-world role 0 data of a box-family clip (as role_solve.py): human [T, 54, 3], names, own nominal poses (mj), box size x s, s."""
    import box_assets as ba
    import omomo_motion as om
    from single_shared import OMOMO_NEW, omomo_palms
    from types import SimpleNamespace
    v = om.load_sequences({name})[seq]
    palms, fit = omomo_palms(seq)
    cwd = os.getcwd()
    os.chdir(RT)
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.examples.robot_retarget import convert_object_poses_to_mujoco_order, create_grounded_nominal_object_poses, create_task_constants, load_motion_data
    from holosoma_retargeting.src.utils import preprocess_motion_data
    mc = MotionDataConfig(data_format="smplh", robot_type="g1")
    constants = create_task_constants(RobotConfig(robot_type="g1"), mc, TaskConfig(object_name=name), "object_interaction")
    human, _, s = load_motion_data("object_interaction", "smplh", OMOMO_NEW, seq, constants, mc)
    os.chdir(cwd)
    bf = ba.box_frame(name, v)
    raw, _ = ba.poses_omomo_new(v, bf, fit)
    a, b = frames
    human, palms, raw = human[a:b], palms[a:b], raw[a:b]
    names = list(constants.DEMO_JOINTS) + ["L_Palm", "R_Palm"]
    hp = np.concatenate([human, palms], 1)
    h0, p0, _ = preprocess_motion_data(hp, SimpleNamespace(demo_joints=names), mc.toe_names, scale=float(s), object_poses=raw)
    nom = convert_object_poses_to_mujoco_order(create_grounded_nominal_object_poses(p0, float(s)))
    return h0, names, nom, bf["extents_m"] * float(s), float(s)


def main():
    import frontback as fb
    import partner as pt
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fr = tuple(int(x) for x in a.frames.split(":"))
    h0, names, nom, size, s = load_role0(a.object, a.sequence, fr)
    O = pt.mj_to_mat(nom)
    res = {}
    for m in (1.0, 1.5):
        C = fb.build(h0, names, O, size, m)
        pel1 = C["human1"][:, names.index("Pelvis")]
        pel0 = h0[:, names.index("Pelvis")]
        tilt = [float(np.degrees(np.arccos(np.clip(((C["P"][t, :3, :3] @ np.linalg.inv(C["A"][t, :3, :3]))[2, 2]), -1, 1)))) for t in range(len(O))]
        res[f"{m:.2f}"] = dict(axis=C["k"], sigma=C["sigma"], walk_axis_angle_deg=round(C["walk_axis_angle_deg"], 2), L_m=round(C["Lm"], 4), **fb.diag(C["D"], s),
                               pelvis_net_role0_m=round(float(np.linalg.norm(pel0[-1, :2] - pel0[0, :2])), 4), pelvis_net_role1_m=round(float(np.linalg.norm(pel1[-1, :2] - pel1[0, :2])), 4),
                               pelvis_net_role0_person_scale_m=round(float(np.linalg.norm(pel0[-1, :2] - pel0[0, :2])) / s, 4),
                               shared_vs_demo_rotation_max_deg=round(max(tilt), 2), partner_initial_pelvis_distance_m=round(float(np.linalg.norm(pel1[0, :2] - pel0[0, :2])), 4))
    out = dict(object=a.object, sequence=a.sequence, frames=list(fr), smpl_scale=s, box_size_solver_m=np.round(size, 4).tolist(), levels=res, rules=fb.__doc__)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rules"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
