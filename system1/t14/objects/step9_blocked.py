#!/usr/bin/env python3
"""T14 step 9: record of why the front-back construction is not built for the three class-a typical clips (they turn while
carrying). Per clip, every 5th frame of the carry episode: person's heading vs the walking direction (screen definitions), box
yaw relative to the first frame, and the gap between the reversed-leg pelvis and the object-attached pelvis of frontback.build
(1.0 x, solver world, / s = person scale); plus a key-frame sheet (person and box, 6 frames, top-oblique view).

    (render env) sonic_s1/bin/python -B system1/t14/objects/step9_blocked.py --out results_system1/T14_retargeting/step9
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "src/holosoma_retargeting", ROOT / "scripts", HERE.parent, HERE):
    sys.path.insert(0, str(p))
CLIPS = (("smallbox", "sub17_smallbox_041", 29, 151), ("largebox", "sub7_largebox_046", 33, 149), ("suitcase", "sub1_suitcase_019", 45, 161))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--diag", type=Path, required=True, help="dir with step9_diag json files (frontback gap per clip)")
    a = ap.parse_args()
    import omomo_motion as om
    from PIL import Image
    import render_screen as rs
    rec, rows = {}, []
    seqs = om.load_sequences({c[0] for c in CLIPS})
    for name, seq, s0, s1 in CLIPS:
        v = seqs[seq]
        J = om.smplx_joints(v)[s0:s1]
        hd = om.heading(J)[:, :2]
        wd = J[-1, 0, :2] - J[0, 0, :2]
        wd = wd / np.linalg.norm(wd)
        ang = (np.degrees(np.arctan2(hd[:, 1], hd[:, 0]) - np.arctan2(wd[1], wd[0])) + 180) % 360 - 180
        R = np.asarray(v["obj_rot"], float)[s0:s1]
        yaw = np.degrees(np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0])))
        d = json.loads((a.diag / f"{name}.json").read_text())["levels"]["1.00"]
        rec[name] = dict(sequence=seq, frames=[s0, s1], heading_vs_walk_deg_every5=np.round(ang[::5], 1).tolist(), box_yaw_rel_deg_every5=np.round(yaw[::5] - yaw[0], 1).tolist(),
                         heading_at_lift_deg=round(float(ang[0]), 1), heading_max_abs_deg=round(float(np.abs(ang).max()), 1), box_yaw_change_deg=round(float(yaw[-1] - yaw[0]), 1),
                         frontback_1p0=d)
        # key frames: 6 frames over the episode, look-at = the person's pelvis at the middle frame, high oblique view
        sq = rs.Seq(v, name)
        frames = [int(round(s0 + f * (s1 - 1 - s0))) for f in np.linspace(0, 1, 6)]
        look = sq.J[(s0 + s1) // 2, 0].copy()
        az = float(np.degrees(np.arctan2(wd[1], wd[0]))) + 90 + 30
        tiles = []
        with tempfile.TemporaryDirectory(dir=ROOT / "logs_system1/t14") as tmp:
            for t in frames:
                img = rs.render_frame(sq, t, look, az, 3.2, 360, 300, tmp)
                tiles.append(rs.caption(img, [f"{seq} 帧 {t}", f"朝向-行走方向 {ang[t - s0]:+.0f}°，箱偏航 {yaw[t - s0] - yaw[0]:+.0f}°"], 12))
        rows.append(np.concatenate(tiles, 1))
    sheet = np.concatenate(rows, 0)
    a.out.mkdir(parents=True, exist_ok=True)
    q = rs.save_jpg(sheet, a.out / "class_a_turns_keyframes.jpg")
    rec["sheet"] = dict(file="class_a_turns_keyframes.jpg", jpeg_quality=q)
    (a.out / "frontback_blocked.json").write_text(json.dumps(dict(rules=__doc__, clips=rec), indent=1, ensure_ascii=False) + "\n")
    for k, v in rec.items():
        if k != "sheet":
            print(k, v["heading_at_lift_deg"], v["heading_max_abs_deg"], v["box_yaw_change_deg"], v["frontback_1p0"]["pelvis_mismatch_max_m"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
