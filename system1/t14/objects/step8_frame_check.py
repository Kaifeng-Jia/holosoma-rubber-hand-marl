#!/usr/bin/env python3
"""T14 step 9c finding: which step-8 role solves had the person-side interaction-mesh frame rotated against the robot-side
(shared target) frame? Read only.

role_solve.py (step 8) gives the Stage-2 interaction mesh the role's own object as the person-side frame (role 0: O; role 1:
S O F) and the shared target P as the robot-side frame. At frame 0 the own frame is P M_r F (F = the table mirror of an improper
role 1, else identity), so the two frames differ by the linear part Q_r F of the role's region map. The mesh's Laplacian
coordinates are translation-invariant but not rotation-invariant: when Q_r F != I the person's relation to its own object is
imposed, unrotated, on the robot's relation to its target region. Per solved step-8 candidate (level 1.0) and role: whether
Q_r F = I, the rotation angle of the proper part of Q_r, and the mean horizontal distance robot pelvis - the role's person pelvis
(information; on the table family the robot is moved to the training table, so that distance is not a clean signal there).

    sonic_s1/bin/python -B system1/t14/objects/step8_frame_check.py --out results_system1/T14_retargeting/step9c/step8_frame_check.csv
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for ef in sorted(glob.glob(str(ROOT / "logs_system1/t14/step8/cands/*/c*/m1p00/eval.json"))):
        d = Path(ef).parent
        e = json.loads(Path(ef).read_text())
        row = dict(sequence=d.parts[-3], candidate=d.parts[-2], step8_applicable_old_window=e["applicable"], step8_failed=" | ".join(e["failed"]))
        for r in (0, 1):
            rec = json.loads((d / f"r{r}/run_record.json").read_text())
            Q = np.asarray(rec["candidate"]["Q0" if r == 0 else "Q1"], float)
            F = np.diag([1.0, 1.0, -1.0]) if (bool(rec.get("improper")) and r == 1) else np.eye(3)
            Qp = Q * np.sign(np.linalg.det(Q))
            ang = float(np.degrees(np.arccos(np.clip((np.trace(Qp) - 1) / 2, -1, 1))))
            z = np.load(d / f"r{r}/{rec['tag']}_new_qpos.npz")
            dist = np.linalg.norm(z["qpos"][:, :2] - z["human_joints"][:, 0, :2], axis=1)
            row.update({f"r{r}_frames_differ_by_translation_only": bool(np.allclose(Q @ F, np.eye(3), atol=1e-6)), f"r{r}_Q_rotation_deg": round(ang, 1),
                        f"r{r}_Q_det": int(np.sign(np.linalg.det(Q))), f"r{r}_robot_person_pelvis_mean_m": round(float(dist.mean()), 3)})
        row["affected"] = not (row["r0_frames_differ_by_translation_only"] and row["r1_frames_differ_by_translation_only"])
        rows.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for r in rows:
        s = summ.setdefault(r["sequence"], dict(solved=0, affected=0, unaffected=[]))
        s["solved"] += 1
        s["affected"] += r["affected"]
        if not r["affected"]:
            s["unaffected"].append(r["candidate"])
    a.out.with_suffix(".json").write_text(json.dumps(dict(rules=__doc__, summary=summ, solved=len(rows), affected=sum(r["affected"] for r in rows)), indent=1) + "\n")
    print(json.dumps(summ), sum(r["affected"] for r in rows), "/", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
