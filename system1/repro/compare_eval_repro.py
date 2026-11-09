#!/usr/bin/env python3
"""Episode-by-episode comparison of a re-run evaluation against the V3 closeout table.

Usage (run with the hssim_s1 interpreter from the repository root):

    python system1/repro/compare_eval_repro.py \
        --new-dir logs_system1/T01/repro_pull_seed721_pm2 \
        --orig-dir <original evaluation directory with *.trajectory.npz> \
        --v3-episodes <V3 results/episodes.csv> \
        --case pull --seed 721 --variant Full --level 2 \
        --out-csv results_system1/T01/repro_pull_seed721_pm2_compare.csv \
        --out-json results_system1/T01/repro_pull_seed721_pm2_compare.json

Tolerances are fixed here, before any result is inspected:
  * completed_reference, termination_reason, executed_steps: exact equality;
  * progress and every continuous metric: |new - v3| <= REL_TOL * max(1, |v3|).
Hand error is recomputed for the new run with the same FK and reference
alignment that produced the V3 column (finalize_v3_results_20261104.handmetrics).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REL_TOL = 1.0e-5
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a: float, b: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= REL_TOL * max(1.0, abs(b))


def fnum(x):
    if x is None or x == "":
        return None
    return float(x)


def hand_errors(case: str, summary: dict, new_dir: Path) -> dict:
    """Same computation as V3 finalize handmetrics (non-variant, non-push path)."""
    import torch
    from prepare_experiments_r2_cpu import BODY, HANDS, BatchFK, reference

    torch.set_num_threads(2)
    fk = BatchFK()
    ref, _ = reference(case, BODY + HANDS, spacing=0.8)
    ref = {k: v.numpy() for k, v in ref.sample(torch.arange(ref.num_frames)).items()}
    names = BODY + HANDS
    hi = [names.index(x) for x in HANDS]
    rows = {}
    for ep in summary["episodes"]:
        if ep["kind"] != "perturbed" or not ep["completed_reference"]:
            continue
        path = new_dir / Path(ep["trajectory"]).name
        with np.load(path, allow_pickle=False) as z:
            f = z["reference_frame"].astype(int)
            offset = z["reference_object_pos_w"] - ref["object_pos_w"][f]
            assert np.max(abs(offset - offset[0])) < 3e-6, "reference alignment"
            expected = ref["agent_body_pos_w"][f][:, :, hi, :] + offset[:, None, None, :]
            actual = fk.poses(z["root_pos"], z["root_quat_xyzw"], z["dof_pos"], HANDS)[..., :3, 3]
            err = np.linalg.norm(actual - expected, axis=-1) * 100
            rows[ep["state_id"]] = float(err.mean())
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-dir", type=Path, required=True)
    p.add_argument("--orig-dir", type=Path, required=True)
    p.add_argument("--v3-episodes", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", default="Full")
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    a = p.parse_args()

    summary = json.loads((a.new_dir / "summary.json").read_text())
    new = {e["state_id"]: e for e in summary["episodes"]}
    with a.v3_episodes.open() as fh:
        v3 = {
            r["state_id"]: r
            for r in csv.DictReader(fh)
            if r["task"] == a.case
            and r["variant"] == a.variant
            and int(r["training_seed"]) == a.seed
            and int(float(r["perturbation_level_cm"])) == a.level
        }
    if set(new) != set(v3):
        raise SystemExit(f"state_id sets differ: new-only={set(new)-set(v3)} v3-only={set(v3)-set(new)}")

    hands = hand_errors(a.case, summary, a.new_dir)
    traj_keys = ("root_pos", "root_quat_xyzw", "dof_pos", "dof_vel", "object_pos_w", "object_quat_xyzw")
    rows, n_fail = [], 0
    max_abs = {}
    for sid in sorted(v3):
        e, r = new[sid], v3[sid]
        term_new = json.dumps(e["terminal_terms"], sort_keys=True)
        term_v3 = json.dumps(json.loads(r["termination_reason"]), sort_keys=True)
        hand_new = hands.get(sid)
        metrics = {
            "progress": (e["progress"], fnum(r["progress"])),
            "object_pos_rmse_cm": (e["object_rmse_3d_m"] * 100, fnum(r["object_pos_rmse_cm"])),
            "object_xy_rmse_cm": (e["object_rmse_xy_m"] * 100, fnum(r["object_xy_rmse_cm"])),
            "height_rmse_cm": (e["height_rmse_m"] * 100, fnum(r["height_rmse_cm"])),
            "raw_reward_sum_common": (e["reward_sum_common"], fnum(r["raw_reward_sum_common"])),
            "hand_pos_error_cm": (hand_new, fnum(r["hand_pos_error_cm"])),
        }
        ok_exact = (
            bool(e["completed_reference"]) == (r["completed_reference"] == "True")
            and term_new == term_v3
            and int(e["steps"]) == int(r["executed_steps"])
        )
        ok_float = all(close(n, o) for n, o in metrics.values())
        # Frame-by-frame trajectory difference against the original evaluation output.
        orig = a.orig_dir / f"{sid}.trajectory.npz"
        traj_diff = None
        if orig.exists():
            with np.load(a.new_dir / f"{sid}.trajectory.npz") as zn, np.load(orig) as zo:
                if all(zn[k].shape == zo[k].shape for k in traj_keys):
                    traj_diff = max(float(np.max(np.abs(zn[k] - zo[k]))) for k in traj_keys)
                else:
                    traj_diff = float("inf")
        for k, (n, o) in metrics.items():
            if n is not None and o is not None:
                max_abs[k] = max(max_abs.get(k, 0.0), abs(n - o))
        ok = ok_exact and ok_float
        n_fail += not ok
        row = dict(
            state_id=sid,
            match=ok,
            completed_new=e["completed_reference"],
            completed_v3=r["completed_reference"] == "True",
            termination_new=term_new,
            termination_v3=term_v3,
            steps_new=e["steps"],
            steps_v3=int(r["executed_steps"]),
            max_abs_traj_diff_vs_original=traj_diff,
        )
        for k, (n, o) in metrics.items():
            row[f"{k}_new"], row[f"{k}_v3"] = n, o
        rows.append(row)

    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = dict(
        case=a.case,
        seed=a.seed,
        variant=a.variant,
        level=a.level,
        n_episodes=len(rows),
        n_mismatch=n_fail,
        rel_tol=REL_TOL,
        max_abs_diff=max_abs,
        max_abs_traj_diff_vs_original=max(
            (r["max_abs_traj_diff_vs_original"] for r in rows if r["max_abs_traj_diff_vs_original"] is not None),
            default=None,
        ),
        new_summary_sha256=sha(a.new_dir / "summary.json"),
        v3_episodes_sha256=sha(a.v3_episodes),
        new_checkpoint=summary["checkpoint"],
        new_checkpoint_sha256=summary["checkpoint_sha256"],
        new_states_sha256=summary["states_sha256"],
        passed=n_fail == 0,
    )
    a.out_json.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
