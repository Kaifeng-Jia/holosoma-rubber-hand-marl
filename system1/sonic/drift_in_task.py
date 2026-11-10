#!/usr/bin/env python3
"""In-task pelvis drift (preregistration: results_system1/T01b/preregistration_drift_in_task.md).

Reads existing evaluation trajectories only (no simulation):
  * any number of System1 evaluation roots, `--scheme LABEL=ROOT` (ROOT/<task>_pm2/summary.json), e.g. A, later B;
  * D = V3 closeout episodes.csv, variant Full, +/-2, seeds 721/722/723 (trajectory_path column).
Writes <out>/episodes.csv (per episode x robot), <out>/summary.csv (task x robot x scheme[/seed]),
<out>/sync_vs_d.csv (D at the step where the other scheme's episode ended, paired by state_id).

    hssim_s1/bin/python -B system1/sonic/drift_in_task.py --scheme A=logs_system1/T01b/schemeA_main \
        --out results_system1/T01b/drift_in_task
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sonic.task_refs import TASKS, load_task_reference  # noqa: E402

V3 = Path("/home/kevin/holosoma-core4d-base/paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/results/episodes.csv")
SEC = 50          # rows per second (50 Hz)
MIN_DQ = 0.05     # m, seconds with smaller reference displacement are not counted for lag
MIN_LREF = 0.10   # m


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Refs:
    def __init__(self):
        self.cache, self.meta = {}, {}

    def __call__(self, task):
        if task not in self.cache:
            ref, bodies, _, meta = load_task_reference(task)
            a = {k: v.numpy() for k, v in ref.sample(torch.arange(ref.num_frames)).items()}
            self.cache[task] = (a["agent_body_pos_w"][:, :, bodies.index("pelvis")].astype(np.float64), a["object_pos_w"].astype(np.float64))
            self.meta[task] = meta
        return self.cache[task]


def episode_metrics(refs, task, traj):
    pel, obj = refs(task)
    with np.load(traj, allow_pickle=False) as z:
        f = z["reference_frame"].astype(int)
        off = z["reference_object_pos_w"].astype(np.float64) - obj[f]
        if np.max(np.abs(off - off[0])) >= 3e-6:
            raise RuntimeError(f"reference alignment offset not constant: {traj}")
        pref = pel[f] + off[:, None, :]                      # [J, 2, 3]
        prob = z["root_pos"].astype(np.float64)              # [J, 2, 3]
    d = np.linalg.norm(prob[..., :2] - pref[..., :2], axis=-1)   # [J, 2]
    out = []
    for r in range(2):
        path = pel[f[0]:f[-1] + 1, r, :2]
        lref = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=-1))) if len(path) > 1 else 0.0
        lag = elig = 0
        for s in range(0, len(f) - 1, SEC):
            e = min(s + SEC, len(f) - 1)
            dq = pref[e, r, :2] - pref[s, r, :2]
            dr = prob[e, r, :2] - prob[s, r, :2]
            n = np.linalg.norm(dq)
            if n >= MIN_DQ:
                elig += 1
                lag += float(np.dot(dr - dq, dq / n)) < 0
        out.append(dict(robot=r, rows=len(f), d_final_cm=100 * float(d[-1, r]), d_max_cm=100 * float(d[:, r].max()),
                        ref_path_m=lref, drift_per_m_ref_cm=100 * float(d[-1, r]) / lref if lref >= MIN_LREF else None,
                        lag_seconds=lag, eligible_seconds=elig, d_series=d[:, r]))
    return out


def summarize(rows):
    def mm(k):
        v = [r[k] for r in rows if r[k] is not None]
        return (float(np.mean(v)) if v else None, float(np.median(v)) if v else None, len(v))
    x = {}
    for k in ("d_final_cm", "d_max_cm", "drift_per_m_ref_cm"):
        x[k + "_mean"], x[k + "_median"], x[k + "_n"] = mm(k)
    lag, elig = sum(r["lag_seconds"] for r in rows), sum(r["eligible_seconds"] for r in rows)
    x.update(lag_seconds=lag, eligible_seconds=elig, lag_fraction=lag / elig if elig else None,
             n_episodes=len(rows), n_completed=sum(bool(r["completed"]) for r in rows),
             executed_steps_mean=float(np.mean([r["executed_steps"] for r in rows])))
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", action="append", required=True, help="LABEL=ROOT with ROOT/<task>_pm2/summary.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--no-d", action="store_true")
    a = p.parse_args()
    tasks = a.tasks.split(",")
    refs = Refs()
    eps = []   # dict per episode x robot
    state_sha = {}
    for spec in a.scheme:
        label, root = spec.split("=", 1)
        root = Path(root) if Path(root).is_absolute() else ROOT / root
        for task in tasks:
            sp = root / f"{task}_pm2/summary.json"
            if not sp.exists():
                continue
            s = json.loads(sp.read_text())
            state_sha[(label, task)] = s["states_sha256"]
            for e in s["episodes"]:
                if e["kind"] == "nominal":
                    continue
                for m in episode_metrics(refs, task, e["trajectory"]):
                    eps.append(dict(scheme=label, seed=None, task=task, state_id=e["state_id"], completed=e["completed_reference"],
                                    executed_steps=e["steps"], trajectory=e["trajectory"], **m))
            print(label, task, flush=True)
    if not a.no_d:
        with V3.open() as f:
            for r in csv.DictReader(f):
                if r["variant"] != "Full" or r["perturbation_level_cm"] != "2" or r["is_nominal"] != "False" or r["task"] not in tasks:
                    continue
                for lab in [k for k in state_sha if k[1] == r["task"]]:
                    if state_sha[lab] != r["state_file_sha256"]:
                        raise RuntimeError(f"initial state files differ: {lab} vs D {r['task']} {r['training_seed']}")
                for m in episode_metrics(refs, r["task"], r["trajectory_path"]):
                    eps.append(dict(scheme="D", seed=int(r["training_seed"]), task=r["task"], state_id=r["state_id"],
                                    completed=r["completed_reference"] == "True", executed_steps=int(r["executed_steps"]),
                                    trajectory=r["trajectory_path"], **m))
        print("D done", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    cols = [k for k in eps[0] if k != "d_series"]
    with (a.out / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: r[k] for k in cols} for r in eps)
    groups = defaultdict(list)
    for r in eps:
        groups[(r["task"], r["robot"], r["scheme"], r["seed"])].append(r)
    summ = []
    for task in tasks:
        for robot in (0, 1):
            schemes = sorted({k[2] for k in groups if k[0] == task and k[1] == robot})
            for sc in schemes:
                seeds = sorted({k[3] for k in groups if k[:3] == (task, robot, sc)}, key=lambda x: -1 if x is None else x)
                per_seed = []
                for sd in seeds:
                    x = summarize(groups[(task, robot, sc, sd)])
                    per_seed.append(x)
                    summ.append(dict(task=task, robot=robot, scheme=sc, seed=sd, **x))
                if sc == "D" and len(per_seed) > 1:
                    mean = {k: float(np.mean([x[k] for x in per_seed if x[k] is not None])) for k in per_seed[0]
                            if isinstance(per_seed[0][k], (int, float)) and all(x[k] is not None for x in per_seed)}
                    summ.append(dict(task=task, robot=robot, scheme="D", seed="mean_3_seeds", **mean))
    with (a.out / "summary.csv").open("w", newline="") as f:
        keys = list(dict.fromkeys(k for r in summ for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summ)
    # synchronous comparison: D's d at the row where the other scheme's episode ended (same state_id)
    idx = {(r["scheme"], r["seed"], r["task"], r["state_id"], r["robot"]): r for r in eps}
    sync = []
    for sc in sorted({r["scheme"] for r in eps if r["scheme"] != "D"}):
        for task in tasks:
            for robot in (0, 1):
                for sd in (721, 722, 723):
                    diffs, d_ended_first = [], 0
                    for i in range(50):
                        sid = f"perturbed_{i:03d}"
                        o, dd = idx.get((sc, None, task, sid, robot)), idx.get(("D", sd, task, sid, robot))
                        if o is None or dd is None:
                            continue
                        j = o["rows"] - 1
                        if j < len(dd["d_series"]):
                            dv = 100 * float(dd["d_series"][j])
                        else:
                            dv = dd["d_final_cm"]
                            d_ended_first += 1
                        diffs.append(dv - o["d_final_cm"])
                    if diffs:
                        sync.append(dict(scheme=sc, task=task, robot=robot, d_seed=sd, n_pairs=len(diffs),
                                         d_ended_before_other=d_ended_first,
                                         median_d_minus_other_cm=float(np.median(diffs)),
                                         mean_d_minus_other_cm=float(np.mean(diffs))))
    with (a.out / "sync_vs_d.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sync[0]))
        w.writeheader()
        w.writerows(sync)
    (a.out / "provenance.json").write_text(json.dumps(dict(
        preregistration="results_system1/T01b/preregistration_drift_in_task.md", schemes=a.scheme,
        v3_episodes=str(V3), v3_episodes_sha256=sha(V3), references=refs.meta, rows_per_second=SEC,
        min_ref_displacement_m=MIN_DQ, min_ref_path_m=MIN_LREF, episode_robot_rows=len(eps)), indent=1) + "\n")
    print("rows", len(eps))


if __name__ == "__main__":
    main()
