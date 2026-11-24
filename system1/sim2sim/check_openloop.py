#!/usr/bin/env python3
"""T10 step 2.3: open-loop replay. MuJoCo starts from the nominal initial state (reset exactly as the evaluator; equal to Isaac's
to 1e-7, check_initial_states.py) and executes, step by step, the PD targets Isaac recorded for the same episode (system1/sim2sim/
run_isaac_dumps.py; D: default + scale * a, C-in: SONIC's q*), with the same PD law per 200 Hz substep. First 1 s (50 control
steps). Reported against Isaac's post-step state (trajectory row k+1): root position error per robot, max joint-angle error, object
position error, and the substep torques. Report only; nothing is tuned.

    sim2sim_s1/bin/python -B system1/sim2sim/check_openloop.py --out results_system1/T10_sim2sim/consistency
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sim2sim.mj_env import PairEnv  # noqa: E402

RUNS = dict(d_table=("table", "D"), cin_table=("table", "C-in"), d_bucket=("bucket", "D"), cin_bucket=("bucket", "C-in"),
            d_push=("push", "D"), d_pull=("pull", "D"), d_kick=("kick", "D"), d_chair=("chair", "D"), cin_pull=("pull", "C-in"),
            cout_table=("table", "C-out"), cout_bucket=("bucket", "C-out"), cout_pull=("pull", "C-out"))  # T13 C1: only with --runs
T10_RUNS = "d_table,cin_table,d_bucket,cin_bucket"
STEPS = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", default=T10_RUNS)
    a = ap.parse_args()
    runs = {k: RUNS[k] for k in a.runs.split(",")}
    a.out.mkdir(parents=True, exist_ok=True)
    rows, summ = [], {}
    for run, (task, scheme) in runs.items():
        base = ROOT / "logs_system1/t10/isaac" / run
        rec = np.load(base / "dump/nominal.sim2sim.npz")
        traj = np.load(base / "eval/nominal.trajectory.npz")
        env = PairEnv(base / "dump", task)
        states = json.loads(Path(json.loads((base / "launch.json").read_text())["argv"][json.loads((base / "launch.json").read_text())["argv"].index("--states") + 1]).read_text())["states"]
        env.reset(states[0])
        assert states[0]["state_id"] == "nominal"
        for k in range(STEPS):
            env.physics_step(rec["pd_target"][k, 0].reshape(2, 29).astype(np.float64))
            st = env.state()
            r = dict(run=run, task=task, scheme=scheme, step=k + 1, time_s=(k + 1) * 0.02,
                     root_pos_err_m_r0=float(np.linalg.norm(st["root"][0, :3] - traj["root_pos"][k + 1, 0])),
                     root_pos_err_m_r1=float(np.linalg.norm(st["root"][1, :3] - traj["root_pos"][k + 1, 1])),
                     joint_err_max_rad=float(np.abs(st["dof_pos"] - traj["dof_pos"][k + 1]).max()),
                     joint_err_mean_rad=float(np.abs(st["dof_pos"] - traj["dof_pos"][k + 1]).mean()),
                     object_pos_err_m=float(np.linalg.norm(st["obj_pos"] - traj["object_pos_w"][k + 1])),
                     torque_err_max_nm=float(np.abs(env.substep_torques - rec["torques_substep"][k, 0]).max()))
            rows.append(r)
        last = [x for x in rows if x["run"] == run][-1]
        summ[run] = {k: v for k, v in last.items() if k.endswith(("_m_r0", "_m_r1", "_rad", "_m"))}
        print(run, json.dumps(summ[run]), flush=True)
    with (a.out / "openloop_1s.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out / "openloop_1s_summary.json").write_text(json.dumps(dict(at_1s=summ, note="report only; no parameter tuned"), indent=1) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    colors = dict(d_table="#eb6834", cin_table="#1baf7a", d_bucket="#b5461a", cin_bucket="#0e6e4c",
                  d_push="#eb6834", d_pull="#b5461a", d_kick="#f2a37a", d_chair="#7a2e0f", cin_pull="#1baf7a",
                  cout_table="#2a78d6", cout_bucket="#164a8c", cout_pull="#6aa8ee")
    for run in runs:
        rr = [x for x in rows if x["run"] == run]
        t = [x["time_s"] for x in rr]
        axes[0].plot(t, [max(x["root_pos_err_m_r0"], x["root_pos_err_m_r1"]) * 100 for x in rr], color=colors[run], label=run)
        axes[1].plot(t, [x["joint_err_max_rad"] for x in rr], color=colors[run], label=run)
        axes[2].plot(t, [x["object_pos_err_m"] * 100 for x in rr], color=colors[run], label=run)
    for ax, title in zip(axes, ("root position error, worse robot (cm)", "max joint angle error (rad)", "object position error (cm)")):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Open-loop replay of Isaac PD targets in MuJoCo, nominal episode, first 1 s (report only)", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out / "openloop_1s.png", dpi=150)


if __name__ == "__main__":
    main()
