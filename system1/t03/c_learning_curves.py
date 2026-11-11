#!/usr/bin/env python3
"""Per-iteration training curves of the 3 scheme C main runs and the D from-scratch runs (Kevin 2026-11-11: C does not
learn; logs only, no run stopped). Same sources as early_progress.py (D: Pull V3-25/26/27, Table T-07, Bin B-RZ0).

Per iteration: entropy (per robot, 29-D diagonal Gaussian) and the geometric-mean action std derived from it,
exp(H/29 - 0.5 ln(2 pi e)); action_noise_std_mean where logged (Pull trainer); mean episode length as logged
(Pull: Episode/mean_length_last100_steps; Table/Bin: Episode/reset_length_mean_steps) and the uniform proxy
num_envs*steps_per_env/reset_count; termination fractions of the episodes that ended in the iteration
(timeout / tracking-or-physical failure); kl; actor and critic learning rates; value_loss; surrogate_loss; actor_grad_norm.

    hssim_s1/bin/python -B system1/t03/c_learning_curves.py --c-snapshot <dir> --out results_system1/T03/c_learning_diag_20261111
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t03.early_progress import runs, sha  # noqa: E402

H_UNIT = 0.5 * math.log(2 * math.pi * math.e)
NDIM = 29
WINDOW = 100
POINTS = (100, 500, 1000, 2000, 3000, 4000)
FIELDS = ["entropy", "sigma_geomean_from_entropy", "action_noise_std_mean", "mean_episode_length_logged", "episode_length_source",
          "episode_length_proxy_steps", "frac_timeout", "frac_failure", "kl", "actor_learning_rate", "critic_learning_rate",
          "value_loss", "surrogate_loss", "actor_grad_norm"]


def env_size(files):
    for f in files:
        rc = Path(f).with_name("run_config.json")
        if rc.exists():
            j = json.loads(rc.read_text())
            if "num_envs" in j and "steps_per_env" in j:
                return int(j["num_envs"]), int(j["steps_per_env"]), str(rc)
    return 2048, 24, "assumed 2048 x 24 (launch commands of all runs)"


def rows_of(files, case):
    E, S, _ = env_size(files)
    out = {}
    for f in files:
        with open(f) as fh:
            for l in fh:
                j = json.loads(l)
                it = int(j["iteration"])
                rc = j.get("reset_count") or 0
                fail = j.get("tracking_failure_count", j.get("physical_failure_count"))
                if case == "pull":
                    L, src = j.get("Episode/mean_length_last100_steps"), "Episode/mean_length_last100_steps"
                else:
                    L, src = j.get("Episode/reset_length_mean_steps"), "Episode/reset_length_mean_steps"
                out[it] = dict(entropy=j.get("entropy"), sigma_geomean_from_entropy=math.exp(j["entropy"] / NDIM - H_UNIT) if j.get("entropy") is not None else None,
                               action_noise_std_mean=j.get("action_noise_std_mean"), mean_episode_length_logged=L, episode_length_source=src,
                               episode_length_proxy_steps=E * S / rc if rc else None,
                               frac_timeout=j.get("timeout_count", 0) / rc if rc else None, frac_failure=fail / rc if (rc and fail is not None) else None,
                               kl=j.get("kl"), actor_learning_rate=j.get("actor_learning_rate"), critic_learning_rate=j.get("critic_learning_rate"),
                               value_loss=j.get("value_loss"), surrogate_loss=j.get("surrogate_loss"), actor_grad_norm=j.get("actor_grad_norm"))
    return out


def wmean(rows, k, key):
    v = [rows[i][key] for i in range(k - WINDOW + 1, k + 1) if i in rows and isinstance(rows[i].get(key), (int, float))]
    return float(np.mean(v)) if len(v) == WINDOW else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c-snapshot", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    R = runs(a.c_snapshot)
    data = {k: rows_of(v, k[0]) for k, v in R.items()}
    src = []
    for k, files in R.items():
        E, S, how = env_size(files)
        src += [dict(task=k[0], scheme=k[1], seed=k[2], file=str(f), sha256=sha(f), num_envs=E, steps_per_env=S, env_size_source=how,
                     first_iteration=min(data[k]), last_iteration=max(data[k])) for f in files]
    summ = []
    for case in ("pull", "table", "bucket"):
        c_last = max(data[(case, "C", "721")])
        with gzip.open(a.out / f"curves_{case}.csv.gz", "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["task", "scheme", "seed", "iteration", *FIELDS])
            w.writeheader()
            for (cs, sch, sd), rows in sorted(data.items()):
                if cs != case:
                    continue
                for it in sorted(rows):
                    if it <= c_last:  # D exported up to the latest C iteration (same iterations)
                        w.writerow(dict(task=cs, scheme=sch, seed=sd, iteration=it, **rows[it]))
        for k_it in [x for x in POINTS if x <= c_last] + [c_last]:
            for (cs, sch, sd), rows in sorted(data.items()):
                if cs == case:
                    summ.append(dict(task=cs, iteration_window_end=k_it, window=WINDOW, scheme=sch, seed=sd,
                                     **{f: wmean(rows, k_it, f) for f in FIELDS if f != "episode_length_source"}))
    with (a.out / "summary_windows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    with (a.out / "sources.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(src[0]))
        w.writeheader()
        w.writerows(src)
    print("done", a.out)


if __name__ == "__main__":
    main()
