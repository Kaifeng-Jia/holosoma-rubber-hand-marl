"""T13 v4 section 4 A: timing breakdown, GPU->CPU sync scan and same-seed traces for the local speed-up pass.

Used only with the training scripts' --t13v4-profile / --t13v4-sync-scan / --t13v4-trace switches (off = no hooks).
  SectionTimer: wall time per section with torch.cuda.synchronize() at the section edges (profiling runs only):
    env_step (whole env.step), physics (_physics_step: PD + PhysX substeps), refresh (_refresh_sim_tensors), termination,
    reward, reset (reset_envs_idx), observations (_compute_observations), collect (learner.collect_rollout: env steps + policy
    inference + storage), update (learner.update: PPO); plus process CPU seconds and GPU utilisation samples per iteration.
  sync_scan: count synchronising CUDA calls by source location while the env steps (torch.cuda.set_sync_debug_mode).
  Trace: per iteration, the rollout's actions / rewards / dones tensors (for the 10-iteration same-seed comparison).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


class SectionTimer:
    def __init__(self, env, learner, out: Path):
        self.env, self.learner, self.out = env, learner, out
        self.acc = defaultdict(float)
        self.calls = Counter()
        self._wrap(env, "step", "env_step")
        for name in ("_physics_step", "_refresh_sim_tensors", "_check_termination", "_compute_reward", "_compute_observations"):
            self._wrap(env, name, name.strip("_").replace("compute_", "").replace("check_", ""))
        self._wrap(env, "reset_envs_idx", "reset")
        self._wrap(learner, "collect_rollout", "collect")
        self._wrap(learner, "update", "update")
        self.gpu = []
        self._stop = False
        self._thread = threading.Thread(target=self._poll_gpu, daemon=True)
        self._thread.start()
        self._cpu0 = self._cpu_seconds()
        self._t0 = time.perf_counter()
        out.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(out, "w")

    @staticmethod
    def _cpu_seconds():
        t = os.times()
        return t.user + t.system + t.children_user + t.children_system

    def _poll_gpu(self):
        while not self._stop:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                self.gpu.append(float(r.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.5)

    def _wrap(self, obj, name, label):
        orig = getattr(obj, name)

        def timed(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                torch.cuda.synchronize()
                self.acc[label] += time.perf_counter() - t
                self.calls[label] += 1
        setattr(obj, name, timed)

    def end_iteration(self, iteration: int) -> dict:
        wall = time.perf_counter() - self._t0
        cpu = self._cpu_seconds() - self._cpu0
        rec = dict(iteration=iteration, wall_s=wall, process_cpu_s=cpu, cores_used=cpu / max(wall, 1e-9),
                   gpu_util_mean=float(np.mean(self.gpu)) if self.gpu else None, sections_s=dict(self.acc), calls=dict(self.calls))
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()
        self.acc.clear()
        self.calls.clear()
        self.gpu = []
        self._cpu0, self._t0 = self._cpu_seconds(), time.perf_counter()
        return rec

    def close(self):
        self._stop = True
        self._f.close()


def sync_scan(env, learner, observations, out: Path, steps: int = 24):
    """Run `steps` rollout steps with sync debug mode on; count synchronising calls per source location."""
    counts = Counter()
    orig_step = env.step

    def step(*a, **k):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode(1)
            try:
                return orig_step(*a, **k)
            finally:
                torch.cuda.set_sync_debug_mode(0)
                for x in w:
                    if "synchroniz" in str(x.message).lower():
                        counts[f"{x.filename}:{x.lineno}"] += 1
    env.step = step
    try:
        saved = learner.num_steps_per_env if hasattr(learner, "num_steps_per_env") else None
        observations = learner.collect_rollout(env, observations)
    finally:
        env.step = orig_step
    rec = dict(steps_per_rollout=saved, sync_sites=dict(counts.most_common()), total=sum(counts.values()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    return observations, rec


class Trace:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    def record(self, learner, iteration: int, metrics: dict):
        st = learner.storage
        arrays = {}
        for kind, getter in (("agent", st.agent), ("team", st.team)):
            for key in ("actions", "rewards", "dones", "values", "actions_log_prob"):
                try:
                    arrays[f"{kind}_{key}"] = getter(key).detach().float().cpu().numpy()
                except (KeyError, AttributeError, ValueError):
                    pass
        np.savez_compressed(self.out_dir / f"iter_{iteration:03d}.npz", **arrays)
        (self.out_dir / f"iter_{iteration:03d}.json").write_text(json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float, type(None)))}, indent=1) + "\n")


__all__ = ["SectionTimer", "Trace", "sync_scan"]
