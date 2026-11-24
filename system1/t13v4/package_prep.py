#!/usr/bin/env python3
"""T13 v4 preparation package: results_system1/T13v4_prep/ (check tables, timing CSV, records, SHA256SUMS; README written by
hand). Reads logs_system1/t13v4/."""
from __future__ import annotations

import csv
import hashlib
import json
import time
import math
import shutil
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L = ROOT / "logs_system1/t13v4"
OUT = ROOT / "results_system1/T13v4_prep"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def nonfinite_metrics(path: Path) -> dict:
    bad, n, last = [], 0, None
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        n += 1
        last = rec
        for k, v in rec.items():
            if isinstance(v, float) and not math.isfinite(v):
                bad.append((rec.get("iteration"), k))
    return dict(iterations=n, nonfinite=bad[:20], nonfinite_count=len(bad), last_iteration=None if last is None else last.get("iteration"),
                elapsed_median_s=st.median(json.loads(l)["elapsed_seconds"] for l in path.read_text().splitlines()[2:]) if n > 2 else None)


def checkpoint_finite(path: Path) -> dict:
    import torch
    s = torch.load(path, map_location="cpu", weights_only=False)
    bad = []
    for key in ("actor_model_state_dict", "critic_model_state_dict"):
        for name, t in s.get(key, {}).items():
            if hasattr(t, "isfinite") and not bool(t.isfinite().all()):
                bad.append(f"{key}.{name}")
    return dict(checkpoint=str(path.relative_to(ROOT)), sha256=sha(path), nonfinite_parameters=bad)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = OUT / "records"
    rec.mkdir(exist_ok=True)
    rows = []
    for name in ("table_noise_off", "table_noise_on", "push_noise_off", "push_noise_on", "table_pa",
                 "table_noise_off_r2", "table_noise_off_r3", "push_noise_off_r2", "push_noise_off_r3", "push_noise_on_r2",
                 "table_noise_off_h2", "table_noise_on_h2", "push_noise_off_h2", "push_noise_on_h2"):
        p = L / "checks" / name / "isaac_checks.json"
        if not p.exists():
            rows.append(dict(check=name, status="missing"))
            continue
        shutil.copy(p, rec / f"isaac_checks_{name}.json")
        r = json.loads(p.read_text())
        o = r["obs"]
        cr = r.get("contact_replay") or {}
        rows.append(dict(check=name, actor=o["group_shapes"]["actor_obs"], teammate=o["group_shapes"]["teammate_obs"], critic=o["group_shapes"]["critic_obs"],
                         teammate_order=" , ".join(f"{k}({o['term_dims']['teammate_obs'][k]})" for k in o["term_order"]["teammate_obs"]),
                         critic_both_commands=(o.get("critic_shared_motion_command") or {}).get("equals_both_robots"),
                         obs_v1_err_ref_ori=o["obs_v1_max_abs_err"]["ref_ori"], obs_v1_err_teammate=o["obs_v1_max_abs_err"]["teammate"],
                         noise=o["noise_enabled"], noise_in_range=all(v["within_range"] for v in o.get("noise_samples", {}).values() if isinstance(v, dict)) if o.get("noise_samples") else None,
                         replay_frames=cr.get("frames"), replay_reward_err=(cr.get("max_abs_err") or {}).get("reward"),
                         replay_lost_mismatch=(cr.get("max_abs_err") or {}).get("lost_mismatch"), replay_terminate_mismatch=(cr.get("max_abs_err") or {}).get("terminate_mismatch"),
                         hand_force_sensor_diff_n=cr.get("hand_force_link_vs_hand_sensor_max_abs_n"),
                         new_term_syncs=json.dumps({k: (v["count_per_3_calls"] if isinstance(v, dict) else v) for k, v in r["sync_counts"].items()}),
                         per_agent=json.dumps(r.get("per_agent")) if r.get("per_agent") else None,
                         object_noise_shape=(o.get("noise_samples") or {}).get("object_noise_shape"),
                         run_finished=time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                         diag_mean_team_reward=cr.get("mean_team_reward"), diag_hand_force_over_1n_in_should=cr.get("hand_force_over_1n_in_should"),
                         diag_hand_force_over_1n_r0=cr.get("hand_force_over_1n_in_should_r0"), diag_should_samples_r0=cr.get("should_samples_r0"),
                         diag_hand_force_over_1n_r1=cr.get("hand_force_over_1n_in_should_r1"), diag_should_samples_r1=cr.get("should_samples_r1"),
                         diag_lost_fraction_in_should=cr.get("lost_fraction_in_should"), diag_frames_terminating=cr.get("frames_terminating"),
                         initial_time_steps=" ".join(map(str, r["initial_time_steps"])) if "initial_time_steps" in r else None,
                         pass2_mean_team_reward=(r.get("contact_replay_pass2") or {}).get("mean_team_reward"),
                         pass2_reward_err=((r.get("contact_replay_pass2") or {}).get("max_abs_err") or {}).get("reward"),
                         pass2_lost_mismatch=((r.get("contact_replay_pass2") or {}).get("max_abs_err") or {}).get("lost_mismatch"),
                         pass2_terminate_mismatch=((r.get("contact_replay_pass2") or {}).get("max_abs_err") or {}).get("terminate_mismatch"),
                         pass2_hand_force_over_1n_in_should=(r.get("contact_replay_pass2") or {}).get("hand_force_over_1n_in_should"),
                         pass2_hand_force_over_1n_r0=(r.get("contact_replay_pass2") or {}).get("hand_force_over_1n_in_should_r0"),
                         pass2_hand_force_over_1n_r1=(r.get("contact_replay_pass2") or {}).get("hand_force_over_1n_in_should_r1")))
        if (L / "checks" / name / "contact_replay_rows.jsonl").exists():
            shutil.copy(L / "checks" / name / "contact_replay_rows.jsonl", rec / f"contact_replay_rows_{name}.jsonl")
    with open(OUT / "isaac_checks.csv", "w", newline="") as f:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for f in ("test_t13v4.log", "test_t13v4_per_agent.log"):
        if (L / f).exists():
            shutil.copy(L / f, rec / f)
    # smokes
    smokes = {}
    for name in ("d_table_200", "d_push_200", "d_table_pa_200"):
        d = L / "smoke" / name
        if (d / "metrics.jsonl").exists():
            s = nonfinite_metrics(d / "metrics.jsonl")
            ck = sorted(d.glob("model_*.pt"))
            if ck:
                s.update(checkpoint_finite(ck[-1]))
            s["status_passed"] = json.loads((d / "status.json").read_text()).get("passed") if (d / "status.json").exists() else None
            for cfg in ("run_config.json",):
                if (d / cfg).exists():
                    shutil.copy(d / cfg, rec / f"{name}_{cfg}")
            smokes[name] = s
    (OUT / "smoke_200.json").write_text(json.dumps(smokes, indent=1) + "\n")
    # speed
    sp = L / "speed"
    tim = []
    for tag, path in (("baseline_profiled_speedup_off", sp / "baseline_run/metrics.jsonl"), ("timing_speedup_off", sp / "timing_off_run/metrics.jsonl"),
                      ("timing_speedup_on", sp / "timing_on_run/metrics.jsonl"), ("smoke_table_200_speedup_on", L / "smoke/d_table_200/metrics.jsonl"),
                      *((f"interleaved_{t}", sp / f"interleaved_{t}_run/metrics.jsonl") for t in ("off1", "on1", "off2", "on2"))):
        if path.exists():
            e = [json.loads(l)["elapsed_seconds"] for l in path.read_text().splitlines()]
            tim.append(dict(run=tag, iterations=len(e), median_s_per_iter=round(st.median(e[2:]), 4), mean_s_per_iter=round(st.mean(e[2:]), 4)))
    with open(OUT / "timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "iterations", "median_s_per_iter", "mean_s_per_iter"])
        w.writeheader()
        w.writerows(tim)
    for f in ("baseline_profile.jsonl", "nocontact_profile.jsonl", "baseline_sync_scan.json"):
        if (sp / f).exists():
            shutil.copy(sp / f, rec / f)
    prof = []
    for tag, f in (("baseline", "baseline_profile.jsonl"), ("no_contact_items", "nocontact_profile.jsonl")):
        if (sp / f).exists():
            r = [json.loads(l) for l in (sp / f).read_text().splitlines()][2:]
            row = dict(run=tag, iterations=len(r) + 2, wall_s=round(st.median(x["wall_s"] for x in r), 3), cores=round(st.median(x["cores_used"] for x in r), 2),
                       gpu_util=round(st.median(x["gpu_util_mean"] for x in r if x["gpu_util_mean"] is not None), 1))
            row.update({k: round(st.median(x["sections_s"].get(k, 0.0) for x in r), 3) for k in r[-1]["sections_s"]})
            prof.append(row)
    with open(OUT / "timing_breakdown.csv", "w", newline="") as f:
        keys = []
        for r in prof:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(prof)
    for f in sorted(sp.glob("compare_*.json")):
        shutil.copy(f, rec / f.name)
    # observation-v1 validation tests (scripts/evaluate_paper_supplement.py --obs-v1 --critic-v1)
    val = []
    for name in ("table_a", "table_b", "table_pa", "push"):
        p = L / "validation" / name / "summary.json"
        if p.exists():
            sm = json.loads(p.read_text())
            eps = sm.get("episodes", [])
            val.append(dict(run=name, complete=sm.get("complete"), checkpoint=sm.get("checkpoint"), checkpoint_sha256=sm.get("checkpoint_sha256"),
                            states_sha256=sm.get("states_sha256"), episodes=len(eps), completed=sum(bool(e.get("completed_reference")) for e in eps),
                            steps=" ".join(str(e.get("steps")) for e in eps), system1_t13v4=json.dumps(sm.get("system1_t13v4")),
                            inference=sm.get("inference"), observation_noise=sm.get("observation_noise")))
    if val:
        with open(OUT / "validation_obs_v1.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(val[0]))
            w.writeheader()
            w.writerows(val)
    # determinism: two identical validation runs (table_a, table_b), every trajectory array compared bitwise
    a_dir, b_dir = L / "validation" / "table_a", L / "validation" / "table_b"
    if a_dir.exists() and b_dir.exists():
        import numpy as np
        n_arr, diff = 0, []
        for fa in sorted(a_dir.glob("*.trajectory.npz")):
            za, zb = np.load(fa), np.load(b_dir / fa.name)
            for k in za.files:
                n_arr += 1
                if not (za[k].shape == zb[k].shape and np.array_equal(za[k], zb[k], equal_nan=True)):
                    diff.append(f"{fa.name}:{k}")
        (rec / "validation_determinism.json").write_text(json.dumps(dict(runs=["table_a", "table_b"], arrays_compared=n_arr, arrays_different=diff,
                                                                          bitwise_identical=not diff), indent=1) + "\n")
    for name in ("table_a", "table_b", "table_pa", "push"):
        p = L / "validation" / name / "summary.json"
        if p.exists():
            shutil.copy(p, rec / f"validation_summary_{name}.json")
    shutil.copy(ROOT / "system1/t13v4/cloud_speed_test_plan.md", rec / "cloud_speed_test_plan.md")
    # --torch-threads switch smoke (2 iterations, 256 envs): run_config "torch_threads" records
    thr = {}
    for name in ("threads_table_2", "threads_push_2"):
        p = L / "smoke" / name / "run_config.json"
        if p.exists():
            rc = json.loads(p.read_text())
            st_ = json.loads((L / "smoke" / name / "status.json").read_text()) if (L / "smoke" / name / "status.json").exists() else None
            thr[name] = dict(torch_threads=rc.get("torch_threads"), status=st_)
    if thr:
        (rec / "torch_threads_smoke.json").write_text(json.dumps(thr, indent=1) + "\n")
    # labels
    for f in sorted((L / "labels").glob("*.json")):
        shutil.copy(f, rec / f"labels_{f.name}")
    # checksums
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in files))
    print(json.dumps(dict(isaac_checks=len(rows), smokes=list(smokes), timing=tim), indent=1))


if __name__ == "__main__":
    main()
