#!/usr/bin/env python3
"""T13 v4 preparation package: results_system1/T13v4_prep/ (check tables, timing CSV, records, SHA256SUMS; README written by
hand). Reads logs_system1/t13v4/."""
from __future__ import annotations

import csv
import hashlib
import json
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
    for name in ("table_noise_off", "table_noise_on", "push_noise_off", "push_noise_on", "table_pa"):
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
                         noise=o["noise_enabled"], noise_in_range=all(v["within_range"] for v in o.get("noise_samples", {}).values()) if o.get("noise_samples") else None,
                         replay_frames=cr.get("frames"), replay_reward_err=(cr.get("max_abs_err") or {}).get("reward"),
                         replay_lost_mismatch=(cr.get("max_abs_err") or {}).get("lost_mismatch"), replay_terminate_mismatch=(cr.get("max_abs_err") or {}).get("terminate_mismatch"),
                         hand_force_sensor_diff_n=cr.get("hand_force_link_vs_hand_sensor_max_abs_n"),
                         new_term_syncs=json.dumps({k: (v["count_per_3_calls"] if isinstance(v, dict) else v) for k, v in r["sync_counts"].items()}),
                         per_agent=json.dumps(r.get("per_agent")) if r.get("per_agent") else None))
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
    # labels
    for f in sorted((L / "labels").glob("*.json")):
        shutil.copy(f, rec / f"labels_{f.name}")
    # checksums
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in files))
    print(json.dumps(dict(isaac_checks=len(rows), smokes=list(smokes), timing=tim), indent=1))


if __name__ == "__main__":
    main()
