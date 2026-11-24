#!/usr/bin/env python3
"""T13 v4 section 0 with the T14 round 6 C references: package results_system1/T13v4_prep/t14ref/ (check table, smokes,
validation tests, states, labels, reference manifests, records, SHA256SUMS of the whole T13v4_prep dir; README by hand).
Reads logs_system1/t13v4/t14/ and system1/t13v4/{states,labels}_t14r6C."""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t13v4.package_prep import checkpoint_finite, nonfinite_metrics, sha  # noqa: E402

L = ROOT / "logs_system1/t13v4/t14"
TOP = ROOT / "results_system1/T13v4_prep"
OUT = TOP / "t14ref"
W = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/system1_t14r6C"
OLD = TOP / "isaac_checks.csv"


def main():
    rec = OUT / "records"
    rec.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("table_noise_off", "table_noise_off_r2", "table_noise_on", "table_pa", "push_noise_off", "push_noise_off_r2", "push_noise_on"):
        p = L / "checks" / name / "isaac_checks.json"
        if not p.exists():
            rows.append(dict(check=name, status="missing"))
            continue
        shutil.copy(p, rec / f"isaac_checks_{name}.json")
        shutil.copy(L / "checks" / name / "contact_replay_rows.jsonl", rec / f"contact_replay_rows_{name}.jsonl")
        r = json.loads(p.read_text())
        o, cr = r["obs"], r.get("contact_replay") or {}
        e = cr.get("max_abs_err") or {}
        rows.append(dict(check=name, reference="T14 round 6 C", actor=o["group_shapes"]["actor_obs"], teammate=o["group_shapes"]["teammate_obs"],
                         critic=o["group_shapes"]["critic_obs"], critic_both_commands=(o.get("critic_shared_motion_command") or {}).get("equals_both_robots"),
                         obs_v1_err_ref_ori=o["obs_v1_max_abs_err"]["ref_ori"], obs_v1_err_teammate=o["obs_v1_max_abs_err"]["teammate"],
                         noise=o["noise_enabled"], noise_in_range=all(v["within_range"] for v in o.get("noise_samples", {}).values() if isinstance(v, dict)) if o.get("noise_samples") else None,
                         clip_partner=(o.get("clip_fraction") or {}).get("partner"), clip_object=(o.get("clip_fraction") or {}).get("object"),
                         replay_frames=cr.get("frames"), replay_reward_err=e.get("reward"), replay_d_err=e.get("d"), replay_penalty_count_err=e.get("count"),
                         replay_lost_mismatch=e.get("lost_mismatch"), replay_terminate_mismatch=e.get("terminate_mismatch"),
                         hand_force_sensor_diff_n=cr.get("hand_force_link_vs_hand_sensor_max_abs_n"),
                         per_agent_max_abs_mean_minus_team=max(v["max_abs_mean_minus_team"] for v in r["per_agent"].values() if isinstance(v, dict) and "max_abs_mean_minus_team" in v) if r.get("per_agent") else None,
                         diag_should_fraction=cr.get("should_fraction"), diag_mean_team_reward=cr.get("mean_team_reward"),
                         diag_hand_force_over_1n_in_should=cr.get("hand_force_over_1n_in_should"),
                         diag_hand_force_over_1n_r0=cr.get("hand_force_over_1n_in_should_r0"), diag_should_samples_r0=cr.get("should_samples_r0"),
                         diag_hand_force_over_1n_r1=cr.get("hand_force_over_1n_in_should_r1"), diag_should_samples_r1=cr.get("should_samples_r1"),
                         diag_lost_fraction_in_should=cr.get("lost_fraction_in_should"), diag_frames_terminating=cr.get("frames_terminating"),
                         diag_frames_with_penalty=cr.get("frames_with_penalty"), initial_time_steps=" ".join(map(str, r.get("initial_time_steps", []))),
                         run_finished=time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))))
    with open(OUT / "isaac_checks_t14ref.csv", "w", newline="") as f:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    smokes = {}
    for name in ("d_table_200", "d_push_200", "d_table_pa_200"):
        d = L / "smoke" / name
        if (d / "metrics.jsonl").exists():
            s = nonfinite_metrics(d / "metrics.jsonl")
            ms = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines()]
            for k in ("mean_reward", "reward_mean", "Train/mean_reward"):
                if k in ms[0]:
                    s["reward_first_last"] = [ms[0][k], ms[-1][k]]
            for k in ("num_resets", "resets", "episode_resets"):
                if k in ms[0]:
                    s["resets_first_last"] = [ms[0][k], ms[-1][k]]
            ck = sorted(d.glob("model_*.pt"))
            if ck:
                s.update(checkpoint_finite(ck[-1]))
            s["status"] = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else None
            for cfg in ("run_config.json",):
                if (d / cfg).exists():
                    shutil.copy(d / cfg, rec / f"{name}_{cfg}")
                    rc = json.loads((d / cfg).read_text())
                    s["system1_t14_reference"] = rc.get("system1_t14_reference") or (rc.get("placement_provenance") or {}).get("t14_reference")
                    s["runtime_reference_check"] = rc.get("runtime_reference_check")
            smokes[name] = s
    (OUT / "smoke_200_t14ref.json").write_text(json.dumps(smokes, indent=1, default=str) + "\n")
    val = []
    for name in ("table", "table_b", "table_pa", "push"):
        p = L / "validation" / name / "summary.json"
        if p.exists():
            sm = json.loads(p.read_text())
            eps = sm.get("episodes", [])
            shutil.copy(p, rec / f"validation_summary_{name}.json")
            val.append(dict(run=name, complete=sm.get("complete"), checkpoint=sm.get("checkpoint"), checkpoint_sha256=sm.get("checkpoint_sha256"),
                            states_sha256=sm.get("states_sha256"), episodes=len(eps), completed=sum(bool(e.get("completed_reference")) for e in eps),
                            steps=" ".join(str(e.get("steps")) for e in eps), system1_t13v4=json.dumps(sm.get("system1_t13v4")),
                            system1_t14_reference=json.dumps(sm.get("system1_t14_reference")), inference=sm.get("inference")))
    if val:
        with open(OUT / "validation_obs_v1_t14ref.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(val[0]))
            w.writeheader()
            w.writerows(val)
    a_dir, b_dir = L / "validation" / "table", L / "validation" / "table_b"
    if a_dir.exists() and b_dir.exists():
        import numpy as np
        n_arr, diff = 0, []
        for fa in sorted(a_dir.glob("*.trajectory.npz")):
            za, zb = np.load(fa), np.load(b_dir / fa.name)
            for k in za.files:
                n_arr += 1
                if not (za[k].shape == zb[k].shape and np.array_equal(za[k], zb[k], equal_nan=True)):
                    diff.append(f"{fa.name}:{k}")
        (rec / "validation_determinism.json").write_text(json.dumps(dict(runs=["table", "table_b"], arrays_compared=n_arr, arrays_different=diff,
                                                                          bitwise_identical=not diff), indent=1) + "\n")
    # states, labels, reference manifests
    st = ROOT / "system1/t13v4/states_t14r6C"
    shutil.copy(st / "summary.json", OUT / "states_summary.json")
    for f in sorted((ROOT / "system1/t13v4/labels_t14r6C").glob("*.json")):
        shutil.copy(f, rec / f"labels_{f.name}")
    for t in ("table", "push"):
        shutil.copy(W / t / "t14_reference.json", OUT / f"t14_reference_{t}.json")
    shutil.copy(W / "table/training_asset_manifest.json", rec / "table_training_asset_manifest.json")
    for f in ("make_states_t14.log",):
        if (ROOT / "logs_system1/t13v4" / f).exists():
            shutil.copy(ROOT / "logs_system1/t13v4" / f, rec / f)
    files = sorted(p for p in TOP.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (TOP / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(TOP)}\n" for p in files))
    print(json.dumps(dict(checks=len(rows), smokes=list(smokes), validation=len(val)), indent=1))


if __name__ == "__main__":
    main()
