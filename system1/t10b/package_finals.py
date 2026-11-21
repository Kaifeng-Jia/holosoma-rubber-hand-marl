#!/usr/bin/env python3
"""T10b B step 5 return pack: final evaluation of a pilot's last checkpoint next to the original policy, Isaac (d_final_5070
protocol, run_finals.py) and MuJoCo (T10 setting, run_mj_finals.py), +/-2 and +/-8 (C-in MuJoCo: PD isaac and official).
Facts only, read only, CPU. Writes results_system1/T10b_friction_margin/retrain/finals/<run>/:

- episodes.csv: one row per episode (pilot and original, both simulators); fields as T10b A episodes.csv plus simulator, policy,
  level, pd. Termination reason (exclusive, T10b A order): object position > 0.25 m, else object orientation > 0.8 rad, else robot
  orientation. Isaac: errors at the last saved row against the reference frame the termination used (phases.termination_errors);
  MuJoCo: the evaluator's own last_termination errors (torso gravity-z error included). T10's non-exclusive MuJoCo counts
  (package_t10.py) are in comparison.csv too. Divergence step: first saved row with object error > 0.1 m against that row's
  reference (evaluator alignment, as object_rmse and T10s). Reference phase / holder at the termination frame and at the
  divergence step from retrain/pilot_progress/references.json.
- comparison.csv: one row per simulator x level x pd x policy (completions of the 50 +/-2 or +/-8 states, nominal, causes,
  failed-episode step median, object RMSE median).
- scene_check.json: the pilot checkpoint's Isaac scene export vs the T10 export of the original policy.
- configs/: launch / exit / run_manifest records (gzip) of every pilot run and of the MuJoCo repro; onnx_check.csv.

    hssim_s1/bin/python -B system1/t10b/package_finals.py --run cin_bucket_seed721_rand
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t10b.phases import divergence_step, termination_errors, termination_reason  # noqa: E402
from system1.t10b.run_finals import ORIGINAL  # noqa: E402

FINALS = ROOT / "logs_system1/t10b/finals"
MJ = ROOT / "logs_system1/t10b/mj_finals"
T10_MJ = ROOT / "logs_system1/t10/mj_eval"
T10_DUMP = {"cin_bucket_seed721_rand": ROOT / "logs_system1/t10/isaac/cin_bucket/dump", "d_table_seed721_rand": ROOT / "logs_system1/t10/isaac/d_table/dump"}
SCHEME = {"cin_bucket_seed721_rand": ("C-in", "bucket", 721), "d_table_seed721_rand": ("D", "table", 721)}
REFS = ROOT / "results_system1/T10b_friction_margin/retrain/pilot_progress/references.json"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def per_frame(segments, n, idx_first=2):
    lab = [None] * n
    for s in segments:
        name, a, b = s[0], s[idx_first], s[idx_first + 1]
        for k in range(a, b + 1):
            lab[k] = name
    return lab


def scene_diff(a, b, p=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(dict(path=p + "/" + k, diff="only in the original export"))
            elif k not in b:
                out.append(dict(path=p + "/" + k, diff="only in the pilot export"))
            else:
                out += scene_diff(a[k], b[k], p + "/" + k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (u, v) in enumerate(zip(a, b)):
            out += scene_diff(u, v, f"{p}[{i}]")
    elif a != b:
        out.append(dict(path=p, diff="value", pilot=json.dumps(a)[:200], original=json.dumps(b)[:200]))
    return out


def episode_rows(sim, policy, level, pd, d, phase, holder):
    s = json.loads((d / "summary.json").read_text())
    if not s.get("complete"):
        raise ValueError(f"incomplete: {d}")
    rows = []
    for e in s["episodes"]:
        r = dict(simulator=sim, policy=policy, level_cm=level, pd=pd, state_id=e["state_id"], is_nominal=e["kind"] == "nominal",
                 completed_reference=e["completed_reference"], executed_steps=e["steps"], horizon_steps=e["horizon_steps"],
                 terminal_terms=json.dumps(e["terminal_terms"], sort_keys=True), object_pos_rmse_cm=e["object_rmse_3d_m"] * 100,
                 checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"], summary_sha256=sha(d / "summary.json"),
                 eval_dir=str(d.relative_to(ROOT)))
        z = np.load(e["trajectory"])
        r["divergence_step_0p1m"] = divergence_step(z["object_pos_w"], z["reference_object_pos_w"])
        if not e["completed_reference"]:
            if sim == "isaac":
                pe, oe, f = termination_errors(z)
                rr, gz = termination_reason(pe, oe), None
            else:
                lt = e["last_termination"]
                pe, oe, gz = lt["object_pos_err"], lt["object_ori_err"], max(lt["gravity_z_err"])
                f = e["steps"] - 1  # mj_eval: termination of step k compares with reference frame k
                rr = "object_position" if pe > 0.25 else "object_orientation" if oe > 0.8 else "robot_orientation" if gz > 0.8 else "none_exceeded"
            r.update(termination_reason=rr, last_object_pos_err_m=pe, last_object_ori_err_rad=oe, last_torso_gravity_z_err_max=gz,
                     termination_reference_frame=f, phase_at_termination=phase[f], holder_at_termination=holder[f])
        k = r["divergence_step_0p1m"]
        if k is not None:
            fr = int(np.asarray(z["reference_frame"])[k])
            r.update(phase_at_divergence=phase[fr], holder_at_divergence=holder[fr])
        rows.append(r)
    return rows, s


def summarize(rows, s, sim, policy, level, pd):
    pr = [r for r in rows if not r["is_nominal"]]
    fails = [r for r in pr if not r["completed_reference"]]
    out = dict(simulator=sim, policy=policy, level_cm=level, pd=pd, n_episodes=len(pr), n_completed=len(pr) - len(fails),
               nominal_completed=[r for r in rows if r["is_nominal"]][0]["completed_reference"],
               fail_object_position=sum(r["termination_reason"] == "object_position" for r in fails),
               fail_object_orientation=sum(r["termination_reason"] == "object_orientation" for r in fails),
               fail_robot_orientation=sum(r["termination_reason"] == "robot_orientation" for r in fails),
               failed_steps_median=statistics.median(r["executed_steps"] for r in fails) if fails else None,
               failed_steps_min=min(r["executed_steps"] for r in fails) if fails else None,
               failed_steps_max=max(r["executed_steps"] for r in fails) if fails else None,
               divergence_step_median_failed=statistics.median(r["divergence_step_0p1m"] for r in fails if r["divergence_step_0p1m"] is not None)
               if any(r["divergence_step_0p1m"] is not None for r in fails) else None,
               phase_at_termination=json.dumps({p: sum(r.get("phase_at_termination") == p for r in fails) for p in sorted({r.get("phase_at_termination") for r in fails})}),
               object_pos_rmse_cm_median=statistics.median(r["object_pos_rmse_cm"] for r in pr),
               checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"], eval_dir=rows[0]["eval_dir"])
    if sim == "mujoco":  # T10 package_t10.py counts (non-exclusive)
        eps = [e for e in s["episodes"] if e["kind"] != "nominal" and not e["completed_reference"]]
        out.update(t10_terminated_object_pos=sum(e["last_termination"]["object_pos_err"] > 0.25 for e in eps),
                   t10_terminated_object_ori=sum(e["last_termination"]["object_ori_err"] > 0.8 for e in eps),
                   t10_terminated_robot_orientation=sum(max(e["last_termination"]["gravity_z_err"]) > 0.8 for e in eps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=tuple(SCHEME), required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results_system1/T10b_friction_margin/retrain/finals")
    a = ap.parse_args()
    run = a.run
    scheme, task, seed = SCHEME[run]
    out = a.out / run
    (out / "configs").mkdir(parents=True, exist_ok=True)
    ref = json.loads(REFS.read_text())[run]
    phase, holder = per_frame(ref["phase_segments"], ref["frames"]), per_frame(ref["holder_segments"], ref["frames"], idx_first=1)
    eps, comp = [], []
    pds = ("isaac",) if scheme == "D" else ("isaac", "official")
    for level in (2, 8):
        src = ORIGINAL[run][level]
        od = Path(json.loads(src.read_text())["argv"][json.loads(src.read_text())["argv"].index("--output-dir") + 1])
        for policy, d in (("pilot", FINALS / f"{run}_pm{level}"), ("original", od)):
            rows, s = episode_rows("isaac", policy, level, None, d, phase, holder)
            eps += rows
            comp.append(summarize(rows, s, "isaac", policy, level, None))
        for pd in pds:
            sfx = "" if scheme == "D" else f"_pd{pd}"
            orig = T10_MJ / f"{scheme.replace('-', '').lower()}_{task}_seed{seed}_pm{level}{sfx}"
            for policy, d in (("pilot", MJ / f"{run}_pm{level}{sfx}"), ("original", orig)):
                rows, s = episode_rows("mujoco", policy, level, pd, d, phase, holder)
                eps += rows
                comp.append(summarize(rows, s, "mujoco", policy, level, pd))
    # MuJoCo repro of the original +/-2 on the current code
    rp = MJ / (f"original_{scheme.replace('-', '').lower()}_{task}_seed{seed}_pm2" + ("" if scheme == "D" else "_pdisaac") + "_repro")
    repro = None
    if rp.exists():
        t10 = T10_MJ / (f"{scheme.replace('-', '').lower()}_{task}_seed{seed}_pm2" + ("" if scheme == "D" else "_pdisaac"))
        r, t = json.loads((rp / "summary.json").read_text()), json.loads((Path(str(t10)) / "summary.json").read_text())
        same = sum(x["steps"] == y["steps"] and x["completed_reference"] == y["completed_reference"] for x, y in zip(r["episodes"], t["episodes"]))
        repro = dict(repro_dir=str(rp.relative_to(ROOT)), t10_dir=str(Path(str(t10)).relative_to(ROOT)), episodes=len(r["episodes"]), same_completion_and_steps=same,
                     object_rmse_3d_max_abs_diff_m=max(abs(x["object_rmse_3d_m"] - y["object_rmse_3d_m"]) for x, y in zip(r["episodes"], t["episodes"])),
                     onnx_sha256_same=r["onnx_sha256"] == t["onnx_sha256"], checkpoint_sha256_same=r["checkpoint_sha256"] == t["checkpoint_sha256"],
                     states_sha256_same=r["states_sha256"] == t["states_sha256"], scene_json_sha256_same=r["scene_json_sha256"] == t["scene_json_sha256"],
                     completed_repro=sum(e["completed_reference"] for e in r["episodes"] if e["kind"] != "nominal"),
                     completed_t10=sum(e["completed_reference"] for e in t["episodes"] if e["kind"] != "nominal"))
        for f in (rp.with_name(rp.name + ".launch.json"), rp.with_name(rp.name + ".exit.json"), rp / "run_manifest.json"):
            (out / "configs" / f"mujoco__{rp.name}__{f.name.replace(rp.name + '.', '')}.gz").write_bytes(gzip.compress(f.read_bytes(), mtime=0))
    # scene export check
    dump = FINALS / f"{run}_dump" / "dump"
    sc = None
    if dump.exists():
        pa, ob = json.loads((dump / "scene.json").read_text()), json.loads((T10_DUMP[run] / "scene.json").read_text())
        ca, cb = np.load(dump / "collision_convexes.npz"), np.load(T10_DUMP[run] / "collision_convexes.npz")
        sc = dict(pilot_export=str(dump.relative_to(ROOT)), original_export=str(T10_DUMP[run].relative_to(ROOT)), differences=scene_diff(pa, ob),
                  collision_convexes_same_keys=sorted(ca.files) == sorted(cb.files),
                  collision_convexes_all_equal=all(np.array_equal(ca[k], cb[k]) for k in ca.files if k in cb.files),
                  pilot_scene_json_sha256=sha(dump / "scene.json"), original_scene_json_sha256=sha(T10_DUMP[run] / "scene.json"))
        (out / "scene_check.json").write_text(json.dumps(sc, indent=1) + "\n")
    # records
    for name in [f"{run}_pm2", f"{run}_pm8", f"{run}_dump"]:
        d = FINALS / name
        for f in (d.with_name(name + ".launch.json"), d.with_name(name + ".exit.json"), (d / "eval" if name.endswith("_dump") else d) / "run_manifest.json"):
            if f.exists():
                (out / "configs" / f"isaac__{name}__{f.name.replace(name + '.', '')}.gz").write_bytes(gzip.compress(f.read_bytes(), mtime=0))
    for level in (2, 8):
        for pd in pds:
            name = f"{run}_pm{level}" + ("" if scheme == "D" else f"_pd{pd}")
            d = MJ / name
            for f in (d.with_name(name + ".launch.json"), d.with_name(name + ".exit.json"), d / "run_manifest.json"):
                (out / "configs" / f"mujoco__{name}__{f.name.replace(name + '.', '')}.gz").write_bytes(gzip.compress(f.read_bytes(), mtime=0))
    oc = ROOT / "logs_system1/t10b/onnx_check/onnx_check.csv"
    if oc.exists():
        (out / "configs" / "onnx_check.csv").write_bytes(oc.read_bytes())
    save_csv(out / "episodes.csv", eps)
    save_csv(out / "comparison.csv", comp)
    prov = dict(run=run, script_sha256=sha(Path(__file__)), references_json_sha256=sha(REFS), repro=repro,
                scene_check_differences=None if sc is None else len(sc["differences"]))
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    for c in comp:
        print({k: c[k] for k in ("simulator", "policy", "level_cm", "pd", "n_completed", "nominal_completed", "fail_object_position",
                                  "fail_object_orientation", "fail_robot_orientation", "failed_steps_median", "divergence_step_median_failed",
                                  "phase_at_termination", "object_pos_rmse_cm_median")})
    print("repro", repro)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
