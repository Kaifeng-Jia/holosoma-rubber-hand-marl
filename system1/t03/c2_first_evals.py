#!/usr/bin/env python3
"""T03 C v2 round, the first cloud evaluation list (system1/cloud/make_c2_eval_jobs.py, 12 jobs, 2026-11-11). CPU only;
reads the copied evaluator outputs, never simulates.

Definitions (written before computing anything from these 12 outputs):
  * Episodes: +/-2 frozen states (pull/table/bucket_states.json, sha256 in each manifest): 50 perturbed + nominal. Summary
    counts use the 50 perturbed states; the nominal episode is reported on its own row.
  * completed = summary.json completed_reference; progress = summary.json progress (mean over the 50); fall = sonic_fell_any
    (SONIC evaluation fall rule, per episode, either robot).
  * C v1 at stop: one row per run (checkpoint = newest model_*.pt at the stop, sha256 checked against
    logs/train/ops/<label>.stopped.json when that record exists; c_pull_m20kg joint8000 completed at 8050, no stop record).
  * Constant-dq check (T03 "修正量有没有用的检查", diagnostic, not a scheme), Pull 20 kg, scheme A path plus a constant dq on
    SONIC's input reference: dq0 / hip_pitch (both) -0.5 x 0.5475 rad and waist_pitch +0.5 x 0.4386 rad / the same at 1.0 x.
    - push ratio: T02 preregistration 9.1 (system1/t02/object_analysis.push_at) at the last executed step: projection of the
      object displacement on the reference object displacement, divided by |reference displacement| (only when >= 0.05 m).
    - hand height relative to the table leg top: system1/t02/pull_arm_diagnosis geometry (results_system1/T02/
      pull_arm_diagnosis/DEFINITIONS.md): rubber_hand_link origin by FK of the executed robot state, in the current table
      frame; leg top = table-local y 0.0110 m (collision box of objects_widetable_plan5_pull_training.urdf). Height = y - 0.0110.
      Reported per robot for the outer hand and the inner hand at t_f = 100 (start of the force phase, from the reference
      table displacement) over the episode-robots that reach step 100 (count reported), and the mean over the force phase
      (step 100 to the episode end). The reference value at frame 100 is a separate row.
    - "behind" at t_f: the T02 definition (outer hand a < 0, horizontal distance to the leg axis < 0.12 m, height <= leg top
      + 0.05 m).
    - Paired against dq0 on the same state: number of states completed only with dq, only with dq0.

    hssim_s1/bin/python -B system1/t03/c2_first_evals.py --src logs_system1/cloud_evals/c2_first --out results_system1/T03/c2_first_evals
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from system1.t02 import pull_arm_diagnosis as pad  # noqa: E402
from system1.t02.object_analysis import push_at  # noqa: E402

LEG_TOP = pad.LEG_Y[1]
CONST = ["constdq_pull_dq0_pm2", "constdq_pull_hipwaist_0p5_pm2", "constdq_pull_hipwaist_1p0_pm2"]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def wcsv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def base_row(d, ep_list):
    pert = [e for e in ep_list if e["kind"] != "nominal"]
    nom = [e for e in ep_list if e["kind"] == "nominal"]
    return dict(n_perturbed=len(pert), n_completed=sum(bool(e["completed_reference"]) for e in pert),
                progress_mean=mean([e["progress"] for e in pert]), n_fell=sum(bool(e.get("sonic_fell_any")) for e in pert),
                executed_steps_median=med([e["steps"] for e in pert]),
                nominal_completed=bool(nom[0]["completed_reference"]) if nom else None, nominal_progress=nom[0]["progress"] if nom else None,
                nominal_steps=nom[0]["steps"] if nom else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stopped", type=Path, default=None, help="local copy of logs/train/ops (for the stop records), optional")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = {j["name"]: j for j in json.loads((a.src / "jobs.json").read_text())}
    if len(jobs) != 12:
        raise SystemExit(f"expected 12 jobs, found {len(jobs)}")
    prov, c1 = [], []
    for name, j in jobs.items():
        d = a.src / ("const_dq" if j["meta"]["kind"] == "const_dq" else "c1_stop") / name
        s = json.loads((d / "summary.json").read_text())
        m = json.loads((d / "run_manifest.json").read_text())
        launch = json.loads((a.src / "ops" / f"{name}.launch.json").read_text())
        ex = json.loads((a.src / "ops" / f"{name}.exit.json").read_text())
        if not s.get("complete") or s["episode_count"] != 51 or ex["returncode"] != 0:
            raise SystemExit(f"incomplete {name}")
        prov.append(dict(name=name, kind=j["meta"]["kind"], git_head=launch["git_head"], git_status_clean=launch["git_status"] == "",
                         started=launch["started"], ended=ex["ended"], gpu=s["gpu_name"], cuda_visible=launch["env"]["CUDA_VISIBLE_DEVICES"],
                         tf32=s["tf32"], checkpoint=s["checkpoint"], checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"],
                         sonic_checkpoint_sha256=s["sonic_checkpoint_sha256"], robot_urdf=s["robot_urdf"], object_urdf=s["object_urdf"],
                         mass_kg=s["physics"]["mass"][0][0], const_dq=json.dumps(m.get("system1_sonic_const_dq")),
                         argv=" ".join(launch["argv"]), summary_sha256=sha(d / "summary.json"), manifest_sha256=sha(d / "run_manifest.json")))
        if j["meta"]["kind"] == "c1_stop":
            st = None
            if a.stopped and j["meta"].get("stopped_record"):
                f = a.stopped / Path(j["meta"]["stopped_record"]).name
                st = json.loads(f.read_text()) if f.exists() else None
            c1.append(dict(run=j["meta"]["run"], case=j["meta"]["case"], mass_kg=j["meta"]["mass_kg"], iteration=j["meta"]["iteration"],
                           checkpoint_sha256=s["checkpoint_sha256"],
                           stop_record_newest_sha256=(st or {}).get("newest_checkpoint_sha256"),
                           stop_record_sha_match=(st or {}).get("newest_checkpoint_sha256") == s["checkpoint_sha256"] if st else None,
                           stop_last_logged_iteration=(st or {}).get("last_logged_iteration"), **base_row(d, s["episodes"])))
    wcsv(a.out / "c1_stop_summary.csv", c1)
    wcsv(a.out / "provenance.csv", prov)

    # constant dq
    ctx = pad.Ctx()
    ep_rows, summ = [], []
    by = {}
    for name in CONST:
        d = a.src / "const_dq" / name
        s = json.loads((d / "summary.json").read_text())
        by[name] = {e["state_id"]: e for e in s["episodes"]}
        for e in s["episodes"]:
            tp = d / Path(e["trajectory"]).name
            z = np.load(tp)
            zz = {k: z[k].astype(np.float64) if z[k].dtype.kind == "f" else z[k] for k in z.files}
            if len(zz["dof_pos"]) != e["steps"] + 1:
                raise SystemExit(f"rows/steps {tp}")
            g = pad.geometry(ctx, zz)
            T = len(zz["dof_pos"]) - 1
            pu = push_at(zz["object_pos_w"], zz["reference_object_pos_w"], T)
            for r in range(2):
                row = dict(setting=name, state_id=e["state_id"], kind=e["kind"], robot=r, executed_steps=e["steps"],
                           completed_reference=e["completed_reference"], progress=e["progress"], sonic_fell_any=e.get("sonic_fell_any"),
                           push_distance_m=pu["push_distance_m"], push_ratio=pu["push_ratio"], ref_disp_m=pu["ref_disp_m"],
                           reached_t_f=T >= pad.T_F, trajectory=str(tp.relative_to(a.src)), trajectory_sha256=sha(tp))
                if T >= pad.T_F:
                    for h in ("outer", "inner"):
                        row[f"tf_{h}_hand_minus_leg_top_m"] = float(g[(r, h, "y")][pad.T_F] - LEG_TOP)
                        row[f"force_{h}_hand_minus_leg_top_mean_m"] = float(np.mean(g[(r, h, "y")][pad.T_F:]) - LEG_TOP)
                    row["tf_behind"] = bool(g[(r, "behind")][pad.T_F])
                for h in ("outer", "inner"):
                    row[f"t0_{h}_hand_minus_leg_top_m"] = float(g[(r, h, "y")][0] - LEG_TOP)
                ep_rows.append(row)
    gref = pad.geometry(ctx, ctx.ref)
    ref_row = dict(setting="reference", n_episode_robots=2)
    for h in ("outer", "inner"):
        ref_row[f"tf_{h}_hand_minus_leg_top_median_m"] = float(np.median([gref[(r, h, "y")][pad.T_F] - LEG_TOP for r in range(2)]))
        ref_row[f"force_{h}_hand_minus_leg_top_mean_median_m"] = float(np.median([np.mean(gref[(r, h, "y")][pad.T_F:]) - LEG_TOP for r in range(2)]))
    ref_row["tf_behind_count"] = int(sum(bool(gref[(r, "behind")][pad.T_F]) for r in range(2)))
    for name in CONST:
        pert = [x for x in ep_rows if x["setting"] == name and x["kind"] != "nominal"]
        ep1 = [x for x in pert if x["robot"] == 0]
        rt = [x for x in pert if x["reached_t_f"]]
        row = dict(setting=name, n_perturbed=len(ep1), n_completed=sum(bool(x["completed_reference"]) for x in ep1),
                   progress_mean=mean([x["progress"] for x in ep1]), n_fell=sum(bool(x["sonic_fell_any"]) for x in ep1),
                   executed_steps_median=med([x["executed_steps"] for x in ep1]),
                   push_ratio_mean=mean([x["push_ratio"] for x in ep1]), push_ratio_median=med([x["push_ratio"] for x in ep1]),
                   n_push_ratio_defined=sum(x["push_ratio"] is not None for x in ep1),
                   push_distance_mean_m=mean([x["push_distance_m"] for x in ep1]),
                   n_episode_robots_reached_t_f=len(rt))
        for h in ("outer", "inner"):
            row[f"tf_{h}_hand_minus_leg_top_median_m"] = med([x.get(f"tf_{h}_hand_minus_leg_top_m") for x in rt])
            row[f"force_{h}_hand_minus_leg_top_mean_median_m"] = med([x.get(f"force_{h}_hand_minus_leg_top_mean_m") for x in rt])
            row[f"t0_{h}_hand_minus_leg_top_median_m"] = med([x[f"t0_{h}_hand_minus_leg_top_m"] for x in pert])
        row["tf_behind_count"] = sum(bool(x.get("tf_behind")) for x in rt)
        if name != CONST[0]:
            sids = [k for k in by[name] if by[name][k]["kind"] != "nominal"]
            row["paired_completed_only_with_dq"] = sum(bool(by[name][k]["completed_reference"]) and not by[CONST[0]][k]["completed_reference"] for k in sids)
            row["paired_completed_only_with_dq0"] = sum(bool(by[CONST[0]][k]["completed_reference"]) and not by[name][k]["completed_reference"] for k in sids)
            row["paired_progress_diff_mean"] = mean([by[name][k]["progress"] - by[CONST[0]][k]["progress"] for k in sids])
        nom = [x for x in ep_rows if x["setting"] == name and x["kind"] == "nominal" and x["robot"] == 0]
        row.update(nominal_completed=nom[0]["completed_reference"], nominal_progress=nom[0]["progress"], nominal_push_ratio=nom[0]["push_ratio"])
        summ.append(row)
    summ.append(ref_row)
    wcsv(a.out / "const_dq_episode_robot.csv", ep_rows)
    wcsv(a.out / "const_dq_summary.csv", summ)
    (a.out / "run.json").write_text(json.dumps(dict(argv=sys.argv, src=str(a.src), leg_top_local_y_m=LEG_TOP, t_f=pad.T_F,
                                                    reference_sha256=ctx.meta.get("sha256"), fk_check_m=ctx.fk_check_m), indent=1))
    print(json.dumps(dict(c1=c1, const=summ), indent=1, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
