#!/usr/bin/env python3
"""T14 front-back training task (Kevin 2026-11-26), local smoke on the laptop 5070 before "可以开卡": package
results_system1/T14_frontback/smoke/ from logs_system1/t14/frontback_train/smoke (run_smoke.sh + run_smoke_part2.sh):
  * isaac_checks.csv: the three in-simulation checks (8 environments; observation v1 vs definitions, contact replay vs
    definitions; noise off twice, noise on), same columns as results_system1/T13v4_prep/t14ref/isaac_checks_t14ref.csv;
  * smoke_train.json: the 200-iteration cloud-command smoke (cloud_smoke.py --run d_frontback, 2048 environments), s/it,
    non-finite keys, the cloud validation command on its last checkpoint;
  * t10_gate.json: check_sim2sim_v1.py summary (initial states, observation v1 per dimension, threshold 1e-4) and its inputs;
  * isaac_finals.csv / mujoco_pm2.csv: per level, episodes, completed, steps (pipeline health only: a 200-iteration policy);
  * records/: scripts, launch records, logs renamed *_log.txt (.gitignore has *.log).
Read only on the logs; writes the package dir only.

    python3 system1/t14/frontback_train/package_smoke.py
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SM = ROOT / "logs_system1/t14/frontback_train/smoke"
OUT = ROOT / "results_system1/T14_frontback/smoke"
RUN = "d_frontback_smoke"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_row(name, p):
    r = json.loads(p.read_text())
    o, cr = r["obs"], r.get("contact_replay") or {}
    e = cr.get("max_abs_err") or {}
    return dict(check=name, reference="T14 front-back sub8_suitcase_020 1.0", actor=o["group_shapes"]["actor_obs"], teammate=o["group_shapes"]["teammate_obs"],
                critic=o["group_shapes"]["critic_obs"], critic_both_commands=(o.get("critic_shared_motion_command") or {}).get("equals_both_robots"),
                obs_v1_err_ref_ori=o["obs_v1_max_abs_err"]["ref_ori"], obs_v1_err_teammate=o["obs_v1_max_abs_err"]["teammate"],
                noise=o["noise_enabled"],
                noise_in_range=all(v["within_range"] for v in o.get("noise_samples", {}).values() if isinstance(v, dict)) if o.get("noise_samples") else None,
                clip_partner=(o.get("clip_fraction") or {}).get("partner"), clip_object=(o.get("clip_fraction") or {}).get("object"),
                replay_frames=cr.get("frames"), replay_reward_err=e.get("reward"), replay_d_err=e.get("d"), replay_penalty_count_err=e.get("count"),
                replay_lost_mismatch=e.get("lost_mismatch"), replay_terminate_mismatch=e.get("terminate_mismatch"),
                hand_force_sensor_diff_n=cr.get("hand_force_link_vs_hand_sensor_max_abs_n"),
                diag_should_fraction=cr.get("should_fraction"), diag_mean_team_reward=cr.get("mean_team_reward"),
                diag_hand_force_over_1n_in_should=cr.get("hand_force_over_1n_in_should"),
                diag_lost_fraction_in_should=cr.get("lost_fraction_in_should"), diag_frames_terminating=cr.get("frames_terminating"),
                diag_frames_with_penalty=cr.get("frames_with_penalty"), initial_time_steps=" ".join(map(str, r.get("initial_time_steps", []))))


def episodes(p):
    ep = json.loads(Path(p).read_text())
    steps = [e["steps"] for e in ep]
    return dict(episodes=len(ep), completed=sum(bool(e.get("completed_reference")) for e in ep), steps_min=min(steps), steps_max=max(steps),
                nominal_steps=next(e["steps"] for e in ep if e["state_id"] == "nominal"),
                terminal_terms_nominal=next(e.get("terminal_terms") for e in ep if e["state_id"] == "nominal"))


def write_csv(p, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    rec = OUT / "records"
    rec.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ("noise_off", "noise_off_r2", "noise_on"):
        p = SM / "checks" / n / "isaac_checks.json"
        rows.append(check_row(n, p))
        shutil.copy2(p, rec / f"isaac_checks_{n}.json")
        shutil.copy2(SM / "checks" / n / "contact_replay_rows.jsonl", rec / f"contact_replay_rows_{n}.jsonl")
        shutil.copy2(SM / "checks" / f"{n}.log", rec / f"check_{n}_log.txt")
    a, b = (json.loads((rec / f"isaac_checks_{n}.json").read_text()) for n in ("noise_off", "noise_off_r2"))
    rows[1]["equal_to_noise_off"] = a == b
    write_csv(OUT / "isaac_checks.csv", rows)

    sj = SM / "train/d_frontback/smoke.json"
    s = json.loads(sj.read_text())
    st = s["stages"][-1]
    ck = ROOT / st["last_checkpoint"]
    val = s["validation"]
    smoke = dict(run=s["run"], passed=s["passed"], num_envs=s["num_envs"], iterations=st["iterations"], train_seconds=st["seconds"],
                 s_per_it_median_after_3=st["s_per_it_median_after_3"], nonfinite_keys=st["nonfinite_keys"], overflow_lines=st["overflow_lines"],
                 train_argv=st["argv"], last_checkpoint=st["last_checkpoint"], last_checkpoint_sha256=sha(ck),
                 validation=dict(argv=val["argv"], returncode=val["returncode"], seconds=val["seconds"], complete=val["complete"], episodes=val["episodes"],
                                 completed_non_nominal=val["completed_non_nominal"], states_sha256=val["states_sha256"]),
                 plan_rules=s["plan"]["rules"] if isinstance(s.get("plan"), dict) else None,
                 fixed_budget=s["plan"].get("fixed_budget") if isinstance(s.get("plan"), dict) else None,
                 git_head=s.get("git_head"), git_status=s.get("git_status"))
    (OUT / "smoke_train.json").write_text(json.dumps(smoke, indent=1) + "\n")
    shutil.copy2(sj, rec / "smoke.json")
    for n in ("train.stdout.log", "validation.stdout.log"):
        txt = (SM / "train/d_frontback" / n).read_text(errors="replace").replace("\r", "\n").splitlines()
        (rec / n.replace(".log", "_tail_log.txt")).write_text("\n".join(txt[-200:]) + "\n")
    md = SM / "train/d_frontback/train"
    for n in ("run_config.json", "status.json"):
        if (md / n).exists():
            shutil.copy2(md / n, rec / f"train_{n}")

    g = json.loads((ROOT / f"logs_system1/t13v4/sim2sim/checks_{RUN}/summary.json").read_text())
    (OUT / "t10_gate.json").write_text(json.dumps(dict(run=RUN, initial_states=g["initial_states"], observations=g["observations"], inputs=g["inputs"]), indent=1) + "\n")

    fin = []
    for lv in (2, 8):
        d = ROOT / f"logs_system1/t13v4/finals/{RUN}_pm{lv}"
        sm = json.loads((d / "summary.json").read_text())
        fin.append(dict(level=f"pm{lv}", complete=sm["complete"], checkpoint_sha256=sm["checkpoint_sha256"], states_sha256=sm["states_sha256"],
                        object_urdf=sm["object_urdf"], reference_frames=sm["reference_frames"], **episodes(d / "episodes.json")))
        lj = ROOT / f"logs_system1/t13v4/finals/{RUN}_pm{lv}.launch.json"
        if lj.exists():
            shutil.copy2(lj, rec / lj.name)
    write_csv(OUT / "isaac_finals.csv", fin)
    d = ROOT / f"logs_system1/t13v4/mj_finals/{RUN}_pm2"
    sm = json.loads((d / "summary.json").read_text())
    write_csv(OUT / "mujoco_pm2.csv", [dict(level="pm2", onnx_sha256=sm["onnx_sha256"], checkpoint_sha256=sm["checkpoint_sha256"], states_sha256=sm["states_sha256"],
                                            scene_json_sha256=sm["scene_json_sha256"], pd=sm["pd"], **episodes(d / "episodes.json"))])
    for n in ("run_smoke.sh", "run_smoke_part2.sh"):
        shutil.copy2(SM / n, rec / n)
    for n in ("run_smoke.log", "run_smoke_part2.log", "train_smoke.log", "finals_pm2.log", "finals_pm8.log", "dump.log", "onnx.log", "checks_sim2sim.log", "mujoco_pm2.log"):
        if (SM / n).exists():
            txt = (SM / n).read_text(errors="replace").replace("\r", "\n").splitlines()
            (rec / n.replace(".log", "_log.txt")).write_text("\n".join(txt[-300:]) + "\n")
    print(json.dumps(dict(checks=[(r["check"], r["obs_v1_err_ref_ori"], r["replay_reward_err"]) for r in rows], smoke=dict(passed=smoke["passed"], s_per_it=smoke["s_per_it_median_after_3"]),
                          t10=dict(init=g["initial_states"]["all_passed"], obs=g["observations"]["obs_passed"], obs_max=g["observations"]["obs_max_abs_diff"]),
                          finals=[(f["level"], f["episodes"], f["completed"]) for f in fin]), indent=1))


if __name__ == "__main__":
    main()
