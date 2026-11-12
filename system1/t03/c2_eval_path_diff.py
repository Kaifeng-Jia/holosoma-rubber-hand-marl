#!/usr/bin/env python3
"""T03 C v2 check 3b (Kevin 2026-11-12): evaluation path of the C v2 evaluations vs D on the same task, +/-2.

Pairs (C v2 evaluation -> D same-protocol evaluation of the paper main result, seed 721, same evaluator):
  pull   C-in it1050 (cloud 4090; and the laptop recheck when present)  vs  D paper it1050 and it8050 (laptop), and D trained
         from scratch like C Pull (V3-25, seed 721) it1050 and it8050 (laptop)
  bucket C-in it1000 (cloud; and the laptop recheck)                    vs  D paper it2000 and it12000 (laptop)
  table  C-in it1000 (cloud)                                            vs  D paper it12000 (laptop)
For every pair: run_manifest.json key by key (equal / different, both values abbreviated). Then the named items:
  * residual kind: system1_scheme_c_kind (C) / controller (D: policy, no residual);
  * checkpoint sha256: the manifest value, recomputed from the file when it is on this machine (C: the cloud file's sha256
    from the copy, exports_system1/train_partial/SHA256SUMS.local); D: also its paper-registry prefix (post_train.D_CURVES);
  * initial-state file sha256: manifest value vs the frozen file (post_train.STATES);
  * termination thresholds: the manifest does not record them; the evaluator builds its config from --case (and the controller
    only through with_sonic_pd, PD gains), so the config is rebuilt here with the evaluator's own calls (no simulator) for
    (case, controller) and config.termination compared term by term;
  * deterministic action: the manifest's inference field and the evaluator code path (both call
    deterministic_core4d_smalltable_actions, the actor mean; C then passes it to SonicResidualEnv.step);
  * TF32: the manifest's tf32 field (C); D's manifest has no tf32 field: the policy path never sets it (only
    system1/sonic_controller.py sets allow_tf32 = self.tf32, False in evaluation), PyTorch's default is off;
  * evaluator code: git head of each launch and git diff of scripts/evaluate_paper_supplement.py between them.

    hssim_s1/bin/python -B system1/t03/c2_eval_path_diff.py --out results_system1/T03/c2_checks/b_eval_path
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src/holosoma", ROOT / "src/holosoma_retargeting", ROOT / "scripts", ROOT):
    sys.path.insert(0, str(p))
from system1.t03.post_train import D_CURVES, STATES, sha  # noqa: E402

CL = ROOT / "logs_system1/cloud_evals"
PT = ROOT / "logs_system1/post_train"
C = {"pull": ("cin_pull_m20kg_seed721_joint8000", 1050), "bucket": ("cin_bucket_m1kg_seed721_fresh12000", 1000),
     "table": ("cin_table_m5kg_seed721_fresh12000", 1000)}
D = {"pull": [("D_pull_paper_seed721", (1050, 8050)), ("D_pull_scratch_V3-25_seed721", (1050, 8050))],
     "bucket": [("D_bucket_paper_B-RZ0_seed721", (2000, 12000))], "table": [("D_table_paper_T-07_seed721", (12000,))]}
EXPECTED_DIFF = {"model_id", "checkpoint", "checkpoint_sha256", "checkpoint_iteration", "training_contract", "inference", "gpu_name",
                 "controller", "tf32", "hip_pitch_actuator", "frozen_action_scales", "effort_limits", "vel_limits", "armature",
                 "pd_target", "sonic_spec", "sonic_spec_sha256", "sonic_pd", "sonic_fall_rule", "sonic_checkpoint_sha256",
                 "system1_scheme_c_kind"}


def build_config(case, controller):
    """The evaluator's config construction (scripts/evaluate_paper_supplement.py lines 38-75) without the simulator."""
    from holosoma.config_values.marl.g1.core4d_smalltable_experiment import g1_29dof_core4d_smalltable_smoke, with_pair_experiment
    from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment
    from holosoma.config_values.marl.g1.core4d_smalltable_observation import g1_29dof_core4d_smalltable_evaluation_observation
    from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_pull_baseline
    from plan5_registered_placement import configure_push_placement
    if case == "pull":
        config = g1_29dof_plan5_pull_baseline
    else:
        from holosoma.config_values.marl.g1.v3_reference_variant import with_reference_variant
        experiment = with_reference_variant(get_core4d_pair_experiment(dict(bucket="bucket003", table="smalltable5kg_A")[case]), None)
        config = with_pair_experiment(g1_29dof_core4d_smalltable_smoke, experiment)
        config = replace(config, env_class="holosoma.envs.marl.core4d_bucket_manager.Core4DBucketManager",
                         simulator=replace(config.simulator, config=replace(config.simulator.config, enable_object_hand_contact=True)))
    config = replace(config, observation=g1_29dof_core4d_smalltable_evaluation_observation,
                     training=replace(config.training, num_envs=1, seed=241003, headless=True))
    config, _ = configure_push_placement(config, None, skill=case, root=ROOT)
    if controller in ("sonic", "sonic_residual", "sonic_planner"):
        from system1.scheme_a import with_sonic_pd
        config = with_sonic_pd(config, hip_pitch_actuator="ours")
    return config


def terms(config):
    return {k: dict(func=v.func, params=v.params, is_timeout=v.is_timeout) for k, v in config.termination.terms.items()}


def short(v, n=160):
    s = json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= n else s[:n] + f"...(len {len(s)})"


def launch_of(d: Path):
    for f in (d.with_name(d.name + ".launch.json"),
              CL / "ops2" / (str(d.relative_to(CL)).replace("/", "__") + ".launch.json") if CL in d.parents else None):
        if f is not None and f.exists():
            return json.loads(f.read_text())
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    term_cfg = {f"{c}/{ctl}": terms(build_config(c, ctl)) for c in C for ctl in ("policy", "sonic_residual")}
    (a.out / "termination_config.json").write_text(json.dumps(term_cfg, indent=1, default=str))
    diff_rows, checks = [], []
    for case, (label, it) in C.items():
        cdirs = [("C-in cloud", CL / "c2_curve" / label / f"it{it:05d}_pm2")]
        r = PT / "recheck" / label / f"it{it:05d}_pm2"
        if (r / "summary.json").exists():
            cdirs.append(("C-in laptop recheck", r))
        ddirs = [(f"{dl.split('_seed')[0]} it{i}", dl, PT / "d_curve" / dl / f"it{i:05d}_pm2") for dl, dits in D[case] for i in dits]
        for ctag, cd in cdirs:
            cm = json.loads((cd / "run_manifest.json").read_text())
            cl = launch_of(cd)
            local_ck = ROOT / "exports_system1/train_partial" / label / f"model_{it:05d}.pt"
            for dtag, dl, dd in ddirs:
                dm = json.loads((dd / "run_manifest.json").read_text())
                dlch = launch_of(dd)
                pair = f"{case}: {ctag} vs {dtag}"
                for k in sorted(set(cm) | set(dm)):
                    cv, dv = cm.get(k, "<absent>"), dm.get(k, "<absent>")
                    diff_rows.append(dict(pair=pair, key=k, equal=cv == dv, expected_to_differ=k in EXPECTED_DIFF,
                                          c_value=short(cv), d_value=short(dv)))
                unexpected = [k for k in set(cm) | set(dm) if cm.get(k, "<absent>") != dm.get(k, "<absent>") and k not in EXPECTED_DIFF]
                st_path, st_sha = STATES[(case, 2)]
                dprefix = D_CURVES[dl][2]
                d_final = max(int(x.name[2:7]) for x in (PT / "d_curve" / dl).glob("it*_pm2") if x.is_dir() and "attempt" not in x.name)
                eh = [(cl or {}).get("git_head", ""), (dlch or {}).get("git_head", "")]
                gd = subprocess.run(["git", "diff", "--stat", eh[0], eh[1], "--", "scripts/evaluate_paper_supplement.py", "system1/scheme_c.py",
                                     "system1/scheme_a.py", "system1/sonic_controller.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip() \
                    if all(eh) else "launch record missing"
                tc, td = term_cfg[f"{case}/sonic_residual"], term_cfg[f"{case}/policy"]
                checks += [
                    dict(pair=pair, item="controller / residual kind", c=f"{cm['controller']} / {cm.get('system1_scheme_c_kind')}",
                         d=f"{dm.get('controller', '<absent: policy>')} / -", verdict="C-in residual vs D policy (as designed)"),
                    dict(pair=pair, item="checkpoint sha256 (manifest)", c=cm["checkpoint_sha256"], d=dm["checkpoint_sha256"],
                         verdict="C: equals local copy of the cloud file" if local_ck.exists() and sha(local_ck) == cm["checkpoint_sha256"] else
                         "C: local copy absent or different",
                         note=("D: no paper-registry value for this run (V3 scratch run; copied from the cloud, sha256 checked at copy)"
                               if dprefix is None else
                               (f"D: paper registry prefix {dprefix} applies to the final checkpoint it{d_final}; "
                                f"this D checkpoint it{dm['checkpoint_iteration']} "
                                + ("matches" if dm["checkpoint_iteration"] != d_final or dm["checkpoint_sha256"].startswith(dprefix) else "DOES NOT match")
                                + ("" if dm["checkpoint_iteration"] == d_final else " (not the final: no registry value)")))),
                    dict(pair=pair, item="initial-state file sha256", c=cm["states_sha256"], d=dm["states_sha256"],
                         verdict="equal, = frozen file" if cm["states_sha256"] == dm["states_sha256"] == st_sha == sha(st_path) else "DIFFERENT"),
                    dict(pair=pair, item="termination thresholds (rebuilt config)", c=short(tc, 400), d=short(td, 400),
                         verdict="identical" if tc == td else "DIFFERENT"),
                    dict(pair=pair, item="deterministic action", c=cm["inference"], d=dm["inference"],
                         verdict="both actor mean (deterministic_core4d_smalltable_actions); C then SonicResidualEnv.step"),
                    dict(pair=pair, item="TF32", c=str(cm.get("tf32", "<absent>")), d=str(dm.get("tf32", "<absent>")),
                         verdict="C off (manifest); D: not recorded, policy path never enables it (PyTorch default off)"),
                    dict(pair=pair, item="observation noise / horizon / endpoint rule",
                         c=f"{cm['observation_noise']} / {cm['horizon_control_steps']} / {cm['endpoint_rule'][:30]}",
                         d=f"{dm['observation_noise']} / {dm['horizon_control_steps']} / {dm['endpoint_rule'][:30]}",
                         verdict="equal" if (cm["observation_noise"], cm["horizon_control_steps"], cm["endpoint_rule"]) ==
                         (dm["observation_noise"], dm["horizon_control_steps"], dm["endpoint_rule"]) else "DIFFERENT"),
                    dict(pair=pair, item="robot / object URDF, object physics read back",
                         c=f"{cm['robot_urdf']} / {Path(cm['object_urdf']).name}", d=f"{dm['robot_urdf']} / {Path(dm['object_urdf']).name}",
                         verdict="equal" if (cm["robot_urdf"], cm["object_urdf"], cm["physics"]) == (dm["robot_urdf"], dm["object_urdf"], dm["physics"]) else "DIFFERENT"),
                    dict(pair=pair, item="GPU", c=cm.get("gpu_name"), d=dm.get("gpu_name"), verdict="same" if cm.get("gpu_name") == dm.get("gpu_name") else "different GPU model"),
                    dict(pair=pair, item="evaluator code (git heads, diff of evaluator / scheme_c / scheme_a / sonic_controller)",
                         c=eh[0] if eh else None, d=eh[1] if len(eh) > 1 else None, verdict="no diff" if gd == "" else gd),
                    dict(pair=pair, item="other manifest keys that differ", c="", d="", verdict=", ".join(sorted(unexpected)) or "none"),
                ]
    with (a.out / "manifest_diff.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diff_rows[0]))
        w.writeheader()
        w.writerows(diff_rows)
    with (a.out / "checks.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "item", "c", "d", "verdict", "note"])
        w.writeheader()
        w.writerows(checks)
    for c in checks:
        print(c["pair"], "|", c["item"], "|", c["verdict"], "|", c.get("note", ""))


if __name__ == "__main__":
    raise SystemExit(main())
