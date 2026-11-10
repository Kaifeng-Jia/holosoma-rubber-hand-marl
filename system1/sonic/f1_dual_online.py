#!/usr/bin/env python3
"""F1 online record (no gate): SonicDualDriver in the paired env with the object removed.

The task's paired env (same config path as scripts/evaluate_paper_supplement.py, SONIC low
level: hip_pitch PD, frozen scales, q* to PD, TF32 off) is reset to reference frame 0, then
the object is moved 50 m away (it no longer interacts with the robots). Terminations are
reduced to the time limit (Plan5: timeout; CORE4D: reference_horizon) because the object
error would otherwise end the episode at once. Both robots follow their own role reference
for the full clip. Per robot: global / pelvis-relative MPJPE, pelvis xy drift, SONIC fall step.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "holosoma"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def build(case):
    from holosoma.config_types.termination import TerminationManagerCfg
    from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment
    from holosoma.config_values.marl.g1.core4d_smalltable_experiment import g1_29dof_core4d_smalltable_smoke, with_pair_experiment
    from holosoma.config_values.marl.g1.core4d_smalltable_observation import g1_29dof_core4d_smalltable_evaluation_observation
    from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_pull_baseline, g1_29dof_plan5_push_baseline
    from holosoma.config_values.marl.g1.kick_experiment import g1_29dof_plan5_kick_baseline
    from plan5_registered_placement import configure_push_placement
    from system1.scheme_a import with_sonic_pd

    if case in ("push", "pull", "kick"):
        config = dict(push=g1_29dof_plan5_push_baseline, pull=g1_29dof_plan5_pull_baseline, kick=g1_29dof_plan5_kick_baseline)[case]
        keep = "timeout"
    else:
        exp = get_core4d_pair_experiment(dict(chair="chair021", bucket="bucket003", table="smalltable5kg_A")[case])
        config = with_pair_experiment(g1_29dof_core4d_smalltable_smoke, exp)
        if case in ("bucket", "table"):
            config = replace(config, env_class="holosoma.envs.marl.core4d_bucket_manager.Core4DBucketManager",
                             simulator=replace(config.simulator, config=replace(config.simulator.config, enable_object_hand_contact=True)))
        keep = "reference_horizon"
    if keep == "timeout":
        term = TerminationManagerCfg(terms={keep: config.termination.terms[keep]})
    else:
        # CORE4D's env requires both terms; disable joint_bad_tracking for this diagnostic only
        jbt = config.termination.terms["joint_bad_tracking"]
        off = dict(jbt.params, bad_object_pos_threshold=1e9, bad_object_ori_threshold=1e9, bad_ref_ori_threshold=1e9)
        term = TerminationManagerCfg(terms={keep: config.termination.terms[keep], "joint_bad_tracking": replace(jbt, params=off)})
    config = replace(config, observation=g1_29dof_core4d_smalltable_evaluation_observation,
                     training=replace(config.training, num_envs=1, seed=241003, headless=True), termination=term)
    proof = ROOT / "system1/evidence/preflight_push_registered.json" if case == "push" else None
    config, _ = configure_push_placement(config, proof, skill=case, root=ROOT)
    return with_sonic_pd(config), keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    config, keep = build(a.case)
    from holosoma.utils.eval_utils import init_sim_imports

    app = init_sim_imports(config)
    import torch
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app
    from system1.scheme_a import SonicDualDriver

    env = None
    try:
        env = get_class(config.env_class)(get_tyro_env_config(config), device=a.device)
        env.set_is_evaluating()
        command = env.command_manager.get_state("paired_motion_command")
        driver = SonicDualDriver(env, a.device)
        ids = torch.tensor([0], device=env.device)
        n = int(command.reference.num_frames)
        env.reset_envs_idx(ids)
        env._refresh_envs_after_reset(ids)
        if int(command.time_steps[0].item()) != 0:
            raise RuntimeError("reference not at frame 0")
        obj = torch.zeros(1, 13, device=env.device)
        obj[0, :3] = env.simulator.scene.env_origins[0] + torch.tensor([50.0, 50.0, 0.6], device=env.device)
        obj[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)  # xyzw identity
        env.simulator.set_actor_states(["object"], ids, obj)
        env._refresh_envs_after_reset(ids)
        driver.reset()
        rec, fall_step, pd_diff = [], [None, None], 0.0
        for step in range(1, n):
            actions, _ = driver.actions(command)
            env.step({"actions": actions})
            pd_diff = max(pd_diff, driver.check_pd_target() or 0.0)
            tr = {k: v[0].detach().cpu().numpy().copy() for k, v in driver.tracking(command, min(step, n - 1)).items()}
            rec.append(tr)
            for r in range(2):
                if fall_step[r] is None and tr["fall"][r]:
                    fall_step[r] = step
            if step < n - 1 and int(env.episode_length_buf[0].item()) != step:
                raise RuntimeError(f"episode reset at step {step} (termination {keep})")
        st = {k: np.stack([x[k] for x in rec]) for k in rec[0]}
        obj_pos = command.simulator_object_pos_w[0] - env.simulator.scene.env_origins[0]
        res = dict(case=a.case, steps=n - 1, termination_kept=keep, object_moved_to_m=[50.0, 50.0], object_final_xy_m=obj_pos[:2].tolist(),
                   pd_target_minus_qstar_max_abs=pd_diff,
                   per_robot=[dict(robot=r, fell=fall_step[r] is not None, fall_step=fall_step[r],
                                   mpjpe_g_mm=float(st["mpjpe_mm"][:, r].mean()),
                                   mpjpe_pelvis_rel_mm=float(st["mpjpe_pelvis_rel_mm"][:, r].mean()),
                                   pelvis_xy_drift_final_mm=float(st["pelvis_dxy_m"][-1, r] * 1000),
                                   pelvis_xy_drift_max_mm=float(st["pelvis_dxy_m"][:, r].max() * 1000)) for r in range(2)])
        np.savez_compressed(a.out / "tracking.npz", **st)
        (a.out / "result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return 0
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(app)


if __name__ == "__main__":
    raise SystemExit(main())
