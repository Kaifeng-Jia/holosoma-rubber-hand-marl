#!/usr/bin/env python3
"""T06 step 1 / T07: load the G1 + two Sharpa hands in the holosoma IsaacSim pair environment and hold the reference
frame-0 pose for 5 s (GPU PhysX, 1 environment, no policy).

Scene = the evaluator's CORE4D pair setup for --case (bucket / table) with the robot replaced by
system1/hand/robot_config.py:with_sharpa_hands and the motion reference by a 73-joint Sharpa runtime reference
(system1/hand/sharpa_reference.py). The object-hand contact sensor is off here (it names the rubber-hand links).
After reset to frame 0 the PD target of every DOF is the frame-0 reference joint position (joint_target_override, the
System1 hook of the action term) for --seconds; recorded: DOF / body lists, masses (robot read back from PhysX),
joint limits, PD gains, action scales, torque limits as the environment sees them, pelvis height and tilt per step,
max finger-joint tracking error, whether a robot fell (pelvis below 0.5 m or tilt > 60 deg).

    hssim_s1/bin/python -B system1/hand/load_test.py --case bucket --reference logs_system1/t06/reference/test_A_track.npz \
        --out logs_system1/t06/load_test/bucket
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/holosoma"))
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--case", choices=("bucket", "table"), default="bucket")
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seconds", type=float, default=5.0)
p.add_argument("--rubber", action="store_true", help="control run: the original rubber-hand robot and reference")
p.add_argument("--device", default="cuda:0")
a = p.parse_args()

from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment  # noqa: E402
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import g1_29dof_core4d_smalltable_smoke, with_pair_experiment  # noqa: E402
from holosoma.config_values.marl.g1.core4d_smalltable_observation import g1_29dof_core4d_smalltable_evaluation_observation  # noqa: E402
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402
from system1.hand.robot_config import with_sharpa_hands  # noqa: E402

experiment = get_core4d_pair_experiment(dict(bucket="bucket003", table="smalltable5kg_A")[a.case])
config = with_pair_experiment(g1_29dof_core4d_smalltable_smoke, experiment)
config = replace(config, observation=g1_29dof_core4d_smalltable_evaluation_observation,
                 training=replace(config.training, num_envs=1, seed=241003, headless=True))
ref_path = a.reference.resolve()
if not a.rubber:
    config = replace(config, robot=with_sharpa_hands(config.robot))

    def remap(terms):
        out = dict(terms)
        t = out["paired_motion_command"]
        prm = dict(t.params)
        prm["motion_config"] = replace(prm["motion_config"], motion_file=str(ref_path))
        prm["paired_reference_file"] = str(ref_path)
        out["paired_motion_command"] = replace(t, params=prm)
        return out
    config = replace(config, command=replace(config.command, setup_terms=remap(config.command.setup_terms),
                                             reset_terms=remap(config.command.reset_terms), step_terms=remap(config.command.step_terms)))
app = init_sim_imports(config)

import torch  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    env = None
    t0 = time.time()
    try:
        a.out.mkdir(parents=True, exist_ok=True)
        env = get_class(config.env_class)(get_tyro_env_config(config), device=a.device)
        env.set_is_evaluating()
        command = env.command_manager.get_state("paired_motion_command")
        sim = env.simulator
        ids = torch.tensor([0], device=env.device)
        env.reset_envs_idx(ids)
        env._refresh_envs_after_reset(ids)
        assert int(command.time_steps[0].item()) == 0
        term = env.action_manager.get_term("joint_control")
        sample = command.reference.sample(torch.tensor([0], device=env.device))
        q0 = sample["agent_joint_pos"].clone()                      # [1, agent, dof] in env.dof_names order
        term.joint_target_override = q0
        dof_names = list(env.dof_names)
        fing = [i for i, n in enumerate(dof_names) if any(f in n for f in ("thumb", "index", "middle", "ring", "pinky"))]
        rv = sim._robot.root_physx_view
        masses = rv.get_masses()[0].detach().cpu().numpy()
        lim = sim._robot.data.joint_pos_limits[0].detach().cpu().numpy() if hasattr(sim._robot.data, "joint_pos_limits") else None
        steps = int(round(a.seconds / env.dt))
        log = []
        zero = torch.zeros(env.num_envs, term.action_dim, device=env.device)
        for k in range(steps):
            env.step({"actions": zero})
            rs = sim.agent_root_states[0].detach().cpu().numpy()  # [agent, 13]
            up = []
            for ag in range(rs.shape[0]):
                x, y, z, w = rs[ag, 3:7]
                up.append(float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))))
            dq = (sim.agent_dof_pos[0] - q0[0]).abs().detach().cpu().numpy()
            log.append(dict(step=k + 1, pelvis_z=rs[:, 2].tolist(), tilt_deg=up, finger_err_max=float(dq[:, fing].max()) if fing else None,
                            body_err_max=float(dq[:, [i for i in range(len(dof_names)) if i not in fing]].max())))
        fell = [any(l["pelvis_z"][ag] < 0.5 or l["tilt_deg"][ag] > 60 for l in log) for ag in range(2)]
        res = dict(case=a.case, rubber=a.rubber, reference=str(ref_path), reference_sha256=sha(ref_path), seconds=a.seconds, dt=env.dt, steps=steps,
                   num_dof=len(dof_names), dof_names=dof_names, body_names=list(sim.body_names), num_bodies=len(sim.body_names),
                   robot_mass_kg_physx=float(masses.sum()), body_masses={n: float(m) for n, m in zip(sim._robot.body_names, masses)},
                   p_gains=env.p_gains.reshape(-1).tolist()[:len(dof_names)], d_gains=env.d_gains.reshape(-1).tolist()[:len(dof_names)],
                   action_scales=env.action_scales.reshape(-1).tolist()[:len(dof_names)],
                   torque_limits=env.torque_limits.reshape(-1).tolist()[:len(dof_names)],
                   joint_pos_limits=None if lim is None else lim.tolist(), fell=fell,
                   pelvis_z_first_last=[log[0]["pelvis_z"], log[-1]["pelvis_z"]], tilt_max_deg=[max(l["tilt_deg"][ag] for l in log) for ag in range(2)],
                   finger_err_max_last=log[-1]["finger_err_max"], body_err_max_last=log[-1]["body_err_max"],
                   robot_urdf=config.robot.asset.urdf_file, gpu=torch.cuda.get_device_name(0), wall_s=time.time() - t0)
        (a.out / "load_test.json").write_text(json.dumps(res, indent=1))
        (a.out / "per_step.json").write_text(json.dumps(log))
        print(json.dumps({k: v for k, v in res.items() if k not in ("dof_names", "body_names", "body_masses", "p_gains", "d_gains", "action_scales",
                                                                     "torque_limits", "joint_pos_limits")}), flush=True)
        return 0
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(app)


if __name__ == "__main__":
    raise SystemExit(main())
