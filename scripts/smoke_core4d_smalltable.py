#!/usr/bin/env python3
"""Run bounded physics and optional one-update checks for CORE4D small-table MAPPO."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--num-envs", type=int, default=1)
PARSER.add_argument("--steps", type=int, default=8)
PARSER.add_argument("--seed", type=int, default=721)
PARSER.add_argument("--ppo-update", action="store_true")
PARSER.add_argument("--output", type=Path, default=None)
ARGS = PARSER.parse_args()
if ARGS.num_envs < 1 or ARGS.steps < 1:
    PARSER.error("--num-envs and --steps must be positive")

from holosoma.agents.mappo.core4d_smalltable_initialization import (  # noqa: E402
    initialize_core4d_smalltable_model_bundle,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (  # noqa: E402
    Core4DSmallTablePPO,
)
from holosoma.agents.mappo.core4d_smalltable_runner import (  # noqa: E402
    Core4DSmallTablePolicyRunner,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (  # noqa: E402
    g1_29dof_core4d_smalltable_smoke,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402


CONFIG = replace(
    g1_29dof_core4d_smalltable_smoke,
    training=replace(
        g1_29dof_core4d_smalltable_smoke.training,
        num_envs=ARGS.num_envs,
        seed=ARGS.seed,
    ),
)
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.managers.command.terms.core4d_smalltable import (  # noqa: E402
    Core4DSmallTableMotionCommand,
    object_origin_velocity_to_com_velocity,
)
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


EXPECTED_MASS_KG = 20.0
EXPECTED_MATERIAL = (0.5, 0.5, 0.0)


def main() -> int:
    env = None
    try:
        torch.manual_seed(ARGS.seed)
        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        env.set_is_evaluating()
        observations = env.reset_all()
        command = env.command_manager.get_state("paired_motion_command")
        if not isinstance(command, Core4DSmallTableMotionCommand):
            raise RuntimeError(f"Unexpected command type: {type(command)}")

        expected_shapes = {
            "actor_obs": (env.num_envs, 2, 154),
            "teammate_obs": (env.num_envs, 2, 4),
            "critic_obs": (env.num_envs, 527),
        }
        for name, expected in expected_shapes.items():
            if observations[name].shape != expected:
                raise RuntimeError(
                    f"{name} shape mismatch: {tuple(observations[name].shape)} vs {expected}"
                )
        if command.reference.fps != 50:
            raise RuntimeError(f"Runtime FPS must be 50, got {command.reference.fps}")

        # Verify the exact reset write before taking another physics step.
        env_ids = torch.arange(env.num_envs, device=env.device)
        command.time_steps.zero_()
        env.reset_envs_idx(env_ids)
        env._refresh_envs_after_reset(env_ids)
        object_state = env.simulator.all_root_states[command.object_indices_in_simulator]
        sample = command.reference.sample(command.time_steps)
        expected_com_velocity = object_origin_velocity_to_com_velocity(
            sample["object_lin_vel_w"],
            sample["object_ang_vel_w"],
            sample["object_quat_w"],
            command.object_com_position_b,
        )
        torch.testing.assert_close(
            object_state[:, 7:10], expected_com_velocity, rtol=0.0, atol=1.0e-5
        )
        torch.testing.assert_close(
            command.simulator_object_lin_vel_w,
            sample["object_lin_vel_w"],
            rtol=0.0,
            atol=1.0e-5,
        )
        torch.testing.assert_close(
            object_state[:, 10:13], sample["object_ang_vel_w"], rtol=0.0, atol=1.0e-5
        )
        env._compute_observations()
        observations = env.obs_buf_dict

        view = env.simulator._object.root_physx_view
        mass = view.get_masses().reshape(-1)
        material = view.get_material_properties()
        torch.testing.assert_close(
            mass,
            torch.full_like(mass, EXPECTED_MASS_KG),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            material,
            torch.as_tensor(
                EXPECTED_MATERIAL,
                dtype=material.dtype,
                device=material.device,
            ).expand_as(material),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

        ppo_config = replace(CONFIG.algo.config, num_steps_per_env=ARGS.steps)
        models = initialize_core4d_smalltable_model_bundle(
            ppo_config,
            device=env.device,
        )
        runner = Core4DSmallTablePolicyRunner(models)
        decision = runner.decide(observations)
        if decision.actions.shape != (env.num_envs, 2, 29):
            raise RuntimeError(f"Unexpected action shape: {tuple(decision.actions.shape)}")

        update_metrics = None
        if ARGS.ppo_update:
            learner = Core4DSmallTablePPO(
                models,
                ppo_config,
                num_envs=env.num_envs,
                num_steps_per_env=ARGS.steps,
                device=env.device,
            )
            observations = learner.collect_rollout(env, observations)
            update_metrics = learner.update().__dict__
        else:
            for _ in range(ARGS.steps):
                decision = runner.decide(observations)
                observations, _, _, _ = env.step({"actions": decision.actions})

        robot_urdf = CONFIG.robot.asset.urdf_file.lower()
        if "rubberhand" not in robot_urdf or "hemisphere" in robot_urdf:
            raise RuntimeError(f"Unexpected robot asset: {CONFIG.robot.asset.urdf_file}")
        report = {
            "passed": True,
            "num_envs": env.num_envs,
            "steps": ARGS.steps,
            "ppo_update": ARGS.ppo_update,
            "ppo_update_metrics": update_metrics,
            "reference_frames": command.reference.num_frames,
            "reference_fps": command.reference.fps,
            "reference_non_looping": True,
            "actor_input_shape": [env.num_envs, 2, 158],
            "critic_input_shape": [env.num_envs, 527],
            "action_shape": [env.num_envs, 2, 29],
            "object_angular_velocity_reset_verified": True,
            "object_origin_com_velocity_conversion_verified": True,
            "table_reset_write_count": command.table_reset_write_count.cpu().tolist(),
            "physics_hz": 1.0 / float(env.sim_dt),
            "control_hz": 1.0 / float(env.dt),
            "mass_kg": mass.cpu().tolist(),
            "material_static_dynamic_restitution": material.cpu().tolist(),
            "robot_urdf": CONFIG.robot.asset.urdf_file,
            "object_urdf": CONFIG.robot.object.object_urdf_path,
        }
        text = json.dumps(report, indent=2, sort_keys=True)
        print(text, flush=True)
        if ARGS.output is not None:
            output = ARGS.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n")
        return 0
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(SIMULATION_APP)


if __name__ == "__main__":
    raise SystemExit(main())
