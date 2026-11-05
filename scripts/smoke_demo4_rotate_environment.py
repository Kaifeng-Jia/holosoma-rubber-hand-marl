#!/usr/bin/env python3
"""Run a bounded real-physics smoke test for cooperative Demo 4 rotation."""

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

from holosoma.config_values.marl.g1.demo4_experiment import (  # noqa: E402
    g1_29dof_demo4_rotate_smoke,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402


PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--steps", type=int, default=8)
PARSER.add_argument("--seed", type=int, default=721)
PARSER.add_argument("--output", type=Path, default=None)
ARGS = PARSER.parse_args()
if ARGS.steps < 1:
    PARSER.error("--steps must be positive")

CONFIG = replace(
    g1_29dof_demo4_rotate_smoke,
    training=replace(g1_29dof_demo4_rotate_smoke.training, seed=ARGS.seed),
)
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.agents.mappo.demo4_initialization import (  # noqa: E402
    initialize_demo4_model_bundle,
)
from holosoma.agents.mappo.demo4_runner import Demo4PolicyRunner  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.envs.marl.demo4_rotate_manager import Demo4RotateManager  # noqa: E402
from holosoma.managers.command.terms.demo4_rotate import (  # noqa: E402
    Demo4RotateMotionCommand,
)
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt"
)
EXPECTED_MASS_KG = 20.0
EXPECTED_MATERIAL = (0.5, 0.5, 0.0)


def _physics(env) -> dict[str, object]:
    view = env.simulator._object.root_physx_view
    mass = view.get_masses().reshape(-1)
    material = view.get_material_properties()
    expected_mass = torch.full_like(mass, EXPECTED_MASS_KG)
    expected_material = torch.as_tensor(
        EXPECTED_MATERIAL,
        device=material.device,
        dtype=material.dtype,
    ).expand_as(material)
    if not torch.allclose(mass, expected_mass, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError(f"Demo 4 table mass mismatch: {mass.detach().cpu().tolist()}")
    if not torch.allclose(material, expected_material, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError(
            "Demo 4 table material mismatch: "
            f"{material.detach().cpu().tolist()}"
        )
    return {
        "mass_kg": mass.detach().cpu().tolist(),
        "material_static_dynamic_restitution": material.detach().cpu().tolist(),
        "inertia_kg_m2_row_major": view.get_inertias().detach().cpu().tolist(),
        "com_pose_body_xyzw": view.get_coms().detach().cpu().tolist(),
    }


def main() -> int:
    env = None
    failure: BaseException | None = None
    try:
        torch.manual_seed(ARGS.seed)
        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        if not isinstance(env, Demo4RotateManager):
            raise RuntimeError(f"Unexpected Demo 4 environment type: {type(env)}")
        env.set_is_evaluating()
        observations = env.reset_all()
        command = env.command_manager.get_state("paired_motion_command")
        if not isinstance(command, Demo4RotateMotionCommand):
            raise RuntimeError(f"Unexpected Demo 4 command type: {type(command)}")

        expected_shapes = {
            "actor_obs": (env.num_envs, 2, 154),
            "teammate_obs": (env.num_envs, 2, 4),
            "table_obs": (env.num_envs, 2, 6),
            "critic_obs": (env.num_envs, 527),
        }
        for key, expected in expected_shapes.items():
            if observations[key].shape != expected:
                raise RuntimeError(
                    f"{key} shape mismatch: {tuple(observations[key].shape)} vs {expected}"
                )

        robot_urdf = CONFIG.robot.asset.urdf_file.lower()
        if "rubberhand" not in robot_urdf or "hemisphere" in robot_urdf:
            raise RuntimeError(f"Unexpected Demo 4 robot asset: {CONFIG.robot.asset.urdf_file}")
        table_write_count = command.table_reset_write_count.clone()
        if not torch.all(table_write_count > 0):
            raise RuntimeError("Demo 4 reset did not write the table exactly through reset")

        models = initialize_demo4_model_bundle(
            SOURCE_CHECKPOINT,
            CONFIG.algo.config,
            device=env.device,
        )
        runner = Demo4PolicyRunner(models)
        initial_table_pos = command.simulator_object_pos_w.clone()
        rewards = []
        values = []
        for _ in range(ARGS.steps):
            decision = runner.decide(
                observations,
                update_critic_normalizer=False,
            )
            observations, reward, done, _ = env.step({"actions": decision.actions})
            if reward.shape != (env.num_envs,):
                raise RuntimeError(f"Demo 4 reward must be team-shaped [E], got {reward.shape}")
            if done.shape != (env.num_envs,):
                raise RuntimeError(f"Demo 4 done must be team-shaped [E], got {done.shape}")
            if decision.values.shape != (env.num_envs, 1):
                raise RuntimeError(f"Demo 4 value must be team-shaped [E,1], got {decision.values.shape}")
            if not torch.equal(command.table_reset_write_count, table_write_count):
                raise RuntimeError("Demo 4 command rewrote the table after reset")
            rewards.append(reward.detach().cpu())
            values.append(decision.values.detach().cpu())

        final_table_pos = command.simulator_object_pos_w.clone()
        report = {
            "passed": True,
            "seed": ARGS.seed,
            "steps": ARGS.steps,
            "environment": type(env).__name__,
            "control_hz": 1.0 / float(env.dt),
            "physics_hz": 1.0 / float(env.sim_dt),
            "reference_frames": command.reference.num_frames,
            "reference_fps": command.reference.fps,
            "actor_input_shape": [env.num_envs, 2, 164],
            "critic_input_shape": list(observations["critic_obs"].shape),
            "action_shape": [env.num_envs, 2, 29],
            "team_reward_shape": list(rewards[-1].shape),
            "team_value_shape": list(values[-1].shape),
            "table_reset_write_count": table_write_count.detach().cpu().tolist(),
            "table_rewritten_during_steps": False,
            "table_displacement_m": (
                final_table_pos - initial_table_pos
            ).detach().cpu().tolist(),
            "signed_yaw_progress_radians": (
                command.signed_yaw_progress_radians.detach().cpu().tolist()
            ),
            "robot_urdf": CONFIG.robot.asset.urdf_file,
            "object_urdf": CONFIG.robot.object.object_urdf_path,
            "object_physics": _physics(env),
        }
        text = json.dumps(report, indent=2, sort_keys=True)
        print(text, flush=True)
        if ARGS.output is not None:
            output = ARGS.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n")
        return 0
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(SIMULATION_APP)
        if failure is not None:
            print(f"Demo 4 environment smoke failed: {failure}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
