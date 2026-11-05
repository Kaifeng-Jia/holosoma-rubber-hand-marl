#!/usr/bin/env python3
"""Run a bounded real-physics smoke test for isolated competitive Demo 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma.config_values.marl.g1.demo3_experiment import (  # noqa: E402
    g1_29dof_demo3_tug_smoke,
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
    g1_29dof_demo3_tug_smoke,
    training=replace(g1_29dof_demo3_tug_smoke.training, seed=ARGS.seed),
)
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.agents.mappo.demo3_initialization import (  # noqa: E402
    initialize_demo3_model_bundle,
)
from holosoma.agents.mappo.demo3_runner import Demo3PolicyRunner  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.envs.marl.demo3_tug_manager import Demo3TugManager  # noqa: E402
from holosoma.managers.command.terms.demo3_tug import Demo3TugMotionCommand  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


CHECKPOINT = REPO_ROOT / "logs/Demo3Tug/checkpoints/model_07999_actor164_table_neutral.pt"
CHECKPOINT_SHA256 = "048f952cad01d5fda42851502347ee626751dccab923905af303eb44807ef01b"
EXPECTED_MASS_KG = 20.0
EXPECTED_COM_M = (0.0, -0.012413473401914, 0.0)
EXPECTED_INERTIA = (
    0.821972006993735,
    0.0,
    0.0,
    0.0,
    1.206183371508506,
    0.0,
    0.0,
    0.0,
    0.821972006993735,
)
EXPECTED_MATERIAL = (0.5, 0.5, 0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physics(env) -> dict[str, object]:
    view = env.simulator._object.root_physx_view
    mass = view.get_masses().reshape(-1)
    inertia = view.get_inertias()
    com = view.get_coms()[..., :3]
    material = view.get_material_properties()
    checks = {
        "mass": (mass, torch.full_like(mass, EXPECTED_MASS_KG)),
        "center_of_mass": (
            com,
            torch.as_tensor(EXPECTED_COM_M, device=com.device, dtype=com.dtype).expand_as(com),
        ),
        "inertia": (
            inertia,
            torch.as_tensor(EXPECTED_INERTIA, device=inertia.device, dtype=inertia.dtype).expand_as(inertia),
        ),
        "material": (
            material,
            torch.as_tensor(EXPECTED_MATERIAL, device=material.device, dtype=material.dtype).expand_as(material),
        ),
    }
    for name, (actual, expected) in checks.items():
        if not torch.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6):
            raise RuntimeError(
                f"Demo 3 {name} mismatch: actual={actual.detach().cpu().tolist()}, "
                f"expected={expected.detach().cpu().tolist()}"
            )
    return {
        "mass_kg": mass.detach().cpu().tolist(),
        "center_of_mass_m": com.detach().cpu().tolist(),
        "inertia_kg_m2": inertia.detach().cpu().tolist(),
        "material_static_dynamic_restitution": material.detach().cpu().tolist(),
    }


def main() -> int:
    env = None
    error: BaseException | None = None
    try:
        torch.manual_seed(ARGS.seed)
        if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
            raise RuntimeError("Demo 3 warm-start checkpoint hash changed")
        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        if not isinstance(env, Demo3TugManager):
            raise RuntimeError(f"Unexpected environment type: {type(env)}")
        env.set_is_evaluating()
        observations = env.reset_all()
        command = env.command_manager.get_state("paired_motion_command")
        if not isinstance(command, Demo3TugMotionCommand):
            raise RuntimeError(f"Unexpected Demo 3 command type: {type(command)}")

        expected_shapes = {
            "actor_obs": (1, 2, 154),
            "teammate_obs": (1, 2, 4),
            "table_obs": (1, 2, 6),
            "critic_obs": (1, 2, 527),
        }
        for key, expected in expected_shapes.items():
            if observations[key].shape != expected:
                raise RuntimeError(
                    f"{key} shape mismatch: {tuple(observations[key].shape)} vs {expected}"
                )
        if not torch.allclose(
            command.agent_pull_axis_w[:, 0],
            -command.agent_pull_axis_w[:, 1],
            atol=1.0e-5,
            rtol=0.0,
        ):
            raise RuntimeError("Demo 3 Pull axes are not opposite")

        models = initialize_demo3_model_bundle(
            CHECKPOINT,
            CONFIG.algo.config,
            expected_sha256=CHECKPOINT_SHA256,
            device=env.device,
        )
        runner = Demo3PolicyRunner(models)
        initial_table = command.simulator_object_pos_w.clone()
        rewards = []
        values = []
        for _ in range(ARGS.steps):
            decision = runner.decide(
                observations,
                update_critic_normalizer=False,
            )
            observations, reward, done, _ = env.step({"actions": decision.actions})
            if reward.shape != (1, 2) or decision.values.shape != (1, 2, 1):
                raise RuntimeError(
                    f"Per-agent reward/value contract failed: {reward.shape}, {decision.values.shape}"
                )
            if done.any() and command.time_steps[0].item() < command.reference.num_frames - 1:
                raise RuntimeError("Demo 3 terminated before the reference horizon during smoke")
            rewards.append(reward.detach().cpu())
            values.append(decision.values.detach().cpu())

        final_table = command.simulator_object_pos_w.clone()
        report = {
            "status": "passed",
            "seed": ARGS.seed,
            "steps": ARGS.steps,
            "environment": type(env).__name__,
            "control_hz": 1.0 / float(env.dt),
            "physics_hz": 1.0 / float(env.sim_dt),
            "reference_frames": command.reference.num_frames,
            "reference_fps": command.reference.fps,
            "actor_observation_shape": list(observations["actor_obs"].shape),
            "teammate_observation_shape": list(observations["teammate_obs"].shape),
            "table_observation_shape": list(observations["table_obs"].shape),
            "critic_observation_shape": list(observations["critic_obs"].shape),
            "reward_shape": list(rewards[-1].shape),
            "value_shape": list(values[-1].shape),
            "pull_axes_world_xy": command.agent_pull_axis_w[0].detach().cpu().tolist(),
            "table_displacement_m": (final_table - initial_table)[0].detach().cpu().tolist(),
            "checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "object_physics": _physics(env),
        }
        text = json.dumps(report, indent=2, sort_keys=True)
        print(text)
        if ARGS.output is not None:
            output = ARGS.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n")
        return 0
    except BaseException as exc:
        error = exc
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            close_env = getattr(env, "close", None)
            if callable(close_env):
                close_env()
        close_simulation_app(SIMULATION_APP)
        if error is not None:
            print(f"Demo 3 smoke failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
