#!/usr/bin/env python3
"""Verify that random reserved teammate channels preserve frozen single-agent A1."""

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

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg
from holosoma.config_values.marl.g1.command import motion_config as frozen_a1_motion
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_w_object,
    g1_29dof_wbt_w_object_marl_compat,
)
from holosoma.utils.eval_utils import init_sim_imports


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--steps", type=int, default=20)
ARGS = PARSER.parse_args()
if ARGS.steps < 1:
    PARSER.error("--steps must be at least 1")

_single_motion_term = CommandTermCfg(
    func="holosoma.managers.command.terms.wbt:MotionCommand",
    params={"motion_config": frozen_a1_motion},
)
_single_a1_command = CommandManagerCfg(
    setup_terms={"motion_command": _single_motion_term},
    reset_terms={"motion_command": _single_motion_term},
    step_terms={"motion_command": _single_motion_term},
)
CONFIG = replace(
    g1_29dof_wbt_w_object_marl_compat,
    training=replace(
        g1_29dof_wbt_w_object_marl_compat.training,
        num_envs=1,
        headless=True,
        project="Plan5Preflight",
        name="single_a1_random_teammate_smoke",
    ),
    command=_single_a1_command,
    robot=replace(
        g1_29dof_wbt_w_object_marl_compat.robot,
        object=replace(
            g1_29dof_wbt_w_object_marl_compat.robot.object,
            object_urdf_path=(
                "holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largetable.urdf"
            ),
        ),
    ),
)
SIMULATION_APP = init_sim_imports(CONFIG)

import torch  # noqa: E402

from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle  # noqa: E402
from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


def main() -> None:
    env = None
    failure: BaseException | None = None
    try:
        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        env.set_is_evaluating()
        observations = env.reset_all()

        checkpoint = REPO_ROOT / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"
        models = initialize_plan5_model_bundle(
            checkpoint,
            g1_29dof_wbt_w_object.algo.config,
            device=env.device,
        )
        generator = torch.Generator(device=env.device).manual_seed(721)
        max_action_difference = 0.0
        reset_count = 0
        teammate_abs_max = 0.0
        reward_values = []

        for _ in range(ARGS.steps):
            base_observation = observations["actor_obs"]
            random_teammate = torch.empty(
                env.num_envs,
                4,
                device=env.device,
            ).uniform_(-1.0, 1.0, generator=generator)
            zero_input = torch.cat((base_observation, torch.zeros_like(random_teammate)), dim=-1)
            random_input = torch.cat((base_observation, random_teammate), dim=-1)
            zero_action = models.actor.act_inference(
                {"actor_obs": models.actor_obs_normalizer(zero_input, update=False)}
            )
            random_action = models.actor.act_inference(
                {"actor_obs": models.actor_obs_normalizer(random_input, update=False)}
            )
            action_difference = torch.max(torch.abs(zero_action - random_action)).item()
            max_action_difference = max(max_action_difference, action_difference)
            teammate_abs_max = max(teammate_abs_max, torch.max(torch.abs(random_teammate)).item())
            observations, reward, done, _ = env.step({"actions": random_action})
            reward_values.append(reward.detach().clone())
            reset_count += int(done.count_nonzero().item())

            if not all(
                torch.isfinite(tensor).all()
                for tensor in (
                    base_observation,
                    random_teammate,
                    zero_action,
                    random_action,
                    env.simulator.robot_root_states[:],
                    env.simulator.dof_pos,
                )
            ):
                raise RuntimeError("Non-finite state in random teammate rollout")

        if max_action_difference > 1.0e-7:
            raise RuntimeError(
                f"Random teammate channels changed frozen A1 actions by {max_action_difference}"
            )
        robot_urdf = (
            REPO_ROOT
            / "src"
            / "holosoma"
            / "holosoma"
            / "data"
            / "robots"
            / CONFIG.robot.asset.urdf_file
        )
        urdf_text = robot_urdf.read_text().lower()
        if "rubber_hand" not in urdf_text or "hemisphere" in urdf_text:
            raise RuntimeError("Single-agent robustness smoke did not use the rubber-hand asset")
        reward_history = torch.stack(reward_values)
        report = {
            "passed": True,
            "rollout_steps": ARGS.steps,
            "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
            "motion_file": frozen_a1_motion.motion_file,
            "object_urdf": CONFIG.robot.object.object_urdf_path,
            "robot_urdf": CONFIG.robot.asset.urdf_file,
            "teammate_range": [-1.0, 1.0],
            "teammate_abs_max_sampled": teammate_abs_max,
            "max_abs_action_difference": max_action_difference,
            "reset_count": reset_count,
            "reward_min": reward_history.min().item(),
            "reward_max": reward_history.max().item(),
        }
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
        print(
            json.dumps(
                {"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(SIMULATION_APP)
    if failure is not None:
        raise RuntimeError("Single-agent A1 random teammate smoke failed") from failure


if __name__ == "__main__":
    main()
