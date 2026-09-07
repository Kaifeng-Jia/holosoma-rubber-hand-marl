#!/usr/bin/env python3
"""Actor-only evaluation with a fixed baseline scoring regime for both reward variants.

For interaction-mesh checkpoints, reward_sum intentionally excludes the new training
reward. Reports explicitly name both regimes; completion/tracking metrics are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--checkpoint", type=Path, required=True)
PARSER.add_argument("--episodes", type=int, default=5)
PARSER.add_argument("--seed", type=int, default=721)
PARSER.add_argument("--output-dir", type=Path, required=True)
ARGS = PARSER.parse_args()
if ARGS.episodes < 1:
    PARSER.error("--episodes must be positive")

from holosoma.agents.mappo.core4d_smalltable_evaluation import (  # noqa: E402
    core4d_smalltable_evaluation_reward_metadata,
    deterministic_core4d_smalltable_actions,
    save_core4d_smalltable_viser,
    validate_core4d_smalltable_checkpoint,
)
from holosoma.agents.mappo.core4d_smalltable_initialization import (  # noqa: E402
    initialize_core4d_smalltable_model_bundle,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (  # noqa: E402
    CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
    CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
    CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
)
from holosoma.config_values.marl.g1.core4d_smalltable_command import (  # noqa: E402
    CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (  # noqa: E402
    CORE4D_SMALLTABLE_OBJECT_URDF,
    g1_29dof_core4d_smalltable_smoke,
)
from holosoma.config_values.marl.g1.core4d_smalltable_observation import (  # noqa: E402
    g1_29dof_core4d_smalltable_evaluation_observation,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402


CONFIG = replace(
    g1_29dof_core4d_smalltable_smoke,
    observation=g1_29dof_core4d_smalltable_evaluation_observation,
    training=replace(
        g1_29dof_core4d_smalltable_smoke.training,
        num_envs=1,
        headless=True,
        seed=ARGS.seed,
        project="Core4DSmallTableEval",
        name="actor_only",
    ),
)
SIMULATION_APP = init_sim_imports(CONFIG)

RUNTIME_REFERENCE_PATH = (
    REPO_ROOT / "src" / "holosoma" / CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE
).resolve()
OBJECT_URDF_PATH = (
    REPO_ROOT / "src" / "holosoma" / CORE4D_SMALLTABLE_OBJECT_URDF
).resolve()
TRAINING_PROMOTION_PATH = RUNTIME_REFERENCE_PATH.with_name(
    "training_asset_manifest.json"
)

import torch  # noqa: E402

from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402


def _require_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"CORE4D {label} is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"CORE4D {label} SHA-256 mismatch: expected {expected}, got {digest}"
        )


def _snapshot(env, command) -> dict[str, np.ndarray]:
    def array(value):
        return value[0].detach().cpu().numpy().copy()

    return {
        "root_pos": array(env.simulator.agent_root_states[..., :3]),
        "root_quat_xyzw": array(env.simulator.agent_root_states[..., 3:7]),
        "dof_pos": array(env.simulator.agent_dof_pos),
        "object_pos_w": array(command.simulator_object_pos_w),
        "object_quat_xyzw": array(command.simulator_object_quat_w),
        "object_position_error_m": np.asarray(
            torch.linalg.vector_norm(
                command.object_pos_w - command.simulator_object_pos_w,
                dim=-1,
            )[0].item()
        ),
    }


def _stack(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        name: np.stack([frame[name] for frame in frames], axis=0)
        for name in frames[0]
    }


def main() -> int:
    env = None
    try:
        checkpoint = ARGS.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        _require_sha256(
            RUNTIME_REFERENCE_PATH,
            CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
            "runtime reference",
        )
        _require_sha256(
            OBJECT_URDF_PATH,
            CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
            "object URDF",
        )
        _require_sha256(
            TRAINING_PROMOTION_PATH,
            CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
            "training-asset promotion",
        )
        output_dir = ARGS.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(ARGS.seed)
        state = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        iteration = validate_core4d_smalltable_checkpoint(state)
        reward_metadata = core4d_smalltable_evaluation_reward_metadata(state)
        print(json.dumps({"reward_regime": reward_metadata}, sort_keys=True), flush=True)
        env_class = get_class(CONFIG.env_class)
        env = env_class(get_tyro_env_config(CONFIG), device="cuda:0")
        env.set_is_evaluating()
        observations = env.reset_all()
        command = env.command_manager.get_state("paired_motion_command")
        models = initialize_core4d_smalltable_model_bundle(
            CONFIG.algo.config,
            device=env.device,
        )
        models.actor.load_state_dict(state["actor_model_state_dict"], strict=True)
        models.actor_obs_normalizer.load_state_dict(
            state["actor_obs_normalizer_state_dict"],
            strict=True,
        )
        models.actor.eval()
        models.actor_obs_normalizer.eval()

        terminal_snapshot = None
        capture_terminal = False
        original_reset = env.reset_envs_idx

        def capturing_reset(env_ids, target_states=None, target_buf=None):
            nonlocal terminal_snapshot
            ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
            if capture_terminal and bool(torch.any(ids == 0).item()):
                terminal_snapshot = _snapshot(env, command)
            return original_reset(env_ids, target_states, target_buf)

        env.reset_envs_idx = capturing_reset
        episode_results = []
        trajectories = []
        max_steps = command.reference.num_frames + 1
        for episode in range(ARGS.episodes):
            frames = [_snapshot(env, command)]
            terminal_snapshot = None
            capture_terminal = True
            reward_sum = 0.0
            terminal_terms = {}
            for step in range(1, max_steps + 1):
                actions = deterministic_core4d_smalltable_actions(models, observations)
                observations, rewards, dones, extras = env.step({"actions": actions})
                reward_sum += float(rewards[0].item())
                if bool(dones[0].item()):
                    terminal_terms = {
                        name: bool(values[0].item())
                        for name, values in extras.get("termination_terms", {}).items()
                    }
                    if terminal_snapshot is None:
                        raise RuntimeError("Terminal physical state was not captured")
                    frames.append(terminal_snapshot)
                    break
                frames.append(_snapshot(env, command))
            else:
                raise RuntimeError(f"Episode {episode} exceeded the reference horizon")
            capture_terminal = False
            trajectory = _stack(frames)
            result = {
                "episode": episode,
                "steps": step,
                "reward_sum": reward_sum,
                "reward_sum_contract": reward_metadata["evaluation_reward_contract"],
                "completed_reference": bool(terminal_terms.get("reference_horizon", False)),
                "bad_tracking": bool(terminal_terms.get("joint_bad_tracking", False)),
                "object_position_rmse_m": float(
                    np.sqrt(np.mean(np.square(trajectory["object_position_error_m"])))
                ),
                "actual_table_displacement_m": float(
                    np.linalg.norm(
                        trajectory["object_pos_w"][-1] - trajectory["object_pos_w"][0]
                    )
                ),
            }
            episode_results.append(result)
            trajectories.append(trajectory)
            print(json.dumps(result, sort_keys=True), flush=True)

        representative = min(
            range(len(episode_results)),
            key=lambda index: (
                not episode_results[index]["completed_reference"],
                episode_results[index]["object_position_rmse_m"],
                index,
            ),
        )
        trajectory = trajectories[representative]
        viser_path = save_core4d_smalltable_viser(
            output_dir / "representative_episode.npz",
            trajectory,
            metadata={
                **reward_metadata,
                "scenario": "core4d_paired_small_table_reference_tracking",
                "checkpoint": str(checkpoint),
                "checkpoint_iteration": iteration,
                "seed": ARGS.seed,
                "actor_only": True,
                "observation_noise": False,
                "episode": representative,
                "fps": 50,
                "result": episode_results[representative],
            },
        )
        summary = {
            **reward_metadata,
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": iteration,
            "seed": ARGS.seed,
            "episodes": episode_results,
            "completion_rate": sum(
                item["completed_reference"] for item in episode_results
            )
            / len(episode_results),
            "representative_episode": representative,
            "representative_viser": str(viser_path),
            "inference": "shared_actor_only_critic_not_called",
            "observation_noise": False,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
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
