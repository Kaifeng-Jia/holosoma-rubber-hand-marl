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
PARSER.add_argument("--experiment", choices=("smalltable", "chair021", "bucket003", "smalltable5kg_A"), default="smalltable")
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
from holosoma.config_values.marl.g1.core4d_pair_experiments import (  # noqa: E402
    get_core4d_pair_experiment,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (  # noqa: E402
    g1_29dof_core4d_smalltable_smoke,
    with_pair_experiment,
)
from holosoma.config_values.marl.g1.core4d_smalltable_observation import (  # noqa: E402
    g1_29dof_core4d_smalltable_evaluation_observation,
)
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402


EXPERIMENT = get_core4d_pair_experiment(ARGS.experiment)
if not EXPERIMENT.training_ready:
    PARSER.error(f"--experiment {ARGS.experiment} is not training_ready; complete asset/physics preparation first")
BASE_CONFIG = with_pair_experiment(g1_29dof_core4d_smalltable_smoke, EXPERIMENT)
if ARGS.experiment in ("bucket003", "smalltable5kg_A"):
    BASE_CONFIG = replace(
        BASE_CONFIG,
        env_class="holosoma.envs.marl.core4d_bucket_manager.Core4DBucketManager",
        simulator=replace(BASE_CONFIG.simulator, config=replace(
            BASE_CONFIG.simulator.config, enable_object_hand_contact=True,
        )),
    )
CONFIG = replace(
    BASE_CONFIG,
    observation=g1_29dof_core4d_smalltable_evaluation_observation,
    training=replace(
        BASE_CONFIG.training,
        num_envs=1,
        headless=True,
        seed=ARGS.seed,
        project=f"{EXPERIMENT.project}Eval",
        name="actor_only",
    ),
)
RUNTIME_REFERENCE_PATH = (
    REPO_ROOT / "src" / "holosoma" / EXPERIMENT.runtime_reference_file
).resolve()
OBJECT_URDF_PATH = (
    REPO_ROOT / "src" / "holosoma" / EXPERIMENT.object_urdf_file
).resolve()
TRAINING_PROMOTION_PATH = (
    REPO_ROOT / "src" / "holosoma" / EXPERIMENT.training_promotion_file
).resolve()
SIMULATION_APP = init_sim_imports(CONFIG)

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

    result = {
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
        "object_height_error_m": np.asarray(
            (command.simulator_object_pos_w[0, 2] - command.object_pos_w[0, 2]).item()
        ),
    }
    if ARGS.experiment in ("bucket003", "smalltable5kg_A"):
        sensor = env.simulator.object_hand_contact_sensor
        result["hand_object_normal_force_w"] = array(sensor.data.force_matrix_w[:, 0].reshape(-1, 2, 2, 3))
        result["contact_valid_after_physics"] = np.asarray(env.episode_length_buf[0].item() > 0)
        result["reference_frame"] = np.asarray(command.time_steps[0].item())
        result["episode_step"] = np.asarray(env.episode_length_buf[0].item())
    return result


def _stack(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        name: np.stack([frame[name] for frame in frames], axis=0)
        for name in frames[0]
    }


def _episode_height_metrics(trajectory: dict[str, np.ndarray]) -> dict[str, float]:
    """Same-state/reference-frame height errors, including the pre-reset terminal state."""
    error = np.asarray(trajectory["object_height_error_m"])
    if error.ndim != 1 or not error.size or not np.isfinite(error).all():
        raise ValueError("Episode height errors must be a nonempty finite vector")
    return {
        "object_height_rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "object_height_bias_m": float(np.mean(error)),
        "object_height_abs_error_max_m": float(np.max(np.abs(error))),
    }


def main() -> int:
    env = None
    try:
        checkpoint = ARGS.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        _require_sha256(
            RUNTIME_REFERENCE_PATH,
            EXPERIMENT.runtime_reference_sha256,
            "runtime reference",
        )
        _require_sha256(
            OBJECT_URDF_PATH,
            EXPERIMENT.object_urdf_sha256,
            "object URDF",
        )
        _require_sha256(
            TRAINING_PROMOTION_PATH,
            EXPERIMENT.training_promotion_sha256,
            "training-asset promotion",
        )
        output_dir = ARGS.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty evaluation directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(ARGS.seed)
        state = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        iteration = validate_core4d_smalltable_checkpoint(
            state, experiment_contract=EXPERIMENT.checkpoint_contract,
        )
        reward_metadata = core4d_smalltable_evaluation_reward_metadata(
            state, experiment_contract=EXPERIMENT.checkpoint_contract,
        )
        if ARGS.experiment in ("bucket003", "smalltable5kg_A"):
            from holosoma.utils.module_utils import get_holosoma_root
            asset_root = CONFIG.robot.asset.asset_root.replace("@holosoma", get_holosoma_root())
            _require_sha256(
                Path(asset_root) / CONFIG.robot.asset.urdf_file,
                state["core4d_smalltable_mappo"]["bucket_reward"]["training_robot_urdf_sha256"],
                "bucket training robot URDF",
            )
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
                **_episode_height_metrics(trajectory),
            }
            episode_path = save_core4d_smalltable_viser(
                output_dir / f"episode_{episode:03d}.npz",
                trajectory,
                metadata={
                    **reward_metadata,
                    "experiment": EXPERIMENT.experiment_id,
                    "checkpoint": str(checkpoint),
                    "checkpoint_iteration": iteration,
                    "seed": ARGS.seed,
                    "actor_only": True,
                    "observation_noise": False,
                    "episode": episode,
                    "fps": EXPERIMENT.reference_fps,
                    "result": dict(result),
                },
            )
            result["episode_viser"] = str(episode_path)
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
                "experiment": EXPERIMENT.experiment_id,
                "scenario": EXPERIMENT.scenario,
                "runtime_reference_file": EXPERIMENT.runtime_reference_file,
                "object_urdf_file": EXPERIMENT.object_urdf_file,
                "checkpoint": str(checkpoint),
                "checkpoint_iteration": iteration,
                "seed": ARGS.seed,
                "actor_only": True,
                "observation_noise": False,
                "episode": representative,
                "fps": EXPERIMENT.reference_fps,
                "result": episode_results[representative],
            },
        )
        summary = {
            **reward_metadata,
            "experiment": EXPERIMENT.experiment_id,
            "scenario": EXPERIMENT.scenario,
            "runtime_reference_file": EXPERIMENT.runtime_reference_file,
            "object_urdf_file": EXPERIMENT.object_urdf_file,
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": iteration,
            "seed": ARGS.seed,
            "episodes": episode_results,
            "completion_rate": sum(
                item["completed_reference"] for item in episode_results
            )
            / len(episode_results),
            "representative_episode": representative,
            "representative_selection": "completed_first_then_minimum_object_position_rmse_not_random",
            "representative_viser": str(viser_path),
            "all_episode_replays_saved": True,
            "height_metric_note": (
                "Each episode reports same-state/reference-frame object-root z errors; "
                "not whole-geometry clearance or hand load-bearing. Compare completion and episode "
                "lengths alongside errors; failed prefixes are not full-trajectory statistics."
            ),
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
