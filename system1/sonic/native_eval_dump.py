#!/usr/bin/env python3
"""Run SONIC's official ``eval_agent_trl.py`` unchanged and record what it sees.

The official entry point and Hydra overrides are passed through verbatim; this
wrapper only adds read-only hooks on ``ImEvalCallback`` so that, for the first
``SONIC_DUMP_ENVS`` environments and ``SONIC_DUMP_STEPS`` control steps, it saves

* the observation dict given to the policy (tokenizer + policy groups),
* the deterministic action returned by the policy,
* robot root / joint state after the step and the per-step MPJPE inputs,
* the full reference arrays of every loaded motion from the motion library,
* the runtime robot model (joint order, defaults, PD gains, limits, masses).

These recordings are the ground truth for checking ``system1/sonic_controller.py``.

Run from the GR00T-WholeBodyControl repository root with the SONIC environment:

    SONIC_DUMP_OUT=/abs/path/native_dump.npz \
    python /home/kevin/duet-system1/system1/sonic/native_eval_dump.py +checkpoint=... <official overrides>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

DUMP_OUT = os.environ["SONIC_DUMP_OUT"]
DUMP_ENVS = int(os.environ.get("SONIC_DUMP_ENVS", "2"))
DUMP_STEPS = int(os.environ.get("SONIC_DUMP_STEPS", "400"))

from gear_sonic.trl.callbacks import im_eval_callback as imcb  # noqa: E402

_records: dict[str, list] = {}
_meta: dict = {}


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _push(key, value):
    _records.setdefault(key, []).append(_np(value)[:DUMP_ENVS].copy())


def _flatten_obs(prefix, obs):
    if isinstance(obs, dict):
        for k, v in obs.items():
            _flatten_obs(f"{prefix}/{k}", v)
    elif isinstance(obs, torch.Tensor):
        _push(prefix, obs)


def _dump_static(env):
    cmd = env.motion_command
    robot = cmd.robot
    lib = cmd.motion_lib
    _meta["joint_names"] = list(robot.joint_names)
    _meta["body_names_robot"] = list(robot.body_names)
    _meta["cmd_body_names"] = list(cmd.cfg.body_names)
    _meta["anchor_body"] = cmd.cfg.anchor_body
    _meta["num_future_frames"] = int(cmd.num_future_frames)
    _meta["frame_skips"] = float(cmd.frame_skips)
    try:
        from omegaconf import OmegaConf

        _meta["env_config"] = OmegaConf.to_container(env.config, resolve=True)
    except Exception as exc:  # keep the dump even if some field cannot be resolved
        _meta["env_config_error"] = repr(exc)
        _meta["env_config_obs"] = json.loads(json.dumps(dict(env.config["obs"]), default=str))
        _meta["env_config_robot"] = json.loads(json.dumps(dict(env.config["robot"]), default=str))
    _meta["step_dt"] = float(env.env.step_dt) if hasattr(env.env, "step_dt") else None
    _meta["physics_dt"] = float(env.env.physics_dt) if hasattr(env.env, "physics_dt") else None
    data = robot.data
    static = {
        "default_joint_pos": data.default_joint_pos[0],
        "joint_stiffness": data.joint_stiffness[0],
        "joint_damping": data.joint_damping[0],
        "joint_armature": data.joint_armature[0],
        "joint_effort_limits": data.joint_effort_limits[0],
        "joint_vel_limits": data.joint_vel_limits[0],
        "joint_pos_limits": data.joint_pos_limits[0],
        "body_masses": robot.root_physx_view.get_masses()[0],
    }
    act = env.env.action_manager.get_term("joint_pos")
    static["action_scale"] = act._scale if isinstance(act._scale, torch.Tensor) else torch.full((len(robot.joint_names),), float(act._scale))
    static["action_offset"] = act._offset if isinstance(act._offset, torch.Tensor) else torch.full((len(robot.joint_names),), float(act._offset))
    for k, v in static.items():
        _records[f"static/{k}"] = [_np(v).reshape(-1) if k != "joint_pos_limits" else _np(v)]
    # Reference arrays of the motions assigned to the dumped environments.
    for e in range(min(DUMP_ENVS, env.num_envs)):
        mid = cmd.motion_ids[e : e + 1]
        n = int(cmd.motion_num_steps[e].item())
        t = torch.arange(n, device=mid.device)
        ids = mid.repeat(n)
        ref = {
            "dof_pos": lib.get_dof_pos(ids, t),
            "dof_vel": lib.get_dof_vel(ids, t),
            "root_pos_w": lib.get_root_pos_w(ids, t),
            "root_quat_w": lib.get_root_quat_w(ids, t),
            "body_pos_w": lib.get_body_pos_w(ids, t),
            "body_quat_w": lib.get_body_quat_w(ids, t),
        }
        for k, v in ref.items():
            _records[f"ref{e}/{k}"] = [_np(v)]
        _meta[f"ref{e}_motion_id"] = int(mid.item())
        _meta[f"ref{e}_motion_start_time_step"] = int(cmd.motion_start_time_steps[e].item())
        _meta[f"ref{e}_num_steps"] = n


_orig_pre = imcb.ImEvalCallback._pre_eval_env_step
_orig_post = imcb.ImEvalCallback._post_eval_env_step
_state = {"steps": 0, "static_done": False}


def _pre(self, actor_state):
    if not _state["static_done"]:
        _dump_static(self.env)
        _state["static_done"] = True
    out = _orig_pre(self, actor_state)
    if _state["steps"] < DUMP_STEPS and "step" in actor_state:
        _flatten_obs("obs", actor_state["obs"])
        _push("action", out["actions"])
        cmd = self.env.motion_command
        _push("time_steps", cmd.time_steps)
        _push("motion_start_time_steps", cmd.motion_start_time_steps)
        r = cmd.robot.data
        _push("pre/root_pos_w", r.root_pos_w)
        _push("pre/root_quat_w", r.root_quat_w)
        _push("pre/root_lin_vel_w", r.root_lin_vel_w)
        _push("pre/root_ang_vel_w", r.root_ang_vel_w)
        _push("pre/root_ang_vel_b", r.root_ang_vel_b)
        _push("pre/joint_pos", r.joint_pos)
        _push("pre/joint_vel", r.joint_vel)
    return out


def _post(self, actor_state):
    out = _orig_post(self, actor_state)
    if _state["steps"] < DUMP_STEPS:
        cmd = self.env.motion_command
        _push("post/robot_body_pos_w", cmd.robot_body_pos_w)
        _push("post/ref_body_pos_w", cmd.body_pos_w)
        _push("post/terminate_state", self.terminate_state)
        if self.mpjpe:
            _push("post/mpjpe_mm", self.mpjpe[-1])
        _state["steps"] += 1
    if out.get("end_eval", False) or _state["steps"] == DUMP_STEPS:
        save()
    return out


def save():
    arrays = {}
    for k, v in _records.items():
        arrays[k] = np.stack(v) if len(v) > 1 else v[0]
    arrays["meta_json"] = np.array(json.dumps(_meta, default=str))
    np.savez_compressed(DUMP_OUT, **arrays)


imcb.ImEvalCallback._pre_eval_env_step = _pre
imcb.ImEvalCallback._post_eval_env_step = _post

if __name__ == "__main__":
    import importlib.util
    import runpy

    # Execute the official script as a script so Hydra resolves its config dir by file path.
    script = importlib.util.find_spec("gear_sonic.eval_agent_trl").origin
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
