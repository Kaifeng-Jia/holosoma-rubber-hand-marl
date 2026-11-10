"""SONIC (GEAR-SONIC release checkpoint) as a frozen low-level controller inside holosoma.

The controller reproduces, for a batch of B robots, exactly the observations SONIC's
own Isaac Lab environment builds for its G1 encoder and its proprioceptive decoder,
runs the released policy (deterministic action mean), and converts the action to
joint position targets with SONIC's action scale / default pose. PD gains returned
by :meth:`pd_gains` are SONIC's (read from its articulation at runtime and stored in
the native dump), so callers can apply SONIC's low-level controller.

Conventions inside this module follow SONIC / Isaac Lab:
* quaternions are (w, x, y, z);
* joint vectors are in Isaac Lab articulation order (``sonic_joint_names``).
Holosoma uses (x, y, z, w) quaternions and its own joint order; the adapter methods
``from_holosoma_*`` and ``to_holosoma_*`` do the conversion by joint name.

Reference semantics (commands.py of gear_sonic):
* future frames: t + k*frame_skips for k = 0..num_future_frames-1, clamped to the
  last frame; frame_skips = 5 at 50 Hz (0.1 s spacing);
* ``command_multi_future_nonflat`` = cat(joint_pos[10*29], joint_vel[10*29]).reshape(10, 58)
  (this mixes frames row-wise exactly as the original code does);
* ``motion_anchor_ori_b_mf_nonflat`` = first two columns of R(q_robot_pelvis^-1 * q_ref_pelvis[k]).
Proprioception (history 10, oldest first, each term flattened then concatenated):
base_ang_vel(3), joint_pos_rel(29), joint_vel_rel(29), last_action(29), gravity_dir(3) (order verified against the native dump).
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SONIC_ROOT = Path("/home/kevin/third_party/GR00T-WholeBodyControl")
DEFAULT_CHECKPOINT = SONIC_ROOT / "sonic_release" / "last.pt"
DEFAULT_SPEC = Path(__file__).resolve().parent / "sonic" / "sonic_release_spec.json"

PROPRIO_TERMS = (("base_ang_vel", 3), ("joint_pos", 29), ("joint_vel", 29), ("actions", 29), ("gravity_dir", 3))
HISTORY = 10
INFER_BLOCK = 128  # fixed inference block (F3 batch invariance)


# ----------------------------------------------------------------------------- quaternion math (w, x, y, z)
def quat_conj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_apply(q, v):
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def matrix_from_quat(q):
    w, x, y, z = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    m = torch.stack(
        (
            1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w),
            two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w),
            two_s * (x * z - y * w), two_s * (y * z + x * w), 1 - two_s * (x * x + y * y),
        ),
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


# ----------------------------------------------------------------------------- observation builders
def future_indices(t, num_frames, n_future=10, skip=5):
    """t: [B] long current frame; num_frames: [B] or int. Returns [B, n_future]."""
    k = torch.arange(n_future, device=t.device) * skip
    idx = t[:, None] + k[None]
    n = torch.as_tensor(num_frames, device=t.device)
    return torch.minimum(idx, (n - 1).reshape(-1, 1) if n.ndim else n - 1)


def g1_encoder_terms(ref_dof_pos_f, ref_dof_vel_f, ref_root_quat_f, robot_root_quat):
    """Future reference windows [B, F, 29] / [B, F, 4] and robot pelvis quat [B, 4] (all SONIC order, wxyz)."""
    b, f, _ = ref_dof_pos_f.shape
    cmd = torch.cat((ref_dof_pos_f.reshape(b, -1), ref_dof_vel_f.reshape(b, -1)), dim=1).reshape(b, f, -1)
    rel = quat_mul(quat_conj(robot_root_quat)[:, None].expand(b, f, 4), ref_root_quat_f)
    ori = matrix_from_quat(rel)[..., :2].reshape(b, f, 6)
    return cmd, ori


def proprio_now(root_quat, root_ang_vel_b, joint_pos, joint_vel, last_action, default_joint_pos):
    down = torch.tensor([0.0, 0.0, -1.0], device=root_quat.device).expand(root_quat.shape[0], 3)
    grav = quat_apply(quat_conj(root_quat), down)
    return dict(
        gravity_dir=grav,
        base_ang_vel=root_ang_vel_b,
        joint_pos=joint_pos - default_joint_pos,
        joint_vel=joint_vel,
        actions=last_action,
    )


@dataclass
class TokenizerLayout:
    names: list
    dims: list  # flattened sizes

    @classmethod
    def from_spec(cls, spec):
        g = spec["env_config"]["obs"]["group_obs_dims"]["tokenizer"]
        names = spec["env_config"]["obs"]["group_obs_names"]["tokenizer"]
        return cls(names=list(names), dims=[int(np.prod(g[n])) for n in names])

    @property
    def total(self):
        return sum(self.dims)

    def assemble(self, parts: dict, batch: int, device):
        cols = []
        for n, d in zip(self.names, self.dims):
            v = parts.get(n)
            cols.append(torch.zeros(batch, d, device=device) if v is None else v.reshape(batch, d))
        return torch.cat(cols, dim=1)


# ----------------------------------------------------------------------------- policy
class _LenientPickle:
    """pickle_module for torch.load: classes of training-only packages (e.g. ``trl``
    trainer state) become inert placeholders; tensors and plain containers load normally."""

    import pickle as _pickle

    class Unpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                def _setstate(obj, s):
                    obj.__dict__.update(s if isinstance(s, dict) else {"_state": s})
                def _init(obj, *args, **kwargs):
                    obj._args, obj._kwargs = args, kwargs

                return type(name, (), {"__module__": f"placeholder.{module}", "__setstate__": _setstate, "__init__": _init})

    load = staticmethod(lambda f, **kw: _LenientPickle.Unpickler(f, **kw).load())
    __name__ = "lenient_pickle"


def load_sonic_policy(checkpoint=DEFAULT_CHECKPOINT, spec=None, device="cuda:0"):
    """Instantiate SONIC's Actor exactly as gear_sonic/eval_agent_trl.py does and load weights."""
    import omegaconf

    from gear_sonic.trl.utils import common as trl_utils_common

    checkpoint = Path(checkpoint)
    raw = (checkpoint.parent / "config.yaml").read_text()
    for a, b in (
        ("groot.rl.trl.", "gear_sonic.trl."),
        ("groot.rl.envs.", "gear_sonic.envs."),
        ("groot.rl.utils.", "gear_sonic.utils."),
        ("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module."),
        ("groot.rl.agents.", "gear_sonic.trl."),
        ("groot/rl/data/", "gear_sonic/data/"),
        ("assets/bm/unitree_description/", "assets/robot_description/"),
        ("1215_bones_seed_filtered", "bones_seed_smpl"),
    ):
        raw = raw.replace(a, b)
    cfg = omegaconf.OmegaConf.load(io.StringIO(raw))
    if cfg.get("eval_overrides") is not None:
        cfg = omegaconf.OmegaConf.merge(cfg, cfg.eval_overrides)
    env_config = omegaconf.OmegaConf.create(spec["env_config"])
    policy = trl_utils_common.custom_instantiate(
        cfg.algo.config.actor,
        env_config=env_config,
        algo_config=cfg.algo.config,
        module_dim_dict=getattr(cfg.algo.config, "module_dim", {}),
        backbone_kwargs={},
        _resolve=False,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False, pickle_module=_LenientPickle)
    sd = state.get("actor_model_state_dict", state.get("policy_state_dict"))
    model_uses_std = "std" in policy.state_dict()
    if model_uses_std and "log_std" in sd and "std" not in sd:
        sd = dict(sd)
        sd["std"] = sd.pop("log_std").exp()
    elif not model_uses_std and "std" in sd and "log_std" not in sd:
        sd = dict(sd)
        sd["log_std"] = sd.pop("std").log()
    policy.load_state_dict(sd)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


class SonicController:
    """Batched SONIC inference for B robots (B = num_envs * num_agents)."""

    def __init__(self, batch, device="cuda:0", checkpoint=DEFAULT_CHECKPOINT, spec_path=DEFAULT_SPEC, holosoma_joint_names=None,
                 tf32=False):
        # F3: formal evaluations run with TF32 off so results do not depend on the batch size.
        # (SONIC's own eval_agent_trl.py turns TF32 on; training of scheme C may opt in.)
        self.tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.backends.cudnn.allow_tf32 = self.tf32
        self.spec = json.loads(Path(spec_path).read_text())
        self.batch, self.device = batch, device
        self.joint_names = list(self.spec["joint_names"])
        st = self.spec["static"]
        t = lambda k: torch.tensor(st[k], dtype=torch.float32, device=device)  # noqa: E731
        self.default_joint_pos = t("default_joint_pos")
        self.action_scale = t("action_scale")
        self.kp, self.kd = t("joint_stiffness"), t("joint_damping")
        self.effort_limit = t("joint_effort_limits")
        self.action_clip = float(self.spec["env_config"].get("action_clip_value", 20.0))
        self.n_future = int(self.spec["num_future_frames"])
        self.skip = int(self.spec["frame_skips"])
        self.layout = TokenizerLayout.from_spec(self.spec)
        self.g1_index = 0
        self.policy = load_sonic_policy(checkpoint, self.spec, device)
        self.policy.init_rollout()
        self.hist = {n: torch.zeros(batch, HISTORY, d, device=device) for n, d in PROPRIO_TERMS}
        self.fresh = torch.ones(batch, dtype=torch.bool, device=device)
        self.last_action = torch.zeros(batch, 29, device=device)
        if holosoma_joint_names is not None:
            hn = list(holosoma_joint_names)
            self.h2s = torch.tensor([hn.index(n) for n in self.joint_names], device=device)
            self.s2h = torch.tensor([self.joint_names.index(n) for n in hn], device=device)

    # -- order adapters
    def to_sonic(self, x_h):
        return x_h[..., self.h2s]

    def to_holosoma(self, x_s):
        return x_s[..., self.s2h]

    def reset(self, ids):
        """Mark robots whose episode restarts; their history is refilled on the next observation."""
        self.fresh[ids] = True
        self.last_action[ids] = 0.0

    def _push_history(self, now):
        for n, _ in PROPRIO_TERMS:
            h = self.hist[n]
            h[:, :-1] = h[:, 1:].clone()
            h[:, -1] = now[n]
            if self.fresh.any():
                h[self.fresh] = now[n][self.fresh][:, None].expand(-1, HISTORY, -1)
        self.fresh[:] = False

    def actor_obs(self):
        return torch.cat([self.hist[n].reshape(self.batch, -1) for n, _ in PROPRIO_TERMS], dim=1)

    def tokenizer_obs(self, ref_dof_pos_f, ref_dof_vel_f, ref_root_quat_f, root_quat):
        cmd, ori = g1_encoder_terms(ref_dof_pos_f, ref_dof_vel_f, ref_root_quat_f, root_quat)
        enc = torch.zeros(self.batch, 3, device=self.device)
        enc[:, self.g1_index] = 1.0
        return self.layout.assemble(
            {"encoder_index": enc, "command_multi_future_nonflat": cmd, "motion_anchor_ori_b_mf_nonflat": ori},
            self.batch, self.device,
        )

    @torch.no_grad()
    def infer(self, obs, dones):
        """Policy action mean, evaluated in fixed blocks of ``INFER_BLOCK`` rows (zero-padded).

        cuBLAS picks kernels by the number of rows, so plain batched inference is not
        bit-identical across batch sizes even with TF32 off (F3). A fixed block size makes
        each row's result independent of the batch size and of its position in the batch.
        """
        n = obs["actor_obs"].shape[0]
        out = []
        for s in range(0, n, INFER_BLOCK):
            blk = {k: v[s : s + INFER_BLOCK] for k, v in obs.items()}
            d = dones[s : s + INFER_BLOCK]
            k = d.shape[0]
            if k < INFER_BLOCK:
                blk = {kk: torch.cat([v, v.new_zeros(INFER_BLOCK - k, v.shape[1])]) for kk, v in blk.items()}
                d = torch.cat([d, d.new_zeros(INFER_BLOCK - k)])
            self.policy.init_rollout()  # history length 1: no cross-call state in the Actor
            out.append(self.policy.act_inference(obs_dict=blk, cur_dones=d, skip_episode_attnmask=True)[:k])
        return torch.cat(out)

    @torch.no_grad()
    def act(self, root_quat, root_ang_vel_b, joint_pos, joint_vel, ref_dof_pos_f, ref_dof_vel_f, ref_root_quat_f, dones=None):
        """All inputs in SONIC order / wxyz. Returns (raw action [B,29], joint targets [B,29])."""
        now = proprio_now(root_quat, root_ang_vel_b, joint_pos, joint_vel, self.last_action, self.default_joint_pos)
        self._push_history(now)
        obs = {"actor_obs": self.actor_obs(), "tokenizer": self.tokenizer_obs(ref_dof_pos_f, ref_dof_vel_f, ref_root_quat_f, root_quat)}
        if dones is None:
            dones = torch.zeros(self.batch, device=self.device)
        action = self.infer(obs, dones)
        self.last_action = action.clamp(-self.action_clip, self.action_clip).clone()
        target = self.default_joint_pos + self.action_scale * self.last_action
        return action, target, obs
