"""T10 sim2sim: SONIC (official deployment ONNX) and the policies (exported ONNX) for the MuJoCo environment.

SonicOnnx reproduces system1/sonic_controller.SonicController (proprioception history of 10, oldest first; g1 encoder inputs;
deterministic action; target = default + action_scale * clip(action)) with the inference replaced by the official
model_encoder.onnx / model_decoder.onnx (encoder input = deploy layout: [mode id 0, 0, 0, 0] + tokenizer without its 3-wide one-hot).
The observation builders are the very functions used in Isaac (imported from system1.sonic_controller).

DController: actor ONNX -> holosoma P target default + scale * clip(a, +/-100).
CInController: scheme C 'in' exactly as system1/scheme_c.py: a = clip(actor, -1, 1); dq = a * paper-D action scale added to
every frame of SONIC's reference window, clamped to the URDF joint limits; SONIC's output q* is the PD target (SONIC PD);
the actor's last-action input is clip(a).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1 import sonic_controller as sc  # noqa: E402

HF = Path("/home/kevin/third_party/GR00T-WholeBodyControl/sonic_hf_release")


def _sess(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


class SonicOnnx:
    def __init__(self, holosoma_joint_names, batch=2, encoder=HF / "model_encoder.onnx", decoder=HF / "model_decoder.onnx"):
        self.spec = json.loads(sc.DEFAULT_SPEC.read_text())
        st = self.spec["static"]
        self.joint_names = list(self.spec["joint_names"])
        self.default = torch.tensor(st["default_joint_pos"], dtype=torch.float32)
        self.scale = torch.tensor(st["action_scale"], dtype=torch.float32)
        self.clip = float(self.spec["env_config"].get("action_clip_value", 20.0))
        self.n_future = int(self.spec["num_future_frames"])
        self.skip = int(self.spec["frame_skips"])
        self.layout = sc.TokenizerLayout.from_spec(self.spec)
        hn = list(holosoma_joint_names)
        self.h2s = np.array([hn.index(n) for n in self.joint_names])
        self.s2h = np.array([self.joint_names.index(n) for n in hn])
        self.batch = batch
        self.enc, self.dec = _sess(encoder), _sess(decoder)
        self.reset()

    def reset(self):
        self.hist = {n: torch.zeros(self.batch, sc.HISTORY, d) for n, d in sc.PROPRIO_TERMS}
        self.fresh = torch.ones(self.batch, dtype=torch.bool)
        self.last_action = torch.zeros(self.batch, 29)

    def _push(self, now):
        for n, _ in sc.PROPRIO_TERMS:
            h = self.hist[n]
            h[:, :-1] = h[:, 1:].clone()
            h[:, -1] = now[n]
            if self.fresh.any():
                h[self.fresh] = now[n][self.fresh][:, None].expand(-1, sc.HISTORY, -1)
        self.fresh[:] = False

    def act(self, root_quat_wxyz, ang_vel_b, jq_s, jqd_s, jp_win_s, jv_win_s, rq_win_wxyz):
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
        q = f(root_quat_wxyz)
        now = sc.proprio_now(q, f(ang_vel_b), f(jq_s), f(jqd_s), self.last_action, self.default)
        self._push(now)
        actor = torch.cat([self.hist[n].reshape(self.batch, -1) for n, _ in sc.PROPRIO_TERMS], dim=1)
        cmd, ori = sc.g1_encoder_terms(f(jp_win_s), f(jv_win_s), f(rq_win_wxyz), q)
        enc = torch.zeros(self.batch, 3)
        enc[:, 0] = 1.0
        tok = self.layout.assemble({"encoder_index": enc, "command_multi_future_nonflat": cmd, "motion_anchor_ori_b_mf_nonflat": ori},
                                   self.batch, "cpu")
        an, tn = actor.numpy(), tok.numpy()
        x = np.concatenate([np.zeros((self.batch, 4), np.float32), tn[:, 3:]], axis=1).astype(np.float32)  # g1 mode id 0
        actions = []
        for i in range(self.batch):
            t = self.enc.run(None, {"obs_dict": x[i:i + 1]})[0]
            actions.append(self.dec.run(None, {"obs_dict": np.concatenate([t, an[i:i + 1]], axis=1).astype(np.float32)})[0][0])
        action = torch.as_tensor(np.stack(actions))
        self.last_action = action.clamp(-self.clip, self.clip).clone()
        target = self.default + self.scale * self.last_action
        return action.numpy(), target.numpy(), dict(sonic_actor_obs=an, sonic_tokenizer=tn)


class ActorOnnx:
    def __init__(self, path):
        self.s = _sess(path)

    def __call__(self, obs158):
        return self.s.run(None, {"obs": np.asarray(obs158, np.float32)})[0].astype(np.float64)


class DController:
    kind = "D"

    def __init__(self, env, actor_onnx):
        self.env, self.actor = env, ActorOnnx(actor_onnx)

    def reset(self):
        self.last = np.zeros((2, self.env.nd))

    def step(self, st):
        actor, mate = self.env.observation(self.last, st=st)
        a = self.actor(np.concatenate([actor, mate], axis=1))
        self.last = a.copy()  # paired_actions = action_manager.action (raw; the clip at 100 never binds in evaluation)
        return self.env.d_target(a), dict(obs=np.concatenate([actor, mate], axis=1), actions=a)


class CInController:
    kind = "C-in"

    def __init__(self, env, actor_onnx, d_action_scales, ref_limits):
        self.env, self.actor = env, ActorOnnx(actor_onnx)
        self.scale_d = np.asarray(d_action_scales, dtype=np.float64)
        self.lo, self.hi = (np.asarray(x, dtype=np.float64) for x in ref_limits)
        self.sonic = SonicOnnx(env.dof_names)

    def reset(self):
        self.last = np.zeros((2, self.env.nd))
        self.sonic.reset()

    def window(self, t):
        env, so = self.env, self.sonic
        idx = np.minimum(t + np.arange(so.n_future) * so.skip, env.n - 1)
        s = env.ref.sample(torch.as_tensor(idx))
        jp = s["agent_joint_pos"].numpy().astype(np.float64).transpose(1, 0, 2)  # [A, F, 29] holosoma order
        jv = s["agent_joint_vel"].numpy().astype(np.float64).transpose(1, 0, 2)
        rq = s["agent_body_quat_w"][:, :, env.body_list.index("pelvis")].numpy().astype(np.float64).transpose(1, 0, 2)  # xyzw
        return jp, jv, rq[..., [3, 0, 1, 2]]

    def sonic_inputs(self, st, a_used, t):
        jp, jv, rq = self.window(t)
        jp = np.clip(jp + (a_used * self.scale_d)[:, None, :], self.lo, self.hi)
        root = st["root"]
        q = root[:, [6, 3, 4, 5]]
        wb = np.stack([sc.quat_apply(sc.quat_conj(torch.tensor(q[i:i + 1])), torch.tensor(root[i:i + 1, 10:13]))[0].numpy() for i in range(2)])
        return q, wb, st["dof_pos"][:, self.sonic.h2s], st["dof_vel"][:, self.sonic.h2s], jp[..., self.sonic.h2s], jv[..., self.sonic.h2s], rq

    def step(self, st):
        actor, mate = self.env.observation(self.last, st=st)
        a = self.actor(np.concatenate([actor, mate], axis=1))
        a_used = np.clip(a, -1.0, 1.0)
        action, target_s, sobs = self.sonic.act(*self.sonic_inputs(st, a_used, self.env.t))
        self.last = a_used.copy()
        return target_s[:, self.sonic.s2h].astype(np.float64), dict(obs=np.concatenate([actor, mate], axis=1), actions=a, sonic_action=action, **sobs)


class COutController(CInController):
    """T13 C1: scheme C 'out' exactly as system1/scheme_c.py kind "out": SONIC receives the unmodified reference (as scheme A);
    q* = SONIC's output joint target (holosoma order) + paper-D action scale * a_c, a_c = a clipped as D's action term
    (clip_actions / action_clip_value of the dumped control config, env.clip_a); the PD gets q* verbatim (SONIC PD of the
    C-out Isaac export); the actor's last-action input is the policy's own raw a (action_manager.policy_action = a, as D);
    SONIC's own last-action input stays its own previous output (SonicOnnx.last_action)."""
    kind = "C-out"

    def __init__(self, env, actor_onnx, d_action_scales):
        self.env, self.actor = env, ActorOnnx(actor_onnx)
        self.scale_d = np.asarray(d_action_scales, dtype=np.float64)
        self.lo = self.hi = None  # no reference clamp in C-out
        self.sonic = SonicOnnx(env.dof_names)

    def sonic_inputs_out(self, st, t):
        jp, jv, rq = self.window(t)
        root = st["root"]
        q = root[:, [6, 3, 4, 5]]
        wb = np.stack([sc.quat_apply(sc.quat_conj(torch.tensor(q[i:i + 1])), torch.tensor(root[i:i + 1, 10:13]))[0].numpy() for i in range(2)])
        return q, wb, st["dof_pos"][:, self.sonic.h2s], st["dof_vel"][:, self.sonic.h2s], jp[..., self.sonic.h2s], jv[..., self.sonic.h2s], rq

    def residual(self, a):
        a_c = a if self.env.clip_a is None else np.clip(a, -self.env.clip_a, self.env.clip_a)
        return a_c * self.scale_d

    def step(self, st):
        actor, mate = self.env.observation(self.last, st=st)
        a = self.actor(np.concatenate([actor, mate], axis=1))
        action, target_s, sobs = self.sonic.act(*self.sonic_inputs_out(st, self.env.t))
        q_star = target_s[:, self.sonic.s2h].astype(np.float64) + self.residual(a)
        self.last = a.copy()
        return q_star, dict(obs=np.concatenate([actor, mate], axis=1), actions=a, sonic_action=action, **sobs)
