"""T06 step 3 / T07: holosoma pieces for the G1 + Sharpa robot (73 DOFs per robot). Nothing here is used unless the
--finger-mode switch of the training script (or evaluator) is on; then:

Action (SharpaDualJointPositionActionTerm, a DualJointPositionActionTerm):
  * the policy outputs 29 values per robot (locked, track) or 29 + 44 (learn); clipping is D's (robot control config);
  * body DOFs: unchanged (action * scale + default pose, the same PD as D);
  * finger DOFs, PD target q_f:  locked = 0 (open pose);  track = q_ref_f(t);  learn = q_ref_f(t) + scale_f * a_f
    (scale_f = 0.25 * effort / kp by the holosoma rule with the Sharpa values, a_f clipped like the body actions), then
    clamped to the finger joint limits. q_ref_f(t) is the finger reference of the paired motion command at the current
    reference frame (the step 2 geometric grip; the "zero point" of the finger actions in learn mode).
    The full 73-DOF action handed to the PD is (q_f - default_f) / scale_f for the fingers, so the base class computes
    exactly target = q_f.
Observations (D's 158 per robot + 88): the body terms are D's terms restricted to the 29 body DOFs; added per robot:
  sharpa_finger_pos (44 finger joint positions) and sharpa_finger_ref (44 finger reference targets). Their names sort
  after D's terms, so the observation manager (sorted concatenation) appends them after D's 154 actor columns; the
  critic gets both robots' finger positions and references (527 + 176).
Rewards: D's terms; the two that read every DOF (action rate, joint limits) are restricted to the 29 body DOFs, so no
finger term enters the reward.
"""
from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.action.terms.marl import DualJointPositionActionTerm

BODY_DOFS = 29
FINGER_DOFS = 44
FINGER_MODES = ("locked", "track", "learn")


def _idx(env: Any):
    cache = getattr(env, "_system1_hand_idx", None)
    if cache is None:
        names = list(env.dof_names)
        fingers = [i for i, n in enumerate(names) if any(f"_{k}_" in n for k in ("thumb", "index", "middle", "ring", "pinky"))]
        body = [i for i in range(len(names)) if i not in fingers]
        if len(body) != BODY_DOFS or len(fingers) != FINGER_DOFS or body != list(range(BODY_DOFS)):
            raise ValueError(f"expected 29 body DOFs first, then 44 finger DOFs; got {len(body)} / {len(fingers)}")
        cache = (torch.as_tensor(body, device=env.device), torch.as_tensor(fingers, device=env.device))
        env._system1_hand_idx = cache
    return cache


def _command(env: Any):
    return env.command_manager.get_state("paired_motion_command")


def reference_joint_pos(env: Any) -> torch.Tensor:
    """[env, agent, dof] reference joint positions at the current reference frame."""
    return _command(env).agent_joint_pos


class SharpaDualJointPositionActionTerm(DualJointPositionActionTerm):
    def __init__(self, cfg, env: Any):
        super().__init__(cfg, env)                       # 73-DOF PD gains, action scales, buffers
        mode = cfg.params.get("finger_mode")
        if mode not in FINGER_MODES:
            raise ValueError(f"finger_mode must be one of {FINGER_MODES}, got {mode!r}")
        self.finger_mode = mode
        self._policy_dim = BODY_DOFS + (FINGER_DOFS if mode == "learn" else 0)
        self._action_dim = self.num_agents * self._policy_dim
        flat = (env.num_envs, self._action_dim)
        self._raw_actions = torch.zeros(flat, device=env.device)
        self._processed_actions = torch.zeros(flat, device=env.device)
        self._full_actions = torch.zeros(env.num_envs, self.num_agents, self._per_agent_action_dim, device=env.device)
        self._body = self._fingers = None
        self._f_lo = self._f_hi = self._f_scale = None
        self.finger_diag: dict[str, torch.Tensor] = {}

    def _bind(self):
        if self._fingers is None:
            body, fingers = _idx(self.env)
            lim = self.env.simulator.hard_dof_pos_limits
            self._body, self._fingers = body, fingers
            self._f_lo, self._f_hi = lim[fingers, 0].clone(), lim[fingers, 1].clone()
            self._f_scale = self.action_scales[fingers].clone()
            if not torch.all(self._f_scale > 0):
                raise ValueError("finger action scales must be positive")

    @property
    def policy_dim(self) -> int:
        return self._policy_dim

    @property
    def policy_action(self) -> torch.Tensor:
        """[env, agent * policy_dim]: what the policy sent (after D's clip); observation term `actions` reads its body part."""
        return self._processed_actions

    @property
    def agent_actions(self) -> torch.Tensor:
        """Full 73-DOF actions [env, agent, dof] in action units for the PD of the base class."""
        self._bind()
        a = self._processed_actions.view(self.env.num_envs, self.num_agents, self._policy_dim)
        default = self.env.default_dof_pos
        default = default.unsqueeze(1) if default.ndim == 2 else default.view(1, 1, -1)
        out = self._full_actions
        out[:, :, self._body] = a[:, :, :BODY_DOFS]
        if self.finger_mode == "locked":
            q_f = torch.zeros_like(out[:, :, self._fingers])
        else:
            q_f = reference_joint_pos(self.env)[:, :, self._fingers]
            if self.finger_mode == "learn":
                corr = a[:, :, BODY_DOFS:] * self._f_scale
                q_raw = q_f + corr
                q_f = torch.maximum(torch.minimum(q_raw, self._f_hi), self._f_lo)
                d = self.finger_diag  # running sums over policy steps; read and reset by system1.hand.hand_mappo.HandTrainDiag
                d["correction_abs_mean"] = d.get("correction_abs_mean", 0.0) + corr.abs().mean().detach()
                d["at_limit_frac"] = d.get("at_limit_frac", 0.0) + (q_raw != q_f).float().mean().detach()
                d["steps"] = d.get("steps", 0) + 1
            else:
                q_f = torch.maximum(torch.minimum(q_f, self._f_hi), self._f_lo)
        out[:, :, self._fingers] = (q_f - default[:, :, self._fingers]) / self._f_scale
        return out


# ----- observations (functions referenced by the observation config) -----

def _body(env, x):
    return x[..., _idx(env)[0]]


def hand_paired_motion_command(env: Any) -> torch.Tensor:
    """D's paired_motion_command restricted to the body DOFs: [E, A, 58] = (q_ref body, dq_ref body)."""
    c = _command(env).command
    n = env.num_dof
    body = _idx(env)[0]
    return torch.cat([c[..., :n][..., body], c[..., n:][..., body]], dim=-1)


def hand_paired_dof_pos(env: Any) -> torch.Tensor:
    return _body(env, env.simulator.agent_dof_pos - env.default_dof_pos[:, None, :])


def hand_paired_dof_vel(env: Any) -> torch.Tensor:
    return _body(env, env.simulator.agent_dof_vel)


def hand_paired_actions(env: Any) -> torch.Tensor:
    term = env.action_manager.get_term("joint_control")
    a = term.policy_action.view(env.num_envs, env.num_agents, term.policy_dim)
    return a[..., :BODY_DOFS]


def sharpa_finger_pos(env: Any) -> torch.Tensor:
    return env.simulator.agent_dof_pos[..., _idx(env)[1]]


def sharpa_finger_ref(env: Any) -> torch.Tensor:
    return reference_joint_pos(env)[..., _idx(env)[1]]


def hand_centralized_agent_dof_pos(env: Any) -> torch.Tensor:
    return hand_paired_dof_pos(env).reshape(env.num_envs, -1)


def hand_centralized_agent_dof_vel(env: Any) -> torch.Tensor:
    return hand_paired_dof_vel(env).reshape(env.num_envs, -1)


def hand_centralized_agent_actions(env: Any) -> torch.Tensor:
    return hand_paired_actions(env).reshape(env.num_envs, -1)


def hand_centralized_shared_motion_command(env: Any) -> torch.Tensor:
    return hand_paired_motion_command(env)[:, 0]


def sharpa_agent_finger_pos(env: Any) -> torch.Tensor:
    return sharpa_finger_pos(env).reshape(env.num_envs, -1)


def sharpa_agent_finger_ref(env: Any) -> torch.Tensor:
    return sharpa_finger_ref(env).reshape(env.num_envs, -1)


# ----- rewards restricted to the body DOFs (same arithmetic as holosoma.managers.reward.terms.marl) -----

def hand_penalty_action_rate(env: Any) -> torch.Tensor:
    term = env.action_manager.get_term("joint_control")
    shape = (env.num_envs, env.num_agents, term.policy_dim)
    actions = env.action_manager.action.reshape(shape)[..., :BODY_DOFS]
    previous = env.action_manager.prev_action.reshape(shape)[..., :BODY_DOFS]
    return torch.sum(torch.square(previous - actions), dim=-1).mean(dim=1)


def hand_limits_dof_pos(env: Any, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    body = _idx(env)[0]
    limits = env.simulator.hard_dof_pos_limits[body]
    midpoint = (limits[:, 0] + limits[:, 1]) / 2
    width = limits[:, 1] - limits[:, 0]
    lower = midpoint - 0.5 * width * soft_dof_pos_limit
    upper = midpoint + 0.5 * width * soft_dof_pos_limit
    positions = env.simulator.agent_dof_pos[..., body]
    excess = -(positions - lower).clip(max=0.0)
    excess += (positions - upper).clip(min=0.0)
    return torch.sum(excess, dim=-1).mean(dim=1)


__all__ = ["BODY_DOFS", "FINGER_DOFS", "FINGER_MODES", "SharpaDualJointPositionActionTerm"]
