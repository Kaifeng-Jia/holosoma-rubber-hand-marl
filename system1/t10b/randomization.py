"""T10b B: domain randomisation for the paired CORE4D training (two robots + one object), on top of holosoma's own terms.

Used only with scripts/train_core4d_smalltable.py --system1-randomization (off: the training is unchanged). Inventory of holosoma's
existing terms and why each is or is not reused: results_system1/T10b_friction_margin/retrain/README_CN.md.

Setup terms (once per environment, at startup):
  s1_object_material_startup  object material static = dynamic = 0.5 x U(0.5, 1.25) per env (object-ground friction follows it:
                              the ground's combine mode is multiply x 1.0); replaces the fixed 0.5 term's value after it ran.
  s1_hand_material_startup    rubber-hand material static = dynamic = 1.0 x U(0.5, 1.25) per env and robot (both hands of a robot
                              share it; shapes found as in system1/t10b/hand_material.py); hand-object friction = average.
  s1_object_mass_startup      object mass x U(0.8, 1.2) per env, inertia scaled by the same factor (COM unchanged).
  s1_target_delay_setup       installs a 0-1 control-step delay on the PD target (the actual joint target, so it covers D's
                              default + scale * a and C-in's SONIC q*): target(t) = target(t-1) for envs with delay 1.
Reset terms (each episode reset):
  s1_pd_gain_reset            kp and kd scales U(0.9, 1.1) per env, robot and joint (holosoma's own scale tensors of the
                              paired joint_control term, [env, agent, dof]).
  s1_target_delay_reset       delay in {0, 1} control steps per env; the first step after a reset is never delayed.
Observation noise is holosoma's existing training observation noise (unchanged; the evaluation configuration disables it).
Every sampled value of the first ``log_envs`` environments is kept in env._s1_rand_log (written to randomization_samples.json).
"""
from __future__ import annotations

from typing import Any

import torch

from system1.t10b.hand_material import HAND_BODIES, shapes_per_body

DEFAULTS = dict(object_friction_base=0.5, hand_friction_base=1.0, friction_scale=(0.5, 1.25), object_mass_scale=(0.8, 1.2),
                kp_scale=(0.9, 1.1), kd_scale=(0.9, 1.1), target_delay_steps=(0, 1))


def _log(env, key, value, n):
    log = getattr(env, "_s1_rand_log", None)
    if log is None:
        log = env._s1_rand_log = {}
    log.setdefault(key, []).append(value[:n].tolist() if hasattr(value, "tolist") else value)


def _u(lo, hi, shape, device="cpu", gen=None):
    return lo + (hi - lo) * torch.rand(shape, device=device, generator=gen)


def s1_object_material_startup(env: Any, *, base: float, scale_range, log_envs: int = 5, **_):
    view = env.simulator._object.root_physx_view
    mats = view.get_material_properties().clone()
    s = _u(scale_range[0], scale_range[1], (env.num_envs,)) * base
    mats[..., 0] = s[:, None]
    mats[..., 1] = s[:, None]
    mats[..., 2] = 0.0
    view.set_material_properties(mats, torch.arange(env.num_envs, dtype=torch.long))
    if not torch.equal(view.get_material_properties(), mats):
        raise RuntimeError("PhysX did not retain the randomised object material")
    _log(env, "object_friction", s, log_envs)


def s1_hand_material_startup(env: Any, *, base: float, scale_range, log_envs: int = 5, **_):
    sim = env.simulator
    out = []
    for art in (sim._robot, getattr(sim, "_robot_1", None)):
        if art is None:
            continue
        mapping, _, _ = shapes_per_body(art)
        idx = [i for b in HAND_BODIES for i in mapping[b]]
        view = art.root_physx_view
        before = view.get_material_properties().clone()
        mats = before.clone()
        s = _u(scale_range[0], scale_range[1], (env.num_envs,)) * base
        for i in idx:
            mats[:, i, 0] = s
            mats[:, i, 1] = s
            mats[:, i, 2] = 0.0
        view.set_material_properties(mats, torch.arange(env.num_envs, dtype=torch.long))
        after = view.get_material_properties()
        if not torch.equal(after, mats):
            raise RuntimeError("PhysX did not retain the randomised hand material")
        other = [i for i in range(after.shape[1]) if i not in idx]
        if not torch.equal(after[:, other], before[:, other]):
            raise RuntimeError("hand material randomisation changed a non-hand shape")
        out.append(s)
    _log(env, "hand_friction_per_robot", torch.stack(out, dim=1), log_envs)


def s1_object_mass_startup(env: Any, *, scale_range, log_envs: int = 5, **_):
    view = env.simulator._object.root_physx_view
    ids = torch.arange(env.num_envs, dtype=torch.long)
    m = view.get_masses().clone()
    inertia = view.get_inertias().clone()
    s = _u(scale_range[0], scale_range[1], (env.num_envs,))
    view.set_masses(m * s.reshape(-1, *([1] * (m.ndim - 1))), ids)
    view.set_inertias(inertia * s.reshape(-1, *([1] * (inertia.ndim - 1))), ids)
    _log(env, "object_mass_kg", (m.reshape(env.num_envs, -1)[:, 0] * s), log_envs)


def s1_pd_gain_reset(env: Any, env_ids, *, kp_range, kd_range, log_envs: int = 5, **_):
    term = env.action_manager.get_term("joint_control")
    idx = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device).long()
    if idx.numel() == 0:
        return
    shape = (idx.numel(), term.num_agents, term.per_agent_action_dim)
    kp = _u(kp_range[0], kp_range[1], shape, device=env.device)
    kd = _u(kd_range[0], kd_range[1], shape, device=env.device)
    term.update_pd_scales(idx, kp, kd)
    if getattr(env, "_s1_pd_logged", 0) < 2:  # first two reset calls only
        env._s1_pd_logged = getattr(env, "_s1_pd_logged", 0) + 1
        _log(env, "pd_reset_env_ids", idx[:log_envs].cpu(), log_envs)
        _log(env, "kp_scale_first_envs_robot0_joints0to3", kp[:log_envs, 0, :4].cpu(), log_envs)


class _TargetDelay:
    def __init__(self, env, lo, hi):
        self.env = env
        self.term = env.action_manager.get_term("joint_control")
        self.lo, self.hi = int(lo), int(hi)
        self.delay = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.prev = None
        self.written = None
        orig = self.term.process_actions

        def process_actions(actions):
            orig(actions)
            t = self.term
            ov = t.joint_target_override
            if ov is not None and ov is not self.written:  # C-in: SONIC's q* written by the driver this step
                cur = ov.clone()
            else:  # D: holosoma's P target default + scale * clipped action
                d = env.default_dof_pos
                d = d.unsqueeze(1) if d.ndim == 2 else d
                cur = t._actions_after_delay * t.action_scales + d
            if self.prev is None:
                self.prev = cur.clone()
            use_prev = (self.delay == 1) & ~self.fresh
            eff = torch.where(use_prev[:, None, None], self.prev, cur)
            self.prev = cur.clone()
            self.fresh[:] = False
            t.joint_target_override = eff
            self.written = eff
        self.term.process_actions = process_actions

    def reset(self, env_ids):
        idx = torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None else torch.as_tensor(env_ids, device=self.env.device).long()
        if idx.numel() == 0:
            return
        self.delay[idx] = torch.randint(self.lo, self.hi + 1, (idx.numel(),), device=self.env.device)
        self.fresh[idx] = True


def s1_target_delay_setup(env: Any, *, steps_range, log_envs: int = 5, **_):
    """Randomisation setup runs before the action manager is set up: keep the range; the delay is installed at the first reset."""
    env._s1_target_delay_range = list(steps_range)
    env._s1_target_delay = None


def s1_target_delay_reset(env: Any, env_ids, *, log_envs: int = 5, **_):
    if getattr(env, "_s1_target_delay_range", None) is None:
        return
    td = getattr(env, "_s1_target_delay", None)
    if td is None:
        td = env._s1_target_delay = _TargetDelay(env, *env._s1_target_delay_range)
    td.reset(env_ids)
    if getattr(env, "_s1_delay_logged", 0) < 2:
        env._s1_delay_logged = getattr(env, "_s1_delay_logged", 0) + 1
        _log(env, "target_delay_steps_first_envs", td.delay[:log_envs].cpu(), log_envs)
        _log(env, "target_delay_fraction_delayed_all_envs", [float(td.delay.float().mean().item())], 1)


def randomization_config(base_cfg, ranges=None):
    """holosoma RandomizationManagerCfg = the training's own config + the terms above (setup order: holosoma's fixed object
    material first, then the randomised values)."""
    from dataclasses import replace

    from holosoma.config_types.randomization import RandomizationTermCfg as T
    r = dict(DEFAULTS, **(ranges or {}))
    f = "system1.t10b.randomization:"
    setup = dict(base_cfg.setup_terms)
    setup.update(
        s1_object_material_startup=T(func=f + "s1_object_material_startup", params=dict(base=r["object_friction_base"], scale_range=list(r["friction_scale"]))),
        s1_hand_material_startup=T(func=f + "s1_hand_material_startup", params=dict(base=r["hand_friction_base"], scale_range=list(r["friction_scale"]))),
        s1_object_mass_startup=T(func=f + "s1_object_mass_startup", params=dict(scale_range=list(r["object_mass_scale"]))),
        s1_target_delay_setup=T(func=f + "s1_target_delay_setup", params=dict(steps_range=list(r["target_delay_steps"]))),
    )
    reset = dict(base_cfg.reset_terms)
    reset.update(
        s1_pd_gain_reset=T(func=f + "s1_pd_gain_reset", params=dict(kp_range=list(r["kp_scale"]), kd_range=list(r["kd_scale"]))),
        s1_target_delay_reset=T(func=f + "s1_target_delay_reset", params={}),
    )
    return replace(base_cfg, setup_terms=setup, reset_terms=reset), r
