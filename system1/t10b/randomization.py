"""T10b B: domain randomisation for the paired CORE4D training (two robots + one object), recipe v2 of
docs/system1/T10b_dr_ranges_zh.md (Kevin confirmed 2026-11-20): peer whole-body-tracking values as-is, the same terms for D
and C-in, PhysX / scene friction combine modes unchanged.

Used only with scripts/train_core4d_smalltable.py --system1-randomization (off: the training is unchanged). Every robot term is
drawn independently per environment and robot.

Robot (SONIC / BeyondMimic, OmniRetarget, holosoma locomotion):
  s1_robot_material_startup   material of every collision shape of both robots (feet and rubber hands included): static U[0.3, 1.6],
                              dynamic U[0.3, 1.2], restitution U[0, 0.5]; holosoma's _isaacsim_randomize_rigid_body_material
                              (Isaac Lab randomize_rigid_body_material, 64 buckets, a bucket per shape, as the holosoma WBT
                              preset), called once per robot.
  s1_robot_torso_com_startup  torso_link COM + U(x +/-0.025, y +/-0.05, z +/-0.075) m; holosoma's randomize_body_com (the WBT
                              preset's randomize_base_com_startup on Isaac Sim), called once per robot.
  s1_dof_pos_bias_startup     default joint position + U(+/-0.01) rad per env, robot and joint (holosoma's setup_dof_pos_bias,
                              per robot: env.default_dof_pos becomes [env, agent, dof]; the paired observation and action terms
                              read it). It acts as a joint-encoder offset b: D's observation q - default - b and PD target
                              default + b + scale * a; for C-in the frozen SONIC driver reads q - b and its target q* is sent as
                              q* + b (system1/scheme_a.py), so both schemes see the same offset.
  push_randomizer_state + randomize_push_schedule (holosoma)  per-env push interval U[1, 3] s;
  s1_apply_pushes             at each due step, both robots' root velocity (world frame) += U(x +/-0.5, y +/-0.5, z +/-0.2 m/s,
                              roll +/-0.52, pitch +/-0.52, yaw +/-0.78 rad/s), SONIC Table 4 per component (= holosoma WBT
                              preset max_push_vel; task book revision 2026-11-20, f4ed7a6), drawn per robot (holosoma's
                              apply_pushes pushes robot 0 only).
  s1_pd_gain_reset            kp, kd scale U[0.9, 1.1] per env, robot and joint at every reset (the paired action term's own
                              [env, agent, dof] scale tensors; holosoma's ActuatorRandomizerState is [env, dof] and shared).
Object (holosoma object preset, OmniRetarget):
  s1_object_material_startup  static U[0.1, 0.6], dynamic U[0.1, 0.6], restitution 0, the body of holosoma's
                              randomize_object_rigid_body_material_startup (same helper and buckets; the function itself rejects
                              the paired simulator class); object-ground and object-hand friction follow from it with the scene's
                              combine modes.
  s1_object_mass_startup      mass x U[0.8, 1.2] with the inertia scaled by the same ratio (holosoma's mass helper, Isaac Lab
                              operation 'scale', recompute_inertia).
  s1_object_inertia_startup   inertia x U[0.5, 1.5] on top (one factor per env for the whole tensor, so it stays a physical
                              inertia; holosoma's per-element inertia term can break the triangle inequality).
  s1_object_com_startup       COM + U(+/-0.08 m) per axis.
Dropped: action delay. Unchanged: holosoma's training observation noise (the evaluation configuration disables it).
s1_readback_check (last setup term) reads every randomised quantity back from PhysX / the env for all environments, checks the
ranges and raises otherwise; final_readback() adds the PD scales and pushes at the end of training. Values of the first
``log_envs`` environments and min / max over all environments are kept in env._s1_rand_log (randomization_samples.json).
"""
from __future__ import annotations

from typing import Any

import torch

DEFAULTS = dict(
    recipe="docs/system1/T10b_dr_ranges_zh.md (Kevin confirmed 2026-11-20; pushes revised f4ed7a6)",
    robot_static_friction=(0.3, 1.6), robot_dynamic_friction=(0.3, 1.2), robot_restitution=(0.0, 0.5), material_buckets=64,
    torso_com_offset_m=dict(x=(-0.025, 0.025), y=(-0.05, 0.05), z=(-0.075, 0.075)),
    dof_pos_bias_rad=(-0.01, 0.01),
    push_interval_s=(1.0, 3.0), push_max_vel=(0.5, 0.5, 0.2, 0.52, 0.52, 0.78),
    kp_scale=(0.9, 1.1), kd_scale=(0.9, 1.1),
    object_static_friction=(0.1, 0.6), object_dynamic_friction=(0.1, 0.6), object_restitution=(0.0, 0.0),
    object_mass_scale=(0.8, 1.2), object_inertia_scale=(0.5, 1.5), object_com_offset_m=(-0.08, 0.08),
)
ROBOTS = ("robot", "robot_1")
TORSO = "torso_link"
EPS = 1e-5


def _log(env, key, value, n=None):
    log = getattr(env, "_s1_rand_log", None)
    if log is None:
        log = env._s1_rand_log = {}
    v = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    if n is not None and hasattr(v, "__getitem__"):
        v = v[:n]
    log[key] = v.tolist() if hasattr(v, "tolist") else v


def _u(lo, hi, shape, device="cpu"):
    return lo + (hi - lo) * torch.rand(shape, device=device)


def _all(env):
    return torch.arange(env.num_envs, dtype=torch.long, device="cpu")


def _cfg(sim, name, bodies=".*"):
    from isaaclab.managers import SceneEntityCfg

    c = SceneEntityCfg(name, body_names=bodies)
    c.resolve(sim.scene)
    return c


# ---------------------------------------------------------------- robot
def s1_robot_material_startup(env: Any, *, static_range, dynamic_range, restitution_range, num_buckets: int, **_):
    from holosoma.managers.randomization.terms.locomotion import _isaacsim_randomize_rigid_body_material

    sim = env.simulator
    obj_before = sim._object.root_physx_view.get_material_properties().clone()
    for name in ROBOTS:
        _isaacsim_randomize_rigid_body_material(sim, _all(env), _cfg(sim, name), static_friction_range=tuple(static_range),
                                                dynamic_friction_range=tuple(dynamic_range), restitution_range=tuple(restitution_range),
                                                num_buckets=int(num_buckets))
    if not torch.equal(sim._object.root_physx_view.get_material_properties(), obj_before):
        raise RuntimeError("robot material randomisation changed the object material")


def s1_robot_torso_com_startup(env: Any, *, com_range, **_):
    from holosoma.simulator.isaacsim.events import randomize_body_com

    sim = env.simulator
    lo = torch.tensor([com_range[k][0] for k in "xyz"], dtype=torch.float)
    hi = torch.tensor([com_range[k][1] for k in "xyz"], dtype=torch.float)
    for name in ROBOTS:
        art = sim.scene[name]
        # holosoma's function resets to sim.default_coms (read from robot 0 at startup); robot_1 is the same model
        if not torch.allclose(art.root_physx_view.get_coms(), sim.default_coms):
            raise RuntimeError(f"{name}: startup COMs differ from robot 0's default_coms")
        randomize_body_com(sim, _all(env), _cfg(sim, name, [TORSO]), (lo, hi), operation="add", distribution="uniform",
                           num_envs=env.num_envs)


def s1_dof_pos_bias_startup(env: Any, *, bias_range, **_):
    base = env.default_dof_pos_base  # [1, dof]
    b = _u(bias_range[0], bias_range[1], (env.num_envs, env.num_agents, base.shape[-1]), device=base.device)
    env._s1_dof_bias = b
    env.default_dof_pos = base[:, None, :] + b  # [env, agent, dof]


def s1_apply_pushes(env: Any, **_):
    state = env.randomization_manager.get_state("push_randomizer_state")
    if state is None:
        raise AttributeError("push_randomizer_state is not registered")
    if env.is_evaluating or not state.enabled:
        return
    ids = state.due_envs(env.dt)
    if ids.numel() == 0:
        return
    state.zero_counters(ids)
    state.resample(ids)
    sim = env.simulator
    m = state.max_push_vel.to(env.device)
    dv = (torch.rand(ids.numel(), env.num_agents, 6, device=env.device) * 2 - 1) * m
    rs = sim.agent_root_states[ids].clone()  # [n, agent, 13] xyzw: pos, quat, lin vel w, ang vel w
    rs[..., 7:13] += dv
    sim.set_agent_root_states(ids, rs)
    st = getattr(env, "_s1_push_stats", None)
    if st is None:
        st = env._s1_push_stats = dict(events=0, pushes=0, max_abs=torch.zeros(6, device=env.device), first=[])
    st["events"] += 1
    st["pushes"] += int(ids.numel()) * env.num_agents
    st["max_abs"] = torch.maximum(st["max_abs"], dv.abs().amax(dim=(0, 1)))
    if len(st["first"]) < 3:
        st["first"].append(dict(step=int(env.common_step_counter) if hasattr(env, "common_step_counter") else None,
                                env_ids=ids[:5].cpu().tolist(), dv_first_env_per_robot=dv[0].cpu().tolist()))


def s1_pd_gain_reset(env: Any, env_ids, *, kp_range, kd_range, log_envs: int = 5, **_):
    term = env.action_manager.get_term("joint_control")
    idx = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device).long()
    if idx.numel() == 0:
        return
    shape = (idx.numel(), term.num_agents, term.per_agent_action_dim)
    kp = _u(kp_range[0], kp_range[1], shape, device=env.device)
    kd = _u(kd_range[0], kd_range[1], shape, device=env.device)
    term.update_pd_scales(idx, kp, kd)
    if not getattr(env, "_s1_pd_logged", False):  # first reset: values of the first envs
        env._s1_pd_logged = True
        _log(env, "pd_first_reset_env_ids", idx, log_envs)
        _log(env, "kp_scale_first_envs_robot0_joints0to3", kp[:, 0, :4], log_envs)
        _log(env, "kp_scale_first_envs_robot1_joints0to3", kp[:, 1, :4], log_envs)
        state = env.randomization_manager.get_state("push_randomizer_state")
        if state is not None and state.push_interval_s is not None:
            _log(env, "push_interval_s_first_envs", state.push_interval_s, log_envs)


# ---------------------------------------------------------------- object
def s1_object_material_startup(env: Any, *, static_range, dynamic_range, restitution_range, num_buckets: int, **_):
    """holosoma's randomize_object_rigid_body_material_startup body (same helper, same 64 buckets, SceneEntityCfg('object'));
    the holosoma function itself refuses the paired simulator (class-name check for 'IsaacSim', ours is DualRobotIsaacSim)."""
    from holosoma.managers.randomization.terms.locomotion import _isaacsim_randomize_rigid_body_material

    sim = env.simulator
    robots_before = [sim.scene[n].root_physx_view.get_material_properties().clone() for n in ROBOTS]
    _isaacsim_randomize_rigid_body_material(sim, _all(env), _cfg(sim, "object"), static_friction_range=tuple(static_range),
                                            dynamic_friction_range=tuple(dynamic_range), restitution_range=tuple(restitution_range),
                                            num_buckets=int(num_buckets))
    if not all(torch.equal(sim.scene[n].root_physx_view.get_material_properties(), b) for n, b in zip(ROBOTS, robots_before)):
        raise RuntimeError("object material randomisation changed a robot material")


def s1_object_mass_startup(env: Any, *, scale_range, **_):
    from holosoma.managers.randomization.terms.locomotion import _isaacsim_randomize_rigid_body_mass

    sim = env.simulator
    _isaacsim_randomize_rigid_body_mass(sim, _all(env), _cfg(sim, "object"), tuple(scale_range), operation="scale")


def s1_object_inertia_startup(env: Any, *, scale_range, **_):
    view = env.simulator._object.root_physx_view
    inertia = view.get_inertias().clone()
    s = _u(scale_range[0], scale_range[1], (env.num_envs,))
    env._s1_object_inertia_scale = s
    view.set_inertias(inertia * s.reshape(-1, *([1] * (inertia.ndim - 1))), _all(env))


def s1_object_com_startup(env: Any, *, offset_range, **_):
    view = env.simulator._object.root_physx_view
    coms = view.get_coms().clone()  # [env, 7] (pos, quat) in the body frame
    env._s1_object_com_nominal = coms.clone()
    off = _u(offset_range[0], offset_range[1], (env.num_envs, 3))
    coms[..., :3] += off
    view.set_coms(coms, _all(env))


# ---------------------------------------------------------------- read back and check
def _mm(x):
    return [float(x.min()), float(x.max())]


def _within(x, lo, hi):
    return bool((x >= lo - EPS).all() and (x <= hi + EPS).all())


def s1_readback_check(env: Any, *, ranges, log_envs: int = 5, **_):
    from system1.t10b.hand_material import HAND_BODIES, shapes_per_body

    r = ranges
    sim = env.simulator
    chk = {}
    for k, name in enumerate(ROBOTS):
        art = sim.scene[name]
        mat = art.root_physx_view.get_material_properties()
        chk[f"{name}_static_friction_minmax"] = _mm(mat[..., 0])
        chk[f"{name}_dynamic_friction_minmax"] = _mm(mat[..., 1])
        chk[f"{name}_restitution_minmax"] = _mm(mat[..., 2])
        chk[f"{name}_material_ok"] = (_within(mat[..., 0], *r["robot_static_friction"]) and _within(mat[..., 1], *r["robot_dynamic_friction"])
                                      and _within(mat[..., 2], *r["robot_restitution"]))
        chk[f"{name}_distinct_materials_shape_level"] = int(torch.unique(mat.reshape(-1, 3), dim=0).shape[0])
        mapping, _, _ = shapes_per_body(art)
        hands = [i for b in HAND_BODIES for i in mapping[b]]
        _log(env, f"{name}_hand_material_first_envs", mat[:, hands], log_envs)
        feet = [i for b, ii in mapping.items() if "ankle_roll" in b for i in ii]
        _log(env, f"{name}_foot_material_first_envs", mat[:, feet], log_envs)
        coms = art.root_physx_view.get_coms()
        t = art.find_bodies(TORSO)[0][0]
        off = coms[:, t, :3] - sim.default_coms[:, t, :3]
        others = torch.ones(coms.shape[1], dtype=torch.bool)
        others[t] = False
        chk[f"{name}_torso_com_offset_minmax_xyz"] = [_mm(off[:, j]) for j in range(3)]
        # other bodies and the torso COM orientation: unchanged up to the PhysX write / read round trip (float32)
        chk[f"{name}_other_bodies_com_max_change"] = float((coms[:, others] - sim.default_coms[:, others]).abs().max())
        chk[f"{name}_torso_com_ok"] = (all(_within(off[:, j], *r["torso_com_offset_m"][a]) for j, a in enumerate("xyz"))
                                       and torch.allclose(coms[:, others], sim.default_coms[:, others], rtol=0.0, atol=1e-6)
                                       and torch.allclose(coms[:, t, 3:], sim.default_coms[:, t, 3:], rtol=0.0, atol=1e-6))
        _log(env, f"{name}_torso_com_offset_first_envs", off, log_envs)
        b = env._s1_dof_bias[:, k]
        chk[f"{name}_dof_pos_bias_minmax"] = _mm(b)
        # (default + b) - default differs from b by float32 round-off (~3e-8)
        chk[f"{name}_dof_pos_bias_ok"] = _within(b, *r["dof_pos_bias_rad"]) and torch.allclose(
            env.default_dof_pos[:, k] - env.default_dof_pos_base, b, rtol=0.0, atol=1e-6)
        _log(env, f"{name}_dof_pos_bias_first_envs_joints0to3", b[:, :4], log_envs)
    obj = sim._object
    view = obj.root_physx_view
    mat = view.get_material_properties()
    chk["object_static_friction_minmax"] = _mm(mat[..., 0])
    chk["object_dynamic_friction_minmax"] = _mm(mat[..., 1])
    chk["object_restitution_minmax"] = _mm(mat[..., 2])
    chk["object_material_ok"] = (_within(mat[..., 0], *r["object_static_friction"]) and _within(mat[..., 1], *r["object_dynamic_friction"])
                                 and _within(mat[..., 2], *r["object_restitution"]))
    chk["object_shapes"] = int(mat.shape[1])
    _log(env, "object_material_first_envs", mat, log_envs)
    m = view.get_masses().reshape(env.num_envs, -1)[:, 0]
    m0 = obj.data.default_mass.reshape(env.num_envs, -1)[:, 0].cpu()
    ratio = m / m0
    chk["object_nominal_mass_kg"] = _mm(m0)
    chk["object_mass_kg_minmax"] = _mm(m)
    chk["object_mass_ok"] = _within(ratio, *r["object_mass_scale"])
    _log(env, "object_mass_kg_first_envs", m, log_envs)
    inertia = view.get_inertias().reshape(env.num_envs, -1)
    i0 = obj.data.default_inertia.reshape(env.num_envs, -1).cpu()
    s = env._s1_object_inertia_scale
    expect = i0 * (ratio * s)[:, None]
    chk["object_inertia_scale_minmax"] = _mm(s)
    chk["object_total_inertia_factor_minmax"] = _mm(ratio * s)
    chk["object_inertia_ok"] = _within(s, *r["object_inertia_scale"]) and torch.allclose(inertia, expect, rtol=1e-4, atol=1e-7)
    _log(env, "object_inertia_scale_first_envs", s, log_envs)
    coms = view.get_coms()
    off = coms[:, :3] - env._s1_object_com_nominal[:, :3]
    chk["object_com_offset_minmax_xyz"] = [_mm(off[:, j]) for j in range(3)]
    chk["object_com_ok"] = all(_within(off[:, j], *r["object_com_offset_m"]) for j in range(3)) and torch.allclose(
        coms[:, 3:], env._s1_object_com_nominal[:, 3:], rtol=0.0, atol=1e-6)
    _log(env, "object_com_offset_first_envs", off, log_envs)
    chk["all_ok"] = all(v for k, v in chk.items() if k.endswith("_ok"))
    _log(env, "startup_range_check", chk)
    if not chk["all_ok"]:
        raise RuntimeError(f"T10b randomisation outside the configured ranges: {chk}")


def final_readback(env: Any, ranges) -> dict:
    """PD scales (all envs, robots, joints, at the end of training) and pushes applied so far, checked against the ranges."""
    term = env.action_manager.get_term("joint_control")
    kp, kd = term.get_pd_scale_tensors()
    st = getattr(env, "_s1_push_stats", None) or dict(events=0, pushes=0, max_abs=torch.zeros(6), first=[])
    mx = torch.as_tensor(ranges["push_max_vel"], dtype=torch.float)
    out = dict(kp_scale_minmax=_mm(kp), kd_scale_minmax=_mm(kd), pd_ok=_within(kp, *ranges["kp_scale"]) and _within(kd, *ranges["kd_scale"]),
               kp_scale_distinct_robots=bool((kp[:, 0] != kp[:, 1]).any()), push_events=st["events"], robot_pushes=st["pushes"],
               push_max_abs_dv=st["max_abs"].cpu().tolist(), push_ok=bool((st["max_abs"].cpu() <= mx + EPS).all()) and st["events"] > 0,
               first_pushes=st["first"])
    _log(env, "final_readback", out)
    return out


def randomization_config(base_cfg, ranges=None):
    """holosoma RandomizationManagerCfg = the training's own config (its fixed object material first) + the terms above."""
    from dataclasses import replace

    from holosoma.config_types.randomization import RandomizationTermCfg as T

    r = dict(DEFAULTS, **(ranges or {}))
    f = "system1.t10b.randomization:"
    loco = "holosoma.managers.randomization.terms.locomotion:"
    push_state = T(func=loco + "PushRandomizerState", params=dict(push_interval_s=list(r["push_interval_s"]), max_push_vel=list(r["push_max_vel"]), enabled=True))
    setup = dict(base_cfg.setup_terms)
    setup.update(
        push_randomizer_state=push_state,
        s1_robot_material_startup=T(func=f + "s1_robot_material_startup", params=dict(
            static_range=list(r["robot_static_friction"]), dynamic_range=list(r["robot_dynamic_friction"]),
            restitution_range=list(r["robot_restitution"]), num_buckets=r["material_buckets"])),
        s1_robot_torso_com_startup=T(func=f + "s1_robot_torso_com_startup", params=dict(com_range={k: list(v) for k, v in r["torso_com_offset_m"].items()})),
        s1_dof_pos_bias_startup=T(func=f + "s1_dof_pos_bias_startup", params=dict(bias_range=list(r["dof_pos_bias_rad"]))),
        s1_object_material_startup=T(func=f + "s1_object_material_startup", params=dict(
            static_range=list(r["object_static_friction"]), dynamic_range=list(r["object_dynamic_friction"]),
            restitution_range=list(r["object_restitution"]), num_buckets=r["material_buckets"])),
        s1_object_mass_startup=T(func=f + "s1_object_mass_startup", params=dict(scale_range=list(r["object_mass_scale"]))),
        s1_object_inertia_startup=T(func=f + "s1_object_inertia_startup", params=dict(scale_range=list(r["object_inertia_scale"]))),
        s1_object_com_startup=T(func=f + "s1_object_com_startup", params=dict(offset_range=list(r["object_com_offset_m"]))),
        s1_readback_check=T(func=f + "s1_readback_check", params=dict(ranges=r)),
    )
    reset = dict(base_cfg.reset_terms)
    reset.update(
        push_randomizer_state=T(func=loco + "PushRandomizerState"),
        randomize_push_schedule=T(func=loco + "randomize_push_schedule"),
        s1_pd_gain_reset=T(func=f + "s1_pd_gain_reset", params=dict(kp_range=list(r["kp_scale"]), kd_range=list(r["kd_scale"]))),
    )
    step = dict(base_cfg.step_terms)
    step.update(
        push_randomizer_state=T(func=loco + "PushRandomizerState"),
        s1_apply_pushes=T(func=f + "s1_apply_pushes"),
    )
    return replace(base_cfg, setup_terms=setup, reset_terms=reset, step_terms=step), r
