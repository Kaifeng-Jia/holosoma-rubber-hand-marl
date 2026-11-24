"""T13 v4 section 2.4 checks inside the real training environment (called by the training scripts' --t13v4-check OUT).

Independent hand computations are numpy float64 from raw simulator / command tensors (the checks may copy to the CPU; the
terms under test may not - torch.cuda sync debug mode counts their synchronisations).
  obs:       group shapes, term order and slices; critic holds both robots' 58-D commands; observation v1 vs the
             definitions (noise off: <= 1e-5; noise on: with the state's offsets / bias, <= 1e-5; noise state ranges);
  contact:   reference replay (every frame written into the simulator, one control step with PD targets = the reference
             joints, the contact terms evaluated), per frame vs the definitions: reward, per-robot penalty count, lost flags
             and the 25-step counter; hand forces of the link sensor vs the object-hand sensor where both exist;
  sync:      GPU->CPU synchronisations inside the new terms (torch.cuda.set_sync_debug_mode).
"""
from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch


def _np(t):
    return t.detach().double().cpu().numpy()


def qmat(q):  # xyzw
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def group_layout(env):
    om = env.observation_manager
    out = {}
    for g, cfg in om.cfg.groups.items():
        names = sorted(cfg.terms)
        out[g] = names
    return out


def expected_v1(env, st, bias):
    from holosoma.managers.observation.terms.marl_v1 import NECK_POINT_TORSO_M, OBJECT_CLIP_M, PARTNER_CLIP_M
    sim = env.simulator
    c = env.command_manager.get_state("paired_motion_command")
    ti = list(sim._body_list).index("torso_link")
    tp, tq = _np(sim.agent_rigid_body_pos[:, :, ti]), _np(sim.agent_rigid_body_rot[:, :, ti])
    Rt = qmat(tq)
    neck = tp + np.einsum("eaij,j->eai", Rt, np.array(NECK_POINT_TORSO_M))
    obj = np.repeat(_np(c.simulator_object_pos_w)[:, None], 2, 1)
    if st.noise_enabled:
        neck = neck + _np(st.neck_offset + st.neck_step)
        obj = obj + _np(st.object_offset + st.object_step)
    yaw = np.arctan2(Rt[..., 1, 0], Rt[..., 0, 0]) + bias
    R = np.zeros(yaw.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = np.cos(yaw), -np.sin(yaw), np.sin(yaw), np.cos(yaw), 1.0
    partner = np.clip(np.einsum("eaji,eaj->eai", R, neck[:, [1, 0]] - neck)[..., :2], -PARTNER_CLIP_M, PARTNER_CLIP_M)
    objrel = np.clip(np.einsum("eaji,eaj->eai", R, obj - neck), -OBJECT_CLIP_M, OBJECT_CLIP_M)
    Rb = np.zeros_like(R)
    Rb[..., 0, 0], Rb[..., 0, 1], Rb[..., 1, 0], Rb[..., 1, 1], Rb[..., 2, 2] = np.cos(bias), -np.sin(bias), np.sin(bias), np.cos(bias), 1.0
    ai = c.ref_body_index
    Ract = np.einsum("eaij,eajk->eaik", Rb, qmat(_np(sim.agent_rigid_body_rot[:, :, ai])))
    rel = np.einsum("eaji,eajk->eaik", Ract, qmat(_np(c.agent_ref_quat_w)))
    return rel[..., :2].reshape(env.num_envs, 2, 6), np.concatenate([partner, objrel], -1)


def check_obs(env, observations, noise_expected: bool) -> dict:
    res = dict(group_shapes={k: list(v.shape) for k, v in observations.items() if hasattr(v, "shape")}, term_order=group_layout(env))
    E = env.num_envs
    om = env.observation_manager
    dims = {}
    for g, names in res["term_order"].items():
        cfg = om.cfg.groups[g]
        dims[g] = {n: int(om._compute_term(g, n, cfg.terms[n]).shape[-1]) for n in names}
    res["term_dims"] = dims
    c = env.command_manager.get_state("paired_motion_command")
    crit = observations["critic_obs"]
    names = res["term_order"]["critic_obs"]
    off = 0
    for n in names:
        w = dims["critic_obs"][n]
        if n == "shared_motion_command":
            block = crit[:, off:off + w]
            res["critic_shared_motion_command"] = dict(width=w, equals_both_robots=bool(torch.allclose(block, c.command.reshape(E, -1).to(block.dtype), atol=1e-6)),
                                                       robot1_block_present=w == 116)
        off += w
    res["critic_width_from_terms"] = off
    # observation v1 vs the definitions on the current state (term instances called directly: exact pre-noise values)
    st = env._system1_obs_v1
    inst = om._term_instances
    ori = inst["actor_obs"]["motion_ref_ori_b"](env)          # owner: draws this step's noise
    par = inst["teammate_obs"]["neck_partner_rel_pos_b"](env)
    obj = inst["teammate_obs"]["object_rel_pos_b"](env)
    bias = _np(st.heading_bias) if st.noise_enabled else np.zeros((E, 2))
    e_ori, e_team = expected_v1(env, st, bias)
    got_team = np.concatenate([np.clip(_np(par), -1, 1), np.clip(_np(obj), -2, 2)], -1)
    res["obs_v1_max_abs_err"] = dict(ref_ori=float(np.abs(_np(ori) - e_ori).max()), teammate=float(np.abs(got_team - e_team).max()))
    res["obs_v1_teammate_group_vs_expected"] = float(np.abs(_np(observations["teammate_obs"]) - e_team).max()) if not st.noise_enabled else None
    res["noise_enabled"] = bool(st.noise_enabled)
    res["noise_expected"] = noise_expected
    if st.noise_enabled:
        from holosoma.managers.observation.terms.marl_v1 import NOISE_V1
        stats = {}
        for name, t, half in (("neck_offset", st.neck_offset, NOISE_V1["neck_episode_offset_m"]), ("neck_step", st.neck_step, NOISE_V1["neck_step_m"]),
                              ("object_offset", st.object_offset, NOISE_V1["object_episode_offset_m"]), ("object_step", st.object_step, NOISE_V1["object_step_m"]),
                              ("heading_bias_rad", st.heading_bias, math.radians(NOISE_V1["heading_bias_deg"]))):
            x = _np(t).ravel()
            stats[name] = dict(n=int(x.size), min=float(x.min()), max=float(x.max()), mean=float(x.mean()), std=float(x.std()), uniform_std=half / math.sqrt(3),
                               within_range=bool(x.min() >= -half and x.max() <= half))
        stats["object_noise_shape"] = list(st.object_offset.shape)  # [E, 1, 3]: one draw per environment, shared by the robots
        res["noise_samples"] = stats
    res["clip_fraction"] = dict(partner=float(st.partner_clipped) / max(float(st.samples), 1.0), object=float(st.object_clipped) / max(float(st.samples), 1.0))
    return res


def replay_contact(env, labels_file: str, *, stride_envs: bool = True) -> dict:
    """Every reference frame once (env e plays a contiguous block of frames), one control step per frame with PD targets =
    reference joints; terms vs numpy definitions."""
    from holosoma.managers.reward.terms.contact_v1 import CONTACT_V1, HANDS, ContactV1Labels
    E = env.num_envs
    c = env.command_manager.get_state("paired_motion_command")
    lab = ContactV1Labels.of(env, labels_file)
    F = min(lab.frames, c.reference.num_frames)
    C = math.ceil(F / E)
    start = torch.arange(E, device=env.device) * C
    rew = env.reward_manager._term_instances["contact_v1"]
    rparams = env.reward_manager.cfg.terms["contact_v1"].params
    term = env.termination_manager._term_instances["lost_contact_v1"]
    term.reset()
    act = env.action_manager.get_term("joint_control")
    labels = np.load(labels_file)
    should_l, point_l, far_l = labels["should_contact"], labels["contact_point"].astype(np.float64), labels["human_far"]
    palm = labels["palm_offset"].astype(np.float64) if lab.palm_frame == "rubber_hand_link" else labels["palm_offset_wrist"].astype(np.float64)
    names = list(env.simulator._body_list)
    hand_idx = [names.index(h) for h in (HANDS if lab.palm_frame == "rubber_hand_link" else ("left_wrist_yaw_link", "right_wrist_yaw_link"))]
    counter = np.zeros((E, 2, 2), int)
    errs = dict(reward=0.0, d=0.0, count=0.0, lost_mismatch=0, terminate_mismatch=0)
    hs = getattr(env.simulator, "object_hand_contact_sensor", None)
    hand_force_diff = 0.0
    k = CONTACT_V1
    rows = []
    for i in range(C):
        frames = start + i
        valid = frames < F
        c.time_steps[:] = frames.clamp(max=F - 1)
        ids = torch.arange(E, device=env.device)
        c._write_reference_state(ids)
        sample = c.reference.sample(c.time_steps)
        act.joint_target_override = sample["agent_joint_pos"].to(env.simulator.agent_dof_pos.dtype)
        env.episode_length_buf[:] = 1
        env._pre_physics_step(torch.zeros(E, 2, env.num_dof, device=env.device))
        env._physics_step()
        env._refresh_sim_tensors()
        act.joint_target_override = None
        r = rew(env, **rparams)
        flags = term(env, labels_file=labels_file)
        # numpy definitions
        f = _np(c.time_steps).astype(int).clip(0, lab.frames - 1)
        Ro = qmat(_np(c.simulator_object_quat_w))
        cp = _np(c.simulator_object_pos_w)[:, None, None] + np.einsum("eij,erhj->erhi", Ro, point_l[f])
        hp, Rh = _np(env.simulator.agent_rigid_body_pos[:, :, hand_idx]), qmat(_np(env.simulator.agent_rigid_body_rot[:, :, hand_idx]))
        d = np.linalg.norm(hp + np.einsum("erhij,hj->erhi", Rh, palm) - cp, axis=-1)
        norms = np.linalg.norm(_np(lab.sensor.data.force_matrix_w)[:, 0], axis=-1)
        fh, fl = norms[:, _np(lab.hand_filter).astype(int)], norms[:, _np(lab.link_filter).astype(int)]
        should, far = should_l[f], far_l[f]
        r_h = np.where(should, k["gain"] * np.exp(-d / k["sigma_m"]) * np.exp(np.minimum(fh - k["force_ref_n"], 0) / k["force_scale_n"]), 1.0)
        count = ((fl > k["undesired_force_n"]) & far).sum(-1)
        robot = r_h.mean(-1) * np.exp(-k["undesired_k"] * count)
        lost = should & ((d > k["lost_distance_m"]) | (fh < k["lost_force_n"]))
        counter = np.where(lost, counter + 1, 0)
        term_np = (counter >= k["lost_steps"]).reshape(E, -1).any(-1)
        v = _np(valid).astype(bool)
        want = robot if _np(r).ndim == 2 else robot.mean(-1)   # per-agent mode returns [E, 2]
        errs["reward"] = max(errs["reward"], float(np.abs(_np(r) - want)[v].max()) if v.any() else 0.0)
        errs["d"] = max(errs["d"], float(np.abs(_np(rew.last_hand) - r_h)[v].max()) if v.any() else 0.0)
        errs["count"] = max(errs["count"], float(np.abs(_np(rew.last_count) - count)[v].max()) if v.any() else 0.0)
        errs["lost_mismatch"] += int((_np(term.last_lost).astype(bool) != lost)[v].sum())
        errs["terminate_mismatch"] += int((_np(flags).astype(bool) != term_np)[v].sum())
        if hs is not None:
            hf = np.linalg.norm(_np(hs.data.force_matrix_w)[:, 0], axis=-1).reshape(E, 2, 2)
            hand_force_diff = max(hand_force_diff, float(np.abs(hf - fh)[v].max()) if v.any() else 0.0)
        for e in np.flatnonzero(v):
            rows.append(dict(frame=int(f[e]), reward_team=float(np.mean(_np(r)[e])), should=should[e].astype(int).tolist(), d_m=d[e].round(4).tolist(),
                             f_hand_n=fh[e].round(2).tolist(), penalty_count=count[e].tolist(), lost=lost[e].astype(int).tolist(),
                             counter=counter[e].tolist(), terminate=bool(term_np[e])))
    rows.sort(key=lambda x: x["frame"])
    arr = lambda key: np.array([x[key] for x in rows], dtype=float)
    summary = dict(frames=len(rows), envs=E, block=C, palm_frame=lab.palm_frame, max_abs_err=errs, hand_force_link_vs_hand_sensor_max_abs_n=hand_force_diff if hs is not None else None,
                   mean_team_reward=float(arr("reward_team").mean()), should_fraction=float(np.mean([np.mean(x["should"]) for x in rows])),
                   hand_force_over_1n_in_should=float(np.mean([fh_ > 1 for x in rows for fh_, s in zip(np.ravel(x["f_hand_n"]), np.ravel(x["should"])) if s])) if any(np.ravel(x["should"]).any() for x in rows) else None,
                   lost_fraction_in_should=float(np.mean([l for x in rows for l, s in zip(np.ravel(x["lost"]), np.ravel(x["should"])) if s])) if rows else None,
                   frames_with_penalty=int(sum(1 for x in rows if np.sum(x["penalty_count"]) > 0)), frames_terminating=int(sum(1 for x in rows if x["terminate"])))
    # internal diagnostic (Kevin 2026-11-24; to be read once the training references are fixed): per robot, in the frames where
    # contact is expected, the share of (frame, hand) samples with hand-object force > 1 N
    for r in (0, 1):
        sel = [(f_, s_) for x in rows for f_, s_ in zip(np.asarray(x["f_hand_n"])[r], np.asarray(x["should"])[r]) if s_]
        summary[f"hand_force_over_1n_in_should_r{r}"] = float(np.mean([f_ > 1.0 for f_, _ in sel])) if sel else None
        summary[f"should_samples_r{r}"] = len(sel)
    return dict(summary=summary, rows=rows)


def count_syncs(env, labels_file: str | None) -> dict:
    """GPU->CPU synchronisations raised inside the new terms (torch.cuda.set_sync_debug_mode warn)."""
    if not torch.cuda.is_available():
        return dict(skipped="no cuda")
    calls = {}
    om = env.observation_manager
    inst = om._term_instances
    for g, n in (("actor_obs", "motion_ref_ori_b"), ("teammate_obs", "neck_partner_rel_pos_b"), ("teammate_obs", "object_rel_pos_b")):
        if n in inst.get(g, {}):
            calls[f"obs:{n}"] = (lambda t=inst[g][n]: t(env))
    from holosoma.managers.observation.terms.marl_v1 import centralized_both_motion_commands
    calls["obs:critic_both_commands"] = lambda: centralized_both_motion_commands(env)
    if labels_file is not None:
        rew = env.reward_manager._term_instances["contact_v1"]
        p = env.reward_manager.cfg.terms["contact_v1"].params
        term = env.termination_manager._term_instances["lost_contact_v1"]
        calls["reward:contact_v1"] = lambda: rew(env, **p)
        calls["termination:lost_contact_v1"] = lambda: term(env, labels_file=labels_file)
    out = {}
    for name, fn in calls.items():
        fn()  # warm-up (label upload on first use)
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode(1)
            try:
                for _ in range(3):
                    fn()
            finally:
                torch.cuda.set_sync_debug_mode(0)
        sites = {}
        for x in w:
            if "synchroniz" in str(x.message).lower():
                key = f"{Path(x.filename).name}:{x.lineno}"
                sites[key] = sites.get(key, 0) + 1
        out[name] = dict(count_per_3_calls=sum(sites.values()), sites=sites)
    return out


def check_per_agent(env) -> dict | None:
    """D-Table-PA: every per-robot term's agent mean equals its team term on the same state; shared terms are [E]."""
    from holosoma.managers.reward.per_agent_team_manager import PerAgentTeamRewardManager, SHARED_TERMS
    rm = env.reward_manager
    if not isinstance(rm, PerAgentTeamRewardManager):
        return None
    import importlib
    import holosoma.managers.reward.terms.marl as team_mod
    from holosoma.managers.reward.terms import contact_v1 as cv
    out = {}
    for name in rm.active_terms:
        cfg = rm.cfg.terms[name]
        mod, fn = cfg.func.split(":")
        if name in SHARED_TERMS:
            out[name] = "shared [E]"
            continue
        if name == "contact_v1":
            pa = rm._term_instances[name](env, **cfg.params)
            team = rm._term_instances[name](env, **{**cfg.params, "per_agent": False})
        elif mod.endswith("marl_per_agent"):
            pa = rm._term_instances[name](env, **cfg.params) if name in rm._term_instances else rm._term_funcs[name](env, **cfg.params)
            team_fn = getattr(team_mod, fn)
            if isinstance(team_fn, type):
                team = team_fn(cfg, env)(env, **cfg.params)
            else:
                team = team_fn(env, **cfg.params)
        else:
            out[name] = f"not checked ({cfg.func})"
            continue
        out[name] = dict(shape=list(pa.shape), max_abs_mean_minus_team=float((pa.mean(dim=1) - team).abs().max()))
    return out


def run(env, observations, *, out: Path, labels_file: str | None, noise_expected: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    c = env.command_manager.get_state("paired_motion_command")
    res = dict(initial_time_steps=_np(c.time_steps).astype(int).tolist())  # reference frames after reset_all (sampled at reset)
    res["obs"] = check_obs(env, observations, noise_expected)
    res["per_agent"] = check_per_agent(env)
    res["sync_counts"] = count_syncs(env, labels_file)
    if labels_file is not None:
        rep = replay_contact(env, labels_file)
        res["contact_replay"] = rep["summary"]
        with open(out / "contact_replay_rows.jsonl", "w") as f:
            for row in rep["rows"]:
                f.write(json.dumps(row) + "\n")
        # history test: S1_T13V4_REPLAY_PASSES=N replays the whole reference again N-1 times in the same process
        for k in range(2, int(os.environ.get("S1_T13V4_REPLAY_PASSES", "1")) + 1):
            rep_k = replay_contact(env, labels_file)
            res[f"contact_replay_pass{k}"] = rep_k["summary"]
            with open(out / f"contact_replay_rows_pass{k}.jsonl", "w") as f:
                for row in rep_k["rows"]:
                    f.write(json.dumps(row) + "\n")
    (out / "isaac_checks.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


__all__ = ["run"]
