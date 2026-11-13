"""T06: holosoma robot config for the G1 + two Sharpa Wave hands (system1/hand/assets/g1_29dof_sharpa.urdf).

with_sharpa_hands(robot) returns a copy of a G1 rubber-hand robot config (the one D trains with) in which
  * the URDF is the assembled G1 + Sharpa file (path relative to the holosoma robot asset root);
  * dof_names = the 29 G1 joints (unchanged order) + the 44 finger joints (left hand then right hand, Sharpa URDF
    order); the per-DOF lists (limits, velocity / effort limits, armature, friction) are the G1 values followed by
    the Sharpa values (limits / velocity / effort from the Sharpa URDF; armature and friction from the official USD,
    system1/hand/sharpa.py: pd_table);
  * finger PD gains (kp, kd per rad, from the official USD drive gains per degree) are added to control.stiffness /
    damping under the full joint name, so the substring matcher of the action term cannot confuse them with G1 keys;
  * default joint angle of every finger joint = 0 (the open pose);
  * body_names = the G1 bodies + the child link of every finger joint (the bodies that remain after fixed joints are
    merged); num_bodies, dof_obs_size and actions_dim follow. The simulator's actual body / DOF lists are checked in the
    step 1 load test (system1/hand/load_test.py) - this module does not assume more than the URDF says.
Every G1 field is left as it is. Nothing here is used unless the T06 switch is on.
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from system1.hand.sharpa import SIDES, SharpaHand, pd_table

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "system1/hand/assets/g1_29dof_sharpa.urdf"
HOLOSOMA_ROBOT_ROOT = ROOT / "src/holosoma/holosoma/data/robots"


def finger_joint_names() -> list[str]:
    return [n for s in SIDES for n in SharpaHand(s).dof_names]


def finger_body_names() -> list[str]:
    out = []
    for s in SIDES:
        h = SharpaHand(s)
        child = {j.name: j.child for j in h.joints}
        out += [child[n] for n in h.dof_names]
    return out


def finger_table() -> dict[str, dict]:
    """Per finger joint: lower, upper, velocity, effort (URDF), kp, kd, armature, friction (USD)."""
    out = {}
    for s in SIDES:
        h = SharpaHand(s)
        pd = pd_table(s)
        jm = {j.name: j for j in h.joints}
        for k, n in enumerate(h.dof_names):
            out[n] = dict(lower=float(h.lower[k]), upper=float(h.upper[k]), velocity=float(getattr(jm[n], "velocity", 0.0) or 0.0),
                          effort=float(pd[n]["effort"]), kp=float(pd[n]["kp"]), kd=float(pd[n]["kd"]),
                          armature=float(pd[n]["armature"]), friction=float(pd[n]["friction"]),
                          action_scale=float(pd[n]["action_scale"]))
    return out


def with_sharpa_hands(robot):
    names = finger_joint_names()
    tab = finger_table()
    col = lambda key: [tab[n][key] for n in names]  # noqa: E731
    stiffness = dict(robot.control.stiffness)
    damping = dict(robot.control.damping)
    for n in names:
        for k in list(stiffness):  # no G1 key may match a finger joint (substring matcher)
            if k in n:
                raise ValueError(f"G1 gain key {k!r} would match finger joint {n!r}")
        stiffness[n] = tab[n]["kp"]
        damping[n] = tab[n]["kd"]
    dof_names = list(robot.dof_names) + names
    bodies = list(robot.body_names) + finger_body_names()
    urdf_rel = os.path.relpath(URDF, HOLOSOMA_ROBOT_ROOT)
    return replace(
        robot,
        num_bodies=len(bodies),
        dof_obs_size=len(dof_names),
        actions_dim=len(dof_names),
        dof_names=dof_names,
        dof_pos_lower_limit_list=list(robot.dof_pos_lower_limit_list) + col("lower"),
        dof_pos_upper_limit_list=list(robot.dof_pos_upper_limit_list) + col("upper"),
        dof_vel_limit_list=list(robot.dof_vel_limit_list) + col("velocity"),
        dof_effort_limit_list=list(robot.dof_effort_limit_list) + col("effort"),
        dof_armature_list=list(robot.dof_armature_list) + col("armature"),
        dof_joint_friction_list=list(robot.dof_joint_friction_list) + col("friction"),
        body_names=bodies,
        init_state=replace(robot.init_state, default_joint_angles={**robot.init_state.default_joint_angles, **{n: 0.0 for n in names}}),
        control=replace(robot.control, stiffness=stiffness, damping=damping),
        asset=replace(robot.asset, urdf_file=urdf_rel),  # xml_file / robot_type left as they are (IsaacSim loads the URDF)
        symmetry_joint_names=None if robot.symmetry_joint_names is None else {
            **robot.symmetry_joint_names,
            **{n: n.replace("left_", "right_", 1) if n.startswith("left_") else n.replace("right_", "left_", 1) for n in names}},
    )


__all__ = ["finger_body_names", "finger_joint_names", "finger_table", "with_sharpa_hands"]
