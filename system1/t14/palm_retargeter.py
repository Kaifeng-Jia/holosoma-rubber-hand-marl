"""T14 step 1: palm points in the interaction mesh, as a subclass (src/holosoma_retargeting is not changed).

InteractionMeshRetargeter puts robot interaction-mesh vertices at the origins of the mapped links
(_calc_manipulator_jacobians: the Jacobian is taken at an optional common offset but the position is always the
link origin). PalmInteractionMeshRetargeter takes palm_offsets = {demo point name: offset in that link's body frame}
for extra keys (L_Palm / R_Palm -> left / right_rubber_hand_link at the rubber-hand palm point) and returns both the
position and the Jacobian of that body point. With palm_offsets=None every call goes to the base class unchanged
(switch off = identical behaviour; test_palm_retargeter.py checks this bit for bit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter  # noqa: E402


class PalmInteractionMeshRetargeter(InteractionMeshRetargeter):
    """anchor_term (T14 step 2, CoCarry form; None = off): dict(keys=("L_Palm", "R_Palm"), weight=w, anchors=[T, 2, 3] in the
    object frame, activation=[T, 2] in [0, 1]); adds sum over hands of w * activation * ||p_palm(q) - anchor||^2, linearised
    in the object frame like the interaction-mesh term (base-class hook extra_objective_terms).
    orient_term (T14 round 4 B; None = off): dict(weight, activation=[T, 2], targets_obj=[T, 2, 3] unit vectors in the object
    frame, palm_normal_local={"left": n, "right": n}); see _orient_terms. Round 5 C: palm_world / finger_world [T, 2, 3]
    (world targets for the palm normal and the finger direction, both with the same weight) instead of targets_obj."""

    def __init__(self, *args, palm_offsets: dict[str, np.ndarray] | None = None, anchor_term: dict | None = None,
                 orient_term: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.palm_offsets = None if palm_offsets is None else {k: np.asarray(v, dtype=float).reshape(3) for k, v in palm_offsets.items()}
        if self.palm_offsets is not None:
            missing = set(self.palm_offsets) - set(self.laplacian_match_links)
            if missing:
                raise ValueError(f"palm offsets for keys not in the interaction mesh: {sorted(missing)}")
        self.anchor_term = anchor_term
        if anchor_term is not None:
            if self.palm_offsets is None or not set(anchor_term["keys"]) <= set(self.palm_offsets):
                raise ValueError("anchor_term needs the palm points")
        self.orient_term = orient_term
        if orient_term is not None:
            self._orient_specs = []
            for side in ("left", "right"):
                arm = []
                for suffix in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"):
                    jid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{suffix}_joint")
                    arm.append(int(np.flatnonzero(self.q_a_indices == int(self.robot_model.jnt_qposadr[jid]))[0]))
                nl = np.asarray(orient_term["palm_normal_local"][side], float)
                nl = nl / np.linalg.norm(nl)
                fl = np.array([1.0, 0.0, 0.0]) - nl[0] * nl  # = _init_hand_orientation's palm_basis finger column
                self._orient_specs.append(dict(body_id=mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_rubber_hand_link"),
                                               arm=np.asarray(arm), normal_local=nl, finger_local=fl / np.linalg.norm(fl)))
        if anchor_term is not None or orient_term is not None:
            self.extra_objective_terms = self._extra_terms

    def _extra_terms(self, q, dqa, frame_idx):
        terms = self._anchor_terms(q, dqa, frame_idx) if self.anchor_term is not None else []
        if self.orient_term is not None:
            terms += self._orient_terms(q, dqa, frame_idx)
        return terms

    def _orient_terms(self, q, dqa, frame_idx):
        """T14 round 4 B: palm normal toward the object surface (inward normal at the anchor), the repo's palm-direction form
        (HandOrientationConfig): w * act * ||n + (-[n]x J_w[:, arm]) dqa[arm] - n*||^2 with the 7 arm joints of that side;
        targets are stored in the object frame and rotated with the (locked) object pose of this frame."""
        import cvxpy as cp
        o = self.orient_term
        act = np.asarray(o["activation"][frame_idx], dtype=float)
        if not np.any(act > 0):
            return []
        qw, qx, qy, qz = q[-4:]
        R_obj = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        terms = []
        for hi, spec in enumerate(self._orient_specs):
            if act[hi] <= 0:
                continue
            rotation, ang_jac, _ = self._calc_body_orientation_linearization(q, spec["body_id"])
            Ja = ang_jac[:, spec["arm"]]
            n = rotation @ spec["normal_local"]
            lin = n + (-self._skew(n) @ Ja) @ dqa[spec["arm"]]
            if "palm_world" in o:  # round 5 C: track the person's palm frame (palm normal and finger direction, world)
                f = rotation @ spec["finger_local"]
                lin_f = f + (-self._skew(f) @ Ja) @ dqa[spec["arm"]]
                w = float(o["weight"]) * float(act[hi])
                terms.append(w * cp.sum_squares(lin - np.asarray(o["palm_world"][frame_idx, hi], float)))
                terms.append(w * cp.sum_squares(lin_f - np.asarray(o["finger_world"][frame_idx, hi], float)))
                continue
            target = R_obj @ np.asarray(o["targets_obj"][frame_idx, hi], float)
            terms.append(float(o["weight"]) * float(act[hi]) * cp.sum_squares(lin - target))
        return terms

    def _anchor_terms(self, q, dqa, frame_idx):
        import cvxpy as cp
        a = self.anchor_term
        act = np.asarray(a["activation"][frame_idx], dtype=float)
        if not np.any(act > 0):
            return []
        keys = {k: self.laplacian_match_links[k] for k, w in zip(a["keys"], act) if w > 0}
        J, p, _ = self._calc_manipulator_jacobians(q, keys, obj_frame=True)
        terms = []
        for hi, k in enumerate(a["keys"]):
            if act[hi] > 0:
                terms.append(float(a["weight"]) * float(act[hi]) * cp.sum_squares(J[k] @ dqa + (p[k] - a["anchors"][frame_idx, hi])))
        return terms

    def _calc_manipulator_jacobians(self, q, links, obj_frame=False, point_offsets=None):
        if self.palm_offsets is None or point_offsets is not None or not (set(self.palm_offsets) & set(links)):
            return super()._calc_manipulator_jacobians(q, links, obj_frame=obj_frame, point_offsets=point_offsets)
        J_XC_dict, p_XC_dict = {}, {}
        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
            else:
                obj_rot = np.eye(3)
                obj_pos = np.zeros(3)
            obj_rot_inv = obj_rot.T
        self.robot_data.qpos[:] = q.copy()
        mujoco.mj_forward(self.robot_model, self.robot_data)
        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            off = self.palm_offsets.get(name, np.zeros(3))
            J = self._calc_contact_jacobian_from_point(body_id, off)
            pos_world = self.robot_data.xpos[body_id] + self.robot_data.xmat[body_id].reshape(3, 3) @ off
            if obj_frame:
                p_XC, J_XC = obj_rot_inv @ (pos_world - obj_pos), obj_rot_inv @ J
            else:
                p_XC, J_XC = pos_world, J
            J_XC_dict[name] = np.array(J_XC[:, self.q_a_indices], dtype=float, copy=True)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)
        P_WO = {"position": obj_pos, "rotation": obj_rot} if obj_frame else None
        return J_XC_dict, p_XC_dict, P_WO
