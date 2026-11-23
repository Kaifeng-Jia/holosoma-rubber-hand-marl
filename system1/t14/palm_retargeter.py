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
    def __init__(self, *args, palm_offsets: dict[str, np.ndarray] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.palm_offsets = None if palm_offsets is None else {k: np.asarray(v, dtype=float).reshape(3) for k, v in palm_offsets.items()}
        if self.palm_offsets is not None:
            missing = set(self.palm_offsets) - set(self.laplacian_match_links)
            if missing:
                raise ValueError(f"palm offsets for keys not in the interaction mesh: {sorted(missing)}")

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
