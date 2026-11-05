from __future__ import annotations

import sys
import time
from pathlib import Path
from types import ModuleType

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from scipy import sparse as sp  # type: ignore[import-untyped]
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from tqdm import tqdm
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.config_types.retargeter import (
    ElasticConstraintConfig,
    FootLockConfig,
    HandOrientationConfig,
    PlanBPalmContactConfig,
    PTFullArmOrientationConfig,
    PTPalmCollisionConfig,
    PTWristDominantSurfaceConfig,
    PTWristOrientationConfig,
    SelfCollisionConfig,
)

# Add src to path for direct execution
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import with type ignore for mypy compatibility
from mujoco_utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    _world_mesh_from_geom,
)
from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
    transform_points_local_to_world,
    transform_points_world_to_local,
)
from viser_utils import create_motion_control_sliders  # type: ignore[import-not-found,no-redef]  # noqa: E402


def _select_pt_wrist_orientation_indices(arm_indices: np.ndarray) -> np.ndarray:
    """Select the wrist roll/pitch/yaw entries from an ordered arm group."""
    arm_indices = np.asarray(arm_indices)
    if arm_indices.shape != (7,):
        raise ValueError(f"Expected seven ordered arm indices, got shape {arm_indices.shape}")
    return arm_indices[-3:].copy()


class InteractionMeshRetargeter:
    """
    A class to perform kinematic retargeting from human motion to a robot,
    preserving spatial relationships using an interaction mesh.
    """

    def __init__(
        self,
        task_constants: ModuleType,
        object_urdf_path: str,
        q_a_init_idx: int = -7,
        activate_foot_sticking: bool = True,
        activate_obj_non_penetration: bool = True,
        activate_joint_limits: bool = True,
        apply_manual_joint_limit_overrides: bool = True,
        step_size: float = 0.2,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        foot_lock: FootLockConfig | None = None,
        self_collision: SelfCollisionConfig | None = None,
        visualize: bool = False,
        debug: bool = False,
        w_nominal_tracking_init: float = 5.0,
        nominal_tracking_tau: float = 10.0,
        scene_xml_path: str | None = None,
        anchor_nominal_foot_height: bool = False,
        nominal_foot_height_tolerance: float = 5e-3,
        hand_orientation: HandOrientationConfig | None = None,
        plan_b_palm_contact: PlanBPalmContactConfig | None = None,
        pt_wrist_orientation: PTWristOrientationConfig | None = None,
        pt_full_arm_orientation: PTFullArmOrientationConfig | None = None,
        pt_wrist_dominant_surface: PTWristDominantSurfaceConfig | None = None,
        pt_palm_collision: PTPalmCollisionConfig | None = None,
        elastic_constraints: ElasticConstraintConfig | None = None,
    ):
        """This kinematic retargeter solves the diffIK problem with hard constraints in SQP style.
        During each SQP iteration, the problem is solved with the following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation in the object frame.
            2. [Constraint] Enforce the non-penetration constraints w/ the ground and (if activated) the object.
            3. [Constraint] Enforce the foot sticking constraints if activated.
            4. [Constraint] Enforce the joint limits if activated.
            5. [Constraint] Enforce trust region of dq.
        The constraints are linearized and the costs are quadratic with a trust region.

        Args:
            q_a_init_idx: the index in robot's configuration where the optimization variables start. -7: starts from the
            floating base, -3: starts from the translation of the floating base, 0: starts from the actuated DOF,
            12: starts from waist, 15: starts from left shoulder
            apply_manual_joint_limit_overrides: whether task-specific manual
                bounds should further tighten the robot URDF joint limits.
            step_size: trust region for each SQP iteration.
            collision_detection_threshold: only start to detect collision
            when the distance is smaller than this threshold.
            penetration_tolerance: tolerance for penetration when enforcing non-penetration constraints.
            foot_sticking_tolerance: tolerance for foot sticking constraints in x, y.
            foot_lock: configuration for explicit frame-range based foot locking constraints.
            elastic_constraints: optional exact-penalty relaxation for
                robot-object and foot kinematic constraints. Ground, joint
                limits, and the SQP trust region remain hard constraints.
            nominal_tracking_tau: the time constant for the nominal tracking cost.
        """

        self.robot_model_path = task_constants.ROBOT_URDF_FILE
        self.object_model_path = object_urdf_path
        self.object_name = task_constants.OBJECT_NAME
        self.collision_detection_threshold = collision_detection_threshold
        self.activate_foot_sticking = activate_foot_sticking
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_joint_limits = activate_joint_limits
        self.apply_manual_joint_limit_overrides = (
            apply_manual_joint_limit_overrides
        )
        self.foot_links = dict(zip(task_constants.FOOT_STICKING_LINKS, task_constants.FOOT_STICKING_LINKS))
        self.penetration_tolerance = penetration_tolerance
        self.step_size = step_size
        self.visualize = visualize
        self.debug = debug
        self.demo_joints = task_constants.DEMO_JOINTS
        self.laplacian_match_links = task_constants.JOINTS_MAPPING
        self.task_constants = task_constants

        self.smplh_mapped_joint_indices = [self.demo_joints.index(name) for name in self.laplacian_match_links]

        # Setup weights and parameters
        self.laplacian_weights = 10
        self.smooth_weight = 0.2
        # Tolerance for foot sticking constraints in x, y.
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self.anchor_nominal_foot_height = anchor_nominal_foot_height
        self.nominal_foot_height_tolerance = nominal_foot_height_tolerance
        self._init_foot_lock(foot_lock)
        self._self_collision_config = self_collision
        self.hand_orientation = hand_orientation or HandOrientationConfig()
        self.plan_b_palm_contact = plan_b_palm_contact or PlanBPalmContactConfig()
        self.pt_wrist_orientation = pt_wrist_orientation or PTWristOrientationConfig()
        self.pt_full_arm_orientation = (
            pt_full_arm_orientation or PTFullArmOrientationConfig()
        )
        self._validate_pt_full_arm_orientation_config()
        self.pt_wrist_dominant_surface = (
            pt_wrist_dominant_surface or PTWristDominantSurfaceConfig()
        )
        self._validate_pt_wrist_dominant_surface_config()
        self.pt_palm_collision = pt_palm_collision or PTPalmCollisionConfig()
        self.elastic_constraints = elastic_constraints or ElasticConstraintConfig()
        if (
            not np.isfinite(self.elastic_constraints.object_collision_weight)
            or self.elastic_constraints.object_collision_weight <= 0.0
        ):
            raise ValueError("Elastic object collision weight must be positive")
        if (
            not np.isfinite(self.elastic_constraints.foot_kinematics_weight)
            or self.elastic_constraints.foot_kinematics_weight <= 0.0
        ):
            raise ValueError("Elastic foot kinematics weight must be positive")
        self._last_elastic_slack_diagnostics = {
            "object_collision_slack_max_m": 0.0,
            "foot_constraint_slack_max_m": 0.0,
        }
        if self.plan_b_palm_contact.enable:
            self.penetration_tolerance = (
                self.plan_b_palm_contact.penetration_tolerance
            )
            self.smooth_weight = (
                self.plan_b_palm_contact.temporal_smooth_weight
            )
        enabled_hand_modes = sum(
            (
                self.hand_orientation.enable,
                self.plan_b_palm_contact.enable,
                self.pt_wrist_orientation.enable,
                self.pt_full_arm_orientation.enable,
                self.pt_wrist_dominant_surface.enable,
                self.pt_palm_collision.enable,
            )
        )
        if enabled_hand_modes > 1:
            raise ValueError(
                "Legacy hand orientation, Plan B palm contact, A.1 PT wrist "
                "orientation, PT full-arm orientation, and PT wrist-dominant "
                "surface refinement, and PT palm collision refinement are mutually exclusive"
            )

        # Setup visualization if requested
        if self.visualize:
            self._setup_visualization()

        # Load Mujoco model
        if scene_xml_path is not None:
            robot_xml_path = scene_xml_path
        elif self.object_name == "ground":
            robot_xml_path = self.robot_model_path.replace(".urdf", ".xml")
        elif self.object_name == "multi_boxes":
            robot_xml_path = self.task_constants.SCENE_XML_FILE
        else:
            robot_xml_path = self.robot_model_path.replace(".urdf", "_w_" + self.object_name + ".xml")

        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print("Loading robot model from: ", robot_xml_path)
        self._object_geom_ids = self._resolve_object_geom_ids()

        self.robot_data = mujoco.MjData(self.robot_model)
        self._init_self_collision(self._self_collision_config)

        if self.robot_data.qpos.shape[0] > 7 + self.task_constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False

        self.nq = self.robot_model.nq

        self.q_a_init_idx = q_a_init_idx
        self.q_a_indices = np.arange(7 + self.q_a_init_idx, 7 + self.task_constants.ROBOT_DOF)

        self.nq_a = len(self.q_a_indices)

        # Create complete limits with floating base (-inf, inf) and actuated joint limits
        n_floating_base = 7
        joint_names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        actuated_joints = [(i, name) for i, name in enumerate(joint_names) if name]  # Filter out None names

        large_number = 1e6
        complete_lower_limits = np.concatenate(
            [-large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 0]]
        )
        complete_upper_limits = np.concatenate(
            [large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 1]]
        )

        self.q_a_lb = complete_lower_limits[self.q_a_indices]
        self.q_a_ub = complete_upper_limits[self.q_a_indices]

        if self.apply_manual_joint_limit_overrides:
            manual_lb_indices = np.asarray(
                list(self.task_constants.MANUAL_LB.keys()),
                dtype=int,
            )
            manual_ub_indices = np.asarray(
                list(self.task_constants.MANUAL_UB.keys()),
                dtype=int,
            )
            self.q_a_lb[manual_lb_indices] = list(
                self.task_constants.MANUAL_LB.values()
            )
            self.q_a_ub[manual_ub_indices] = list(
                self.task_constants.MANUAL_UB.values()
            )

        self._init_hand_orientation()
        self._init_plan_b_tabletop()

        # Prevent too much waist twist
        self.Q_diag = np.zeros(self.nq_a) * 1e-3
        self.Q_diag[np.array(list(self.task_constants.MANUAL_COST.keys())).astype(int)] = list(
            self.task_constants.MANUAL_COST.values()
        )

        self.w_nominal_tracking_init = w_nominal_tracking_init
        self.nominal_tracking_tau = nominal_tracking_tau
        self.track_nominal_indices = task_constants.NOMINAL_TRACKING_INDICES

    def _validate_pt_full_arm_orientation_config(self) -> None:
        """Validate the opt-in full-arm refinement before loading its targets."""
        config = self.pt_full_arm_orientation
        positive_weights = {
            "orientation_weight": config.orientation_weight,
            "hand_position_weight": config.hand_position_weight,
        }
        nonnegative_weights = {
            "arm_prior_weight": config.arm_prior_weight,
            "correction_temporal_weight": config.correction_temporal_weight,
        }
        for name, value in positive_weights.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"PT full-arm {name} must be finite and positive")
        for name, value in nonnegative_weights.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"PT full-arm {name} must be finite and non-negative")
        if (
            isinstance(config.max_nfev, bool)
            or not isinstance(config.max_nfev, (int, np.integer))
            or config.max_nfev <= 0
        ):
            raise ValueError("PT full-arm max_nfev must be a positive integer")

    def _validate_pt_wrist_dominant_surface_config(self) -> None:
        """Validate wrist-dominant surface-point refinement weights."""
        config = self.pt_wrist_dominant_surface
        positive_weights = {
            "normal_weight": config.normal_weight,
            "surface_position_weight": config.surface_position_weight,
        }
        nonnegative_weights = {
            "finger_weight": config.finger_weight,
            "proximal_prior_weight": config.proximal_prior_weight,
            "wrist_prior_weight": config.wrist_prior_weight,
            "proximal_correction_temporal_weight": (
                config.proximal_correction_temporal_weight
            ),
            "wrist_correction_temporal_weight": (
                config.wrist_correction_temporal_weight
            ),
        }
        for name, value in positive_weights.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"PT wrist-dominant surface {name} must be finite and positive"
                )
        for name, value in nonnegative_weights.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"PT wrist-dominant surface {name} must be finite and non-negative"
                )
        if (
            isinstance(config.max_nfev, bool)
            or not isinstance(config.max_nfev, (int, np.integer))
            or config.max_nfev <= 0
        ):
            raise ValueError(
                "PT wrist-dominant surface max_nfev must be a positive integer"
            )

    def _init_hand_orientation(self) -> None:
        """Resolve hand links, palm frames, and upper-limb optimization joints."""
        self._hand_orientation_specs: list[dict[str, object]] = []
        if not (
            self.hand_orientation.enable
            or self.plan_b_palm_contact.enable
            or self.pt_wrist_orientation.enable
            or self.pt_full_arm_orientation.enable
            or self.pt_wrist_dominant_surface.enable
            or self.pt_palm_collision.enable
        ):
            return
        if not self.has_dynamic_object:
            raise ValueError("Hand orientation requires a dynamic interaction object")

        candidates = {
            "left": ("L_Wrist", "LeftHand"),
            "right": ("R_Wrist", "RightHand"),
        }
        palm_normals = {
            "left": np.array([-0.07513681, -0.99540367, -0.05938011]),
            "right": np.array([-0.07514936, 0.99540878, -0.05927846]),
        }
        for side, demo_candidates in candidates.items():
            demo_joint = next(
                (name for name in demo_candidates if name in self.laplacian_match_links),
                None,
            )
            if demo_joint is None:
                raise ValueError(f"No mapped {side} wrist joint is available for hand orientation")
            link_name = self.laplacian_match_links[demo_joint]
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id < 0:
                raise ValueError(f"Hand link '{link_name}' was not found in the MuJoCo model")
            self._hand_orientation_specs.append(
                {
                    "side": side,
                    "demo_joint_idx": self.demo_joints.index(demo_joint),
                    "link_name": link_name,
                    "body_id": body_id,
                    "palm_normal": palm_normals[side],
                }
            )

        for spec in self._hand_orientation_specs:
            palm_normal = np.asarray(spec["palm_normal"], dtype=float)
            palm_normal /= np.linalg.norm(palm_normal)
            finger_direction = np.array([1.0, 0.0, 0.0])
            finger_direction -= np.dot(finger_direction, palm_normal) * palm_normal
            finger_direction /= np.linalg.norm(finger_direction)
            across_direction = np.cross(palm_normal, finger_direction)
            across_direction /= np.linalg.norm(across_direction)
            spec["palm_basis"] = np.column_stack(
                [finger_direction, across_direction, palm_normal]
            )
            if (
                self.plan_b_palm_contact.enable
                or self.pt_wrist_dominant_surface.enable
            ):
                spec["palm_contact_point"] = self._derive_palm_contact_point(
                    str(spec["side"]),
                    palm_normal,
                )
            if self.plan_b_palm_contact.enable:
                spec["palm_geom_id"] = mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{spec['side']}_rubber_hand_link",
                )

            side = str(spec["side"])
            arm_joint_suffixes = (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
            arm_indices = []
            arm_qpos_indices = []
            arm_lower_limits = []
            arm_upper_limits = []
            for suffix in arm_joint_suffixes:
                joint_name = f"{side}_{suffix}_joint"
                joint_id = mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                if joint_id < 0:
                    raise ValueError(f"Joint '{joint_name}' was not found in the MuJoCo model")
                qpos_idx = int(self.robot_model.jnt_qposadr[joint_id])
                local_idx = np.flatnonzero(self.q_a_indices == qpos_idx)
                if local_idx.size != 1:
                    raise ValueError(f"Joint '{joint_name}' is outside the optimized configuration")
                arm_indices.append(int(local_idx[0]))
                arm_qpos_indices.append(qpos_idx)
                arm_lower_limits.append(float(self.robot_model.jnt_range[joint_id, 0]))
                arm_upper_limits.append(float(self.robot_model.jnt_range[joint_id, 1]))
            spec["arm_orientation_indices"] = np.asarray(arm_indices, dtype=int)
            spec["pt_full_arm_qpos_indices"] = np.asarray(
                arm_qpos_indices,
                dtype=int,
            )
            spec["pt_full_arm_lower_limits"] = np.asarray(
                arm_lower_limits,
                dtype=float,
            )
            spec["pt_full_arm_upper_limits"] = np.asarray(
                arm_upper_limits,
                dtype=float,
            )
            spec["pt_wrist_qpos_indices"] = _select_pt_wrist_orientation_indices(
                np.asarray(arm_qpos_indices, dtype=int)
            )
            spec["pt_wrist_lower_limits"] = _select_pt_wrist_orientation_indices(
                np.asarray(arm_lower_limits, dtype=float)
            )
            spec["pt_wrist_upper_limits"] = _select_pt_wrist_orientation_indices(
                np.asarray(arm_upper_limits, dtype=float)
            )
            if self.plan_b_palm_contact.enable:
                wrist_indices = np.asarray(arm_indices[-3:], dtype=int)
                self.q_a_lb[wrist_indices] = np.asarray(
                    arm_lower_limits[-3:],
                    dtype=float,
                )
                self.q_a_ub[wrist_indices] = np.asarray(
                    arm_upper_limits[-3:],
                    dtype=float,
                )

    def _derive_palm_contact_point(
        self,
        side: str,
        palm_normal: np.ndarray,
    ) -> np.ndarray:
        """Derive a stable palm support point from the rubber-hand collision mesh."""
        geom_name = f"{side}_rubber_hand_link"
        geom_id = mujoco.mj_name2id(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )
        if geom_id < 0:
            raise ValueError(f"Plan B requires collision geom '{geom_name}'")
        mesh_id = int(self.robot_model.geom_dataid[geom_id])
        if mesh_id < 0:
            raise ValueError(f"Plan B palm geom '{geom_name}' must be a mesh")

        vertex_start = int(self.robot_model.mesh_vertadr[mesh_id])
        vertex_count = int(self.robot_model.mesh_vertnum[mesh_id])
        vertices = np.asarray(
            self.robot_model.mesh_vert[vertex_start : vertex_start + vertex_count],
            dtype=float,
        )
        geom_quat = self.robot_model.geom_quat[geom_id]
        geom_rotation = Rotation.from_quat(
            [geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]]
        ).as_matrix()
        vertices_body = (
            vertices @ geom_rotation.T + self.robot_model.geom_pos[geom_id]
        )

        normal = np.asarray(palm_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        support = float(np.max(vertices_body @ normal))
        center = 0.5 * (
            np.min(vertices_body, axis=0) + np.max(vertices_body, axis=0)
        )
        return center + (support - float(center @ normal)) * normal

    def _init_plan_b_tabletop(self) -> None:
        """Resolve the tabletop side used by the independent Plan B stage."""
        self._plan_b_tabletop_spec: dict[str, object] | None = None
        if not self.plan_b_palm_contact.enable:
            return
        if not self.has_dynamic_object:
            raise ValueError("Plan B palm contact requires a dynamic interaction object")

        cfg = self.plan_b_palm_contact
        axes = (cfg.face_axis, cfg.vertical_axis, cfg.lateral_axis)
        if sorted(axes) != [0, 1, 2]:
            raise ValueError(
                "Plan B face_axis, vertical_axis, and lateral_axis must be "
                "distinct values from {0, 1, 2}"
            )
        if cfg.face_sign not in (-1, 1):
            raise ValueError("Plan B face_sign must be -1 or +1")
        if cfg.hand_spacing <= 0:
            raise ValueError("Plan B hand_spacing must be positive")
        if (
            cfg.surface_gap < 0
            or cfg.approach_clearance < 0
        ):
            raise ValueError("Plan B clearances must be non-negative")
        if cfg.max_sqp_iterations <= 0:
            raise ValueError("Plan B max_sqp_iterations must be positive")
        if cfg.temporal_smooth_weight < 0:
            raise ValueError("Plan B temporal_smooth_weight must be non-negative")
        if (
            cfg.fade_frames < 0
            or cfg.release_frames < 0
        ):
            raise ValueError(
                "Plan B fade/release frame counts must be non-negative"
            )

        geom_id = mujoco.mj_name2id(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            cfg.tabletop_geom_name,
        )
        if geom_id < 0:
            raise ValueError(
                f"Plan B tabletop geom '{cfg.tabletop_geom_name}' was not found"
            )
        if int(self.robot_model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("Plan B tabletop contact currently requires a box geom")

        half_size = np.asarray(self.robot_model.geom_size[geom_id], dtype=float)
        if 0.5 * cfg.hand_spacing >= half_size[cfg.lateral_axis]:
            raise ValueError("Plan B hand targets lie outside the tabletop side")
        if abs(cfg.vertical_offset) >= half_size[cfg.vertical_axis]:
            raise ValueError("Plan B vertical target lies outside the tabletop side")

        geom_quat = self.robot_model.geom_quat[geom_id]
        geom_rotation = Rotation.from_quat(
            [geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]]
        ).as_matrix()
        outward_geom = np.zeros(3)
        outward_geom[cfg.face_axis] = float(cfg.face_sign)
        vertical_geom = np.zeros(3)
        vertical_geom[cfg.vertical_axis] = 1.0

        target_points_body = {}
        downward_body = -geom_rotation @ vertical_geom
        for side, lateral_sign in (("left", 1.0), ("right", -1.0)):
            point_geom = np.zeros(3)
            point_geom[cfg.face_axis] = (
                float(cfg.face_sign) * half_size[cfg.face_axis]
            )
            point_geom[cfg.vertical_axis] = cfg.vertical_offset
            point_geom[cfg.lateral_axis] = lateral_sign * 0.5 * cfg.hand_spacing
            target_points_body[side] = (
                self.robot_model.geom_pos[geom_id] + geom_rotation @ point_geom
            )

        self._plan_b_tabletop_spec = {
            "geom_id": geom_id,
            "target_points_body": target_points_body,
            "outward_body": geom_rotation @ outward_geom,
            "twist_direction_body": downward_body,
        }
        posture_joint_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        for side in ("left", "right"):
            posture_joint_names.extend(
                [
                    f"{side}_shoulder_pitch_joint",
                    f"{side}_shoulder_roll_joint",
                    f"{side}_shoulder_yaw_joint",
                    f"{side}_elbow_joint",
                    f"{side}_wrist_roll_joint",
                    f"{side}_wrist_pitch_joint",
                    f"{side}_wrist_yaw_joint",
                ]
            )
        posture_indices = []
        posture_weights = []
        for joint_name in posture_joint_names:
            joint_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            qpos_idx = int(self.robot_model.jnt_qposadr[joint_id])
            local_idx = np.flatnonzero(self.q_a_indices == qpos_idx)
            if local_idx.size != 1:
                raise ValueError(
                    f"Plan B full-body optimization requires joint '{joint_name}'"
                )
            posture_indices.append(int(local_idx[0]))
            weight = cfg.posture_weight
            if joint_name in {"waist_yaw_joint", "waist_roll_joint"}:
                weight += cfg.waist_yaw_roll_weight
            posture_weights.append(weight)
        self._plan_b_posture_indices = np.asarray(posture_indices, dtype=int)
        self._plan_b_posture_weights = np.asarray(posture_weights, dtype=float)

    def _init_foot_lock(self, foot_lock: FootLockConfig | None) -> None:
        """Initialize foot lock configuration and normalize window mappings."""
        self.foot_lock = foot_lock or FootLockConfig()
        self._foot_lock_windows: dict[str, tuple[tuple[int, int], ...]] = {"left": (), "right": ()}
        if self.foot_lock.windows is None:
            return
        for key, windows in self.foot_lock.windows.items():
            key_lower = key.lower()
            side = None
            if key_lower.startswith("l") or ("left" in key_lower):
                side = "left"
            elif key_lower.startswith("r") or ("right" in key_lower):
                side = "right"
            if side is None:
                continue

            normalized_windows: list[tuple[int, int]] = []
            for window in windows:
                if len(window) != 2:
                    raise ValueError(f"Invalid foot lock window for {key}: {window}")
                start, end = int(window[0]), int(window[1])
                if end < start:
                    raise ValueError(f"Invalid foot lock window with end < start for {key}: {window}")
                normalized_windows.append((start, end))
            self._foot_lock_windows[side] = tuple(normalized_windows)

    def _init_self_collision(self, self_collision: SelfCollisionConfig | None) -> None:
        """Initialize self-collision configuration and precompute geom pairs."""
        sc = self_collision or SelfCollisionConfig()
        self._self_collision_enabled = sc.enable and len(sc.pairs) > 0
        self._self_collision_tolerance = sc.tolerance
        self._self_collision_windows: list[tuple[int, int]] | None = sc.windows
        self._self_collision_geom_pairs: list[tuple[int, int]] = []

        self._sc_last_vis_frame = -1

        if not self._self_collision_enabled:
            return

        m = self.robot_model

        # Build body_name → [geom_ids] mapping (only geoms with collision enabled)
        body_to_geoms: dict[str, list[int]] = {}
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            body_id = m.geom_bodyid[g]
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            body_to_geoms.setdefault(body_name, []).append(g)

        # Build geom pairs from body name pairs
        for body_a, body_b in sc.pairs:
            geoms_a = body_to_geoms.get(body_a, [])
            geoms_b = body_to_geoms.get(body_b, [])
            if not geoms_a:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_a}'")
            if not geoms_b:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_b}'")
            for ga in geoms_a:
                for gb in geoms_b:
                    self._self_collision_geom_pairs.append((ga, gb))

        print(
            f"[SelfCollision] Initialized with {len(self._self_collision_geom_pairs)} geom pairs "
            f"from {len(sc.pairs)} body pairs, tolerance={sc.tolerance}m"
        )

    def _setup_visualization(self):
        """Setup Viser visualization components."""
        self.server = viser.ViserServer()

        # 1) Ensure a world frame exists (absolute path!)
        try:
            self.server.scene.add_frame("/world", show_axes=False)
        except Exception:
            print("Starting viser")

        # Create parent frames for robot and object
        self.robot_base = self.server.scene.add_frame("/world/robot", show_axes=False)

        print("robot_model_path: ", self.robot_model_path)

        # Load robot URDF
        self.robot_urdf = yourdfpy.URDF.load(
            self.robot_model_path,
            load_meshes=True,
            build_scene_graph=True,
        )

        print("Viser using robot URDF: ", self.robot_model_path)

        # Create ViserUrdf instance for robot, attaching it to the robot_base frame
        self.viser_robot = ViserUrdf(
            self.server,
            urdf_or_path=self.robot_urdf,
            root_node_name="/world/robot",  # This links to the robot_base frame we created
        )

        # Similarly for object
        if self.object_model_path:
            self.object_base = self.server.scene.add_frame("/world/object", show_axes=False)

            self.object_urdf = yourdfpy.URDF.load(
                self.object_model_path,
                load_meshes=True,
                build_scene_graph=True,
            )

            # Create ViserUrdf instance for object, attaching it to the object_base frame
            self.viser_object = ViserUrdf(
                self.server,
                urdf_or_path=self.object_urdf,
                root_node_name="/world/object",  # This links to the object_base frame we created
            )
            print("Viser using object URDF: ", self.object_model_path)

        else:
            self.viser_object = None

        # Check the number of actuated joints and their names
        robot_joint_limits = self.viser_robot.get_actuated_joint_limits()
        print("\nRobot joints:")
        print("Number of actuated joints:", len(robot_joint_limits))
        print("Joint names:", list(robot_joint_limits.keys()))

        # Initialize robot with this configuration
        robot_initial_config = np.zeros(len(robot_joint_limits))
        self.viser_robot.update_cfg(robot_initial_config)

        # Add grid
        self.server.scene.add_grid(
            "/world/grid",
            width=8,
            height=8,
            position=(0.0, 0.0, 0.0),
        )

    def draw_mesh_from_geom(self, model, data, geom_id, geom_name, name="/mesh", color=(50, 150, 255), opacity=0.5):
        """
        Draw a single MuJoCo mesh geom (already baked to world coords) in viser.
        color is [0, 255] RGB ints; opacity is [0,1].
        """
        if not hasattr(self, "server"):
            return
        V, F = _world_mesh_from_geom(model, data, geom_id, geom_name)
        self.server.scene.add_mesh_simple(
            name,
            vertices=V.astype(np.float32),
            faces=F.astype(np.int32),
            position=(0.0, 0.0, 0.0),  # already world-frame
            color=tuple(int(c) for c in color),
            opacity=float(opacity),
        )

    def draw_mesh_pair_with_contact(
        self,
        model,
        data,
        geom_id1,
        geom_id2,
        geom1_name,
        geom2_name,
        fromto=None,
        group_name="pair",
        color1=(50, 150, 255),
        color2=(255, 120, 60),
        opacity=0.45,
        show_segment=True,
    ):
        """
        Draw two meshes and (optionally) a contact/query segment.
        Uses the existing self.draw_keypoints(...) to visualize points.
        """
        # Note: sometime geom does not have mesh, mesh_id will be -1
        if int(model.geom_dataid[geom_id1]) == -1 or int(model.geom_dataid[geom_id2]) == -1:
            return

        base = f"/{group_name}"
        # meshes
        self.draw_mesh_from_geom(model, data, geom_id1, geom1_name, name=f"{base}/mesh1", color=color1, opacity=opacity)
        self.draw_mesh_from_geom(model, data, geom_id2, geom2_name, name=f"{base}/mesh2", color=color2, opacity=opacity)

        # contact points (q: green, c: red) via your draw_keypoints
        if fromto is not None:
            q = np.asarray(fromto[:3], dtype=float)
            c = np.asarray(fromto[3:], dtype=float)

            # your existing helper (rgba expects floats 0..1)
            self.draw_keypoints(q, name=f"{group_name}_q", rgba=(0.0, 1.0, 0.0, 1.0))
            self.draw_keypoints(c, name=f"{group_name}_c", rgba=(1.0, 0.0, 0.0, 1.0))

    def retarget_motion(
        self,
        human_joint_motions,
        object_poses,
        object_poses_augmented,
        object_points_local_demo,
        object_points_local,
        foot_sticking_sequences,
        q_a_init=None,
        q_nominal_list=None,
        original=True,
        dest_res_path=None,
    ):
        """
        The main function to retarget an entire motion sequence frame by frame.

        Args:
            human_joint_motions (np.ndarray): (num_frames, num_joints, 3) array.
            object_poses (np.ndarray): (num_frames, 7) array of demo object poses (quat, trans).
            object_poses_augmented (np.ndarray): (num_frames, 7) array of augmented object poses (quat, trans).
            object_points_local_demo (np.ndarray): Demo object points in local frame (rest pose).
            object_points_local (np.ndarray): Current object points in local frame (rest pose).
            foot_sticking_sequences (list): List of foot sticking sequences for each frame.
            q_a_init (np.ndarray, optional): Initial robot configuration.
            q_a_nominal (np.ndarray, optional): Nominal robot configuration.

        Returns:
            tuple: (retargeted_motions, obj_pts_demo_list, obj_pts_list, tetrahedra)
        """
        num_frames = human_joint_motions.shape[0]
        if q_nominal_list is not None:
            q_locked_list = q_nominal_list
        else:
            q_locked_list = np.zeros((num_frames, self.nq))
            q_locked_list[0, self.q_a_indices] = q_a_init

        q_locked_list[:, -7:] = object_poses_augmented
        q = np.copy(q_locked_list[0])
        retargeted_motions = [q]
        elastic_diagnostics: list[dict[str, float]] = []

        hand_orientation_weights = self._compute_hand_orientation_weights(
            human_joint_motions,
            object_poses,
            object_points_local_demo,
        )
        hand_palm_targets, hand_finger_targets = self._compute_fixed_push_hand_orientation_targets(
            human_joint_motions,
            object_poses,
            object_poses_augmented,
            hand_orientation_weights,
        )
        plan_b_weights = self._compute_plan_b_contact_weights(
            human_joint_motions,
            object_poses,
            object_points_local_demo,
        )
        (
            plan_b_position_task_weights,
            plan_b_position_weights,
        ) = self._compute_plan_b_phase_weights(
            plan_b_weights,
            self.plan_b_palm_contact.fade_frames,
        )
        (
            plan_b_position_targets,
            plan_b_normal_targets,
            plan_b_twist_targets,
        ) = self._compute_plan_b_palm_targets(
            object_poses_augmented,
            plan_b_position_weights,
        )
        tetrahedra = []
        obj_pts_demo_list = []  # scaled object pts
        obj_pts_list = []  # original size object pts

        print(f"\nStarting motion retargeting for {num_frames} frames...")

        with tqdm(range(num_frames)) as pbar:
            for i in pbar:
                # Get object poses and transform points
                object_quat_demo = object_poses[i, 3:]
                object_trans_demo = object_poses[i, :3]

                # Get human joint positions and create interaction mesh in object frame
                human_mapped_joints = human_joint_motions[i, self.smplh_mapped_joint_indices]

                if self.object_name == "ground":
                    human_mapped_joints_in_object = human_mapped_joints
                else:
                    human_mapped_joints_in_object = transform_points_world_to_local(
                        object_quat_demo, object_trans_demo, human_mapped_joints
                    )

                source_vertices, source_tetrahedra = create_interaction_mesh(
                    np.vstack([human_mapped_joints_in_object, object_points_local_demo])
                )
                tetrahedra.append(source_tetrahedra)

                if self.debug:
                    # Only for visualization
                    object_quat = object_poses_augmented[i, 3:]
                    object_trans = object_poses_augmented[i, :3]
                    obj_pts_demo = transform_points_local_to_world(
                        object_quat_demo, object_trans_demo, object_points_local_demo
                    )
                    obj_pts = transform_points_local_to_world(object_quat, object_trans, object_points_local)

                    obj_pts_demo_list.append(obj_pts_demo)
                    obj_pts_list.append(obj_pts)
                    human_kpts_handle_list = self.draw_keypoints(human_mapped_joints, name="human_kpts")  # 15 X 3
                    obj_kpts_demo_handle_list = self.draw_keypoints(
                        obj_pts_demo, name="object_demo_kpts", rgba=(1, 0, 0, 1)
                    )  # 100 X 3
                    obj_kpts_handle_list = self.draw_keypoints(
                        obj_pts, name="object_kpts", rgba=(0, 1, 1, 1)
                    )  # 100 X 3

                # Create adjacency list and calculate target Laplacian coordinates
                adj_list = get_adjacency_list(source_tetrahedra, len(source_vertices))
                target_laplacian = calculate_laplacian_coordinates(source_vertices, adj_list)

                # Run optimization
                if original:
                    w_nominal_tracking = self.w_nominal_tracking_init
                else:
                    w_nominal_tracking = self.w_nominal_tracking_init * np.exp(-i / self.nominal_tracking_tau)

                q, cost = self.iterate(
                    q_locked=q_locked_list[i],
                    q_n=q,
                    q_t_last=retargeted_motions[-1],
                    target_laplacian=target_laplacian,
                    adj_list=adj_list,
                    obj_pts_local=object_points_local,
                    foot_sticking=foot_sticking_sequences[i],
                    w_nominal_tracking=w_nominal_tracking,
                    q_a_nominal=(q_nominal_list[i, self.q_a_indices] if q_nominal_list is not None else None),
                    init_t=i == 0,
                    n_iter=(
                        50
                        if i == 0
                        else (
                            self.plan_b_palm_contact.max_sqp_iterations
                            if self.plan_b_palm_contact.enable
                            else 10
                        )
                    ),
                    frame_idx=i,
                    hand_orientation_weights=hand_orientation_weights[i],
                    hand_palm_targets=hand_palm_targets[i],
                    hand_finger_targets=hand_finger_targets[i],
                    plan_b_weights=plan_b_weights[i],
                    plan_b_position_task_weights=(
                        plan_b_position_task_weights[i]
                    ),
                    plan_b_position_weights=plan_b_position_weights[i],
                    plan_b_position_targets=plan_b_position_targets[i],
                    plan_b_normal_targets=plan_b_normal_targets[i],
                    plan_b_twist_targets=plan_b_twist_targets[i],
                )
                if self.debug:
                    robot_link_positions = self._get_robot_link_positions(
                        q, self.laplacian_match_links.values()
                    )  # 15 X 3
                    robot_kpts_handle_list = self.draw_keypoints(
                        robot_link_positions, name="robot_kpts", rgba=(0, 1, 0, 1)
                    )

                retargeted_motions.append(q)
                if self.elastic_constraints.enable:
                    elastic_diagnostics.append(
                        self._measure_elastic_constraint_diagnostics(
                            q=q,
                            q_t_last=retargeted_motions[-2],
                            q_a_nominal=(
                                q_nominal_list[i, self.q_a_indices]
                                if q_nominal_list is not None
                                else None
                            ),
                            foot_sticking=foot_sticking_sequences[i],
                        )
                    )
                if self.visualize and self.debug:
                    self.draw_q(q)

                pbar.set_postfix(cost=cost)

        # Remove previous debug visualization
        if self.debug:
            for handle in human_kpts_handle_list:
                handle.remove()
            human_kpts_handle_list.clear()

            for handle in obj_kpts_demo_handle_list:
                handle.remove()
            obj_kpts_demo_handle_list.clear()

            for handle in obj_kpts_handle_list:
                handle.remove()
            obj_kpts_handle_list.clear()

            for handle in robot_kpts_handle_list:
                handle.remove()
            robot_kpts_handle_list.clear()

        # Save results
        result_arrays: dict[str, object] = {
            "qpos": np.array(retargeted_motions)[1:],
            "human_joints": human_joint_motions,
            "fps": 30,
            "cost": cost,
        }
        if self.elastic_constraints.enable:
            for key in elastic_diagnostics[0] if elastic_diagnostics else ():
                result_arrays[key] = np.asarray(
                    [frame[key] for frame in elastic_diagnostics],
                    dtype=np.float64,
                )
        np.savez(dest_res_path, **result_arrays)
        print("Saving results to path:", dest_res_path)

        if self.visualize:
            robot_dof = len(self.viser_robot.get_actuated_joint_limits())

            create_motion_control_sliders(
                server=self.server,
                viser_robot=self.viser_robot,
                robot_base_frame=self.robot_base,
                motion_sequence=np.asarray(retargeted_motions)[1:],
                robot_dof=robot_dof,
                viser_object=self.viser_object,
                object_base_frame=getattr(self, "object_base", None) if self.viser_object else None,
                contains_object_in_qpos=bool(self.viser_object) and bool(self.has_dynamic_object),
                initial_fps=30,
                initial_interp_mult=2,
                loop=False,
            )

            # 4) optional: visibility toggle
            with self.server.gui.add_folder("Visibility"):
                show_meshes_cb = self.server.gui.add_checkbox("Show meshes", self.viser_robot.show_visual)

                @show_meshes_cb.on_update
                def _(_):
                    self.viser_robot.show_visual = show_meshes_cb.value
                    if self.viser_object is not None:
                        self.viser_object.show_visual = show_meshes_cb.value

        return (
            np.array(retargeted_motions)[1:],
            obj_pts_demo_list,
            obj_pts_list,
            tetrahedra,
        )

    def _compute_hand_orientation_weights(
        self,
        human_joint_motions: np.ndarray,
        object_poses: np.ndarray,
        object_points_local_demo: np.ndarray,
    ) -> np.ndarray:
        """Detect demo wrist/object contact and fade the optional objective."""
        if not self.hand_orientation.enable:
            return np.zeros(
                (human_joint_motions.shape[0], len(self._hand_orientation_specs)),
                dtype=float,
            )
        return self._compute_proximity_contact_weights(
            human_joint_motions,
            object_poses,
            object_points_local_demo,
            contact_distance=self.hand_orientation.contact_distance,
            fade_frames=self.hand_orientation.fade_frames,
            label="Hand orientation",
            extend_fade=False,
        )

    def _compute_plan_b_contact_weights(
        self,
        human_joint_motions: np.ndarray,
        object_poses: np.ndarray,
        object_points_local_demo: np.ndarray,
    ) -> np.ndarray:
        """Detect the demonstrated contact phase used by Plan B."""
        if not self.plan_b_palm_contact.enable:
            return np.zeros(
                (human_joint_motions.shape[0], len(self._hand_orientation_specs)),
                dtype=float,
            )
        return self._compute_proximity_contact_weights(
            human_joint_motions,
            object_poses,
            object_points_local_demo,
            contact_distance=self.plan_b_palm_contact.contact_distance,
            fade_frames=self.plan_b_palm_contact.fade_frames,
            release_frames=self.plan_b_palm_contact.release_frames,
            label="Plan B palm contact",
            extend_fade=True,
        )

    @staticmethod
    def _compute_plan_b_phase_weights(
        orientation_weights: np.ndarray,
        delay_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split Plan B into early retraction and delayed approach phases.

        The position task starts one delay before orientation so the hands can
        move to their clearance poses first. The approach progress starts one
        delay after orientation so rotation happens while clearance is held.
        """
        orientation_weights = np.asarray(orientation_weights, dtype=float)
        position_task_weights = orientation_weights.copy()
        approach_weights = np.zeros_like(orientation_weights)
        delay = max(0, int(delay_frames))
        if delay == 0:
            approach_weights[:] = orientation_weights
            return position_task_weights, approach_weights
        if delay >= orientation_weights.shape[0]:
            return position_task_weights, approach_weights

        position_task_weights[:-delay] = np.maximum(
            position_task_weights[:-delay],
            orientation_weights[delay:],
        )
        approach_weights[delay:] = np.minimum(
            orientation_weights[delay:],
            orientation_weights[:-delay],
        )
        return position_task_weights, approach_weights

    def _compute_proximity_contact_weights(
        self,
        human_joint_motions: np.ndarray,
        object_poses: np.ndarray,
        object_points_local_demo: np.ndarray,
        *,
        contact_distance: float,
        fade_frames: int,
        release_frames: int | None = None,
        label: str,
        extend_fade: bool,
    ) -> np.ndarray:
        """Detect wrist/object proximity runs and add symmetric temporal fades."""
        num_frames = human_joint_motions.shape[0]
        num_hands = len(self._hand_orientation_specs)
        weights = np.zeros((num_frames, num_hands), dtype=float)

        contact = np.zeros_like(weights, dtype=bool)
        for frame_idx in range(num_frames):
            object_quat = object_poses[frame_idx, 3:]
            object_trans = object_poses[frame_idx, :3]
            for hand_idx, spec in enumerate(self._hand_orientation_specs):
                wrist_world = human_joint_motions[frame_idx, int(spec["demo_joint_idx"])]
                wrist_local = transform_points_world_to_local(
                    object_quat,
                    object_trans,
                    wrist_world[None, :],
                )[0]
                distance = np.linalg.norm(object_points_local_demo - wrist_local, axis=1).min()
                contact[frame_idx, hand_idx] = distance <= contact_distance

        fade_frames = max(0, int(fade_frames))
        release_frames = (
            fade_frames
            if release_frames is None
            else max(0, int(release_frames))
        )
        for hand_idx in range(num_hands):
            active_frames = np.flatnonzero(contact[:, hand_idx])
            if active_frames.size == 0:
                continue
            split_points = np.flatnonzero(np.diff(active_frames) > 1) + 1
            for run in np.split(active_frames, split_points):
                start, end = int(run[0]), int(run[-1])
                if extend_fade:
                    for frame_idx in range(start, end + 1):
                        weights[frame_idx, hand_idx] = min(
                            1.0,
                            (frame_idx - start + 1) / (fade_frames + 1),
                        )
                    for offset in range(1, release_frames + 1):
                        fade_weight = (release_frames - offset + 1) / (
                            release_frames + 1
                        )
                        after = end + offset
                        if after < num_frames:
                            weights[after, hand_idx] = max(
                                weights[after, hand_idx],
                                fade_weight,
                            )
                    continue
                for frame_idx in range(start, end + 1):
                    if fade_frames == 0:
                        weights[frame_idx, hand_idx] = 1.0
                        continue
                    fade_in = (frame_idx - start + 1) / (fade_frames + 1)
                    fade_out = (end - frame_idx + 1) / (fade_frames + 1)
                    weights[frame_idx, hand_idx] = min(1.0, fade_in, fade_out)

        for hand_idx, spec in enumerate(self._hand_orientation_specs):
            active = np.flatnonzero(weights[:, hand_idx] > 0)
            if active.size:
                print(
                    f"{label} ({spec['side']}): frames "
                    f"{active[0]}..{active[-1]} (including fade)"
                )
        return weights

    def _map_pt_palm_orientations_to_robot_links(
        self,
        palm_orientations: np.ndarray | None,
        num_frames: int,
    ) -> np.ndarray:
        """Convert anatomical PT palm frames to robot hand-link rotations."""
        num_hands = len(self._hand_orientation_specs)
        targets = np.zeros((num_frames, num_hands, 3, 3), dtype=float)
        if not (
            self.pt_wrist_orientation.enable
            or getattr(
                self,
                "pt_full_arm_orientation",
                PTFullArmOrientationConfig(),
            ).enable
            or getattr(
                self,
                "pt_wrist_dominant_surface",
                PTWristDominantSurfaceConfig(),
            ).enable
            or getattr(self, "pt_palm_collision", PTPalmCollisionConfig()).enable
        ):
            return targets
        if palm_orientations is None:
            raise ValueError("PT wrist orientation tracking requires palm orientation targets")

        palm_orientations = np.asarray(palm_orientations, dtype=float)
        if palm_orientations.shape != (num_frames, 2, 3, 3):
            raise ValueError(
                "Expected PT palm orientations with shape "
                f"({num_frames}, 2, 3, 3), got {palm_orientations.shape}"
            )

        source_indices = {"left": 0, "right": 1}
        for hand_idx, spec in enumerate(self._hand_orientation_specs):
            source_idx = source_indices[str(spec["side"])]
            palm_basis = np.asarray(spec["palm_basis"], dtype=float)
            targets[:, hand_idx] = palm_orientations[:, source_idx] @ palm_basis.T
        return targets

    def apply_pt_palm_collision_postprocess(
        self,
        qpos: np.ndarray,
        palm_orientations: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Refine only the arms, reusing the existing MuJoCo collision geometry."""
        from holosoma_retargeting.src.pt_palm_collision import refine_pt_palms_with_collision

        return refine_pt_palms_with_collision(self, qpos, palm_orientations)

    def apply_pt_wrist_orientation_postprocess(
        self,
        qpos_sequence: np.ndarray,
        palm_orientations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply A.1 by solving only wrist roll/pitch/yaw in every frame.

        All floating-base, leg, waist, shoulder, elbow, and object coordinates
        are copied unchanged from the fixed-object baseline. Object collision
        constraints are intentionally not re-solved in this final stage: A.1
        preserves the demonstrated rigid-hand orientation as the training
        reference, even when the larger rubber-hand mesh visually overlaps the
        demonstration object.

        Returns:
            ``(qpos, orientation_errors_deg)``. Errors have shape ``(T, 2)``
            in left/right hand order.
        """
        if not self.pt_wrist_orientation.enable:
            raise ValueError("PT wrist orientation post-processing is disabled")

        result = np.asarray(qpos_sequence, dtype=float).copy()
        if result.ndim != 2 or result.shape[1] != self.nq:
            raise ValueError(
                f"Expected qpos with shape (T, {self.nq}), got {result.shape}"
            )

        targets = self._map_pt_palm_orientations_to_robot_links(
            palm_orientations,
            result.shape[0],
        )
        errors_deg = np.zeros((result.shape[0], len(self._hand_orientation_specs)))
        previous_solutions = [
            result[0, np.asarray(spec["pt_wrist_qpos_indices"], dtype=int)].copy()
            for spec in self._hand_orientation_specs
        ]

        for frame_idx in range(result.shape[0]):
            frame_q = result[frame_idx].copy()
            for hand_idx, spec in enumerate(self._hand_orientation_specs):
                qpos_indices = np.asarray(spec["pt_wrist_qpos_indices"], dtype=int)
                lower = np.asarray(spec["pt_wrist_lower_limits"], dtype=float)
                upper = np.asarray(spec["pt_wrist_upper_limits"], dtype=float)
                body_id = int(spec["body_id"])
                target_rotation = targets[frame_idx, hand_idx]

                def orientation_residual(wrist_qpos: np.ndarray) -> np.ndarray:
                    self.robot_data.qpos[:] = frame_q
                    self.robot_data.qpos[qpos_indices] = wrist_qpos
                    mujoco.mj_forward(self.robot_model, self.robot_data)
                    current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                    return Rotation.from_matrix(
                        target_rotation @ current_rotation.T
                    ).as_rotvec()

                initial = np.clip(
                    previous_solutions[hand_idx],
                    lower + 1e-9,
                    upper - 1e-9,
                )
                solution = least_squares(
                    orientation_residual,
                    initial,
                    bounds=(lower, upper),
                    xtol=1e-13,
                    ftol=1e-13,
                    gtol=1e-13,
                    max_nfev=120,
                )
                frame_q[qpos_indices] = solution.x
                previous_solutions[hand_idx] = solution.x.copy()
                errors_deg[frame_idx, hand_idx] = np.degrees(
                    np.linalg.norm(orientation_residual(solution.x))
                )

            result[frame_idx] = frame_q

        max_error = float(errors_deg.max(initial=0.0))
        if max_error > self.pt_wrist_orientation.max_solver_error_deg:
            raise RuntimeError(
                "A.1 wrist solve exceeded the configured orientation error: "
                f"{max_error:.6f} deg > "
                f"{self.pt_wrist_orientation.max_solver_error_deg:.6f} deg"
            )
        return result, errors_deg

    def apply_pt_full_arm_orientation_postprocess(
        self,
        qpos_sequence: np.ndarray,
        palm_orientations: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        """Refine demonstrated palms with each seven-DoF arm independently.

        The floating base, waist, legs, opposite arm, and interaction object are
        copied exactly from ``qpos_sequence``.  Each hand solve trades off the
        raw demonstrated palm orientation against the baseline hand-link
        position, a baseline arm-pose prior, and temporal continuity of the
        *correction*.  Smoothing the correction rather than the output motion
        preserves intentional fast motion already present in the baseline.

        Returns:
            ``(qpos, metrics)``.  Per-frame metrics use left/right hand order
            and retain the complete seven-joint corrections for auditing.
        """
        config = self.pt_full_arm_orientation
        if not config.enable:
            raise ValueError("PT full-arm orientation post-processing is disabled")

        baseline = np.asarray(qpos_sequence, dtype=float)
        if baseline.ndim != 2 or baseline.shape[1] != self.nq:
            raise ValueError(
                f"Expected qpos with shape (T, {self.nq}), got {baseline.shape}"
            )
        if baseline.shape[0] == 0:
            raise ValueError("PT full-arm orientation requires at least one frame")
        if not np.isfinite(baseline).all():
            raise ValueError("PT full-arm qpos input contains non-finite values")

        targets = self._map_pt_palm_orientations_to_robot_links(
            palm_orientations,
            baseline.shape[0],
        )
        if not np.isfinite(targets).all():
            raise ValueError("PT full-arm palm targets contain non-finite values")

        num_frames = baseline.shape[0]
        num_hands = len(self._hand_orientation_specs)
        if num_hands != 2:
            raise ValueError(
                f"PT full-arm orientation requires two hand specifications, got {num_hands}"
            )

        result = baseline.copy()
        orientation_errors_deg = np.zeros((num_frames, num_hands), dtype=float)
        hand_position_errors_m = np.zeros((num_frames, num_hands), dtype=float)
        arm_corrections_rad = np.zeros((num_frames, num_hands, 7), dtype=float)
        solver_success = np.zeros((num_frames, num_hands), dtype=bool)
        solver_nfev = np.zeros((num_frames, num_hands), dtype=np.int64)
        solver_cost = np.zeros((num_frames, num_hands), dtype=float)
        arm_qpos_indices = np.empty((num_hands, 7), dtype=np.int64)

        previous_corrections = [np.zeros(7, dtype=float) for _ in range(num_hands)]
        sqrt_orientation_weight = np.sqrt(config.orientation_weight)
        sqrt_position_weight = np.sqrt(config.hand_position_weight)
        sqrt_prior_weight = np.sqrt(config.arm_prior_weight)
        sqrt_temporal_weight = np.sqrt(config.correction_temporal_weight)

        for hand_idx, spec in enumerate(self._hand_orientation_specs):
            indices = np.asarray(spec["pt_full_arm_qpos_indices"], dtype=int)
            if indices.shape != (7,):
                raise ValueError(
                    "PT full-arm hand specification must contain seven qpos indices"
                )
            arm_qpos_indices[hand_idx] = indices

        for frame_idx in range(num_frames):
            baseline_q = baseline[frame_idx].copy()
            self.robot_data.qpos[:] = baseline_q
            mujoco.mj_forward(self.robot_model, self.robot_data)
            baseline_positions = np.stack(
                [
                    self.robot_data.xpos[int(spec["body_id"])].copy()
                    for spec in self._hand_orientation_specs
                ]
            )

            frame_q = baseline_q.copy()
            for hand_idx, spec in enumerate(self._hand_orientation_specs):
                qpos_indices = arm_qpos_indices[hand_idx]
                lower = np.asarray(spec["pt_full_arm_lower_limits"], dtype=float)
                upper = np.asarray(spec["pt_full_arm_upper_limits"], dtype=float)
                if lower.shape != (7,) or upper.shape != (7,):
                    raise ValueError(
                        "PT full-arm hand specification must contain seven joint bounds"
                    )
                body_id = int(spec["body_id"])
                baseline_arm = baseline_q[qpos_indices].copy()
                previous_correction = previous_corrections[hand_idx]
                target_rotation = targets[frame_idx, hand_idx]
                target_position = baseline_positions[hand_idx]

                def full_arm_residual(arm_qpos: np.ndarray) -> np.ndarray:
                    candidate_q = baseline_q.copy()
                    candidate_q[qpos_indices] = arm_qpos
                    self.robot_data.qpos[:] = candidate_q
                    mujoco.mj_forward(self.robot_model, self.robot_data)
                    current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                    current_position = self.robot_data.xpos[body_id]
                    correction = arm_qpos - baseline_arm
                    residuals = [
                        sqrt_orientation_weight
                        * Rotation.from_matrix(
                            target_rotation @ current_rotation.T
                        ).as_rotvec(),
                        sqrt_position_weight
                        * (current_position - target_position),
                    ]
                    if sqrt_prior_weight > 0.0:
                        residuals.append(sqrt_prior_weight * correction)
                    if sqrt_temporal_weight > 0.0:
                        residuals.append(
                            sqrt_temporal_weight
                            * (correction - previous_correction)
                        )
                    return np.concatenate(residuals)

                initial = np.clip(
                    baseline_arm + previous_correction,
                    lower + 1.0e-9,
                    upper - 1.0e-9,
                )
                solution = least_squares(
                    full_arm_residual,
                    initial,
                    bounds=(lower, upper),
                    xtol=1.0e-11,
                    ftol=1.0e-11,
                    gtol=1.0e-11,
                    max_nfev=int(config.max_nfev),
                )
                if not np.isfinite(solution.x).all():
                    raise RuntimeError(
                        "PT full-arm solver produced non-finite joint values at "
                        f"frame {frame_idx}, hand {spec['side']}"
                    )

                frame_q[qpos_indices] = solution.x
                correction = solution.x - baseline_arm
                previous_corrections[hand_idx] = correction.copy()
                arm_corrections_rad[frame_idx, hand_idx] = correction
                solver_success[frame_idx, hand_idx] = bool(solution.success)
                solver_nfev[frame_idx, hand_idx] = int(solution.nfev)
                solver_cost[frame_idx, hand_idx] = float(solution.cost)

                self.robot_data.qpos[:] = frame_q
                mujoco.mj_forward(self.robot_model, self.robot_data)
                current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                current_position = self.robot_data.xpos[body_id]
                orientation_errors_deg[frame_idx, hand_idx] = np.degrees(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            target_rotation @ current_rotation.T
                        ).as_rotvec()
                    )
                )
                hand_position_errors_m[frame_idx, hand_idx] = np.linalg.norm(
                    current_position - target_position
                )

            result[frame_idx] = frame_q

        if not np.isfinite(result).all():
            raise RuntimeError("PT full-arm result contains non-finite qpos values")
        for hand_idx, spec in enumerate(self._hand_orientation_specs):
            indices = arm_qpos_indices[hand_idx]
            lower = np.asarray(spec["pt_full_arm_lower_limits"], dtype=float)
            upper = np.asarray(spec["pt_full_arm_upper_limits"], dtype=float)
            if np.any(result[:, indices] < lower - 1.0e-8) or np.any(
                result[:, indices] > upper + 1.0e-8
            ):
                raise RuntimeError(
                    f"PT full-arm result violates {spec['side']} arm joint limits"
                )

        arm_steps_rad = np.zeros_like(arm_corrections_rad)
        correction_steps_rad = np.zeros_like(arm_corrections_rad)
        if num_frames > 1:
            for hand_idx in range(num_hands):
                indices = arm_qpos_indices[hand_idx]
                arm_steps_rad[1:, hand_idx] = np.diff(
                    result[:, indices],
                    axis=0,
                )
            correction_steps_rad[1:] = np.diff(arm_corrections_rad, axis=0)

        metrics: dict[str, np.ndarray | float] = {
            "orientation_errors_deg": orientation_errors_deg,
            "hand_position_errors_m": hand_position_errors_m,
            "arm_corrections_rad": arm_corrections_rad,
            "arm_steps_rad": arm_steps_rad,
            "correction_steps_rad": correction_steps_rad,
            "arm_qpos_indices": arm_qpos_indices,
            "solver_success": solver_success,
            "solver_nfev": solver_nfev,
            "solver_cost": solver_cost,
            "orientation_error_p95_deg": np.percentile(
                orientation_errors_deg,
                95,
                axis=0,
            ),
            "orientation_error_max_deg": orientation_errors_deg.max(axis=0),
            "hand_position_error_p95_m": np.percentile(
                hand_position_errors_m,
                95,
                axis=0,
            ),
            "hand_position_error_max_m": hand_position_errors_m.max(axis=0),
            "arm_correction_abs_max_rad": np.max(
                np.abs(arm_corrections_rad),
                axis=(0, 2),
            ),
            "correction_step_abs_max_rad": np.max(
                np.abs(correction_steps_rad),
                axis=(0, 2),
            ),
            "solver_success_rate": solver_success.mean(axis=0),
        }
        return result, metrics

    def apply_pt_wrist_dominant_surface_postprocess(
        self,
        qpos_sequence: np.ndarray,
        palm_orientations: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        """Refine palm surfaces while keeping shoulder/elbow motion conservative.

        Both arms are solved independently.  The rubber-hand support point is
        held near its baseline world position, palm-normal alignment is the
        primary orientation task, and the demonstrated finger direction is a
        weaker twist cue.  Priors and temporal correction weights are larger
        for the proximal four joints than for the three wrist joints.
        """
        config = self.pt_wrist_dominant_surface
        if not config.enable:
            raise ValueError(
                "PT wrist-dominant surface post-processing is disabled"
            )

        baseline = np.asarray(qpos_sequence, dtype=float)
        if baseline.ndim != 2 or baseline.shape[1] != self.nq:
            raise ValueError(
                f"Expected qpos with shape (T, {self.nq}), got {baseline.shape}"
            )
        if baseline.shape[0] == 0:
            raise ValueError(
                "PT wrist-dominant surface refinement requires at least one frame"
            )
        if not np.isfinite(baseline).all():
            raise ValueError(
                "PT wrist-dominant surface qpos input contains non-finite values"
            )

        targets = self._map_pt_palm_orientations_to_robot_links(
            palm_orientations,
            baseline.shape[0],
        )
        if not np.isfinite(targets).all():
            raise ValueError(
                "PT wrist-dominant surface palm targets contain non-finite values"
            )

        num_frames = baseline.shape[0]
        num_hands = len(self._hand_orientation_specs)
        if num_hands != 2:
            raise ValueError(
                "PT wrist-dominant surface refinement requires two hand "
                f"specifications, got {num_hands}"
            )

        result = baseline.copy()
        orientation_errors_deg = np.zeros((num_frames, num_hands), dtype=float)
        palm_normal_errors_deg = np.zeros((num_frames, num_hands), dtype=float)
        finger_direction_errors_deg = np.zeros(
            (num_frames, num_hands),
            dtype=float,
        )
        surface_position_errors_m = np.zeros(
            (num_frames, num_hands),
            dtype=float,
        )
        link_origin_errors_m = np.zeros((num_frames, num_hands), dtype=float)
        arm_corrections_rad = np.zeros((num_frames, num_hands, 7), dtype=float)
        solver_success = np.zeros((num_frames, num_hands), dtype=bool)
        solver_nfev = np.zeros((num_frames, num_hands), dtype=np.int64)
        solver_cost = np.zeros((num_frames, num_hands), dtype=float)
        arm_qpos_indices = np.empty((num_hands, 7), dtype=np.int64)

        previous_corrections = [np.zeros(7, dtype=float) for _ in range(num_hands)]
        sqrt_normal_weight = np.sqrt(config.normal_weight)
        sqrt_finger_weight = np.sqrt(config.finger_weight)
        sqrt_surface_position_weight = np.sqrt(config.surface_position_weight)
        sqrt_proximal_prior_weight = np.sqrt(config.proximal_prior_weight)
        sqrt_wrist_prior_weight = np.sqrt(config.wrist_prior_weight)
        sqrt_proximal_temporal_weight = np.sqrt(
            config.proximal_correction_temporal_weight
        )
        sqrt_wrist_temporal_weight = np.sqrt(
            config.wrist_correction_temporal_weight
        )

        for hand_idx, spec in enumerate(self._hand_orientation_specs):
            indices = np.asarray(spec["pt_full_arm_qpos_indices"], dtype=int)
            if indices.shape != (7,):
                raise ValueError(
                    "PT wrist-dominant surface hand specification must contain "
                    "seven qpos indices"
                )
            if "palm_contact_point" not in spec:
                raise ValueError(
                    "PT wrist-dominant surface hand specification is missing "
                    "the rubber-palm support point"
                )
            arm_qpos_indices[hand_idx] = indices

        def unit_vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
            cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
            return float(np.degrees(np.arccos(cosine)))

        for frame_idx in range(num_frames):
            baseline_q = baseline[frame_idx].copy()
            self.robot_data.qpos[:] = baseline_q
            mujoco.mj_forward(self.robot_model, self.robot_data)
            baseline_link_origins = np.stack(
                [
                    self.robot_data.xpos[int(spec["body_id"])].copy()
                    for spec in self._hand_orientation_specs
                ]
            )
            baseline_surface_points = np.stack(
                [
                    self.robot_data.xpos[int(spec["body_id"])]
                    + self.robot_data.xmat[int(spec["body_id"])].reshape(3, 3)
                    @ np.asarray(spec["palm_contact_point"], dtype=float)
                    for spec in self._hand_orientation_specs
                ]
            )

            frame_q = baseline_q.copy()
            for hand_idx, spec in enumerate(self._hand_orientation_specs):
                qpos_indices = arm_qpos_indices[hand_idx]
                lower = np.asarray(spec["pt_full_arm_lower_limits"], dtype=float)
                upper = np.asarray(spec["pt_full_arm_upper_limits"], dtype=float)
                if lower.shape != (7,) or upper.shape != (7,):
                    raise ValueError(
                        "PT wrist-dominant surface hand specification must "
                        "contain seven joint bounds"
                    )
                body_id = int(spec["body_id"])
                baseline_arm = baseline_q[qpos_indices].copy()
                previous_correction = previous_corrections[hand_idx]
                palm_normal_local = np.asarray(spec["palm_normal"], dtype=float)
                palm_normal_local /= np.linalg.norm(palm_normal_local)
                finger_local = np.asarray(spec["palm_basis"], dtype=float)[:, 0]
                finger_local /= np.linalg.norm(finger_local)
                palm_support_local = np.asarray(
                    spec["palm_contact_point"],
                    dtype=float,
                )
                target_rotation = targets[frame_idx, hand_idx]
                target_normal = target_rotation @ palm_normal_local
                target_finger = target_rotation @ finger_local
                target_surface_point = baseline_surface_points[hand_idx]

                def surface_residual(arm_qpos: np.ndarray) -> np.ndarray:
                    candidate_q = baseline_q.copy()
                    candidate_q[qpos_indices] = arm_qpos
                    self.robot_data.qpos[:] = candidate_q
                    mujoco.mj_forward(self.robot_model, self.robot_data)
                    current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                    current_surface_point = (
                        self.robot_data.xpos[body_id]
                        + current_rotation @ palm_support_local
                    )
                    current_normal = current_rotation @ palm_normal_local
                    current_finger = current_rotation @ finger_local
                    correction = arm_qpos - baseline_arm
                    residuals = [
                        sqrt_normal_weight * (current_normal - target_normal),
                        sqrt_surface_position_weight
                        * (current_surface_point - target_surface_point),
                    ]
                    if sqrt_finger_weight > 0.0:
                        residuals.append(
                            sqrt_finger_weight * (current_finger - target_finger)
                        )
                    if sqrt_proximal_prior_weight > 0.0:
                        residuals.append(
                            sqrt_proximal_prior_weight * correction[:4]
                        )
                    if sqrt_wrist_prior_weight > 0.0:
                        residuals.append(
                            sqrt_wrist_prior_weight * correction[4:]
                        )
                    temporal_correction = correction - previous_correction
                    if sqrt_proximal_temporal_weight > 0.0:
                        residuals.append(
                            sqrt_proximal_temporal_weight
                            * temporal_correction[:4]
                        )
                    if sqrt_wrist_temporal_weight > 0.0:
                        residuals.append(
                            sqrt_wrist_temporal_weight
                            * temporal_correction[4:]
                        )
                    return np.concatenate(residuals)

                initial = np.clip(
                    baseline_arm + previous_correction,
                    lower + 1.0e-9,
                    upper - 1.0e-9,
                )
                solution = least_squares(
                    surface_residual,
                    initial,
                    bounds=(lower, upper),
                    xtol=1.0e-11,
                    ftol=1.0e-11,
                    gtol=1.0e-11,
                    max_nfev=int(config.max_nfev),
                )
                if not np.isfinite(solution.x).all():
                    raise RuntimeError(
                        "PT wrist-dominant surface solver produced non-finite "
                        f"joint values at frame {frame_idx}, hand {spec['side']}"
                    )

                frame_q[qpos_indices] = solution.x
                correction = solution.x - baseline_arm
                previous_corrections[hand_idx] = correction.copy()
                arm_corrections_rad[frame_idx, hand_idx] = correction
                solver_success[frame_idx, hand_idx] = bool(solution.success)
                solver_nfev[frame_idx, hand_idx] = int(solution.nfev)
                solver_cost[frame_idx, hand_idx] = float(solution.cost)

                self.robot_data.qpos[:] = frame_q
                mujoco.mj_forward(self.robot_model, self.robot_data)
                current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                current_origin = self.robot_data.xpos[body_id]
                current_surface_point = (
                    current_origin + current_rotation @ palm_support_local
                )
                current_normal = current_rotation @ palm_normal_local
                current_finger = current_rotation @ finger_local
                orientation_errors_deg[frame_idx, hand_idx] = np.degrees(
                    Rotation.from_matrix(
                        target_rotation @ current_rotation.T
                    ).magnitude()
                )
                palm_normal_errors_deg[frame_idx, hand_idx] = (
                    unit_vector_angle_deg(current_normal, target_normal)
                )
                finger_direction_errors_deg[frame_idx, hand_idx] = (
                    unit_vector_angle_deg(current_finger, target_finger)
                )
                surface_position_errors_m[frame_idx, hand_idx] = np.linalg.norm(
                    current_surface_point - target_surface_point
                )
                link_origin_errors_m[frame_idx, hand_idx] = np.linalg.norm(
                    current_origin - baseline_link_origins[hand_idx]
                )

            result[frame_idx] = frame_q

        if not np.isfinite(result).all():
            raise RuntimeError(
                "PT wrist-dominant surface result contains non-finite qpos values"
            )
        for hand_idx, spec in enumerate(self._hand_orientation_specs):
            indices = arm_qpos_indices[hand_idx]
            lower = np.asarray(spec["pt_full_arm_lower_limits"], dtype=float)
            upper = np.asarray(spec["pt_full_arm_upper_limits"], dtype=float)
            if np.any(result[:, indices] < lower - 1.0e-8) or np.any(
                result[:, indices] > upper + 1.0e-8
            ):
                raise RuntimeError(
                    "PT wrist-dominant surface result violates "
                    f"{spec['side']} arm joint limits"
                )

        arm_steps_rad = np.zeros_like(arm_corrections_rad)
        correction_steps_rad = np.zeros_like(arm_corrections_rad)
        if num_frames > 1:
            for hand_idx in range(num_hands):
                indices = arm_qpos_indices[hand_idx]
                arm_steps_rad[1:, hand_idx] = np.diff(
                    result[:, indices],
                    axis=0,
                )
            correction_steps_rad[1:] = np.diff(arm_corrections_rad, axis=0)

        metrics: dict[str, np.ndarray | float] = {
            "orientation_errors_deg": orientation_errors_deg,
            "palm_normal_errors_deg": palm_normal_errors_deg,
            "finger_direction_errors_deg": finger_direction_errors_deg,
            "surface_position_errors_m": surface_position_errors_m,
            "link_origin_errors_m": link_origin_errors_m,
            "arm_corrections_rad": arm_corrections_rad,
            "arm_steps_rad": arm_steps_rad,
            "correction_steps_rad": correction_steps_rad,
            "arm_qpos_indices": arm_qpos_indices,
            "solver_success": solver_success,
            "solver_nfev": solver_nfev,
            "solver_cost": solver_cost,
            "palm_normal_error_p95_deg": np.percentile(
                palm_normal_errors_deg,
                95,
                axis=0,
            ),
            "palm_normal_error_max_deg": palm_normal_errors_deg.max(axis=0),
            "finger_direction_error_p95_deg": np.percentile(
                finger_direction_errors_deg,
                95,
                axis=0,
            ),
            "finger_direction_error_max_deg": finger_direction_errors_deg.max(
                axis=0
            ),
            "surface_position_error_p95_m": np.percentile(
                surface_position_errors_m,
                95,
                axis=0,
            ),
            "surface_position_error_max_m": surface_position_errors_m.max(
                axis=0
            ),
            "solver_success_rate": solver_success.mean(axis=0),
        }
        return result, metrics

    def _compute_fixed_push_hand_orientation_targets(
        self,
        human_joint_motions: np.ndarray,
        object_poses: np.ndarray,
        object_poses_augmented: np.ndarray,
        hand_orientation_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a palm-forward, long-axis-along-edge rigid pushing pose."""
        num_frames = human_joint_motions.shape[0]
        targets_shape = (num_frames, len(self._hand_orientation_specs), 3)
        palm_targets = np.zeros(targets_shape, dtype=float)
        finger_targets = np.zeros_like(palm_targets)
        if not self.hand_orientation.enable:
            return palm_targets, finger_targets

        for hand_idx, spec in enumerate(self._hand_orientation_specs):
            active_frames = np.flatnonzero(hand_orientation_weights[:, hand_idx] > 0.0)
            if active_frames.size == 0:
                continue

            # Determine which side of the object this hand pushes from, then
            # keep that direction constant in the object's local frame.
            inward_directions_local = []
            wrist_idx = int(spec["demo_joint_idx"])
            for frame_idx in active_frames:
                demo_quat = object_poses[frame_idx, 3:]
                demo_rotation = Rotation.from_quat(
                    [demo_quat[1], demo_quat[2], demo_quat[3], demo_quat[0]]
                ).as_matrix()
                inward_world = (
                    object_poses[frame_idx, :3]
                    - human_joint_motions[frame_idx, wrist_idx]
                )
                inward_world[2] = 0.0
                inward_norm = np.linalg.norm(inward_world)
                if inward_norm > 1e-8:
                    inward_directions_local.append(
                        demo_rotation.T @ (inward_world / inward_norm)
                    )

            if not inward_directions_local:
                raise ValueError(
                    f"Cannot determine the {spec['side']} hand pushing direction"
                )
            inward_local = np.mean(inward_directions_local, axis=0)
            inward_local /= np.linalg.norm(inward_local)

            for frame_idx in range(num_frames):
                target_quat = object_poses_augmented[frame_idx, 3:]
                target_rotation = Rotation.from_quat(
                    [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
                ).as_matrix()
                inward_world = target_rotation @ inward_local
                inward_world[2] = 0.0
                inward_norm = np.linalg.norm(inward_world)
                if inward_norm < 1e-8:
                    raise ValueError("Fixed pushing direction is parallel to world up")
                inward_world /= inward_norm
                world_up = np.array([0.0, 0.0, 1.0])
                lateral_world = np.cross(world_up, inward_world)
                lateral_world /= np.linalg.norm(lateral_world)

                finger_targets[frame_idx, hand_idx] = lateral_world
                palm_targets[frame_idx, hand_idx] = inward_world

        return palm_targets, finger_targets

    def _compute_plan_b_palm_targets(
        self,
        object_poses_augmented: np.ndarray,
        contact_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build table-local Plan B contact points and orientations in world coordinates."""
        num_frames = object_poses_augmented.shape[0]
        num_hands = len(self._hand_orientation_specs)
        target_shape = (num_frames, num_hands, 3)
        position_targets = np.zeros(target_shape, dtype=float)
        normal_targets = np.zeros(target_shape, dtype=float)
        twist_targets = np.zeros(target_shape, dtype=float)
        if not self.plan_b_palm_contact.enable:
            return position_targets, normal_targets, twist_targets
        if self._plan_b_tabletop_spec is None:
            raise RuntimeError("Plan B tabletop geometry was not initialized")

        target_points_body = self._plan_b_tabletop_spec["target_points_body"]
        outward_body = np.asarray(
            self._plan_b_tabletop_spec["outward_body"],
            dtype=float,
        )
        twist_direction_body = np.asarray(
            self._plan_b_tabletop_spec["twist_direction_body"],
            dtype=float,
        )
        cfg = self.plan_b_palm_contact

        for frame_idx in range(num_frames):
            object_position = object_poses_augmented[frame_idx, :3]
            object_quat = object_poses_augmented[frame_idx, 3:]
            object_rotation = Rotation.from_quat(
                [
                    object_quat[1],
                    object_quat[2],
                    object_quat[3],
                    object_quat[0],
                ]
            ).as_matrix()
            outward_world = object_rotation @ outward_body
            outward_world /= np.linalg.norm(outward_world)
            for hand_idx, spec in enumerate(self._hand_orientation_specs):
                side = str(spec["side"])
                face_point_world = (
                    object_position
                    + object_rotation @ np.asarray(target_points_body[side], dtype=float)
                )
                weight = float(contact_weights[frame_idx, hand_idx])
                clearance = (
                    cfg.surface_gap
                    + cfg.approach_clearance * (1.0 - weight)
                )
                position_targets[frame_idx, hand_idx] = (
                    face_point_world + clearance * outward_world
                )
                normal_targets[frame_idx, hand_idx] = -outward_world
                twist_world = object_rotation @ twist_direction_body
                twist_world -= float(twist_world @ outward_world) * outward_world
                twist_targets[frame_idx, hand_idx] = (
                    twist_world / np.linalg.norm(twist_world)
                )

        return position_targets, normal_targets, twist_targets

    def solve_single_iteration(
        self,
        q_locked: np.ndarray,
        q_a_n_last: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        verbose=False,
        init_t=False,
        frame_idx: int = 0,
        hand_orientation_weights: np.ndarray | None = None,
        hand_palm_targets: np.ndarray | None = None,
        hand_finger_targets: np.ndarray | None = None,
        plan_b_weights: np.ndarray | None = None,
        plan_b_position_task_weights: np.ndarray | None = None,
        plan_b_position_weights: np.ndarray | None = None,
        plan_b_position_targets: np.ndarray | None = None,
        plan_b_normal_targets: np.ndarray | None = None,
        plan_b_twist_targets: np.ndarray | None = None,
    ):
        """The main function to solve a single iteration of the DiffIK problem.
        Args:
            q_locked: the locked robot and object configuration.
            q_a_n_last: the last optimized robot configuration at current time step.
            q_t_last: the robot and object configuration at the last time step.
            foot_sticking: a sequence of booleans indicating whether the foot [left, right] is sticking to the ground.
            smpl_joints: the (possibly scaled) SMPL joint positions to match for IK.
            q_ref: the reference robot configuration.
            smpl_joints_original: the original SMPL joint positions (used for contact matching).
            obj_original: the original object pose (used for contact matching).
            init_t: the current time step is the first time step.
            frame_idx: frame index used by explicit foot lock window constraints.
        """
        assert len(q_a_n_last) == self.nq_a

        # Lock the object pose and set the current robot slice to last accepted solution
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last

        # Compute Laplacian pieces
        J_OC_dict, p_OC_dict, _ = self._calc_manipulator_jacobians(
            q, links=self.laplacian_match_links, obj_frame=(self.object_name != "ground")
        )
        robot_link_keys = list(self.laplacian_match_links.keys())
        V_r = len(robot_link_keys)
        V_o = len(obj_pts_local)
        V = V_r + V_o

        # Stack Jacobians for robot points
        J_V = np.zeros((3 * V, self.nq_a))
        for i, key in enumerate(robot_link_keys):
            J_V[3 * i : 3 * (i + 1), :] = J_OC_dict[key]

        robot_pts_local = np.array([p_OC_dict[k] for k in robot_link_keys])
        vertices = np.vstack([robot_pts_local, obj_pts_local])  # (V x 3)

        L = calculate_laplacian_matrix(vertices, adj_list)  # (V x V), EXPECT SPARSE OR SMALL
        if not sp.issparse(L):
            L = sp.csr_matrix(L)

        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        J_L = Kron @ J_V

        lap0 = L @ vertices
        lap0_vec = lap0.reshape(-1)  # (3V,)
        target_lap_vec = target_laplacian.reshape(-1)  # (3V,)

        w_v = (self.laplacian_weights * np.ones(V)).astype(float)  # (V,)
        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

        # Decision variables
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")
        lap_var = cp.Variable(3 * V, name="laplacian")

        # Constraints list
        constraints = []
        object_collision_slacks: list[cp.Variable] = []
        foot_constraint_slacks: list[cp.Variable] = []

        # Linear equality
        constraints += [cp.Constant(J_L[:, self.q_a_indices]) @ dqa - lap_var == -lap0_vec]

        # Foot constraints (sticking + foot lock window Z pinning)
        apply_foot_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking
        apply_foot_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        apply_nominal_foot_height = (
            (self.q_a_init_idx < 12)
            and self.anchor_nominal_foot_height
            and q_a_nominal is not None
        )
        if apply_foot_sticking or apply_foot_lock or apply_nominal_foot_height:
            J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)

            if apply_nominal_foot_height:
                q_nominal = np.copy(q)
                q_nominal[self.q_a_indices] = q_a_nominal
                _, p_WF_nominal_dict, _ = self._calc_manipulator_jacobians(
                    q_nominal, links=self.foot_links, obj_frame=False
                )
                tolerance = self.nominal_foot_height_tolerance
                for key, J_WF in J_WF_dict.items():
                    z_delta = p_WF_nominal_dict[key][2] - p_WF_dict[key][2]
                    Jz = J_WF[2, self.q_a_indices]
                    if self.elastic_constraints.enable:
                        slack = cp.Variable(
                            nonneg=True,
                            name=f"foot_height_slack_{frame_idx}_{key}",
                        )
                        foot_constraint_slacks.append(slack)
                        constraints += [
                            Jz @ dqa >= z_delta - tolerance - slack,
                            Jz @ dqa <= z_delta + tolerance + slack,
                        ]
                    else:
                        constraints += [
                            Jz @ dqa >= z_delta - tolerance,
                            Jz @ dqa <= z_delta + tolerance,
                        ]

            # Foot sticking: constrain XY to stay near previous frame position
            if apply_foot_sticking:
                _, p_WF_t_last_dict, _ = self._calc_manipulator_jacobians(
                    q_t_last, links=self.foot_links, obj_frame=False
                )
                left_key = right_key = None
                for key in foot_sticking:
                    if key.lower().startswith("l"):
                        left_key = key
                    elif key.lower().startswith("r"):
                        right_key = key
                if left_key is None or right_key is None:
                    raise ValueError("foot_sticking must include one left* and one right* key")

                for key, J_WF in J_WF_dict.items():
                    apply_left = ("left" in key) and foot_sticking[left_key]
                    apply_right = ("right" in key) and foot_sticking[right_key]
                    if apply_left or apply_right:
                        p_lb = p_WF_t_last_dict[key] - p_WF_dict[key] - self.foot_sticking_tolerance
                        p_ub = p_lb + 2 * self.foot_sticking_tolerance  # symmetric window

                        Jxy = J_WF[:2, self.q_a_indices]  # (2 x nq_act)
                        if self.elastic_constraints.enable:
                            slack = cp.Variable(
                                2,
                                nonneg=True,
                                name=f"foot_xy_slack_{frame_idx}_{key}",
                            )
                            foot_constraint_slacks.append(slack)
                            constraints += [
                                Jxy @ dqa >= p_lb[:2] - slack,
                                Jxy @ dqa <= p_ub[:2] + slack,
                            ]
                        else:
                            constraints += [
                                Jxy @ dqa >= p_lb[:2],
                                Jxy @ dqa <= p_ub[:2],
                            ]

            # Foot lock windows: pin Z to floor within configured frame ranges
            if apply_foot_lock:
                for key, J_WF in J_WF_dict.items():
                    if not self._is_foot_locked_in_window(key, frame_idx):
                        continue

                    z_anchor = self.foot_lock.z_floor
                    z_delta = z_anchor - p_WF_dict[key][2]
                    Jz = J_WF[2, self.q_a_indices]
                    constraints += [
                        Jz @ dqa >= z_delta - self.foot_lock.tolerance,
                        Jz @ dqa <= z_delta + self.foot_lock.tolerance,
                    ]

        # Non-penetration constraints
        Js, phis = self._update_jacobians_and_phis_from_q(q)
        for key, phi in phis.items():
            is_object_pair = self._is_object_collision_pair(key)
            if is_object_pair and not self.activate_obj_non_penetration:
                continue
            Ja_n_full = Js[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            rhs = -phi - self.penetration_tolerance
            if self.elastic_constraints.enable and is_object_pair:
                slack = cp.Variable(
                    nonneg=True,
                    name=(
                        f"object_collision_slack_{frame_idx}_{key[0]}_{key[1]}"
                    ),
                )
                object_collision_slacks.append(slack)
                constraints += [Ja_n @ dqa + slack >= rhs]
            else:
                constraints += [Ja_n @ dqa >= rhs]

        # Self-collision constraints
        Js_sc, phis_sc = self._compute_self_collision_constraints(frame_idx)
        for key, phi in phis_sc.items():
            Ja_n_full = Js_sc[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            # Enforce: new_distance >= tolerance  =>  phi + J @ dqa >= tol
            rhs = self._self_collision_tolerance - phi
            constraints += [Ja_n @ dqa >= rhs]

        # Joint limits constraints (actuated)
        if self.activate_joint_limits:
            constraints += [
                dqa >= (self.q_a_lb - q_a_n_last),
                dqa <= (self.q_a_ub - q_a_n_last),
            ]

        # Step size constraints (Lorentz cone)
        constraints += [cp.SOC(self.step_size, dqa)]

        # Objective
        obj_terms = []

        if object_collision_slacks:
            obj_terms.append(
                self.elastic_constraints.object_collision_weight
                * cp.sum(cp.hstack(object_collision_slacks))
            )
        if foot_constraint_slacks:
            obj_terms.append(
                self.elastic_constraints.foot_kinematics_weight
                * cp.sum(cp.hstack(foot_constraint_slacks))
            )

        obj_terms.append(cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)))

        # nominal tracking for selected indices
        if (w_nominal_tracking > 0) and (q_a_nominal is not None):
            idx = np.array(self.track_nominal_indices, dtype=int)
            if idx.size > 0:
                z = dqa[idx] - (q_a_nominal[idx] - q_a_n_last[idx])
                obj_terms.append(w_nominal_tracking * cp.sum_squares(z))

        # Q_diag cost
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)
        obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last)))

        if (
            self.hand_orientation.enable
            and hand_orientation_weights is not None
            and hand_palm_targets is not None
            and hand_finger_targets is not None
        ):
            for hand_idx, spec in enumerate(self._hand_orientation_specs):
                fade_weight = float(hand_orientation_weights[hand_idx])
                rotation, angular_jacobian, _ = self._calc_body_orientation_linearization(
                    q,
                    int(spec["body_id"]),
                )
                arm_indices = np.asarray(spec["arm_orientation_indices"], dtype=int)

                if fade_weight > 0.0:
                    finger_direction = rotation[:, 0]
                    finger_jacobian = (
                        -self._skew(finger_direction) @ angular_jacobian[:, arm_indices]
                    )
                    finger_linear = finger_direction + finger_jacobian @ dqa[arm_indices]
                    finger_target = np.asarray(hand_finger_targets[hand_idx], dtype=float)
                    obj_terms.append(
                        self.hand_orientation.finger_direction_weight
                        * fade_weight
                        * cp.sum_squares(finger_linear - finger_target)
                    )

                    palm_normal = rotation @ np.asarray(spec["palm_normal"], dtype=float)
                    palm_jacobian = (
                        -self._skew(palm_normal) @ angular_jacobian[:, arm_indices]
                    )
                    palm_linear = palm_normal + palm_jacobian @ dqa[arm_indices]
                    palm_target = np.asarray(hand_palm_targets[hand_idx], dtype=float)
                    obj_terms.append(
                        self.hand_orientation.palm_direction_weight
                        * fade_weight
                        * cp.sum_squares(palm_linear - palm_target)
                    )

        if (
            self.plan_b_palm_contact.enable
            and plan_b_weights is not None
            and plan_b_position_task_weights is not None
            and plan_b_position_weights is not None
            and plan_b_position_targets is not None
            and plan_b_normal_targets is not None
            and plan_b_twist_targets is not None
        ):
            cfg = self.plan_b_palm_contact
            if q_a_nominal is not None:
                posture_indices = self._plan_b_posture_indices
                posture_delta = dqa[posture_indices] - (
                    q_a_nominal[posture_indices]
                    - q_a_n_last[posture_indices]
                )
                obj_terms.append(
                    cp.sum_squares(
                        cp.multiply(
                            np.sqrt(self._plan_b_posture_weights),
                            posture_delta,
                        )
                    )
                )
            for hand_idx, spec in enumerate(self._hand_orientation_specs):
                fade_weight = float(plan_b_weights[hand_idx])
                if fade_weight <= 0.0:
                    continue
                orientation_task_weight = fade_weight**3
                position_task_weight = (
                    float(plan_b_position_task_weights[hand_idx]) ** 3
                )
                approach_weight = float(
                    plan_b_position_weights[hand_idx]
                )
                contact_task_weight = approach_weight**3
                palm_geom_id = int(spec["palm_geom_id"])
                tabletop_geom_id = int(
                    self._plan_b_tabletop_spec["geom_id"]
                )
                contact_pair = None
                for pair in (
                    (palm_geom_id, tabletop_geom_id),
                    (tabletop_geom_id, palm_geom_id),
                ):
                    if pair in phis:
                        contact_pair = pair
                        break
                normal_point_blend = (
                    (1.0 - approach_weight) ** 2
                    if contact_pair is not None
                    else 1.0
                )

                (
                    contact_position,
                    position_jacobian,
                    rotation,
                    angular_jacobian,
                ) = self._calc_body_point_linearization(
                    q,
                    int(spec["body_id"]),
                    np.asarray(spec["palm_contact_point"], dtype=float),
                )
                position_target = np.asarray(
                    plan_b_position_targets[hand_idx],
                    dtype=float,
                )
                normal_target = np.asarray(
                    plan_b_normal_targets[hand_idx],
                    dtype=float,
                )
                position_error = (
                    contact_position
                    + position_jacobian @ dqa
                    - position_target
                )
                tangent_projector = np.eye(3) - np.outer(
                    normal_target,
                    normal_target,
                )
                obj_terms.append(
                    cfg.normal_position_weight
                    * position_task_weight
                    * normal_point_blend
                    * cp.sum_squares(normal_target @ position_error)
                )
                obj_terms.append(
                    cfg.tangent_position_weight
                    * position_task_weight
                    * cp.sum_squares(tangent_projector @ position_error)
                )

                palm_normal_local = np.asarray(spec["palm_normal"], dtype=float)
                palm_normal_world = rotation @ palm_normal_local
                palm_normal_jacobian = (
                    -self._skew(palm_normal_world) @ angular_jacobian
                )
                palm_normal_linear = (
                    palm_normal_world + palm_normal_jacobian @ dqa
                )
                obj_terms.append(
                    cfg.palm_normal_weight
                    * orientation_task_weight
                    * cp.sum_squares(palm_normal_linear - normal_target)
                )

                long_axis_local = np.asarray(spec["palm_basis"], dtype=float)[:, 0]
                long_axis_world = rotation @ long_axis_local
                long_axis_jacobian = (
                    -self._skew(long_axis_world) @ angular_jacobian
                )
                long_axis_linear = long_axis_world + long_axis_jacobian @ dqa
                twist_target = np.asarray(
                    plan_b_twist_targets[hand_idx],
                    dtype=float,
                )
                obj_terms.append(
                    cfg.twist_weight
                    * orientation_task_weight
                    * cp.sum_squares(long_axis_linear - twist_target)
                )
                if contact_pair is not None and contact_task_weight > 0.0:
                    distance_jacobian = Js[contact_pair][self.q_a_indices]
                    distance_linear = (
                        phis[contact_pair]
                        + distance_jacobian @ dqa
                    )
                    obj_terms.append(
                        cfg.contact_distance_weight
                        * contact_task_weight
                        * cp.sum_squares(distance_linear)
                    )

        # Smoothness cost
        dqa_smooth = q_t_last[self.q_a_indices] - q_a_n_last
        if np.isscalar(self.smooth_weight):
            obj_terms.append(self.smooth_weight * cp.sum_squares(dqa - dqa_smooth))
        else:
            Wsmooth = np.asarray(self.smooth_weight, dtype=float)
            if Wsmooth.ndim == 1:
                obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Wsmooth), dqa - dqa_smooth)))
            else:
                # if a full matrix was supplied, fall back to quad_form
                obj_terms.append(cp.quad_form(dqa - dqa_smooth, Wsmooth))

        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

        # -------- Solve with Clarabel --------
        solver_kwargs = {"verbose": verbose}
        problem.solve(solver=cp.CLARABEL, **solver_kwargs)
        if (
            problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            and (init_t or self.plan_b_palm_contact.enable)
            and not self.elastic_constraints.enable
        ):
            constraints = [c for c in constraints if not isinstance(c, cp.constraints.second_order.SOC)]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"CVXPY solve failed: {problem.status}")

        def _slack_max(slacks: list[cp.Variable]) -> float:
            values = [
                np.asarray(slack.value, dtype=float).reshape(-1)
                for slack in slacks
                if slack.value is not None
            ]
            return float(np.max(np.concatenate(values))) if values else 0.0

        self._last_elastic_slack_diagnostics = {
            "object_collision_slack_max_m": _slack_max(
                object_collision_slacks
            ),
            "foot_constraint_slack_max_m": _slack_max(
                foot_constraint_slacks
            ),
        }

        dqa_star = dqa.value
        cost = problem.value

        q_star = np.copy(q)
        q_star[self.q_a_indices] = dqa_star + q_a_n_last
        q_star[3:7] /= np.linalg.norm(q_star[3:7]) + 1e-12

        return q_star, cost

    def _is_object_collision_pair(self, pair: tuple[int, int]) -> bool:
        """Return whether a collision pair contains the interaction object."""
        return pair[0] in self._object_geom_ids or pair[1] in self._object_geom_ids

    def _resolve_object_geom_ids(self) -> frozenset[int]:
        """Resolve interaction-object geoms by MuJoCo body ownership."""
        model = self.robot_model
        object_body_id = -1
        for body_name in (self.object_name, f"{self.object_name}_link"):
            object_body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            if object_body_id >= 0:
                break

        if object_body_id < 0:
            free_joint_ids = np.flatnonzero(
                model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE)
            )
            if free_joint_ids.size >= 2:
                object_body_id = int(model.jnt_bodyid[int(free_joint_ids[-1])])
        if object_body_id < 0:
            return frozenset()

        object_body_ids: set[int] = set()
        for body_id in range(model.nbody):
            ancestor = body_id
            while ancestor > 0:
                if ancestor == object_body_id:
                    object_body_ids.add(body_id)
                    break
                ancestor = int(model.body_parentid[ancestor])
        return frozenset(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in object_body_ids
        )

    def _measure_elastic_constraint_diagnostics(
        self,
        *,
        q: np.ndarray,
        q_t_last: np.ndarray,
        q_a_nominal: np.ndarray | None,
        foot_sticking: dict[str, bool],
    ) -> dict[str, float]:
        """Measure nonlinear constraint violations after an accepted SQP frame."""
        diagnostics = dict(self._last_elastic_slack_diagnostics)
        _, collision_distances = self._update_jacobians_and_phis_from_q(q)
        object_distances = [
            distance
            for pair, distance in collision_distances.items()
            if self._is_object_collision_pair(pair)
        ]
        diagnostics["object_penetration_max_m"] = (
            max(0.0, -float(min(object_distances)))
            if object_distances
            else 0.0
        )

        diagnostics["foot_xy_deviation_max_m"] = 0.0
        diagnostics["foot_height_deviation_max_m"] = 0.0
        if self.q_a_init_idx >= 12:
            return diagnostics

        _, current_positions, _ = self._calc_manipulator_jacobians(
            q,
            links=self.foot_links,
            obj_frame=False,
        )
        if self.activate_foot_sticking:
            _, previous_positions, _ = self._calc_manipulator_jacobians(
                q_t_last,
                links=self.foot_links,
                obj_frame=False,
            )
            active_sides = {
                "left": any(
                    bool(active) and str(key).lower().startswith("l")
                    for key, active in foot_sticking.items()
                ),
                "right": any(
                    bool(active) and str(key).lower().startswith("r")
                    for key, active in foot_sticking.items()
                ),
            }
            xy_deviations = [
                float(
                    np.linalg.norm(
                        current_positions[key][:2]
                        - previous_positions[key][:2]
                    )
                )
                for key in current_positions
                for side in ("left", "right")
                if side in key.lower() and active_sides[side]
            ]
            if xy_deviations:
                diagnostics["foot_xy_deviation_max_m"] = max(xy_deviations)

        if self.anchor_nominal_foot_height and q_a_nominal is not None:
            q_nominal = np.copy(q)
            q_nominal[self.q_a_indices] = q_a_nominal
            _, nominal_positions, _ = self._calc_manipulator_jacobians(
                q_nominal,
                links=self.foot_links,
                obj_frame=False,
            )
            diagnostics["foot_height_deviation_max_m"] = max(
                abs(
                    float(current_positions[key][2])
                    - float(nominal_positions[key][2])
                )
                for key in current_positions
            )
        return diagnostics

    def _is_foot_locked_in_window(self, foot_link_key: str, frame_idx: int) -> bool:
        """Check whether a foot link is locked by configured frame windows."""
        key_lower = foot_link_key.lower()
        side = None
        if "left" in key_lower:
            side = "left"
        elif "right" in key_lower:
            side = "right"
        if side is None:
            return False

        return any(start <= frame_idx <= end for start, end in self._foot_lock_windows.get(side, ()))

    def _compute_self_collision_constraints(self, frame_idx: int):
        """Compute Jacobians and distances for self-collision body pairs.

        Assumes ``mj_forward`` has already been called with the current q
        (done by ``_update_jacobians_and_phis_from_q`` which runs first).

        Returns:
            Js: dict mapping (geom_a, geom_b) -> relative Jacobian (1 x nq)
            phis: dict mapping (geom_a, geom_b) -> signed distance
        """
        if not self._self_collision_enabled:
            return {}, {}

        # Check frame windows
        if self._self_collision_windows is not None:
            if not any(start <= frame_idx <= end for start, end in self._self_collision_windows):
                return {}, {}

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        if not hasattr(self, "_geom_names"):
            raise RuntimeError(
                "[SelfCollision] _geom_names not initialized. Please run _prefilter_pairs_with_mj_collision first."
            )

        _first_iter = self._sc_last_vis_frame != frame_idx
        if _first_iter:
            self._sc_last_vis_frame = frame_idx

        for geom_a, geom_b in self._self_collision_geom_pairs:
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, geom_a, geom_b, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(geom_a),
                    m.geom(geom_b),
                    self._geom_names[geom_a],
                    self._geom_names[geom_b],
                    fromto,
                    dist,
                )
                key = ("self", geom_a, geom_b)
                Js[key] = J_rel
                phis[key] = float(dist)

        if _first_iter and self.visualize:
            self._draw_self_collision_geoms()

        return Js, phis

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        init_t: bool = False,
        n_iter: int = 10,
        frame_idx: int = 0,
        hand_orientation_weights: np.ndarray | None = None,
        hand_palm_targets: np.ndarray | None = None,
        hand_finger_targets: np.ndarray | None = None,
        plan_b_weights: np.ndarray | None = None,
        plan_b_position_task_weights: np.ndarray | None = None,
        plan_b_position_weights: np.ndarray | None = None,
        plan_b_position_targets: np.ndarray | None = None,
        plan_b_normal_targets: np.ndarray | None = None,
        plan_b_twist_targets: np.ndarray | None = None,
    ):
        """Iterate the solver for multiple iterations."""
        last_cost = np.inf
        object_collision_feasible = True
        minimum_object_collision: tuple[tuple[int, int], float] | None = None
        for _ in range(n_iter):
            q_before_iteration = np.copy(q_n)
            q_a_n_last = q_n[self.q_a_indices]
            q_n, cost = self.solve_single_iteration(
                q_locked=q_locked,
                q_a_n_last=q_a_n_last,
                q_t_last=q_t_last,
                target_laplacian=target_laplacian,
                adj_list=adj_list,
                obj_pts_local=obj_pts_local,
                foot_sticking=foot_sticking,
                q_a_nominal=q_a_nominal,
                w_nominal_tracking=w_nominal_tracking,
                init_t=init_t,
                frame_idx=frame_idx,
                hand_orientation_weights=hand_orientation_weights,
                hand_palm_targets=hand_palm_targets,
                hand_finger_targets=hand_finger_targets,
                plan_b_weights=plan_b_weights,
                plan_b_position_task_weights=plan_b_position_task_weights,
                plan_b_position_weights=plan_b_position_weights,
                plan_b_position_targets=plan_b_position_targets,
                plan_b_normal_targets=plan_b_normal_targets,
                plan_b_twist_targets=plan_b_twist_targets,
            )
            object_collision_feasible = True
            if self.plan_b_palm_contact.enable:
                _, collision_distances = (
                    self._update_jacobians_and_phis_from_q(q_n)
                )
                object_distances = [
                    distance
                    for (geom_a, geom_b), distance in collision_distances.items()
                    if (
                        self.object_name in self._geom_names[geom_a]
                        or self.object_name in self._geom_names[geom_b]
                    )
                ]
                if object_distances:
                    minimum_object_collision = min(
                        (
                            (pair, distance)
                            for pair, distance in collision_distances.items()
                            if (
                                self.object_name in self._geom_names[pair[0]]
                                or self.object_name in self._geom_names[pair[1]]
                            )
                        ),
                        key=lambda item: item[1],
                    )
                    object_collision_feasible = (
                        minimum_object_collision[1]
                        >= -self.penetration_tolerance
                        - self.plan_b_palm_contact.collision_validation_tolerance
                    )
            cost_converged = np.isclose(cost, last_cost)
            if self.elastic_constraints.enable:
                step_converged = (
                    np.linalg.norm(
                        q_n[self.q_a_indices]
                        - q_before_iteration[self.q_a_indices]
                    )
                    <= 1.0e-6
                )
            else:
                step_converged = True
            if cost_converged and step_converged and object_collision_feasible:
                break
            last_cost = cost
        if self.plan_b_palm_contact.enable and not object_collision_feasible:
            pair_text = "unknown pair"
            distance_text = "unknown distance"
            task_error_text = ""
            if minimum_object_collision is not None:
                (geom_a, geom_b), distance = minimum_object_collision
                pair_text = (
                    f"{self._geom_names[geom_a]} / {self._geom_names[geom_b]}"
                )
                distance_text = f"{distance:.9f}m"
            if (
                plan_b_position_targets is not None
                and plan_b_normal_targets is not None
                and plan_b_twist_targets is not None
            ):
                task_errors = []
                for hand_idx, spec in enumerate(self._hand_orientation_specs):
                    (
                        contact_position,
                        _,
                        rotation,
                        _,
                    ) = self._calc_body_point_linearization(
                        q_n,
                        int(spec["body_id"]),
                        np.asarray(spec["palm_contact_point"], dtype=float),
                    )
                    palm_normal = rotation @ np.asarray(
                        spec["palm_normal"],
                        dtype=float,
                    )
                    long_axis = rotation @ np.asarray(
                        spec["palm_basis"],
                        dtype=float,
                    )[:, 0]
                    position_error_vector = (
                        contact_position
                        - np.asarray(plan_b_position_targets[hand_idx], dtype=float)
                    )
                    normal_target = np.asarray(
                        plan_b_normal_targets[hand_idx],
                        dtype=float,
                    )
                    signed_normal_error = float(
                        normal_target @ position_error_vector
                    )
                    tangent_error = np.linalg.norm(
                        position_error_vector
                        - signed_normal_error * normal_target
                    )
                    normal_error = np.degrees(
                        np.arccos(
                            np.clip(
                                palm_normal
                                @ np.asarray(
                                    plan_b_normal_targets[hand_idx],
                                    dtype=float,
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    twist_error = np.degrees(
                        np.arccos(
                            np.clip(
                                long_axis
                                @ np.asarray(
                                    plan_b_twist_targets[hand_idx],
                                    dtype=float,
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    palm_geom_id = mujoco.mj_name2id(
                        self.robot_model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        f"{spec['side']}_rubber_hand_link",
                    )
                    palm_mesh_id = int(
                        self.robot_model.geom_dataid[palm_geom_id]
                    )
                    vertex_start = int(
                        self.robot_model.mesh_vertadr[palm_mesh_id]
                    )
                    vertex_count = int(
                        self.robot_model.mesh_vertnum[palm_mesh_id]
                    )
                    vertices_geom = np.asarray(
                        self.robot_model.mesh_vert[
                            vertex_start : vertex_start + vertex_count
                        ],
                        dtype=float,
                    )
                    palm_geom_rotation = np.asarray(
                        self.robot_data.geom_xmat[palm_geom_id],
                        dtype=float,
                    ).reshape(3, 3)
                    vertices_world = (
                        vertices_geom @ palm_geom_rotation.T
                        + self.robot_data.geom_xpos[palm_geom_id]
                    )
                    tabletop_geom_id = int(
                        self._plan_b_tabletop_spec["geom_id"]
                    )
                    tabletop_rotation = np.asarray(
                        self.robot_data.geom_xmat[tabletop_geom_id],
                        dtype=float,
                    ).reshape(3, 3)
                    vertices_tabletop = (
                        vertices_world
                        - self.robot_data.geom_xpos[tabletop_geom_id]
                    ) @ tabletop_rotation
                    contact_tabletop = (
                        contact_position
                        - self.robot_data.geom_xpos[tabletop_geom_id]
                    ) @ tabletop_rotation
                    normal_tabletop = (
                        tabletop_rotation.T @ normal_target
                    )
                    support_delta = float(
                        np.max(vertices_world @ normal_target)
                        - contact_position @ normal_target
                    )
                    bounds_min = vertices_tabletop.min(axis=0)
                    bounds_max = vertices_tabletop.max(axis=0)
                    task_errors.append(
                        f"{spec['side']}: normal_position="
                        f"{signed_normal_error:+.6f}m, "
                        f"tangent_position={tangent_error:.6f}m, "
                        f"normal={normal_error:.3f}deg, "
                        f"twist={twist_error:.3f}deg, "
                        f"tabletop_local_bounds="
                        f"{np.array2string(bounds_min, precision=5)}.."
                        f"{np.array2string(bounds_max, precision=5)}, "
                        f"contact_local="
                        f"{np.array2string(contact_tabletop, precision=5)}, "
                        f"normal_local="
                        f"{np.array2string(normal_tabletop, precision=5)}, "
                        f"support_delta={support_delta:+.6f}m"
                    )
                task_error_text = "; task errors: " + "; ".join(task_errors)
            raise RuntimeError(
                "Plan B SQP did not reach a collision-free configuration "
                f"within {n_iter} iterations: {pair_text}, {distance_text}"
                f"{task_error_text}"
            )
        return q_n, cost

    @staticmethod
    def _skew(vector: np.ndarray) -> np.ndarray:
        """Return the matrix whose product with x is vector cross x."""
        x, y, z = vector
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    def _calc_body_orientation_linearization(
        self,
        q: np.ndarray,
        body_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return body rotation, angular Jacobian, and position in world coordinates."""
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        position = np.array(self.robot_data.xpos[body_id], dtype=float, copy=True)
        rotation = np.array(self.robot_data.xmat[body_id], dtype=float, copy=True).reshape(3, 3)
        jacobian_pos = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        jacobian_rot = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(
            self.robot_model,
            self.robot_data,
            jacobian_pos,
            jacobian_rot,
            position.reshape(3, 1),
            body_id,
        )
        qdot_to_qvel = self._build_transform_qdot_to_qvel_fast()
        angular_jacobian = (jacobian_rot @ qdot_to_qvel)[:, self.q_a_indices]
        return rotation, angular_jacobian, position

    def _calc_body_point_linearization(
        self,
        q: np.ndarray,
        body_id: int,
        point_body: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a body-fixed point and its full-body position/orientation Jacobians."""
        rotation, angular_jacobian, body_position = (
            self._calc_body_orientation_linearization(q, body_id)
        )
        point_body = np.asarray(point_body, dtype=float)
        point_world = body_position + rotation @ point_body
        point_jacobian = self._calc_contact_jacobian_from_point(
            body_id,
            point_body,
        )[:, self.q_a_indices]
        return (
            np.asarray(point_world, dtype=float),
            np.asarray(point_jacobian, dtype=float),
            rotation,
            angular_jacobian,
        )

    def _draw_self_collision_geoms(self):
        """Draw collision cylinders for self-collision geom pairs in viser."""
        if not hasattr(self, "server") or not self._self_collision_enabled:
            return
        m, d = self.robot_model, self.robot_data
        seen_geoms: set[int] = set()
        colors = [(255, 80, 80), (80, 80, 255)]  # red for first body, blue for second
        for geom_a, geom_b in self._self_collision_geom_pairs:
            for idx, gid in enumerate([geom_a, geom_b]):
                if gid in seen_geoms:
                    continue
                seen_geoms.add(gid)
                gtype = int(m.geom_type[gid])
                if gtype not in (3, 5):  # 3 = capsule, 5 = cylinder
                    continue
                radius = float(m.geom_size[gid][0])
                half_len = float(m.geom_size[gid][1])
                cyl = trimesh.creation.capsule(radius=radius, height=2 * half_len, count=[16, 16])
                # World transform from MuJoCo data
                pos = d.geom_xpos[gid]
                rot_mat = d.geom_xmat[gid].reshape(3, 3)
                transform = np.eye(4)
                transform[:3, :3] = rot_mat
                transform[:3, 3] = pos
                cyl.apply_transform(transform)
                body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""
                self.server.scene.add_mesh_simple(
                    f"/world/sc_geom/{body_name}_g{gid}",
                    vertices=cyl.vertices.astype(np.float32),
                    faces=cyl.faces.astype(np.int32),
                    color=colors[idx % 2],
                    opacity=0.35,
                )

    def draw_q(self, q: np.ndarray):
        """Draw a single robot configuration."""
        # Update robot joint configurations
        robot_joint_positions = q[7 : 7 + self.task_constants.ROBOT_DOF]
        self.viser_robot.update_cfg(robot_joint_positions)

        # Update robot base pose using set_transform
        robot_quat = q[3:7]  # Base orientation
        robot_pos = q[:3]  # Base position

        # Update robot base frame
        self.robot_base.position = robot_pos
        self.robot_base.wxyz = robot_quat  # Assuming quaternion is in wxyz order

        # Update object pose if it exists
        if hasattr(self, "viser_object") and self.viser_object is not None:
            if self.has_dynamic_object:
                object_quat = q[-4:]
                object_pos = q[-7:-4]
            else:
                object_quat = np.asarray([1, 0, 0, 0])
                object_pos = np.zeros(3)

            # Update object base frame
            self.object_base.position = object_pos
            self.object_base.wxyz = object_quat  # Assuming quaternion is in wxyz order

    def draw_keypoints(self, p, name="keypoint", rgba=(0, 0, 1, 1)):
        """Draw keypoints in visualization."""
        if not hasattr(self, "server"):
            return None

        # Create a sphere mesh using trimesh
        sphere = trimesh.primitives.Sphere(radius=0.02)
        vertices = sphere.vertices
        faces = sphere.faces

        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])

        kpts_handle_list = []

        # Draw keypoints
        if len(p.shape) == 1:
            # Single point
            kpts_handle = self.server.scene.add_mesh_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                position=p,
                color=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)
        elif len(p.shape) == 2:
            # Multiple points
            kpts_handle = self.server.scene.add_batched_meshes_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                batched_positions=p,
                batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (p.shape[0], 1)),
                batched_colors=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)

        return kpts_handle_list

    def visualize_motion(
        self,
        human_joint_motions,
        obj_pts_demo,
        obj_pts,
        retargeted_motions,
        tetrahedra,
        dt=1 / 30,
        visualize_tetrahedra=False,
    ):
        for i in range(len(human_joint_motions)):
            object_pts_demo = obj_pts_demo[i]
            object_pts = obj_pts[i]
            self.draw_keypoints(human_joint_motions[i, self.smplh_mapped_joint_indices], name="human")
            self.draw_keypoints(object_pts_demo, name="object_demo", rgba=(1, 0, 0, 1))
            self.draw_keypoints(object_pts, name="object", rgba=(0, 1, 0, 1))
            self.draw_q(retargeted_motions[i])
            robot_link_positions = self._get_robot_link_positions(
                retargeted_motions[i], self.laplacian_match_links.values()
            )
            self.draw_keypoints(robot_link_positions, name="robot", rgba=(0, 1, 0, 1))
            input()
            if visualize_tetrahedra:
                self.visualize_tetrahedra(
                    np.vstack(
                        [
                            human_joint_motions[i, self.smplh_mapped_joint_indices],
                            object_pts_demo,
                        ]
                    ),
                    tetrahedra[i],
                    name="human_tetrahedra",
                )
                self.visualize_tetrahedra(
                    np.vstack([robot_link_positions, object_pts]),
                    tetrahedra[i],
                    name="robot_tetrahedra",
                    rgba=(0, 1, 1, 1),
                )
            else:
                time.sleep(dt)

    def visualize_tetrahedra(self, vertices, tetrahedra, name="tetrahedra", color=(0, 0, 0, 1)):
        # Convert color to 0-255 range
        color_255 = np.array(color[:3]) * 255

        # Prepare points and colors for all edges
        points = []
        colors = []

        for tet in tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = tet[i], tet[j]
                    points.extend([vertices[u], vertices[v]])
                    colors.extend([color_255, color_255])

        # Convert to numpy arrays
        points = np.array(points)
        colors = np.array(colors)

        # Add line segments for all edges at once
        self.server.scene.add_line_segments(
            f"/{name}",
            points=points,
            colors=colors,
            line_width=0.01,
        )

    def _compute_jacobian_for_contact_relative(self, geom1, geom2, geom1_name, geom2_name, fromto, dist):
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            nhat_BA_W = np.sign(dist) * (v / norm_v)
        # Degenerate: points coincide. Heuristics fallback.
        # If one side is a plane/ground, use its known normal.
        elif "ground" in geom2_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            nhat_BA_W = np.array([0.0, 0.0, 0.0])

        J_bodyA = self._calc_contact_jacobian_from_point(geom1.bodyid, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(geom2.bodyid, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        return nhat_BA_W @ Jc

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold

        # Run collision. This runs broad→narrow and fills d.contact.
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs that involve at least one masked geom
        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore margins to keep physics untouched
        m.geom_margin[:] = self._saved_margins

        return candidates

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        self.robot_data.qpos[:] = q

        mujoco.mj_forward(self.robot_model, self.robot_data)  # kinematics & AABBs valid

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # 1) Fast prefilter via mj_collision with temporary margins
        candidates = self._prefilter_pairs_with_mj_collision(threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        # 2) Precise distance only on candidates (early-exit at threshold)
        contype, conaff = m.geom_contype, m.geom_conaffinity

        def masks_ok(g1, g2):
            if contype[g1] == 0 and conaff[g1] == 0:
                return False
            if contype[g2] == 0 and conaff[g2] == 0:
                return False
            object_g1 = g1 in self._object_geom_ids
            object_g2 = g2 in self._object_geom_ids
            ground_g1 = "ground" in self._geom_names[g1]
            ground_g2 = "ground" in self._geom_names[g2]
            if object_g1 and ground_g2:
                return False
            if ground_g1 and object_g2:
                return False
            return object_g1 or object_g2 or ground_g1 or ground_g2

        for g1, g2 in candidates:
            # Optional: keep your own filters here (e.g., skip object-ground, only keep interaction with object/ground)
            if not masks_ok(g1, g2):
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(g1), m.geom(g2), self._geom_names[g1], self._geom_names[g2], fromto, dist
                )
                Js[(g1, g2)] = J_rel
                phis[(g1, g2)] = float(dist)

                # For debug
                # self.draw_mesh_pair_with_contact(self.robot_model, self.robot_data, g1, g2,   \
                #     self._geom_names[g1], self._geom_names[g2], fromto=fromto)

        return Js, phis

    def _world_to_body_frame(self, p_w: np.ndarray, body_idx: int) -> np.ndarray:
        """Transform point from world frame to body frame."""
        p_w = np.asarray(p_w).reshape(3)
        body_pos = self.robot_data.xpos[body_idx].reshape(3)
        body_mat = self.robot_data.xmat[body_idx].reshape(3, 3)
        return body_mat.T @ (p_w - body_pos)

    def _get_geometry_name(self, geom_id: int) -> str:
        """Get geometry name from ID."""
        return mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
        """
        Return T(q) (nv x nq) such that v = T(q) @ qdot.
        - Free root: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, ωx,ωy,ωz]
        where ω and v are WORLD-expressed in MuJoCo.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses BODY-omega mapping (for debugging).
        """
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        # ---- root free joint (assumed joint 0) ----
        j0 = 0
        assert self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE
        qadr = self.robot_model.jnt_qposadr[j0]  # 0
        dadr = self.robot_model.jnt_dofadr[j0]  # 0

        # Linear block: v_lin = xyz_dot
        T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)

        # Angular block: ω_* = 2 * E_*(q) * quat_dot
        w, x, y, z = self.robot_data.qpos[qadr + 3 : qadr + 7]

        def get_e_world(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, qz, -qy],
                    [-qy, -qz, qw, qx],
                    [-qz, qy, -qx, qw],
                ]
            )

        def get_e_body(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, -qz, qy],
                    [-qy, qz, qw, -qx],
                    [-qz, -qy, qx, qw],
                ]
            )

        E_fn = get_e_world if use_world_omega else get_e_body

        # ---- FREE joint #1 (human/root): use model addresses, but this should be the first joint ----
        j_free1 = 0
        assert self.robot_model.jnt_type[j_free1] == mujoco.mjtJoint.mjJNT_FREE
        qadr1 = int(self.robot_model.jnt_qposadr[j_free1])  # expect 0
        dadr1 = int(self.robot_model.jnt_dofadr[j_free1])  # start of its 6 qvel dofs

        qw, qx, qy, qz = self.robot_data.qpos[qadr1 + 3 : qadr1 + 7]
        E1 = 2.0 * E_fn(qw, qx, qy, qz)
        # linear-first: v_W = rdot, ω_W = 2E(q) * quat_dot
        T[dadr1 + 0 : dadr1 + 3, qadr1 + 0 : qadr1 + 3] = np.eye(3)  # v block
        T[dadr1 + 3 : dadr1 + 6, qadr1 + 3 : qadr1 + 7] = E1  # ω block

        if self.has_dynamic_object:
            # ---- FREE joint #2 (object): assume it's the last FREE joint; fill its 6x7 block ----
            # Find it by type (safer than hardcoding tail indices)
            free_joints = [
                j for j in range(self.robot_model.njnt) if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ]
            assert len(free_joints) >= 2, "Expected two FREE joints (human + object)."
            j_free2 = free_joints[1]  # second FREE joint
            qadr2 = int(self.robot_model.jnt_qposadr[j_free2])  # expect nq-7
            dadr2 = int(self.robot_model.jnt_dofadr[j_free2])  # its 6 qvel dofs (often at nv-6)

            qw, qx, qy, qz = self.robot_data.qpos[qadr2 + 3 : qadr2 + 7]
            E2 = 2.0 * E_fn(qw, qx, qy, qz)
            T[dadr2 + 0 : dadr2 + 3, qadr2 + 0 : qadr2 + 3] = np.eye(3)  # v block
            T[dadr2 + 3 : dadr2 + 6, qadr2 + 3 : qadr2 + 7] = E2  # ω block

        # ---- remaining hinge/slide joints: v = qdot ----
        for j in range(1, self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint block not implemented.")

        return T

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray, input_world=False):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)
        """

        p_body = np.asarray(p_body, dtype=float).reshape(3)

        # 1) Make sure kinematics are current once
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # 2) World point (3,1) for mj_jac
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        # 3) J_v: translational Jacobian wrt generalized velocities (3 x nv)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))  # Jp = J_v

        T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | None = None,
    ):
        """Compute position-based Jacobians using MuJoCo."""
        J_XC_dict = {}
        p_XC_dict = {}

        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
                obj_rot_inv = obj_rot.T
            else:
                obj_rot = Rotation.from_quat([0, 0, 0, 1]).as_matrix()
                obj_rot_inv = obj_rot.T
                obj_pos = np.zeros(3)

        q_mujoco = q.copy()
        self.robot_data.qpos[:] = q_mujoco

        mujoco.mj_forward(self.robot_model, self.robot_data)

        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

            if point_offsets is not None:
                pC_B = point_offsets
            else:
                pC_B = np.zeros(3)

            J = self._calc_contact_jacobian_from_point(body_id, pC_B)
            pos_world = self.robot_data.xpos[body_id]

            if obj_frame:
                p_XC = obj_rot_inv @ (pos_world - obj_pos)
                J_XC = obj_rot_inv @ J
            else:
                p_XC = pos_world
                J_XC = J

            # Store reduced Jacobian and position with hard copies to avoid aliasing
            J_XC_dict[name] = np.array(J_XC[:, self.q_a_indices], dtype=float, copy=True)  # FIX (copy)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)

        P_WO = {"position": obj_pos, "rotation": obj_rot} if obj_frame else None

        return J_XC_dict, p_XC_dict, P_WO

    def _get_robot_link_positions(self, q, link_names):
        """Get robot link positions for given configuration using Mujoco."""
        mujoco_q = q.copy()

        # Set the configuration
        if mujoco_q.shape != self.robot_data.qpos.shape:
            self.robot_data.qpos = mujoco_q[:-7]  # Exclude object information from q
        else:
            self.robot_data.qpos = mujoco_q
        # Forward kinematics to update all positions
        mujoco.mj_forward(self.robot_model, self.robot_data)

        robot_link_positions = []

        for link_name in link_names:
            # Get body ID from name
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                raise ValueError(f"Body {link_name} not found in Mujoco model")

            # Get position in world frame
            # xpos gives us the position of the body's center of mass in world coordinates
            pos = self.robot_data.xpos[body_id].copy()
            robot_link_positions.append(pos)

        return np.array(robot_link_positions)
