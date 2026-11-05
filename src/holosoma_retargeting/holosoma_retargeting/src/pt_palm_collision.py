"""Collision-aware demonstrated-palm refinement, independent of data source.

The existing retargeter supplies palm calibration, FK and signed-distance
Jacobians. Only named arm hinge joints are variables here: no floating-base
quaternion optimization, no table motion change, no hand-designed contact pose.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import cvxpy as cp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

if TYPE_CHECKING:
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter


def _arm_geom_ids(retargeter: InteractionMeshRetargeter, qpos_indices: np.ndarray) -> set[int]:
    model = retargeter.robot_model
    arm_bodies = {
        int(model.jnt_bodyid[j])
        for j in range(model.njnt)
        if int(model.jnt_qposadr[j]) in qpos_indices
    }
    movable = set()
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body > 0:
            if body in arm_bodies:
                movable.add(geom)
                break
            body = int(model.body_parentid[body])
    return movable


def _contact_manifold_rows(retargeter, pairs, indices):
    """Keep every contact witness, not just one closest point per geom pair.

    A flat palm can touch a tabletop at multiple points. Constraining only one
    witness can let rotation push another witness inside the object. MuJoCo's
    contact normal is directed from contact.geom1 to contact.geom2; retain that
    order even though the broad-phase pair keys are sorted.
    """
    pair_set = {tuple(sorted(pair)) for pair in pairs}
    contacts = []
    for i in range(retargeter.robot_data.ncon):
        contact = retargeter.robot_data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if tuple(sorted((g1, g2))) in pair_set:
            contacts.append((g1, g2, float(contact.dist), contact.pos.copy(), contact.frame[:3].copy()))
    rows = []
    for g1, g2, distance, point, normal in contacts:
        j1 = retargeter._calc_contact_jacobian_from_point(
            int(retargeter.robot_model.geom_bodyid[g1]), point, input_world=True
        )
        j2 = retargeter._calc_contact_jacobian_from_point(
            int(retargeter.robot_model.geom_bodyid[g2]), point, input_world=True
        )
        rows.append((normal @ (j2 - j1)[:, indices], distance))
    return rows


def refine_pt_palms_with_collision(
    retargeter: InteractionMeshRetargeter,
    qpos: np.ndarray,
    palm_orientations: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return refined qpos and honest residuals; never silently relax collisions.

    Existing frozen-body collisions are measured, not 'fixed' by moving legs.
    Final nonlinear checks cover all movable arm/object and arm/ground pairs
    reported by the same collision backend as the baseline. This is kinematic
    feasibility, not a guarantee of dynamically executable contact forces.
    """
    cfg = retargeter.pt_palm_collision
    if not cfg.enable:
        raise ValueError("PT palm collision refinement is disabled")
    if not retargeter.activate_obj_non_penetration or not retargeter.activate_joint_limits:
        raise ValueError("PT palm collision refinement requires collisions and joint limits")
    if retargeter.elastic_constraints.enable:
        raise ValueError("PT palm collision refinement does not use elastic collision slack")
    for name in ("orientation_weight", "hand_position_weight", "arm_prior_weight"):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("correction_temporal_weight", "clearance", "validation_tolerance"):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if isinstance(cfg.max_iterations, bool) or not isinstance(cfg.max_iterations, int) or cfg.max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(retargeter.step_size) or retargeter.step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    if cfg.clearance >= retargeter.collision_detection_threshold:
        raise ValueError("clearance must be smaller than the collision detection threshold")

    base = np.asarray(qpos, dtype=float)
    if base.ndim != 2 or base.shape[1] != retargeter.nq or not len(base) or not np.isfinite(base).all():
        raise ValueError(f"Expected finite nonempty qpos (T, {retargeter.nq})")
    palms = np.asarray(palm_orientations, dtype=float)
    if palms.shape != (len(base), 2, 3, 3) or not np.isfinite(palms).all():
        raise ValueError("Expected finite palm rotations (T, 2, 3, 3)")
    if not np.allclose(palms.swapaxes(-1, -2) @ palms, np.eye(3), atol=1e-6) or not np.allclose(
        np.linalg.det(palms), 1.0, atol=1e-6
    ):
        raise ValueError("Palm targets must be proper rotation matrices")
    specs = retargeter._hand_orientation_specs
    if len(specs) != 2:
        raise ValueError("Expected left and right hand specifications")
    indices = np.concatenate([spec["pt_full_arm_qpos_indices"] for spec in specs]).astype(int)
    if indices.shape != (14,) or len(np.unique(indices)) != 14:
        raise ValueError("Expected fourteen distinct named arm hinge joints")
    lower = np.concatenate([spec["pt_full_arm_lower_limits"] for spec in specs])
    upper = np.concatenate([spec["pt_full_arm_upper_limits"] for spec in specs])
    # Respect the effective baseline shoulder/elbow limits; A.1 uses physical
    # URDF wrist limits instead of old task-specific fixed-wrist overrides.
    local = np.array([int(np.flatnonzero(retargeter.q_a_indices == i)[0]) for i in indices])
    proximal = np.array([0, 1, 2, 3, 7, 8, 9, 10])
    lower[proximal] = np.maximum(lower[proximal], retargeter.q_a_lb[local[proximal]])
    upper[proximal] = np.minimum(upper[proximal], retargeter.q_a_ub[local[proximal]])
    targets = retargeter._map_pt_palm_orientations_to_robot_links(palms, len(base))
    movable = _arm_geom_ids(retargeter, indices)
    model = retargeter.robot_model
    object_collision_geoms = [
        g for g in retargeter._object_geom_ids
        if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
    ]
    if not object_collision_geoms:
        raise ValueError("No active object collision geometry")
    result = base.copy()
    corrections = np.zeros((len(base), 14))
    errors = np.zeros((len(base), 2))
    position_drift = np.zeros_like(errors)
    min_arm_distance = np.full(len(base), retargeter.collision_detection_threshold)
    frozen_penetration = np.zeros(len(base))
    iteration_counts = np.zeros(len(base), dtype=int)
    converged = np.zeros(len(base), dtype=bool)
    hand_distance = np.full_like(errors, retargeter.collision_detection_threshold)

    def collision_state(q):
        jacobians, distances = retargeter._update_jacobians_and_phis_from_q(q)
        pairs = [p for p in distances if p[0] in movable or p[1] in movable]
        return jacobians, distances, pairs

    for frame in tqdm(range(len(base)), desc="PT palm + collision"):
        position_targets = []
        for spec in specs:
            _, _, position = retargeter._calc_body_orientation_linearization(base[frame], int(spec["body_id"]))
            position_targets.append(position)
        previous_correction = corrections[frame - 1] if frame else np.zeros(14)
        current = base[frame].copy()
        current[indices] = np.clip(base[frame, indices] + previous_correction, lower, upper)
        # Reusing last frame's correction is only an initial guess. A moving
        # table can make it less feasible than this frame's original baseline.
        baseline_start = base[frame].copy()
        baseline_start[indices] = np.clip(baseline_start[indices], lower, upper)
        _, warm_distances, warm_pairs = collision_state(current)
        _, base_distances, base_pairs = collision_state(baseline_start)
        warm_violation = max((max(0.0, cfg.clearance - warm_distances[p]) for p in warm_pairs), default=0.0)
        base_violation = max((max(0.0, cfg.clearance - base_distances[p]) for p in base_pairs), default=0.0)
        if warm_violation > max(cfg.validation_tolerance, base_violation):
            current = baseline_start

        def objective(q):
            correction = q[indices] - base[frame, indices]
            value = cfg.arm_prior_weight * np.dot(correction, correction)
            if frame:
                delta = correction - previous_correction
                value += cfg.correction_temporal_weight * np.dot(delta, delta)
            for hand, spec in enumerate(specs):
                rotation, _, point = retargeter._calc_body_orientation_linearization(q, int(spec["body_id"]))
                error = Rotation.from_matrix(targets[frame, hand] @ rotation.T).as_rotvec()
                offset = point - position_targets[hand]
                value += cfg.orientation_weight * np.dot(error, error)
                value += cfg.hand_position_weight * np.dot(offset, offset)
            return value

        for iteration in range(cfg.max_iterations):
            step = cp.Variable(14)
            terms = [cfg.arm_prior_weight * cp.sum_squares(current[indices] + step - base[frame, indices])]
            if frame:
                terms.append(cfg.correction_temporal_weight * cp.sum_squares(
                    current[indices] + step - base[frame, indices] - previous_correction
                ))
            for hand, spec in enumerate(specs):
                point, jp, rotation, jr = retargeter._calc_body_point_linearization(
                    current, int(spec["body_id"]), np.zeros(3)
                )
                # Rotation error and angular Jacobian are both world-frame.
                error = Rotation.from_matrix(targets[frame, hand] @ rotation.T).as_rotvec()
                terms.append(cfg.orientation_weight * cp.sum_squares(jr[:, local] @ step - error))
                terms.append(cfg.hand_position_weight * cp.sum_squares(
                    point + jp[:, local] @ step - position_targets[hand]
                ))
            jacobians, distances, pairs = collision_state(current)
            manifold_rows = _contact_manifold_rows(retargeter, pairs, indices)
            constraints = [
                current[indices] + step >= lower,
                current[indices] + step <= upper,
                cp.norm(step, 2) <= retargeter.step_size,
            ]
            for pair in pairs:
                constraints.append(jacobians[pair][indices] @ step >= cfg.clearance - distances[pair])
            for jacobian, distance in manifold_rows:
                constraints.append(jacobian @ step >= cfg.clearance - distance)
            problem = cp.Problem(cp.Minimize(sum(terms)), constraints)
            problem.solve(solver=cp.CLARABEL)
            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or step.value is None:
                # A failed local step is not evidence that an already checked
                # feasible pose is invalid. Keep it as an explicitly
                # non-converged preview rather than applying a relaxed step.
                feasible = all(distances[p] >= cfg.clearance - cfg.validation_tolerance for p in pairs)
                if feasible:
                    final_distances, final_pairs = distances, pairs
                    iteration_counts[frame] = iteration + 1
                    break
                worst = min(pairs, key=lambda p: distances[p]) if pairs else None
                raise RuntimeError(
                    f"PT palm collision frame {frame}, SQP {iteration}: {problem.status}; "
                    f"worst pair {worst}, distance {distances.get(worst)}, "
                    f"Jacobian norm {np.linalg.norm(jacobians[worst][indices]) if worst else None}; "
                    "no constraints relaxed"
                )
            if not np.isfinite(step.value).all():
                raise RuntimeError(f"PT palm collision frame {frame}: non-finite solver step")
            old_violation = max((max(0.0, cfg.clearance - distances[p]) for p in pairs), default=0.0)
            old_objective = objective(current)
            old_merit = old_objective + 1e5 * sum(max(0.0, cfg.clearance - distances[p]) for p in pairs)
            accepted_step = None
            # Recheck *nonlinear* geometry and new collision pairs after every
            # proposed step. This scales an IK solver step, not robot actions.
            for backtrack in range(18):
                alpha = 0.5 ** backtrack
                candidate = current.copy()
                candidate[indices] += alpha * np.asarray(step.value)
                _, candidate_distances, candidate_pairs = collision_state(candidate)
                violation = max((max(0.0, cfg.clearance - candidate_distances[p]) for p in candidate_pairs), default=0.0)
                merit = objective(candidate) + 1e5 * sum(
                    max(0.0, cfg.clearance - candidate_distances[p]) for p in candidate_pairs
                )
                restoring_feasibility = old_violation > cfg.validation_tolerance
                if (violation <= max(cfg.validation_tolerance, old_violation * (1.0 - 0.1 * alpha))
                        and (restoring_feasibility or merit <= old_merit + 1e-8)):
                    current = candidate
                    accepted_step = alpha * np.asarray(step.value)
                    break
            _, final_distances, final_pairs = collision_state(current)
            feasible = all(final_distances[p] >= cfg.clearance - cfg.validation_tolerance for p in final_pairs)
            iteration_counts[frame] = iteration + 1
            if accepted_step is None:
                # A feasible stationary iterate is a preview candidate, not a
                # proof of global optimality. Leave converged=False explicitly.
                break
            if np.linalg.norm(step.value) < 1e-5 and feasible:
                converged[frame] = True
                break
        if not feasible:
            worst = min(final_pairs, key=lambda p: final_distances[p])
            names = [retargeter._get_geometry_name(g) for g in worst]
            raise RuntimeError(f"PT palm collision frame {frame}: nonlinear distance {final_distances[worst]:.6g} m, {names}")
        result[frame, indices] = current[indices]
        corrections[frame] = current[indices] - base[frame, indices]
        min_arm_distance[frame] = min((final_distances[p] for p in final_pairs), default=retargeter.collision_detection_threshold)
        frozen_penetration[frame] = max(
            (max(0.0, -d) for p, d in final_distances.items() if p not in final_pairs), default=0.0
        )
        for hand, spec in enumerate(specs):
            rotation, _, point = retargeter._calc_body_orientation_linearization(current, int(spec["body_id"]))
            errors[frame, hand] = np.degrees(Rotation.from_matrix(targets[frame, hand] @ rotation.T).magnitude())
            position_drift[frame, hand] = np.linalg.norm(point - position_targets[hand])
            geom = mujoco.mj_name2id(retargeter.robot_model, mujoco.mjtObj.mjOBJ_GEOM, f"{spec['side']}_rubber_hand_link")
            if geom < 0:
                raise ValueError(f"Missing rubber-hand collision geom for {spec['side']}")
            hand_distance[frame, hand] = min(
                mujoco.mj_geomDistance(retargeter.robot_model, retargeter.robot_data, geom, obj,
                                      retargeter.collision_detection_threshold, None)
                for obj in object_collision_geoms
                if ((model.geom_contype[geom] & model.geom_conaffinity[obj])
                    or (model.geom_contype[obj] & model.geom_conaffinity[geom]))
            )

    fixed_indices = np.setdiff1d(np.arange(base.shape[1]), indices)
    if not np.array_equal(result[:, fixed_indices], base[:, fixed_indices]):
        raise AssertionError("Frozen base, legs, waist or object was modified")
    if np.any(result[:, indices] < lower - 1e-6) or np.any(result[:, indices] > upper + 1e-6):
        raise RuntimeError("PT palm collision result violates arm joint limits")
    return result, {
        "orientation_errors_deg": errors,
        "hand_position_drift_m": position_drift,
        "hand_object_distance_m": hand_distance,
        "arm_qpos_indices": indices,
        "arm_correction_rad": corrections,
        "minimum_arm_collision_distance_m": min_arm_distance,
        "inherited_frozen_body_penetration_m": frozen_penetration,
        "sqp_iterations": iteration_counts,
        "sqp_converged": converged,
    }
