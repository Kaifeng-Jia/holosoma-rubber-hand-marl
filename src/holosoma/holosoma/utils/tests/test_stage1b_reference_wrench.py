import numpy as np

from holosoma.analyze_stage1b_reference_wrench import solve_planar_contact_wrench


def _solve(
    hand_points: np.ndarray,
    *,
    required_force: np.ndarray = np.array([0.0, 10.0, 0.0]),
    required_yaw: float = 0.0,
    hand_friction: float = 0.0,
):
    foot_points = np.array(
        [
            [-0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    return solve_planar_contact_wrench(
        hand_points_w_m=hand_points,
        table_com_w_m=np.zeros(3),
        push_normal_w=np.array([0.0, 1.0, 0.0]),
        lateral_tangent_w=np.array([1.0, 0.0, 0.0]),
        foot_points_w_m=foot_points,
        foot_slip_velocities_w_m_s=np.tile(np.array([0.0, 1.0, 0.0]), (4, 1)),
        fallback_slip_velocity_w_m_s=np.array([0.0, 1.0, 0.0]),
        required_planar_force_w_n=required_force,
        required_yaw_moment_nm=required_yaw,
        table_weight_n=20.0,
        hand_friction=hand_friction,
        ground_friction=0.0,
    )


def test_symmetric_hands_supply_translation_without_yaw() -> None:
    solution = _solve(np.array([[-0.4, -0.25, 0.1], [0.4, -0.25, 0.1]]))

    assert solution.feasible
    assert np.isclose(solution.objective_n, 10.0)
    assert np.allclose(solution.hand_forces_w_n.sum(axis=0), [0.0, 10.0, 0.0])


def test_off_center_hand_needs_sufficient_friction_for_coupled_lateral_wrench() -> None:
    hand_point = np.array([[-0.4, -0.25, 0.1]])
    required_force = np.array([16.0, 10.0, 0.0])

    assert not _solve(hand_point, required_force=required_force, hand_friction=1.0).feasible
    feasible = _solve(hand_point, required_force=required_force, hand_friction=2.0)
    assert feasible.feasible
    force = feasible.hand_forces_w_n[0]
    assert np.allclose(force, required_force)
    assert np.isclose(np.cross(hand_point[0], force)[2], 0.0, atol=1.0e-8)


def test_single_hand_can_supply_requested_natural_yaw() -> None:
    hand_point = np.array([[-0.4, -0.25, 0.1]])
    solution = _solve(hand_point, required_yaw=-4.0, hand_friction=0.0)

    assert solution.feasible
    assert np.allclose(solution.hand_forces_w_n[0], [0.0, 10.0, 0.0])


def test_yaw_slack_reports_minimum_moment_relaxation() -> None:
    hand_point = np.array([[-0.4, -0.25, 0.1]])
    solution = solve_planar_contact_wrench(
        hand_points_w_m=hand_point,
        table_com_w_m=np.zeros(3),
        push_normal_w=np.array([0.0, 1.0, 0.0]),
        lateral_tangent_w=np.array([1.0, 0.0, 0.0]),
        foot_points_w_m=np.array(
            [[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0]]
        ),
        foot_slip_velocities_w_m_s=np.tile(np.array([0.0, 1.0, 0.0]), (4, 1)),
        fallback_slip_velocity_w_m_s=np.array([0.0, 1.0, 0.0]),
        required_planar_force_w_n=np.array([0.0, 10.0, 0.0]),
        required_yaw_moment_nm=0.0,
        table_weight_n=20.0,
        hand_friction=0.0,
        ground_friction=0.0,
        allow_yaw_moment_slack=True,
    )

    assert solution.feasible
    assert np.isclose(solution.yaw_moment_residual_nm, -4.0)


def test_no_active_hand_is_not_feasible() -> None:
    solution = _solve(np.empty((0, 3)))

    assert not solution.feasible
    assert solution.message == "no active rubber-hand contact"


def test_tilted_local_axes_are_orthogonalized_in_the_planar_model() -> None:
    solution = solve_planar_contact_wrench(
        hand_points_w_m=np.array([[-0.4, -0.25, 0.1], [0.4, -0.25, 0.1]]),
        table_com_w_m=np.zeros(3),
        push_normal_w=np.array([0.0, 1.0, 0.1]),
        lateral_tangent_w=np.array([1.0, 0.02, 0.1]),
        foot_points_w_m=np.array(
            [[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0]]
        ),
        foot_slip_velocities_w_m_s=np.tile(np.array([0.0, 1.0, 0.0]), (4, 1)),
        fallback_slip_velocity_w_m_s=np.array([0.0, 1.0, 0.0]),
        required_planar_force_w_n=np.array([0.0, 10.0, 0.0]),
        required_yaw_moment_nm=0.0,
        table_weight_n=20.0,
        hand_friction=0.1,
        ground_friction=0.0,
    )

    assert solution.feasible
