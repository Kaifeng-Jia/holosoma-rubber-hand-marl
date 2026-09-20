"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Example: {"L_Toe": [(30, 60)], "R_Toe": [(10, 20), (80, 95)]}"""

    z_floor: float = 0.0
    """Floor height used by Z pinning constraints."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Configuration for self-collision avoidance constraints."""

    enable: bool = False
    """Whether to enforce self-collision constraints."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    """Body name pairs to check for self-collision.
    Example: [("left_elbow_link", "left_knee_link"), ("left_wrist_yaw_link", "left_knee_link")]"""

    windows: list[tuple[int, int]] | None = None
    """Inclusive frame windows during which self-collision is enforced.
    If None, enforced on all frames.
    Example: [(50, 120)] means only enforce on frames 50..120."""

    tolerance: float = 0.02
    """Minimum distance (meters) to maintain between body pairs."""


@dataclass(frozen=True)
class HandOrientationConfig:
    """Optional standardized G1 rubber-hand pushing pose."""

    enable: bool = False
    """Whether to use the fixed palm-forward, fingers-up pose near the object."""

    contact_distance: float = 0.10
    """Demo wrist-to-object distance (meters) used to detect contact."""

    fade_frames: int = 8
    """Number of frames used to fade the objective in and out around contact."""

    palm_direction_weight: float = 5.0
    """Weight for pointing the palm toward the object during contact."""

    finger_direction_weight: float = 0.0
    """Optional roll-stabilization weight; zero preserves the stable palm-only solution."""


@dataclass(frozen=True)
class PlanBPalmContactConfig:
    """Task-space palm contact used by the rubber-hand Plan B pipeline."""

    enable: bool = False
    """Whether to optimize a collision-safe, full-body palm pushing pose."""

    tabletop_geom_name: str = "largetable_tabletop"
    """MuJoCo box geom whose vertical side is used as the pushing surface."""

    face_axis: int = 2
    """Tabletop geom-local axis normal to the contact face (0=x, 1=y, 2=z)."""

    face_sign: int = -1
    """Signed side of ``face_axis`` facing the robot; must be -1 or +1."""

    vertical_axis: int = 1
    """Tabletop geom-local vertical axis."""

    lateral_axis: int = 0
    """Tabletop geom-local axis along which the two palms are separated."""

    hand_spacing: float = 0.24
    """Distance in meters between the left and right palm target centers."""

    vertical_offset: float = 0.0
    """Contact-center offset from the tabletop side center along the vertical axis."""

    surface_gap: float = 1e-3
    """Numerical offset retained in the virtual contact target."""

    penetration_tolerance: float = 0.0
    """Plan B robot/object penetration tolerance; zero enforces separation."""

    collision_validation_tolerance: float = 1e-4
    """Numerical tolerance used only to validate MuJoCo mesh signed distances."""

    approach_clearance: float = 0.05
    """Extra clearance at the beginning and end of a faded contact window."""

    contact_distance: float = 0.10
    """Demo wrist-to-object distance used to identify the contact phase."""

    fade_frames: int = 24
    """Frames over which the Plan B task fades in and out."""

    release_frames: int = 36
    """Frames used only to release Plan B after demonstrated contact ends."""

    normal_position_weight: float = 10000.0
    """Weight for placing the palm support point on the side plane."""

    contact_distance_weight: float = 100000.0
    """Weight for driving the real palm/table signed distance to zero."""

    tangent_position_weight: float = 300.0
    """Weight for placing the palm at the requested side-face coordinates."""

    palm_normal_weight: float = 1000.0
    """Weight for pointing the palm normal into the table."""

    twist_weight: float = 100.0
    """Weight for pointing both rubber-hand long axes downward."""

    posture_weight: float = 0.5
    """Weak Baseline tracking weight for waist and upper-body joints."""

    waist_yaw_roll_weight: float = 5.0
    """Additional Baseline tracking weight for waist yaw and roll."""

    max_sqp_iterations: int = 60
    """Maximum SQP iterations per Plan B frame before strict validation fails."""

    temporal_smooth_weight: float = 0.2
    """Plan B cost weight for staying close to the previous frame."""


@dataclass(frozen=True)
class PTWristOrientationConfig:
    """Optional A.1 raw-PT wrist orientation post-processing."""

    enable: bool = False
    """Whether to replace only wrist roll/pitch/yaw with the demonstrated orientation."""

    max_calibration_error_deg: float = 1.0
    """Maximum allowed 90th-percentile wrist-to-palm calibration error in degrees."""

    max_solver_error_deg: float = 0.01
    """Maximum allowed per-frame robot hand-link orientation error in degrees."""


@dataclass(frozen=True)
class PTFullArmOrientationConfig:
    """Optional demonstrated-palm refinement using each complete G1 arm."""

    enable: bool = False
    """Whether shoulder, elbow, and wrist jointly track the demonstrated palm."""

    orientation_weight: float = 100.0
    """Quadratic weight for hand-link orientation error in radians."""

    hand_position_weight: float = 1.0e4
    """Quadratic weight for drift from the baseline hand-link position in metres."""

    arm_prior_weight: float = 1.0
    """Quadratic weight for changes from the baseline seven-joint arm pose."""

    correction_temporal_weight: float = 5.0
    """Quadratic weight for frame-to-frame changes in the refinement correction."""

    max_nfev: int = 200
    """Maximum nonlinear least-squares evaluations for each hand and frame."""


@dataclass(frozen=True)
class PTWristDominantSurfaceConfig:
    """Demonstrated palm refinement that keeps proximal arm motion conservative."""

    enable: bool = False
    """Whether wrist-led palm-normal and support-point tracking is enabled."""

    normal_weight: float = 100.0
    """Quadratic weight for aligning the rubber-palm surface normal."""

    finger_weight: float = 10.0
    """Weaker quadratic weight for aligning the rubber-hand long axis."""

    surface_position_weight: float = 1.0e4
    """Quadratic weight for preserving the baseline palm support point in metres."""

    proximal_prior_weight: float = 100.0
    """Quadratic prior on shoulder and elbow corrections."""

    wrist_prior_weight: float = 1.0
    """Quadratic prior on wrist corrections."""

    proximal_correction_temporal_weight: float = 100.0
    """Temporal weight on shoulder/elbow correction changes."""

    wrist_correction_temporal_weight: float = 20.0
    """Temporal weight on wrist correction changes."""

    max_nfev: int = 200
    """Maximum nonlinear least-squares evaluations for each hand and frame."""


@dataclass(frozen=True)
class PTPalmCollisionConfig:
    """Demonstrated palm orientation with collision-constrained arm refinement."""

    enable: bool = False
    """Opt-in final stage; does not change existing A.1 or Plan B."""

    orientation_weight: float = 100.0
    hand_position_weight: float = 1.0e4
    arm_prior_weight: float = 1.0
    correction_temporal_weight: float = 5.0
    """Soft costs, not exact-pose requirements; positions are in metres."""

    clearance: float = 0.0
    """Minimum distance for movable-arm/object and ground pairs; zero allows touch."""

    validation_tolerance: float = 1.0e-4
    """Tolerance on final nonlinear distance, NOT an optimization slack."""

    max_iterations: int = 60
    """SQP iterations per frame, with the existing trust-region radius."""


@dataclass(frozen=True)
class ElasticConstraintConfig:
    """Optional exact-penalty relaxation for selected SQP constraints."""

    enable: bool = False
    """Whether object and foot constraints may use penalized slack variables."""

    object_collision_weight: float = 1.0e5
    """L1 penalty per metre of robot-object non-penetration slack."""

    foot_kinematics_weight: float = 1.0e4
    """L1 penalty per metre of foot XY or nominal-height slack."""


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    apply_manual_joint_limit_overrides: bool = True
    """Whether to tighten URDF joint limits with task-specific manual bounds."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""

    hand_orientation: HandOrientationConfig = field(default_factory=HandOrientationConfig)
    """Optional contact-phase hand orientation for fixed-size object adaptation."""

    plan_b_palm_contact: PlanBPalmContactConfig = field(default_factory=PlanBPalmContactConfig)
    """Plan B collision-safe palm contact and full-body pose optimization."""

    pt_wrist_orientation: PTWristOrientationConfig = field(default_factory=PTWristOrientationConfig)
    """A.1 raw-PT wrist-only orientation post-processing for rigid robot hands."""

    pt_full_arm_orientation: PTFullArmOrientationConfig = field(
        default_factory=PTFullArmOrientationConfig
    )
    """Raw-PT palm-orientation refinement using shoulder, elbow, and wrist."""

    pt_wrist_dominant_surface: PTWristDominantSurfaceConfig = field(
        default_factory=PTWristDominantSurfaceConfig
    )
    """Wrist-led demonstrated-palm refinement with a fixed support point."""

    pt_palm_collision: PTPalmCollisionConfig = field(default_factory=PTPalmCollisionConfig)
    """PT palm targets with hard arm/object non-penetration in the final stage."""

    elastic_constraints: ElasticConstraintConfig = field(default_factory=ElasticConstraintConfig)
    """Optional slack variables for otherwise infeasible object/foot constraints."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""
