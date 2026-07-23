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
class PTWristOrientationConfig:
    """Optional A.1 raw-PT wrist orientation post-processing."""

    enable: bool = False
    """Whether to replace only wrist roll/pitch/yaw with the demonstrated orientation."""

    max_calibration_error_deg: float = 1.0
    """Maximum allowed 90th-percentile wrist-to-palm calibration error in degrees."""

    max_solver_error_deg: float = 0.01
    """Maximum allowed per-frame robot hand-link orientation error in degrees."""


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

    pt_wrist_orientation: PTWristOrientationConfig = field(default_factory=PTWristOrientationConfig)
    """A.1 raw-PT wrist-only orientation post-processing for rigid robot hands."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""
