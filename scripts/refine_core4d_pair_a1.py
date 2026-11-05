#!/usr/bin/env python3
"""Refine demonstrated palm orientation on a frozen CORE4D pair baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    REPO_ROOT
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
)
sys.path.insert(0, str(PACKAGE_ROOT.parent))

from holosoma_retargeting.data_utils.core4d_adapter import (  # noqa: E402
    load_canonical_core4d_sequence,
    resample_core4d_pair_sequence,
)
from holosoma_retargeting.data_utils.core4d_retarget import (  # noqa: E402
    CORE4D_A1_PALM_LANDMARKS,
    build_core4d_a1_palm_orientation_targets,
    require_shared_physical_object_qpos,
)


LOGGER = logging.getLogger(__name__)
BASE_G1_URDF = PACKAGE_ROOT / "models" / "g1" / "g1_29dof.urdf"
ROBOT_QPOS_WIDTH = 36
OBJECT_QPOS_WIDTH = 7
WRIST_QPOS_INDICES = np.asarray([26, 27, 28, 33, 34, 35], dtype=np.int64)
FULL_ARM_QPOS_INDICES = np.asarray(
    [
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
    ],
    dtype=np.int64,
)
SOLVER_MODES = ("wrist-only", "full-arm", "wrist-dominant")
TIME_SERIES_DIAGNOSTICS = frozenset(
    {
        "orientation_errors_deg",
        "palm_normal_errors_deg",
        "finger_direction_errors_deg",
        "hand_position_errors_m",
        "surface_position_errors_m",
        "link_origin_errors_m",
        "arm_corrections_rad",
        "arm_steps_rad",
        "correction_steps_rad",
        "solver_success",
        "solver_nfev",
        "solver_cost",
    }
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Frozen position-only CORE4D pair directory.",
    )
    parser.add_argument(
        "--baseline-kind",
        choices=("fixed-object", "shared-scaled-preview"),
        default="fixed-object",
        help="Explicit input contract; the original fixed-object checks remain the default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty output directory for the A.1-refined pair.",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=None,
        help="Canonical source override; defaults to the parent manifest input.",
    )
    parser.add_argument(
        "--solver-mode",
        choices=SOLVER_MODES,
        default="wrist-only",
        help=(
            "Palm-orientation refinement solver. 'wrist-only' preserves the "
            "existing A.1 behavior. 'full-arm' and 'wrist-dominant' may change "
            "only each arm's shoulder, elbow, and wrist coordinates."
        ),
    )
    parser.add_argument(
        "--max-calibration-error-deg",
        type=_positive_float,
        default=1.0,
        help="Existing A.1 wrist-to-palm calibration threshold.",
    )
    parser.add_argument(
        "--max-solver-error-deg",
        type=_positive_float,
        default=0.01,
        help=(
            "Existing A.1 robot hand-link orientation threshold; applies only "
            "to --solver-mode wrist-only. The other modes use weighted soft "
            "tradeoffs."
        ),
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_empty_output_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to mix an A.1 run with existing files: {path}")


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_npz_copy(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required NPZ file does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _validate_pair_data(pair: dict[str, np.ndarray]) -> tuple[int, int, str]:
    required = {"robot_qpos", "object_qpos", "fps", "object_name"}
    missing = sorted(required.difference(pair))
    if missing:
        raise ValueError(f"Baseline pair is missing keys: {missing}")
    robot_qpos = np.asarray(pair["robot_qpos"], dtype=np.float64)
    object_qpos = np.asarray(pair["object_qpos"], dtype=np.float64)
    if robot_qpos.ndim != 3 or robot_qpos.shape[1:] != (2, ROBOT_QPOS_WIDTH):
        raise ValueError(
            "robot_qpos must have shape (T, 2, 36), got "
            f"{robot_qpos.shape}"
        )
    frames = len(robot_qpos)
    if frames < 1 or object_qpos.shape != (frames, OBJECT_QPOS_WIDTH):
        raise ValueError(
            f"object_qpos must have shape ({frames}, 7), got {object_qpos.shape}"
        )
    if not np.isfinite(robot_qpos).all() or not np.isfinite(object_qpos).all():
        raise ValueError("Baseline pair contains non-finite qpos")
    fps_array = np.asarray(pair["fps"])
    if fps_array.shape != ():
        raise ValueError(f"fps must be scalar, got {fps_array.shape}")
    fps_value = fps_array.item()
    fps = int(fps_value)
    if fps <= 0 or float(fps_value) != float(fps):
        raise ValueError(f"fps must be a positive integer, got {fps_value!r}")
    object_name_array = np.asarray(pair["object_name"])
    if object_name_array.shape != ():
        raise ValueError("object_name must be scalar")
    object_name = str(object_name_array.item())
    if not object_name:
        raise ValueError("object_name must be non-empty")
    return frames, fps, object_name


def _compose_person_qpos(pair: dict[str, np.ndarray], person_index: int) -> np.ndarray:
    if person_index not in (0, 1):
        raise ValueError(f"person_index must be 0 or 1, got {person_index}")
    return np.concatenate(
        (
            np.asarray(pair["robot_qpos"], dtype=np.float64)[:, person_index],
            np.asarray(pair["object_qpos"], dtype=np.float64),
        ),
        axis=1,
    )


def _load_shared_scaled_source(
    baseline_dir: Path,
    manifest: dict[str, object],
    pair: dict[str, np.ndarray],
) -> dict[str, object]:
    """Validate the existing scaled-preview export without pretending it is full-size."""
    if (
        manifest.get("kind") != "diagnostic_preview_not_training_asset"
        or manifest.get("robot_reference") != "per-person Stage1 nominal_scaled qpos"
        or manifest.get("training_ready") is not False
    ):
        raise ValueError("Expected an unrefined shared-scaled Stage1 preview")
    object_name = str(pair["object_name"].item())
    if Path(object_name).name != object_name or object_name in {"", ".", ".."}:
        raise ValueError("Preview object_name must be a plain filename stem")
    for path, hash_key in (
        (baseline_dir / "core4d_pair_reference.npz", "pair_sha256"),
        (baseline_dir / "assets" / f"{object_name}.urdf", "object_urdf_sha256"),
        (Path(str(manifest["source_manifest"])), "source_manifest_sha256"),
    ):
        if not manifest.get(hash_key) or _sha256(path) != manifest[hash_key]:
            raise ValueError(f"Shared-scaled preview {hash_key} mismatch")
    source_manifest = _load_json_object(Path(str(manifest["source_manifest"])))
    scales = np.asarray(pair["human_to_robot_scales"], dtype=np.float64)
    shared_scale = np.asarray(pair["shared_object_scale"], dtype=np.float64)
    if (
        scales.shape != (2,)
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or shared_scale.shape != ()
        or float(shared_scale) != float(scales.mean())
        or float(shared_scale) != manifest.get("shared_object_scale")
        or not np.array_equal(scales, source_manifest.get("human_to_robot_scales"))
    ):
        raise ValueError("Shared-scaled preview scale differs from its source")
    paths = manifest.get("source_person_nominal")
    if not isinstance(paths, list) or len(paths) != 2:
        raise ValueError("Preview must identify both Stage1 nominal files")
    nominal_objects = []
    for index, path in enumerate(paths):
        nominal = _load_npz_copy(Path(str(path)))
        qpos = np.asarray(nominal["qpos"], dtype=np.float64)
        if (
            qpos.shape != (len(pair["robot_qpos"]), 43)
            or not np.isfinite(qpos).all()
            or not np.array_equal(nominal["fps"], pair["fps"])
            or not np.array_equal(qpos[:, :36], pair["robot_qpos"][:, index])
        ):
            raise ValueError(f"Preview person{index + 1} differs from Stage1 nominal")
        nominal_objects.append(qpos[:, 36:])
    first, second = nominal_objects
    norms = np.linalg.norm(first[:, 3:], axis=1, keepdims=True)
    if (
        np.any(norms < 1e-12)
        or not np.allclose(first[:, 3:], second[:, 3:], atol=1e-12, rtol=0)
        or not np.array_equal(pair["object_qpos"][:, :3], (first[:, :3] + second[:, :3]) / 2)
        or not np.allclose(pair["object_qpos"][:, 3:], first[:, 3:] / norms, atol=1e-12, rtol=0)
    ):
        raise ValueError("Preview shared object differs from the Stage1 averaging contract")
    return source_manifest


def _require_frozen_object(qpos_sequence: list[np.ndarray], object_qpos: np.ndarray) -> None:
    """The frozen object may be full-size or scaled; never substitute another trajectory."""
    for qpos in qpos_sequence:
        if not np.array_equal(np.asarray(qpos)[:, ROBOT_QPOS_WIDTH:], object_qpos):
            raise RuntimeError("A.1 changed the frozen baseline object trajectory")


def _allowed_qpos_indices(solver_mode: str) -> np.ndarray:
    if solver_mode == "wrist-only":
        return WRIST_QPOS_INDICES.copy()
    if solver_mode in ("full-arm", "wrist-dominant"):
        return FULL_ARM_QPOS_INDICES.copy()
    raise ValueError(f"Unsupported solver mode: {solver_mode!r}")


def _result_suffix(solver_mode: str) -> str:
    try:
        return {
            "wrist-only": "a1",
            "full-arm": "a1_full_arm",
            "wrist-dominant": "a1_wrist_dominant",
        }[solver_mode]
    except KeyError as error:
        raise ValueError(f"Unsupported solver mode: {solver_mode!r}") from error


def _require_only_allowed_change(
    baseline_qpos: np.ndarray,
    refined_qpos: np.ndarray,
    *,
    allowed_qpos_indices: np.ndarray,
    solver_mode: str,
) -> None:
    baseline = np.asarray(baseline_qpos, dtype=np.float64)
    refined = np.asarray(refined_qpos, dtype=np.float64)
    if baseline.shape != refined.shape or baseline.ndim != 2:
        raise ValueError(
            f"Baseline/refined qpos shapes must match, got {baseline.shape} and {refined.shape}"
        )
    indices = np.asarray(allowed_qpos_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= baseline.shape[1]):
        raise ValueError(f"Invalid allowed qpos indices: {indices.tolist()}")
    allowed = np.zeros(baseline.shape[1], dtype=bool)
    allowed[indices] = True
    if not np.array_equal(refined[:, ~allowed], baseline[:, ~allowed]):
        changed = np.flatnonzero(np.any(refined != baseline, axis=0))
        forbidden = changed[~np.isin(changed, indices)]
        if solver_mode == "wrist-only" and np.array_equal(
            indices,
            WRIST_QPOS_INDICES,
        ):
            raise RuntimeError(
                "A.1 changed coordinates outside the six wrist joints: "
                f"{forbidden.tolist()}"
            )
        raise RuntimeError(
            f"A.1 {solver_mode} changed coordinates outside its allowed joints: "
            f"{forbidden.tolist()}"
        )


def _require_wrist_only_change(
    baseline_qpos: np.ndarray,
    refined_qpos: np.ndarray,
) -> None:
    """Preserve the original wrist-only contract helper for callers/tests."""
    _require_only_allowed_change(
        baseline_qpos,
        refined_qpos,
        allowed_qpos_indices=WRIST_QPOS_INDICES,
        solver_mode="wrist-only",
    )


def _build_a1_retargeter(
    *,
    object_name: str,
    object_urdf_path: Path,
    scene_xml_path: Path,
    foot_links: list[str],
    max_solver_error_deg: float,
    solver_mode: str = "wrist-only",
):
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    from holosoma_retargeting.config_types.retargeter import (
        PTFullArmOrientationConfig,
        PTWristDominantSurfaceConfig,
        PTWristOrientationConfig,
        RetargeterConfig,
    )
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.examples.robot_retarget import (
        build_retargeter_kwargs_from_config,
        create_task_constants,
    )
    from holosoma_retargeting.src.interaction_mesh_retargeter import (
        InteractionMeshRetargeter,
    )

    robot_config = RobotConfig(
        robot_type="g1",
        robot_urdf_file=str(BASE_G1_URDF.resolve()),
        foot_sticking_links=foot_links,
    )
    motion_config = MotionDataConfig(data_format="smplx", robot_type="g1")
    task_config = TaskConfig(object_name=object_name)
    wrist_config = PTWristOrientationConfig(
        enable=solver_mode == "wrist-only",
        max_solver_error_deg=max_solver_error_deg,
    )
    full_arm_config = PTFullArmOrientationConfig(
        enable=solver_mode == "full-arm",
    )
    wrist_dominant_config = PTWristDominantSurfaceConfig(
        enable=solver_mode == "wrist-dominant",
    )
    retargeter_config = RetargeterConfig(
        visualize=False,
        debug=False,
        pt_wrist_orientation=wrist_config,
        pt_full_arm_orientation=full_arm_config,
        pt_wrist_dominant_surface=wrist_dominant_config,
    )
    constants = create_task_constants(
        robot_config,
        motion_config,
        task_config,
        "object_interaction",
    )
    constants.OBJECT_URDF_FILE = str(object_urdf_path)
    constants.SCENE_XML_FILE = str(scene_xml_path)
    kwargs = build_retargeter_kwargs_from_config(
        retargeter_config,
        constants,
        str(object_urdf_path),
        "object_interaction",
    )
    kwargs["scene_xml_path"] = str(scene_xml_path)
    # Keep this script self-contained: the shared factory intentionally still
    # serves older RetargeterConfig versions that do not know this opt-in mode.
    kwargs["pt_full_arm_orientation"] = full_arm_config
    kwargs["pt_wrist_dominant_surface"] = wrist_dominant_config
    return InteractionMeshRetargeter(**kwargs)


def _error_summary(errors_deg: np.ndarray) -> dict[str, dict[str, float]]:
    values = np.asarray(errors_deg, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected left/right errors with shape (T, 2), got {values.shape}")
    return {
        side: {
            "p90": float(np.percentile(values[:, index], 90)),
            "p95": float(np.percentile(values[:, index], 95)),
            "max": float(np.max(values[:, index])),
        }
        for index, side in enumerate(("left", "right"))
    }


def _joint_motion_summary(
    baseline_qpos: np.ndarray,
    refined_qpos: np.ndarray,
    fps: int,
    qpos_indices: np.ndarray,
) -> dict[str, float | list[int]]:
    indices = np.asarray(qpos_indices, dtype=np.int64)
    delta = np.asarray(refined_qpos)[:, indices] - np.asarray(baseline_qpos)[:, indices]
    speed = np.diff(np.asarray(refined_qpos)[:, indices], axis=0) * fps
    changed = indices[np.any(delta != 0.0, axis=0)]
    speed_abs = np.abs(speed)
    return {
        "allowed_qpos_indices": indices.tolist(),
        "actually_changed_qpos_indices": changed.tolist(),
        "absolute_delta_rad_p90": float(np.percentile(np.abs(delta), 90)),
        "absolute_delta_rad_max": float(np.max(np.abs(delta))),
        "absolute_velocity_rad_s_p90": (
            float(np.percentile(speed_abs, 90)) if speed_abs.size else 0.0
        ),
        "absolute_velocity_rad_s_max": (
            float(np.max(speed_abs)) if speed_abs.size else 0.0
        ),
    }


def _wrist_motion_summary(
    baseline_qpos: np.ndarray,
    refined_qpos: np.ndarray,
    fps: int,
) -> dict[str, float | list[int]]:
    """Preserve the wrist-only manifest helper and its field names."""
    return _joint_motion_summary(
        baseline_qpos,
        refined_qpos,
        fps,
        WRIST_QPOS_INDICES,
    )


def _validate_refinement_diagnostics(
    diagnostics: Mapping[str, object],
    *,
    frames: int,
    solver_mode: str,
) -> dict[str, np.ndarray | float | int]:
    """Validate and copy mode-specific diagnostics from a refinement solver."""
    copied: dict[str, np.ndarray | float | int] = {}
    for name, value in diagnostics.items():
        array = np.asarray(value)
        if array.shape == ():
            scalar = array.item()
            if isinstance(scalar, (bool, np.bool_)):
                copied[name] = int(bool(scalar))
            elif isinstance(scalar, (int, np.integer)):
                copied[name] = int(scalar)
            elif isinstance(scalar, (float, np.floating)):
                if not np.isfinite(scalar):
                    raise ValueError(f"Diagnostic {name!r} is non-finite")
                copied[name] = float(scalar)
            else:
                raise ValueError(f"Diagnostic {name!r} has unsupported scalar type")
            continue
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"Diagnostic {name!r} contains non-finite values")
        copied[name] = array.copy()

    orientation = np.asarray(copied.get("orientation_errors_deg"))
    if orientation.shape != (frames, 2):
        raise ValueError(
            "orientation_errors_deg must have shape "
            f"({frames}, 2), got {orientation.shape}"
        )
    if solver_mode in ("full-arm", "wrist-dominant"):
        expected_shapes = {
            "arm_corrections_rad": (frames, 2, 7),
            "arm_steps_rad": (frames, 2, 7),
            "correction_steps_rad": (frames, 2, 7),
            "arm_qpos_indices": (2, 7),
            "solver_success": (frames, 2),
            "solver_nfev": (frames, 2),
            "solver_cost": (frames, 2),
        }
        if solver_mode == "full-arm":
            expected_shapes["hand_position_errors_m"] = (frames, 2)
        else:
            expected_shapes.update(
                {
                    "palm_normal_errors_deg": (frames, 2),
                    "finger_direction_errors_deg": (frames, 2),
                    "surface_position_errors_m": (frames, 2),
                    "link_origin_errors_m": (frames, 2),
                }
            )
        for name, expected in expected_shapes.items():
            actual = np.asarray(copied.get(name))
            if actual.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {actual.shape}"
                )
        resolved = np.sort(
            np.asarray(copied["arm_qpos_indices"], dtype=np.int64).reshape(-1)
        )
        if not np.array_equal(resolved, FULL_ARM_QPOS_INDICES):
            raise RuntimeError(
                "Full-arm solver reported qpos indices "
                f"{resolved.tolist()}, expected {FULL_ARM_QPOS_INDICES.tolist()}"
            )
    return copied


def _apply_refinement_solver(
    retargeter,
    *,
    solver_mode: str,
    baseline_qpos: np.ndarray,
    palm_targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    """Adapt both solver APIs to one explicit CLI-level result contract."""
    if solver_mode == "wrist-only":
        refined, orientation_errors = (
            retargeter.apply_pt_wrist_orientation_postprocess(
                baseline_qpos,
                palm_targets,
            )
        )
        raw_diagnostics: Mapping[str, object] = {
            "orientation_errors_deg": orientation_errors,
        }
    elif solver_mode == "full-arm":
        refined, raw_diagnostics = (
            retargeter.apply_pt_full_arm_orientation_postprocess(
                baseline_qpos,
                palm_targets,
            )
        )
        if not isinstance(raw_diagnostics, Mapping):
            raise TypeError("Full-arm solver diagnostics must be a mapping")
    elif solver_mode == "wrist-dominant":
        refined, raw_diagnostics = (
            retargeter.apply_pt_wrist_dominant_surface_postprocess(
                baseline_qpos,
                palm_targets,
            )
        )
        if not isinstance(raw_diagnostics, Mapping):
            raise TypeError("Wrist-dominant solver diagnostics must be a mapping")
    else:
        raise ValueError(f"Unsupported solver mode: {solver_mode!r}")
    diagnostics = _validate_refinement_diagnostics(
        raw_diagnostics,
        frames=len(np.asarray(baseline_qpos)),
        solver_mode=solver_mode,
    )
    return np.asarray(refined, dtype=np.float64), diagnostics


def _array_distribution_summary(values: object) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"p90": 0.0, "p95": 0.0, "max": 0.0}
    absolute = np.abs(array)
    return {
        "p90": float(np.percentile(absolute, 90)),
        "p95": float(np.percentile(absolute, 95)),
        "max": float(np.max(absolute)),
    }


def _left_right_distribution_summary(values: object) -> dict[str, object]:
    array = np.asarray(values)
    if array.ndim < 2 or array.shape[1] != 2:
        return {"all": _array_distribution_summary(array)}
    return {
        side: _array_distribution_summary(array[:, index])
        for index, side in enumerate(("left", "right"))
    }


def _diagnostics_manifest_summary(
    diagnostics: Mapping[str, np.ndarray | float | int],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "orientation_errors_deg": _error_summary(
            np.asarray(diagnostics["orientation_errors_deg"])
        )
    }
    for name in (
        "palm_normal_errors_deg",
        "finger_direction_errors_deg",
        "hand_position_errors_m",
        "surface_position_errors_m",
        "link_origin_errors_m",
    ):
        if name in diagnostics:
            summary[name] = _error_summary(np.asarray(diagnostics[name]))
    for name in (
        "arm_corrections_rad",
        "arm_steps_rad",
        "correction_steps_rad",
        "solver_nfev",
        "solver_cost",
    ):
        if name in diagnostics:
            summary[name] = _left_right_distribution_summary(diagnostics[name])
    if "solver_success" in diagnostics:
        success = np.asarray(diagnostics["solver_success"], dtype=bool)
        summary["solver_success"] = {
            "successful": int(np.count_nonzero(success)),
            "total": int(success.size),
            "rate": float(np.mean(success)) if success.size else 0.0,
        }
    reported_scalars = {
        name: value
        for name, value in diagnostics.items()
        if np.asarray(value).shape == ()
    }
    if reported_scalars:
        summary["reported_scalars"] = reported_scalars
    return summary


def _solver_config_snapshot(retargeter, solver_mode: str) -> dict[str, object]:
    config_name = {
        "wrist-only": "pt_wrist_orientation",
        "full-arm": "pt_full_arm_orientation",
        "wrist-dominant": "pt_wrist_dominant_surface",
    }.get(solver_mode)
    if config_name is None:
        raise ValueError(f"Unsupported solver mode: {solver_mode!r}")
    config = getattr(retargeter, config_name)
    if not is_dataclass(config):
        raise TypeError(f"{config_name} must be a dataclass configuration")
    return asdict(config)


def run(args: argparse.Namespace) -> Path:
    baseline_dir = args.baseline_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline directory does not exist: {baseline_dir}")
    _require_empty_output_dir(output_dir)

    parent_manifest_path = baseline_dir / "manifest.json"
    parent_pair_path = baseline_dir / "core4d_pair_reference.npz"
    parent_manifest = _load_json_object(parent_manifest_path)
    pair = _load_npz_copy(parent_pair_path)
    frames, fps, object_name = _validate_pair_data(pair)
    scaled_preview = args.baseline_kind == "shared-scaled-preview"
    source_manifest = (
        _load_shared_scaled_source(baseline_dir, parent_manifest, pair)
        if scaled_preview
        else parent_manifest
    )
    if source_manifest.get("robot") != "g1_29dof_rubber_hands":
        raise ValueError("The baseline manifest is not a rubber-hand G1 run")
    if not str(source_manifest.get("wrist_mode", "")).startswith("position_only_baseline"):
        raise ValueError("The baseline manifest is not the frozen position-only wrist run")
    if any(value.get("object_name") != object_name for value in (parent_manifest, source_manifest)):
        raise ValueError("The baseline manifest and pair object names differ")

    canonical_path = (
        args.canonical.expanduser().resolve()
        if args.canonical is not None
        else Path(str(source_manifest["input"])).expanduser().resolve()
    )
    if not canonical_path.is_file():
        raise FileNotFoundError(f"Canonical CORE4D source does not exist: {canonical_path}")
    expected_input_hash = str(source_manifest.get("input_sha256", ""))
    if expected_input_hash and _sha256(canonical_path) != expected_input_hash:
        raise ValueError("Canonical source SHA256 differs from the frozen baseline manifest")

    sequence = load_canonical_core4d_sequence(canonical_path)
    if sequence.fps != fps:
        sequence = resample_core4d_pair_sequence(sequence, fps)
    if len(sequence.human_joints) != frames:
        raise ValueError(
            f"Canonical sequence has {len(sequence.human_joints)} frames at {fps} Hz; "
            f"baseline has {frames}"
        )
    if sequence.object_name != object_name:
        raise ValueError("Canonical and baseline object names differ")

    baseline_qpos = [_compose_person_qpos(pair, index) for index in range(2)]
    if not scaled_preview:
        require_shared_physical_object_qpos(baseline_qpos, sequence.object_poses)
    _require_frozen_object(baseline_qpos, pair["object_qpos"])
    person_baselines: list[dict[str, np.ndarray]] = []
    for person_index, qpos in enumerate(baseline_qpos):
        if scaled_preview:
            person_baselines.append({"qpos": qpos.copy(), "fps": pair["fps"].copy()})
            continue
        person_path = (
            baseline_dir
            / f"person{person_index + 1}"
            / f"person{person_index + 1}_fixed_object.npz"
        )
        person_data = _load_npz_copy(person_path)
        if "qpos" not in person_data or not np.array_equal(person_data["qpos"], qpos):
            raise ValueError(f"{person_path} does not exactly match the baseline pair")
        person_baselines.append(person_data)

    object_urdf_path = baseline_dir / "assets" / f"{object_name}.urdf"
    scene_xml_path = baseline_dir / "assets" / f"g1_29dof_w_{object_name}.xml"
    if scaled_preview:
        scene_xml_path = Path(str(source_manifest["scene_xml"]))
    for asset in (BASE_G1_URDF, object_urdf_path, scene_xml_path):
        if not asset.is_file():
            raise FileNotFoundError(f"Required A.1 asset does not exist: {asset}")
    foot_anchor = source_manifest.get("foot_anchor")
    if not isinstance(foot_anchor, dict) or not isinstance(foot_anchor.get("links"), list):
        raise ValueError("Baseline manifest is missing the frozen foot-anchor links")
    foot_links = [str(value) for value in foot_anchor["links"]]

    palm_targets: list[np.ndarray] = []
    calibration_errors: list[np.ndarray] = []
    for person_index in range(2):
        targets, errors = build_core4d_a1_palm_orientation_targets(
            sequence.human_joints_full[:, person_index],
            sequence.wrist_quat_xyzw[:, person_index],
            max_calibration_error_deg=args.max_calibration_error_deg,
        )
        palm_targets.append(targets)
        calibration_errors.append(errors)
        LOGGER.info(
            "Person %d calibration p90: left=%.9f deg, right=%.9f deg",
            person_index + 1,
            errors[0],
            errors[1],
        )

    if scaled_preview:
        from holosoma_retargeting.src.utils import create_uniformly_scaled_object_scene_xml

        # Start from the original-size scene, scaling the object exactly once.
        # The preview URDF is already scaled and is copied verbatim.
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(baseline_dir / "assets", output_dir / "assets")
        scene_xml_path = Path(create_uniformly_scaled_object_scene_xml(
            scene_xml_path,
            object_name,
            float(pair["shared_object_scale"]),
            output_dir / "assets" / f"g1_29dof_w_{object_name}.xml",
        ))
        object_urdf_path = output_dir / "assets" / object_urdf_path.name

    retargeter = _build_a1_retargeter(
        object_name=object_name,
        object_urdf_path=object_urdf_path,
        scene_xml_path=scene_xml_path,
        foot_links=foot_links,
        max_solver_error_deg=args.max_solver_error_deg,
        solver_mode=args.solver_mode,
    )
    spec_index_key = (
        "pt_wrist_qpos_indices"
        if args.solver_mode == "wrist-only"
        else "pt_full_arm_qpos_indices"
    )
    resolved_indices = np.sort(
        np.concatenate(
            [
                np.asarray(spec[spec_index_key], dtype=np.int64)
                for spec in retargeter._hand_orientation_specs
            ]
        )
    )
    allowed_qpos_indices = _allowed_qpos_indices(args.solver_mode)
    if not np.array_equal(resolved_indices, allowed_qpos_indices):
        raise RuntimeError(
            f"Resolved {args.solver_mode} qpos indices {resolved_indices.tolist()} "
            f"differ from the frozen contract {allowed_qpos_indices.tolist()}"
        )

    refined_qpos: list[np.ndarray] = []
    solver_diagnostics: list[dict[str, np.ndarray | float | int]] = []
    for person_index in range(2):
        refined, diagnostics = _apply_refinement_solver(
            retargeter,
            solver_mode=args.solver_mode,
            baseline_qpos=baseline_qpos[person_index],
            palm_targets=palm_targets[person_index],
        )
        _require_only_allowed_change(
            baseline_qpos[person_index],
            refined,
            allowed_qpos_indices=allowed_qpos_indices,
            solver_mode=args.solver_mode,
        )
        refined_qpos.append(refined)
        solver_diagnostics.append(diagnostics)
        errors = np.asarray(diagnostics["orientation_errors_deg"])
        LOGGER.info(
            "Person %d %s orientation errors: left max=%.9f deg, right max=%.9f deg",
            person_index + 1,
            args.solver_mode,
            errors[:, 0].max(),
            errors[:, 1].max(),
        )
    if not scaled_preview:
        require_shared_physical_object_qpos(refined_qpos, sequence.object_poses)
    _require_frozen_object(refined_qpos, pair["object_qpos"])

    output_dir.mkdir(parents=True, exist_ok=True)
    if not scaled_preview:
        shutil.copytree(baseline_dir / "assets", output_dir / "assets")
    person_result_paths: list[Path] = []
    for person_index in range(2):
        source_data = {key: value.copy() for key, value in person_baselines[person_index].items()}
        source_data["qpos"] = refined_qpos[person_index]
        source_data["a1_calibration_errors_deg"] = calibration_errors[person_index]
        source_data["a1_solver_errors_deg"] = solver_diagnostics[person_index][
            "orientation_errors_deg"
        ]
        source_data["a1_solver_mode"] = np.asarray(args.solver_mode)
        for name, value in solver_diagnostics[person_index].items():
            source_data[f"a1_refinement_{name}"] = np.asarray(value)
        person_dir = output_dir / f"person{person_index + 1}"
        person_dir.mkdir()
        result_suffix = _result_suffix(args.solver_mode)
        object_label = "shared_scaled" if scaled_preview else "fixed_object"
        result_path = (
            person_dir
            / f"person{person_index + 1}_{object_label}_{result_suffix}.npz"
        )
        np.savez_compressed(result_path, **source_data)
        person_result_paths.append(result_path)

    pair_output = {name: value.copy() for name, value in pair.items()}
    pair_output["robot_qpos"] = np.stack(
        [value[:, :ROBOT_QPOS_WIDTH] for value in refined_qpos],
        axis=1,
    )
    pair_output["a1_calibration_errors_deg"] = np.stack(calibration_errors)
    pair_output["a1_solver_errors_deg"] = np.stack(
        [
            np.asarray(values["orientation_errors_deg"])
            for values in solver_diagnostics
        ],
        axis=1,
    )
    pair_output["a1_solver_mode"] = np.asarray(args.solver_mode)
    if scaled_preview:
        provenance = json.loads(str(pair["provenance_json"].item()))
        provenance.update({
            "robot_reference": f"Stage1 nominal plus existing A1 {args.solver_mode} postprocess",
            "parent_baseline_pair": str(parent_pair_path),
            "parent_pair_sha256": _sha256(parent_pair_path),
            "training_ready": False,
        })
        pair_output["provenance_json"] = np.asarray(json.dumps(provenance, sort_keys=True))
    common_diagnostics = set.intersection(
        *(set(values) for values in solver_diagnostics)
    )
    for name in sorted(common_diagnostics):
        per_person = [np.asarray(values[name]) for values in solver_diagnostics]
        stack_axis = 1 if name in TIME_SERIES_DIAGNOSTICS else 0
        pair_output[f"a1_refinement_{name}"] = np.stack(
            per_person,
            axis=stack_axis,
        )
    pair_path = output_dir / "core4d_pair_reference.npz"
    np.savez_compressed(pair_path, **pair_output)

    manifest = dict(parent_manifest)
    solver_label = {
        "wrist-only": (
            "frozen_v4_fixed_object_baseline_plus_existing_A1_"
            "wrist_only_postprocess"
        ),
        "full-arm": "frozen_v4_fixed_object_baseline_plus_A1_full_arm_postprocess",
        "wrist-dominant": (
            "frozen_v4_fixed_object_baseline_plus_A1_"
            "wrist_dominant_surface_postprocess"
        ),
    }[args.solver_mode]
    if scaled_preview:
        solver_label = f"shared_scaled_Stage1_baseline_plus_existing_A1_{args.solver_mode}_postprocess"
        manifest.update({
            "robot": source_manifest["robot"],
            "input": str(canonical_path),
            "input_sha256": _sha256(canonical_path),
            "robot_reference": provenance["robot_reference"],
            "training_ready": False,
        })
    wrist_mode = {
        "wrist-only": "A1_demonstrated_palm_orientation_wrist_only",
        "full-arm": "A1_demonstrated_palm_orientation_full_arm",
        "wrist-dominant": "A1_demonstrated_palm_surface_wrist_dominant",
    }[args.solver_mode]
    motion_key = "wrist_motion" if args.solver_mode == "wrist-only" else "arm_motion"
    exactness: dict[str, object] = {
        "non_allowed_qpos_exact": True,
        "root_waist_legs_object_qpos_exact": True,
        "shared_object_qpos_exact": True,
        "collision_reoptimized": False,
    }
    if args.solver_mode == "wrist-only":
        exactness["non_wrist_qpos_exact"] = True
    manifest.update(
        {
            "parent_baseline_dir": str(baseline_dir),
            "baseline_kind": args.baseline_kind,
            "parent_manifest_sha256": _sha256(parent_manifest_path),
            "parent_pair_sha256": _sha256(parent_pair_path),
            "pair_reference": str(pair_path),
            "pair_sha256": _sha256(pair_path),
            "person_result_paths": [str(path) for path in person_result_paths],
            "object_urdf": str(output_dir / "assets" / object_urdf_path.name),
            "object_urdf_sha256": _sha256(output_dir / "assets" / object_urdf_path.name),
            "scene_xml": str(output_dir / "assets" / scene_xml_path.name),
            "solver": solver_label,
            "solver_mode": args.solver_mode,
            "wrist_mode": wrist_mode,
            "a1": {
                "solver_mode": args.solver_mode,
                "solver_configuration": _solver_config_snapshot(
                    retargeter,
                    args.solver_mode,
                ),
                "allowed_qpos_indices": allowed_qpos_indices.tolist(),
                "palm_landmarks": [
                    {"name": name, "smplx_index": index}
                    for name, index in CORE4D_A1_PALM_LANDMARKS
                ],
                "quaternion_convention": "canonical wrist xyzw",
                "max_calibration_error_deg": args.max_calibration_error_deg,
                "max_solver_error_deg": (
                    args.max_solver_error_deg
                    if args.solver_mode == "wrist-only"
                    else None
                ),
                "calibration_errors_deg": {
                    f"person{index + 1}": {
                        "left": float(errors[0]),
                        "right": float(errors[1]),
                    }
                    for index, errors in enumerate(calibration_errors)
                },
                "solver_errors_deg": {
                    f"person{index + 1}": _error_summary(
                        np.asarray(values["orientation_errors_deg"])
                    )
                    for index, values in enumerate(solver_diagnostics)
                },
                "solver_diagnostics": {
                    f"person{index + 1}": _diagnostics_manifest_summary(values)
                    for index, values in enumerate(solver_diagnostics)
                },
                motion_key: {
                    f"person{index + 1}": _joint_motion_summary(
                        baseline_qpos[index],
                        refined_qpos[index],
                        fps,
                        allowed_qpos_indices,
                    )
                    for index in range(2)
                },
                **exactness,
            },
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("A.1 paired reference written to %s", pair_path)
    return pair_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
