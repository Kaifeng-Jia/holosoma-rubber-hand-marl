#!/usr/bin/env python3
"""Retarget one canonical CORE4D pair with two isolated OmniRetarget solves."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import replace
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
    Core4DPairSequence,
    load_canonical_core4d_sequence,
    resample_core4d_pair_sequence,
)
from holosoma_retargeting.data_utils.core4d_object_assets import (  # noqa: E402
    create_core4d_retarget_assets,
)
from holosoma_retargeting.data_utils.core4d_retarget import (  # noqa: E402
    G1_HEIGHT_M,
    Core4DPersonRetargetInput,
    prepare_core4d_person_retarget_input,
    require_shared_physical_object_qpos,
    save_core4d_pair_reference,
)


LOGGER = logging.getLogger(__name__)
BASE_G1_URDF = PACKAGE_ROOT / "models" / "g1" / "g1_29dof.urdf"
BASE_G1_XML = PACKAGE_ROOT / "models" / "g1" / "g1_29dof.xml"
RETARGET_METHODS = ("omni", "two-stage")
OBJECT_SAMPLE_REQUESTED_COUNT = 100
OBJECT_SAMPLE_SEED = 42
CORE4D_NOMINAL_FOOT_HEIGHT_TOLERANCE_M = 5e-3
CORE4D_STAGE2_FOOT_XY_TOLERANCE_M = 1e-3
CORE4D_STAGE2_ELASTIC_OBJECT_WEIGHT = 1.0e5
CORE4D_STAGE2_ELASTIC_FOOT_WEIGHT = 1.0e4
ELASTIC_DIAGNOSTIC_FIELDS = (
    "object_collision_slack_max_m",
    "foot_constraint_slack_max_m",
    "object_penetration_max_m",
    "foot_xy_deviation_max_m",
    "foot_height_deviation_max_m",
)


def _summarize_elastic_diagnostics(path: Path) -> dict[str, float]:
    """Read per-frame elastic diagnostics and return reproducible maxima."""
    with np.load(path, allow_pickle=False) as result:
        missing = [field for field in ELASTIC_DIAGNOSTIC_FIELDS if field not in result]
        if missing:
            raise RuntimeError(
                f"Elastic Stage 2 result is missing diagnostics: {missing}"
            )
        return {
            field: float(np.max(np.asarray(result[field], dtype=np.float64)))
            for field in ELASTIC_DIAGNOSTIC_FIELDS
        }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Canonical, pickle-free CORE4D paired NPZ.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for per-person and paired outputs.",
    )
    parser.add_argument(
        "--method",
        choices=RETARGET_METHODS,
        default="two-stage",
        help=(
            "Retargeting method: official OmniRetarget single-stage solve or "
            "the local nominal-to-physical two-stage adaptation."
        ),
    )
    parser.add_argument(
        "--output-fps",
        type=_positive_int,
        default=30,
        help="Unified solve rate; the default matches the existing OmniRetarget baseline.",
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        default=None,
        help="Optional prefix length for a smoke test after resampling.",
    )
    parser.add_argument(
        "--human-heights",
        type=_positive_float,
        nargs=2,
        metavar=("PERSON1_M", "PERSON2_M"),
        default=None,
        help="Optional explicit height overrides in meters.",
    )
    parser.add_argument(
        "--robot-height",
        type=_positive_float,
        default=G1_HEIGHT_M,
        help=f"G1 reference height in meters (default: {G1_HEIGHT_M}).",
    )
    parser.add_argument(
        "--retarget-scene-mass",
        type=_positive_float,
        default=1.0,
        help=(
            "Placeholder object mass used only to make the kinematic MuJoCo scene "
            "well formed; this is not an Isaac training mass."
        ),
    )
    return parser.parse_args(argv)


def _slice_sequence(
    sequence: Core4DPairSequence,
    max_frames: int | None,
) -> Core4DPairSequence:
    if max_frames is None:
        return sequence
    if max_frames > len(sequence.human_joints):
        raise ValueError(
            f"max_frames={max_frames} exceeds the available {len(sequence.human_joints)} frames"
        )
    provenance = dict(sequence.provenance)
    provenance["retarget_prefix_frames"] = max_frames
    return replace(
        sequence,
        human_joints=np.asarray(sequence.human_joints[:max_frames]).copy(),
        human_joints_full=np.asarray(sequence.human_joints_full[:max_frames]).copy(),
        wrist_quat_xyzw=np.asarray(sequence.wrist_quat_xyzw[:max_frames]).copy(),
        object_poses=np.asarray(sequence.object_poses[:max_frames]).copy(),
        aligned_frame_ids=np.asarray(sequence.aligned_frame_ids[:max_frames]).copy(),
        provenance=provenance,
    )


def _slice_person_input(
    person_input: Core4DPersonRetargetInput,
    max_frames: int | None,
) -> Core4DPersonRetargetInput:
    """Slice already-prepared data so smoke and full runs share preprocessing."""
    if max_frames is None:
        return person_input
    return replace(
        person_input,
        human_joints=np.asarray(person_input.human_joints[:max_frames]).copy(),
        human_joints_full=np.asarray(
            person_input.human_joints_full[:max_frames]
        ).copy(),
        wrist_quat_xyzw=np.asarray(
            person_input.wrist_quat_xyzw[:max_frames]
        ).copy(),
        nominal_object_poses=np.asarray(
            person_input.nominal_object_poses[:max_frames]
        ).copy(),
        physical_object_poses=np.asarray(
            person_input.physical_object_poses[:max_frames]
        ).copy(),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    """Hash an array together with its shape and dtype."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _require_empty_output_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a CORE4D run with existing files: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _retarget_person(
    *,
    sequence: Core4DPairSequence,
    person_input: Core4DPersonRetargetInput,
    output_dir: Path,
    object_urdf_path: Path,
    scene_xml_path: Path,
    method: str,
) -> tuple[np.ndarray, Path, dict[str, object]]:
    """Retarget one person using a selected wrapper around the same solver."""
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    from holosoma_retargeting.config_types.retargeter import RetargeterConfig
    from holosoma_retargeting.config_types.retargeting import RetargetingConfig
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.examples.robot_retarget import (
        build_retargeter_kwargs_from_config,
        convert_object_poses_to_mujoco_order,
        create_task_constants,
        _compute_q_init_base,
        extract_foot_sticking_sequence_velocity,
        run_fixed_object_size_adaptation,
        setup_object_data,
    )
    from holosoma_retargeting.src.interaction_mesh_retargeter import (
        InteractionMeshRetargeter,
    )
    if method not in RETARGET_METHODS:
        raise ValueError(
            f"method must be one of {RETARGET_METHODS}, got {method!r}"
        )

    person_name = f"person{person_input.person_index + 1}"
    person_dir = output_dir / person_name
    person_dir.mkdir(parents=True, exist_ok=False)

    robot_config = RobotConfig(
        robot_type="g1",
        robot_urdf_file=str(BASE_G1_URDF.resolve()),
    )
    stage2_nominal_tracking_indices: np.ndarray | None = None
    if method == "two-stage":
        stage2_nominal_tracking_indices = np.arange(
            7 + robot_config.ROBOT_DOF,
            dtype=np.int64,
        )
        robot_config = replace(
            robot_config,
            nominal_tracking_indices=stage2_nominal_tracking_indices.copy(),
        )
    motion_config = MotionDataConfig(data_format="smplx", robot_type="g1")
    task_config = TaskConfig(object_name=sequence.object_name)
    retargeter_config = RetargeterConfig(visualize=False, debug=False)
    active_retargeter_config = (
        replace(
            retargeter_config,
            apply_manual_joint_limit_overrides=False,
            foot_sticking_tolerance=(
                CORE4D_STAGE2_FOOT_XY_TOLERANCE_M
            ),
            elastic_constraints=replace(
                retargeter_config.elastic_constraints,
                enable=True,
                object_collision_weight=(
                    CORE4D_STAGE2_ELASTIC_OBJECT_WEIGHT
                ),
                foot_kinematics_weight=(
                    CORE4D_STAGE2_ELASTIC_FOOT_WEIGHT
                ),
            ),
        )
        if method == "two-stage"
        else retargeter_config
    )
    cfg = RetargetingConfig(
        task_type="object_interaction",
        robot="g1",
        data_format="smplx",
        task_name=person_name,
        save_dir=person_dir,
        fixed_object_size_adaptation=method == "two-stage",
        robot_config=robot_config,
        motion_data_config=motion_config,
        task_config=task_config,
        retargeter=retargeter_config,
    )
    constants = create_task_constants(
        robot_config,
        motion_config,
        task_config,
        "object_interaction",
    )
    # Stage 2 adapts the Stage 1 motion to the physical-size object.  The
    # two-stage RobotConfig above keeps that complete robot pose as a soft
    # prior so redundant arms cannot drift to a different kinematic branch.
    # Stage 1 is unchanged because it receives q_nominal_list=None.
    constants.OBJECT_MESH_FILE = sequence.object_mesh_path
    constants.OBJECT_URDF_FILE = str(object_urdf_path)
    constants.SCENE_XML_FILE = str(scene_xml_path)

    object_local_pts, object_local_pts_demo, configured_urdf_path = setup_object_data(
        "object_interaction",
        constants,
        None,
        person_input.human_to_robot_scale,
        task_config,
        False,
    )
    if object_local_pts is None or object_local_pts_demo is None:
        raise RuntimeError("CORE4D object point sampling unexpectedly returned no points")

    retargeter_kwargs = build_retargeter_kwargs_from_config(
        active_retargeter_config,
        constants,
        configured_urdf_path,
        "object_interaction",
    )
    retargeter_kwargs["nominal_tracking_tau"] = (
        active_retargeter_config.nominal_tracking_tau
    )
    retargeter_kwargs["anchor_nominal_foot_height"] = method == "two-stage"
    retargeter_kwargs["nominal_foot_height_tolerance"] = (
        CORE4D_NOMINAL_FOOT_HEIGHT_TOLERANCE_M
    )
    retargeter_kwargs["scene_xml_path"] = str(scene_xml_path)
    retargeter = InteractionMeshRetargeter(**retargeter_kwargs)

    run_metadata = {
        "object_sampling": {
            "requested_count": OBJECT_SAMPLE_REQUESTED_COUNT,
            "actual_count": int(len(object_local_pts)),
            "seed": OBJECT_SAMPLE_SEED,
            "physical_points_sha256": _array_sha256(object_local_pts),
            "nominal_points_sha256": _array_sha256(object_local_pts_demo),
        },
        "foot_anchor_links": list(robot_config.FOOT_STICKING_LINKS),
        "retargeter_config": {
            "q_a_init_idx": active_retargeter_config.q_a_init_idx,
            "activate_joint_limits": (
                active_retargeter_config.activate_joint_limits
            ),
            "apply_manual_joint_limit_overrides": (
                active_retargeter_config.apply_manual_joint_limit_overrides
            ),
            "activate_object_nonpenetration": (
                active_retargeter_config.activate_obj_non_penetration
            ),
            "activate_foot_sticking": (
                active_retargeter_config.activate_foot_sticking
            ),
            "penetration_tolerance_m": (
                active_retargeter_config.penetration_tolerance
            ),
            "foot_xy_tolerance_m": (
                active_retargeter_config.foot_sticking_tolerance
            ),
            "step_size": active_retargeter_config.step_size,
            "elastic_constraints": {
                "enable": (
                    active_retargeter_config.elastic_constraints.enable
                ),
                "object_collision_weight": (
                    active_retargeter_config.elastic_constraints.object_collision_weight
                ),
                "foot_kinematics_weight": (
                    active_retargeter_config.elastic_constraints.foot_kinematics_weight
                ),
            },
            "w_nominal_tracking_init": (
                active_retargeter_config.w_nominal_tracking_init
            ),
            "nominal_tracking_tau": (
                active_retargeter_config.nominal_tracking_tau
            ),
            "stage2_nominal_tracking_indices": (
                stage2_nominal_tracking_indices.tolist()
                if stage2_nominal_tracking_indices is not None
                else None
            ),
            "stage2_nominal_tracking_scope": (
                "full_robot_qpos_soft_prior"
                if stage2_nominal_tracking_indices is not None
                else "inactive"
            ),
            "nominal_tracking_active_stages": (
                [2] if stage2_nominal_tracking_indices is not None else []
            ),
            "anchor_nominal_foot_height": method == "two-stage",
            "nominal_foot_height_tolerance_m": (
                CORE4D_NOMINAL_FOOT_HEIGHT_TOLERANCE_M
            ),
            "stage1_apply_manual_joint_limit_overrides": (
                retargeter_config.apply_manual_joint_limit_overrides
            ),
            "stage1_foot_xy_tolerance_m": (
                retargeter_config.foot_sticking_tolerance
            ),
            "stage2_profile": (
                "urdf_limits_elastic_object_foot_full_nominal_v2"
                if method == "two-stage"
                else None
            ),
        },
    }

    if method == "two-stage":
        qpos, result_path = run_fixed_object_size_adaptation(
            cfg=cfg,
            constants=constants,
            data_format="smplx",
            task_name=person_name,
            save_dir=person_dir,
            smpl_scale=person_input.human_to_robot_scale,
            human_joints=person_input.human_joints,
            nominal_object_poses=person_input.nominal_object_poses,
            physical_object_poses=person_input.physical_object_poses,
            object_local_pts=object_local_pts,
            object_local_pts_demo=object_local_pts_demo,
            object_urdf_path=configured_urdf_path,
            toe_names=motion_config.toe_names,
            retargeter=retargeter,
        )
        run_metadata["elastic_diagnostics_max"] = (
            _summarize_elastic_diagnostics(result_path)
        )
        return qpos, result_path, run_metadata

    q_init = _compute_q_init_base(
        "object_interaction",
        "smplx",
        person_input.human_joints,
        person_input.nominal_object_poses,
        constants,
    )
    nominal_object_poses_mj = convert_object_poses_to_mujoco_order(
        person_input.nominal_object_poses
    )
    physical_object_poses_mj = convert_object_poses_to_mujoco_order(
        person_input.physical_object_poses
    )
    foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
        person_input.human_joints,
        retargeter.demo_joints,
        motion_config.toe_names,
    )
    foot_sticking_sequences[0][motion_config.toe_names[0]] = False
    foot_sticking_sequences[0][motion_config.toe_names[1]] = False
    result_path = person_dir / f"{person_name}_omni_single_stage.npz"
    qpos, _, _, _ = retargeter.retarget_motion(
        human_joint_motions=person_input.human_joints,
        object_poses=nominal_object_poses_mj,
        object_poses_augmented=physical_object_poses_mj,
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        foot_sticking_sequences=foot_sticking_sequences,
        q_a_init=q_init,
        q_nominal_list=None,
        original=True,
        dest_res_path=str(result_path),
    )
    return qpos, result_path, run_metadata


def run(args: argparse.Namespace) -> Path:
    source_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Canonical CORE4D artifact does not exist: {source_path}")
    for robot_asset in (BASE_G1_URDF, BASE_G1_XML):
        if not robot_asset.is_file():
            raise FileNotFoundError(f"Required rubber-hand G1 asset is missing: {robot_asset}")
    _require_empty_output_dir(output_dir)

    source_sequence = load_canonical_core4d_sequence(source_path)
    full_sequence = (
        resample_core4d_pair_sequence(source_sequence, args.output_fps)
        if source_sequence.fps != args.output_fps
        else source_sequence
    )
    sequence = _slice_sequence(full_sequence, args.max_frames)

    generated_assets = create_core4d_retarget_assets(
        sequence.object_mesh_path,
        BASE_G1_XML,
        sequence.object_name,
        output_dir / "assets",
        mass_kg=args.retarget_scene_mass,
    )
    height_overrides = args.human_heights or (None, None)
    full_person_inputs = [
        prepare_core4d_person_retarget_input(
            full_sequence,
            person_index,
            robot_height_m=args.robot_height,
            human_height_override_m=height_overrides[person_index],
        )
        for person_index in range(2)
    ]
    person_inputs = [
        _slice_person_input(person_input, args.max_frames)
        for person_input in full_person_inputs
    ]

    qpos_sequences: list[np.ndarray] = []
    result_paths: list[Path] = []
    person_run_metadata: list[dict[str, object]] = []
    for person_input in person_inputs:
        LOGGER.info(
            "Retargeting person %d with height %.6f m and scale %.9f",
            person_input.person_index + 1,
            person_input.human_height_m,
            person_input.human_to_robot_scale,
        )
        qpos, result_path, run_metadata = _retarget_person(
            sequence=sequence,
            person_input=person_input,
            output_dir=output_dir,
            object_urdf_path=generated_assets.object_urdf_path,
            scene_xml_path=generated_assets.scene_xml_path,
            method=args.method,
        )
        qpos_sequences.append(np.asarray(qpos, dtype=np.float64))
        result_paths.append(result_path)
        person_run_metadata.append(run_metadata)

    physical_point_hashes = [
        metadata["object_sampling"]["physical_points_sha256"]
        for metadata in person_run_metadata
    ]
    if len(set(physical_point_hashes)) != 1:
        raise RuntimeError(
            "The two independent solves did not receive the same physical "
            "object sample points"
        )

    require_shared_physical_object_qpos(qpos_sequences, sequence.object_poses)
    pair_path = save_core4d_pair_reference(
        output_dir / "core4d_pair_reference.npz",
        qpos_sequences=qpos_sequences,
        sequence=sequence,
        person_inputs=person_inputs,
    )
    manifest = {
        "dataset": "CORE4D",
        "method": args.method,
        "input": str(source_path),
        "input_sha256": _sha256(source_path),
        "source_frames": len(source_sequence.human_joints),
        "source_fps": source_sequence.fps,
        "retarget_frames": len(sequence.human_joints),
        "retarget_fps": sequence.fps,
        "object_name": sequence.object_name,
        "object_mesh_path": sequence.object_mesh_path,
        "object_mesh_sha256": _sha256(Path(sequence.object_mesh_path)),
        "object_collision": {
            "strategy": generated_assets.collision_strategy,
            "geometry_type": (
                "box_primitives"
                if generated_assets.collision_boxes
                else "source_mesh"
            ),
            "box_count": len(generated_assets.collision_boxes),
            "boxes": [
                {
                    "name": box.name,
                    "center_m": list(box.center),
                    "size_m": list(box.size),
                }
                for box in generated_assets.collision_boxes
            ],
        },
        "retarget_scene_mass_kg": args.retarget_scene_mass,
        "retarget_scene_mass_scope": "kinematic_retarget_only_not_training",
        "robot": "g1_29dof_rubber_hands",
        "solver": {
            "core": "InteractionMeshRetargeter",
            "pair_strategy": "two_independent_person_solves",
            "pipeline": (
                "official_omniretarget_single_stage"
                if args.method == "omni"
                else "nominal_scaled_then_physical_two_stage"
            ),
            "stages": 1 if args.method == "omni" else 2,
            "q_nominal_policy": (
                "none"
                if args.method == "omni"
                else "stage1_output_tracks_during_stage2"
            ),
            "sqp_iterations_first_frame": 50,
            "sqp_iterations_subsequent_frames": 10,
            **person_run_metadata[0]["retargeter_config"],
        },
        "foot_anchor": {
            "strategy": "official_g1_default_eight_surface_points",
            "links": person_run_metadata[0]["foot_anchor_links"],
            "nominal_height_tolerance_m": (
                CORE4D_NOMINAL_FOOT_HEIGHT_TOLERANCE_M
            ),
        },
        "object_sampling": {
            "requested_count": OBJECT_SAMPLE_REQUESTED_COUNT,
            "actual_count_per_person": [
                metadata["object_sampling"]["actual_count"]
                for metadata in person_run_metadata
            ],
            "seed": OBJECT_SAMPLE_SEED,
            "physical_points_sha256": physical_point_hashes,
            "nominal_points_sha256": [
                metadata["object_sampling"]["nominal_points_sha256"]
                for metadata in person_run_metadata
            ],
        },
        "elastic_diagnostics_max_per_person": [
            metadata.get("elastic_diagnostics_max")
            for metadata in person_run_metadata
        ],
        "wrist_mode": "position_only_baseline; canonical wrist quaternions preserved for later A1",
        "height_method": [value.human_height_method for value in person_inputs],
        "human_heights_m": [value.human_height_m for value in person_inputs],
        "human_to_robot_scales": [value.human_to_robot_scale for value in person_inputs],
        "ground_offsets_m": [value.ground_offset_m for value in person_inputs],
        "scale_anchor_policy": "initial_object_origin_projected_to_ground",
        "scale_anchors_m": [value.scale_anchor.tolist() for value in person_inputs],
        "shared_physical_object_qpos_exact": True,
        "person_result_paths": [str(path) for path in result_paths],
        "pair_reference": str(pair_path),
        "object_urdf": str(generated_assets.object_urdf_path),
        "scene_xml": str(generated_assets.scene_xml_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Paired CORE4D reference written to %s", pair_path)
    return pair_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
