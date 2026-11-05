from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from holosoma_retargeting.config_types.data_type import SMPLX_DEMO_JOINTS
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.data_utils.core4d_adapter import (
    CORE4D_HEIGHT_METHOD,
    Core4DPairSequence,
)
from holosoma_retargeting.data_utils.core4d_retarget import (
    prepare_core4d_person_retarget_input,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_SCRIPT = REPO_ROOT / "scripts" / "retarget_core4d_pair.py"


def _load_pair_script():
    spec = importlib.util.spec_from_file_location("retarget_core4d_pair", PAIR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair_sequence(frames: int = 5) -> Core4DPairSequence:
    human_joints = np.arange(
        frames * 2 * len(SMPLX_DEMO_JOINTS) * 3,
        dtype=np.float64,
    ).reshape(frames, 2, len(SMPLX_DEMO_JOINTS), 3)
    human_joints_full = np.arange(
        frames * 2 * 127 * 3,
        dtype=np.float64,
    ).reshape(frames, 2, 127, 3)
    wrist_quat_xyzw = np.zeros((frames, 2, 2, 4), dtype=np.float64)
    wrist_quat_xyzw[..., 3] = 1.0
    object_poses = np.zeros((frames, 7), dtype=np.float64)
    object_poses[:, 0] = 1.0
    object_poses[:, 4:] = np.column_stack(
        (
            np.arange(frames, dtype=np.float64),
            2.0 * np.arange(frames, dtype=np.float64),
            0.4 + 0.01 * np.arange(frames, dtype=np.float64),
        )
    )
    return Core4DPairSequence(
        human_joints=human_joints,
        human_joints_full=human_joints_full,
        betas=np.asarray([[0.0] * 10, [0.1] * 10], dtype=np.float64),
        human_heights=np.asarray([1.8, 1.7], dtype=np.float64),
        height_method=CORE4D_HEIGHT_METHOD,
        wrist_quat_xyzw=wrist_quat_xyzw,
        object_poses=object_poses,
        fps=30,
        joint_names=np.asarray(SMPLX_DEMO_JOINTS),
        aligned_frame_ids=np.column_stack(
            (
                np.arange(frames, dtype=np.int64),
                100 + np.arange(frames, dtype=np.int64),
            )
        ),
        object_name="desk001",
        object_mesh_path="/fixtures/desk/desk001_m.obj",
        provenance={
            "source_sequence": "fixture/sequence",
            "person_order": ["person1", "person2"],
        },
    )


def test_parse_args_accepts_defaults_and_explicit_valid_values() -> None:
    module = _load_pair_script()

    defaults = module.parse_args(
        ["--input", "source.npz", "--output-dir", "results"]
    )
    assert defaults.input == Path("source.npz")
    assert defaults.output_dir == Path("results")
    assert defaults.method == "two-stage"
    assert defaults.output_fps == 30
    assert defaults.max_frames is None
    assert defaults.human_heights is None
    assert defaults.robot_height == pytest.approx(1.32)
    assert defaults.retarget_scene_mass == pytest.approx(1.0)

    explicit = module.parse_args(
        [
            "--input",
            "source.npz",
            "--output-dir",
            "results",
            "--output-fps",
            "60",
            "--method",
            "omni",
            "--max-frames",
            "12",
            "--human-heights",
            "1.81",
            "1.72",
            "--robot-height",
            "1.3",
            "--retarget-scene-mass",
            "2.5",
        ]
    )
    assert explicit.method == "omni"
    assert explicit.output_fps == 60
    assert explicit.max_frames == 12
    assert explicit.human_heights == pytest.approx([1.81, 1.72])
    assert explicit.robot_height == pytest.approx(1.3)
    assert explicit.retarget_scene_mass == pytest.approx(2.5)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--output-fps", "0"],
        ["--max-frames", "-1"],
        ["--human-heights", "1.8", "0"],
        ["--human-heights", "1.8"],
        ["--robot-height", "nan"],
        ["--retarget-scene-mass", "inf"],
        ["--method", "unsupported"],
    ],
)
def test_parse_args_rejects_invalid_positive_parameters(extra_args: list[str]) -> None:
    module = _load_pair_script()
    base_args = ["--input", "source.npz", "--output-dir", "results"]

    with pytest.raises(SystemExit) as error:
        module.parse_args(base_args + extra_args)

    assert error.value.code == 2


def test_slice_sequence_slices_all_paired_time_fields_without_mutation() -> None:
    module = _load_pair_script()
    sequence = _pair_sequence()
    time_fields = (
        "human_joints",
        "human_joints_full",
        "wrist_quat_xyzw",
        "object_poses",
        "aligned_frame_ids",
    )
    snapshots = {name: np.asarray(getattr(sequence, name)).copy() for name in time_fields}
    fixed_snapshots = {
        "betas": sequence.betas.copy(),
        "human_heights": sequence.human_heights.copy(),
        "joint_names": sequence.joint_names.copy(),
        "provenance": json.loads(json.dumps(dict(sequence.provenance))),
    }

    sliced = module._slice_sequence(sequence, 3)

    assert sliced is not sequence
    for name in time_fields:
        source = np.asarray(getattr(sequence, name))
        result = np.asarray(getattr(sliced, name))
        assert len(result) == 3
        np.testing.assert_array_equal(result, source[:3])
        assert not np.shares_memory(result, source)
        np.testing.assert_array_equal(source, snapshots[name])
    np.testing.assert_array_equal(sliced.betas, sequence.betas)
    np.testing.assert_array_equal(sliced.human_heights, sequence.human_heights)
    np.testing.assert_array_equal(sliced.joint_names, sequence.joint_names)
    assert sliced.height_method == sequence.height_method
    assert sliced.fps == sequence.fps
    assert sliced.object_name == sequence.object_name
    assert sliced.object_mesh_path == sequence.object_mesh_path
    assert sliced.provenance == {
        **sequence.provenance,
        "retarget_prefix_frames": 3,
    }
    np.testing.assert_array_equal(sequence.betas, fixed_snapshots["betas"])
    np.testing.assert_array_equal(
        sequence.human_heights,
        fixed_snapshots["human_heights"],
    )
    np.testing.assert_array_equal(sequence.joint_names, fixed_snapshots["joint_names"])
    assert sequence.provenance == fixed_snapshots["provenance"]

    assert module._slice_sequence(sequence, None) is sequence
    with pytest.raises(ValueError, match="exceeds the available 5 frames"):
        module._slice_sequence(sequence, 6)


def test_person_input_is_prepared_before_smoke_prefix_is_sliced() -> None:
    module = _load_pair_script()
    sequence = _pair_sequence()
    joints = sequence.human_joints.copy()
    foot_indices = [
        SMPLX_DEMO_JOINTS.index("L_Foot"),
        SMPLX_DEMO_JOINTS.index("R_Foot"),
    ]
    joints[-1, 0, foot_indices, 2] = -10.0
    sequence = replace(sequence, human_joints=joints)

    full_input = prepare_core4d_person_retarget_input(sequence, 0)
    smoke_input = module._slice_person_input(full_input, 3)
    prefix_only_input = prepare_core4d_person_retarget_input(
        module._slice_sequence(sequence, 3),
        0,
    )

    assert smoke_input.ground_offset_m == full_input.ground_offset_m
    assert smoke_input.ground_offset_m != prefix_only_input.ground_offset_m
    np.testing.assert_array_equal(
        smoke_input.human_joints,
        full_input.human_joints[:3],
    )
    np.testing.assert_array_equal(
        smoke_input.nominal_object_poses,
        full_input.nominal_object_poses[:3],
    )
    assert not np.shares_memory(smoke_input.human_joints, full_input.human_joints)
    assert module._slice_person_input(full_input, None) is full_input


def test_require_empty_output_dir_accepts_only_new_or_empty_directories(
    tmp_path: Path,
) -> None:
    module = _load_pair_script()

    new_directory = tmp_path / "new" / "run"
    module._require_empty_output_dir(new_directory)
    assert new_directory.is_dir()

    module._require_empty_output_dir(new_directory)
    assert list(new_directory.iterdir()) == []

    existing_file = new_directory / "previous-result.npz"
    existing_file.write_bytes(b"do not mix runs")
    with pytest.raises(FileExistsError, match="Refusing to mix a CORE4D run"):
        module._require_empty_output_dir(new_directory)
    assert existing_file.read_bytes() == b"do not mix runs"

    output_is_file = tmp_path / "not-a-directory"
    output_is_file.write_text("keep", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="Output path is not a directory"):
        module._require_empty_output_dir(output_is_file)
    assert output_is_file.read_text(encoding="utf-8") == "keep"


def test_retarget_person_uses_official_g1_foot_anchor_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_pair_script()
    sequence = _pair_sequence(frames=1)
    person_input = prepare_core4d_person_retarget_input(sequence, 0)

    captured: dict[str, object] = {}

    class FakeInteractionMeshRetargeter:
        def __init__(self, **kwargs) -> None:
            self.task_constants = kwargs.get("task_constants")
            self.apply_manual_joint_limit_overrides = kwargs.get(
                "apply_manual_joint_limit_overrides",
                True,
            )
            self.foot_sticking_tolerance = kwargs.get(
                "foot_sticking_tolerance",
            )
            self.anchor_nominal_foot_height = kwargs.get(
                "anchor_nominal_foot_height",
                False,
            )
            self.nominal_foot_height_tolerance = kwargs.get(
                "nominal_foot_height_tolerance",
            )
            self.elastic_constraints = kwargs.get("elastic_constraints")

    def fake_run_fixed_object_size_adaptation(**kwargs):
        captured["retargeter"] = kwargs["retargeter"]
        captured["stage1_config"] = kwargs["cfg"].retargeter
        result_path = tmp_path / "person1" / "person1_fixed_object.npz"
        np.savez(
            result_path,
            **{
                field: np.zeros(1, dtype=np.float64)
                for field in module.ELASTIC_DIAGNOSTIC_FIELDS
            },
        )
        return np.zeros((1, 43), dtype=np.float64), result_path

    robot_retarget_module = ModuleType(
        "holosoma_retargeting.examples.robot_retarget"
    )

    def fake_create_task_constants(robot_config, *_args, **_kwargs):
        captured["robot_config"] = robot_config
        return SimpleNamespace(
            NOMINAL_TRACKING_INDICES=(
                robot_config.NOMINAL_TRACKING_INDICES.copy()
            ),
        )

    robot_retarget_module.create_task_constants = fake_create_task_constants
    robot_retarget_module.setup_object_data = lambda *_args, **_kwargs: (
        np.zeros((4, 3), dtype=np.float64),
        np.zeros((4, 3), dtype=np.float64),
        str(tmp_path / "Box026.urdf"),
    )
    robot_retarget_module.build_retargeter_kwargs_from_config = (
        lambda config, constants, *_args, **_kwargs: {
            "task_constants": constants,
            "apply_manual_joint_limit_overrides": (
                config.apply_manual_joint_limit_overrides
            ),
            "foot_sticking_tolerance": config.foot_sticking_tolerance,
            "elastic_constraints": config.elastic_constraints,
        }
    )
    robot_retarget_module.convert_object_poses_to_mujoco_order = (
        lambda poses: np.asarray(poses)[:, [4, 5, 6, 0, 1, 2, 3]]
    )
    robot_retarget_module._compute_q_init_base = (
        lambda *_args, **_kwargs: np.zeros(36, dtype=np.float64)
    )
    robot_retarget_module.extract_foot_sticking_sequence_velocity = (
        lambda joints, _demo_joints, toe_names: [
            {toe_names[0]: False, toe_names[1]: False} for _ in range(len(joints))
        ]
    )
    robot_retarget_module.run_fixed_object_size_adaptation = (
        fake_run_fixed_object_size_adaptation
    )
    interaction_retargeter_module = ModuleType(
        "holosoma_retargeting.src.interaction_mesh_retargeter"
    )
    interaction_retargeter_module.InteractionMeshRetargeter = (
        FakeInteractionMeshRetargeter
    )
    monkeypatch.setitem(
        sys.modules,
        robot_retarget_module.__name__,
        robot_retarget_module,
    )
    monkeypatch.setitem(
        sys.modules,
        interaction_retargeter_module.__name__,
        interaction_retargeter_module,
    )

    module._retarget_person(
        sequence=sequence,
        person_input=person_input,
        output_dir=tmp_path,
        object_urdf_path=tmp_path / "Box026.urdf",
        scene_xml_path=tmp_path / "g1_29dof_w_Box026.xml",
        method="two-stage",
    )

    retargeter = captured["retargeter"]
    assert isinstance(retargeter, FakeInteractionMeshRetargeter)
    assert retargeter.apply_manual_joint_limit_overrides is False
    assert (
        retargeter.foot_sticking_tolerance
        == module.CORE4D_STAGE2_FOOT_XY_TOLERANCE_M
    )
    assert retargeter.anchor_nominal_foot_height is True
    assert retargeter.elastic_constraints.enable is True
    np.testing.assert_array_equal(
        retargeter.task_constants.NOMINAL_TRACKING_INDICES,
        np.arange(36, dtype=np.int64),
    )
    assert (
        retargeter.nominal_foot_height_tolerance
        == module.CORE4D_NOMINAL_FOOT_HEIGHT_TOLERANCE_M
    )
    robot_config = captured["robot_config"]
    assert robot_config.FOOT_STICKING_LINKS == RobotConfig(
        robot_type="g1"
    ).FOOT_STICKING_LINKS
    assert len(robot_config.FOOT_STICKING_LINKS) == 8
    stage1_config = captured["stage1_config"]
    assert stage1_config.apply_manual_joint_limit_overrides is True
    assert stage1_config.foot_sticking_tolerance == pytest.approx(1e-3)
    assert stage1_config.elastic_constraints.enable is False


def test_retarget_person_dispatches_methods_with_the_same_data_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_pair_script()
    sequence = _pair_sequence(frames=2)
    person_input = prepare_core4d_person_retarget_input(sequence, 0)
    physical_points = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )
    nominal_points = physical_points * person_input.human_to_robot_scale
    captures: dict[str, dict[str, object]] = {}

    def to_mujoco(poses: np.ndarray) -> np.ndarray:
        return np.asarray(poses)[:, [4, 5, 6, 0, 1, 2, 3]]

    class FakeInteractionMeshRetargeter:
        def __init__(self, **kwargs) -> None:
            self.demo_joints = list(SMPLX_DEMO_JOINTS)
            self.init_kwargs = kwargs

        def retarget_motion(self, **kwargs):
            captures["omni"] = kwargs
            qpos = np.zeros((2, 43), dtype=np.float64)
            qpos[:, -7:] = kwargs["object_poses_augmented"]
            return qpos, [], [], []

    def fake_run_fixed_object_size_adaptation(**kwargs):
        captures["two-stage"] = kwargs
        qpos = np.zeros((2, 43), dtype=np.float64)
        qpos[:, -7:] = to_mujoco(kwargs["physical_object_poses"])
        result_path = Path(kwargs["save_dir"]) / "person1_fixed_object.npz"
        np.savez(
            result_path,
            **{
                field: np.zeros(2, dtype=np.float64)
                for field in module.ELASTIC_DIAGNOSTIC_FIELDS
            },
        )
        return qpos, result_path

    robot_retarget_module = ModuleType(
        "holosoma_retargeting.examples.robot_retarget"
    )
    robot_retarget_module.create_task_constants = (
        lambda robot_config, *_args, **_kwargs: SimpleNamespace(
            FOOT_STICKING_LINKS=robot_config.FOOT_STICKING_LINKS,
            NOMINAL_TRACKING_INDICES=(
                robot_config.NOMINAL_TRACKING_INDICES.copy()
            ),
        )
    )
    robot_retarget_module.setup_object_data = lambda *_args, **_kwargs: (
        physical_points.copy(),
        nominal_points.copy(),
        str(tmp_path / "Box026.urdf"),
    )
    robot_retarget_module.build_retargeter_kwargs_from_config = (
        lambda *_args, **_kwargs: {}
    )
    robot_retarget_module.convert_object_poses_to_mujoco_order = to_mujoco
    robot_retarget_module._compute_q_init_base = (
        lambda *_args, **_kwargs: np.arange(36, dtype=np.float64)
    )
    robot_retarget_module.extract_foot_sticking_sequence_velocity = (
        lambda joints, _demo_joints, toe_names: [
            {toe_names[0]: True, toe_names[1]: True} for _ in range(len(joints))
        ]
    )
    robot_retarget_module.run_fixed_object_size_adaptation = (
        fake_run_fixed_object_size_adaptation
    )
    interaction_retargeter_module = ModuleType(
        "holosoma_retargeting.src.interaction_mesh_retargeter"
    )
    interaction_retargeter_module.InteractionMeshRetargeter = (
        FakeInteractionMeshRetargeter
    )
    monkeypatch.setitem(
        sys.modules,
        robot_retarget_module.__name__,
        robot_retarget_module,
    )
    monkeypatch.setitem(
        sys.modules,
        interaction_retargeter_module.__name__,
        interaction_retargeter_module,
    )

    results = {}
    metadata = {}
    for method in ("omni", "two-stage"):
        method_dir = tmp_path / method
        method_dir.mkdir()
        qpos, result_path, run_metadata = module._retarget_person(
            sequence=sequence,
            person_input=person_input,
            output_dir=method_dir,
            object_urdf_path=tmp_path / "Box026.urdf",
            scene_xml_path=tmp_path / "g1_29dof_w_Box026.xml",
            method=method,
        )
        results[method] = (qpos, result_path)
        metadata[method] = run_metadata

    omni = captures["omni"]
    two_stage = captures["two-stage"]
    np.testing.assert_array_equal(
        omni["human_joint_motions"],
        two_stage["human_joints"],
    )
    np.testing.assert_array_equal(
        omni["object_poses"],
        to_mujoco(two_stage["nominal_object_poses"]),
    )
    np.testing.assert_array_equal(
        omni["object_poses_augmented"],
        to_mujoco(two_stage["physical_object_poses"]),
    )
    np.testing.assert_array_equal(
        omni["object_points_local_demo"],
        two_stage["object_local_pts_demo"],
    )
    np.testing.assert_array_equal(
        omni["object_points_local"],
        two_stage["object_local_pts"],
    )
    assert omni["q_nominal_list"] is None
    assert omni["original"] is True
    assert captures.keys() == {"omni", "two-stage"}
    np.testing.assert_array_equal(
        two_stage["constants"].NOMINAL_TRACKING_INDICES,
        np.arange(36, dtype=np.int64),
    )
    assert results["omni"][1].name == "person1_omni_single_stage.npz"
    assert results["two-stage"][1].name == "person1_fixed_object.npz"
    for method in ("omni", "two-stage"):
        np.testing.assert_array_equal(
            results[method][0][:, -7:],
            to_mujoco(person_input.physical_object_poses),
        )
        assert len(metadata[method]["foot_anchor_links"]) == 8
        assert metadata[method]["object_sampling"]["actual_count"] == 2
    assert (
        metadata["omni"]["object_sampling"]["physical_points_sha256"]
        == metadata["two-stage"]["object_sampling"]["physical_points_sha256"]
    )
    assert (
        metadata["omni"]["retargeter_config"]["anchor_nominal_foot_height"]
        is False
    )
    assert (
        metadata["two-stage"]["retargeter_config"][
            "anchor_nominal_foot_height"
        ]
        is True
    )
    assert (
        metadata["omni"]["retargeter_config"][
            "apply_manual_joint_limit_overrides"
        ]
        is True
    )
    assert (
        metadata["two-stage"]["retargeter_config"][
            "apply_manual_joint_limit_overrides"
        ]
        is False
    )
    assert metadata["omni"]["retargeter_config"][
        "foot_xy_tolerance_m"
    ] == pytest.approx(1e-3)
    assert metadata["two-stage"]["retargeter_config"][
        "foot_xy_tolerance_m"
    ] == pytest.approx(module.CORE4D_STAGE2_FOOT_XY_TOLERANCE_M)
    assert metadata["two-stage"]["retargeter_config"][
        "stage1_foot_xy_tolerance_m"
    ] == pytest.approx(1e-3)
    assert metadata["two-stage"]["retargeter_config"][
        "stage2_profile"
    ] == "urdf_limits_elastic_object_foot_full_nominal_v2"
    assert metadata["omni"]["retargeter_config"][
        "stage2_nominal_tracking_indices"
    ] is None
    assert metadata["two-stage"]["retargeter_config"][
        "stage2_nominal_tracking_indices"
    ] == list(range(36))
    assert metadata["omni"]["retargeter_config"][
        "nominal_tracking_active_stages"
    ] == []
    assert metadata["two-stage"]["retargeter_config"][
        "nominal_tracking_active_stages"
    ] == [2]
    assert metadata["omni"]["retargeter_config"][
        "elastic_constraints"
    ]["enable"] is False
    assert metadata["two-stage"]["retargeter_config"][
        "elastic_constraints"
    ]["enable"] is True


def test_run_records_method_and_reproducible_shared_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_pair_script()
    source_path = tmp_path / "canonical.npz"
    mesh_path = tmp_path / "object.obj"
    source_path.write_bytes(b"canonical fixture")
    mesh_path.write_bytes(b"mesh fixture")
    sequence = replace(_pair_sequence(frames=2), object_mesh_path=str(mesh_path))
    generated_assets = SimpleNamespace(
        object_urdf_path=tmp_path / "object.urdf",
        scene_xml_path=tmp_path / "scene.xml",
        collision_strategy="source_mesh_visual_and_collision",
        collision_boxes=(),
    )
    calls: list[tuple[str, int]] = []

    def fake_retarget_person(*, person_input, method, **_kwargs):
        calls.append((method, person_input.person_index))
        is_two_stage = method == "two-stage"
        qpos = np.zeros((2, 43), dtype=np.float64)
        qpos[:, -7:] = sequence.object_poses[:, [4, 5, 6, 0, 1, 2, 3]]
        result_path = (
            tmp_path
            / f"person{person_input.person_index + 1}_{method}.npz"
        )
        return qpos, result_path, {
            "object_sampling": {
                "requested_count": 100,
                "actual_count": 72,
                "seed": 42,
                "physical_points_sha256": "shared-physical-hash",
                "nominal_points_sha256": (
                    f"nominal-person-{person_input.person_index + 1}"
                ),
            },
            "foot_anchor_links": RobotConfig(
                robot_type="g1"
            ).FOOT_STICKING_LINKS,
            "retargeter_config": {
                "q_a_init_idx": -7,
                "activate_joint_limits": True,
                "apply_manual_joint_limit_overrides": not is_two_stage,
                "activate_object_nonpenetration": True,
                "activate_foot_sticking": True,
                "penetration_tolerance_m": 0.001,
                "foot_xy_tolerance_m": (
                    0.001
                ),
                "elastic_constraints": {
                    "enable": is_two_stage,
                    "object_collision_weight": 1.0e5,
                    "foot_kinematics_weight": 1.0e4,
                },
                "step_size": 0.2,
                "w_nominal_tracking_init": 5.0,
                "nominal_tracking_tau": 1.0e6,
                "stage2_nominal_tracking_indices": (
                    list(range(36)) if is_two_stage else None
                ),
                "stage2_nominal_tracking_scope": (
                    "full_robot_qpos_soft_prior"
                    if is_two_stage
                    else "inactive"
                ),
                "nominal_tracking_active_stages": (
                    [2] if is_two_stage else []
                ),
                "anchor_nominal_foot_height": method == "two-stage",
                "nominal_foot_height_tolerance_m": 0.005,
                "stage1_apply_manual_joint_limit_overrides": True,
                "stage1_foot_xy_tolerance_m": 0.001,
                "stage2_profile": (
                    "urdf_limits_elastic_object_foot_full_nominal_v2"
                    if is_two_stage
                    else None
                ),
            },
        }

    monkeypatch.setattr(module, "load_canonical_core4d_sequence", lambda _path: sequence)
    monkeypatch.setattr(
        module,
        "create_core4d_retarget_assets",
        lambda *_args, **_kwargs: generated_assets,
    )
    monkeypatch.setattr(module, "_retarget_person", fake_retarget_person)

    manifests = {}
    for method in ("omni", "two-stage"):
        output_dir = tmp_path / f"output-{method}"
        args = module.parse_args(
            [
                "--input",
                str(source_path),
                "--output-dir",
                str(output_dir),
                "--method",
                method,
            ]
        )
        module.run(args)
        manifests[method] = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )

    assert calls == [
        ("omni", 0),
        ("omni", 1),
        ("two-stage", 0),
        ("two-stage", 1),
    ]
    omni = manifests["omni"]
    two_stage = manifests["two-stage"]
    assert omni["dataset"] == two_stage["dataset"] == "CORE4D"
    assert omni["method"] == "omni"
    assert two_stage["method"] == "two-stage"
    assert omni["solver"]["pipeline"] == "official_omniretarget_single_stage"
    assert omni["solver"]["stages"] == 1
    assert omni["solver"]["q_nominal_policy"] == "none"
    assert two_stage["solver"]["pipeline"] == (
        "nominal_scaled_then_physical_two_stage"
    )
    assert two_stage["solver"]["stages"] == 2
    assert two_stage["solver"]["q_nominal_policy"] == (
        "stage1_output_tracks_during_stage2"
    )
    assert omni["solver"]["apply_manual_joint_limit_overrides"] is True
    assert two_stage["solver"]["apply_manual_joint_limit_overrides"] is False
    assert omni["solver"]["foot_xy_tolerance_m"] == pytest.approx(1e-3)
    assert two_stage["solver"]["foot_xy_tolerance_m"] == pytest.approx(1e-3)
    assert two_stage["solver"]["stage1_foot_xy_tolerance_m"] == pytest.approx(
        1e-3
    )
    assert two_stage["solver"]["stage2_profile"] == (
        "urdf_limits_elastic_object_foot_full_nominal_v2"
    )
    assert omni["solver"]["stage2_nominal_tracking_indices"] is None
    assert two_stage["solver"]["stage2_nominal_tracking_indices"] == list(
        range(36)
    )
    assert omni["solver"]["nominal_tracking_active_stages"] == []
    assert two_stage["solver"]["nominal_tracking_active_stages"] == [2]
    assert omni["foot_anchor"]["links"] == two_stage["foot_anchor"]["links"]
    assert len(omni["foot_anchor"]["links"]) == 8
    assert omni["object_sampling"] == two_stage["object_sampling"]
    assert omni["object_sampling"]["actual_count_per_person"] == [72, 72]
    assert omni["object_sampling"]["physical_points_sha256"] == [
        "shared-physical-hash",
        "shared-physical-hash",
    ]
    for field in (
        "input_sha256",
        "object_mesh_sha256",
        "source_frames",
        "source_fps",
        "retarget_frames",
        "retarget_fps",
        "height_method",
        "human_heights_m",
        "human_to_robot_scales",
        "ground_offsets_m",
        "scale_anchor_policy",
        "scale_anchors_m",
        "shared_physical_object_qpos_exact",
    ):
        assert omni[field] == two_stage[field]
