from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refine_core4d_pair_a1.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("refine_core4d_pair_a1", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(frames: int = 3) -> dict[str, np.ndarray]:
    robot_qpos = np.arange(frames * 2 * 36, dtype=np.float64).reshape(frames, 2, 36)
    object_qpos = np.arange(frames * 7, dtype=np.float64).reshape(frames, 7)
    return {
        "robot_qpos": robot_qpos,
        "object_qpos": object_qpos,
        "fps": np.asarray(30, dtype=np.int64),
        "object_name": np.asarray("Box026"),
    }


def test_parse_args_keeps_existing_a1_thresholds() -> None:
    module = _load_script()

    args = module.parse_args(
        ["--baseline-dir", "v4", "--output-dir", "v5"]
    )

    assert args.baseline_dir == Path("v4")
    assert args.output_dir == Path("v5")
    assert args.canonical is None
    assert args.baseline_kind == "fixed-object"
    assert args.solver_mode == "wrist-only"
    assert args.max_calibration_error_deg == pytest.approx(1.0)
    assert args.max_solver_error_deg == pytest.approx(0.01)


def test_parse_args_accepts_explicit_full_arm_mode() -> None:
    module = _load_script()

    args = module.parse_args(
        [
            "--baseline-dir",
            "v4",
            "--output-dir",
            "full-arm",
            "--solver-mode",
            "full-arm",
        ]
    )

    assert args.solver_mode == "full-arm"


def test_parse_args_accepts_explicit_wrist_dominant_mode() -> None:
    module = _load_script()

    args = module.parse_args(
        [
            "--baseline-dir",
            "v4",
            "--output-dir",
            "wrist-dominant",
            "--solver-mode",
            "wrist-dominant",
        ]
    )

    assert args.solver_mode == "wrist-dominant"
    np.testing.assert_array_equal(
        module._allowed_qpos_indices("wrist-dominant"),
        module.FULL_ARM_QPOS_INDICES,
    )
    assert module._result_suffix("wrist-only") == "a1"
    assert module._result_suffix("full-arm") == "a1_full_arm"
    assert module._result_suffix("wrist-dominant") == "a1_wrist_dominant"


def test_pair_validation_and_person_qpos_preserve_shared_object() -> None:
    module = _load_script()
    pair = _pair()

    assert module._validate_pair_data(pair) == (3, 30, "Box026")
    person0 = module._compose_person_qpos(pair, 0)
    person1 = module._compose_person_qpos(pair, 1)

    np.testing.assert_array_equal(person0[:, :36], pair["robot_qpos"][:, 0])
    np.testing.assert_array_equal(person1[:, :36], pair["robot_qpos"][:, 1])
    np.testing.assert_array_equal(person0[:, 36:], pair["object_qpos"])
    np.testing.assert_array_equal(person1[:, 36:], pair["object_qpos"])


def test_wrist_only_contract_accepts_six_wrist_coordinates() -> None:
    module = _load_script()
    baseline = np.arange(4 * 43, dtype=np.float64).reshape(4, 43)
    refined = baseline.copy()
    refined[:, module.WRIST_QPOS_INDICES] += 0.25

    module._require_wrist_only_change(baseline, refined)

    refined[2, 25] += 1.0e-12
    with pytest.raises(RuntimeError, match="outside the six wrist joints"):
        module._require_wrist_only_change(baseline, refined)


def test_wrist_only_contract_rejects_object_change() -> None:
    module = _load_script()
    baseline = np.zeros((2, 43), dtype=np.float64)
    refined = baseline.copy()
    refined[0, 36] = 1.0e-12

    with pytest.raises(RuntimeError, match="outside the six wrist joints"):
        module._require_wrist_only_change(baseline, refined)


def test_full_arm_contract_allows_only_shoulder_elbow_and_wrist() -> None:
    module = _load_script()
    baseline = np.zeros((3, 43), dtype=np.float64)
    refined = baseline.copy()
    refined[:, module.FULL_ARM_QPOS_INDICES] = 0.25

    module._require_only_allowed_change(
        baseline,
        refined,
        allowed_qpos_indices=module._allowed_qpos_indices("full-arm"),
        solver_mode="full-arm",
    )

    refined[1, 21] = 1.0e-12
    with pytest.raises(RuntimeError, match="full-arm.*outside its allowed joints"):
        module._require_only_allowed_change(
            baseline,
            refined,
            allowed_qpos_indices=module.FULL_ARM_QPOS_INDICES,
            solver_mode="full-arm",
        )


def test_full_arm_contract_rejects_shared_object_change() -> None:
    module = _load_script()
    baseline = np.zeros((2, 43), dtype=np.float64)
    refined = baseline.copy()
    refined[0, 36] = 1.0e-12

    with pytest.raises(RuntimeError, match="outside its allowed joints"):
        module._require_only_allowed_change(
            baseline,
            refined,
            allowed_qpos_indices=module.FULL_ARM_QPOS_INDICES,
            solver_mode="full-arm",
        )


def _full_arm_diagnostics(module, frames: int) -> dict[str, np.ndarray | float]:
    return {
        "orientation_errors_deg": np.zeros((frames, 2)),
        "hand_position_errors_m": np.zeros((frames, 2)),
        "arm_corrections_rad": np.zeros((frames, 2, 7)),
        "arm_steps_rad": np.zeros((frames, 2, 7)),
        "correction_steps_rad": np.zeros((frames, 2, 7)),
        "arm_qpos_indices": module.FULL_ARM_QPOS_INDICES.reshape(2, 7),
        "solver_success": np.ones((frames, 2), dtype=bool),
        "solver_nfev": np.ones((frames, 2), dtype=np.int64),
        "solver_cost": np.zeros((frames, 2)),
        "orientation_error_max_deg": 2.0,
    }


def _wrist_dominant_diagnostics(
    module,
    frames: int,
) -> dict[str, np.ndarray | float]:
    diagnostics = _full_arm_diagnostics(module, frames)
    diagnostics.pop("hand_position_errors_m")
    diagnostics.update(
        {
            "palm_normal_errors_deg": np.zeros((frames, 2)),
            "finger_direction_errors_deg": np.zeros((frames, 2)),
            "surface_position_errors_m": np.zeros((frames, 2)),
            "link_origin_errors_m": np.zeros((frames, 2)),
        }
    )
    return diagnostics


def test_full_arm_diagnostics_enforce_shapes_and_indices() -> None:
    module = _load_script()
    diagnostics = _full_arm_diagnostics(module, frames=4)

    copied = module._validate_refinement_diagnostics(
        diagnostics,
        frames=4,
        solver_mode="full-arm",
    )
    np.testing.assert_array_equal(
        copied["arm_qpos_indices"],
        module.FULL_ARM_QPOS_INDICES.reshape(2, 7),
    )

    diagnostics["hand_position_errors_m"] = np.zeros((4, 1))
    with pytest.raises(ValueError, match="hand_position_errors_m must have shape"):
        module._validate_refinement_diagnostics(
            diagnostics,
            frames=4,
            solver_mode="full-arm",
        )


def test_full_arm_diagnostics_reject_wrong_joint_contract() -> None:
    module = _load_script()
    diagnostics = _full_arm_diagnostics(module, frames=2)
    diagnostics["arm_qpos_indices"] = np.arange(21, 35).reshape(2, 7)

    with pytest.raises(RuntimeError, match="reported qpos indices"):
        module._validate_refinement_diagnostics(
            diagnostics,
            frames=2,
            solver_mode="full-arm",
        )


def test_wrist_dominant_diagnostics_require_surface_metrics() -> None:
    module = _load_script()
    diagnostics = _wrist_dominant_diagnostics(module, frames=3)

    copied = module._validate_refinement_diagnostics(
        diagnostics,
        frames=3,
        solver_mode="wrist-dominant",
    )
    assert copied["surface_position_errors_m"].shape == (3, 2)

    del diagnostics["finger_direction_errors_deg"]
    with pytest.raises(ValueError, match="finger_direction_errors_deg must have shape"):
        module._validate_refinement_diagnostics(
            diagnostics,
            frames=3,
            solver_mode="wrist-dominant",
        )


def test_solver_dispatch_preserves_all_three_modes() -> None:
    module = _load_script()
    baseline = np.zeros((2, 43), dtype=np.float64)
    targets = np.zeros((2, 2, 3, 3), dtype=np.float64)

    class FakeRetargeter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def apply_pt_wrist_orientation_postprocess(self, qpos, palm):
            self.calls.append("wrist-only")
            return np.asarray(qpos).copy(), np.zeros((2, 2))

        def apply_pt_full_arm_orientation_postprocess(self, qpos, palm):
            self.calls.append("full-arm")
            return np.asarray(qpos).copy(), _full_arm_diagnostics(module, 2)

        def apply_pt_wrist_dominant_surface_postprocess(self, qpos, palm):
            self.calls.append("wrist-dominant")
            return np.asarray(qpos).copy(), _wrist_dominant_diagnostics(module, 2)

    retargeter = FakeRetargeter()
    wrist_result, wrist_diagnostics = module._apply_refinement_solver(
        retargeter,
        solver_mode="wrist-only",
        baseline_qpos=baseline,
        palm_targets=targets,
    )
    full_result, full_diagnostics = module._apply_refinement_solver(
        retargeter,
        solver_mode="full-arm",
        baseline_qpos=baseline,
        palm_targets=targets,
    )
    dominant_result, dominant_diagnostics = module._apply_refinement_solver(
        retargeter,
        solver_mode="wrist-dominant",
        baseline_qpos=baseline,
        palm_targets=targets,
    )

    assert retargeter.calls == ["wrist-only", "full-arm", "wrist-dominant"]
    np.testing.assert_array_equal(wrist_result, baseline)
    np.testing.assert_array_equal(full_result, baseline)
    np.testing.assert_array_equal(dominant_result, baseline)
    assert set(wrist_diagnostics) == {"orientation_errors_deg"}
    assert "hand_position_errors_m" in full_diagnostics
    assert "palm_normal_errors_deg" in dominant_diagnostics


def test_single_frame_motion_summary_has_zero_velocity() -> None:
    module = _load_script()
    baseline = np.zeros((1, 43), dtype=np.float64)
    refined = baseline.copy()
    refined[:, module.WRIST_QPOS_INDICES] = 0.1

    summary = module._wrist_motion_summary(baseline, refined, fps=30)

    assert summary["absolute_velocity_rad_s_p90"] == pytest.approx(0.0)
    assert summary["absolute_velocity_rad_s_max"] == pytest.approx(0.0)


def test_require_empty_output_dir_never_mixes_runs(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "v5"

    module._require_empty_output_dir(output)
    assert not output.exists()
    output.mkdir()
    module._require_empty_output_dir(output)
    (output / "keep.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to mix"):
        module._require_empty_output_dir(output)


def test_parse_args_shared_scaled_is_explicit_and_keeps_wrist_solver() -> None:
    module = _load_script()
    args = module.parse_args([
        "--baseline-dir", "preview", "--output-dir", "a1",
        "--baseline-kind", "shared-scaled-preview",
    ])
    assert args.baseline_kind == "shared-scaled-preview"
    assert args.solver_mode == "wrist-only"
    assert args.max_solver_error_deg == pytest.approx(0.01)


def _scaled_preview_fixture(module, tmp_path: Path):
    preview = tmp_path / "preview"
    (preview / "assets").mkdir(parents=True)
    pair = _pair()
    scales = np.array([0.78, 0.72])
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"human_to_robot_scales": scales.tolist()}))
    nominal_paths = []
    for index in range(2):
        obj = np.tile([0.1 + index * 0.02, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], (3, 1))
        nominal = np.concatenate([pair["robot_qpos"][:, index], obj], axis=1)
        path = tmp_path / f"person{index + 1}_nominal.npz"
        np.savez_compressed(path, qpos=nominal, fps=pair["fps"])
        nominal_paths.append(str(path))
    pair["object_qpos"] = np.tile([0.11, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], (3, 1))
    # Match the exporter's exact floating-point averaging, not a decimal literal.
    pair["object_qpos"][:, 0] = (0.1 + (0.1 + 0.02)) / 2
    pair["human_to_robot_scales"] = scales
    pair["shared_object_scale"] = np.asarray(scales.mean())
    np.savez_compressed(preview / "core4d_pair_reference.npz", **pair)
    urdf = preview / "assets" / "Box026.urdf"
    urdf.write_text('<robot name="Box026"/>')
    manifest = {
        "kind": "diagnostic_preview_not_training_asset",
        "robot_reference": "per-person Stage1 nominal_scaled qpos",
        "training_ready": False,
        "shared_object_scale": float(scales.mean()),
        "source_manifest": str(source),
        "source_manifest_sha256": module._sha256(source),
        "source_person_nominal": nominal_paths,
        "pair_sha256": module._sha256(preview / "core4d_pair_reference.npz"),
        "object_urdf_sha256": module._sha256(urdf),
    }
    return preview, manifest, pair


def test_shared_scaled_input_preserves_export_recipe(tmp_path: Path) -> None:
    module = _load_script()
    preview, manifest, pair = _scaled_preview_fixture(module, tmp_path)
    source = module._load_shared_scaled_source(preview, manifest, pair)
    assert source["human_to_robot_scales"] == [0.78, 0.72]
    qposes = [module._compose_person_qpos(pair, index) for index in range(2)]
    module._require_frozen_object(qposes, pair["object_qpos"])
    qposes[1][0, 36] += 1e-12
    with pytest.raises(RuntimeError, match="frozen baseline object"):
        module._require_frozen_object(qposes, pair["object_qpos"])


@pytest.mark.parametrize("change", ["hash", "scale", "robot", "object", "already_refined"])
def test_shared_scaled_input_rejects_mismatched_sources(tmp_path: Path, change: str) -> None:
    module = _load_script()
    preview, manifest, pair = _scaled_preview_fixture(module, tmp_path)
    if change == "hash":
        manifest["pair_sha256"] = "invalid"
    elif change == "scale":
        pair["shared_object_scale"] = np.asarray(0.75**2)
    elif change == "robot":
        pair["robot_qpos"][0, 0, 26] += 0.01
    elif change == "object":
        pair["object_qpos"][0, 0] += 0.01
    else:
        manifest["robot_reference"] = "already wrist refined"
    with pytest.raises(ValueError):
        module._load_shared_scaled_source(preview, manifest, pair)


def test_scaled_scene_changes_only_object_once_and_preserves_source(tmp_path: Path) -> None:
    from holosoma_retargeting.src.utils import create_uniformly_scaled_object_scene_xml
    import xml.etree.ElementTree as ET

    source = tmp_path / "original.xml"
    original = '''<mujoco><asset>
      <mesh name="robot_mesh" file="robot.obj" scale="1 1 1"/>
      <mesh name="chair_mesh" file="chair.obj" scale="1 1 1"/>
      </asset><worldbody>
      <body name="robot" pos="0 0 1"><geom mesh="robot_mesh"/></body>
      <body name="chair021" pos="1 2 3">
        <geom mesh="chair_mesh"/><inertial pos="0 1 0" mass="1" diaginertia="2 3 4"/>
      </body></worldbody></mujoco>'''
    source.write_text(original)
    output = tmp_path / "output.xml"
    create_uniformly_scaled_object_scene_xml(source, "chair021", 0.75, output)
    root = ET.parse(output).getroot()
    assert source.read_text() == original
    assert root.find(".//mesh[@name='robot_mesh']").get("scale") == "1 1 1"
    assert root.find(".//body[@name='robot']").get("pos") == "0 0 1"
    np.testing.assert_allclose(
        np.fromstring(root.find(".//mesh[@name='chair_mesh']").get("scale"), sep=" "),
        [0.75, 0.75, 0.75],
    )
    inertial = root.find(".//body[@name='chair021']/inertial")
    assert inertial.get("mass") == "1"
    np.testing.assert_allclose(np.fromstring(inertial.get("diaginertia"), sep=" "),
                               np.array([2, 3, 4]) * 0.75**2)
