#!/usr/bin/env python3
"""Export a shared small-table preview from a completed two-stage CORE4D run."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROBOT_QPOS_WIDTH = 36
OBJECT_QPOS_WIDTH = 7
NOMINAL_QPOS_WIDTH = ROBOT_QPOS_WIDTH + OBJECT_QPOS_WIDTH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        required=True,
        help="Completed --method two-stage output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for the diagnostic preview.",
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
        raise FileExistsError(f"Output directory is not empty: {path}")


def _load_nominal_qpos(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Nominal Stage 1 result does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted({"qpos", "fps"}.difference(archive.files))
        if missing:
            raise ValueError(f"{path} is missing keys: {missing}")
        qpos = np.asarray(archive["qpos"], dtype=np.float64).copy()
        fps_value = np.asarray(archive["fps"])
    if qpos.ndim != 2 or qpos.shape[1] != NOMINAL_QPOS_WIDTH or len(qpos) == 0:
        raise ValueError(
            f"{path} qpos must have shape (T, {NOMINAL_QPOS_WIDTH}), got {qpos.shape}"
        )
    if fps_value.shape != ():
        raise ValueError(f"{path} fps must be a scalar, got {fps_value.shape}")
    fps_scalar = fps_value.item()
    fps = int(fps_scalar)
    if not np.isfinite(fps_scalar) or float(fps_scalar) != float(fps) or fps <= 0:
        raise ValueError(f"{path} fps must be a positive finite integer")
    if not np.isfinite(qpos).all():
        raise ValueError(f"{path} qpos contains non-finite values")
    return qpos, fps


def _load_pair_metadata(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Source pair reference does not exist: {path}")
    required = {
        "robot_qpos",
        "object_qpos",
        "fps",
        "human_heights",
        "human_to_robot_scales",
        "scale_anchor",
        "object_name",
        "object_mesh_path",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path} is missing keys: {missing}")
        return {key: np.asarray(archive[key]).copy() for key in required}


def _scale_vector_attribute(
    element: ET.Element,
    attribute: str,
    scale: float,
    *,
    expected_size: int = 3,
) -> None:
    raw_value = element.get(attribute)
    if raw_value is None:
        return
    try:
        values = [float(value) for value in raw_value.split()]
    except ValueError as exc:
        raise ValueError(f"Invalid URDF {element.tag} {attribute}: {raw_value!r}") from exc
    if len(values) != expected_size or not np.isfinite(values).all():
        raise ValueError(
            f"URDF {element.tag} {attribute} must contain {expected_size} finite values"
        )
    element.set(attribute, " ".join(f"{value * scale:.12g}" for value in values))


def _write_scaled_preview_urdf(source: Path, output: Path, scale: float) -> None:
    try:
        tree = ET.parse(source)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid source object URDF: {source}") from exc
    root = tree.getroot()

    for origin in root.findall(".//origin"):
        _scale_vector_attribute(origin, "xyz", scale)
    for mesh in root.findall(".//mesh"):
        if mesh.get("scale") is None:
            mesh.set("scale", f"{scale:.12g} {scale:.12g} {scale:.12g}")
        else:
            _scale_vector_attribute(mesh, "scale", scale)
    for box in root.findall(".//box"):
        _scale_vector_attribute(box, "size", scale)

    # This is a fixed-mass kinematic preview.  Position scales with length and
    # rotational inertia therefore scales with length squared; mass is kept.
    for inertia in root.findall(".//inertia"):
        for attribute in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            raw_value = inertia.get(attribute)
            if raw_value is not None:
                value = float(raw_value)
                if not np.isfinite(value):
                    raise ValueError(f"URDF inertia {attribute} must be finite")
                inertia.set(attribute, f"{value * scale**2:.12g}")

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def export_small_table_pair(source_run_dir: Path, output_dir: Path) -> Path:
    source_run_dir = source_run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _require_empty_output_dir(output_dir)

    source_manifest_path = source_run_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest does not exist: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("method") != "two-stage":
        raise ValueError("Small-table export requires a completed --method two-stage run")

    pair_path = source_run_dir / "core4d_pair_reference.npz"
    nominal_paths = [
        source_run_dir / "person1" / "person1_nominal_scaled.npz",
        source_run_dir / "person2" / "person2_nominal_scaled.npz",
    ]
    pair = _load_pair_metadata(pair_path)
    qpos1, fps1 = _load_nominal_qpos(nominal_paths[0])
    qpos2, fps2 = _load_nominal_qpos(nominal_paths[1])
    if qpos1.shape != qpos2.shape:
        raise ValueError(f"Nominal qpos shapes differ: {qpos1.shape} versus {qpos2.shape}")
    pair_frames = int(np.asarray(pair["robot_qpos"]).shape[0])
    pair_fps = int(np.asarray(pair["fps"]).item())
    if len(qpos1) != pair_frames or fps1 != fps2 or fps1 != pair_fps:
        raise ValueError(
            "Nominal results and pair reference must have identical frame counts and fps"
        )

    object_name = str(np.asarray(pair["object_name"]).item())
    if Path(object_name).name != object_name or object_name in {"", ".", ".."}:
        raise ValueError(f"object_name must be a plain filename stem, got {object_name!r}")
    source_urdf = source_run_dir / "assets" / f"{object_name}.urdf"
    if not source_urdf.is_file():
        raise FileNotFoundError(f"Source object URDF does not exist: {source_urdf}")

    scales = np.asarray(pair["human_to_robot_scales"], dtype=np.float64)
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError(f"human_to_robot_scales must contain two positive values, got {scales}")
    shared_scale = float(np.mean(scales))

    object1 = qpos1[:, ROBOT_QPOS_WIDTH:]
    object2 = qpos2[:, ROBOT_QPOS_WIDTH:]
    quaternion_delta = float(np.max(np.abs(object1[:, 3:] - object2[:, 3:])))
    if quaternion_delta > 1e-12:
        raise ValueError(
            "The two nominal object rotations differ; a shared preview cannot be formed "
            f"(max abs difference {quaternion_delta:.9g})"
        )
    shared_object = object1.copy()
    shared_object[:, :3] = 0.5 * (object1[:, :3] + object2[:, :3])
    quaternion_norms = np.linalg.norm(shared_object[:, 3:], axis=1, keepdims=True)
    if np.any(quaternion_norms <= 0.0):
        raise ValueError("Nominal object trajectory contains a zero quaternion")
    shared_object[:, 3:] /= quaternion_norms
    robot_qpos = np.stack(
        (qpos1[:, :ROBOT_QPOS_WIDTH], qpos2[:, :ROBOT_QPOS_WIDTH]), axis=1
    )
    if not np.isfinite(robot_qpos).all() or not np.isfinite(shared_object).all():
        raise ValueError("Small-table pair contains non-finite values")

    provenance = {
        "kind": "diagnostic_preview_not_training_asset",
        "robot_reference": "per-person Stage1 nominal_scaled qpos",
        "scale_choice": "arithmetic_mean_balancing_two_person_scales",
        "shared_object_pose": (
            "arithmetic mean of the two person-specific nominal translations; "
            "common rotation"
        ),
        "source_pair": str(pair_path),
        "source_person_nominal": [str(path) for path in nominal_paths],
        "source_object_urdf": str(source_urdf),
        "training_ready": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pair = output_dir / "core4d_pair_reference.npz"
    np.savez_compressed(
        output_pair,
        robot_qpos=robot_qpos,
        object_qpos=shared_object,
        fps=np.asarray(fps1, dtype=np.int64),
        human_heights=np.asarray(pair["human_heights"], dtype=np.float64),
        human_to_robot_scales=scales,
        scale_anchor=np.asarray(pair["scale_anchor"], dtype=np.float64),
        object_name=np.asarray(object_name),
        object_mesh_path=np.asarray(pair["object_mesh_path"]),
        shared_object_scale=np.asarray(shared_scale, dtype=np.float64),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    output_urdf = output_dir / "assets" / f"{object_name}.urdf"
    _write_scaled_preview_urdf(source_urdf, output_urdf, shared_scale)

    manifest = {
        **provenance,
        "frames": len(robot_qpos),
        "fps": fps1,
        "object_name": object_name,
        "shared_object_scale": shared_scale,
        "source_human_to_robot_scales": scales.tolist(),
        "maximum_scale_mismatch_fraction": float(
            np.max(np.abs(scales - shared_scale) / shared_scale)
        ),
        "maximum_nominal_object_translation_disagreement_m": float(
            np.max(np.linalg.norm(object1[:, :3] - object2[:, :3], axis=1))
        ),
        "pair_reference": str(output_pair),
        "object_urdf": str(output_urdf),
        "pair_sha256": _sha256(output_pair),
        "object_urdf_sha256": _sha256(output_urdf),
        "mass_contract": (
            "unchanged retarget placeholder mass; preview only, not approved for training"
        ),
        "inertia_contract": (
            "fixed-mass geometric preview: inertia multiplied by scale^2"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_pair


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = export_small_table_pair(args.source_run_dir, args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
