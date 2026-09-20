"""Explicit asset/config selection for reusable CORE4D paired MAPPO.

This module selects data and physics, not different learning algorithms.
The legacy small-table contract remains the default; new experiments must be
selected explicitly and keep their own references, assets and checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[4]
MOTION_ROOT = "holosoma/data/motions/g1_29dof/whole_body_tracking"
CHAIR_DATA_DIR = f"{MOTION_ROOT}/core4d_chair021_20231020_074_a1"
CHAIR_SOURCE_SHA256 = "1e1999a179655abaf577f78fa6eeea5a7ecd4b1e8ae86a1144118b4dca54679b"
CHAIR_RUNTIME_SHA256 = "9de64128209230229e40a675ed6b204e1c4b7655a81cbef773f76557a0350ba5"
CHAIR_MESH_SHA256 = "7fb3283f1b67c8d08bb1f68c51d15aab5e6c940b8571e704bca265701b776161"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Core4DPairExperiment:
    experiment_id: str
    object_name: str
    project: str
    scenario: str
    source_pair_sha256: str
    runtime_reference_file: str
    runtime_reference_sha256: str
    object_urdf_file: str
    object_urdf_sha256: str
    training_promotion_file: str
    training_promotion_sha256: str
    reference_frames: int
    reference_fps: int
    object_mass_kg: float
    material_static_dynamic_restitution: tuple[float, float, float]
    object_collider_type: str
    physics_hz: int
    control_hz: int
    allow_interaction_mesh: bool
    training_ready: bool

    @property
    def checkpoint_contract(self) -> dict[str, object] | None:
        if self.experiment_id == "smalltable":
            return None  # Retain the exact metadata of all existing checkpoints.
        return {
            "experiment_id": self.experiment_id,
            "object_name": self.object_name,
            "source_pair_sha256": self.source_pair_sha256,
            "runtime_reference_sha256": self.runtime_reference_sha256,
            "object_urdf_sha256": self.object_urdf_sha256,
            "training_promotion_sha256": self.training_promotion_sha256,
            "reference_frames": self.reference_frames,
            "reference_fps": self.reference_fps,
            "physics_contract": {
                "object_mass_kg": self.object_mass_kg,
                "material_static_dynamic_restitution": self.material_static_dynamic_restitution,
                "object_collider_type": self.object_collider_type,
                "physics_hz": self.physics_hz,
                "control_hz": self.control_hz,
            },
        }


def get_core4d_pair_experiment(name: str = "smalltable") -> Core4DPairExperiment:
    if name == "smalltable5kg_A":
        data_dir = f"{MOTION_ROOT}/core4d_smalltable5kg_A"
        promotion_file = f"{data_dir}/training_asset_manifest.json"
        promotion_path = PACKAGE_ROOT / promotion_file
        promotion = json.loads(promotion_path.read_text())
        old = get_core4d_pair_experiment("smalltable")
        runtime_file = f"{data_dir}/core4d_pair_runtime_fps50.npz"
        object_file = f"{data_dir}/desk001_5kg_training.urdf"
        expected = {
            "experiment_id": name, "source_pair_sha256": old.source_pair_sha256,
            "runtime_reference_sha256": old.runtime_reference_sha256,
            "training_object_urdf_sha256": _sha256(PACKAGE_ROOT / object_file),
            "interaction_artifact_sha256": _sha256(PACKAGE_ROOT / data_dir / "interaction_vectors_v1.npz"),
            "training_robot_urdf_sha256": _sha256(PACKAGE_ROOT / "holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"),
            "object_mass_kg": 5.0, "inertia_scale_from_20kg": .25,
            "object_material_static_dynamic_restitution": [.5, .5, 0.],
            "object_collider_type": "convex_hull", "reference_frames": 687, "reference_fps": 50,
            "contact_target": "not_used_for_smalltable_A", "allowed_reward_variant": "A",
        }
        if any(promotion.get(k) != v for k, v in expected.items()):
            raise ValueError("5kg table manifest differs from selected reference/physics/reward")
        if _sha256(PACKAGE_ROOT / runtime_file) != old.runtime_reference_sha256:
            raise ValueError("5kg table must preserve the original runtime reference exactly")
        if type(promotion.get("training_ready")) is not bool:
            raise ValueError("5kg table requires explicit training_ready")
        return Core4DPairExperiment(
            experiment_id=name, object_name="desk001", project="Core4DSmallTableA",
            scenario="core4d_smalltable5kg_bucket_A_transfer",
            source_pair_sha256=old.source_pair_sha256,
            runtime_reference_file=runtime_file, runtime_reference_sha256=old.runtime_reference_sha256,
            object_urdf_file=object_file, object_urdf_sha256=expected["training_object_urdf_sha256"],
            training_promotion_file=promotion_file, training_promotion_sha256=_sha256(promotion_path),
            reference_frames=687, reference_fps=50, object_mass_kg=5.0,
            material_static_dynamic_restitution=(.5, .5, 0.), object_collider_type="convex_hull",
            physics_hz=200, control_hz=50, allow_interaction_mesh=False,
            training_ready=promotion["training_ready"],
        )
    if name == "smalltable":
        from holosoma.agents.mappo.core4d_smalltable_ppo import (
            CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
            CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
            CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
        )

        return Core4DPairExperiment(
            experiment_id=name, object_name="desk001", project="Core4DSmallTable",
            scenario="core4d_paired_small_table_reference_tracking",
            source_pair_sha256="d9d17e96f5f0b73aa76a1bf32f1ec50cfb7fa7fe78a847dd890882add4bbfd05",
            runtime_reference_file=f"{MOTION_ROOT}/core4d_smalltable/core4d_pair_runtime_fps50.npz",
            runtime_reference_sha256=CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
            object_urdf_file=f"{MOTION_ROOT}/objects_core4d_desk001_small_training.urdf",
            object_urdf_sha256=CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
            training_promotion_file=f"{MOTION_ROOT}/core4d_smalltable/training_asset_manifest.json",
            training_promotion_sha256=CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
            reference_frames=687, reference_fps=50, object_mass_kg=20.0,
            material_static_dynamic_restitution=(0.5, 0.5, 0.0),
            object_collider_type="convex_hull", physics_hz=200, control_hz=50,
            allow_interaction_mesh=True, training_ready=True,
        )
    if name == "bucket003":
        from holosoma.config_values.marl.g1.core4d_bucket_contract import (
            BUCKET_DATA_DIR, BUCKET_SOURCE_SHA256,
        )
        promotion_file = f"{BUCKET_DATA_DIR}/training_asset_manifest.json"
        promotion_path = PACKAGE_ROOT / promotion_file
        promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
        object_file = f"{BUCKET_DATA_DIR}/bucket003_training.urdf"
        runtime_file = f"{BUCKET_DATA_DIR}/core4d_pair_runtime_fps50.npz"
        expected = {
            "source_pair_sha256": BUCKET_SOURCE_SHA256,
            "runtime_reference_sha256": _sha256(PACKAGE_ROOT / runtime_file),
            "training_object_urdf_sha256": _sha256(PACKAGE_ROOT / object_file),
            "object_mesh_sha256": _sha256(PACKAGE_ROOT / BUCKET_DATA_DIR / "bucket003_m.obj"),
            "object_mass_kg": 1.0,
            "object_material_static_dynamic_restitution": [0.5, 0.5, 0.0],
            "object_collider_type": "convex_decomposition",
        }
        if any(promotion.get(key) != value for key, value in expected.items()):
            raise ValueError("Bucket asset manifest differs from the selected reference/physics")
        if type(promotion.get("training_ready")) is not bool:
            raise ValueError("Bucket manifest requires an explicit training_ready boolean")
        return Core4DPairExperiment(
            experiment_id=name, object_name="bucket003", project="Core4DBucket",
            scenario="core4d_paired_bucket003_handover_reference_tracking",
            source_pair_sha256=BUCKET_SOURCE_SHA256,
            runtime_reference_file=runtime_file,
            runtime_reference_sha256=expected["runtime_reference_sha256"],
            object_urdf_file=object_file,
            object_urdf_sha256=expected["training_object_urdf_sha256"],
            training_promotion_file=promotion_file,
            training_promotion_sha256=_sha256(promotion_path),
            reference_frames=497, reference_fps=50, object_mass_kg=1.0,
            material_static_dynamic_restitution=(0.5, 0.5, 0.0),
            object_collider_type="convex_decomposition", physics_hz=200, control_hz=50,
            allow_interaction_mesh=False, training_ready=promotion["training_ready"],
        )
    if name != "chair021":
        raise ValueError(f"Unknown CORE4D pair experiment: {name!r}")

    promotion_file = f"{CHAIR_DATA_DIR}/training_asset_manifest.json"
    promotion_path = PACKAGE_ROOT / promotion_file
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    object_file = f"{CHAIR_DATA_DIR}/chair021_training.urdf"
    object_hash = _sha256(PACKAGE_ROOT / object_file)
    mesh_hash = _sha256(PACKAGE_ROOT / CHAIR_DATA_DIR / "chair021_m.obj")
    if mesh_hash != CHAIR_MESH_SHA256:
        raise ValueError("Chair collision mesh differs from the reviewed geometry")
    expected = {
        "source_pair_sha256": CHAIR_SOURCE_SHA256,
        "runtime_reference_sha256": CHAIR_RUNTIME_SHA256,
        "training_object_urdf_sha256": object_hash,
        "object_mass_kg": 5.0,
        "object_material_static_dynamic_restitution": [0.5, 0.5, 0.0],
        "object_collider_type": "convex_decomposition",
        "object_mesh_sha256": mesh_hash,
    }
    if any(promotion.get(key) != value for key, value in expected.items()):
        raise ValueError("Chair asset manifest differs from the reviewed reference/physics")
    if type(promotion.get("training_ready")) is not bool:
        raise ValueError("Chair asset manifest requires an explicit training_ready boolean")
    return Core4DPairExperiment(
        experiment_id=name, object_name="chair021", project="Core4DChair",
        scenario="core4d_paired_chair021_reference_tracking",
        source_pair_sha256=CHAIR_SOURCE_SHA256,
        runtime_reference_file=f"{CHAIR_DATA_DIR}/core4d_pair_runtime_fps50.npz",
        runtime_reference_sha256=CHAIR_RUNTIME_SHA256,
        object_urdf_file=object_file, object_urdf_sha256=object_hash,
        training_promotion_file=promotion_file,
        training_promotion_sha256=_sha256(promotion_path),
        reference_frames=391, reference_fps=50, object_mass_kg=5.0,
        material_static_dynamic_restitution=(0.5, 0.5, 0.0),
        object_collider_type="convex_decomposition", physics_hz=200, control_hz=50,
        allow_interaction_mesh=False, training_ready=promotion["training_ready"],
    )


__all__ = ["Core4DPairExperiment", "get_core4d_pair_experiment"]
