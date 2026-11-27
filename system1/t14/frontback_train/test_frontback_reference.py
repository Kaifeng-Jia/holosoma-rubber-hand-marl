#!/usr/bin/env python3
"""Smoke test of the front-back object-replacement switch (t14_reference.table_experiment) and the prepared assets.

    hssim_s1/bin/python -B system1/t14/frontback_train/test_frontback_reference.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/holosoma"))
sys.path.insert(0, str(ROOT))
W = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    from holosoma.config_values.marl.g1.core4d_bucket_contract import build_bucket_reward_contract
    from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment
    from system1.t13v4.t14_reference import table_experiment
    base = get_core4d_pair_experiment("smalltable5kg_A")
    # 1. switch off (the r6C Table manifest, no object_replacement): only reference identities change, as before
    old, rec = table_experiment(base, W / "system1_t14r6C/table/t14_reference.json")
    assert "object_replacement" not in rec
    assert (old.object_urdf_file, old.object_urdf_sha256, old.reference_frames, old.object_name) == \
        (base.object_urdf_file, base.object_urdf_sha256, base.reference_frames, base.object_name)
    # 2. switch on (front-back manifest): box URDF, its sha, 99 frames; mass, material, collider, rates = Table Transport
    new, rec = table_experiment(base, W / "system1_t14fb/table/t14_reference.json")
    man = json.loads((W / "system1_t14fb/table/t14_reference.json").read_text())
    ob = man["object_replacement"]
    assert new.object_urdf_file == ob["object_urdf"] and new.object_urdf_sha256 == sha(ROOT / "src/holosoma" / ob["object_urdf"])
    z = np.load(ROOT / "src/holosoma" / new.runtime_reference_file)
    assert new.reference_frames == len(z["agent_joint_pos"]) == 99
    for k in ("object_mass_kg", "material_static_dynamic_restitution", "object_collider_type", "physics_hz", "control_hz", "reference_fps", "experiment_id"):
        assert getattr(new, k) == getattr(base, k), k
    # 3. the bucket artifact is consistent with the replaced experiment (the training-time contract check)
    art = ROOT / "src/holosoma" / Path(new.training_promotion_file).parent / "interaction_vectors_v1.npz"
    c = build_bucket_reward_contract(art, "A_no_rel_no_height", experiment=new)
    assert c["runtime_reference_sha256"] == new.runtime_reference_sha256
    a = np.load(art)
    assert a["reference_body_points_world"].shape[0] == 99 and a["reference_object_points_world"].shape[0] == 99
    # 4. URDF: one box of the construction's level-1.0 size, 5 kg, uniform-density inertia
    import xml.etree.ElementTree as ET
    r = ET.parse(ROOT / "src/holosoma" / ob["object_urdf"]).getroot()
    size = [float(x) for x in r.find(".//collision/geometry/box").get("size").split()]
    assert np.allclose(size, ob["box_size_m"]) and float(r.find(".//mass").get("value")) == 5.0
    I = r.find(".//inertia")
    assert abs(float(I.get("ixx")) - 5 / 12 * (size[1] ** 2 + size[2] ** 2)) < 1e-9
    print("FRONTBACK_REFERENCE_SMOKE_OK", json.dumps(dict(frames=new.reference_frames, urdf=new.object_urdf_file, bucket_variant=c["variant"])))


if __name__ == "__main__":
    main()
