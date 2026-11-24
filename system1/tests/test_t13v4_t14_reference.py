"""T13 v4 section 0 with the T14 round 6 C references: switch tests for system1/t13v4/t14_reference.py, the state files and
the cloud runner's rules. Run with the hssim_s1 interpreter (no simulator needed):

    hssim_s1/bin/python -m pytest -q system1/tests/test_t13v4_t14_reference.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src/holosoma", ROOT / "src/holosoma_retargeting", ROOT / "scripts"):
    sys.path.insert(0, str(p))
from system1.t13v4 import t14_reference as tr  # noqa: E402

W = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/system1_t14r6C"


def test_table_experiment_switch():
    from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment
    old = get_core4d_pair_experiment("smalltable5kg_A")
    same, rec = tr.table_experiment(old, None)
    assert same is old and rec is None  # switch off: unchanged
    new, rec = tr.table_experiment(old, W / "table/t14_reference.json")
    assert new.runtime_reference_sha256 == tr.sha(ROOT / "src/holosoma" / new.runtime_reference_file) != old.runtime_reference_sha256
    assert new.object_urdf_file == old.object_urdf_file and new.object_urdf_sha256 == old.object_urdf_sha256
    assert (new.reference_frames, new.reference_fps, new.object_mass_kg) == (old.reference_frames, old.reference_fps, old.object_mass_kg)
    assert new.checkpoint_contract != old.checkpoint_contract  # old checkpoints cannot be resumed / evaluated as new ones
    prom = json.loads((ROOT / "src/holosoma" / new.training_promotion_file).read_text())
    assert prom["training_ready"] is True and prom["runtime_reference_sha256"] == new.runtime_reference_sha256
    from holosoma.config_values.marl.g1.core4d_bucket_contract import build_bucket_reward_contract
    art = ROOT / "src/holosoma" / Path(new.training_promotion_file).parent / "interaction_vectors_v1.npz"
    c = build_bucket_reward_contract(art, "A_no_rel_no_height", experiment=new)
    assert c["runtime_reference_sha256"] == new.runtime_reference_sha256


def test_push_config_switch():
    from plan5_registered_placement import configure_push_placement
    from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_push_baseline
    cfg, prov = configure_push_placement(g1_29dof_plan5_push_baseline, ROOT / "system1/evidence/preflight_push_registered.json", skill="push", root=ROOT)
    same, prov0 = tr.push_config(cfg, None, prov)
    assert same is cfg and prov0 is prov
    new, prov1 = tr.push_config(cfg, W / "push/t14_reference.json", prov)
    for name in ("setup_terms", "reset_terms", "step_terms"):
        t_old, t_new = getattr(cfg.command, name)["paired_motion_command"], getattr(new.command, name)["paired_motion_command"]
        assert t_new.params["lateral_spacing_m"] == t_old.params["lateral_spacing_m"] == pytest.approx(0.8780472)
        assert t_new.params["motion_config"].motion_file.endswith("sub6_largetable_033_t14r6C_mj_fps50_w_obj.npz")
        assert t_new.params["motion_config"].body_names_to_track == t_old.params["motion_config"].body_names_to_track
        assert t_new.params.get("paired_reference_file") is None
    assert new.robot.object.object_urdf_path == cfg.robot.object.object_urdf_path  # training table unchanged
    assert prov1 != prov and prov1["t14_reference"]["name"] == "t14_r6C_m1p00"
    with pytest.raises(ValueError):
        tr.push_config(cfg, W / "push/t14_reference.json", None)  # needs the registered placement


def test_check_labels():
    lab = ROOT / "system1/t13v4/labels_t14r6C"
    assert tr.check_labels(lab / "table_t14r6C_k1p0_labels.npz", W / "table/t14_reference.json", "table")["runtime_sha256"]
    assert tr.check_labels(lab / "push_t14r6C_m1p00_labels.npz", W / "push/t14_reference.json", "push")["runtime_sha256"]
    with pytest.raises(ValueError):
        tr.check_labels(lab / "push_t14r6C_m1p00_labels.npz", W / "table/t14_reference.json", "table")


def test_manifest_tamper(tmp_path):
    d = json.loads((W / "table/t14_reference.json").read_text())
    d["files"]["runtime_reference"]["sha256"] = "0" * 64
    p = tmp_path / "m.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        tr.load_manifest(p, "table")


def test_state_files():
    st = ROOT / "system1/t13v4/states_t14r6C"
    key = lambda s: json.dumps([s["robot_xy_m"], s["robot_yaw_rad"]])  # noqa: E731
    for case in ("table", "push"):
        recs = {n: json.loads((st / f"{case}_{n}_states.json").read_text()) for n in ("pm2", "pm8", "validation")}
        for n, r in recs.items():
            assert r["case"] == case and len(r["states"]) == 51 and r["states"][0]["kind"] == "nominal"
            assert r["system1_t13v4"]["t14_reference_manifest_sha256"] == tr.sha(W / f"{case}/t14_reference.json")
        assert recs["pm2"]["generation_seed"] == 241003 and recs["validation"]["generation_seed"] == 20261120
        for a, b in zip(recs["pm2"]["states"], recs["pm8"]["states"]):
            assert np.array_equal(np.asarray(b["robot_xy_m"]), 4.0 * np.asarray(a["robot_xy_m"]))
        test = {key(s) for n in ("pm2", "pm8") for s in recs[n]["states"] if s["kind"] != "nominal"}
        assert not any(key(s) in test for s in recs["validation"]["states"] if s["kind"] != "nominal")


def test_cloud_rules():
    from system1.t13v4 import cloud_runs as cr
    r2, r4 = cr.rules(2048), cr.rules(4096)
    assert (r2["save"], r2["val"], r2["stop_after"], r2["patience"], r2["max_it"], r2["earliest_stop"]) == (1000, 2000, 8000, 4000, 20000, 12000)
    assert (r4["save"], r4["val"], r4["stop_after"], r4["patience"], r4["max_it"], r4["earliest_stop"]) == (500, 1000, 4000, 2000, 10000, 6000)
    curve = {2000: 1, 4000: 3, 6000: 5, 8000: 7, 10000: 7}
    assert not cr.stop_decision(curve, r2)  # before 12000
    assert not cr.stop_decision({**curve, 12000: 8}, r2)  # improved
    assert cr.stop_decision({**curve, 12000: 7}, r2)  # 10000, 12000 <= best before (7 at 8000)
    assert not cr.stop_decision({2000: 0, 4000: 0, 6000: 0, 8000: 0, 10000: 0, 12000: 1}, r2)
    assert cr.stop_decision({1000: 2, 2000: 2, 3000: 4, 4000: 4, 5000: 3, 6000: 4}, r4)
