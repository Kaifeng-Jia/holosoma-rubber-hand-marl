#!/usr/bin/env python3
"""T14: stack the two per-person solves of one variant into the compact 30 Hz pair (same fields as
core4d_smalltable/20231030_001_desk001_move_compact_fps30.npz), write its source manifest, and build the 50 Hz runtime
reference with the existing synthesizer (scripts/synthesize_core4d_smalltable_runtime_reference.py, custom-reference
mode: --expected-source-sha256 / --source-manifest / separate output directory).

    hsretargeting/bin/python -B system1/t14/table_combine.py --variant-dir logs_system1/t14/table/new --variant new \
        --out exports_system1/T14_references/table_k1p0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/20231030_001_desk001_move_compact_fps30.npz"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-dir", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cur = np.load(CURRENT, allow_pickle=False)
    qs, recs = [], []
    for p in (1, 2):
        z = np.load(a.variant_dir / f"person{p}/person{p}_{a.variant}_qpos.npz")
        qs.append(z["qpos"])
        recs.append(json.loads((a.variant_dir / f"person{p}/run_record.json").read_text()))
        obj = z["shared_object_qpos_mj"]
    assert qs[0].shape == qs[1].shape == (len(cur["robot_qpos"]), 43), [q.shape for q in qs]
    assert np.abs(qs[0][:, -7:] - qs[1][:, -7:]).max() == 0.0, "the two solves must share one object track"
    s_bar = recs[0]["s_bar"]
    assert abs(s_bar - float(cur["shared_object_scale"])) < 1e-12, (s_bar, float(cur["shared_object_scale"]))
    obj_cur_err = float(np.abs(obj - cur["object_qpos"]).max())  # same shared object track as the current reference
    prov = dict(kind="t14_step1_shared_object_resolve" if a.variant == "new" else "t14_omniretarget_as_is_shared_object", variant=a.variant,
                robot_reference=f"per-person re-solve around the shared object ({a.variant})", shared_object_scale=s_bar,
                person_run_records=[str(a.variant_dir / f"person{p}/run_record.json") for p in (1, 2)],
                person_qpos_sha256=[sha(a.variant_dir / f"person{p}/person{p}_{a.variant}_qpos.npz") for p in (1, 2)],
                object_track_vs_current_reference_max_abs=obj_cur_err)
    pair = a.out / f"20231030_001_desk001_move_{a.variant}_fps30.npz"
    np.savez(pair, robot_qpos=np.stack([q[:, :36] for q in qs], 1).astype(np.float64), object_qpos=obj.astype(np.float64), fps=np.int64(30),
             human_heights=cur["human_heights"], human_to_robot_scales=cur["human_to_robot_scales"], scale_anchor=cur["scale_anchor"],
             object_name=cur["object_name"], object_mesh_path=cur["object_mesh_path"], shared_object_scale=np.float64(s_bar),
             provenance_json=np.array(json.dumps(prov)))
    manifest = a.out / "source_manifest.json"
    manifest.write_text(json.dumps(dict(pair_sha256=sha(pair), **prov), indent=1) + "\n")
    runtime = a.out / "core4d_pair_runtime_fps50.npz"
    cmd = [sys.executable, "-B", str(ROOT / "scripts/synthesize_core4d_smalltable_runtime_reference.py"), "--source", str(pair), "--expected-source-sha256", sha(pair),
           "--source-manifest", str(manifest), "--output", str(runtime), "--manifest", str(a.out / "core4d_pair_runtime_fps50.manifest.json")]
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    rec = dict(pair=str(pair), pair_sha256=sha(pair), runtime=str(runtime), runtime_sha256=sha(runtime), command=cmd, object_track_vs_current_reference_max_abs=obj_cur_err)
    (a.out / "combine_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
