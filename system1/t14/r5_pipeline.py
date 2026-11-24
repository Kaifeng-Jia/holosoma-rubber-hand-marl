#!/usr/bin/env python3
"""T14 round 5 orchestration (tmux, after run_r5.sh): wait for the solves; evaluate + render every task (version C, close-ups
with the person / robot hands and their axis arrows; 2 at a time behind the memory gate of r4_pipeline); Push leg check (C);
3-D contact-position error for round 5 (added to step4's table, all rounds); Table shrink check (C); axis-arrow verification
sheets (Push, Table; C next to round 4 A / B). Return codes in logs_system1/t14/r5/pipeline.json."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import r4_pipeline as rp

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "logs_system1/t14/r5"
rp.S = S
rp.log = dict(started=time.strftime("%F %T"), steps=[])
SO = rp.SO
OUT = ROOT / "results_system1/T14_retargeting/step5"
R4, R5 = ROOT / "exports_system1/T14_references/r4", ROOT / "exports_system1/T14_references/r5"
P = "20231030_001_desk001_move_new_fps30.npz"


def main():
    while "ALL_DONE" not in (S / "run.log").read_text():
        time.sleep(15)
    rp.log["solves_done"] = time.strftime("%T")
    rp.save()
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda t: rp.sh(["bash", "system1/t14/finish_r5.sh", t, "C"], f"fin_{t}_C"), ("table", "push", "bin", "chair", "pull")))
    rp.sh([SO, "-B", "system1/t14/push_legs.py", "--single", str(S / "single_push_C/push_new_qpos.npz"), "--m", "1.0", "--out", str(OUT / "push_legs/push_legs_C.json")], "push_legs_C", 1.5)
    rp.sh([SO, "-B", "system1/t14/recompute_contact_error.py", "--rounds", "round5C", "--out", str(OUT / "contact_error_3d")], "contact_error_r5", 1.5)
    refs = []
    for k in ("0.8", "0.9", "1.0"):
        kk = "k" + k.replace(".", "p")
        refs += ["--ref", f"r5C_k{k}={R5}/table_{kk}_C/{P}@{S}/table_{kk}_C@{k}"]
    rp.sh([SO, "-B", "system1/t14/table_shrink_check.py", *refs, "--out", str(OUT / "diagnostics/table_shrink_check.json")], "shrink_r5", 1.5)
    env_note = "verify sheets need MUJOCO_GL=egl"
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    R4S = ROOT / "logs_system1/t14/r4"
    rp.sh([SO, "-B", "system1/t14/verify_hand_axes.py", "--task", "push", "--ref", f"C={S}/single_push_C/push_new_qpos.npz@1.0",
           "--ref", f"A={R4S}/single_push_A/push_new_qpos.npz@1.0", "--ref", f"B={R4S}/single_push_B/push_new_qpos.npz@1.0", "--out", str(OUT / "axes_check/push_axes.jpg")], "verify_push", 4.5)
    rp.sh([SO, "-B", "system1/t14/verify_hand_axes.py", "--task", "table", "--ref", f"C={R5}/table_k1p0_C/{P}@1.0", "--ref", f"A={R4}/table_k1p0_A/{P}@1.0",
           "--ref", f"B={R4}/table_k1p0_B/{P}@1.0", "--out", str(OUT / "axes_check/table_axes.jpg")], "verify_table", 4.5)
    rp.log["done"] = time.strftime("%T")
    rp.save()
    print("PIPELINE_DONE", env_note, flush=True)


if __name__ == "__main__":
    main()
