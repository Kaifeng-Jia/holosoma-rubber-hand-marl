#!/usr/bin/env python3
"""T14 round 6 orchestration (tmux, after run_r6.sh): evaluate + render C and C' (prior tau = 10) for every task (2 at a time
behind the memory gate); Push leg check; 3-D contact error (all rounds, round 6 added); Table shrink check; posture metrics
(round 5 C, round 6 C, round 6 C') for every level; side-by-side sheets with the torso and whole arms (person, round 5 C,
round 6 C, round 6 C'), Push and Table, 3 frames. Return codes in logs_system1/t14/r6/pipeline.json."""
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import r4_pipeline as rp

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "logs_system1/t14/r6"
rp.S = S
rp.log = dict(started=time.strftime("%F %T"), steps=[])
SO = rp.SO
OUT = ROOT / "results_system1/T14_retargeting/step6"
R5, R6 = ROOT / "exports_system1/T14_references/r5", ROOT / "exports_system1/T14_references/r6"
S5 = ROOT / "logs_system1/t14/r5"
P = "20231030_001_desk001_move_new_fps30.npz"


def main():
    while "ALL_DONE" not in (S / "run.log").read_text():
        time.sleep(15)
    rp.log["solves_done"] = time.strftime("%T")
    rp.save()
    jobs = [(t, v) for v in ("C", "Cp") for t in ("table", "push", "bin", "chair", "pull")]
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda tv: rp.sh(["bash", "system1/t14/finish_r6.sh", tv[0], tv[1]], f"fin_{tv[0]}_{tv[1]}"), jobs))
    for name, q in (("r6C", S / "single_push_C/push_new_qpos.npz"), ("r6Cp", S / "single_push_Cp/push_new_qpos.npz"), ("r5C", S5 / "single_push_C/push_new_qpos.npz")):
        rp.sh([SO, "-B", "system1/t14/push_legs.py", "--single", str(q), "--m", "1.0", "--out", str(OUT / f"push_legs/push_legs_{name}.json")], f"push_legs_{name}", 1.5)
    (OUT / "contact_error_3d").mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "results_system1/T14_retargeting/step5/contact_error_3d").glob("contact_error_3d*.csv"):
        shutil.copy(f, OUT / "contact_error_3d" / f.name)
    # Push / Pull rows of every earlier round again: the person's target now on the OMOMO capture table (CORE4D rows unchanged)
    rp.sh([SO, "-B", "system1/t14/recompute_contact_error.py", "--rounds", "round2", "round3", "comparisons", "round4A", "round4B", "round5C",
           "--tasks", "push", "pull", "--out", str(OUT / "contact_error_3d")], "contact_error_omomo_all_rounds", 1.5)
    rp.sh([SO, "-B", "system1/t14/recompute_contact_error.py", "--rounds", "round6C", "round6Cp", "--out", str(OUT / "contact_error_3d")], "contact_error_r6", 1.5)
    refs = []
    for v in ("C", "Cp"):
        for k in ("0.8", "0.9", "1.0"):
            kk = "k" + k.replace(".", "p")
            refs += ["--ref", f"r6{v}_k{k}={R6}/table_{kk}_{v}/{P}@{S}/table_{kk}_{v}@{k}"]
    (OUT / "diagnostics").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "results_system1/T14_retargeting/step5/diagnostics/table_shrink_check.json", OUT / "diagnostics/table_shrink_check.json")
    rp.sh([SO, "-B", "system1/t14/table_shrink_check.py", *refs, "--out", str(OUT / "diagnostics/table_shrink_check.json")], "shrink_r6", 1.5)
    prefs = []
    for name, d, v in (("r5C", S5, "C"), ("r6C", S, "C"), ("r6Cp", S, "Cp")):
        for t in ("table", "bin", "chair"):
            for k in ("0.8", "0.9", "1.0", "1.1", "1.2"):
                kk = "k" + k.replace(".", "p")
                if (d / f"{t}_{kk}_{v}/person2/run_record.json").exists():
                    prefs += ["--ref", f"{name}_{t}_k{k}={d}/{t}_{kk}_{v}"]
        for t in ("push", "pull"):
            prefs += ["--ref", f"{name}_{t}={d}/single_{t}_{v}"]
    rp.sh([SO, "-B", "system1/t14/posture_metrics.py", *prefs, "--out", str(OUT / "posture/posture.json")], "posture", 1.5)
    os.environ.update(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    rp.sh([SO, "-B", "system1/t14/verify_hand_axes.py", "--task", "push", "--wide", "--ref", f"第5轮C={S5}/single_push_C/push_new_qpos.npz@1.0",
           "--ref", f"第6轮C={S}/single_push_C/push_new_qpos.npz@1.0", "--ref", f"第6轮C'={S}/single_push_Cp/push_new_qpos.npz@1.0", "--out", str(OUT / "sidebyside/push.jpg")], "sbs_push", 4.5)
    rp.sh([SO, "-B", "system1/t14/verify_hand_axes.py", "--task", "table", "--wide", "--ref", f"第5轮C={R5}/table_k1p0_C/{P}@1.0",
           "--ref", f"第6轮C={R6}/table_k1p0_C/{P}@1.0", "--ref", f"第6轮C'={R6}/table_k1p0_Cp/{P}@1.0", "--out", str(OUT / "sidebyside/table.jpg")], "sbs_table", 4.5)
    rp.log["done"] = time.strftime("%T")
    rp.save()
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
