#!/usr/bin/env python3
"""T14 step 7 layer 1, after run_solves.py (tmux): per solve, one at a time (render peaks at 4 GB): role check -> levels
(+ references of the reported levels to exports_system1/T14_references/step7) -> pictures + video; then posture of all solves.
Return codes in <log>/post.json."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SO = "/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python"
S = ROOT / "logs_system1/t14/step7"
OUT = ROOT / "results_system1/T14_retargeting/step7"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="2",
           __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
log = dict(started=time.strftime("%F %T"), steps=[])


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2


def sh(cmd, name, need_gb):
    while avail_gb() - need_gb < 8.0:  # Kevin's rule: keep available - 8 GB for the rest
        time.sleep(20)
    t0 = time.time()
    with open(S / "post" / f"{name}.log", "w") as f:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    log["steps"].append(dict(name=name, rc=rc, seconds=round(time.time() - t0, 1), cmd=" ".join(map(str, cmd))))
    (S / "post" / "post.json").write_text(json.dumps(log, indent=1) + "\n")
    return rc


def main():
    (S / "post").mkdir(parents=True, exist_ok=True)
    while "ALL_DONE" not in (S / "solves/run.log").read_text():
        time.sleep(20)
    ch = json.loads((OUT / "layer1_choice.json").read_text())["choice"]
    tags = [(f"{e['object']}{cls}", cls) for cls in "bc" for e in ch[cls]]
    done = []
    for tag, cls in tags:
        solve = S / f"solves/single_{tag}_C"
        if not (solve / "run_record.json").exists():
            log["steps"].append(dict(name=tag, skipped="solve did not finish"))
            continue
        if sh([SO, "-B", "system1/t14/objects/role_check_box.py", "--solve", solve, "--out", S / f"role/{tag}"], f"role_{tag}", 1.0):
            continue
        lv = S / f"levels/{tag}"
        if lv.exists():
            lv.rename(ROOT / f"_to_delete/t14_step7/levels_{tag}_{time.strftime('%H%M%S')}")
        if sh([SO, "-B", "system1/t14/objects/dual_box.py", "--solve", solve, "--role", S / f"role/{tag}/role_check.json", "--out", lv,
               "--refs-out", ROOT / "exports_system1/T14_references/step7"], f"levels_{tag}", 2.3):
            continue
        sh([SO, "-B", "system1/t14/objects/render_box.py", "--solve", solve, "--levels", lv / "levels.json", "--cls", cls, "--out", OUT / "figures",
            "--video-out", ROOT / "exports_system1/T14_videos/step7/layer1"], f"render_{tag}", 4.2)
        done.append(tag)
    refs = []
    for tag in done:
        refs += ["--ref", f"{tag}={S / f'solves/single_{tag}_C'}"]
    sh([SO, "-B", "system1/t14/posture_metrics.py", *refs, "--out", OUT / "posture/posture.json"], "posture", 1.5)
    log["done"] = time.strftime("%F %T")
    (S / "post" / "post.json").write_text(json.dumps(log, indent=1) + "\n")
    print("POST_DONE", flush=True)


if __name__ == "__main__":
    main()
