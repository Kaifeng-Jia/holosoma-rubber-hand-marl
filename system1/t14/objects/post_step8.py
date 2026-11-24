#!/usr/bin/env python3
"""T14 step 8 after run_step8.py (tmux): per sequence the step-7-format levels.json (step8_levels_json.py) and the pictures +
video (render_box.py, one at a time: 4 GB peak), then the tables (step8_tables.py). Return codes in logs_system1/t14/step8/post.json."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SO = "/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python"
L = ROOT / "logs_system1/t14/step8"
OUT = ROOT / "results_system1/T14_retargeting/step8"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="2",
           __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
CLS = dict(smallboxb="b", largeboxb="b", suitcaseb="b", smallboxc="c", largeboxc="c", suitcasec="c", push="b", pull="b")
log = dict(started=time.strftime("%F %T"), steps=[])


def avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2


def sh(cmd, name, need):
    while avail() - need < 8.0:
        time.sleep(20)
    t0 = time.time()
    with open(L / "post" / f"{name}.log", "w") as f:
        rc = subprocess.call(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    log["steps"].append(dict(name=name, rc=rc, seconds=round(time.time() - t0, 1)))
    (L / "post.json").write_text(json.dumps(log, indent=1) + "\n")
    return rc


def main():
    (L / "post").mkdir(exist_ok=True)
    while "done" not in json.loads((L / "driver.json").read_text()):
        time.sleep(30)
    for tag, cls in CLS.items():
        d = L / f"levels/{tag}"
        if sh([SO, "-B", "system1/t14/objects/step8_levels_json.py", "--tag", tag, "--out", d], f"levels_{tag}", 1.0) or not (d / "levels.json").exists():
            continue
        lv = json.loads((d / "levels.json").read_text())
        (OUT / "levels" / tag).mkdir(parents=True, exist_ok=True)
        (OUT / "levels" / tag / "levels.json").write_text((d / "levels.json").read_text())
        sh([SO, "-B", "system1/t14/objects/render_box.py", "--solve", lv["solve_r0"], "--levels", d / "levels.json", "--cls", cls, "--out", OUT / "figures",
            "--video-out", ROOT / "exports_system1/T14_videos/step8"], f"render_{tag}", 4.3)
    with open(L / "post" / "tables.md", "w") as f:
        rc = subprocess.call([SO, "-B", "system1/t14/objects/step8_tables.py"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    log["steps"].append(dict(name="tables", rc=rc))
    log["done"] = time.strftime("%F %T")
    (L / "post.json").write_text(json.dumps(log, indent=1) + "\n")
    print("POST8_DONE", flush=True)


if __name__ == "__main__":
    main()
