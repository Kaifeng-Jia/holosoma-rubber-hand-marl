#!/usr/bin/env python3
"""T14 step 9g item 4 (Kevin 2026-11-26): side-by-side sheet, level grid and (optionally) video of the clips of a finished step-9
driver run, labelled with both verdicts (non-hand penetration on the actual collision shapes = "新量法", on the per-link boxes =
"旧量法"; rejudge_nonhand_exact.py). Same pictures as the driver's render (run_step9.Clip.render -> render_box.py: levels a, 1.0,
1.25, 1.5 and the first free level above 1.5 that have an evaluation; side-by-side at 1.0, else the lowest solved level); the
levels json is written next to the driver's as levels_both.json (the driver's levels.json is not touched). Render only, CPU.
--clearance-exact (step 9g second fix, Kevin 2026-11-26 10:29): "新量法" also judges the two-robot clearance on the actual collision
shapes (rejudge_clearance_exact.py: the eval's two-robot minimum distance >= -1 cm); "旧量法" = the eval's own verdict (both
on boxes). Default: as before.

    sonic_s1/bin/python -B system1/t14/objects/step9g_render_both.py --logdir logs_system1/t14/step9g --rejudge logs_system1/t14/rejudge_exact/step9g \
        --fig-out results_system1/T14_retargeting/step9g/figures_all22 [--video-out exports_system1/T14_videos/step9g_all22] [--no-video CLIP ...] CLIP ...
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
SO = "/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
           __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
SHORT = {"V0 gravity": "重力", "V0 support drift r0": "漂移 r0", "V0 support drift r1": "漂移 r1", "V0 non-hand penetration r0": "非手 r0",
         "V0 non-hand penetration r1": "非手 r1", "V0 robot-robot clearance": "两机间隙", "section-4 contact retention >= 0.99": "保持率",
         "main motion": "主运动"}


def tagm(m):
    return f"m{m:.2f}".replace(".", "p")


def verdict(app, failed):
    return "适用" if app else "不适用（" + "、".join(SHORT.get(x, x) for x in failed) + "）"


def one(clip, sm, a, video):
    from rejudge_nonhand_exact import replace_nonhand
    from step8_levels_json import conv
    d = a.logdir / clip
    rep = sm.get("report_levels") or []
    free = sm.get("first_free_above_1p5")
    out, both = {}, {}
    for m in rep:
        f = d / tagm(m) / "eval.json"
        if not f.exists():
            continue
        e = json.loads(f.read_text())
        x = json.loads((a.rejudge / clip / f"{tagm(m)}.json").read_text())
        nf = replace_nonhand(e["failed"], x["new"])
        if a.clearance_exact:
            from rejudge_clearance_exact import LIMIT_M, with_clearance
            nf = with_clearance(nf, e["robot_robot_min_distance_cm"] / 100.0 >= LIMIT_M - 1e-10)
        both[m] = dict(new=(not nf, nf), old=(e["applicable"], e["failed"]))
        lv = conv(e, m, d / tagm(m), json.loads((d / tagm(m) / "r0/run_record.json").read_text()))
        lv["verdict_lines"] = [f"新量法：{verdict(not nf, nf)}", f"旧量法：{verdict(e['applicable'], e['failed'])}"]
        out[f"{m:.2f}"] = lv
    if not out:
        return clip, None
    side = 1.0 if "1.00" in out else min(float(k) for k in out)
    b = both[side]
    status = f"{side:.2f} 倍：新量法{'适用' if b['new'][0] else '不适用'}，旧量法{'适用' if b['old'][0] else '不适用'}"  # failed items per level: level grid
    rec0 = json.loads((d / tagm(side) / "r0/run_record.json").read_text())
    e1 = json.loads((d / tagm(side) / "eval.json").read_text())
    src = Path(e1["command"].split("--source ")[1].split()[0])
    srec = json.loads((src / "run_record.json").read_text())
    lv = dict(tag=clip, role_status=status, applicable=bool(b["new"][0]), applicable_old_measure=bool(b["old"][0]), selected=None, a=sm.get("a"),
              first_free_level_above_1=free, first_free_label="1.5 以上第一个不相交档", report_levels=[float(k) for k in out if free is None or float(k) != free], levels=out,
              source_box_m=srec["target"]["size"], solve_r0=str(d / tagm(side) / "r0"), clip_start=rec0["clip_start"], side_level=side,
              verdicts={f"{m:.2f}": dict(new_applicable=v["new"][0], new_failed=v["new"][1], old_applicable=v["old"][0], old_failed=v["old"][1]) for m, v in both.items()})
    (d / "levels_both.json").write_text(json.dumps(lv, indent=1, ensure_ascii=False) + "\n")
    cmd = [SO, "-B", "system1/t14/objects/render_box.py", "--solve", d / tagm(side) / "r0", "--levels", d / "levels_both.json", "--cls", "a",
           "--clip-start", rec0["clip_start"], "--side-level", side, "--out", a.fig_out]
    if video:
        cmd += ["--video-out", a.video_out]
    with open(d / "render_both.log", "w") as f:
        rc = subprocess.call(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    return clip, rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--driver", default="driver.json")
    ap.add_argument("--rejudge", type=Path, required=True)
    ap.add_argument("--fig-out", type=Path, required=True)
    ap.add_argument("--video-out", type=Path, default=None)
    ap.add_argument("--no-video", nargs="*", default=[], help="clips rendered without video (video already made from the same solves)")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--clearance-exact", action="store_true", help="the new verdict also uses the exact two-robot clearance (step 9g second fix)")
    a = ap.parse_args()
    for k in ("logdir", "rejudge", "fig_out", "video_out"):
        v = getattr(a, k)
        if v is not None and not v.is_absolute():
            setattr(a, k, ROOT / v)
    drv = json.loads((a.logdir / a.driver).read_text())["summary"]
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        res = list(ex.map(lambda c: one(c, drv[c], a, a.video_out is not None and c not in a.no_video), a.clips))
    print(json.dumps(dict(res)))
    return 0 if all(rc == 0 for _, rc in res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
