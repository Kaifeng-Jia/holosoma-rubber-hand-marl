#!/usr/bin/env python3
"""T14 step 9 driver (Kevin 2026-11-25 00:30; tmux, CPU only, at most --parallel worker processes, each started only when
MemAvailable - need >= 8 GB; no new start after --deadline, America/New_York).

Sequences: class-a clips with |heading - walking direction| < 45 deg at every frame (class_a_turns.csv "straight"); first the
three of Kevin's rule (per object the smallest |box yaw change|): smallbox sub4_smallbox_047, largebox sub7_largebox_045,
suitcase sub8_suitcase_006 ("main"). Per clip: source solve (role_solve --mode source --frames), pre-solve gates on the level grid
(step9_gates.py), role solves (role_solve --construction frontback, both roles) and the verdict (step9_eval.py) per level.
Every clip is solved whatever its gates say (the verdict lists the failed gate).

Main clips (Kevin item 2: levels a, 1.0, 1.25, 1.5; crowded -> lengthen along the walking axis only):
  * 1.0, 1.25, 1.5;
  * a (step-8 rule): only when 1.0 has no robot-robot intersection frame: 0.95, 0.90, ... (not below 0.55) until a level
    intersects; a = the last level without intersection;
  * when 1.5 still intersects: 1.55, 1.60, ... (at most 3.0) until the first level without intersection (fully evaluated);
  * levels.json (step-7 format) and pictures + video (render_box.py --cls a).
Extension (Kevin item 3, only if at least one main clip is applicable at one of its reported levels): every other straight
clip at 1.0 and at its smallest level without intersection on the same grid (1.0 clear -> the a search; else 1.25, 1.5, 1.55,
... <= 3.0). Record: logs_system1/t14/step9b/driver.json.

    sonic_s1/bin/python -B system1/t14/objects/run_step9.py --parallel 6 --deadline 08:15
Step 9c (Kevin 2026-11-25 01:52): --step step9c --fb-turn world --fb-legs walk --source-root logs_system1/t14/step9b/src
--no-extension (the three main clips only; the source solves of step 9b are reused: same clips, same source mode).
Step 9e (Kevin 2026-11-25 15:10): --step step9e --fb-turn world --fb-legs full_scopy --fb-human-frame target --skip-failed-up
--all-main --r1-source construction --source-root logs_system1/t14/step9b/src logs_system1/t14/step9d/src.
"""
import argparse
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
SO = "/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python"
HS = "/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python"
L = ROOT / "logs_system1/t14/step9b"
EXP = ROOT / "exports_system1/T14_references/step9"
VID = ROOT / "exports_system1/T14_videos/step9"
FIG = ROOT / "results_system1/T14_retargeting/step9b/figures"
TURNS = ROOT / "results_system1/T14_retargeting/step9/class_a_turns.csv"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
           __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
NY = ZoneInfo("America/New_York")
MAIN = {"sub4_smallbox_047": "smallboxa", "sub7_largebox_045": "largeboxa", "sub8_suitcase_006": "suitcasea"}
DOWN = [round(1.0 - 0.05 * i, 2) for i in range(1, 10)]  # 0.95 .. 0.55
UP = [round(1.5 + 0.05 * i, 2) for i in range(1, 31)]  # 1.55 .. 3.0
GRID = sorted(set(DOWN + [1.0, 1.25, 1.5] + UP))
log = dict(started=datetime.now(NY).isoformat(timespec="seconds"), steps=[], summary={})
lock = threading.Lock()
slots = None
DEADLINE = None
LOGFILE = "driver.json"
FB_ARGS = []
SOLVE_EXTRA = []
SRC_ROOT = None
EXTENSION = True
SKIP_FAILED_UP = False
ALL_MAIN = False
EVAL_EXTRA = []
NO_RENDER = False  # --no-render: no pictures / video from the driver (step 9g second fix: step9g_render_both.py afterwards, at most 2 renders)
T13_YIELD = None  # --t13-yield N: while a T13 post-run (local evaluation) is running, at most N workers of this driver run
RUNNING = [0]


def t13_running():
    """True while any results_system1/T13_runs/*/post_run_status.json says state "running" (T13 local evaluation first)."""
    for f in (ROOT / "results_system1/T13_runs").glob("*/post_run_status.json"):
        try:
            if json.loads(f.read_text()).get("state") == "running":
                return True
        except (OSError, ValueError):
            return True  # being written: treat as running
    return False


def now():
    return datetime.now(NY)


def avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2


def save():
    with lock:
        (L / LOGFILE).write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n")


def tagm(m):
    return f"m{m:.2f}".replace(".", "p")


def run(cmd, name, need, logf):
    """one worker process; None = not started (deadline)."""
    slots.acquire()
    try:
        while True:
            if now() >= DEADLINE:
                with lock:
                    log["steps"].append(dict(name=name, skipped="deadline", time=now().strftime("%T")))
                save()
                return None
            with lock:
                ok = avail() - need >= 8.0 and (T13_YIELD is None or RUNNING[0] < T13_YIELD or not t13_running())
                if ok:
                    RUNNING[0] += 1
                    rec = dict(name=name, cmd=" ".join(map(str, cmd)), started=now().strftime("%T"), avail_gb=round(avail(), 1))
                    log["steps"].append(rec)
                    logf.parent.mkdir(parents=True, exist_ok=True)
                    f = open(logf, "w")
                    p = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=ENV)
                    time.sleep(3)  # let it allocate before the next memory reading
            if ok:
                break
            time.sleep(10)
        save()
        rc = p.wait()
        with lock:
            RUNNING[0] -= 1
        f.close()
        rec.update(rc=rc, done=now().strftime("%T"))
        save()
        return rc
    finally:
        slots.release()


def parallel(jobs):
    """run (cmd, name, need, logf) jobs concurrently (each takes a slot); returns their return codes."""
    out = [None] * len(jobs)

    def go(i, j):
        out[i] = run(*j)
    ts = [threading.Thread(target=go, args=(i, j)) for i, j in enumerate(jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


class Clip:
    def __init__(self, row):
        self.obj, self.seq = row["object"], row["sequence"]
        self.frames = f"{row['start']}:{row['stop']}"
        self.tag = MAIN.get(self.seq, self.seq)
        self.d = L / self.tag
        # reuse a finished source solve (first root that has it); else solve it under L/src
        reuse = [r / self.tag for r in (SRC_ROOT or []) if (r / self.tag / "run_record.json").exists()]
        self.src = reuse[0] if reuse else L / "src" / self.tag
        self.main = self.seq in MAIN
        self.evals = {}
        self.sm = log["summary"].setdefault(self.tag, dict(object=self.obj, sequence=self.seq, frames=self.frames, main=self.main))

    def source(self):
        if (self.src / "run_record.json").exists():
            return True
        if self.src.exists():
            dst = ROOT / f"_to_delete/t14_step9b/src_{self.tag}_{time.strftime('%H%M%S')}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            self.src.rename(dst)
        rc = run([HS, "-B", "system1/t14/objects/role_solve.py", "--family", "box", "--object", self.obj, "--sequence", self.seq, "--mode", "source",
                  "--frames", self.frames, "--tag", self.tag, "--out", self.src], f"source_{self.tag}", 2.0, L / "src" / f"{self.tag}.log")
        return rc == 0 and (self.src / "run_record.json").exists()

    def gates(self):
        if all((self.d / f"gates_{tagm(m)}.json").exists() for m in GRID):
            return True
        rc = run([HS, "-B", "system1/t14/objects/step9_gates.py", "--object", self.obj, "--sequence", self.seq, "--frames", self.frames,
                  "--levels", ",".join(f"{m:.2f}" for m in GRID), "--out", self.d, *FB_ARGS], f"gates_{self.tag}", 1.5, self.d / "gates.log")
        return rc == 0

    def solve_jobs(self, m):
        d = self.d / tagm(m)
        jobs = []
        for r in (0, 1):
            if (d / f"r{r}" / "run_record.json").exists():
                continue
            if (d / f"r{r}").exists():
                dst = ROOT / f"_to_delete/t14_step9b/{self.tag}_{tagm(m)}_r{r}_{time.strftime('%H%M%S')}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                (d / f"r{r}").rename(dst)
            cmd = [HS, "-B", "system1/t14/objects/role_solve.py", "--family", "box", "--object", self.obj, "--sequence", self.seq, "--mode", "role",
                   "--construction", "frontback", "--role", r, "--m", m, "--frames", self.frames, "--tag", self.tag, "--out", d / f"r{r}", *FB_ARGS, *SOLVE_EXTRA]
            if r == 0:
                cmd += ["--source", self.src]
            jobs.append((cmd, f"solve_{self.tag}_{tagm(m)}_r{r}", 2.0, d / f"r{r}.log"))
        return jobs

    def eval_job(self, m):
        d = self.d / tagm(m)
        return ([SO, "-B", "system1/t14/objects/step9_eval.py", "--cand-dir", d, "--source", self.src, "--gates", self.d / f"gates_{tagm(m)}.json",
                 "--out", d / "eval.json", "--pair-out", EXP / f"{self.tag}_{tagm(m)}" / "pair_fps30.npz", *EVAL_EXTRA], f"eval_{self.tag}_{tagm(m)}", 2.5, d / "eval.log")

    def levels(self, ms):
        """solve + evaluate the levels (concurrently); returns {m: eval or None}."""
        ms = [m for m in ms if m not in self.evals]
        parallel([j for m in ms for j in self.solve_jobs(m)])
        ok = [m for m in ms if all((self.d / tagm(m) / f"r{r}" / "run_record.json").exists() for r in (0, 1))]
        parallel([self.eval_job(m) for m in ok if not (self.d / tagm(m) / "eval.json").exists()])
        for m in ms:
            f = self.d / tagm(m) / "eval.json"
            self.evals[m] = json.loads(f.read_text()) if f.exists() else None
            e = self.evals[m]
            self.sm.setdefault("levels", {})[f"{m:.2f}"] = None if e is None else dict(applicable=e["applicable"], failed=e["failed"],
                                                                                    intersect_frames=e["robot_robot_intersect_frames"], rr_cm=e["robot_robot_min_distance_cm"])
        save()
        return {m: self.evals[m] for m in ms}

    def inter(self, m):
        e = self.evals.get(m)
        return None if e is None else e["robot_robot_intersect_frames"] > 0

    def search_down(self):
        a = 1.0
        for m in DOWN:
            self.levels([m])
            x = self.inter(m)
            if x is None or x:
                break
            a = m
        return a

    def search_up(self, grid):
        for m in grid:
            self.levels([m])
            x = self.inter(m)
            if x is None:
                if SKIP_FAILED_UP:  # step 9c run 2: a failed solve says nothing about crowding; go on lengthening
                    continue
                return None
            if not x:
                return m
        return None

    def prepare(self):
        if not self.source():
            self.sm["error"] = "source solve failed or not started"
            save()
            return False
        if not self.gates():
            self.sm["error"] = "gates failed or not started"
            save()
            return False
        return True

    def run_main(self):
        try:
            self._run_main()
        except Exception:
            import traceback
            self.sm["exception"] = traceback.format_exc()[-2000:]
            save()
        finally:
            decide.update(self)

    def _run_main(self):
        if not self.prepare():
            return
        self.levels([1.0, 1.25, 1.5])
        decide.early(self)
        a = self.search_down() if self.inter(1.0) is False else None
        crowded = self.inter(1.5) or (SKIP_FAILED_UP and self.inter(1.5) is None and (self.inter(1.25) or self.inter(1.0)))
        free = self.search_up(UP) if crowded else None
        rep = sorted({1.0, 1.25, 1.5} | ({a} if a is not None else set()) | ({free} if free is not None else set()))
        self.sm.update(a=a, first_free_above_1p5=free, report_levels=rep,
                       applicable_levels=[m for m in rep if (self.evals.get(m) or {}).get("applicable")])
        self.sm["min_free"] = min((m for m in self.evals if self.inter(m) is False), default=None)
        save()
        decide.update(self)
        if (self.main or not ALL_MAIN) and not NO_RENDER:  # step 9e (--all-main): pictures + video of the three debug clips only
            self.render(rep, a, free)
        self.done = True

    def run_ext(self):
        try:
            self._run_ext()
        except Exception:
            import traceback
            self.sm["exception"] = traceback.format_exc()[-2000:]
            save()

    def _run_ext(self):
        if not self.prepare():
            return
        self.levels([1.0])
        if self.inter(1.0) is False:
            mf = self.search_down()
        elif self.inter(1.0):
            mf = self.search_up([1.25, 1.5] + UP)
        else:
            mf = None
        self.sm.update(min_free=mf, applicable_levels=[m for m in sorted({1.0} | ({mf} if mf else set())) if (self.evals.get(m) or {}).get("applicable")])
        save()

    def render(self, rep, a, free):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from step8_levels_json import conv
        out = {}
        for m in rep:
            e = self.evals.get(m)
            if e is not None:
                out[f"{m:.2f}"] = conv(e, m, self.d / tagm(m), json.loads((self.d / tagm(m) / "r0/run_record.json").read_text()))
        if not out:
            return
        side = 1.0 if "1.00" in out else min(float(k) for k in out)  # the side-by-side sheet: 1.0, else the lowest solved level
        e1 = self.evals[side]
        from step9_tables import ZH
        status = f"一前一后 {side:.2f} 倍：" + ("适用" if e1["applicable"] else "不适用（" + "、".join(ZH.get(x, x) for x in e1["failed"]) + "）")
        rec0 = json.loads((self.d / tagm(side) / "r0/run_record.json").read_text())
        srec = json.loads((self.src / "run_record.json").read_text())
        lv = dict(tag=self.tag, role_status=status, applicable=bool(e1["applicable"]), selected=None, a=a, first_free_level_above_1=free,
                  report_levels=[float(k) for k in out if free is None or float(k) != free], levels=out,
                  source_box_m=srec["target"]["size"], solve_r0=str(self.d / tagm(side) / "r0"), clip_start=rec0["clip_start"], side_level=side)
        (self.d / "levels.json").write_text(json.dumps(lv, indent=1, ensure_ascii=False) + "\n")
        rc = run([SO, "-B", "system1/t14/objects/render_box.py", "--solve", self.d / tagm(side) / "r0", "--levels", self.d / "levels.json", "--cls", "a",
                  "--clip-start", rec0["clip_start"], "--side-level", side, "--out", FIG, "--video-out", VID], f"render_{self.tag}", 4.3, self.d / "render.log")
        self.sm["render_rc"] = rc
        save()


class Decide:
    def __init__(self):
        self.ev = threading.Event()
        self.n = 0
        self.yes = False
        self.counted = set()

    def early(self, clip):
        """after 1.0 / 1.25 / 1.5: a main clip already applicable there decides the extension (later levels only add)."""
        if ALL_MAIN:
            return
        if any((clip.evals.get(m) or {}).get("applicable") for m in (1.0, 1.25, 1.5)):
            with lock:
                if not self.yes:
                    self.yes = True
                    log["extension_decision"] = dict(run=True, time=now().isoformat(timespec="seconds"), by=clip.tag, after="levels 1.0 / 1.25 / 1.5",
                                                     rule="at least one main clip applicable at one of its reported levels")
                self.ev.set()

    def update(self, clip):
        if ALL_MAIN:
            return
        with lock:
            if clip.tag in self.counted:
                return
            self.counted.add(clip.tag)
            self.n += 1
            if clip.sm.get("applicable_levels"):
                self.yes = True
            if (self.yes or self.n == len(MAIN)) and not self.ev.is_set():
                log["extension_decision"] = dict(run=self.yes, time=now().isoformat(timespec="seconds"),
                                                 rule="at least one main clip applicable at one of its reported levels")
                self.ev.set()


decide = Decide()


def render_only(tags, parallel):
    """pictures + video of main clips from the finished driver's summary (driver.json is read, not written; record in
    render_only.json): levels a, 1.0, 1.25, 1.5 and the first free level above 1.5 that have an evaluation."""
    global LOGFILE
    done = json.loads((L / "driver.json").read_text())["summary"]
    LOGFILE = "render_only.json"
    rows = {MAIN.get(r["sequence"], r["sequence"]): r for r in csv.DictReader(open(TURNS)) if r["straight"] == "True"}

    def one(tag):
        c = Clip(rows[tag])
        sm = done[tag]
        rep = [m for m in (sm.get("report_levels") or [1.0, 1.25, 1.5])]
        for m in rep:
            f = c.d / tagm(m) / "eval.json"
            c.evals[m] = json.loads(f.read_text()) if f.exists() else None
        c.render([m for m in rep if c.evals.get(m) is not None], sm.get("a"), sm.get("first_free_above_1p5"))
    ts = [threading.Thread(target=one, args=(t,)) for t in tags]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("RENDER_ONLY_DONE", flush=True)


def search_up_only(tags, start):
    """step 9c: continue the 'lengthen while crowded' search of main clips whose 1.5 level has no evaluation (a failed
    solve) and whose 1.0 / 1.25 levels intersect: levels start, start + 0.05, ... <= 3.0 until the first level without
    intersection (fully evaluated). Record in search_up.json (driver.json is read, not written)."""
    global LOGFILE
    LOGFILE = "search_up.json"
    rows = {MAIN.get(r["sequence"], r["sequence"]): r for r in csv.DictReader(open(TURNS)) if r["straight"] == "True"}
    grid = [m for m in UP if m >= start - 1e-9]

    def one(tag):
        c = Clip(rows[tag])
        free = c.search_up(grid)
        c.sm.update(search_from=start, first_free_above_1p5=free, min_free=free)
        save()
    ts = [threading.Thread(target=one, args=(t,)) for t in tags]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("SEARCH_UP_DONE", flush=True)


def main():
    global slots, DEADLINE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--deadline", default="08:15", help="HH:MM America/New_York today: no new worker after it")
    ap.add_argument("--render-only", nargs="*", default=None, help="tags of main clips: pictures + video only (after the driver)")
    ap.add_argument("--step", default="step9b", help="output name: logs_system1/t14/<step>, exports, results_system1/T14_retargeting/<step>")
    ap.add_argument("--fb-turn", choices=("box", "world"), default="box")
    ap.add_argument("--fb-legs", choices=("all", "walk", "walk_scopy", "full_scopy", "reverse_warp", "mirror"), default="all")
    ap.add_argument("--fb-walk-distance", choices=("pelvis", "pelvis_feet"), default="pelvis",
                    help="step9_gates / role_solve --fb-walk-distance (step 9g second fix: pelvis_feet); default = not passed")
    ap.add_argument("--source-root", type=Path, nargs="*", default=None, help="reuse the source solves under these directories (first that has the clip)")
    ap.add_argument("--all-main", action="store_true", help="step 9e: every straight clip (the 22) runs the main protocol (a, 1.0, 1.25, 1.5, "
                    "first free above 1.5), all at once, no extension gate; pictures + video for the three debug clips only")
    ap.add_argument("--r1-source", choices=("reversed", "construction"), default="reversed", help="step9_eval.py --r1-source (step 9e: construction)")
    ap.add_argument("--no-extension", action="store_true")
    ap.add_argument("--search-up", nargs="*", default=None, help="tags: continue the up-search from --search-from (step 9c)")
    ap.add_argument("--fb-human-frame", choices=("own", "target"), default="own", help="role_solve --fb-human-frame (step 9c run 2: target)")
    ap.add_argument("--skip-failed-up", action="store_true", help="the up-search goes on past a level whose solve failed, and starts "
                    "when 1.5 failed while 1.0 or 1.25 intersect (step 9c run 2)")
    ap.add_argument("--search-from", type=float, default=1.55)
    ap.add_argument("--no-render", action="store_true", help="no pictures / video from the driver (default: as before)")
    ap.add_argument("--turns", type=Path, default=None, help="final test after step 9g2 (Kevin 2026-11-26): clip list of other objects "
                    "(step9_turns.py --objects ...), every straight clip of it; needs --all-main (default: the 22 debug clips)")
    ap.add_argument("--t13-yield", type=int, default=None, help="while a T13 post-run is running (post_run_status.json state "
                    "'running'), start a worker only if fewer than N of this driver's are running (default: off)")
    a = ap.parse_args()
    global L, EXP, VID, FIG, FB_ARGS, SRC_ROOT, EXTENSION, SOLVE_EXTRA, SKIP_FAILED_UP, ALL_MAIN, EVAL_EXTRA, T13_YIELD, NO_RENDER
    T13_YIELD = a.t13_yield
    NO_RENDER = a.no_render
    SOLVE_EXTRA = ["--fb-human-frame", a.fb_human_frame] if a.fb_human_frame != "own" else []
    SKIP_FAILED_UP = a.skip_failed_up
    L = ROOT / f"logs_system1/t14/{a.step}"
    EXP = ROOT / f"exports_system1/T14_references/{a.step.replace('step9b', 'step9')}"
    VID = ROOT / f"exports_system1/T14_videos/{a.step.replace('step9b', 'step9')}"
    FIG = ROOT / f"results_system1/T14_retargeting/{a.step}/figures"
    FB_ARGS = ["--fb-turn", a.fb_turn, "--fb-legs", a.fb_legs] + (["--fb-walk-distance", a.fb_walk_distance] if a.fb_walk_distance != "pelvis" else [])
    SRC_ROOT = None if a.source_root is None else [r if r.is_absolute() else ROOT / r for r in a.source_root]
    EXTENSION = not a.no_extension
    ALL_MAIN = a.all_main
    EVAL_EXTRA = ["--r1-source", a.r1_source] if a.r1_source != "reversed" else []
    L.mkdir(parents=True, exist_ok=True)
    log.update(t13_yield=T13_YIELD, no_render=NO_RENDER, step=a.step, fb_args=FB_ARGS, solve_extra=SOLVE_EXTRA, eval_extra=EVAL_EXTRA, skip_failed_up=SKIP_FAILED_UP,
               source_root=None if SRC_ROOT is None else [str(r) for r in SRC_ROOT], extension_enabled=EXTENSION, all_main=ALL_MAIN)
    slots = threading.BoundedSemaphore(a.parallel)
    if a.render_only is not None:
        DEADLINE = now().replace(hour=23, minute=59)
        return render_only(a.render_only, a.parallel)
    hh, mm = (int(x) for x in a.deadline.split(":"))
    DEADLINE = now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    if a.search_up is not None:
        return search_up_only(a.search_up, a.search_from)
    log.update(parallel=a.parallel, deadline=DEADLINE.isoformat(timespec="seconds"), grid=GRID)
    turns = TURNS if a.turns is None else (a.turns if a.turns.is_absolute() else ROOT / a.turns)
    rows = [r for r in csv.DictReader(open(turns)) if r["straight"] == "True"]
    main_rows = [r for r in rows if r["sequence"] in MAIN]
    ext_rows = [r for r in rows if r["sequence"] not in MAIN]
    if a.turns is None:
        assert len(main_rows) == 3 and len(rows) == 22, (len(main_rows), len(rows))
    else:  # final test: other objects, no debug clip among them, the main protocol for every straight clip
        assert ALL_MAIN and not main_rows and rows, (ALL_MAIN, len(main_rows), len(rows))
        log.update(turns=str(turns), turns_sha256=__import__("hashlib").sha256(turns.read_bytes()).hexdigest())
    log["clips"] = [dict(object=r["object"], sequence=r["sequence"], frames=f"{r['start']}:{r['stop']}", main=r["sequence"] in MAIN) for r in rows]
    save()
    if ALL_MAIN:  # step 9e: the frozen construction on all 22 straight clips at once, main protocol for each
        ts = [threading.Thread(target=Clip(r).run_main) for r in main_rows + ext_rows]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        log["done"] = now().isoformat(timespec="seconds")
        save()
        print("STEP9_ALL_MAIN_DONE", flush=True)
        return
    ts = [threading.Thread(target=Clip(r).run_main) for r in main_rows]
    for t in ts:
        t.start()
    decide.ev.wait()
    ets = []
    if decide.yes and EXTENSION:
        ets = [threading.Thread(target=Clip(r).run_ext) for r in ext_rows]
        for t in ets:
            t.start()
    for t in ts + ets:
        t.join()
    log["done"] = now().isoformat(timespec="seconds")
    save()
    print("STEP9B_DONE", flush=True)


if __name__ == "__main__":
    main()
