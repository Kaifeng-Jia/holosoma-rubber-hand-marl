#!/usr/bin/env python3
"""T14 step 9g item 1 (Kevin 2026-11-26): re-judge finished evaluations with the non-hand penetration measured on the actual
collision shapes (nonhand_exact.py); no solve is re-run. Read only on the solves, writes only under --out.

Sets and the verdict that is re-judged ("old"):
  * step8b: the 57 candidates of the step-8 rerun (logs_system1/t14/step8b/cands/<seq>/<cand>/m1p00/eval.json); old verdict =
    the round-6 re-judge of these solves (results_system1/T14_retargeting/step8b/step8b_rejudge.csv, new_applicable / new_failed),
    the verdict step 8b reported; Push / Pull selection rule as step8b_compare.py (applicable candidates, smallest joint fit
    error rounded to 1e-10, ties by ID);
  * step9f, step9g: every level with an eval.json under <logdir>/<clip>/m<level>/; old verdict = eval.json (applicable, failed);
    reported levels and clip summary from the driver json (--driver; summary[clip].report_levels).
New verdict = the old failed items with "V0 non-hand penetration r0 / r1" replaced by the new measure (limit 0.01 m, unchanged);
every other item as it was. The source solve of each level is the one in the evaluation's own command (--source).

    sonic_s1/bin/python -B system1/t14/objects/rejudge_nonhand_exact.py --set step9f --driver logs_system1/t14/step9f/driver.json \
        --out logs_system1/t14/rejudge_exact/step9f --workers 1
    ... --table-only (tables from the per-level jsons) --results <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NH = ("V0 non-hand penetration r0", "V0 non-hand penetration r1")
ORDER = ["V0 gravity", "V0 support drift r0", "V0 support drift r1", "V0 non-hand penetration r0", "V0 non-hand penetration r1",
         "V0 robot-robot clearance", "section-4 contact retention >= 0.99", "main motion"]


def arg_of(cmd, flag):
    t = shlex.split(cmd)
    return t[t.index(flag) + 1]


def one(ev_path, out_path):
    sys.path.insert(0, str(HERE))
    import nonhand_exact as nx
    e = json.loads(Path(ev_path).read_text())
    cand, src = Path(arg_of(e["command"], "--cand-dir")), Path(arg_of(e["command"], "--source"))
    t0 = time.time()
    m = nx.measure(cand, src)
    res = dict(eval=str(ev_path), cand_dir=str(cand), source=str(src), new=m, seconds=round(time.time() - t0, 2),
               old_proxy={f"r{r}": e["V0"][f"nonhand_penetration_r{r}"]["value_m"] for r in (0, 1)},
               old_applicable=e["applicable"], old_failed=e["failed"], module_rules=nx.__doc__)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=1) + "\n")
    return str(out_path)


def replace_nonhand(failed, new):
    f = [x for x in failed if x not in NH] + [NH[r] for r in (0, 1) if not new[f"r{r}"]["passed"]]
    return sorted(set(f), key=lambda x: ORDER.index(x) if x in ORDER else 99)


def jobs(a):
    L = a.logdir
    if a.set == "step8b":
        evs = sorted(L.glob("cands/*/c*/m1p00/eval.json"))
    else:
        evs = sorted(L.glob("*/m*/eval.json"))
    return [(ev, a.out / ev.relative_to(L).parent.with_suffix(".json")) for ev in evs]


def run(a):
    todo = [(e, o) for e, o in jobs(a) if not o.exists()]
    print(f"{len(todo)} to measure", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs = [ex.submit(one, str(e), str(o)) for e, o in todo]
        for f in as_completed(fs):
            print(f.result(), flush=True)


def lvl(tag):
    return float(tag[1:].replace("p", "."))


def nh_cols(row, x):
    for r in (0, 1):
        n = x["new"][f"r{r}"]
        row[f"old_proxy_nonhand_r{r}_m"] = x["old_proxy"][f"r{r}"]
        row[f"new_exact_nonhand_r{r}_m"] = n["value_m"]
        row[f"new_exact_link_r{r}"] = n["link"]
        row[f"new_exact_frame_r{r}"] = n["frame"]
        row[f"new_exact_max_depth_r{r}_m"] = n["max_depth_m"]
        row[f"new_exact_frames_over_limit_r{r}"] = n["frames_added_over_limit"]
        row[f"new_exact_links_over_limit_r{r}"] = " | ".join(n["links_over_limit"])
    row["source_exact_max_depth_m"] = x["new"]["source_max_depth_m"]


def table_step8b(a):
    R = ROOT / "results_system1/T14_retargeting/step8b"
    rj = {(r["sequence"], r["candidate"]): r for r in csv.DictReader(open(R / "step8b_rejudge.csv"))}
    fit = {(r["sequence"], r["candidate"]): r for r in csv.DictReader(open(R / "step8_old_vs_new.csv"))}
    rows = []
    for ev, o in jobs(a):
        x = json.loads(o.read_text())
        seq, cand = ev.parts[-4], ev.parts[-3]
        old = rj[(seq, cand)]
        old_failed = [s for s in old["new_failed"].split(" | ") if s]
        nf = replace_nonhand(old_failed, x["new"])
        row = dict(sequence=seq, candidate=cand, id=old["id"], fit_rmse_m=fit[(seq, cand)]["fit_rmse_m"],
                   old_applicable=old["new_applicable"] == "True", old_failed=" | ".join(old_failed))
        nh_cols(row, x)
        row.update(new_applicable=not nf, new_failed=" | ".join(nf), verdict_changed=(not nf) != (old["new_applicable"] == "True"))
        rows.append(row)
    summ = {}
    for seq in dict.fromkeys(r["sequence"] for r in rows):
        rs = [r for r in rows if r["sequence"] == seq]

        def sel(key):
            ok = [r for r in rs if r[key]]
            return sorted(ok, key=lambda r: (round(float(r["fit_rmse_m"]) / 1e-10), r["id"]))[0]["candidate"] if ok else None
        summ[seq] = dict(evaluated=len(rs), applicable_old=[r["candidate"] for r in rs if r["old_applicable"]],
                         applicable_new=[r["candidate"] for r in rs if r["new_applicable"]], selected_old=sel("old_applicable"),
                         selected_new=sel("new_applicable"))
        summ[seq]["selection_changed"] = summ[seq]["selected_old"] != summ[seq]["selected_new"]
    return rows, summ


def table_step9(a):
    drv = json.loads(a.driver.read_text())["summary"]
    rows, clips = [], {}
    by = {}
    for ev, o in jobs(a):
        by.setdefault(ev.parts[-3], {})[lvl(ev.parts[-2])] = json.loads(o.read_text())
    for clip, sm in drv.items():
        rep = sm.get("report_levels") or []
        got = by.get(clip, {})
        for m in sorted(got):
            x = got[m]
            nf = replace_nonhand(x["old_failed"], x["new"])
            row = dict(clip=clip, sequence=sm["sequence"], level=f"{m:.2f}", reported=m in [round(v, 2) for v in rep],
                       old_applicable=x["old_applicable"], old_failed=" | ".join(x["old_failed"]))
            nh_cols(row, x)
            row.update(new_applicable=not nf, new_failed=" | ".join(nf), verdict_changed=(not nf) != x["old_applicable"])
            rows.append(row)
        rr = [r for r in rows if r["clip"] == clip and r["reported"]]
        clips[clip] = dict(sequence=sm["sequence"], report_levels=rep, evaluated_report_levels=[float(r["level"]) for r in rr],
                           unsolved_report_levels=[v for v in rep if round(v, 2) not in [round(float(r["level"]), 2) for r in rr]],
                           applicable_old=[float(r["level"]) for r in rr if r["old_applicable"]],
                           applicable_new=[float(r["level"]) for r in rr if r["new_applicable"]],
                           driver_applicable_levels=sm.get("applicable_levels"))
        clips[clip]["clip_applicable_old"] = bool(clips[clip]["applicable_old"])
        clips[clip]["clip_applicable_new"] = bool(clips[clip]["applicable_new"])
        assert clips[clip]["applicable_old"] == [float(v) for v in (sm.get("applicable_levels") or [])], (clip, clips[clip])
    return rows, clips


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=("step8b", "step9f", "step9g", "step9g2"), required=True)
    ap.add_argument("--logdir", type=Path, default=None)
    ap.add_argument("--driver", type=Path, default=None, help="step 9: driver json with summary[clip].report_levels")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--results", type=Path, default=None, help="write <set>_nonhand_exact.csv / .json here")
    ap.add_argument("--name", default=None, help="file stem of the tables (default <set>_nonhand_exact)")
    a = ap.parse_args()
    a.logdir = (a.logdir or ROOT / f"logs_system1/t14/{a.set}")
    a.logdir = a.logdir if a.logdir.is_absolute() else ROOT / a.logdir
    a.out = a.out if a.out.is_absolute() else ROOT / a.out
    if not a.table_only:
        run(a)
    if a.results is None:
        return 0
    rows, summ = table_step8b(a) if a.set == "step8b" else table_step9(a)
    a.results.mkdir(parents=True, exist_ok=True)
    stem = a.name or f"{a.set}_nonhand_exact"
    write_csv(a.results / f"{stem}.csv", rows)
    (a.results / f"{stem}.json").write_text(json.dumps(dict(rules=__doc__, command=" ".join(sys.argv), summary=summ), indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
