#!/usr/bin/env python3
"""Paper figure / table drafts (docs/system1/paper_results_plan_zh.md section 3, third revision 2026-11-13: rubber hand,
Push / Pull / Table / Bin in the main text, Kick / Chair in the appendix). Every output is computed here from committed
CSVs; nothing is typed in. Missing data are left empty and named (a "missing" column / a note on the figure).

Inputs:
  * A: results_system1/T01b/schemeA/episodes.csv (variant schemeA; laptop 5070);
  * B: results_system1/T02/episodes.csv (B0 shown; every B variant listed in table1_b_variants.csv);
  * C-in, C-out, D (Pull / Table / Bin): results_system1/T03/c2_main/per_seed_summary.csv (laptop 5070);
  * training curves: results_system1/T03/train_efficiency/curve_points.csv, per_run.csv (cloud 4090);
  * load capacity: results_system1/T04/mass_sweep_summary.csv, T04/physics_check/physics_check.csv.
Outputs: results_system1/figures/ (table1_main.csv / .md, table1_b_variants.csv, fig2_training_efficiency.png / .csv,
fig3_load_capacity.png / .csv, fig5_summary_radar.png, fig5_summary_raw.csv).

    hssim_s1/bin/python -B system1/render/paper_figs.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.render.scheme_colors import PAPER_COLOR, PAPER_LINESTYLE, PAPER_NAME, PAPER_ORDER  # noqa: E402

R = ROOT / "results_system1"
OUT = R / "figures"
TASKS_MAIN = ("push", "pull", "table", "bucket")
TASKS_APPX = ("kick", "chair")
TASK_NAME = dict(push="Push", pull="Pull", table="Table", bucket="Bin", kick="Kick", chair="Chair")
C_TASKS = ("pull", "table", "bucket")
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"


def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def wr(p, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def episodes_cells(path, variant_of, scheme):
    """{(task, level): (n_completed, object_rmse_mean_completed, gpu set)} for a deterministic scheme."""
    cnt = defaultdict(lambda: [0, 0, [], set()])
    for x in rd(path):
        if x["is_nominal"] == "True" or x["variant"] != variant_of:
            continue
        c = cnt[(x["task"], int(x["perturbation_level_cm"]))]
        c[1] += 1
        c[3].add(x.get("gpu_model"))
        if x["completed_reference"] == "True":
            c[0] += 1
            c[2].append(float(x["object_pos_rmse_cm"]))
    return {k: dict(scheme=scheme, n=v[1], completion=[2 * v[0] if v[1] == 50 else 100 * v[0] / v[1]],
                    obj=statistics.mean(v[2]) if v[2] else None, gpus=v[3]) for k, v in cnt.items()}


def table1():
    cells = {}
    a = episodes_cells(R / "T01b/schemeA/episodes.csv", "schemeA", "A")
    b_all = {}
    for v in ("B0", "B1-0", "B1-2", "B1-4", "B-open"):
        b_all[v] = episodes_cells(R / "T02/episodes.csv", f"schemeB_{v}", "B")
    for (t, l), c in a.items():
        cells[("A", t, l)] = c
    for (t, l), c in b_all["B0"].items():
        cells[("B", t, l)] = c
    ps = rd(R / "T03/c2_main/per_seed_summary.csv")
    for sch in ("C-in", "C-out", "D"):
        for t in C_TASKS:
            for l in (2, 8):
                rr = [x for x in ps if x["variant"] == sch and x["task"] == t and int(x["perturbation_level_cm"]) == l and x["n_completed"]]
                if rr:
                    obj = [float(x["object_pos_rmse_cm_mean"]) for x in rr if x["object_pos_rmse_cm_mean"]]
                    cells[(sch, t, l)] = dict(scheme=sch, n=50, completion=[float(x["completion_pct"]) for x in rr],
                                              obj=statistics.mean(obj) if obj else None, gpus={x["gpu_model"] for x in rr},
                                              seeds=[x["training_seed"] for x in rr])
    rows = []
    for t in TASKS_MAIN + TASKS_APPX:
        for sch in PAPER_ORDER:
            for l in (2, 8):
                c = cells.get((sch, t, l))
                note = ""
                if c is None:
                    note = ("C not trained on this task (plan: C only Pull / Table / Bin)" if sch.startswith("C") else
                            "D not evaluated on the laptop 5070 for this task (V3 numbers are from the cloud 4090, not mixed in)" if sch == "D" else
                            "B not applicable (the planner drops the kick motion, T02)" if (sch == "B" and t == "kick") else "missing")
                    rows.append(dict(task=TASK_NAME[t], section="main" if t in TASKS_MAIN else "appendix", scheme=sch, level=f"+/-{l}",
                                     completion_pct_mean="", completion_pct_sd="", n_seeds="", completion_per_seed="", object_rmse_cm_completed_mean="",
                                     gpu="", note=note))
                    continue
                v = c["completion"]
                rows.append(dict(task=TASK_NAME[t], section="main" if t in TASKS_MAIN else "appendix", scheme=sch, level=f"+/-{l}",
                                 completion_pct_mean=statistics.mean(v), completion_pct_sd=statistics.stdev(v) if len(v) > 1 else "",
                                 n_seeds=len(v) if sch in ("C-in", "C-out", "D") else "deterministic",
                                 completion_per_seed=";".join(f"{x:g}" for x in v), object_rmse_cm_completed_mean=c["obj"] if c["obj"] is not None else "",
                                 gpu=";".join(sorted(str(g) for g in c["gpus"])),
                                 note="single seed (preliminary)" if sch == "C-out" else ""))
    wr(OUT / "table1_main.csv", rows)
    brows = []
    for v, cc in b_all.items():
        for (t, l), c in sorted(cc.items()):
            brows.append(dict(variant=v, task=TASK_NAME[t], level=f"+/-{l}", n=c["n"], completion_pct=c["completion"][0], gpu=";".join(map(str, c["gpus"]))))
    wr(OUT / "table1_b_variants.csv", brows)
    # markdown view
    md = ["| Task | Level | " + " | ".join(PAPER_NAME[s] for s in PAPER_ORDER) + " |", "|---|---|" + "---|" * len(PAPER_ORDER)]
    for t in TASKS_MAIN + TASKS_APPX:
        for l in (2, 8):
            cellstr = []
            for s in PAPER_ORDER:
                r = next(x for x in rows if x["task"] == TASK_NAME[t] and x["scheme"] == s and x["level"] == f"+/-{l}")
                if r["completion_pct_mean"] == "":
                    cellstr.append("—")
                elif r["completion_pct_sd"] != "":
                    cellstr.append(f"{r['completion_pct_mean']:.1f} ± {r['completion_pct_sd']:.1f}")
                else:
                    cellstr.append(f"{r['completion_pct_mean']:.0f}")
            md.append(f"| {TASK_NAME[t]}{' (appendix)' if t in TASKS_APPX else ''} | ±{l} | " + " | ".join(cellstr) + " |")
    (OUT / "table1_main.md").write_text("\n".join(md) + "\n")
    return rows


def fig2():
    pts = [x for x in rd(R / "T03/train_efficiency/curve_points.csv")]
    groups = [("C-in", "C-in", "-"), ("C-out", "C-out", "-"), ("D", "D (paper)", "-"), ("D", "D (paper, single-person prior)", "-"),
              ("D", "D (from scratch, V3-25-27)", ":")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    out = []
    for ax, t in zip(axes, C_TASKS):
        style(ax)
        for key, scheme, ls in groups:
            rr = [x for x in pts if x["task"] == t and x["scheme"] == scheme]
            if not rr:
                continue
            seeds = sorted({x["seed"] for x in rr})
            its = sorted({int(x["iteration"]) for x in rr})
            xs, mean, lo, hi, nmiss = [], [], [], [], []
            for it in its:
                v = [float(x["completion_pct"]) for x in rr if int(x["iteration"]) == it and x["status"] == "complete"]
                miss = len(seeds) - len(v)
                out.append(dict(task=t, scheme=scheme, iteration=it, env_steps=it * 49152, n_seeds=len(seeds), n_present=len(v),
                                mean_pct=statistics.mean(v) if v and not miss else "", min_pct=min(v) if v and not miss else "",
                                max_pct=max(v) if v and not miss else "", per_seed=";".join(f"{x:g}" for x in v),
                                missing="" if not miss else f"{miss} of {len(seeds)} seeds missing (not drawn)"))
                if v and not miss:
                    xs.append(it * 49152 / 1e8)
                    mean.append(statistics.mean(v))
                    lo.append(min(v))
                    hi.append(max(v))
                elif miss:
                    nmiss.append(it * 49152 / 1e8)
            col = PAPER_COLOR[key]
            label = PAPER_NAME[key] if key != "D" else {"D (paper)": PAPER_NAME["D"], "D (paper, single-person prior)": PAPER_NAME["D"] + ", single-person prior",
                                                         "D (from scratch, V3-25-27)": PAPER_NAME["D"] + ", from scratch"}[scheme]
            if len(seeds) > 1:
                label += f" (mean, range of {len(seeds)} seeds)"
                ax.fill_between(xs, lo, hi, color=col, alpha=0.15, linewidth=0)
            else:
                label += " (1 seed)"
            ax.plot(xs, mean, color=col, linestyle=ls, linewidth=2, marker="o", markersize=3.5, label=label)
            for xm in nmiss:
                ax.axvline(xm, color=col, linewidth=0.8, linestyle=(0, (1, 2)), alpha=0.6)
        ax.set_title(TASK_NAME[t], color=INK, fontsize=10)
        ax.set_xlabel("environment steps (×10⁸)", color=MUTED, fontsize=8)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("completion, ±2 states (%)", color=MUTED, fontsize=8)
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.99, 0.99, "DRAFT - thin dotted verticals: checkpoints with a missing curve point (not drawn)", ha="right", va="top", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.2, 1, 0.97))
    fig.savefig(OUT / "fig2_training_efficiency.png", dpi=200)
    plt.close(fig)
    wr(OUT / "fig2_training_efficiency.csv", out)


def fig3():
    ms = rd(R / "T04/mass_sweep_summary.csv")
    pc = rd(R / "T04/physics_check/physics_check.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
    out = []
    for ax, t in zip(axes, C_TASKS):
        style(ax)
        orig = {"pull": 20.0, "table": 5.0, "bucket": 1.0}[t]
        for sch in ("C-in", "D"):
            rr = [x for x in ms if x["task"] == t and x["scheme"] == sch and x["perturbation_level_cm"] == "2" and x["n_completed"] not in ("", None)
                  and (x["retrained"].startswith("yes") or x["retrained"].startswith("T03 main") or x["retrained"].startswith("paper result"))]
            pts = sorted((float(x["mass_kg"]) / orig, float(x["completion_pct"]), x["retrained"]) for x in rr)
            for p in pts:
                out.append(dict(task=t, scheme=sch, mass_ratio=p[0], completion_pm2_pct=p[1], source=p[2]))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PAPER_COLOR[sch], linewidth=2, marker="o", markersize=4,
                    label=PAPER_NAME[sch] + " (retrained per mass, seed 721)")
        ax.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_title(TASK_NAME[t], color=INK, fontsize=10)
        ax.set_xlabel("mass / original mass", color=MUTED, fontsize=8)
        ax.set_ylim(-3, 103)
        for x in pc:
            if x.get("task") != t:
                continue
            ratio = float(x["mass_kg"]) / orig
            proposed = x.get("proposed") not in ("", None)  # level proposed by the add-level / midpoint rule (not trained)
            if x.get("physically_impossible") == "True":
                ax.axvspan(ratio / 1.15, ratio * 1.15, color=GRID, alpha=0.8)
                ax.text(ratio, 102, "beyond physical\nlimit (not trained)", ha="center", va="top", fontsize=6.5, color=MUTED)
            elif proposed:
                ax.text(ratio, 102, "rule triggered,\nnot trained", ha="center", va="top", fontsize=6.5, color=MUTED,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
                ax.axvline(ratio, color=MUTED, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("completion, ±2 states (%)", color=MUTED, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.text(0.99, 0.99, "DRAFT - A and B: 0 % at every mass (table only); dashed line: 50 % boundary", ha="right", va="top", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(OUT / "fig3_load_capacity.png", dpi=200)
    plt.close(fig)
    wr(OUT / "fig3_load_capacity.csv", out)
    return pc


def fig5(t1):
    """Raw values per scheme and dimension, then a radar draft. Normalisation (also in the README): completion and AUC in % / 100;
    load boundary = largest mass ratio with +/-2 >= 50 % divided by the largest ratio trained (8); 'no training needed' = 1 for A, B."""
    pr = rd(R / "T03/train_efficiency/per_run.csv")
    ms = rd(R / "T04/mass_sweep_summary.csv") + rd(R / "T04/B/mass_sweep_summary_B.csv")
    raw = []
    for s in PAPER_ORDER:
        def comp(level):
            v = [float(x["completion_pct_mean"]) for x in t1 if x["scheme"] == s and x["level"] == level and x["task"] in ("Pull", "Table", "Bin")
                 and x["completion_pct_mean"] != ""]
            return statistics.mean(v) if len(v) == 3 else None
        auc = [float(x["auc_mean_completion_pct"]) for x in pr if x["scheme"] in ((s,) if s != "D" else ("D (paper)", "D (paper, single-person prior)"))
               and x["auc_mean_completion_pct"] not in ("", None) and int(x["n_points_missing"]) == 0]
        n_auc_runs = len([x for x in pr if x["scheme"] in ((s,) if s != "D" else ("D (paper)", "D (paper, single-person prior)"))])
        bnd = []
        for t, orig in (("pull", 20.0), ("table", 5.0), ("bucket", 1.0)):
            rr = [x for x in ms if x["task"] == t and x["scheme"] == ("C-in" if s == "C-in" else s) and x["perturbation_level_cm"] == "2"
                  and x["n_completed"] not in ("", None) and (x["retrained"].startswith(("yes", "T03 main", "paper result", "not_applicable")))]
            ok = [float(x["mass_kg"]) / orig for x in rr if float(x["completion_pct"]) >= 50]
            bnd.append(max(ok) if ok else (0.0 if rr else None))
        raw.append(dict(scheme=s, completion_pm2_mean_pull_table_bin=comp("+/-2"), completion_pm8_mean_pull_table_bin=comp("+/-8"),
                        training_auc_mean=statistics.mean(auc) if auc and len(auc) == n_auc_runs else None,
                        training_auc_runs_complete=f"{len(auc)}/{n_auc_runs}" if n_auc_runs else "not applicable (no training)",
                        load_boundary_ratio_per_task=";".join("" if b is None else f"{b:g}" for b in bnd),
                        load_boundary_ratio_mean=statistics.mean(bnd) if all(b is not None for b in bnd) else None,
                        needs_training=s not in ("A", "B")))
    wr(OUT / "fig5_summary_raw.csv", raw)
    dims = ["completion ±2", "completion ±8", "training efficiency\n(AUC)", "load boundary", "no training\nneeded"]
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, polar=True)
    ax.set_xticks(ang)
    ax.set_xticklabels(dims, fontsize=7.5, color=INK)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=6, color=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    for r in raw:
        vals = [None if r["completion_pm2_mean_pull_table_bin"] is None else r["completion_pm2_mean_pull_table_bin"] / 100,
                None if r["completion_pm8_mean_pull_table_bin"] is None else r["completion_pm8_mean_pull_table_bin"] / 100,
                None if r["training_auc_mean"] is None else r["training_auc_mean"] / 100,
                None if r["load_boundary_ratio_mean"] is None else r["load_boundary_ratio_mean"] / 8,
                0.0 if r["needs_training"] else 1.0]
        s = r["scheme"]
        v = np.array([np.nan if x is None else x for x in vals])
        vv = np.concatenate([v, v[:1]])
        aa = np.concatenate([ang, ang[:1]])
        ax.plot(aa, vv, color=PAPER_COLOR[s], linestyle=PAPER_LINESTYLE[s], linewidth=2, marker="o", markersize=3.5, label=PAPER_NAME[s])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=7, frameon=False, ncol=1)
    fig.text(0.02, 0.98, "DRAFT - a scheme without a value on an axis has a gap there\n(missing or not applicable, see fig5_summary_raw.csv)",
             fontsize=6.5, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_summary_radar.png", dpi=200)
    plt.close(fig)
    return raw


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t1 = table1()
    fig2()
    fig3()
    raw = fig5(t1)
    for r in raw:
        print(r)


if __name__ == "__main__":
    main()
