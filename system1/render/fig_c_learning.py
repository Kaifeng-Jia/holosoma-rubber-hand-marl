#!/usr/bin/env python3
"""Figure: per-iteration training curves, C vs D from scratch (system1/t03/c_learning_curves.py output).
Rows = tasks, columns = action std (from entropy), episode length, timeout share, KL, actor LR, value loss.
C: one line; D: mean of three seeds (line) and min-max over seeds (band). 50-iteration rolling mean for display.

    hssim_s1/bin/python -B system1/render/fig_c_learning.py --dir results_system1/T03/c_learning_diag_20261111
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.render.scheme_colors import SCHEME_COLOR, SCHEME_NAME, use_cjk_font  # noqa: E402

TASK_NAME = {"pull": "Pull（拉桌子）", "table": "Table（搬桌子）", "bucket": "Bin（递箱子）"}
PANELS = [("sigma_geomean_from_entropy", "动作标准差（由熵换算，几何平均）", "linear"),
          ("mean_episode_length_logged", "平均回合长度（步）", "linear"),
          ("frac_timeout", "结束回合中超时（走完参考）的比例", "linear"),
          ("kl", "KL", "log"),
          ("actor_learning_rate", "actor 学习率", "log"),
          ("value_loss", "价值损失", "log")]
SMOOTH = 50


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, required=True)
    a = p.parse_args()
    use_cjk_font()
    fig, ax = plt.subplots(3, len(PANELS), figsize=(4.0 * len(PANELS), 9.6), sharex="row")
    for r, task in enumerate(("pull", "table", "bucket")):
        df = pd.read_csv(a.dir / f"curves_{task}.csv.gz")
        for c, (key, title, scale) in enumerate(PANELS):
            axx = ax[r, c]
            cs = df[df.scheme == "C"].sort_values("iteration")
            y = cs[key].rolling(SMOOTH, min_periods=1).mean()
            axx.plot(cs.iteration, y, color=SCHEME_COLOR["C"], lw=2, label=SCHEME_NAME["C"] + "（种子 721）")
            d = df[df.scheme == "D"].pivot_table(index="iteration", columns="seed", values=key).sort_index()
            d = d.rolling(SMOOTH, min_periods=1).mean()
            axx.fill_between(d.index, d.min(axis=1), d.max(axis=1), color=SCHEME_COLOR["D"], alpha=0.18, lw=0)
            axx.plot(d.index, d.mean(axis=1), color=SCHEME_COLOR["D"], lw=2, label=SCHEME_NAME["D"] + "（从零训练，三种子均值，带=最小–最大）")
            axx.set_yscale(scale)
            axx.grid(True, color="#e6e6e3", lw=0.6)
            for s in ("top", "right"):
                axx.spines[s].set_visible(False)
            if r == 0:
                axx.set_title(title, fontsize=11)
            if c == 0:
                axx.set_ylabel(TASK_NAME[task], fontsize=12)
            if r == 2:
                axx.set_xlabel("迭代数")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, f"曲线为 {SMOOTH} 次迭代滑动平均；D 只画到 C 当前的迭代数。Pull 的 D 为 V3-25–27，Table 为论文 T-07，Bin 为论文 B-RZ0。",
             ha="center", fontsize=10, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    out = a.dir / "fig_c_vs_d_training_curves.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
