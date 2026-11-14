"""Fixed scheme colours for every System 1 figure (CLAUDE.md: one colour per scheme in all plots, legend in full).

Steps from the dataviz reference palette (light mode). Checked with OKLab Delta E x100 under normal vision and
Machado-2009 protan / deutan / tritan simulation (2026-11-11; the JS validator needs node, not installed here):
adjacent pairs in A-B-C-D order all clear normal >= 15 and CVD >= 8. The non-adjacent B-D pair (yellow / orange,
normal 14) does not: a line chart that shows B and D together draws B dashed (LINESTYLE). Yellow and aqua sit
below 3:1 contrast on white, so figures carry a legend and a table/CSV view of the values.
"""
SCHEME_COLOR = {"A": "#2a78d6", "B": "#eda100", "C": "#1baf7a", "D": "#eb6834"}
SCHEME_NAME = {"A": "A 静态目标 + SONIC", "B": "B 动态目标 + SONIC", "C": "C MARL + SONIC", "D": "D 端到端 MARL"}
LINESTYLE = {"A": "-", "B": "--", "C": "-", "D": "-"}
# Paper figures (docs/system1/paper_results_plan_zh.md section 3, 2026-11-13): five schemes; C-in and C-out share the green
# hue at two lightness steps (C-in = the C colour above, C-out darker). Checked with system1/render/palette_check.py
# --pairs all on white: every pair normal >= 15 and CVD >= 8 except B-D (normal 13.7, deutan 9.7), so B stays dashed.
# On a dark surface C-out is below 3:1, so paper figures use the white surface.
PAPER_COLOR = {"A": "#2a78d6", "B": "#eda100", "C-in": "#1baf7a", "C-out": "#0e6e4c", "D": "#eb6834"}
PAPER_NAME = {"A": "A: static target + SONIC", "B": "B: dynamic target + SONIC", "C-in": "C-in: RL corrects SONIC's input",
              "C-out": "C-out: RL corrects SONIC's output", "D": "D: end-to-end MARL (DuetMimic)"}
PAPER_LINESTYLE = {"A": "-", "B": "--", "C-in": "-", "C-out": "-", "D": "-"}
PAPER_ORDER = ("A", "B", "C-in", "C-out", "D")


def use_cjk_font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from pathlib import Path
    for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
