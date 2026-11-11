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
