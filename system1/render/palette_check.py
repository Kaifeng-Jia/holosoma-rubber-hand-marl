#!/usr/bin/env python3
"""Palette check for the System 1 figures (the method of the dataviz reference validator, in Python because node is not
installed): OKLab Delta E x100 between colour pairs under normal vision and the Machado et al. 2009 protan / deutan /
tritan simulations (severity 1.0, applied in linear sRGB), and WCAG contrast against the surface.
Targets: normal >= 15 for every checked pair, CVD >= 8 (6-8 only with a second encoding such as a dash or marker).

    python3 system1/render/palette_check.py "#2a78d6,#eda100,#1baf7a,#0e6e4c,#eb6834" --surface "#ffffff" [--pairs all]
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def oklab(lrgb):
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return M2 @ np.cbrt(M1 @ np.clip(lrgb, 0, 1))


def lum(lrgb):
    return float(lrgb @ [0.2126, 0.7152, 0.0722])


def check(colors, surface="#ffffff", pairs="adjacent"):
    L = [lin(hex_rgb(c)) for c in colors]
    out = []
    idx = list(itertools.combinations(range(len(colors)), 2)) if pairs == "all" else [(i, i + 1) for i in range(len(colors) - 1)]
    for i, j in idx:
        row = dict(pair=f"{colors[i]}-{colors[j]}", normal=100 * float(np.linalg.norm(oklab(L[i]) - oklab(L[j]))))
        for k, M in MACHADO.items():
            row[k] = 100 * float(np.linalg.norm(oklab(M @ L[i]) - oklab(M @ L[j])))
        out.append(row)
    ls = lum(lin(hex_rgb(surface)))
    contrast = {c: (max(lum(x), ls) + 0.05) / (min(lum(x), ls) + 0.05) for c, x in zip(colors, L)}
    return out, contrast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("colors")
    ap.add_argument("--surface", default="#ffffff")
    ap.add_argument("--pairs", choices=("adjacent", "all"), default="adjacent")
    a = ap.parse_args()
    rows, con = check(a.colors.split(","), a.surface, a.pairs)
    for r in rows:
        worst = min(r["protan"], r["deutan"], r["tritan"])
        print(f"{r['pair']}: normal {r['normal']:.1f}  protan {r['protan']:.1f}  deutan {r['deutan']:.1f}  tritan {r['tritan']:.1f}  "
              f"{'PASS' if r['normal'] >= 15 and worst >= 8 else 'WARN(6-8, needs 2nd encoding)' if r['normal'] >= 15 and worst >= 6 else 'FAIL'}")
    for c, v in con.items():
        print(f"contrast {c} vs {a.surface}: {v:.2f}{'' if v >= 3 else '  (<3:1: labels / table view needed)'}")


if __name__ == "__main__":
    main()
