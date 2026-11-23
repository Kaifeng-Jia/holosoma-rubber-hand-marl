#!/usr/bin/env python3
"""T14: uniformly scale an object URDF by k about the object frame origin (box sizes, box origins, mesh scale, inertial
origin x k; inertia x k^2 at the same mass). For rendering and kinematic evaluation of the size-scaled references; a
training asset at another size would need its own reviewed manifest.

    python3 system1/t14/scale_box_urdf.py --urdf <in.urdf> --k 1.2 --out <out.urdf>
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def scale_urdf(src: Path, k: float, dst: Path) -> Path:
    tree = ET.parse(src)
    root = tree.getroot()
    for el in root.iter():
        if el.tag in ("visual", "collision", "inertial"):
            o = el.find("origin")
            if o is not None and o.get("xyz"):
                o.set("xyz", fmt([float(x) * k for x in o.get("xyz").split()]))
        if el.tag == "box":
            el.set("size", fmt([float(x) * k for x in el.get("size").split()]))
        if el.tag == "mesh":  # mesh objects (Bin, Chair): scale attribute x k, file made absolute
            sc = [float(x) for x in (el.get("scale") or "1 1 1").split()]
            el.set("scale", fmt([x * k for x in sc]))
            f = Path(el.get("filename"))
            el.set("filename", str(f if f.is_absolute() else (src.parent / f).resolve()))
        if el.tag == "inertia":
            for a in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                el.set(a, f"{float(el.get(a)) * k * k:.12g}")
    root.set("name", f"{root.get('name')}_k{k:g}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(scale_urdf(a.urdf, a.k, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
