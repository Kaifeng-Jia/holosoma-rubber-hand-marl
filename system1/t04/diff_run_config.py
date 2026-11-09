#!/usr/bin/env python3
"""Field-by-field diff of two run_config.json files (new vs reference).

Large arrays (e.g. per-env PhysX read-backs) are compared by their unique values.
Output: JSON list of {key, reference, new, category}. ``category`` marks fields that
the T04 rule allows to differ (mass, seed, directory) and pure run metadata
(timestamps, commit, process info), so anything else stands out as ``unexpected``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED_MASS = ("runtime_object_physics", "object_physics", "object_urdf", "object_urdf_sha256",
                "system1_object_mass_variant", "experiment_contract", "training_object_urdf_sha256")
ALLOWED_DIR = ("output_dir",)
ALLOWED_SEED = ("seed",)
METADATA = ("started_unix_time", "git_commit", "runtime_process", "resume_checkpoint", "resume_checkpoint_sha256")


def summarize(v):
    if isinstance(v, dict) and set(v) == {"shape", "values"}:
        flat = json.dumps(v["values"])
        uniq = sorted({json.dumps(x) for x in v["values"]}) if isinstance(v["values"], list) else [flat]
        return {"shape": v["shape"], "unique_rows": uniq[:3], "n_unique": len(uniq)}
    if isinstance(v, list) and len(v) > 20:
        uniq = sorted({json.dumps(x) for x in v})
        return {"len": len(v), "unique": uniq[:3], "n_unique": len(uniq)}
    return v


def walk(a, b, prefix=""):
    out = []
    keys = sorted(set(a) | set(b)) if isinstance(a, dict) and isinstance(b, dict) else None
    if keys is None:
        if summarize(a) != summarize(b):
            out.append((prefix, summarize(a), summarize(b)))
        return out
    for k in keys:
        p = f"{prefix}.{k}" if prefix else k
        if k not in a or k not in b:
            out.append((p, summarize(a.get(k, "<absent>")), summarize(b.get(k, "<absent>"))))
        elif isinstance(a[k], dict) and isinstance(b[k], dict) and set(a[k]) != {"shape", "values"}:
            out.extend(walk(a[k], b[k], p))
        elif summarize(a[k]) != summarize(b[k]):
            out.append((p, summarize(a[k]), summarize(b[k])))
    return out


def category(key):
    top = key.split(".")[0]
    if top in ALLOWED_MASS or "object_mass" in key or "object_urdf" in key:
        return "allowed:mass"
    if top in ALLOWED_DIR:
        return "allowed:directory"
    if top in ALLOWED_SEED:
        return "allowed:seed"
    if top in METADATA:
        return "metadata"
    return "unexpected"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ref, new = json.loads(a.reference.read_text()), json.loads(a.new.read_text())
    rows = [dict(key=k, reference=r, new=n, category=category(k)) for k, r, n in walk(ref, new)]
    a.out.write_text(json.dumps(dict(reference=str(a.reference), new=str(a.new), diffs=rows), indent=1, default=str))
    for r in rows:
        print(f"[{r['category']}] {r['key']}: {json.dumps(r['reference'], default=str)[:120]} -> {json.dumps(r['new'], default=str)[:120]}")
    print(f"{sum(r['category'] == 'unexpected' for r in rows)} unexpected of {len(rows)} differing fields")


if __name__ == "__main__":
    main()
