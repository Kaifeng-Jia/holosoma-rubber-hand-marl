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


SCHEME_C = ("system1_scheme_c",)  # T03: fields that exist because actions go through SONIC
C_SCHEMES = ("c", "cin", "cout")
C_IN_PPO = ("entropy_coef", "init_noise_std")  # T03 C v2: C-in uses init std 0.3 and entropy 0 (preregistered)


def category(key, ref=None, new=None, scheme=None):
    top = key.split(".")[0]
    if top == "bucket_reference" and isinstance(ref, str) and isinstance(new, str) and "/src/holosoma/" in ref \
            and ref.split("/src/holosoma/")[1] == new.split("/src/holosoma/")[1]:
        return "allowed:directory"  # same file inside the checkout; only the checkout root differs
    if scheme in C_SCHEMES and top in SCHEME_C:
        return "allowed:scheme_c"
    if scheme in C_SCHEMES and top == "save_interval":
        return "allowed:save_interval(T03)"
    if scheme == "cin" and top in C_IN_PPO:
        return "allowed:c_in_ppo(T03 C v2)"
    if top in ALLOWED_MASS or "object_mass" in key or "object_urdf" in key:
        return "allowed:mass"
    if top in ALLOWED_DIR:
        return "allowed:directory"
    if top in ALLOWED_SEED:
        return "allowed:seed"
    if top in METADATA:
        return "metadata"
    return "unexpected"


def diff_files(reference, new, scheme="d"):
    ref, nw = json.loads(Path(reference).read_text()), json.loads(Path(new).read_text())
    return [dict(key=k, reference=r, new=n, category=category(k, r, n, scheme)) for k, r, n in walk(ref, nw)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scheme", choices=("d", "c", "cin", "cout"), default="d")
    a = p.parse_args()
    rows = diff_files(a.reference, a.new, a.scheme)
    a.out.write_text(json.dumps(dict(reference=str(a.reference), new=str(a.new), scheme=a.scheme, diffs=rows), indent=1, default=str))
    for r in rows:
        print(f"[{r['category']}] {r['key']}: {json.dumps(r['reference'], default=str)[:120]} -> {json.dumps(r['new'], default=str)[:120]}")
    print(f"{sum(r['category'] == 'unexpected' for r in rows)} unexpected of {len(rows)} differing fields")


if __name__ == "__main__":
    main()
