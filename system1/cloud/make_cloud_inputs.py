#!/usr/bin/env python3
"""Pack the read-only inputs the cloud needs and the repository does not contain.

Output: exports_system1/cloud_inputs/cloud_inputs.tar (+ manifest with source paths and
sha256). Contents:
  priors/marl_compat_pull_v1/      Pull single-person prior (D T04 init; C Pull normalizer)
  repro/Pull-baseline_seed721/     paper Table Pulling seed721 checkpoint (cloud repro check)
  states/                          frozen pull states +/-2 and +/-8
  v3/episodes.csv                  V3 closeout episodes table (repro comparison)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = Path("/home/kevin/holosoma-core4d-base")
M = Path("/home/kevin/holosoma-rubber-hand-marl")
FILES = {
    "priors/marl_compat_pull_v1/model_07999_actor158.pt": M / "logs/WholeBodyTracking/marl_compat_pull_v1/model_07999_actor158.pt",
    "priors/marl_compat_pull_v1/conversion_manifest.json": M / "logs/WholeBodyTracking/marl_compat_pull_v1/conversion_manifest.json",
    "repro/Pull-baseline_seed721/model_08050.pt": M / "logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt",
    "states/pull_states.json": B / "logs/PaperSupplement20260929/unified_evaluation_20261003/pull_states.json",
    "states/pull_pm8.json": B / "logs/V3_20261102/cloud_backup_20261104/raw/v3_extended_eval_20261104_ops/states/pull_pm8.json",
    "v3/episodes.csv": B / "paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/results/episodes.csv",
}
EXPECT = {
    "priors/marl_compat_pull_v1/model_07999_actor158.pt": "f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364",
    "repro/Pull-baseline_seed721/model_08050.pt": "727630e9cec654d88bfb454d5eaabd70d7db629478598a2039140653a57cbf43",
    "states/pull_states.json": "473c1fc0563228a770393c2ccd78f3b051df258a408e6481d44c12fa2450c572",
    "states/pull_pm8.json": "3c2a60c7982db92b59b1f3343103e711409a6c756faaf83b86016b8163c8d5b2",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = ROOT / "exports_system1/cloud_inputs"
    stage = out / "stage"
    stage.mkdir(parents=True, exist_ok=False)
    manifest, sums = {}, []
    for rel, src in FILES.items():
        h = sha(src)
        if rel in EXPECT and h != EXPECT[rel]:
            raise RuntimeError(f"sha mismatch for {src}")
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[rel] = dict(source=str(src), sha256=h)
        sums.append(f"{h}  {rel}")
    (stage / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with tarfile.open(out / "cloud_inputs.tar", "w") as tar:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(stage)))
    (out / "cloud_inputs.tar.sha256").write_text(f"{sha(out / 'cloud_inputs.tar')}  cloud_inputs.tar\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
