#!/usr/bin/env python3
"""Export reference pelvis routes [N,2,3] and object centre [N,3] per task (paper loaders) for figures_t01b.py."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sonic.task_refs import TASKS, load_task_reference
out = ROOT / "logs_system1/T01b/figures_cache"; out.mkdir(parents=True, exist_ok=True)
for t in TASKS:
    ref, bodies, _, meta = load_task_reference(t)
    s = {k: v.numpy() for k, v in ref.sample(torch.arange(ref.num_frames)).items()}
    np.savez(out / f"{t}_ref.npz", pelvis=s["agent_body_pos_w"][:, :, bodies.index("pelvis")], object=s["object_pos_w"],
             reference_sha256=meta["sha256"])
    print(t, ref.num_frames)
