"""Load each task's paired runtime reference exactly as the paper environment does.

* push : PairedA1Reference.from_motion_loader(MotionLoader(single A1 file)), lateral spacing
         from the registered-placement proof (0.8780472 m);
* pull / kick : PairedMotionReference(paired runtime file of the Plan5 command config);
* chair / bucket / table : PairedMotionReference(runtime file of the CORE4D experiment).
Arrays come out in simulator body order (``g1_29dof_w_object.body_names``), quaternions xyzw.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "holosoma"))
PUSH_PROOF = ROOT / "system1/evidence/preflight_push_registered.json"
TASKS = ("push", "pull", "kick", "chair", "bucket", "table")


def sim_names():
    from holosoma.config_values.robot import g1_29dof_w_object as R

    return list(R.body_names), list(R.dof_names)


def reference_file(task):
    from holosoma.config_values.marl.g1.command import PLAN5_PULL_PAIRED_REFERENCE_FILE, motion_config
    from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment
    from holosoma.config_values.marl.g1.kick_command import PLAN5_KICK_RUNTIME_REFERENCE_FILE

    if task == "push":
        return motion_config.motion_file
    if task == "pull":
        return PLAN5_PULL_PAIRED_REFERENCE_FILE
    if task == "kick":
        return PLAN5_KICK_RUNTIME_REFERENCE_FILE
    exp = get_core4d_pair_experiment(dict(chair="chair021", bucket="bucket003", table="smalltable5kg_A")[task])
    return exp.runtime_reference_file


def load_task_reference(task, device="cpu"):
    from holosoma.envs.marl import PairedA1Reference, PairedMotionReference
    from holosoma.managers.command.terms.wbt import FAKE_BODY_NAME_ALIASES, MotionLoader
    from holosoma.utils.path import resolve_data_file_path

    bodies, dofs = sim_names()
    aliases = [FAKE_BODY_NAME_ALIASES.get(n, n) for n in bodies]
    rel = reference_file(task)
    path = resolve_data_file_path(str(rel))
    meta = dict(task=task, file=str(rel), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    if task == "push":
        spacing = float(json.loads(PUSH_PROOF.read_text())["lateral_spacing_m"])
        ref = PairedA1Reference.from_motion_loader(MotionLoader(str(rel), aliases, dofs, device=device), lateral_spacing_m=spacing)
        meta.update(loader="PairedA1Reference", lateral_spacing_m=spacing,
                    proof_sha256=hashlib.sha256(PUSH_PROOF.read_bytes()).hexdigest())
    else:
        ref = PairedMotionReference(str(rel), aliases, dofs, device=device)
        meta.update(loader="PairedMotionReference")
    return ref, bodies, dofs, meta
