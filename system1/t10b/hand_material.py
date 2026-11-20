"""T10b A: set the PhysX material of the rubber-hand collision shapes only (both robots), for the friction-margin evaluation.

The rubber hands are their own rigid bodies in the Isaac articulation ('left_rubber_hand_link', 'right_rubber_hand_link'; one
convex-hull shape each, palm and fingers in one mesh). Shape indices per body are found exactly as Isaac Lab's
randomize_rigid_body_material does (one rigid-body view per articulation link path, its max_shapes, consecutive slices of the
articulation's material table). The new values are written with the PhysX tensor API (the same API holosoma uses for the object
material) and read back; every other shape (other robot links) must stay bit-identical, otherwise this raises.

Used by scripts/evaluate_paper_supplement.py --hand-friction (off: unchanged).
"""
from __future__ import annotations

import torch

HAND_BODIES = ("left_rubber_hand_link", "right_rubber_hand_link")


def shapes_per_body(art):
    view = art.root_physx_view
    paths = list(view.link_paths[0])
    counts = [int(art._physics_sim_view.create_rigid_body_view(p).max_shapes) for p in paths]
    if sum(counts) != int(view.max_shapes):
        raise RuntimeError(f"shape count per body {sum(counts)} != articulation max_shapes {view.max_shapes}")
    out, start = {}, 0
    for p, c in zip(paths, counts):
        out[p.rsplit("/", 1)[-1]] = list(range(start, start + c))
        start += c
    return out, paths, counts


def set_hand_material(env, static_friction: float, dynamic_friction: float, restitution: float = 0.0):
    if dynamic_friction > static_friction:
        raise ValueError("dynamic friction cannot exceed static friction")
    sim = env.simulator
    report = {}
    for key, art in (("robot", sim._robot), ("robot_1", getattr(sim, "_robot_1", None))):
        if art is None:
            continue
        mapping, paths, counts = shapes_per_body(art)
        idx = [i for b in HAND_BODIES for i in mapping[b]]
        if len(idx) != 2:
            raise RuntimeError(f"expected one collision shape per rubber hand, got {[(b, mapping[b]) for b in HAND_BODIES]}")
        view = art.root_physx_view
        before = view.get_material_properties().clone()
        mats = before.clone()
        mats[:, idx, 0] = float(static_friction)
        mats[:, idx, 1] = float(dynamic_friction)
        mats[:, idx, 2] = float(restitution)
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")
        view.set_material_properties(mats, env_ids)
        after = view.get_material_properties()
        if not torch.equal(after, mats):
            raise RuntimeError("PhysX did not retain the requested hand material")
        changed = sorted(set((after != before).any(-1).nonzero()[:, 1].tolist()))
        if not set(changed) <= set(idx):
            raise RuntimeError(f"material changed outside the hand shapes: {changed} vs {idx}")
        body_of = {i: b for b, ii in mapping.items() for i in ii}
        report[key] = dict(link_paths=paths, shapes_per_body=counts, hand_shape_indices=idx, changed_shape_indices=changed,
                           changed_bodies=[body_of[i] for i in changed], before=before[0].tolist(), after=after[0].tolist(),
                           unchanged_shapes_identical=bool(torch.equal(after[:, [i for i in range(after.shape[1]) if i not in idx]],
                                                                       before[:, [i for i in range(before.shape[1]) if i not in idx]])))
    return dict(static_friction=static_friction, dynamic_friction=dynamic_friction, restitution=restitution, hand_bodies=list(HAND_BODIES),
                api="PhysX tensor API set_material_properties (as holosoma's object material); combine mode of these materials: PhysX default",
                per_robot=report)
