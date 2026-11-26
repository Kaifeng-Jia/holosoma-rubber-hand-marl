#!/usr/bin/env python3
"""T14 step 9g item 3 (Kevin 2026-11-26): which constraint group makes the partner's Stage-1 solve of smallboxa infeasible
(frame 105). Report only: the solver is not changed. Read only on the step-9g inputs; writes only under --out-dir.

Method: role_solve.py is run with the failing level's exact command (output dir moved to --out-dir), with three read-only hooks
on the retargeter (holosoma_retargeting.src.interaction_mesh_retargeter):
  * cvxpy.Problem is recorded when solve_single_iteration builds it (objective, constraint list in construction order);
  * _update_jacobians_and_phis_from_q / _compute_self_collision_constraints record their pairs and signed distances;
  * solve_single_iteration: when it raises "CVXPY solve failed: infeasible", the recorded QP is analysed, then the error is
    re-raised (so the run ends as before).
Constraint groups in construction order (solve_single_iteration): Laplacian equality (1, always satisfiable through its free
variable) | foot constraints (nominal foot height 2 per foot link when enabled, foot sticking XY 2 per sticking foot, foot-lock Z
2 per locked foot) | non-penetration, one per pair within the collision threshold (robot-object pairs, robot-ground pairs) |
self-collision | joint limits (2) | step size (one second-order cone ||dq_a|| <= step_size). The analysis:
  1. feasibility (objective 0, Clarabel) of: all groups; all groups but one (each group left out in turn); each group alone
     (with the Laplacian equality);
  2. L1 elastic relaxation: every inequality row gets its own non-negative slack (the Laplacian equality and the step-size cone
     stay hard; dq_a = 0 satisfies both), minimise the sum of the slacks; per group the slack sum, and the rows that need slack
     (pair geoms and their signed distance, joint names) -> the smallest total violation and where it sits;
  3. the state at the failing call: frame, iteration of that frame, foot sticking flags, the closest pairs per group, joints at
     a limit.

    hsretargeting_s1/bin/python -B system1/t14/objects/step9g_s105_diag.py --out-dir logs_system1/t14/step9g_s105/m1p00 -- \
        <role_solve.py arguments without --out>
"""
from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))

import cvxpy  # noqa: E402
import holosoma_retargeting.src.interaction_mesh_retargeter as imr  # noqa: E402

REC = dict(problems=[], phis=[], sc=[], calls=0, frame_calls={}, out=None, done=False)


class CP:
    """the retargeter module's cvxpy, with Problem recorded."""

    def __getattr__(self, n):
        return getattr(cvxpy, n)

    @staticmethod
    def Problem(objective, constraints=None):
        REC["problems"].append((objective, list(constraints or [])))
        return cvxpy.Problem(objective, constraints)


imr.cp = CP()
K = imr.InteractionMeshRetargeter
_upd, _sc, _solve = K._update_jacobians_and_phis_from_q, K._compute_self_collision_constraints, K.solve_single_iteration


def upd(self, q):
    Js, phis = _upd(self, q)
    REC["phis"].append((np.array(q, float), dict(phis)))
    return Js, phis


def sc(self, frame_idx):
    Js, phis = _sc(self, frame_idx)
    REC["sc"].append(dict(phis))
    return Js, phis


def solve(self, *args, **kw):
    REC["problems"].clear()
    REC["phis"].clear()
    REC["sc"].clear()
    f = kw.get("frame_idx", 0)
    REC["frame_calls"][f] = REC["frame_calls"].get(f, 0) + 1
    try:
        return _solve(self, *args, **kw)
    except RuntimeError as e:
        if "infeasible" in str(e) and not REC["done"]:
            REC["done"] = True
            t0 = time.time()
            res = analyse(self, kw)
            res["analysis_seconds"] = round(time.time() - t0, 1)
            res["error"] = str(e)
            REC["out"].write_text(json.dumps(res, indent=1, default=str) + "\n")
            print("S105_DIAG_WRITTEN", REC["out"], flush=True)
        raise


K._update_jacobians_and_phis_from_q, K._compute_self_collision_constraints, K.solve_single_iteration = upd, sc, solve


def joint_names(self):
    import mujoco
    m = self.robot_model
    by_q = {}
    for j in range(m.njnt):
        a = int(m.jnt_qposadr[j])
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            for i, s in enumerate(("x", "y", "z", "qw", "qx", "qy", "qz")):
                by_q[a + i] = f"{nm}:{s}"
        else:
            by_q[a] = nm
    return [by_q.get(int(i), f"q{int(i)}") for i in self.q_a_indices]


def status(cons):
    p = cvxpy.Problem(cvxpy.Minimize(0), cons)
    try:
        p.solve(solver=cvxpy.CLARABEL)
    except Exception as ex:  # noqa: BLE001
        return f"error: {ex}"
    return p.status


def analyse(self, kw):
    obj, cons = REC["problems"][0]
    retried = len(REC["problems"]) > 1
    q, phis = REC["phis"][-1]
    scp = REC["sc"][-1] if REC["sc"] else {}
    np_keys = [k for k in phis if not (self._is_object_collision_pair(k) and not self.activate_obj_non_penetration)]
    Np, Ns = len(np_keys), len(scp)
    JL = 2 if self.activate_joint_limits else 0
    assert isinstance(cons[-1], cvxpy.constraints.second_order.SOC), type(cons[-1])
    Nf = len(cons) - 1 - Np - Ns - JL - 1
    assert Nf >= 0, (len(cons), Np, Ns, JL)
    gn = self._geom_names
    fs = kw.get("foot_sticking") or {}
    feet = list(self.foot_links)
    apply_h = (self.q_a_init_idx < 12) and self.anchor_nominal_foot_height and kw.get("q_a_nominal") is not None
    nh = 2 * len(feet) if apply_h else 0
    lk = next((k for k in fs if k.lower().startswith("l")), None)
    rk = next((k for k in fs if k.lower().startswith("r")), None)
    nst = 2 * sum(1 for key in feet if ("left" in key and lk is not None and fs[lk]) or ("right" in key and rk is not None and fs[rk])) \
        if (self.q_a_init_idx < 12 and self.activate_foot_sticking) else 0
    nlk = Nf - nh - nst
    labels = ["laplacian equality"] + ["foot height"] * nh + ["foot sticking"] * nst + ["foot lock"] * nlk
    pair_of = {}
    for k in np_keys:
        labels.append("non-penetration object" if self._is_object_collision_pair(k) else "non-penetration ground")
        pair_of[len(labels) - 1] = k
    for k in scp:
        labels.append("self-collision")
        pair_of[len(labels) - 1] = k
    labels += ["joint limit lower", "joint limit upper"][:JL] + ["step size"]
    assert len(labels) == len(cons), (len(labels), len(cons))
    groups = list(dict.fromkeys(labels))
    idx = {g: [i for i, l in enumerate(labels) if l == g] for g in groups}
    jn = joint_names(self)
    # 1. feasibility
    feas = dict(all=status(cons))
    for g in groups:
        if g == "laplacian equality":
            continue
        feas[f"all but {g}"] = status([c for i, c in enumerate(cons) if labels[i] != g])
        feas[f"{g} alone"] = status([cons[0]] + [cons[i] for i in idx[g]])
    for a_, b_ in (("non-penetration object", "step size"), ("non-penetration object", "joint limit lower"),
                   ("non-penetration object", "joint limit upper"), ("self-collision", "step size")):
        if a_ in idx and b_ in idx:
            feas[f"{a_} + {b_}"] = status([cons[0]] + [cons[i] for i in idx[a_] + idx[b_]])
    # 2. L1 elastic relaxation
    slack, new = {}, [cons[0], cons[-1]]
    for i, c in enumerate(cons[1:-1], start=1):
        assert isinstance(c, cvxpy.constraints.Inequality), (i, type(c))
        s = cvxpy.Variable(c.expr.shape, nonneg=True)
        slack[i] = s
        new.append(c.expr <= s)
    p = cvxpy.Problem(cvxpy.Minimize(sum(cvxpy.sum(s) for s in slack.values())), new)
    p.solve(solver=cvxpy.CLARABEL)
    rows = []
    per_group = {}
    for i, s in slack.items():
        v = np.atleast_1d(np.asarray(s.value, float))
        g = labels[i]
        per_group.setdefault(g, dict(rows=0, slack_sum=0.0, slack_max=0.0, rows_with_slack=0))
        per_group[g]["rows"] += v.size
        per_group[g]["slack_sum"] += float(v.sum())
        per_group[g]["slack_max"] = max(per_group[g]["slack_max"], float(v.max()))
        for j in np.flatnonzero(v > 1e-7):
            per_group[g]["rows_with_slack"] += 1
            r = dict(group=g, row=i, entry=int(j), slack=round(float(v[j]), 6))
            if i in pair_of:
                k = pair_of[i]
                r.update(pair=[gn[k[0]], gn[k[1]]], signed_distance_m=round(float((phis if g != "self-collision" else scp)[k]), 6))
            if g.startswith("joint limit"):
                r["joint"] = jn[j]
            rows.append(r)
    for g in per_group.values():
        g["slack_sum"] = round(g["slack_sum"], 6)
        g["slack_max"] = round(g["slack_max"], 6)
    rows.sort(key=lambda r: -r["slack"])
    # 3. state
    qa = q[self.q_a_indices]
    at_lim = [dict(joint=jn[i], q=round(float(qa[i]), 4), lb=round(float(self.q_a_lb[i]), 4), ub=round(float(self.q_a_ub[i]), 4))
              for i in range(len(qa)) if qa[i] <= self.q_a_lb[i] + 1e-3 or qa[i] >= self.q_a_ub[i] - 1e-3]
    close = {}
    for g in ("non-penetration object", "non-penetration ground", "self-collision"):
        ks = [pair_of[i] for i in idx.get(g, [])]
        src = scp if g == "self-collision" else phis
        close[g] = [dict(pair=[gn[k[0]], gn[k[1]]], signed_distance_m=round(float(src[k]), 6)) for k in sorted(ks, key=lambda k: src[k])[:12]]
    f = kw.get("frame_idx", 0)
    return dict(rules=__doc__, frame=int(f), iteration_of_frame=REC["frame_calls"].get(f), init_t=bool(kw.get("init_t", False)),
                retried_without_step_size=retried, step_size=float(self.step_size), penetration_tolerance=float(self.penetration_tolerance),
                collision_detection_threshold=float(self.collision_detection_threshold), elastic=bool(self.elastic_constraints.enable),
                foot_sticking=fs, foot_links=feet, counts={g: len(idx[g]) for g in groups}, feasibility=feas,
                l1_relaxation=dict(status=p.status, total_slack=round(float(p.value), 6) if p.value is not None else None, per_group=per_group,
                                   rows_needing_slack=rows[:40]),
                joints_at_limit=at_lim, closest_pairs=close)


def main():
    argv = sys.argv[1:]
    i = argv.index("--out-dir")
    out_dir = Path(argv[i + 1])
    rest = argv[argv.index("--") + 1:]
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    REC["out"] = out_dir / "s105_diag.json"
    sys.argv = [str(HERE / "role_solve.py")] + rest + ["--out", str(out_dir / "r1")]
    try:
        runpy.run_path(str(HERE / "role_solve.py"), run_name="__main__")
    except SystemExit as e:
        print("role_solve exit", e.code, flush=True)
    except RuntimeError as e:
        print("role_solve RuntimeError", e, flush=True)
    return 0 if REC["done"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
