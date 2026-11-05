# Step 1 Paired 8,000-Iteration Physics Evaluation

## Status and scope

- Evaluation date: 2026-07-31
- Pilot closure date: 2026-08-05
- Protocol: [`STEP1_PHYSICS_EVAL.md`](STEP1_PHYSICS_EVAL.md), schema `holosoma.wbt_physics_eval.v1`
- Training seed: 42 for both checkpoints
- Evaluation seeds: 42, 43, 44
- Evaluated attempts: first 20 closed attempts per seed
- Total evaluated attempts: 60 per checkpoint
- Policy steps recorded per seed: 6,500

This is the paired physics evaluation of the A1 and Plan B
`model_07999.pt` checkpoints. Evaluation seeds randomize the environment and
initial state; they are not independent training seeds.

## Checkpoints

```text
A1:
logs/WholeBodyTracking/20260728_051638-rubberhand_a1_8000_seed42-locomotion/
  model_07999.pt

Plan B:
logs/WholeBodyTracking/20260728_132654-rubberhand_plan_b_8000_seed42-locomotion/
  model_07999.pt
```

## Evaluation artifacts

| Policy | Eval seed | Recording and JSON directory |
|---|---:|---|
| A1 | 42 | `logs/WholeBodyTracking/20260731_224124-rubberhand_a1_8000_seed42-eval/` |
| A1 | 43 | `logs/WholeBodyTracking/20260731_224602-rubberhand_a1_8000_seed42-eval/` |
| A1 | 44 | `logs/WholeBodyTracking/20260731_225034-rubberhand_a1_8000_seed42-eval/` |
| Plan B | 42 | `logs/WholeBodyTracking/20260731_225505-rubberhand_plan_b_8000_seed42-eval/` |
| Plan B | 43 | `logs/WholeBodyTracking/20260731_225937-rubberhand_plan_b_8000_seed42-eval/` |
| Plan B | 44 | `logs/WholeBodyTracking/20260731_230410-rubberhand_plan_b_8000_seed42-eval/` |

All six runs:

1. loaded the intended checkpoint and motion file on `cuda:0`;
2. completed all 6,500 requested policy steps;
3. recorded reference, robot, table, termination, torque, and 39-body raw
   contact-sensor channels;
4. detected at least 22 closed attempts; and
5. produced a finite metric summary over the first 20 attempts.

No CUDA, PhysX, NaN, timeout, channel, or analyzer failure occurred.

## Per-seed results

| Policy | Seed | Completion | Early termination | Mean length | Correct direction | Hand contact | Interaction proxy | Invalid contact | Fall proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 42 | 80% | 20% | 278.45 | 100% | 100% | 100% | 48.21% | 0% |
| A1 | 43 | 85% | 15% | 287.10 | 100% | 100% | 100% | 55.92% | 5% |
| A1 | 44 | 80% | 20% | 269.05 | 100% | 100% | 90% | 34.75% | 5% |
| Plan B | 42 | 85% | 15% | 274.20 | 100% | 100% | 90% | 54.32% | 0% |
| Plan B | 43 | 60% | 40% | 235.70 | 100% | 100% | 85% | 60.22% | 5% |
| Plan B | 44 | 85% | 15% | 286.85 | 95% | 100% | 95% | 43.80% | 5% |

Plan B's seed-43 completion drop is visible rather than hidden by its
three-seed mean.

## Aggregate results

Values are mean +/- sample standard deviation across the three evaluation
seeds. Each seed contributes the same 20 attempts.

| Metric | A1 | Plan B | Preferred direction |
|---|---:|---:|---|
| Full-motion completion | 81.67 +/- 2.89% | 76.67 +/- 14.43% | Higher |
| Early termination | 18.33 +/- 2.89% | 23.33 +/- 14.43% | Lower |
| Mean attempt length | 278.20 +/- 9.03 steps | 265.58 +/- 26.64 steps | Higher |
| Joint-position RMSE | 0.1741 +/- 0.0028 rad | 0.2351 +/- 0.0027 rad | Lower |
| Key-body position RMSE | 0.0346 +/- 0.0030 m | 0.0426 +/- 0.0016 m | Lower |
| Key-body orientation mean | 11.98 +/- 0.22 deg | 14.32 +/- 0.11 deg | Lower |
| Table trajectory position RMSE | 0.0590 +/- 0.0061 m | 0.0669 +/- 0.0078 m | Lower |
| Table orientation mean | 8.02 +/- 0.97 deg | 6.45 +/- 1.30 deg | Lower |
| Correct displacement direction | 100.00 +/- 0.00% | 98.33 +/- 2.89% | Higher |
| Displacement-direction cosine | 0.9944 +/- 0.0069 | 0.9974 +/- 0.0023 | Higher |
| Rubber-hand contact attempts | 100.00 +/- 0.00% | 100.00 +/- 0.00% | Higher |
| Interaction-proxy success | 96.67 +/- 5.77% | 90.00 +/- 5.00% | Higher |
| Invalid-contact step fraction | 46.30 +/- 10.71% | 52.78 +/- 8.32% | Lower |
| Fall-proxy attempts | 3.33 +/- 2.89% | 3.33 +/- 2.89% | Lower |
| Action-delta RMSE | 0.2190 +/- 0.0023 | 0.2271 +/- 0.0120 | Lower |
| Mean torque utilization | 0.1193 +/- 0.0058 | 0.1036 +/- 0.0026 | Lower |

## Change from the 500-iteration pilots

The 500-iteration numbers used one evaluation seed and therefore are a
diagnostic reference, not a matched three-seed confidence comparison.

| Metric | A1: 500 -> 8,000 | Plan B: 500 -> 8,000 |
|---|---:|---:|
| Completion | 10.0% -> 81.7% | 5.0% -> 76.7% |
| Early termination | 90.0% -> 18.3% | 95.0% -> 23.3% |
| Joint-position RMSE | 0.3080 -> 0.1741 rad | 0.2724 -> 0.2351 rad |
| Key-body position RMSE | 0.0659 -> 0.0346 m | 0.0738 -> 0.0426 m |
| Table position RMSE | 0.0681 -> 0.0590 m | 0.0730 -> 0.0669 m |
| Interaction proxy | 70.0% -> 96.7% | 65.0% -> 90.0% |
| Fall proxy | 25.0% -> 3.3% | 65.0% -> 3.3% |
| Action-delta RMSE | 0.2160 -> 0.2190 | 0.2940 -> 0.2271 |

Both policies show clear physical-learning progress. The paired
8,000-iteration progression gate passes for both A1 and Plan B.

## Invalid-contact diagnosis

The invalid-contact metric remains high. It is computed from the union of all
non-allowed robot bodies exceeding 5 N, so per-body fractions below do not sum
to the union fraction.

Mean per-step contact fractions across evaluation seeds:

| A1 body | Fraction | Plan B body | Fraction |
|---|---:|---|---:|
| `left_knee_link` | 15.90% | `right_hip_yaw_link` | 20.05% |
| `right_hip_yaw_link` | 15.61% | `left_hip_yaw_link` | 18.62% |
| `left_hip_yaw_link` | 11.62% | `right_knee_link` | 12.14% |
| `right_knee_link` | 6.61% | `left_knee_link` | 7.18% |
| `left_wrist_pitch_link` | 0.71% | `torso_link` | 1.35% |

The residual contact problem is primarily a hip/knee issue, not a missing
rubber-hand sensor. The body-level net-force signal still cannot distinguish
self-collision, table contact, and ground contact without pairwise PhysX
reports, so visual replay of representative failed attempts remains necessary.

## Post-evaluation visual audit

Visual replay changed the interpretation of Plan B's task-level completion
metric. In the seed-43 attempt 13 replay, the robot maintained hand contact but
also braced or pushed the table with its leg/hip region. The attempt completed
the table trajectory, but it did not reproduce the intended hand-dominant Plan
B technique.

The recording supports this diagnosis without proving pairwise contact: the
`right_hip_yaw_link` exceeded 5 N in 33.1% of the attempt and overlapped table
motion in 32.8% of the attempt. Both rubber hands also carried external force.
Consequently, the current `hand contact + object motion` interaction proxy
detects temporal coincidence, not which body caused the table displacement.
Plan B's reported completion rate must therefore not be presented as a
hand-push success rate.

The audit initially suspected an asset/reward semantic mismatch because the
allowed-contact expression named the wrist-yaw links rather than the fixed
rubber-hand links. Follow-up inspection disproved that interpretation.
`UndesiredContacts` selects bodies from the simulator's 32 articulation-body
list, which does not contain `left_rubber_hand_link` or
`right_rubber_hand_link`. The 39-body contact sensor records those fixed links,
but the reward term never indexes them. Therefore, the exact-main runs did not
directly penalize rubber-hand contact through this term.

The later regex change that explicitly excluded both rubber-hand names was a
no-op for training behavior. From-scratch 8,000-iteration reruns saved the
changed configuration, but all 76 checkpoint tensors were identical to the
corresponding exact-main checkpoint. Their seed-43 evaluation recordings were
also byte-identical. These reruns are reproducibility confirmations, not a
separate "semantic recovery" method. Artifact paths and hashes are frozen in
[`STEP1_PUSH_PILOT_ARTIFACT_INDEX.md`](STEP1_PUSH_PILOT_ARTIFACT_INDEX.md).

Evaluation should primarily measure similarity to each frozen retargeted ideal
motion: robot joint/body tracking, rubber-hand pose relative to the table,
table trajectory, and standing stability. Contact channels remain diagnostic
unless pairwise contact attribution is added.

The table URDF base mass is 0.1 kg, but the object randomizer uses an additive
mass operation with parameters `[1.0, 4.0]`. The effective training mass range
is therefore approximately 1.1--4.1 kg, not a fixed 0.1 kg. These results test
the inherited exact-main mass distribution. Any future fixed-mass or expanded
robustness experiment must be labeled as a different configuration.

## Interpretation

A1 currently has:

- more stable completion across evaluation seeds;
- lower joint and key-body tracking error;
- lower table position error;
- higher interaction-proxy success;
- fewer invalid-contact steps; and
- slightly smoother actions.

Plan B currently has:

- lower table orientation error;
- slightly better displacement-direction cosine; and
- lower mean torque utilization.

These are complementary tradeoffs. The result does not justify deleting
either reference or declaring that training reward selected a universal
winner. A1 is accepted as the current push-motion prior. Plan B is retained as
a distinct behavioral prior and as evidence that a kinematically valid
palm-edge target does not uniquely determine the learned physical technique.
Its task completion cannot be treated as correct hand-push imitation, but its
alternative hand, leg, and hip behavior remains relevant to later multi-agent
emergence experiments.

## Gate decision and pilot closure

The 8,000-iteration progression gate passes technically and shows substantial
learning progress for both policies. It is not the final mixed-data Step 1
baseline:

- mean completion remains below 90%;
- Plan B has a 60% worst-seed completion result; and
- hip/knee invalid contacts remain frequent.

The single-motion push pilot is now closed. No further A1 or Plan B training is
scheduled: additional iterations would not resolve the reference/contact
ambiguity demonstrated by the visual and geometric audit. The next active work
is to build a versioned behavior-prior library, beginning with kick and a
bounded pull feasibility check, before training the shared mixed-motion prior.
