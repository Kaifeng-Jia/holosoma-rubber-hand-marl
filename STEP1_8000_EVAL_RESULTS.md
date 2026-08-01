# Step 1 Paired 8,000-Iteration Physics Evaluation

## Status and scope

- Evaluation date: 2026-07-31
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

The audit also found an asset/reward semantic mismatch in the exact-main
control. The original allowed-contact expression excluded
`left_wrist_yaw_link` and `right_wrist_yaw_link` from the undesired-contact
penalty, while the added collision bodies are named `left_rubber_hand_link`
and `right_rubber_hand_link`. The two rubber-hand bodies were consequently
penalized during these 8,000-iteration runs.

The post-audit semantic recovery only adds those two rubber-hand body names to
the allowed-contact expression. It does not add a special leg, knee, hip, or
torso penalty and does not change any reward weight or formula. The six results
in this document remain the unmodified exact-main control; training after the
semantic recovery must be reported separately.

Evaluation should primarily measure similarity to each frozen retargeted ideal
motion: robot joint/body tracking, rubber-hand pose relative to the table,
table trajectory, and standing stability. Contact channels remain diagnostic
unless pairwise contact attribution is added.

The table mass remains frozen at 0.1 kg for this low-resistance stage. This is
appropriate for testing stable stance, reference imitation, and a short initial
push, but it is not evidence of load-robust pushing. Heavier-table tests belong
to a later, separately reported robustness stage.

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
winner. It does show that A1 is presently the more stable all-round tracking
baseline. Plan B retains useful object-orientation and torque behavior, but its
task completion cannot be treated as correct hand-push technique after the
visual audit.

## Gate decision and next question

The 8,000-iteration progression gate passes technically and shows substantial
learning progress for both policies. It is not yet the final strong baseline:

- mean completion remains below 90%;
- Plan B has a 60% worst-seed completion result; and
- hip/knee invalid contacts remain frequent.

Before starting the full 30,000-iteration runs, freeze the final numeric
acceptance thresholds and validate the rubber-hand semantic-recovery variant at
a smaller gate. Any result produced after the allowed-contact correction must
remain separate from the exact-main control.
