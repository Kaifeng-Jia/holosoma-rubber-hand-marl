# Step 1 Rubber-Hand Single-Motion Baseline

## Scope

This document freezes the single-motion A1 and Plan B pilot that precedes the
mixed-data Step 1 policy in `LongTermGoal.md`.

The pilot answers:

> Can the original object-aware `main` WBT setup learn each frozen
> rubber-hand/large-table reference without a pipeline, numerical, or early
> physical-interaction failure?

It does not establish convergence and must not be used as the final A1 versus
Plan B comparison.

The single-motion push pilot was formally closed on 2026-08-05. A1 is retained
as the accepted push prior, while Plan B is retained as a distinct behavioral
prior whose palm-edge reference was not reproduced by the learned policy.
Neither policy is scheduled for additional single-motion training.

## Four-action WBT milestone

The warm-start/pre-training milestone was completed on 2026-08-07. Four
separate 8,000-iteration WBT checkpoints are preserved for downstream task
learning:

| Prior | Frozen checkpoint | SHA-256 | Interpretation |
|---|---|---|---|
| Push A1 | `logs/WholeBodyTracking/20260728_051638-rubberhand_a1_8000_seed42-locomotion/model_07999.pt` | `54fbfa4b38ee69ac1be33da8b5d3f6849e587b6bd02da3f407b0b27844be3c25` | Accepted push prior |
| Push Plan B | `logs/WholeBodyTracking/20260728_132654-rubberhand_plan_b_8000_seed42-locomotion/model_07999.pt` | `5ba3d28bfb862321812e1f9475f69371b40ee657d639ef0a3488fbccc4a5e89e` | Distinct push/contact prior with documented palm-edge mismatch |
| Kick | `logs/WholeBodyTracking/20260806_055921-rubberhand_kick_sub16_028_8000_seed42-locomotion/model_07999.pt` | `a84928cf4a8ba6ca91905a4105ff82bdb331f11ac0acb6ab56a034f5b6660e47` | Directionally effective kick/leg-contact prior |
| Pull | `logs/WholeBodyTracking/20260807_071217-rubberhand_pull_sub3_010_8000_seed42-locomotion/model_07999.pt` | `fc5a5d3b66bc076598b28f6c1e8ad822798ce6ce44c9f0370056dc8ad44495dd` | Feasible pull prior with accepted leg/corner variants |

Here, "completed WBT" means that all four retained reference motions have a
reproducible full-length PPO pre-training checkpoint suitable for warm-starting
later experiments. It does not mean that every checkpoint passed the stricter
goal-conditioned expert gate, nor that each learned policy uses only the body
part implied by its action label. The exact motions, evaluations, known
mismatches, artifact paths, and hashes are frozen in
[`STEP1_MOTION_MANIFEST.yaml`](STEP1_MOTION_MANIFEST.yaml).

## Frozen source state

```text
branch: rubber_hand_largetable_baseline
A1/Plan B training assets: commit 349a088a
CUDA recovery validation: commit 2cdb5fd5
simulator: Isaac Sim
algorithm: PPO
experiment: exp:g1-29dof-wbt-w-object
reward design: inherited from main without modification
robot: G1 29-DoF with rubber hands
object: objects_largetable.urdf
table base mass: 0.1 kg
training mass randomization: add 1--4 kg (effective approximately 1.1--4.1 kg)
```

## GPU capacity calibration

All rows used A1, seed 42, 24 rollout steps per environment, and the original
PPO batch structure. Peak framebuffer memory was sampled with `nvidia-smi
dmon`.

| Environments | Iterations | Timesteps/iteration | Throughput | Peak VRAM |
|---:|---:|---:|---:|---:|
| 32 | 1 | 768 | 545 steps/s | 3.02 GB |
| 128 | 1 | 3,072 | 1,575 steps/s | 3.03 GB |
| 512 | 1 | 12,288 | 5,103 steps/s | 3.44 GB |
| 1,024 | 1 | 24,576 | 8,021 steps/s | 3.75 GB |
| 1,024 | 10 | 24,576 | 7,415–10,119 steps/s | 3.93 GB |
| 2,048 | 1 | 49,152 | 13,669 steps/s | 4.66 GB |
| 4,096 | 1 | 98,304 | 18,530 steps/s | 6.23 GB |
| 4,096 | 10 | 98,304 | 17,863–22,581 steps/s | 6.25 GB |

The 4,096-environment run completed ten consecutive iterations without a CUDA,
PhysX, out-of-memory, or numerical failure. It preserves the original `main`
rollout batch:

```text
4096 environments × 24 rollout steps = 98,304 transitions/iteration
```

## Pilot matrix

Only the motion file and run name differ.

| Run | Motion | Seed | Environments | Iterations | Save interval |
|---|---|---:|---:|---:|---:|
| A1 pilot | A1 | 42 | 4,096 | 500 | 100 |
| Plan B pilot | Plan B | 42 | 4,096 | 500 | 100 |

Each pilot processes:

```text
98,304 transitions/iteration × 500 iterations = 49,152,000 transitions
```

The save interval is shortened only to inspect early learning. It does not
change rollout collection, reward, PPO optimization, or policy inputs.

## Commands

Run from the repository root in the `hssim` environment.

A1:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object \
  --training.name=rubberhand_a1_pilot_seed42 \
  --training.num-envs=4096 \
  --training.headless=True \
  --training.seed=42 \
  --algo.config.num-learning-iterations=500 \
  --algo.config.save-interval=100 \
  --command.setup-terms.motion-command.params.motion-config.motion-file=holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz \
  --robot.object.object-urdf-path=holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largetable.urdf
```

Plan B:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object \
  --training.name=rubberhand_plan_b_pilot_seed42 \
  --training.num-envs=4096 \
  --training.headless=True \
  --training.seed=42 \
  --algo.config.num-learning-iterations=500 \
  --algo.config.save-interval=100 \
  --command.setup-terms.motion-command.params.motion-config.motion-file=holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/plan_b/sub6_largetable_033_plan_b_mj_fps50_w_obj.npz \
  --robot.object.object-urdf-path=holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largetable.urdf
```

## Pilot acceptance

A pilot passes the pipeline gate when:

1. all 500 iterations complete on `cuda:0`;
2. no NaN, infinity, CUDA, PhysX, or out-of-memory failure occurs;
3. checkpoints are written at the expected intervals;
4. reward and episode-length curves show no catastrophic collapse;
5. motion, body, joint, and object tracking diagnostics remain finite; and
6. a checkpoint can be loaded for deterministic physics-based evaluation.

A pass does not mean the policy has converged.

## Pilot results

Both pilots completed on 2026-07-28 and processed the full frozen budget:

```text
A1:
  logs/WholeBodyTracking/20260728_031249-rubberhand_a1_pilot_seed42-locomotion

Plan B:
  logs/WholeBodyTracking/20260728_035030-rubberhand_plan_b_pilot_seed42-locomotion

iterations per run: 500
transitions per run: 49,152,000
device: cuda:0
```

The saved `holosoma_config.yaml` files were compared in full. Their only
differences are the expected run `name` and `motion_file`.

Both runs wrote checkpoints and ONNX exports at iterations 0, 100, 200, 300,
400, and 499. All 500 recorded values of every metric below are finite.

The following table reports the mean over the final 20 iterations. These are
training diagnostics, not deterministic policy-evaluation scores.

| Metric | A1 | Plan B |
|---|---:|---:|
| Mean reward | 14.6870 | 16.2368 |
| Mean episode length | 244.503 | 283.897 |
| Global reference position error | 0.175157 | 0.195329 |
| Global reference rotation error | 0.187159 | 0.193931 |
| Relative body position error | 0.074190 | 0.078144 |
| Relative body rotation error | 0.338084 | 0.345862 |
| Joint position error | 1.39447 | 1.53521 |
| Joint velocity error | 13.4723 | 13.8109 |
| Raw episode object-position reward | 11.2243 | 12.6206 |
| Raw episode object-orientation reward | 10.7764 | 12.0680 |

Plan B's larger raw episode object rewards partly reflect its longer episodes;
they must not be interpreted as lower instantaneous table error. At this early
gate, A1 has lower tracking errors while Plan B has longer episodes and a
higher aggregate reward. The runs are not converged, so this is a diagnostic
tradeoff rather than a winner selection.

The final checkpoints were also loaded through `eval_agent.py` with one
environment, evaluation phase fixed to the start of the motion, and 309 control
steps:

```text
A1 model_00499.pt: passed
Plan B model_00499.pt: passed
```

Both evaluations loaded the saved experiment configuration, selected
`cuda:0`, loaded the correct motion file, executed all requested steps, and
shut down normally.

Pilot gate result:

```text
A1: passed
Plan B: passed
```

This result authorizes the 8,000-iteration single-seed comparison gate. It does
not authorize an A1-versus-Plan-B quality conclusion without fixed
physics-based evaluation metrics.

The fixed recorder, metrics, attempt definition, thresholds, and paired
evaluation procedure are documented in
[`STEP1_PHYSICS_EVAL.md`](STEP1_PHYSICS_EVAL.md). Validation on the two
500-iteration checkpoints confirms that the policies can be evaluated with
actual table state and rubber-hand contact-sensor data. It also shows high
early-termination rates, reinforcing that these checkpoints are pipeline
pilots rather than converged policies.

The paired 8,000-iteration checkpoints and their three-seed physics evaluation
are documented in
[`STEP1_8000_EVAL_RESULTS.md`](STEP1_8000_EVAL_RESULTS.md). Both policies pass
the progression gate and improve substantially over their 500-iteration
pilots. A1 is currently more stable across evaluation seeds, while Plan B has
lower table-orientation error and mean torque utilization. Neither result is
treated as final Step 1 convergence.

The frozen checkpoints, evaluation recordings, tensor-equivalence evidence,
motion assets, and hashes are indexed in
[`STEP1_PUSH_PILOT_ARTIFACT_INDEX.md`](STEP1_PUSH_PILOT_ARTIFACT_INDEX.md).

## Pilot closure decision

The push pilot established the required engineering result:

- the A1 and Plan B retargeted motions can train under the inherited PPO WBT
  pipeline;
- the policies execute physically coupled robot/table trajectories;
- A1 is the more stable all-round push prior;
- Plan B exposes objective ambiguity: its valid palm-edge reference leads to a
  policy that often uses top contact and leg/hip bracing instead;
- the later rubber-hand allowed-contact regex change is behaviorally inert
  because the reward body list excludes the fixed rubber-hand links;
- completion and hand-force coincidence do not identify which body caused
  object motion.

These findings are sufficient to proceed to the multi-behavior prior library.
Perfect palm-edge imitation and advanced object-structure understanding are not
prerequisites for the multi-agent research direction.

## Historical training gates

```text
500 iterations:
  early pilot only

4,000 iterations:
  first original-main checkpoint boundary

8,000 iterations:
  original nightly-scale, single-seed comparison gate

30,000 iterations:
  complete original-main training budget
```

At 4,096 environments, 30,000 iterations equal 2,949,120,000 simulated
transitions per run. This historical budget is no longer scheduled for the A1
or Plan B single-motion push policies. New skills use staged 50, 500, 2,000,
and 8,000-iteration gates, and advance only while both numeric and semantic
checks pass.
