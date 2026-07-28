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

## Later training gates

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
transitions per run. Multiple seeds are started only after the paired
single-seed pilots and the 8,000-iteration gate pass.
