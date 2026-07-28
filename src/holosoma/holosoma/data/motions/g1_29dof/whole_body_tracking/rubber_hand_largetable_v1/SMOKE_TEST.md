# A1 and Plan B WBT Smoke Test

## Purpose

This is an end-to-end pipeline check for the frozen A1 and Plan B
rubber-hand/large-table motions. It verifies that each motion reaches:

1. motion and asset loading;
2. environment creation and reset;
3. observation, reward, and termination computation;
4. rollout collection;
5. one PPO update; and
6. checkpoint export.

It is not a policy-quality benchmark. Four environments and one learning
iteration are too small for comparing A1 with Plan B.

## Date and source state

```text
run date: 2026-07-27 (America/New_York)
branch: rubber_hand_largetable_baseline
motion asset commit: 349a088a
seed: 42
environments: 4
learning iterations: 1
```

## Common command

Run from the repository root after sourcing the Isaac Sim environment. Replace
`<motion-file>` with one of the two paths in the next section.

```bash
PYTHONPATH="$PWD/src/holosoma" \
OMNI_KIT_ACCEPT_EULA=1 \
PYTHONDONTWRITEBYTECODE=1 \
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object \
  --training.num-envs=4 \
  --training.headless=True \
  --training.seed=42 \
  --algo.config.num-learning-iterations=1 \
  --command.setup-terms.motion-command.params.motion-config.motion-file=<motion-file> \
  --robot.object.object-urdf-path=holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largetable.urdf
```

Motion paths:

```text
A1:
holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz

Plan B:
holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/plan_b/sub6_largetable_033_plan_b_mj_fps50_w_obj.npz
```

The large-table URDF override is required. The inherited experiment preset
otherwise selects `objects_largebox.urdf`.

## Results

Both runs exited successfully and wrote `model_00000.pt` and
`model_00000.onnx`.

| Check | A1 | Plan B |
|---|---:|---:|
| Motion load | passed | passed |
| Environment/reset | passed | passed |
| Reward/termination | passed | passed |
| Rollout collection | passed | passed |
| PPO update | passed | passed |
| Checkpoint export | passed | passed |
| Timesteps | 96 | 96 |
| Mean episode length | 11.75 | 15.50 |
| Mean reward | -0.42 | -0.53 |
| Collection time | 0.385 s | 0.375 s |
| Learning time | 0.069 s | 0.060 s |

Run directories (gitignored):

```text
A1:
logs/WholeBodyTracking/20260728_022020-g1_29dof_wbt_manager-locomotion

Plan B:
logs/WholeBodyTracking/20260728_022052-g1_29dof_wbt_manager-locomotion
```

The saved `holosoma_config.yaml` files were compared in full. The only
difference is the expected `motion_file` value. Therefore the two smoke runs
used the same seed and training/environment configuration.

The reward and error values above are diagnostics from an initial stochastic
rollout. They must not be interpreted as evidence that one motion is easier to
learn or yields a better policy.

## Current machine limitation

Isaac Sim reported:

```text
NVML_ERROR_LIB_RM_VERSION_MISMATCH
CUDA error 804: forward compatibility was attempted on non supported HW
No CUDA devices found
Environment device: cpu
```

The smoke tests therefore validate the code, data, environment, and PPO path
through Isaac Sim's CPU fallback. They do not validate GPU training readiness
or representative throughput. Resolve the host NVIDIA driver/library mismatch
and confirm that Isaac Sim selects a CUDA device before starting a long run.

## Non-blocking warnings observed

- unresolved visual prim references in the generated rubber-hand USD;
- Isaac Sim joint-name ordering warning;
- inability to modify articulation-root properties below an instanced object
  prim; and
- CPU collision-filtering warning caused by the fallback execution mode.

These warnings did not prevent either smoke test from completing. The
articulation and collision warnings should be rechecked once GPU execution is
restored and before treating a long training result as the baseline.
