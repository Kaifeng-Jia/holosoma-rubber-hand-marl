# Step 1 Physics Evaluation Protocol

## Status and scope

This protocol is frozen before the paired 8,000-iteration A1 and Plan B
training gate.

It evaluates whether a learned policy physically executes a WBT reference. It
does not judge retargeting quality from a kinematic Viser replay, and it does
not use training reward as a substitute for physical evaluation.

The protocol applies equally to A1 and Plan B:

- the same robot and table assets;
- the same checkpoint budget;
- the same evaluation seeds;
- the same number of closed motion attempts;
- the same metric formulas and thresholds; and
- no A1- or Plan-B-specific reward or success rule.

## Evaluation unit

The frozen motion contains 309 frames at 50 Hz. One policy step is 0.02 s and
contains four 0.005 s physics substeps.

`MotionCommand` loops a completed clip internally without emitting an
environment `done`. Therefore, an attempt is reconstructed from the recorded
motion-frame sequence:

- a high-frame-to-low-frame wrap with no termination is a completed attempt;
- a reset carrying `done=True` is an early tracking termination;
- a terminal attempt must reach frame 307 or later to count as complete; and
- an unfinished tail at the end of a recording is excluded.

The primary 8,000-iteration evaluation uses the first 20 closed attempts from
each recording. The simulator runs for 6,500 policy steps so that even a
policy completing every 308-step attempt supplies at least 20 closed attempts.
An evaluation is invalid if fewer than 20 closed attempts are available.

## Evaluation distribution

The primary gate uses evaluation seeds 42, 43, and 44:

```text
20 attempts/seed × 3 seeds = 60 attempts/checkpoint
```

Evaluation mode fixes motion sampling to the beginning of the clip and disables
scheduled external pushes. The saved training distribution is otherwise
retained: startup material/mass/CoM randomization and reset-time initial-pose
noise remain active. A1 and Plan B are evaluated with the same seed list.

This is a seeded in-distribution physics evaluation, not a noise-free
kinematic replay.

## Recorded state

`EvalRecordingCallback` records the state used to calculate each action before
the physics transition, plus transition results after the step. Pre-step
recording is required because a failed WBT environment resets inside
`env.step`; a post-step-only recorder would replace the failure state with the
next attempt's initial state.

The NPZ contains:

- reference and actual joint position/velocity;
- reference and actual 14 tracked-body poses;
- root pose;
- reference and actual table pose/velocity;
- policy actions, PD targets, torques, and physics-substep states;
- reward, done, timeout, and non-timeout termination flags;
- motion ID, motion frame, and environment episode step; and
- raw current/history contact force for all 39 contact-sensor robot bodies.

The raw contact list includes `left_rubber_hand_link` and
`right_rubber_hand_link` directly. The older 32-body simulator tensor omits
these fixed hand links and must not be used as the primary rubber-hand contact
signal.

## Frozen metrics

### Completion and termination

- Full-motion completion rate.
- Early tracking-termination rate.
- Mean closed-attempt length.
- Timeout count.

The current WBT termination configuration aggregates tracking failures under
`bad_tracking`; it does not expose a more specific per-term cause.

### Robot tracking

- Joint-position RMSE and MAE in radians.
- Key-body position RMSE in metres.
- Mean key-body quaternion error in radians and degrees.

Quaternion error uses the shortest rotation and treats `q` and `-q` as the
same orientation.

### Table motion

- Table trajectory position RMSE.
- Mean table orientation error.
- Actual and reference displacement magnitude per attempt.
- Displacement-direction cosine.
- Correct-direction rate for attempts whose reference displacement is at least
  0.01 m.

Correct direction means a positive dot product between actual and reference
displacement, and actual displacement of at least 0.01 m.

### Contact and interaction

- Rubber-hand contact-attempt rate.
- First rubber-hand contact time.
- Left/right/either-hand contact fraction.
- Peak left/right rubber-hand net force.
- Invalid-body contact fraction.
- Interaction-proxy success rate.

Frozen contact force threshold:

```text
5 N
```

The contact sensor reports net external force on a body, not an exact
hand-table contact pair. Consequently, `rubber-hand contact` is explicitly a
proxy. The stronger interaction proxy requires:

1. rubber-hand net force above 5 N while table speed exceeds 0.02 m/s;
2. at least 0.01 m actual table displacement; and
3. displacement in the reference direction.

This combines hand loading with table motion, but it is still not a PhysX
pairwise-contact report. Pairwise contact reporting may be added later without
changing the current proxy definition.

### Stability and control

- Fall-proxy rate.
- Minimum root height.
- Peak and mean torque-limit utilization.
- Action-delta RMSE.

The fall proxy is true if either:

```text
root height falls more than 0.30 m below the reference
root up-axis tilts more than 60 degrees from world up
```

This is reported as a proxy because the current termination configuration has
no dedicated fall term.

## 8,000-iteration command template

For each A1/Plan B checkpoint and each evaluation seed:

```bash
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint <checkpoint.pt> \
  --recording.config.enabled \
  --recording.config.output-path=<label>_seed<seed>_physics_eval_v1.npz \
  --training.headless=True \
  --training.num-envs=1 \
  --training.seed=<seed> \
  --training.max-eval-steps=6500 \
  --training.export-onnx=False
```

Analyze the recording:

```bash
python -m holosoma.analyze_wbt_eval_recording \
  <recording.npz> \
  --max-attempts=20 \
  --output=<recording-stem>.json
```

The analyzer defaults are the frozen thresholds in this document. Changing a
threshold creates a different protocol version and must not silently overwrite
v1 results.

## Gate interpretation

The paired 8,000-iteration run is a convergence/comparison gate, not the final
Step 1 result.

The gate passes technically when:

1. all six evaluations complete without CUDA, PhysX, NaN, or channel errors;
2. each evaluation provides 20 closed attempts;
3. every required metric is finite where mathematically defined; and
4. no aggregate metric hides per-seed completion, interaction, or direction
   collapse.

Progression to the full 30,000-iteration budget requires a consistent
improvement over the corresponding 500-iteration pilot in completion and
early-termination rates. A1 and Plan B remain parallel assets; one is not
deleted merely because it is worse on a single aggregate score.

Numeric final-baseline acceptance thresholds and confidence-interval reporting
will be frozen after the 8,000-iteration calibration and before the
30,000-iteration final runs.

## Protocol validation on the 500-iteration pilots

These results validate the recorder and formulas. They are not evidence of
convergence and must not be used to select a winner.

Both policies were evaluated at seed 42 for 3,090 policy steps. The table below
uses the first 20 closed attempts from each recording.

| Metric | A1 | Plan B |
|---|---:|---:|
| Full-motion completion | 10.0% | 5.0% |
| Early termination | 90.0% | 95.0% |
| Mean attempt length | 100.5 steps | 79.1 steps |
| Joint-position RMSE | 0.3080 rad | 0.2724 rad |
| Key-body position RMSE | 0.0659 m | 0.0738 m |
| Key-body orientation mean | 23.30 deg | 23.82 deg |
| Table trajectory position RMSE | 0.0681 m | 0.0730 m |
| Table orientation mean | 12.78 deg | 7.65 deg |
| Correct displacement direction | 100.0% (15 evaluable) | 86.7% (15 evaluable) |
| Rubber-hand contact attempts | 95.0% | 100.0% |
| Interaction-proxy success | 70.0% | 65.0% |
| Invalid-contact step fraction | 44.8% | 54.3% |
| Fall-proxy attempts | 25.0% | 65.0% |
| Peak torque utilization | 100.0% | 100.0% |
| Action-delta RMSE | 0.2160 | 0.2940 |

The high early-termination rates confirm the previous conclusion: 500
iterations are enough to validate the training/evaluation pipeline, but not
enough to establish a strong learned policy.

Validation recordings:

```text
A1:
logs/WholeBodyTracking/20260728_044857-rubberhand_a1_pilot_seed42-eval/
  a1_physics_eval_v1.npz

Plan B:
logs/WholeBodyTracking/20260728_045133-rubberhand_plan_b_pilot_seed42-eval/
  plan_b_physics_eval_v1.npz
```
