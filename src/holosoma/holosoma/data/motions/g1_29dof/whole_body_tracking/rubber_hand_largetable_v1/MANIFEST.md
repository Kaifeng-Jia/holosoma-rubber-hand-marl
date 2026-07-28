# Rubber-hand Large-table WBT Motions v1

## Scope

This directory contains Whole-Body Tracking motion files converted from the
frozen A1 and Plan B retargeting references:

```text
task: sub6_largetable_033
input: 186 frames at 30 Hz
output: 309 frames at 50 Hz
dynamic object: largetable
conversion code commit: 7b54e6d9a213177a2737f991b72a84baefe2f29d
generated: 2026-07-27
```

A1 and Plan B remain separate experiments. Neither converted file combines,
selects between, or modifies the two reference trajectories.

## Files

```text
a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz
plan_b/sub6_largetable_033_plan_b_mj_fps50_w_obj.npz
```

The table asset for both motions is:

```text
../objects_largetable.urdf
```

Do not use the existing `objects_largebox.urdf` default with these motions.

## Source references

```text
SHA-256 a902e79abb69974c4b4af41ee73efa3101b150e4f9a41d06c7c48d0ebfceb511
  data/retargeted/rubber_hand_largetable_v1/sub6_largetable_033/a1/
  sub6_largetable_033_fixed_object.npz

SHA-256 230bd421cbfb5fb613953347d5140598fe16cff00cd38933f1095fa2c3659ca1
  data/retargeted/rubber_hand_largetable_v1/sub6_largetable_033/plan_b/
  sub6_largetable_033_fixed_object_plan_b.npz
```

## Conversion commands

Run from:

```text
src/holosoma_retargeting/holosoma_retargeting
```

A1:

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file ../../../data/retargeted/rubber_hand_largetable_v1/sub6_largetable_033/a1/sub6_largetable_033_fixed_object.npz \
  --output-fps 50 \
  --output-name <output>/sub6_largetable_033_a1_mj_fps50_w_obj.npz \
  --data-format smplh \
  --object-name largetable \
  --has-dynamic-object \
  --once \
  --headless
```

Plan B:

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file ../../../data/retargeted/rubber_hand_largetable_v1/sub6_largetable_033/plan_b/sub6_largetable_033_fixed_object_plan_b.npz \
  --output-fps 50 \
  --output-name <output>/sub6_largetable_033_plan_b_mj_fps50_w_obj.npz \
  --data-format smplh \
  --object-name largetable \
  --has-dynamic-object \
  --once \
  --headless
```

## SHA-256

```text
b669468ade12f8c271119d6c8c120463223ab65489805c486e43b0c3a4ba2207  a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz
6bcc3111a08fc2c714a2fe671411b122b52eb49bd1712c7670b7b61d032d9d14  plan_b/sub6_largetable_033_plan_b_mj_fps50_w_obj.npz
```

## WBT schema

Both files contain:

```text
fps                 (1,)
joint_pos           (309, 36)  # root xyz+wxyz + 29 joints
joint_vel           (309, 35)  # root linear+angular + 29 joints
body_pos_w          (309, 52, 3)
body_quat_w         (309, 52, 4)
body_lin_vel_w      (309, 52, 3)
body_ang_vel_w      (309, 52, 3)
object_pos_w        (309, 3)
object_quat_w       (309, 4)
object_lin_vel_w    (309, 3)
object_ang_vel_w    (309, 3)
joint_names         (29,)
body_names          (52,)
```

## Validation

```text
all numeric values finite: yes
joint order vs MuJoCo G1 model: exact
joint order vs training rubber-hand URDF: exact
required WBT tracking body names present: yes
converted joint/root trajectory vs 50 Hz interpolation: exact
converted object trajectory and velocities vs 50 Hz interpolation: exact
A1 vs Plan B object position/orientation trajectories: byte-identical
training MotionLoader A1 load: passed
training MotionLoader Plan B load: passed
A1 training CLI overrides parse: passed
Plan B training CLI overrides parse: passed
```

Quaternion maximum norm errors:

```text
A1 root: 8.23e-08
A1 body: 6.66e-16
A1 object: 8.73e-08
Plan B root: 7.07e-08
Plan B body: 6.66e-16
Plan B object: 8.73e-08
```

Maximum converted joint velocities:

```text
A1: 6.030754 rad/s, frame 0, left_hip_pitch_joint
Plan B: 11.157419 rad/s, frame 14, left_wrist_roll_joint
```

The Plan B wrist-speed peak is a known baseline limitation inherited from the
retargeted motion. Conversion does not smooth or hide it.

## Training entry points

Use the original object-aware WBT experiment and override only the motion and
table asset. This retains the original reward design.

A1:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object \
  --command.setup-terms.motion-command.params.motion-config.motion-file=holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz \
  --robot.object.object-urdf-path=holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largetable.urdf
```

Plan B:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object \
  --command.setup-terms.motion-command.params.motion-config.motion-file=holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/plan_b/sub6_largetable_033_plan_b_mj_fps50_w_obj.npz \
  --robot.object.object-urdf-path=holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largetable.urdf
```

These are single-motion baseline runs. They do not yet implement the later
mixed-action Step 1 policy.
