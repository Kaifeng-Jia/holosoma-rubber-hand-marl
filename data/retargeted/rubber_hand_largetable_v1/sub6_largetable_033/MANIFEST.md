# Rubber-hand Large-table Retargeting v1

## Scope

This directory freezes the Step 2 reference motions for:

```text
task: sub6_largetable_033
frames: 186
fps: 30
source code commit: d2a30b970f1c95c4e87076ec6b9db03d1a7c71bf
generated: 2026-07-27
```

A1 and Plan B are parallel branches from the same fixed-object baseline:

```text
nominal scaled retarget
        |
fixed physical-object baseline
        |
        +-- A1: PT wrist orientation, wrist-only
        |
        +-- Plan B: designed palm contact, full-body SQP
```

Plan B does not read or inherit the A1 result.

## Files

```text
a1/sub6_largetable_033_nominal_scaled.npz
a1/sub6_largetable_033_fixed_object_base.npz
a1/sub6_largetable_033_fixed_object.npz

plan_b/sub6_largetable_033_nominal_scaled.npz
plan_b/sub6_largetable_033_fixed_object_base.npz
plan_b/sub6_largetable_033_fixed_object_plan_b.npz
```

The final training-reference candidates are:

```text
a1/sub6_largetable_033_fixed_object.npz
plan_b/sub6_largetable_033_fixed_object_plan_b.npz
```

The nominal and baseline files are retained in both branches for independent
provenance. Their duplicate copies are byte-identical.

## Generation commands

Run from:

```text
src/holosoma_retargeting/holosoma_retargeting
```

A1:

```bash
/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  examples/robot_retarget.py \
  --data-path /home/kevin/holosoma/OMOMO_new \
  --save-dir <output>/a1 \
  --task-type object_interaction \
  --task-name sub6_largetable_033 \
  --data-format smplh \
  --task-config.object-name largetable \
  --fixed-object-size-adaptation \
  --retargeter.pt-wrist-orientation.enable \
  --retargeter.foot-sticking-tolerance 0.05
```

Plan B:

```bash
/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  examples/robot_retarget.py \
  --data-path /home/kevin/holosoma/OMOMO_new \
  --save-dir <output>/plan_b \
  --task-type object_interaction \
  --task-name sub6_largetable_033 \
  --data-format smplh \
  --task-config.object-name largetable \
  --fixed-object-size-adaptation \
  --retargeter.plan-b-palm-contact.enable \
  --retargeter.foot-sticking-tolerance 0.05
```

## SHA-256

```text
bc3c934e2aa0d9ca44e03d03973d965ca9e766c5e3d9f2ccfc159782cb260e63  a1/sub6_largetable_033_nominal_scaled.npz
27f9f6c73579aab479aadd4991b08231e4459431c50b625976f8fcc4775cbc29  a1/sub6_largetable_033_fixed_object_base.npz
a902e79abb69974c4b4af41ee73efa3101b150e4f9a41d06c7c48d0ebfceb511  a1/sub6_largetable_033_fixed_object.npz
bc3c934e2aa0d9ca44e03d03973d965ca9e766c5e3d9f2ccfc159782cb260e63  plan_b/sub6_largetable_033_nominal_scaled.npz
27f9f6c73579aab479aadd4991b08231e4459431c50b625976f8fcc4775cbc29  plan_b/sub6_largetable_033_fixed_object_base.npz
230bd421cbfb5fb613953347d5140598fe16cff00cd38933f1095fa2c3659ca1  plan_b/sub6_largetable_033_fixed_object_plan_b.npz
```

## Validation

Shared provenance:

```text
A1 vs Plan B nominal maximum field error: 0
A1 vs Plan B fixed-object baseline maximum field error: 0
qpos shape: (186, 43)
all values finite: yes
```

A1:

```text
changed qpos indices: [26, 27, 28, 33, 34, 35]
maximum change outside six wrist qpos: 0
object pose vs fixed-object baseline maximum error: 0
left/right hand-link orientation error p90/max: 0 / 0 deg
minimum wrist joint-limit margin: 0.515910856 rad
wrist speed p90/p99/max: 0.847484136 / 2.663013877 / 5.112121166 rad/s
left/right minimum palm-table distance: -0.085605658 / -0.085362415 m
```

A1 deliberately permits reference-mesh penetration because its final stage
prioritizes demonstrated wrist orientation. Collision must remain enabled in
simulation and training.

Plan B:

```text
object pose vs fixed-object baseline maximum error: 0
joint-limit maximum violation: 0
minimum robot-object signed distance: -0.040405545 mm
collision validation tolerance: 0.100000000 mm
stable contact frames: 56..147
```

Stable-contact orientation errors:

```text
left palm normal p90/max: 0.049241208 / 0.049520300 deg
right palm normal p90/max: 0.046100686 / 0.047060244 deg
left downward twist p90/max: 0.130405540 / 0.148366666 deg
right downward twist p90/max: 0.153023855 / 0.174826848 deg
```

Stable-contact real palm/table distances:

```text
left p50/p90/max: 0.000000000 / 0.004782989 / 0.441985186 mm
right p50/p90/max: 0.002272938 / 0.003212620 / 0.004561732 mm
```

Known Plan B limitation:

```text
maximum robot joint speed: 11.292800181 rad/s
location: frame 8 -> 9, left_wrist_roll_joint
```

The speed peak is preserved as a documented baseline limitation. It should be
reported separately from tracking quality during training evaluation.
