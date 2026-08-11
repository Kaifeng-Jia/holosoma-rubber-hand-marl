# Step 1 Push Pilot Artifact Index

## Status

- Frozen: 2026-08-05
- Branch: `rubber_hand_largetable_baseline`
- Artifact/code commit: `e4b27a1400749a4d0ec8c224c53ff4238132ffd2`
- Algorithm: PPO whole-body tracking
- Training seed: 42
- Evaluation protocol: `holosoma.wbt_physics_eval.v1`
- Canonical evaluation seeds: 42, 43, and 44

This index closes the A1/Plan B single-motion push pilot. It distinguishes the
canonical exact-main controls from later reproducibility reruns and prevents a
configuration-only difference from being reported as a behavioral method.

## Canonical training checkpoints

| Prior | Checkpoint | SHA-256 |
|---|---|---|
| A1 | `logs/WholeBodyTracking/20260728_051638-rubberhand_a1_8000_seed42-locomotion/model_07999.pt` | `54fbfa4b38ee69ac1be33da8b5d3f6849e587b6bd02da3f407b0b27844be3c25` |
| Plan B | `logs/WholeBodyTracking/20260728_132654-rubberhand_plan_b_8000_seed42-locomotion/model_07999.pt` | `5ba3d28bfb862321812e1f9475f69371b40ee657d639ef0a3488fbccc4a5e89e` |

## Canonical three-seed evaluation

| Prior | Seed | Evaluation directory |
|---|---:|---|
| A1 | 42 | `logs/WholeBodyTracking/20260731_224124-rubberhand_a1_8000_seed42-eval/` |
| A1 | 43 | `logs/WholeBodyTracking/20260731_224602-rubberhand_a1_8000_seed42-eval/` |
| A1 | 44 | `logs/WholeBodyTracking/20260731_225034-rubberhand_a1_8000_seed42-eval/` |
| Plan B | 42 | `logs/WholeBodyTracking/20260731_225505-rubberhand_plan_b_8000_seed42-eval/` |
| Plan B | 43 | `logs/WholeBodyTracking/20260731_225937-rubberhand_plan_b_8000_seed42-eval/` |
| Plan B | 44 | `logs/WholeBodyTracking/20260731_230410-rubberhand_plan_b_8000_seed42-eval/` |

Each seed records 6,500 policy steps and contributes the first 20 closed
attempts to the frozen report in `STEP1_8000_EVAL_RESULTS.md`.

## Reproducibility reruns

The allowed-contact regex was later changed to explicitly exclude
`left_rubber_hand_link` and `right_rubber_hand_link`. This configuration change
does not affect `UndesiredContacts`, because those fixed links are absent from
the simulator articulation-body list used by that reward term.

| Prior | Rerun checkpoint | File SHA-256 |
|---|---|---|
| A1 | `logs/WholeBodyTracking/20260801_184233-rubberhand_a1_contact_semantic_8000_seed42-locomotion/model_07999.pt` | `70ccba9454fe33497a11c3abf2a1050058b4a9c99fc92bd51efe11ffb4d85559` |
| Plan B | `logs/WholeBodyTracking/20260802_043910-rubberhand_plan_b_contact_semantic_8000_seed42-locomotion/model_07999.pt` | `8a73661f0a958e467ea12bec2c9bf5bd2211d8af22079df6ac83cce64ec58a71` |

Checkpoint files differ because their saved experiment configurations contain
different run names, save intervals, and contact regex strings. A recursive
comparison excluding `experiment_config` found all 76 tensors identical:

| Prior | Composite tensor SHA-256 | Canonical equals rerun |
|---|---|---|
| A1 | `297277bdfbfb144100d5df76231180dbe9f61ec4f857e1754c770985a752bef3` | Yes |
| Plan B | `f9f4c1c2a72b52fee1afeeef8cf21d959e669c5f07ca37eb938ccd8d4a77ebee` | Yes |

The seed-43 recordings are also byte-identical:

| Prior | Canonical recording | Rerun recording | Shared SHA-256 |
|---|---|---|---|
| A1 | `logs/WholeBodyTracking/20260731_224602-rubberhand_a1_8000_seed42-eval/a1_seed43_physics_eval_v1.npz` | `logs/WholeBodyTracking/20260802_183713-rubberhand_a1_contact_semantic_8000_seed42-eval/a1_contact_semantic_8000_seed43_physics_eval_v1.npz` | `cd47315b3bed5e17cf5d977438bf01d4898036e5ef7221ef13b162b2e40dbfb0` |
| Plan B | `logs/WholeBodyTracking/20260731_225937-rubberhand_plan_b_8000_seed42-eval/plan_b_seed43_physics_eval_v1.npz` | `logs/WholeBodyTracking/20260802_184204-rubberhand_plan_b_contact_semantic_8000_seed42-eval/plan_b_contact_semantic_8000_seed43_physics_eval_v1.npz` | `e86cc774482991f74afdc9c38fbb61d49d8ec5e917e42334b76b8f797d52d604` |

The reruns are therefore reproducibility confirmations, not a separate
semantic-recovery policy family.

## Frozen motion and asset hashes

| Artifact | Path | SHA-256 |
|---|---|---|
| A1 50 Hz motion | `src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz` | `b669468ade12f8c271119d6c8c120463223ab65489805c486e43b0c3a4ba2207` |
| Plan B 50 Hz motion | `src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/plan_b/sub6_largetable_033_plan_b_mj_fps50_w_obj.npz` | `6bcc3111a08fc2c714a2fe671411b122b52eb49bd1712c7670b7b61d032d9d14` |
| Rubber-hand robot URDF | `src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf` | `7ed217f28ed6e3b1fa864bf0aadd527ed5ad319361ecefaeccd56e81306a4a25` |
| Training table URDF | `src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largetable.urdf` | `7e69319366888310890e1accac8fd137bb9cbd85ad04a922e20e97cade3109e2` |

## Frozen interpretation

- A1 is the accepted push-motion prior and the more stable all-round tracking
  checkpoint.
- Plan B is a valid palm-edge retarget reference, but its policy does not
  reproduce palm-edge contact reliably.
- Plan B's top-contact and leg/hip behavior is retained as a potentially useful
  behavioral prior for multi-agent emergence experiments.
- Full-motion completion, object trajectory tracking, and body-level hand-force
  coincidence do not prove hand-caused table motion.
- The table base mass is 0.1 kg, while inherited startup randomization adds
  1--4 kg, producing an effective training range of about 1.1--4.1 kg.
- No additional A1 or Plan B single-motion training is scheduled.

## Next active work

The push pilot is complete. The next behavior-prior work is:

1. create the versioned motion manifest;
2. inventory and select a kick demonstration;
3. retarget and validate the kick prior;
4. run the staged kick expert training ladder;
5. perform a bounded pull feasibility audit;
6. train the shared mixed-motion prior after the retained library is frozen.
