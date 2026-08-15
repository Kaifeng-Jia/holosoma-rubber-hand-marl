# Multi-Agent Emergence Roadmap

## 1. Status and authority

- Status: active execution guide
- Confirmed: 2026-08-14
- Advisor guidance incorporated: 2026-08-12 meeting
- Platform: Unitree G1 with fixed rubber hands and a physically simulated table
- Active branch: `rubber_hand_marl_baseline`
- Baseline commit: `8038c092` (`wbt-four-action-priors-v1`)
- Primary research direction: multi-agent physical cooperation and competition
- Supersedes as the active guide: `LongTermGoal.md`

This is the single canonical roadmap for subsequent implementation, training,
evaluation, and documentation on this branch. `LongTermGoal.md` remains an
unchanged historical record. Later changes to the research direction must edit
this file instead of creating parallel roadmap documents.

## 2. Research question

The current research question is:

> Can single-agent whole-body-tracking motion priors be transferred through
> multi-agent reinforcement learning so that homogeneous G1 agents, acting
> from local observations under a shared team objective, develop effective
> physical cooperation or competition around a common object?

The first objective is a strong, reproducible baseline. Paper-level innovation
will be selected only after the baseline exposes a concrete multi-agent
limitation.

The initial hypotheses are:

1. WBT initialization improves multi-agent learning speed or final performance
   relative to training from scratch.
2. Joint multi-agent fine-tuning outperforms simply copying an independently
   trained single-agent policy to two robots.
3. In a calibrated cooperative regime, both agents make measurable causal
   contributions to object motion.

## 3. Scope correction from the 2026-08-12 meeting

The active pipeline is:

```text
frozen single-agent WBT priors
    -> MARL-compatible single-agent actor interface
    -> homogeneous two-agent environment
    -> shared-policy multi-agent reinforcement learning
    -> quantitative cooperation analysis
    -> later cooperative or adversarial extensions
```

The following are paused and are not prerequisites for multi-agent training:

- arbitrary target-point single-agent manipulation;
- plus-or-minus 30-degree target-direction generalization;
- teacher-guidance annealing to a reference-free policy;
- four separate goal-conditioned task experts;
- push/kick/pull mixture training, skill conditioning, CVAE, or distillation;
- heterogeneous deployment in which one agent pushes and another kicks;
- general object-structure understanding;
- semantic contact-point planning;
- raw-camera perception or sim-to-real deployment.

These are valid independent research directions, but combining them with the
first MARL baseline would obscure the main question and add unnecessary
engineering dependencies.

## 4. Preserved milestone and invariants

### 4.1 Frozen WBT milestone

The branch starts at `wbt-four-action-priors-v1`, which preserves:

- Push A1;
- Push Plan B;
- Kick;
- Pull;
- the motion manifest and artifact identities;
- validated WBT training and evaluation infrastructure;
- rubber-hand retargeting, visualization, and collision assets.

The four priors remain useful assets, but they no longer all need to be merged
or converted into target-conditioned experts before MARL begins.

### 4.2 First behavior

Push A1 is the frozen first behavior because it is the most reliable retained
prior and supports the simplest symmetric two-agent arrangement.

Plan B, Kick, and Pull are deferred extensions or ablations. They are not
blockers for the first baseline.

### 4.3 Embodiment invariant

All active retargeting, training, evaluation, and visualization must use the
fixed rubber-hand embodiment. Hemisphere, half-sphere, or sphere-hand assets
must not be referenced by active configurations or commands on this branch.

## 5. Frozen first-baseline architecture

### 5.1 Task

The first task uses two homogeneous G1 robots to push one table in the same
demonstrated direction with the same A1 motion type.

- The robots begin on the same suitable pushing side of the table.
- Their initial locations are symmetric or nearly symmetric.
- Both use locally transformed versions of the A1 reference.
- Robot-robot, robot-table, and rubber-hand-table collisions remain enabled.
- The first task does not ask the policy to choose among push, kick, and pull.
- The first task does not require an arbitrary target point or final table yaw.

The environment should make the intended cooperation easy to express. The
research subject is what happens after compatible agents are placed together,
not how to combine deliberately incompatible actions.

The current `objects_largetable.urdf` tabletop is only about 0.522 m wide. It
is retained as the frozen single-agent WBT asset, but it is not assumed to be
wide enough for the first same-side two-agent layout. Before freezing the
ghost-teammate distribution, the project must validate a separate symmetric
wide-table asset. The push-direction depth, tabletop height, A1 contact edge,
and object reference origin must remain unchanged while only the lateral span
is widened. Detailed geometry decisions and validation results are maintained
in `WIDETABLE_GEOMETRY_DESIGN_CN.md`.

### 5.2 Shared actor

- Both agents use one parameter-shared actor.
- The action remains the existing continuous G1 joint-position command.
- Each actor receives only its own local observation at execution time.
- Reference-motion conditioning is retained for the first baseline.
- No agent ID, role label, action-class label, or privileged global truth is
  supplied to the actor.

### 5.3 Teammate observation

The actor interface must be identical in the single-agent compatibility stage
and the later multi-agent stage. The minimal first interface appends four local
values to the existing WBT actor observation:

- teammate relative planar position in the observing robot's heading frame: 2;
- teammate relative planar velocity in the observing robot's heading frame: 2.

Full teammate joint state, contact forces, full pose, and explicit intent are
excluded initially. Additional information requires evidence that the minimal
interface is insufficient.

During the compatibility stage, no physical teammate exists. A ghost teammate
follows the opposite shifted A1 reference. Its relative position and velocity
are derived consistently from that reference and the observing robot's actual
root state; they are not independently sampled or forced near zero. Bounded
layout variation may be added only after the nominal left/right retention gate
passes. During MARL, these four values are replaced by the corresponding real
teammate observations.

### 5.4 Centralized critic

Training uses centralized training with decentralized execution.

The critic may receive:

- both robots' proprioceptive and root states;
- both robots' actions or action history when required by the existing PPO
  implementation;
- the table pose and velocity;
- reference and episode-phase information;
- other simulator truth required for stable value estimation.

Privileged critic information must never silently enter the deployed actor.
Actor and critic observation dimensions must be reported separately.

### 5.5 Shared team reward

All agents receive the same scalar team reward. The first implementation should
reuse the validated WBT reward components and change only what is required to
aggregate the two-agent task:

```text
team reward
    = shared table/reference objective
    + mean of the agents' motion-tracking and stability terms
    - shared fall, invalid-state, and safety penalties
```

The exact formula and weights must be audited against the frozen WBT
configuration before implementation.

The first reward must not:

- force hand-only contact;
- prohibit leg, hip, knee, or torso contact solely for semantic appearance;
- prescribe separate roles;
- reward a named action class;
- introduce arbitrary target-point progress.

Reference similarity, physical task success, and contact technique remain
separate evaluation concepts.

## 6. Execution stages

### Stage 0 -- Isolate and preserve the new mainline

#### Work

- create `rubber_hand_marl_baseline` from `wbt-four-action-priors-v1`;
- use a separate worktree;
- leave the previous object-goal worktree and its uncommitted diagnostics
  untouched;
- establish this file as the canonical roadmap.

#### Exit gate

- the new worktree is clean;
- its parent is exactly `8038c092` before the roadmap commit;
- no object-goal bridge or teacher-annealing implementation is present;
- the previous worktrees remain unchanged.

### Stage 1A -- Build the lossless MARL-compatible A1 interface

Status: **passed on 2026-08-15**.

#### Work

1. Audit the exact A1 actor observation and checkpoint tensor shapes.
2. Preserve the original 154-dimensional `actor_obs` group and append a
   separate four-dimensional `teammate_obs` group.
3. Transfer the original input-layer weights exactly.
4. Expand the empirical actor normalizer without changing the original 154
   statistics.
5. Initialize the four new input columns to zero so the converted policy
   reproduces the old actor independently of teammate input.
6. Keep the 298-dimensional WBT critic and the original WBT reward unchanged
   during this compatibility stage.

#### Exit gate

- the expanded actor loads without silently dropping or reordering inputs;
- deterministic converted-policy output matches the frozen A1 policy within a
  documented numerical tolerance;
- the conversion records the source checkpoint hash and all added dimensions;
- no training is required to claim lossless interface conversion.

#### Recorded result

- source checkpoint:
  `logs/WholeBodyTracking/20260728_051638-rubberhand_a1_8000_seed42-locomotion/model_07999.pt`
  in the frozen `holosoma-rubber-hand-largetable` worktree;
- source SHA-256:
  `54fbfa4b38ee69ac1be33da8b5d3f6849e587b6bd02da3f407b0b27844be3c25`;
- converted checkpoint:
  `logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt`;
- converted SHA-256:
  `11ef1fe7a4b340a47218f040e7675af4073067c5ce931e169818f903150e3224`;
- reproducible manifest:
  `logs/WholeBodyTracking/marl_compat_a1_v1/conversion_manifest.json`;
- actor first layer: `512 x 154 -> 512 x 158`;
- actor normalizer: `154 -> 158`, with all original statistics and the scalar
  count `786432000` preserved exactly;
- critic first layer: unchanged at `512 x 298`;
- new Actor columns: exactly zero;
- optimizer state: removed; the embedded configuration sets
  `load_optimizer=False`;
- maximum deterministic output error over 257 random old observations and
  random teammate inputs: `0.0` with an acceptance tolerance of `1e-6`;
- runtime gate: the standard `eval_agent` path loaded the embedded config,
  explicit ghost manager, 158-D Actor, 298-D Critic, and rubber-hand A1 assets,
  then completed two Isaac Sim evaluation steps with one environment.

The existing empirical normalizer uses one scalar count shared by all input
dimensions. Resetting it would damage the frozen 154-D statistics, while
preserving it means the four new dimensions will not acquire useful empirical
statistics during short adaptation. Therefore `teammate_obs` uses a fixed
physical scale before the normalizer: planar relative position in metres and
planar relative velocity in metres per second, both clipped to `[-1, 1]`; its
new normalizer entries are identity statistics (`mean=0`, `var=std=1`). This is
an explicit baseline contract rather than an accidental normalizer behavior.

### Geometry preflight -- Freeze the wide-table partner layout

This is a bounded dependency between Stage 1A and Stage 1B, not a new research
stage. It exists because ghost-teammate values must be based on the real future
layout rather than arbitrary numbers.

#### Work

1. Confirm the A1 table local axes and runtime world transform.
2. Create a separate primitive-based wide-table candidate without modifying the
   frozen single-agent table asset.
3. Preserve table depth, height, push-side contact edge, and object origin.
4. Begin with a 1.4 m lateral-width candidate; use 1.2 m and 1.6 m only as
   bounded alternatives if the first layout fails.
5. Place two A1 robots on the same pushing side using symmetric lateral
   reference offsets.
6. Validate visual geometry in Viser and actual collisions and stability in
   Isaac Sim.
7. Freeze the baseline-v1 table width and agent center spacing, and measure the
   nominal teammate-state envelope used to design the Stage 1B variation.

Mass and friction are explicitly not selected by this geometry preflight. A
temporary inertial value may be used for static validation, but it is not a
training asset or reported physical result. Final mass, COM, inertia, and
friction are frozen only after the Stage 4 capacity calibration.

#### Exit gate

- both robots fit on the same pushing side without initial interpenetration;
- both A1 hand-contact regions lie on the unchanged push-side surface;
- feet and lower bodies are not blocked by table legs at reset;
- visual and collision primitives agree;
- the table origin and push-direction contact geometry remain compatible with
  the frozen A1 object trajectory;
- a concrete symmetric partner offset and measured nominal teammate envelope
  are recorded before Stage 1B; the reset-variation range is frozen in Stage
  1B rather than guessed during geometry preflight.

### Stage 1B -- Validate A1 retention with a ghost teammate

Status: **nominal left/right gate failed on 2026-08-15; do not train or enter
Stage 2 yet**.

#### Work

1. Evaluate each observer side separately on the 1.4 m table using the frozen
   0.8 m partner spacing.
2. Derive the ghost position and velocity from the opposite shifted A1
   reference and express them in the observing robot's yaw frame.
3. Evaluate the converted policy before any further training.
4. Freeze bounded reset variation only after both nominal sides pass.
5. Run a 50-iteration smoke test only after the no-training equivalence gate.
6. Run a 500-iteration retention test only if the smoke gate passes and further
   WBT exposure is justified.

#### Required comparisons

- frozen A1 checkpoint with the original interface;
- expanded A1 checkpoint with the nominal ghost state;
- expanded A1 checkpoint across the frozen ghost-state randomization range.

#### Exit gate

- A1 completion retains at least 90% of the frozen checkpoint's measured
  completion performance;
- fall and early-termination rates do not regress materially;
- ghost-state variation does not cause policy collapse;
- the same actor input schema can consume a real teammate state.

#### Recorded nominal result

The detailed protocol and per-channel evidence are recorded in
`STAGE1B_A1_RETENTION_REPORT_CN.md`. The isolated retention table keeps the
1.4 m geometry but preserves the frozen A1 table's 0.1 kg mass and contact
parameters; its COM and inertia are recomputed from the five-box geometry.

Twenty closed attempts at seed 42 produced:

- original 0.522 m A1 table, zero ghost: `16/20 = 80%` completion;
- 1.4 m table, left observer, trajectory ghost: `1/20 = 5%` completion;
- 1.4 m table, right observer, trajectory ghost: `17/20 = 85%` completion.

For both left and right sides, a paired 650-step zero-ghost versus
trajectory-ghost run had exactly `0.0` maximum difference in actions, joint
states, and object positions. The frozen actor therefore ignores the new
inputs exactly as designed; ghost values are not the cause of the left-side
failure. The left side retained only 6.25% of the original completion rate,
far below the required 90%. No 50- or 500-iteration adaptation is authorized.

Exact post-physics termination attribution and the geometry audit are now
complete. Left failures are dominated by the object-position/orientation
gates rather than robot collapse. Centering at `+/-0.4 m`, reducing spacing to
0.7 m, reference-only mirroring, dynamic policy-I/O mirroring, and a fixed
mirror-plane candidate all failed to make the frozen one-sided actor reliable.
An offline sensitivity audit reproduced checkpoint actions within `7.6e-6`
and isolated the initial left/right action difference almost entirely to the
three-dimensional base-angular-velocity observation, but making that initial
observation match did not improve the closed-loop rollout. The remaining gap
is therefore distributional: the one-sided A1 actor was not trained to retain
the skill on the opposite side.

This result authorizes a bounded left-side adaptation ladder, not Stage 2:

1. freeze a table-centered `+/-0.4 m` layout contract;
2. initialize from `model_07999_actor158.pt` and run a 50-iteration left-side
   WBT adaptation smoke without changing reward, termination, table physics,
   or teammate-column weights;
3. proceed to 500 iterations only if the smoke improves the exact metrics;
4. evaluate the resulting frozen checkpoint on both sides for 20 attempts;
5. require at least `15/20` on each side before Stage 1B-2/Stage 2.

If 500 iterations cannot pass both sides, the next fallback is training-time
symmetry augmentation or mixed left/right references, not another deployment
geometry patch.

The raw right-side teammate position reaches 1.0902 m and velocity reaches
1.0600 m/s in the formal rollout. The current `[-1, 1]` observation clip is
therefore slightly saturated on that side. Observation rescaling remains an
open decision and must be resolved before teammate-aware training; it does not
affect the frozen actor because all four new input columns are zero.

### Stage 2 -- Build the two-agent environment

#### Work

1. Instantiate two rubber-hand G1 agents and one table.
2. Define mirrored or near-mirrored initial transforms.
3. Transform the A1 reference consistently for each robot.
4. Verify collision groups, resets, terminations, and recording.
5. Verify that each actor sees self-local data plus real teammate position and
   velocity.
6. Verify that the critic sees the intended global state.
7. Implement one identical shared reward value for both agents.

#### Exit gate

- deterministic reset produces the intended geometry;
- neither robot nor table begins in collision interpenetration;
- both actors receive the same schema in their respective local frames;
- reward changes have the correct sign under controlled object motion;
- a short rollout records both robots and the table correctly;
- no hemisphere-hand asset is active.

### Stage 3 -- Establish the MARL learning baseline

#### Required experimental groups

1. **Single A1:** one robot executing the frozen prior.
2. **Copied A1:** two copies of the compatible A1 actor without joint
   fine-tuning.
3. **MARL from scratch:** the same shared-actor architecture without WBT
   initialization.
4. **WBT-initialized MARL:** the main group, initialized from the compatible A1
   checkpoint and jointly fine-tuned.

The primary optimizer is a MAPPO-style centralized-critic PPO baseline unless
the code audit shows that the repository already provides a more appropriate
equivalent implementation.

#### Training ladder

```text
50 iterations     environment and channel smoke gate
500 iterations    reward direction and learning sanity gate
2,000 iterations  early cooperation and stability gate
8,000 iterations  first complete baseline
```

Training to 30,000 iterations is authorized only if the 8,000-iteration curves
are still improving and the learned behavior matches the intended objective.

#### Exit gate

- the main group trains without numerical or reset failures;
- metrics are evaluated at frozen checkpoints rather than selected only by
  visual preference;
- WBT initialization is compared fairly with training from scratch;
- joint fine-tuning is compared with copied actors without fine-tuning.

### Stage 4 -- Calibrate and verify genuine cooperation

The final table mass and friction must be physically plausible and fixed for
the reported experiment. They should not be chosen only by intuition.

#### Calibration

1. Use an easy table for environment smoke testing.
2. Run a bounded mass/friction capacity sweep.
3. Select a cooperative regime in which one robot usually fails or makes little
   progress, while two jointly trained robots can succeed.
4. Freeze the selected asset and physics parameters before the final runs.

#### Core metrics

- table displacement and direction;
- full-episode task success;
- reference and robot-motion tracking error;
- completion time;
- robot fall and early-termination rates;
- object motion produced per agent;
- performance under teammate removal;
- performance when one teammate is replaced by a standing/no-op policy;
- performance under bounded initial-position or teammate perturbations.

The main causal cooperation measure is the loss of team performance when one
agent is removed or neutralized. Visualization alone is not evidence of
cooperation.

#### Exit gate

- two-agent joint training materially outperforms the single-agent condition in
  the frozen cooperative regime;
- joint training improves on copied A1 actors without fine-tuning;
- removing or neutralizing either agent causes a measurable performance loss;
- the claimed behavior is reproducible across frozen seeds.

### Stage 5 -- Controlled extensions and innovation selection

Only after the A1+A1 baseline passes Stage 4 should the project expand to:

1. Pull + Pull;
2. Kick + Kick;
3. heavier, larger, or disturbed objects;
4. asymmetric initial positions;
5. temporary teammate failure and compensation;
6. adversarial pulling or table-control tasks;
7. improved credit assignment or opponent/teammate modeling.

Distinct semantic roles are not required in the first homogeneous baseline.
Later work may analyze whether asymmetric force, contact, or recovery roles
emerge from local state without role labels.

CVAE, multi-skill merging, heterogeneous checkpoints, arbitrary target points,
and object-structure learning remain separate future directions. They should be
reintroduced only when they directly address a measured limitation of the
multi-agent baseline.

## 7. Frozen first-run decisions

- Research focus: multi-agent physical cooperation.
- First motion: Push A1.
- Number of agents: two.
- Embodiment: fixed rubber hands only.
- Initialization: the same MARL-compatible A1 prior for both agents.
- Actor: homogeneous, parameter-shared, and locally observed.
- Added teammate input: local planar relative position and velocity.
- Critic: centralized and globally observed during training.
- Reward: one shared team reward built from audited WBT components.
- Direction: the fixed demonstrated push direction, not an arbitrary target.
- First environment: symmetric or near-symmetric same-side cooperative layout
  using a separately validated wide table.
- Wide-table geometry: freeze 1.4 m width and 0.8 m symmetric agent spacing for
  baseline v1; this is an accepted first-run geometry, not a claim of global
  optimality or a mathematically minimal width. Depth, height, push-side
  contact geometry, and reference origin are preserved.
- A1 prior budget: reuse the frozen 8,000-iteration checkpoint; do not run a
  new 8,000- or 30,000-iteration A1 job before MARL. Only a bounded 50/500
  retention adaptation is allowed if the converted policy fails its measured
  retention gate.
- Ghost distribution: frozen only after the real two-agent geometry preflight.
- Physics: easy smoke asset followed by a bounded capacity sweep and a frozen
  final mass/friction setting.
- First full budget: 8,000 iterations after the earlier gates pass.

Changes to any frozen decision must be recorded in this document before the
corresponding long-running experiment begins.

## 8. Immediate next steps

Completed at the current checkpoint:

- the A1 actor/checkpoint/reward audit;
- the A1 object-axis and loader audit;
- the separate primitive 1.4 m candidate URDF and structural tests;
- a two-copy A1 Viser preview with an adjustable 0.8 m initial spacing;
- human visual acceptance of the 1.4 m table and 0.8 m spacing;
- a one-environment Isaac Sim import/reset/24-step smoke test on CUDA.
- an Isaac Sim dual-A1 frame-0 reset collision gate with explicit far-away
  controls: 0 N excess contact force, no table drift, and rubber-hand bodies
  present in both articulations and contact sensors.
- the lossless Stage 1A checkpoint conversion, strict model/normalizer loads,
  zero deterministic output error, and a two-step Isaac Sim runtime smoke.

The next gates are:

1. Attribute left-side early terminations exactly at the post-physics failure
   state and audit the off-center contact/yaw-moment mechanism.
2. Compare the accepted midpoint-preserving layout with a table-centerline
   symmetric placement and, only if justified, a mirrored-reference variant.
3. Re-run the nominal left/right no-training gate after one geometry/reference
   correction is explicitly approved.
4. Decide the teammate position/velocity observation scale from the measured
   envelope before any teammate-aware update.
5. Do not freeze reset randomization, run a 50/500-iteration adaptation, or
   begin the full two-agent environment until both nominal sides pass Stage 1B.

The Isaac Gym runtime asset check remains pending because the local machine has
no `hsgym` environment. It does not block the current Isaac Sim baseline, but
must not be reported as validated.
