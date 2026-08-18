# Multi-Agent Emergence Roadmap

## 1. Status and authority

- Status: active execution guide
- Confirmed: 2026-08-14
- Scope realigned: 2026-08-17
- Advisor guidance incorporated: 2026-08-12 meeting
- Platform: Unitree G1 with fixed rubber hands and a physically simulated table
- Active branch: `rubber_hand_marl_baseline`
- Baseline commit: `8038c092` (`wbt-four-action-priors-v1`)
- Primary research direction: multi-agent physical cooperation and competition
- Active work item: reference-state collision replay and rubber-hand wrench-feasibility check
- Supersedes as the active guide: `LongTermGoal.md`

This is the single canonical roadmap for subsequent implementation, training,
evaluation, and documentation on this branch. `LongTermGoal.md` remains an
unchanged historical record. Later changes to the research direction must edit
this file instead of creating parallel roadmap documents.

### 1.1 Execution and consultation contract

The overall direction in this roadmap governs local implementation choices.
Diagnostics may refine how an approved step is executed, but they must not
silently redefine the research question, add a new training objective, skip a
decision gate, or promote a diagnostic artifact into a baseline.

Before execution continues, discuss with the user any uncertainty that would
change one or more of the following:

- the active plan or the order of plans;
- the robot/reference/object trajectory contract;
- table geometry, mass, inertia, friction, or contact semantics;
- actor or critic observations and parameter-sharing structure;
- reward, termination, reset, or success definitions;
- checkpoint initialization or promotion;
- whether a step is single-agent, a frozen two-entity diagnostic, or MARL;
- the approved iteration budget or the transition to a long-running job;
- the inclusion of task-oriented, target-conditioned, or other out-of-scope
  research elements.

Read-only diagnosis, tests, and short explicitly approved smoke checks may
proceed within the active contract. Any material change must first be explained
by file/module purpose, proposed modification, expected evidence, and rollback
condition. If evidence contradicts the roadmap, stop at the next decision gate,
record the result here, and consult the user instead of improvising a new path.

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

### Current-stage option register -- five plans

These five plans preserve the original escalation order discussed before the
Stage 1B experiments. They are alternative experiments, not five mandatory
stages. The former proposal to train separate left and right checkpoints was
explicitly rejected and is not an active plan.

| Plan | Method | Status and decision |
|---|---|---|
| **1. Bounded left-side adaptation** | Initialize from `model_07999_actor158.pt`, preserve the shifted reference/object/WBT contract, run a 50-iteration left smoke, and continue to 500 only if the same checkpoint improves both left and right gates. | **Failed.** The 50-iteration run caused severe first-update drift and completed only `2/20` on each side. The 500-iteration continuation was not authorized. |
| **2. Balanced left/right reference training** | Train one shared checkpoint with both shifted references and report per-side metrics rather than aggregate reward. | **Failed under the tested recipes.** The conservative full-actor variant preserved the right side but completed `0/20` left; the backbone-frozen teammate-input adapter also completed no left trajectory. Do not add iterations to either recipe. |
| **3. Training-time symmetry augmentation** | Define strict left/right transforms for observation, reference, action, critic state, joints, bodies, and contact channels, then train one shared actor with symmetry-consistency supervision. | **Pending, blocked by the post-Plan-2 feasibility gate.** Symmetry training is justified only if the single-agent side-end reference is dynamically feasible. |
| **4. Full symmetric A1 retraining** | Re-run the complete A1 WBT configuration with validated left/right references and symmetry support, potentially up to the full 8,000-iteration budget. | **Pending high-cost fallback.** Use only if Plan 3 is valid but insufficient; do not start directly. This was the original Plan 5 before removal of the rejected separate-checkpoint proposal. |
| **5. Paired-reference multi-agent WBT** | Materialize the accepted Viser layout as two synchronized robot references, one shared table reference, and one shared phase; initialize a shared actor from A1 and train a centralized critic for the dual-robot pushing demonstration. | **New parallel candidate, not yet selected.** It bypasses the remaining single-agent side-adaptation stage but preserves the A1 prior. |

#### Rejected proposal

Training independent `A1-left` and `A1-right` checkpoints was the former Plan
4. It remains rejected because it breaks the simplest homogeneous shared-policy
baseline, introduces role specialization before it is required, and creates an
unresolved checkpoint selection/merging problem for the multi-agent stage.

#### Post-Plan-2 feasibility gate

The failure of Plan 2 does not authorize Plan 3, Plan 4, or Plan 5
automatically. First determine whether a single off-centre robot can reproduce
the required table trajectory under a frozen physical contract. This is a
diagnostic decision gate, not a sixth training plan.

```text
Plans 1 and 2 failed
        |
        v
single-agent side-reference dynamics feasibility audit
        |
        +-- feasible within accepted margins
        |       -> generate/validate corrected side references
        |       -> discuss Plan 3
        |       -> Plan 4 only if Plan 3 is valid but insufficient
        |
        +-- infeasible or only physically brittle
                -> discuss a frozen two-entity mechanics preflight
                -> measure opposite-side yaw-moment cancellation
                -> only then decide whether to select Plan 5
```

The feasibility audit first freezes representative mass, inertia, friction,
contact semantics, torque limits, acceptable table-position/yaw errors, and
allowed body contacts. It then compares the force and yaw moment required by
the object reference with the contact wrench that one rubber-hand G1 can
produce from the side while remaining balanced. It must distinguish a clearly
feasible solution, a near-limit/brittle solution, and no acceptable solution.
It must not claim mathematical impossibility from one failed PPO run.

#### Confirmed feasibility-audit contract -- 2026-08-17

- Evaluate the historical effective table-mass envelope of approximately
  `1.1--4.1 kg`; do not treat the `0.1 kg` URDF base mass as the complete
  simulated mass or silently replace the envelope with one favorable mass.
- Permit a small, bounded amount of natural table yaw rather than requiring an
  off-centre single robot to reproduce the central A1 table orientation
  exactly. Recover the existing reference/termination yaw scales first and
  review the numerical tolerance before using it to classify feasibility.
- Permit incidental contact by legs or other robot bodies. Such contact must be
  reported separately, and a rollout does not count as rubber-hand pushing if
  non-hand contact supplies the dominant propulsive impulse.
- Keep table geometry, friction/contact settings, actuator limits, reference
  trajectories, reward, and termination logic unchanged during the initial
  read-only audit. Any later change requires a separate review.

#### Read-only contact audit result -- 2026-08-17

The opt-in Isaac Sim audit records the actual runtime object mass, inertia, COM,
material properties, angular velocity, filtered table-to-all-robot contacts,
and filtered table-to-rubber-hand contacts. The implementation is isolated in
commits `042b85a7` and `8c96de33`; the default sensor graph remains unchanged
when diagnostics are disabled. Seventeen related tests and a 24-step CUDA smoke
passed before the fixed-physics evaluations.

The bounded audit used seed 42, deterministic PPO inference, static and dynamic
table friction `0.5`, restitution `0`, 650 evaluation steps per case, and total
table masses `1.1`, `2.6`, and `4.1 kg`. The Plan-2 candidate was
`model_08002.pt`; the exact frozen comparison was
`marl_compat_a1_v1/model_07999_actor158.pt`. Raw NPZ recordings and generated
JSON reports are stored under
`logs/WholeBodyTracking/stage1b_a1_retention_v1/contact_audit_*`.

| Policy | Mass | Side | Completed attempts | Endpoint-yaw pass | Median rubber-hand propulsion fraction | Dominant propulsive bodies |
|---|---:|---|---:|---:|---:|---|
| Plan 2 | 1.1 kg | left | 1/3 | 1/3 | 0.00045% | hips/knees |
| Plan 2 | 1.1 kg | right | 2/2 | 2/2 | 0.01094% | hips/knees |
| Plan 2 | 2.6 kg | left | 0/3 | 0/3 | 0.44680% | hips/knees |
| Plan 2 | 2.6 kg | right | 2/2 | 0/2 | 0.00668% | hips/knees |
| Plan 2 | 4.1 kg | left | 2/2 | 1/2 | 0.01493% | hips |
| Plan 2 | 4.1 kg | right | 1/2 | 0/2 | 0.00096% | hips/knees |
| Frozen A1 | 2.6 kg | left | 2/2 | 0/2 | 1.36875% | hips, then hands/knees |
| Frozen A1 | 2.6 kg | right | 0/7 | 0/7 | 0.00000% | hips/knees/ankle |

Completion and endpoint-yaw gates are reported separately from contact
semantics. A completed rollout does not pass this audit when non-hand bodies
supply the dominant positive propulsive impulse. None of the eight cases passed
the rubber-hand-dominance requirement. Plan 2 changed the left/right completion
distribution but did not create a rubber-hand-driven side push; it remains a
diagnostic artifact and is not promoted.

The dominant hip/knee contact points lie on the tabletop push-side vertical
edge at table-local `z` approximately `-0.262 m`, matching half of the frozen
`0.5219528 m` table depth. These contacts can begin as early as reference frame
10. At sampled first-contact frames, the corresponding reference hip/knee body
origins are farther from the edge than the physical rollout bodies. The current
evidence therefore indicates rollout drift into an easier body-contact solution,
but does not prove that the shifted reference itself is collision-free or that
a single-robot side reference is dynamically impossible.

The next bounded step is consequently a no-training reference-state collision
replay followed, if collision-free, by a rubber-hand contact-wrench feasibility
check. It must determine whether the exact reference states introduce table
interpenetration, whether reachable palm contacts can supply the required
translation and bounded yaw wrench, and what balance/actuator margin remains.
Only after that evidence is reviewed may the project discuss Plan 3 or a frozen
two-entity mechanics preflight. This result does not authorize Plan 4, Plan 5,
new PPO iterations, hand-only rewards, or hidden support forces.

A frozen two-entity mechanics preflight is not MARL and is not the rejected
separate-checkpoint plan. It uses two physical robots with documented frozen
policies, synchronized phase, no artificial teammate force, and no joint
learning solely to test whether their physical yaw moments cancel. If that
preflight supports Plan 5, selecting and implementing Plan 5 still requires a
separate user decision.

Plan 5 is reference-guided multi-agent reinforcement learning. Its current
scope is deliberately narrow: reproduce the synchronized dual-robot pushing
reference, maintain both robots' WBT behavior, track the one shared table
trajectory, and test whether the two physical contact moments cancel. It does
**not** include a target point, task reward, reference annealing, object
navigation, autonomous skill selection, or task-oriented MARL. Those topics
belong to a separate research direction and are not active here.

### Stage 1B -- Validate A1 retention with a ghost teammate

Status: **Plans 1 and 2 failed the bilateral single-agent gate on 2026-08-15.
The 2026-08-17 read-only contact audit additionally showed that both the Plan-2
candidate and frozen A1 comparison are dominated by hip/knee table propulsion
in the shifted side layouts. Do not continue either Plan-2 recipe. The active
next gate is the no-training reference-state collision/wrench-feasibility check.
No later plan is selected, and selecting Plan 5 would not retroactively mark
Stage 1B as passed.**

#### Historical protocol (closed)

The work items, comparisons, and exit gate below record the contract used for
the completed Stage 1B experiments. They are retained for reproducibility and
do not authorize another adaptation run.

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
far below the required 90%. Adaptation was withheld until the exact termination
and geometry audits below were complete.

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

The completed audit authorized one bounded left-side adaptation smoke, not
Stage 2:

1. freeze a table-centered `+/-0.4 m` layout contract;
2. initialize from `model_07999_actor158.pt` and run a 50-iteration left-side
   WBT adaptation smoke without changing reward, termination, table physics,
   or teammate-column weights;
3. proceed to 500 iterations only if the smoke improves the exact metrics;
4. evaluate the resulting frozen checkpoint on both sides for 20 attempts;
5. require at least `15/20` on each side before Stage 1B-2/Stage 2.

The 50-iteration smoke was run from `model_07999_actor158.pt` with teammate
inputs scaled to zero. It consumed 4,915,200 transitions. Although online
reward and episode length improved, the first PPO update reached KL `10.321`.
The frozen `model_08048.pt` then completed only `2/20` centered-left and `2/20`
centered-right attempts, with 85% and 90% fall-proxy rates. On the same
centered 650-step layouts, reset counts changed from frozen `left=3, right=0`
to adapted `left=4, right=7`. The smoke therefore failed and exposed severe
right-side forgetting; `model_08048.pt` is a diagnostic artifact, not a new
baseline.

This run was not continued to 500 iterations. Its failure motivated the
balanced left/right Plan 2 experiments recorded below; those experiments also
failed the bilateral gate and are now closed. Do not revive this one-sided
recipe or add another deployment-geometry patch.

The balanced conservative adaptation audit is now complete. It restarted from
`model_07999_actor158.pt`, sampled the centered left/right references at
`0.5/0.5`, kept all teammate inputs scaled to zero, fixed actor and critic
learning rates at `1e-5`, used one learning epoch, and reduced PPO clipping to
`0.05`. KL remained between approximately `0.0001` and `0.0003`, so the severe
first-update drift and right-side forgetting from the one-sided smoke were
eliminated. A 0--5 update scan selected the fourth update, `model_08002.pt`, as
the best short-gate candidate. Its SHA-256 is
`cce6aad3a3906ed5d28e3610592ee284109f739f2fb141b42679c0d3e6c8cd69`.

The 6500-step formal gate did not validate that candidate for bilateral use.
Centered right passed with `18/20` completed attempts, but centered left
completed `0/20`. On the left, all 20 attempts made rubber-hand contact, 17/20
passed the interaction proxy, and all 18 direction-evaluable attempts moved
the table in the correct direction, with 0.826 m mean displacement. The
failure is therefore not absence of pushing behavior. It is dominated by the
strict object position/orientation tracking gates (7 and 11 attributions,
respectively), which prevent the physical left-side rollout from reaching the
reference endpoint.

Balanced conservative adaptation is recorded as a partial technical success,
not a Stage 1B pass. `model_08002.pt` remains a diagnostic candidate and
`model_07999_actor158.pt` remains the frozen baseline. Do not increase PPO
iterations on this recipe: the 0--5 scan was non-monotonic and the best short
candidate still failed the formal left gate. Before Stage 2, re-examine the
compatibility among the centered-left object reference, physical object
dynamics, and the Stage 1B completion/termination contract. Do not hide this
failure by adding a body-part contact restriction or another geometry patch.

The raw right-side teammate position reaches 1.0902 m and velocity reaches
1.0600 m/s in the formal rollout. The current `[-1, 1]` observation clip is
therefore slightly saturated on that side. Observation rescaling remains an
open decision and must be resolved before teammate-aware training; it does not
affect the frozen actor because all four new input columns are zero.

A subsequent identifiability audit found that the translated left/right
references are indistinguishable in the original 154-D actor observation before
physical contact. A mixed-side trajectory-ghost manager now infers a constant
side per clip and supplies the correct relative teammate position. This removed
the observation aliasing but did not pass the left-side gate. A 1/7/12/17/25
update scan with the full actor reached a best short left mean episode length of
177.7 steps at update 7, with zero completed trajectories; later updates traded
better table orientation for worse body tracking.

A second bounded audit froze the complete A1 backbone and trained only the
first-layer columns belonging to the appended teammate observation. Checkpoint
diffs confirmed exactly zero change in the original 154 input columns, all
downstream actor layers, biases, and action noise. Neither 25 updates at
`1e-5` nor 7 updates at `1e-4` completed a left trajectory; increasing the
adapter learning rate by ten times left the first two termination points at
approximately 186 and 181 steps. More iterations on either tested adaptation
recipe are therefore not authorized by current evidence.

The isolated retention URDF has a `0.1 kg` base mass, but inherited WBT startup
randomization adds `1--4 kg`; the simulated run is approximately
`1.1--4.1 kg`, not a fixed `0.1 kg`. Do not attribute the asymmetric failure to
the URDF base mass alone.

Stage 1B remains failed and `model_07999_actor158.pt` remains the frozen
baseline. The post-Plan-2 feasibility gate now owns the next decision. Do not
start symmetry augmentation, full retraining, the frozen two-entity preflight,
or paired-reference MARL until the corresponding branch is supported by the
audit and explicitly confirmed. Do not silently relax the WBT object gate and
do not promote any checkpoint from the failed Plan 1/2 diagnostic runs.

### Stage 2 -- Build the two-agent environment

Status: **Blocked until the post-Plan-2 feasibility audit is reviewed and a
specific branch is explicitly selected. The existing geometry and reset
smokes are preserved evidence, not authorization to begin Plan 5.**

#### Work

1. Instantiate two rubber-hand G1 agents and one table.
2. Define mirrored or near-mirrored initial transforms.
3. Materialize the Viser construction as a paired reference asset containing
   synchronized agent-0 and agent-1 trajectories, one shared table trajectory,
   and one shared motion phase. Do not sample two independent single-agent
   motions.
4. Verify collision groups, resets, terminations, and recording.
5. Verify that each actor sees self-local data plus real teammate position and
   velocity.
6. Verify that the critic sees the intended global state.
7. Implement one identical shared reward value for both agents.
8. For Plan 5, initialize both calls to the shared 158-D actor from
   `model_07999_actor158.pt`; initialize the dimensionally new centralized
   critic separately.

#### Exit gate

- deterministic reset produces the intended geometry;
- neither robot nor table begins in collision interpenetration;
- both actors receive the same schema in their respective local frames;
- reward changes have the correct sign under controlled object motion;
- a short rollout records both robots and the table correctly;
- no hemisphere-hand asset is active.

### Stage 3 -- Establish the MARL learning baseline

Stage 3 remains reference-guided WBT in the current scope. It trains the paired
dual-robot demonstration and is not a target-conditioned or task-oriented
stage.

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
- A1 prior budget: reuse the frozen 8,000-iteration checkpoint. The bounded
  Plan 1/2 adaptations failed and are closed. Do not run a new 8,000- or
  30,000-iteration A1 job unless the feasibility gate supports Plan 3 and an
  explicit decision subsequently authorizes the Plan 4 fallback.
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

The active next work remains the post-Plan-2 single-agent side-reference
dynamics feasibility gate. Its read-only rollout-contact phase is complete and
it proceeds in reviewable steps:

1. **Apply the confirmed audit contract.** Cover the `1.1--4.1 kg` effective
   mass envelope, permit small natural yaw and incidental body contact, and
   recover the exact existing friction, actuator, reference, and termination
   values without changing them. Review the numerical yaw tolerance before it
   is used as a feasibility boundary.
2. **Reuse existing recordings for a read-only wrench audit -- complete.** Compute the
   object-reference linear/yaw acceleration requirements, extract actual
   rubber-hand contact forces and moment arms, and compare required versus
   realized table force and yaw moment at the approach, first-contact, sustained
   push, and termination intervals.
3. **Run a bounded reference-state collision and contact-feasibility check --
   active.** Replay exact reference states against the frozen table collision
   geometry, then test whether reachable rubber-hand contacts and admissible
   forces can satisfy the reference wrench while respecting balance, friction,
   and actuator limits. This is analysis, not PPO training. The current physical
   rollouts are not a valid hand-wrench proof because hip/knee propulsion
   dominates all audited cases.
4. **Present one of three evidence-backed outcomes:** feasible with margin,
   feasible only near physical limits, or no acceptable solution under the
   frozen contract. Review uncertainties with the user before interpreting the
   outcome.
5. **Choose the branch only after review.** A robust feasible result permits a
   corrected-reference proposal and later discussion of Plan 3. An infeasible
   or brittle result permits discussion of the frozen two-entity mechanics
   preflight. Neither outcome automatically authorizes Plan 4 or Plan 5.

Until this gate is reviewed, do not edit reference trajectories, change table
physics, add a second physical robot, modify reward/termination logic, or start
another training run.

The Isaac Gym runtime asset check remains pending because the local machine has
no `hsgym` environment. It does not block the current Isaac Sim baseline, but
must not be reported as validated.
