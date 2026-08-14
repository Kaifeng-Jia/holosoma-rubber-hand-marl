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
is sampled near the intended symmetric partner location, with bounded position
and velocity variation. During MARL, these four values are replaced by the
corresponding real teammate observations.

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

### Stage 1 -- Build a MARL-compatible A1 prior

#### Work

1. Audit the exact A1 actor observation and checkpoint tensor shapes.
2. Append the four teammate observation channels.
3. Transfer the original input-layer weights exactly.
4. Initialize the four new input columns to zero or a documented near-zero
   value so the initial policy reproduces the old actor.
5. Add bounded ghost-teammate sampling around the intended partner layout.
6. Continue A1 WBT training only as needed to validate the new interface.

#### Required comparisons

- frozen A1 checkpoint with the original interface;
- expanded A1 checkpoint with the nominal ghost state;
- expanded A1 checkpoint across the frozen ghost-state randomization range.

#### Exit gate

- the expanded actor loads without silently dropping or reordering inputs;
- A1 completion retains at least 90% of the frozen checkpoint's measured
  completion performance;
- fall and early-termination rates do not regress materially;
- ghost-state variation does not cause policy collapse;
- the same actor input schema can consume a real teammate state.

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
- First environment: symmetric or near-symmetric cooperative layout.
- Physics: easy smoke asset followed by a bounded capacity sweep and a frozen
  final mass/friction setting.
- First full budget: 8,000 iterations after the earlier gates pass.

Changes to any frozen decision must be recorded in this document before the
corresponding long-running experiment begins.

## 8. Immediate next steps

1. Complete and verify Stage 0.
2. Perform a read-only audit of the A1 actor observations, network input layer,
   checkpoint format, and WBT reward configuration.
3. Present the exact files and proposed Stage 1 code changes before editing.
4. Implement and test the four-channel ghost-teammate interface.
5. Produce the MARL-compatible A1 retention comparison.
6. Begin the two-agent environment only after Stage 1 passes its gate.
