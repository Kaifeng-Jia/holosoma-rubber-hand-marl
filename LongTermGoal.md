# Long-Term Goal

## Status

- Confirmed: 2026-07-27
- Scope: establish a strong, reproducible baseline across Steps 1–5 before pursuing paper-level methodological innovation
- Platform: Unitree G1 with rubber hands, physically simulated table interaction

## Overall Objective

Build a complete learning pipeline that starts from short human-object interaction demonstrations and ends with multiple G1 robots cooperatively moving a table to a target:

```text
Human demonstration data
    → single-agent physical motion priors
    → autonomous goal-conditioned table manipulation
    → shared-policy multi-agent cooperation
    → emergent roles from local observations
    → measurable credit assignment and division of labor
```

The final system should demonstrate that:

1. A single G1 can physically reproduce multiple demonstrated interaction skills.
2. A single G1 can move a table toward a task goal without reference joint trajectories or a predefined action dictionary.
3. Single-agent skills can be transferred to multiple G1 agents using a shared policy and centralized training.
4. Agents using only their own local observations can adopt different, useful roles without explicit role labels.
5. Each agent's contribution to the cooperative task can be measured quantitatively.

## Step 1 — Mixed-Data Single-Agent Tracking

### Objective

Train one reference-conditioned whole-body tracking policy on multiple valid interaction motions, including pushing, kicking, and verified pulling motions.

The policy answers:

> Given a reference motion, how should the G1 physically execute it?

It does not yet decide which interaction technique should be used.

### Baseline Design

- Start from the original `main` reward design.
- Keep robot-object, rubber-hand-object, and other physically necessary collisions enabled.
- Retain complete short clips, including preparation, initial contact, and post-contact motion.
- Use balanced sampling across action categories rather than sampling proportional to total frame count.
- Train push-only, kick-only, and valid-pull-only expert policies as diagnostic upper bounds.
- Train one mixed-motion policy as the primary Step 1 result.
- Do not provide an action-class label to the mixed policy.
- After reproducing the exact `main` baseline, evaluate a transfer-ready variant in which the actor also observes local table dynamics while the reward remains unchanged.

### Required Outputs

- A versioned motion manifest containing action category, source, retargeting version, contact validity, and observed table displacement.
- Physics-based replay and quality-control results for every retained motion.
- Per-action diagnostic expert checkpoints.
- One mixed-motion tracking checkpoint.
- Per-action videos and quantitative evaluation results.

### Evaluation

- Full-motion completion rate.
- Average successfully tracked episode length.
- Joint and key-body tracking error.
- Table trajectory error.
- Contact establishment success rate.
- Correct table displacement direction.
- Fall and early-termination rates.
- Per-skill performance retention of the mixed policy relative to the corresponding expert.

No action category may silently collapse behind a good aggregate score.

## Step 2 — Goal-Conditioned Single-Agent Manipulation

### Objective

Train one autonomous single-agent policy that observes the G1, the table, and a target, then moves the table toward that target.

The deployed policy should have the form:

```text
robot state + local table state + target
    → continuous 29-DoF joint-position command
```

It must not depend on:

- Human reference joint trajectories.
- A discrete push/kick/pull action dictionary.
- A manually selected motion clip.
- A high-level action-class ID.

### Observations

- G1 proprioception and root state.
- Previous action and necessary contact signals.
- Table pose, orientation, linear velocity, and angular velocity in the G1-local heading frame.
- Target position and target orientation relative to the current table state.
- Fixed table geometry cues only when needed to represent meaningful contact regions.

### Training Transition

1. Initialize from the Step 1 mixed-motion policy.
2. Add table-state and goal observations.
3. Introduce goal progress, terminal accuracy, stability, safety, and smoothness rewards.
4. Initially retain a limited imitation prior to keep physically meaningful movement.
5. Gradually anneal reference-state inputs and imitation regularization.
6. Evaluate a fully autonomous policy with no human reference trajectory at inference time.

### Required Outputs

- One autonomous goal-conditioned single-agent checkpoint.
- A reference-free inference configuration.
- Comparisons between training from scratch, direct Step 1 initialization, and gradual imitation-to-task transfer.
- Evaluation across multiple initial poses, target directions, and target distances.

### Evaluation

- Goal success rate.
- Final table position and orientation error.
- Normalized final error relative to commanded displacement.
- Time required to reach the goal.
- Fall rate and invalid-contact rate.
- Unwanted table rotation.
- Contact establishment time.
- Sample efficiency.
- Worst-direction and worst-initialization performance.

The success tolerance must be selected relative to the intended “push a little” displacement range and frozen before final evaluation; a generic 0.2 m threshold must not be adopted without validation.

## Step 3 — Shared-Policy Multi-Agent Cooperation

### Objective

Transfer the Step 2 single-agent capability to multiple homogeneous G1 agents.

### Baseline Design

- Initialize all agents from the Step 2 checkpoint.
- Use one parameter-shared actor.
- Give each actor only its own local observation.
- Use a centralized critic with global agent and object state during training.
- Use MAPPO/centralized training with decentralized execution.
- Begin with two agents, then test additional agents.
- Keep physical inter-agent and agent-object collisions enabled in the main baseline.

### Required Comparisons

- Single-agent policy copied to multiple agents without joint fine-tuning.
- Multi-agent training from scratch.
- Step 2 initialization followed by multi-agent fine-tuning.

### Required Output

A shared-policy multi-agent checkpoint that improves cooperative success, training stability, or manipulable object range over the single-agent baseline.

## Step 4 — Emergent Roles from Local Observations

### Objective

Determine whether agents sharing the same policy can adopt different useful behaviors solely because they occupy different local states.

### Baseline Design

- No explicit role labels.
- No agent-specific push, kick, or pull command.
- No agent ID unless introduced later as a separately reported ablation.
- Randomize initial positions, orientations, distances, target directions, and agent count.
- Preserve a shared actor and local actor observations.

### Required Output

A policy and evaluation protocol showing whether distinct roles emerge and whether agents can exchange roles when their spatial conditions change.

### Evaluation

- Contact location on the table.
- Contacting body part.
- Contact-force direction and magnitude.
- Object-directed mechanical contribution.
- Role duration and role-switch frequency.
- Role diversity across initial conditions and random seeds.
- Task success with and without individual role patterns.

Visual differences alone are not sufficient evidence of role emergence.

## Step 5 — Credit Assignment and Quantifiable Division of Labor

### Objective

Measure and improve how cooperative reward is attributed to individual agents.

### Baseline Design

Compare:

1. Shared team reward with standard MAPPO advantages.
2. Agent-specific advantages produced by a centralized critic.
3. A standard counterfactual or difference-reward baseline when computationally practical.

Separate training-time credit assignment from evaluation-time causal contribution measurement.

### Causal Evaluation

For each agent, measure the change in team performance when:

- The agent is removed.
- Its action is replaced with a standing/no-op action.
- Its object collision is disabled for evaluation.
- Its policy is frozen while the remaining agents act normally.

### Required Output

- Per-agent contribution estimates.
- Leave-one-agent-out performance measurements.
- Correlation between assigned credit and causal performance loss.
- Credit-assignment ablations covering success rate, sample efficiency, stability, and contribution balance.

## Cross-Stage Acceptance Principles

- Use multiple random seeds and report variance.
- Report per-skill and worst-case results, not only aggregate averages.
- Separate kinematic similarity from physically valid interaction.
- Keep robot, actuator, torque, balance, contact, and collision constraints central to evaluation.
- Freeze evaluation splits and success definitions before final experiments.
- Record negative results and failed motion subsets instead of silently removing them.
- Do not advance to the next step when the current step lacks a reproducible checkpoint and fixed evaluation protocol.
- Preserve exact-main and from-scratch controls throughout the pipeline.

## Lessons Adopted from Related Work

### CooHOI

[CooHOI](https://arxiv.org/abs/2406.14558) supports:

- Single-agent skill learning followed by multi-agent policy transfer.
- Shared object dynamics as feedback and an implicit communication channel.
- Parameter sharing with centralized training and decentralized execution.
- Including backward and lateral movement priors to avoid cooperative deadlock.
- Comparing transfer against multi-agent training from scratch.

The baseline will not directly copy its box-specific standing points, held points, phase rewards, or generic box proxies because those designs could predetermine contacts and roles.

### SynAgent

[SynAgent](https://arxiv.org/abs/2604.18557) supports:

- Interaction-aware retargeting and strict motion quality control.
- Per-subset experts as diagnostics or teachers before unifying diverse skills.
- Gradual transition from imitation to autonomous object-level control.
- Physically rolled-out state-action pairs as higher-quality supervision.
- Shared decentralized policies initialized from single-agent experience.

Its interaction-mesh retargeting, multi-teacher DAgger pipeline, and trajectory-conditioned CVAE remain possible later extensions rather than requirements for the initial strong baseline.

## Current Non-Goals

The initial strong baseline does not require:

- Paper-level methodological novelty.
- Sim-to-real deployment.
- Raw-camera perception.
- Learned object category generalization.
- Large-scale geometry generalization.
- Explicit language commands.
- A hierarchical discrete action selector.
- Manually assigned cooperative roles.
- A complex generative policy or mixture-of-experts architecture.

These may be revisited only after Steps 1–5 are reproducible and quantitatively validated.

## Next Planning Task

Review the current repository and experimental state, identify the exact short-term milestone currently in progress, and create an ordered implementation and verification plan leading to the next acceptance gate.
