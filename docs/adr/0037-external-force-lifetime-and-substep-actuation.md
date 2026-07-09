---
project: rlevo
status: active
type: decision
date: 2026-07-09
tags: [adr, decision, physics, rapier, box2d, locomotion, force-accumulation, substep, actuation]
---

# ADR 0037: External-force lifetime and substep actuation

## Status

**Accepted (2026-07-09).** Resolves the force-accumulation half of issue #98
across both physics backends. Supersedes nothing.

## Context

Both physics backends drive actuated environments by pushing an external force
or torque onto a rigid body every environment step and then advancing the
solver. Both use engines whose "apply force" primitive is **additive**, not
set-and-hold:

- rapier3d 0.32 `RigidBody::add_force` / `add_torque` do `user_force += force`.
  The pipeline **never clears** the accumulator on its own — the widely-repeated
  "external forces are auto-cleared each step" claim is **false** for 0.32. Only
  `reset_forces` / `reset_torques` zero it, and before this ADR there were **zero
  callers**. Every environment that re-applies control each step (InvertedPendulum,
  InvertedDoublePendulum, Reacher, Swimmer) was therefore integrating a
  *monotonically growing sum* of every force it had ever applied, not the force
  for the current step. A constant action produced a linearly-growing effective
  force; velocities grew super-linearly and, for the low-inertia Swimmer chain
  under a monotone action, diverged.
- The box2d backend has the analogous additive-accumulator shape and is fixed in
  the same spirit by a parallel engineer (see below).

MuJoCo — the reference these environments track — has no such hazard: an
actuator force is `gear · ctrl`, **set** (not accumulated) fresh each substep and
held **constant** across the `frame_skip` substeps of one control step. The bug
is a backend-integration defect, not a modeling choice.

## Decision

**Core contract: an external force lives exactly ONE integration step.** A force
applied before a solver step is integrated once and then cleared; it never
carries into the next step. This is enforced inside the step *primitive* of each
backend, so no environment can opt out or forget.

### rapier3d (`crates/rlevo-environments/src/locomotion/backend/rapier3d.rs`)

1. `Rapier3DWorld::step_once` calls a private `reset_external_forces(&mut self)`
   **after** `pipeline.step(...)`. `reset_external_forces` iterates
   `bodies.iter_mut()` and calls `reset_forces(false)` + `reset_torques(false)`
   on each body. This makes the primitive self-consistent: forces applied before
   a `step_once` are integrated once, then zeroed.
2. `wake_up = false` on the reset: zeroing an accumulator must not wake a
   sleeping body. Actuated bodies are already awake from their
   `add_force(.., true)` / `add_torque(.., true)` calls; a body with no applied
   force has nothing to clear and must be left asleep.
3. New primitive `Rapier3DWorld::step_actuated(&mut self, apply: impl FnMut(&mut Self))`
   advances `frame_skip.max(1)` substeps, invoking `apply` immediately **before
   each** `step_once`. This is the substep-actuation convention: to hold an
   actuator constant across the frame skip (MuJoCo `ctrl`-held-across-substeps),
   the caller re-applies the *same* precomputed control every substep, and drag /
   velocity-dependent forces are recomputed each substep from live velocity.

### box2d (parallel engineer, this issue)

The box2d half (`box2d/physics.rs`, `car_racing`, `lunar_lander`) is implemented
separately under the same contract: `RapierWorld::step` resets forces after
`pipeline.step`, so a force applied by an environment lives exactly one step
there too. The two backends share the *contract* (one-integration-step lifetime)
even though the reset call sites differ.

### Environment convention (`frame_skip > 1`)

Actuated environments **precompute** all control values before stepping, then
pass a closure to `step_actuated` that captures `Copy` body handles and the
control array and borrows only the world (never `self` — that would be a borrow
conflict; `RefCell` is not used). Per env:

- InvertedPendulum (`frame_skip = 1`): re-apply the cart force each substep.
- InvertedDoublePendulum (`frame_skip = 1`): same.
- Reacher (`frame_skip = 2`): re-apply both joint torques each substep.
- Swimmer (`frame_skip = 8`): re-apply the constant actuator torque **and**
  recompute viscous drag from current velocity each substep. The Swimmer world is
  now built with `frame_skip = config.frame_skip` (was `1` with a hand-rolled
  substep loop) so `step_actuated` owns the loop. `apply_drag` became a free
  helper `apply_drag_to(world, handles, k, k_ang)` so it can run inside the
  world-only closure. Swimmer's `gear [5, 5]` and both drag coefficients are
  **frozen**: re-tuning toward canonical Swimmer-v5 (`gear [150, 150]`) is a
  deferred follow-up — #98 fixes only accumulation and must not silently change
  gear.

### Unaffected paths

Motor-driven joints (the joint-motor API, e.g. the box2d bipedal walker) do not
use the `add_force` / `add_torque` accumulator and are unaffected. Snapshot /
restore does not capture external forces — moot under the one-step lifetime,
since after any `step_once` the accumulator is always zero.

## Consequences

### Positive
- **Correct physics by construction.** A constant action yields a constant
  effective force; per-step velocity increments are stationary (verified by
  backend and per-env regression tests). The defect cannot recur in a new env,
  because the primitive clears the accumulator regardless of caller behavior.
- **Backend parity.** Both backends now honor the same one-integration-step
  contract, described in one place.
- **Swimmer stability.** A monotone constant action no longer diverges the chain.

### Neutral
- `step_actuated` replaces the `apply_action`-then-`step` call shape in the four
  rapier3d envs; `apply_action` methods became pure `control_*` computations.
- Swimmer's manual substep loop is gone; the world's `frame_skip` now drives it.

### Negative / accepted costs
- **Behavioral change (alpha).** Non-accumulating dynamics differ from the old
  buggy trajectories; a driven pendulum/tip now falls on the correct (slower)
  timescale rather than being flung by an accumulating force. Existing
  termination tests still pass (the poles sit at unstable equilibria, so even a
  correct constant force eventually tips them); no test was weakened to preserve
  the bug.
- **Per-substep re-application cost.** `frame_skip` closure invocations per env
  step instead of one force application. Negligible against the solver step.

## Alternatives considered

- **Clear at the top of the next env step (in `Environment::step`).** Rejected:
  leaves the accumulator dirty between calls, keeps the hazard reachable by any
  caller that steps the world directly, and does not give the "force lives one
  substep" guarantee within a `frame_skip` loop.
- **Set (overwrite) the accumulator instead of add-then-clear.** rapier3d exposes
  no set primitive; drag and actuator forces on the same body must sum, so an
  add-then-clear-after-integrate scheme is the natural fit and keeps the additive
  API usable.
- **Wake bodies on reset (`wake_up = true`).** Rejected: needlessly wakes idle
  bodies every step, defeating the island sleep optimization, for no benefit —
  a body with no applied force has a zero accumulator.
- **Keep Swimmer's hand-rolled substep loop and only fix the accumulator.**
  Rejected: `step_actuated` is the shared convention; duplicating the loop keeps
  the false "rapier auto-clears" folklore alive in a second place.

## References
- Issue #98 — env/rapier `user_force` never cleared; SteeringEngine forces
  accumulate across steps. This ADR resolves the force-accumulation scope; RNG
  reseeding, finiteness guards, terminal-state guards, and config validation
  named in the issue are split to separate follow-ups.
- Swimmer gear/drag re-tuning toward canonical Swimmer-v5 — deferred follow-up.
- Code: `crates/rlevo-environments/src/locomotion/backend/rapier3d.rs`
  (`step_once` reset, `step_actuated`, `reset_external_forces`);
  `locomotion/{inverted_pendulum,inverted_double_pendulum,reacher,swimmer}/env.rs`
  (substep actuation); box2d half (`box2d/physics.rs`, `car_racing`,
  `lunar_lander`) by a parallel engineer under the shared contract.
- ADR [0029](0029-host-rng-seeding-convention.md) — the per-reset RNG convention
  whose #98 follow-up is explicitly *not* part of this ADR.
