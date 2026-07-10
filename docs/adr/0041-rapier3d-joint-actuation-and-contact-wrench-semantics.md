---
project: rlevo
status: active
type: decision
date: 2026-07-10
tags: [adr, decision, physics, rapier, locomotion, joint-torque, contact-force, cfrc_ext, mujoco]
---

# ADR 0041: Rapier3D joint actuation and contact-wrench semantics

## Status

**Accepted (2026-07-10).** Resolves the three 🔴 High findings of issue #123
(review §5.1, §1.1, §1.2) for the rapier3d locomotion backend. Extends ADR 0037
(external-force lifetime / substep actuation); supersedes nothing.

## Context

`Rapier3DBackend` (`crates/rlevo-environments/src/locomotion/backend/`) exposes
two seams that were under-specified and, in one case, unimplemented:

1. **`apply_joint_torque`** was an `unimplemented!` landmine documented as
   "always panics". It had **zero callers** — all four locomotion envs bypass it
   and apply torque directly to bodies via `RigidBodySet::add_torque` inside
   their `step_actuated` hooks. It was on the v0.2 actuation path, so a joint-
   level torque primitive is needed before any env can drive an articulated
   chain through the trait.
2. **`contact_force`** returns a 6D wrench read by `InvertedDoublePendulum` (and,
   later, Ant/Humanoid `contact_cost`). Its trait/impl docs made a **false claim**
   ("matches MuJoCo's `cfrc_ext` layout") and there were **zero tests** pinning
   its magnitude, sign, or frame-skip behaviour. Review §1.1 further feared the
   force was `1/frame_skip`-scaled.

The literature reconciliation (vault note
`docs/.private/research/2026-07-10-issue-123-rapier3d-joint-torque-contact-force.md`)
established the *time semantics* of the contact math are already correct:
MuJoCo's `cfrc_ext` is an **instantaneous** per-timestep force recomputed by
`mj_rnePostConstraint` after the last of `frame_skip` substeps (Gymnasium calls
it once after `mj_step(nstep = frame_skip)`), so rapier's last-substep
`impulse / substep_dt` is the right analogue. The real defects there are
**doc imprecision** and **missing tests**, not the arithmetic.

Two rapier-0.32-specific facts, verified against the vendored source, shape the
decision:

- **`GenericJoint::local_axis1()` is anchor-contaminated under the glam backend.**
  parry3d 0.26 wraps **glam** (`Vector = glam::Vec3`, `Rotation = glam::Quat`),
  and glamx's `Pose3 * Vec3` is `transform_point` (applies rotation **and**
  translation). rapier's `local_axis1()` is `local_frame1 * Vector::X`, written
  for nalgebra's `Isometry * Vector` (rotation-only) semantics, so under glam it
  returns `axis + anchor` — a **non-unit, contaminated** vector. Empirically a
  hinge built with axis `+Z` and anchor `(0,0,0.5)` reports `local_axis1()`
  `= (0,0,1.5)`.
- **Same-parent contact pairs are unconditionally cleared** (rapier3d-0.32.0
  `narrow_phase.rs:841`, "Same parents. Ignore collisions."), matching MuJoCo's
  undisableable same-body geom filter. Jointed *neighbour* pairs, however, still
  collide by default (`contacts_enabled` defaults `true`).

## Decision

### 1. Joint-kind-dispatched torque application (`apply_joint_torque`)

New signature (was `-> ()` with an `unimplemented!` body):

```rust
fn apply_joint_torque(
    world: &mut Self::World,
    joint: Self::JointHandle,
    torque: f32,
) -> Result<(), BackendError>;
```

`torque` is the **generalized force on a revolute joint's single free angular
DOF** — the analogue of a MuJoCo `motor` actuator on a hinge (`gear × ctrl`).
**Gear scaling stays the caller's responsibility**, unchanged. The torque lives
exactly **one physics substep** (ADR 0037): callers hold it across `frame_skip`
by re-applying inside `Rapier3DWorld::step_actuated`.

The impl **dispatches on joint kind** (this split is load-bearing — the maximal-
vs reduced-coordinate mechanics differ):

- **`Impulse` (maximal coordinates).** Both bodies are free and the solver
  enforces the hinge, so apply **equal-and-opposite** world-axis torques:
  `add_torque(+τ·â)` to `body2` and `add_torque(−τ·â)` to `body1`. The pair
  injects **zero net external torque**, so the generalized force lands on the
  hinge DOF alone (regression-tested via angular-momentum cancellation).
- **`Multibody` (reduced coordinates).** The hinge is baked into the
  parameterization, so torque the **child only** (`body2` of the parent→child
  insertion): `add_torque(+τ·â)` on the child projects onto the hinge DOF and
  the solver supplies the parent reaction. Also torquing the parent would inject
  spurious generalized force on upstream DOFs — this is the existing
  `swimmer/env.rs` child-only convention (its "double-torquing" warning).

**Sign convention.** A positive `τ` drives `body2` positively about the joint's
free axis `+â` relative to `body1`, where `â` = `body1`'s world rotation applied
to the unit hinge axis.

**Insertion invariant.** Multibody joints are inserted **parent→child**
(`b1 = parent`, `b2 = child`); the `MultibodyJointHandle` indexes the child
link, whose `parent_id` gives `body1`. `Rapier3DWorld::add_multibody_joint`
callers (reacher, swimmer, …) already follow this.

**Axis extraction — avoid `local_axis1()`.** Because `local_axis1()` is anchor-
contaminated under the glam backend (see Context), the world hinge axis is
computed from the **rotation only**: `â = R_body1 · (local_frame1.rotation · X)`.
This yields the correct **unit** axis regardless of anchor offset. Using
`local_axis1()` instead would scale the applied torque by the anchor magnitude
(a `1.5×` error in the two-body regression) and, for off-axis anchors, tilt the
axis outright.

### 2. Revolute-only domain; `Result`-based rejection

A scalar torque is a generalized force on **one** free angular axis, so the
domain is **revolute joints only**. `BackendError` (new, in `backend/mod.rs`,
`thiserror`-derived to match the workspace's 26 existing error enums — none use
`#[non_exhaustive]`, so this one does not either) carries:

- `InvalidJointHandle` — stale/unknown handle, or a missing attached body.
- `UnsupportedJoint(&'static str)` — no single free angular axis (prismatic,
  spherical, fixed). Detection: all linear axes locked **and** exactly one free
  angular axis (`ANG_AXES − locked_axes` has one bit) — accepts rapier's
  `LOCKED_REVOLUTE_AXES`, rejects the rest.

Rejecting rather than mis-actuating honours `docs/rules.md §4` (never silently
mishandle an out-of-domain argument). **Prismatic / linear (cart-force)
actuation is explicitly deferred** — the linear-actuation seam is separate, and
envs continue to drive cart forces directly via `bodies_mut().add_force` for
now. (Deferral filed per `rules.md §12`.)

### 3. `contact_force` semantics — docs + tests (sign later corrected)

The *time-scaling* arithmetic is **not** touched (review §1.1's feared
`1/frame_skip` rescale would be *wrong* — see Context). The trait/impl docs are
rewritten and pinning tests added. The **sign attribution was subsequently
corrected under this same issue** (see Negative / accepted costs): the returned
wrench is the external contact force-torque acting **ON the queried body**, so a
resting ball reports `wrench[2] ≈ +m·g`. Precise semantics:

- **Sign — force ON the queried body.** `−force_mag·normal` when the body owns
  `collider1`, `+force_mag·normal` when it owns `collider2`; insertion-order
  invariant; Newton's-third-law antisymmetric across a contacting pair.

- **Instantaneous last-substep force.** Per-substep average
  `impulse / substep_dt` from the **last** solved substep — not a step-integrated
  or frame-skip-averaged quantity. Matches Gymnasium's read-after-last-substep
  `cfrc_ext` semantics; **independent of `frame_skip`** (pinned by a
  `frame_skip = 1` vs `5` steady-state test).
- **Layout `[fx, fy, fz, tx, ty, tz]` (force first)** — an **intentional
  deviation** from MuJoCo's `cfrc_ext`, which is `[torque(3), force(3)]`
  ("rotation(3), translation(3)" per `mj_rnePostConstraint`). The prior "matches
  `cfrc_ext` layout" claim was false and is removed.
- **Torque reference point: the body's own centre of mass**, whereas MuJoCo uses
  the kinematic-**subtree** CoM — identical for a leaf body, materially different
  for an internal one (a documented Ant/Humanoid `contact_cost` calibration
  caveat).
- **Content: contact-manifold forces only** — a subset of `cfrc_ext`'s full RNE
  post-constraint external wrench.
- **Self-contacts never contribute:** same-parent collider pairs are
  unconditionally cleared by rapier 0.32 (`narrow_phase.rs:841`), matching
  MuJoCo's undisableable same-body filter — pinned by a two-collider-one-body
  zero-wrench test, with the rapier version + file cited in a code comment.

### 4. Skeleton-builder convention: disable jointed-neighbour contacts

Adjacent skeleton links (an inner cap and its neighbour) overlap at the shared
joint anchor and would otherwise generate **permanent internal contacts**
polluting `contact_force`. MuJoCo's parent–child geom filter excludes exactly
these. The four rapier3d env builders (InvertedPendulum, InvertedDoublePendulum,
Reacher, Swimmer) therefore set **`.contacts_enabled(false)`** on their skeleton
joints. *(That env-side change is applied by a concurrent change under this same
issue; this ADR records the convention — reacher/swimmer already cite ADR 0041.)*

## Consequences

### Positive

- `apply_joint_torque` is a correct, testable primitive: envs can migrate off
  direct `add_torque` onto the trait when convenient, with backend-uniform
  semantics and a documented sign convention.
- `contact_force` semantics are now precisely specified, with the MuJoCo
  deviations (layout, torque frame, content) called out so downstream calibration
  (Ant/Humanoid `contact_cost`) is not silently miscalibrated.
- Regression coverage: sign, equal-and-opposite (no net torque), multibody
  child-only equivalence, error paths, contact magnitude, frame-skip invariance,
  and same-body zero-wrench are all pinned.

### Neutral

- `BackendError` is additive; the trait method's return type changed but had zero
  callers, so nothing downstream breaks.

### Negative / accepted costs

- **`contact_force` sign — RESOLVED (2026-07-10, same issue).** The initial
  pinning test recorded an *inverted* vertical sign (a resting ball reported
  **negative** `wrench[2]`, the opposite of the physical "ground pushes up"),
  left pinned-but-unfixed under Change 2's original "docs + tests only" scope.
  That restriction was subsequently lifted **for the sign only**, and the
  attribution is now corrected: `contact_force` returns the external contact
  force-torque acting **ON the queried body**, so a resting ball reports
  `wrench[2] ≈ +m·g`. Root cause was a two-branch sign inversion in the `n`
  computation: parry's manifold normal points from `collider1` toward
  `collider2` (parry3d-0.26.1 `contact_manifolds/contact_manifold.rs:449`) and
  the solver drives the non-negative `contact.data.impulse` along `dir1 =
  −normal` on collider1's body / `+normal` on collider2's body
  (`solver/contact_constraint/contact_with_coulomb_friction.rs:83`,
  `contact_constraint_element.rs:282`/`285`), so the force ON the queried body is
  `−force_mag·normal` when it owns `collider1` and `+force_mag·normal` when it
  owns `collider2` — both branches were previously reversed. The fix flips
  `InvertedDoublePendulum`'s `obs[8]` sign, which is ≈ 0 in normal operation
  (jointed contacts disabled), and is `contact_cost`-neutral (that squares the
  wrench). Pinned by a resting-ball sign test, a Newton's-third-law antisymmetry
  test, and an insertion-order-robustness test.
- **Contact-force sign is insertion-order-INDEPENDENT — RESOLVED.** Swapping
  which collider is `collider1` flips **both** `manifold.data.normal` and the
  `flipped` flag, so the attributed force is unchanged. The earlier concern that
  the sign depended on insertion order was disproved by source analysis and is
  pinned by the insertion-order-robustness test (ground-first vs ball-first both
  yield `wrench[2] > 0`).
- `contact_force` magnitude carries rapier's steady-state penetration bias
  (~1.25× m·g at rest), so the magnitude test asserts a band, not equality.

## Alternatives considered

- **Motor-API actuation (high damping / zero stiffness) for `apply_joint_torque`.**
  Rejected: a motor drives toward a target velocity/position, not a pure torque;
  mapping a generalized force onto motor gains is indirect and backend-fragile.
  Equal-and-opposite `add_torque` (impulse) / child-only `add_torque` (multibody)
  is the direct MuJoCo-`motor` analogue.
- **Torque both bodies in the multibody case (mirror the impulse path).**
  Rejected: in reduced coordinates the hinge reaction is solver-supplied;
  torquing the parent too injects spurious force on upstream DOFs (the
  swimmer "double-torquing" hazard).
- **Use `local_axis1()` as the brief's design named.** Rejected on evidence: it
  is anchor-contaminated under the glam backend (non-unit / off-axis). Extract
  the axis from `local_frame1.rotation` instead.
- **Fix the `contact_force` sign now.** Initially rejected (out of the declared
  "docs + tests only" scope, mid-flight with a concurrent env change) and pinned
  in the interim; the restriction was then lifted **for the sign only** and the
  fix landed under this same issue (see Negative / accepted costs). The
  time-scaling and layout decisions above are unaffected.
- **Rescale `contact_force` by `frame_skip` (review §1.1).** Rejected: `cfrc_ext`
  is instantaneous; the current last-substep `impulse/dt` is already correct.

## References

- Issue #123 — rapier3d backend: unimplemented joint torque, wrong contact force.
- Vault research note (literature/source reconciliation, with citations):
  `docs/.private/research/2026-07-10-issue-123-rapier3d-joint-torque-contact-force.md`.
- ADR [0037](0037-external-force-lifetime-and-substep-actuation.md) — one-substep
  external-force lifetime and `step_actuated`, on which the torque re-application
  contract rests.
- Code: `crates/rlevo-environments/src/locomotion/backend/mod.rs` (`BackendError`,
  trait docs), `.../backend/rapier3d.rs` (`apply_joint_torque` dispatch,
  `contact_force` docs + tests); `.../locomotion/swimmer/env.rs` (child-only
  multibody torque convention cited in the multibody branch).
- rapier3d 0.32 source: `dynamics/joint/generic_joint.rs` (`local_frame1`,
  `JointAxesMask`), `geometry/contact_pair.rs` (`ContactData` normal-impulse
  convention), `geometry/narrow_phase.rs:841` (same-parent clearing);
  parry3d 0.26 / glamx 0.1 (`Pose3 * Vec3 = transform_point`).
