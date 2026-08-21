---
project: rlevo
status: active
type: decision
date: 2026-07-30
tags: [adr, decision, bounds, config, validation, rlevo-core, rlevo-reinforcement-learning, log-std, guard-test]
---

# ADR 0068: `Bounds` strictness is enforced asymmetrically by crate — a named helper everywhere, a mechanical guard only in `rlevo-reinforcement-learning`

## Status

**Accepted (2026-07-30).** Resolves issue #387 ("migrating an ordered scalar
pair to a `Bounds` field silently drops the strictness half of the old check").

**Supersedes nothing.** It refines ADR [0027](0027-bounds-newtype-for-closed-ranges.md)'s
inclusive-invariant decision (Decision 2) (the deliberate inclusive `lo <= hi`
invariant) and ADR [0054](0054-policy-head-construction-is-fallible.md)'s
Bounds-field decision (Decision 3) (the two explicit `config::distinct` lines
that preserve strictness for `log_std`) into a named mechanism plus one
mechanical check. `Bounds`'s inclusive invariant is **not** re-litigated — see
this ADR's own Alternatives-considered section, below.

**Chosen shape.** Three parts, deliberately unequal:

1. `config::nondegenerate_bounds(C, field, b: Bounds)` in `rlevo-core::config`,
   delegating to `config::distinct` — **not** to `b.span() > 0.0`.
2. A source-text guard test scoped to **`rlevo-reinforcement-learning` only**,
   in the shape of `crates/rlevo-environments/tests/rng_seeding_guards.rs`.
   Its allowlist is 2 rows today, with an empty exemption list.
3. Prose convention only (`docs/rules.md`'s Config Validation Contract, in
   its Error Handling section, plus the `Bounds` rustdoc) for
   `rlevo-evolution` and `rlevo-environments`, where zero width is legitimate.

Additive: no existing public signature changes, no numerics change, no
serialized-form change. Both affected configs already carry the check this ADR
names (ADR 0054's Bounds-field decision, Decision 3); the helper renames it
and the guard pins it.

## Context

### The trap is real and the mechanism is exact

`config::ordered` (`rlevo-core::config`) requires strict `low < high`.
`Bounds::try_new` (`crates/rlevo-core/src/bounds.rs`) accepts `lo <= hi` and
**deliberately permits** `lo == hi` — ADR 0027's inclusive-invariant decision
(Decision 2) gives the reasons, and they are good ones: `clamp` is
well-defined on a single point, and every search-space consumer samples
`lo + (hi - lo) * r`, which degenerates to the constant `lo` with no
division and no `NaN`.

So folding an `ordered`-checked scalar pair into one `Bounds` field discharges
the **ordering** half of the old check and silently drops the **strictness**
half. The diff reads as a pure refactor — a `config::ordered` line deleted
because the type now guarantees it — while being a semantic loosening. That is
the whole content of #387, and it is correct as a description of the mechanism.

### It has no live defect, and the audit that says so had not been run

#387 files this as a general trap, and it is right about the trap. It is not a
bug report: **there is no un-guarded instance in the workspace.** The one
genuine case — the two Gaussian policy-head configs' `log_std` — was already
closed by ADR 0054's Bounds-field decision (Decision 3), which foresaw
exactly this loosening and added an explicit line to both configs:

- `crates/rlevo-reinforcement-learning/src/algorithms/ppo/policies/gaussian.rs:284`
- `crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_policy.rs:123`

The issue notes that the sweep across the other migrations had not been done.
It has been done for this ADR, and the result is that every other migration is
the deliberate, documented choice of ADR 0027's inclusive-invariant decision
(Decision 2) and migration-scope decision (Decision 6) — and that ADR 0027's
error-surface decision (Decision 5)'s stated mitigation actually landed,
rather than being promised and forgotten: `debug_assert!(span > 0.0, ..)`
sits in `crates/rlevo-evolution/src/local_search/simulated_annealing.rs:79` and
`crates/rlevo-evolution/src/local_search/random_restart.rs:83`, in the two
`default_for` constructors whose `step_size = 0.1 * (hi - lo)` a zero-width
range would flatten to `0.0`.

Recorded plainly because it decides the shape below: **the population needing
strictness is 2, and both were already fixed before this ADR was written.**

### The ratio is perfectly crate-partitioned, and that is the whole argument

Every `Bounds`-typed struct field in the workspace's production crates was
enumerated (`rg -n '^\s+(pub )?[a-z_]+: (Bounds|Option<Bounds>),$' crates/`,
excluding `tests/`, `examples/`, `benches/`). 32 fields:

| crate | fields | zero width legitimate? |
|---|---|---|
| `rlevo-environments` | 10 — 4 `action_clip` (`reacher`, `swimmer`, `inverted_pendulum`, `inverted_double_pendulum`), 3 `HealthyCheck` `Option<Bounds>` ranges (`locomotion/common.rs:107,109,112`), 3 mountain-car (`pos_bounds` $\times 2$, `action_bounds`) | **Yes** — a pinned clamp is meaningful |
| `rlevo-evolution` | 20 — 18 `pub bounds` on the metaheuristic/EA configs, plus 2 **private** `bounds` on `HillClimbingParams` (`hill_climbing.rs:50`) and `SimulatedAnnealingParams` (`simulated_annealing.rs:53`) | **Yes** — a zero-width search space is degenerate-but-safe (ADR 0027's inclusive-invariant decision, Decision 2) |
| `rlevo-reinforcement-learning` | 2 — `log_std` on `TanhGaussianPolicyHeadConfig` and `SquashedGaussianPolicyHeadConfig` | **No** — a zero-width $\log \sigma$ range is a silent $\sigma$ collapse |
| `rlevo-core`, `rlevo-hybrid`, `rlevo-benchmarks` | 0 | — |

**The minority is not scattered. It is one crate.** That fact, not the 2-vs-30
ratio on its own, is what licenses a guard whose scan covers one crate: a
crate-scoped guard is only worth writing if the population it must see is
crate-shaped, and here it is. A workspace-wide scan of the same pattern would
need ~32 allowlist rows, 30 of them saying "zero width is fine here, see
ADR 0027's inclusive-invariant decision (Decision 2)" — which is 30 rows of
restating an accepted ADR, and 30 rows of stale-row maintenance, to protect 2
fields.

Two details of the enumeration matter for the guard's implementation and are
recorded so they are not rediscovered. First, the pattern must **not** be
`pub`-only: two of the evolution fields are private, and while both are outside
the guard's scope today, a `pub`-only scan is the kind of near-miss that reads
as complete. Second, `crates/rlevo-core/src/config.rs` matches the same regex
on `nondegenerate_bounds`'s own `b: Bounds` **parameter** — the guard resolves
matches to struct fields, not to any `: Bounds` occurrence.

### Why the $\sigma$ collapse is not merely "useless"

Both RL sites already document this in-file, and it is the reason the two-item
minority is worth mechanizing at all rather than left to review:

- **PPO** (`gaussian.rs`): `log_std` is a *single shared `Param`*. Pinning it to
  a constant freezes $\sigma$ **and its gradient** from step 0, with no path back.
- **SAC** (`sac_policy.rs`): the `log_std` head still receives gradient through
  the mean path, so nothing is permanently frozen — but the policy becomes
  state-independently deterministic in scale, which is precisely the quantity
  SAC's entropy temperature is tuned against.

Neither produces a `NaN`, a panic, or a failed assertion. Both produce a run
that trains and reports finite numbers — the same failure class ADR 0054's
Context section identified for the inverted case, minus the backend
divergence that made the inverted case findable.

## Decision

### 1. `config::nondegenerate_bounds` delegates to `config::distinct`, not to `span() > 0.0`

```rust
pub fn nondegenerate_bounds(
    config: &'static str,
    field: &'static str,
    b: Bounds,
) -> Result<(), ConfigError>
```

This is the subtlest clause in the ADR and the one most likely to be
"simplified" by a well-meaning reviewer, so the reasoning is recorded here and
repeated as a comment at the implementation.

A `b.span() > 0.0` test is the obvious spelling and it is **wrong**:
`Bounds::new(-20.0, f32::INFINITY)` has span `inf`, and `inf > 0.0` is `true`.
It would be accepted. Nothing downstream catches it on the SAC head — ADR 0049's
floor (`>= -35`) and span (`< 40`) checks exist on the PPO path, but the SAC
head has no span check, so the infinite endpoint would reach
`exp(log_std.hi())` unremarked. Delegating to `config::distinct` inherits ADR
[0060](0060-config-values-must-be-finite.md)'s finiteness guard on both
arguments and rejects it.

Three consequences of the delegation, all intended:

- **The helper is strictly stronger than "nonzero span."** It rejects a
  non-finite endpoint too. Per ADR 0060 that is correct: a `Bounds` **field of a
  config** is a config *value*, not `in_range`'s schema-level bound.
- **The name is therefore slightly wider than the check.** "Nondegenerate" also
  excludes $\pm\infty$. This naming wart is documented at the helper rather than
  papered over; a field whose legitimate domain *includes* a one-sided infinite
  range — `HealthyCheck::z_range`'s $[0.7, \infty)$ — simply does not call this
  helper and relies on the `Bounds` invariant alone.
- **Error semantics are preserved byte-for-byte** against the two lines from
  ADR 0054's Bounds-field decision (Decision 3) it replaces: same
  `ConstraintKind::DegenerateInterval`, same `field`.
  This is what makes the change to `gaussian.rs`/`sac_policy.rs` a rename rather
  than a behaviour change, and it is why the existing rejection tests survive
  unmodified.

It also keeps one explanation of `clippy::float_cmp` in the workspace, in
`distinct` — the reason that predicate is spelled `(a - b).abs() > 0.0` rather
than `a != b`.

### 2. The mechanical guard is scoped to `rlevo-reinforcement-learning`

A source-text guard test in `crates/rlevo-reinforcement-learning/tests/`,
modelled on `crates/rlevo-environments/tests/rng_seeding_guards.rs` and
inheriting its four properties without change:

- **Bidirectional allowlist.** Every `Bounds` field found in the crate's `src/`
  must be on the list, *and* every list row must still resolve to a live field.
  A row that no longer matches is a stale row and fails — the property ADR
  0062's guard-mechanization decision (Decision 4) borrowed from
  `landscape_dim_guards.rs`, and the reason the check does not rot into a
  permanently-green no-op.
- **`#[cfg(test)]` regions are skipped**, with the same cost accepted for the
  same reason: a test-only config cannot ship, and requiring every test fixture
  to edit the allowlist is how a guard gets deleted.
- **Fail loud, with the repair in the message** — the failure text names this
  ADR, `config::nondegenerate_bounds`, and ADR 0027's inclusive-invariant
  decision (Decision 2) (so a reader who
  concludes "zero width is fine for my field" learns immediately that the answer
  is to move the field out of this crate's scope or justify an exemption row,
  not to widen the check).
- **Its limits are the same and are stated, not discovered.** It reads source
  text: brittle to reformatting, defeatable by aliasing or by a type alias for
  `Bounds`. It catches the accident, not the adversary. That is the correct
  threat model — the failure this exists to prevent is a contributor adding a
  Gaussian head and copying the `Bounds` field without the `validate()` line.

The allowlist ships with **2 rows and an empty exemption list**. That the
exemption list is empty is a property worth having explicitly: it means "every
`Bounds` field in this crate needs a strictness check" is currently a
crate-level *invariant*, not a per-field judgement, so the guard's question to a
future author is a yes/no with a default, not an essay.

### 3. `rlevo-evolution` and `rlevo-environments` get prose, not a scan

For the other 30 fields the convention lives in `docs/rules.md`'s Config
Validation Contract, in its Error Handling section, and in the `Bounds`
rustdoc: a `Bounds` field is zero-width-permissive by design; if a consumer
needs a strictly positive span, it calls `config::nondegenerate_bounds` in
its own `validate()`, and the reason goes in the field doc.

This is not "docs-only enforcement" in the sense this ADR's own
Alternatives-considered section, below, rejects. The one place in these two
crates where zero width actually bites — the SA and random-restart
`step_size` defaults — already carries ADR 0027's error-surface decision
(Decision 5)'s `debug_assert!`, and those two constructors are not reachable
from a
`Validate` impl at all (`default_for` is a constructor, not a validation
seam), so `nondegenerate_bounds` could not be wired there without changing
their signatures to fallible. The prose covers what is *unenforced*; it is not
being asked to cover what the guard covers.

## Assumptions and the reopen trigger

The crate partition is **a fact today, not a law.** This ADR's central
justification is an enumeration of the workspace as of 2026-07-30, and it is
load-bearing: remove the partition and this ADR's own Decision 2's scope
choice loses its argument entirely.

**Review trigger — any one of these reopens this ADR:**

1. A second real instance (an un-guarded zero-width-sensitive `Bounds`) appearing
   **anywhere**, including inside `rlevo-reinforcement-learning`.
2. The strictness-needing population rising above ~5.
3. That population crossing a crate boundary — i.e. a strictness-needing
   `Bounds` field outside `rlevo-reinforcement-learning`.

Any of those flips this ADR's own Decision 2 to a workspace-wide guard, and (2) or (3) would
also justify revisiting the `NonDegenerateBounds` newtype rejected below: its
cost amortizes differently at 5-vs-30 than at 2-vs-30, because the newtype's
price is paid once per *permissive* adopter (each must answer "which `Bounds`?")
while its benefit scales with the *strict* population.

The concrete plausible shape of (3) is already visible. A future
`rlevo-evolution` operator deriving a step size or a perturbation scale from
its search bounds wants a strictly positive span — `simulated_annealing.rs:79`
and `random_restart.rs:83` **already want exactly that** and settle for a
`debug_assert!` because they sit in an infallible constructor. A config
formalizing that want would need `nondegenerate_bounds` and would fall
**outside** the guard's scan, in a crate the guard does not read. That is not a
hypothetical: it is the existing `debug_assert!` being promoted.

## Consequences

### Positive

- The strictness check acquires a **name**. The migration trap #387 describes
  now has a one-line, greppable answer at the point of use, instead of a
  four-argument `config::distinct` call whose relationship to the `Bounds` field
  beside it is legible only to a reader who knows ADR 0027's
  inclusive-invariant decision (Decision 2).
- **`span() > 0.0` is closed off before anyone writes it.** The infinite-endpoint
  hole is the kind of thing found by a reviewer once and reintroduced by the next
  refactor; putting the reasoning in the helper and in this ADR is the only
  durable form.
- The two RL sites' checks become **checked-not-conventional**. Deleting either
  `nondegenerate_bounds` line now fails a test that names the ADR, rather than
  passing every gate in the repository.
- **The guard's scope is itself recorded as a decision**, following ADR
  0062's guard-mechanization decision (Decision 4)'s precedent that a
  source-text guard's scope is architectural (0062 chose
  crate-wide *against* a `grids/`-only scan, on the same kind of population
  argument reaching the opposite conclusion — the reasoning transfers, the answer
  does not, and the difference is exactly that #104's population was crate-wide
  while #387's is one crate's two fields).
- No migration. Both call sites already exist; both keep their error semantics.

### Negative / accepted costs — do not soften these

- **This invokes a mechanical remedy at recurrence-count zero.** The
  `rng_seeding_guards.rs` precedent was written after the same bug recurred
  *twice* (#104, then #282's nine envs). Here the count is zero: one instance,
  found by review, fixed by ADR 0054 before the issue that names the trap was
  even filed. That is the strongest argument against this ADR's own
  Decision 2, and it is a good one — a guard is a permanent maintenance
  obligation and an allowlist is a file that must be edited by people who
  did not read this ADR.

  The countervailing input, which is what decided it: the project roadmap's near
  horizon is model-based RL / MPC for robotics plus continued CleanRL build-out.
  Both add **continuous-action Gaussian heads** — precisely the shape whose
  `log_std: Bounds` is the entire strict population — into precisely the crate
  the guard scans. Recurrence is *expected*, not hypothetical, and it is expected
  to arrive by copy-paste from one of the two existing heads, which is the
  failure mode a bidirectional allowlist catches and review demonstrably did not
  (`Validate` was implemented on all four head configs and called on none, per
  ADR 0054's Context section). Stated as a forecast, because that is what it
  is: if the roadmap changes, cost 1 stops being paid for and this ADR's own
  Decision 2 should be deleted, which costs nothing.
- **A fourth spelling of "these two numbers differ" now exists in
  `rlevo-core::config`** — `ordered`, `distinct`, `nondegenerate_bounds`, and (in
  spirit) `Bounds`'s own invariant. Mitigated by delegation, not by
  documentation: `nondegenerate_bounds` contains no comparison of its own, so
  there is one predicate with three entry points, not three predicates.
- **The helper's name over-promises slightly** (this ADR's own Decision 1):
  it also rejects $\pm\infty$. A reader who wants "nonzero span, infinity allowed"
  must notice this. The wart is documented at the helper; the alternative —
  a helper that permits an infinite endpoint — is the loosening the helper
  exists to prevent.
- **Enforcement is now asymmetric across the workspace by design**, which is a
  thing a reader must be told rather than infer. `rules.md`'s section 4
  (Error Handling) and this ADR are the only places the asymmetry is
  stated; a contributor who sees the guard in
  `rlevo-reinforcement-learning` and none in `rlevo-evolution` and concludes the
  latter is an oversight will file the ADR-0027-relitigating issue this ADR is
  trying to make unnecessary.
- **The guard is source-text, not semantic.** Brittle to reformatting;
  defeatable by aliasing. Inherited from the precedent, along with the threat
  model that makes it acceptable.

### Neutral

- No new dependency. No `proptest`: the invariant is "this call exists in this
  `validate()`", which is a source property, not an input-space property (ADR
  [0036](0036-adopt-proptest-for-property-tests.md)).
- `Bounds` is untouched — no new method, no invariant change, no serde change.
  `validate()` keeps doing real work on both Gaussian configs regardless (ADR
  0054's Consequences section: ADR 0049's floor and span checks stay).
- `ConstraintKind` is unchanged. `DegenerateInterval` already says the right
  thing for this failure, and it is `#[non_exhaustive]` (ADR 0060) if that ever
  stops being true.

## Alternatives considered

- **A `NonDegenerateBounds` newtype wrapping `Bounds`.** The most attractive
  rejection, because the newtype is the workspace's own blessed answer to
  "an invariant that must survive struct-literal construction" (ADR 0055: privacy
  on the *value*, not the container; transitive through `Deserialize`, `Clone`,
  and literals in a way a `validate()` line is not).

  Rejected on the cost side, and specifically **not** on the ground that it would
  retire `validate()` — it would not. ADR 0054's Consequences section is
  explicit that ADR 0049's absolute floor (`>= -35`) and span (`< 40`) checks
  are not expressible as a `Bounds` invariant and stay in `validate()` either
  way, with the `log_std_min = -120, log_std_max = -100` counterexample given
  there. So the newtype removes **one line** from two `validate()` bodies
  that keep running, while adding a second public range type that all 30
  permissive adopters must now answer "which `Bounds` does my field hold?"
  about — every config author, every doc example, every `From`/`TryFrom`
  seam. That is the inverse of ADR 0054's uniformity decision (Decision 2)'s
  own uniformity reasoning ("one answer, not four"): it makes a
  per-field question out of something that is currently a type-level fact. At
  2-vs-30 it does not earn its keep. At 5-vs-30 the arithmetic changes, which is
  why this is on the reopen trigger rather than closed.

- **`Bounds::try_new_nondegenerate`.** Rejected, and worse than the newtype on
  the newtype's own strongest ground: it returns a plain `Bounds`, so the
  invariant **does not travel with the value** — the field type is still
  `Bounds`, a struct literal still bypasses it, and it is therefore a fourth
  spelling of the same check with none of the newtype's compensating guarantee.
  It also fails on error shape: distinguishing "degenerate" from "inverted" in
  the returned error requires enum-ifying the currently-`pub struct BoundsError
  { lo, hi }` (ADR 0027's error-surface decision, Decision 5, `Copy`, two
  public fields), a breaking change to a public type in service of a check
  that is **not `Bounds`'s business** — ADR 0027's inclusive-invariant
  decision (Decision 2) decided that `Bounds` is the closed-clamp-range type
  and `config::*` owns the strict scalar invariants.

- **Docs-only enforcement for the RL crate too** (state the convention, add no
  guard). Rejected on this workspace's own history rather than on principle: ADR
  0054's Alternatives-considered section already rejected exactly this shape
  for exactly these configs, citing ADR 0026's "exactly the drift that
  produced 87 un-validated configs" — and #386 *was* that drift arriving,
  with `Validate` implemented on four head configs and called on none. Note
  that this ADR nonetheless chooses prose for the other 30 fields (this
  ADR's own Decision 3): the distinction is that there, prose is describing
  a *permission*, and a permission does not drift into a defect.

- **A workspace-wide guard, same shape, same predicate.** Rejected on this
  ADR's own Context section's partition: ~32 allowlist rows, 30 of which
  restate ADR 0027's inclusive-invariant decision (Decision 2), each of which
  must be maintained against reformatting and field renames, to protect 2 fields
  that live in one crate. The failure mode is specific and worth naming — a guard
  whose rows are 94 % "this one is fine" trains its readers to add a row without
  thinking, which is the state in which it stops catching anything. Revisit under
  the reopen trigger.

- **Changing `Bounds`'s inclusive invariant to strict `lo < hi`.** Out of scope
  and explicitly not re-litigated. ADR 0027's inclusive-invariant decision
  (Decision 2) and its Alternatives-considered section decided this with
  reasons (`clamp` is well-defined on a point; every sampler is
  zero-width-safe), and 30 of 32 fields depend on the answer. This ADR supersedes
  nothing.

- **Add the check to `Validate` for every `Bounds` field and exempt the
  permissive ones.** The mirror image of the status quo, and it inverts the
  default in the wrong direction: it would make the *documented, intended,
  30-field* behaviour the thing requiring justification. Rejected for the same
  reason ADR 0027's inclusive-invariant decision (Decision 2) rejected strict
  `<` on the type.

## References

- Issue **#387** — the migration trap this ADR records. Accurate about the
  mechanism, with **no live defect**: its one genuine instance was already closed
  by ADR 0054's Bounds-field decision (Decision 3), and the audit it notes
  was not run has been run for this ADR (this ADR's own Context section).
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — its inclusive-invariant
  decision (Decision 2) (the deliberate inclusive invariant and the
  documented divergence from `config::ordered`), its error-surface decision
  (Decision 5) (the `debug_assert!` mitigation that landed), its
  migration-scope decision (Decision 6) (migration scope). Refined, not
  superseded.
- ADR [0054](0054-policy-head-construction-is-fallible.md) — its Bounds-field
  decision (Decision 3) (the two `config::distinct` lines this ADR renames),
  its uniformity decision (Decision 2) (the uniformity reasoning the newtype
  alternative would invert), its Consequences section (why `Bounds` does not
  subsume `validate()`, and the `-120/-100` counterexample), its
  Alternatives-considered section (the docs-only rejection reused here).
- ADR [0049](0049-ppo-gaussian-log-std-is-bounded.md) — the floor (`>= -35`) and
  span (`< 40`) checks that stay in `validate()` and that the newtype alternative
  would not have retired.
- ADR [0060](0060-config-values-must-be-finite.md) — the finiteness guard on
  `ordered`/`distinct` that `nondegenerate_bounds` inherits, and the
  value-versus-schema distinction that makes rejecting $\pm\infty$ correct here while
  `Bounds::new(0.7, f32::INFINITY)` stays legitimate elsewhere.
- ADR [0026](0026-shared-config-validation-convention.md), ADR
  [0055](0055-config-invariant-enforcement-allocation.md) — the `Validate`
  chokepoint convention, and the "privacy on the value, not the container" rule
  the newtype alternative appeals to.
- ADR [0062](0062-grid-layout-fidelity-and-no-dead-rng.md) — its
  guard-mechanization decision (Decision 4), the precedent that a source-text
  guard's **scope** is an ADR-recorded decision, and the source of the
  bidirectional-allowlist / stated-limits shape adopted here.
- ADR [0036](0036-adopt-proptest-for-property-tests.md) — why this invariant is
  not a proptest.
- `docs/rules.md`'s Config Validation Contract, in its Error Handling section
  ("Config Validation Contract (ADR 0026, 0055, 0060)") — gains this ADR's
  own Decision 3's convention.
- Code — the helper: `crates/rlevo-core/src/config.rs`
  (`config::nondegenerate_bounds`, `config::distinct`, `config::ordered`),
  `crates/rlevo-core/src/bounds.rs` (`Bounds::try_new`, the `lo <= hi`
  invariant).
- Code — the strict population (2):
  `crates/rlevo-reinforcement-learning/src/algorithms/ppo/policies/gaussian.rs:284`
  and `crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_policy.rs:123`
  (the lines from ADR 0054's Bounds-field decision, Decision 3, with the
  in-file comments explaining PPO's frozen shared `Param` and SAC's
  state-independent $\sigma$).
- Code — the permissive population's live mitigation:
  `crates/rlevo-evolution/src/local_search/simulated_annealing.rs:79`,
  `crates/rlevo-evolution/src/local_search/random_restart.rs:83` (ADR 0027's
  error-surface decision, Decision 5's, `debug_assert!`, and the plausible
  shape of this ADR's reopen trigger).
- Tests: `crates/rlevo-environments/tests/rng_seeding_guards.rs` (the guard shape
  this ADR's own Decision 2 copies — bidirectional allowlist,
  `#[cfg(test)]`-region skipping, stated limits),
  `crates/rlevo-environments/tests/landscape_dim_guards.rs` (its own
  ancestor).
- Provenance: the roadmap forecast in this ADR's own Consequences section
  (model-based RL / MPC for robotics; continued CleanRL build-out) comes
  from the maintainer's working notes, which are **gitignored**, so the
  forecast is reproduced here as a stated assumption rather than deferred to
  an unreachable file.
