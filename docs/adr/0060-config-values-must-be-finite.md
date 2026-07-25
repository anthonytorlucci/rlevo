---
project: rlevo
status: active
type: decision
date: 2026-07-25
tags: [adr, decision, config, validation, finiteness, nan, infinity, non-exhaustive, issue-353]
---

# ADR 0060: Config values must be finite; bounds may be infinite

## Status

**Accepted (2026-07-25).** Resolves issue #353 (`config::positive` accepts
`f64::INFINITY` as a positive value). **Extends** ADR 0026 — it does not
supersede it. The `Validate` trait, the `ConfigError` shape, and the
construction-chokepoint rule all survive unchanged; what 0026 left unstated is
the *domain* the float predicates check over, and that is what this ADR fixes.
Complements ADR 0027 (`Bounds`) and ADR 0031 (`NonNegativeRate`), whose
newtypes already enforced finiteness locally, at a lower layer.
`docs/rules.md §4` reconciled on acceptance (one bullet, and the section
heading's ADR list).

**Chosen shape** — one distinction, applied to the shared predicates:

> A config **value** must be finite. A config **bound** may be `±∞`.

`positive` and `in_range` reject a non-finite `got`; `ordered` and `distinct`
require both arguments finite. A new `ConstraintKind::NotFinite { got: f64 }`
is checked **first**, so it wins over `NotPositive` / `OutOfRange` /
`NotOrdered` / `DegenerateInterval`. `ConstraintKind` becomes
`#[non_exhaustive]`. No helper signature changes and no call site migrates.

## Context

### The hole, and its measured width

`config::positive(f64::INFINITY)` returned `Ok`. Verified by execution, not by
reading: the predicate is `got > 0.0`, and `f64::INFINITY > 0.0` is `true`.
**105** `config::positive` call sites were affected — learning rates, physics
constants (`cartpole.rs:225` `masscart`, `:228` `force_mag`, `:229` `tau`;
`pendulum.rs:162` and `acrobot.rs:357` `dt`), and evolution parameters.

The consequence is not theoretical, because the value does not get sanitized on
the way down. `f64::INFINITY as f32` is `f32::INFINITY`, **not** `NaN`. So an
infinite `alpha_lr` (`sac_config.rs:98`, guarded by `config::positive`) reaches
SAC's `LogAlpha::adam_step`, where at `grad == 0` the step is `inf · 0 = NaN`,
and the `[−88, 88]` clamp (`sac_alpha.rs:164-168`) **propagates** rather than
rescues — `NaN.clamp(..)` is `NaN`, as `sac_alpha.rs:246-247` already documents.
Issue #184 patched that one call site defensively. The other 104 had no guard.

### The bound case is the opposite, and it is legitimate

A further **37** call sites (40 textual occurrences of `in_range(.., INFINITY,
..)` across `crates/`, less the three inside `config.rs`'s own definition and
rustdoc) spell `in_range(C, f, lo, f64::INFINITY, got)`, where `hi = ∞`
correctly means *unbounded above* — `cmsa_es.rs:164`, `ppo_config.rs:162,169`,
`td3_config.rs:100,107,114`, the locomotion configs, and so on. The same
comparison chain that made `hi = ∞` work also accepted `got = ∞`: `∞ >= lo &&
∞ <= ∞` is `true`. One predicate, two roles, one domain — that is the defect.

`Bounds` (ADR 0027) makes the legitimacy of an infinite *bound* concrete rather
than hypothetical: `locomotion/common.rs:312` constructs
`Bounds::new(0.7, f32::INFINITY)` as a healthy `z_range`, and `bounds.rs:202-203`
tests both one-sided infinite ranges deliberately. "Reject infinity everywhere"
would have been wrong.

### The workspace already had the cure, twice, at the wrong layers

- **`NonNegativeRate`** (ADR 0031) enforces `is_finite() && r >= 0.0`
  (`rate.rs:82,95`) — the **newtype** layer, covering only fields that adopted it.
- **`QrDqnTrainingConfig`** hand-rolls `if !(0.5 * self.kappa * self.kappa).is_finite()`
  (`qrdqn_config.rs:262`) — the **call-site** layer, covering one field.

Two independent local cures for the same problem is evidence of a missing
shared-layer rule, not of two unrelated fixes. The predicate layer sits between
them and had no such rule.

### `ConstraintKind` was never semantically closed

Its rustdoc calls it "the closed set of configuration-invariant violations", but
`Custom(&'static str)` is an escape hatch that any config can reach for — and
does (`gaussian.rs:882,926` match on it). A variant set with a `Custom` arm was
never closed in practice; declaring `#[non_exhaustive]` records what was already
true.

### The `Deserialize` vector makes this live, not dormant

`docs/rules.md §4` and ADR 0055 §Consequences record that **no config loader
exists in the workspace yet** and that ~22 configs derive `Deserialize` plainly.
A non-finite float is exactly what a JSON/TOML/bincode decoder smuggles past
validation once the first loader is written — `inf` is a representable IEEE-754
value in every one of those encodings. Closing the predicate now means the
loader author inherits the check rather than having to rediscover it.

## Decision

### 1. Values are finite; bounds may not be

A config **value** (`got`) must be finite. A config **bound** (`lo` / `hi`) may
be `±∞`. This is the whole rule; everything below is its mechanical application.

The asymmetry is not a special case — it is a type distinction the signature
does not express. `lo`/`hi` are the *schema*, supplied by the config author as
source literals; `got` is the *data*, and a field is never legitimately `NaN` or
`±∞`.

### 2. The predicates enforce it

`positive` and `in_range` reject a non-finite `got`. `ordered` and `distinct`
require **both** arguments finite — both of their arguments are values. `lo` and
`hi` are deliberately **not** checked.

### 3. `ConstraintKind::NotFinite { got: f64 }`, checked first

The finiteness check runs before the range / sign / ordering comparison, so
`NotFinite` wins. A non-finite value is a wrong-kind-of-number problem, and
naming it as one is a better diagnosis than reporting the downstream comparison
it also happens to fail.

### 4. `ConstraintKind` becomes `#[non_exhaustive]`

Downstream `match` carries a trailing `_` (or `other =>`) arm. This is the
`#[non_exhaustive]`-is-for-enums use ADR 0055 §5 blessed; it is not a struct
adoption.

### 5. No `finite` / `finite_nonneg` / `finite_min` helpers

`in_range(C, f, 0.0, f64::INFINITY, x)` remains the blessed spelling of
"non-negative, unbounded above". A second spelling would mean 37 call sites each
*choosing* between two helpers, and each choice is a chance to choose wrong. One
spelling that is correct by construction beats two that are correct by
discipline.

### 6. An infinite *bound* belongs in `Bounds`, never in `ordered`

`ordered(C, f, low, high)` can no longer express a half-open interval, because
both of its arguments are values. That is the intended outcome, not a casualty:
a half-open *range* is a `Bounds` (ADR 0027), which was built for exactly the
one-sided infinite case and tests it.

### One deliberate deviation, recorded so it is not rediscovered

Two of the 37 `in_range(.., ∞, ..)` parameters admit `+∞` as a *numerically*
well-defined sentinel. **We reject them anyway.**

- **TD3 `noise_clip` (`c`)** — `clip(x, −∞, ∞) ≡ x`, so `c = ∞` denotes
  "unclipped target-policy smoothing" (`td3_config.rs:43`, validated at
  `:114-119`).
- **CMSA-ES `tau_c`** — the covariance blend is `1 / τ_c` (`cmsa_es.rs:475`), so
  `τ_c = ∞` gives blend `0`, freezing the covariance update
  (`cmsa_es.rs:107`, validated at `:164`).

Rejected for three independent reasons, any one of which suffices:

1. **Neither is the canonical formulation.** Fujimoto et al. (2018) fix
   `c = 0.5`; Beyer & Sendhoff (2008) *derive*
   `τ_c = 1 + N(N+1)/(2μ)` — always finite, and the in-tree default computes
   exactly that (`cmsa_es.rs:136`).
2. **Both fields are `f32`.** `f32::MAX` expresses "effectively unclipped" and
   "effectively frozen" exactly, without putting an infinite value in a config.
3. **A predicate whose strictness depends on which field it guards is the
   looseness this ADR removes.** Per-field exemptions reintroduce, one level
   down, the "two roles, one domain" defect described in §Context.

If either sentinel is ever wanted back, this paragraph is the record of what
would have to change — a `Bounds`-typed field or an explicit `Option` sentinel,
not a relaxed predicate.

## Consequences

### Positive

- **One predicate edit closes 142 call sites with zero migration** (105
  `config::positive` + 37 `in_range(.., ∞, ..)`), and makes every *future*
  `in_range(.., ∞, ..)` correct for free. That matters more than it sounds: the
  issue's own call-site count drifted **103 → 105** while it sat open, so any
  remedy that scales with the call-site count is a remedy that is already stale
  when it lands.
- **Error text becomes actionable.** `NotPositive { got: inf }` renders as
  `value inf must be strictly positive` — a message that argues against itself
  and offers the user no repair. `NotFinite { got: inf }` renders as
  `value inf must be finite`, which names the actual fault.
- **The `Deserialize` obligation shrinks.** The first config loader (ADR 0055
  §Consequences, cost 3) inherits finiteness rejection rather than owning it.

### Negative / accepted costs

- **Behaviour-breaking without a source change.** A downstream `Validate` impl
  inherits stricter behaviour by recompiling — no signature moves, so nothing
  flags the change at the call site. Accepted: the strictness is in the
  rejecting direction only, and the workspace is alpha with no external
  consumers.
- **Kind assertions change for inputs that were *already* rejected.** `NaN` and
  `−∞` move from `NotPositive` / `OutOfRange` to `NotFinite`. Any test asserting
  the old kind must be updated — the *accept/reject* outcome is unchanged for
  these inputs; only the diagnosis moves. (`config.rs`'s own
  `in_range_rejects_nan_as_out_of_range` is precisely such a test, and it
  documents in its doc comment why it pins the kind rather than `is_err()`.)
- **`ordered` loses half-open intervals.** §6. No in-tree caller wanted one.

### Neutral

- All helper signatures unchanged. No call-site migration. No serialized-schema
  change (no config in the workspace is persisted — ADR 0058 §Context,
  correction 1).

## Alternatives Considered

### Per-site `finite_*` helpers, migrating the 37 `in_range(.., ∞, ..)` sites

Add `finite`, `finite_nonneg`, `finite_min` and rewrite the unbounded-above
sites to use them. Rejected on three counts: it creates **two spellings of one
check**, so every future author picks; correctness becomes **contingent on 37
migrations all landing**, rather than on one predicate being right; and the hole
stays open in whichever config is forgotten — which is the failure mode the
issue documents, reproduced with extra steps.

### Reuse `NotPositive` / `OutOfRange` for infinity

Free — no new variant, no `#[non_exhaustive]`, no test-kind churn. Rejected
because it leaves a **misleading message on the exact input class the rule exists
to reject**: "value inf must be strictly positive" tells a user that `inf` is not
positive, which is false, and gives them no repair. The variant is the cheap part;
the diagnosis is the product.

### Defer `#[non_exhaustive]` to a follow-up

Rejected. Adding it **now** is free — all five in-workspace `match` sites on
`ConstraintKind` (`config.rs` ×2, `qrdqn_config.rs:563`, `gaussian.rs:882,926`)
already carry a wildcard `other =>` arm, so nothing breaks. Adding it **later**
is a breaking change to every downstream `match`. Deferring converts a two-way
door into a scheduled one-way door, for zero benefit today.

## References

- Issue #353 — "`config::positive` accepts `inf` as a positive value (103 call
  sites affected)"; the true figure at acceptance is 105, plus 37
  `in_range(.., ∞, ..)` sites the issue does not name.
- Issue #184 — the SAC-α optimizer guard; the one call site that was already
  defended against the `inf · 0 = NaN` step this ADR closes at the source.
- ADR [0026](0026-shared-config-validation-convention.md) — `Validate` /
  `ConfigError` / `ConstraintKind`. **Extended, not superseded**: the trait, the
  error shape, and the construction-chokepoint rule are unchanged.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — `Bounds`; the typed home
  for a legitimately one-sided infinite *range*, and the reason "reject infinity
  everywhere" is wrong.
- ADR [0031](0031-probability-rate-newtypes.md) — `NonNegativeRate`'s
  `is_finite() && r >= 0.0`; the same rule, already enforced one layer down, for
  the subset of fields that adopted the newtype.
- ADR [0055](0055-config-invariant-enforcement-allocation.md) — §5
  (`#[non_exhaustive]` is for enums, which §4 above obeys) and §Consequences
  cost 3 (the dormant, unowned `Deserialize` obligation this narrows).
- ADR [0058](0058-target-update-type-unifies-cadence-and-tau.md) — the adjacent
  precedent for *narrowing by construction* (`PolyakTau` excluding `τ = 0`) and
  for deliberately **not** narrowing `config::in_range` for one field's sake;
  this ADR narrows the shared helper only for the value/bound distinction, which
  is universal.
- Fujimoto, S., van Hoof, H., Meger, D. (2018). *Addressing Function
  Approximation Error in Actor-Critic Methods.* arXiv:1802.09477. — TD3's
  `c = 0.5`; the canonical, finite `noise_clip`.
- Beyer, H.-G., Sendhoff, B. (2008). *Covariance Matrix Adaptation Revisited —
  The CMSA Evolution Strategy.* PPSN X. — derives
  `τ_c = 1 + N(N+1)/(2μ)`, always finite.
- Code: `crates/rlevo-core/src/config.rs` (the predicates and
  `ConstraintKind`); `crates/rlevo-core/src/rate.rs:82,95` and
  `crates/rlevo-core/src/bounds.rs:202-203` (the two pre-existing local cures /
  the legitimate infinite bound);
  `crates/rlevo-reinforcement-learning/src/algorithms/qrdqn/qrdqn_config.rs:262`
  (the hand-rolled call-site guard);
  `crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_config.rs:98` and
  `sac/sac_alpha.rs:164-168,246-247` (the `inf` → `NaN` path);
  `crates/rlevo-environments/src/locomotion/common.rs:312` (`Bounds::new(0.7,
  f32::INFINITY)`); `crates/rlevo-environments/src/classic/cartpole.rs:225-229`
  (physics constants guarded by `config::positive`).
