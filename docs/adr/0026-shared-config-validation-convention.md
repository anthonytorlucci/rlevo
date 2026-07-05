---
project: rlevo
status: active
type: decision
date: 2026-07-04
tags: [adr, decision, config, validation, error-handling, rlevo-core, conventions]
---

# ADR 0026: Shared `Validate` config-validation convention

## Status

**Accepted (2026-07-04).** `docs/rules.md §4` reconciled on acceptance (line
142 narrowed, "Config Validation Contract" subsection added, setter-guard
panic rows retained). Filed to unblock the per-crate config-validation
issues #102/#103 (environments), #137/#138/#139 (evolution), and
#164/#165/#166 (reinforcement-learning), all of which independently flagged
"the config/builder should validate before use" as the single most repeated
gap across three code reviews (issue #193).

**Chosen shape:** a standalone `pub trait Validate { fn validate(&self) ->
Result<(), ConfigError>; }` plus a structured `ConfigError` in a new
`rlevo_core::config` module — **not** a macro, **not** a bare `docs/rules.md`
convention. Fallibility is `Result`, not panic. Adoption is incremental; this
ADR lands only the trait, the error type, one reference adopter, and the
`docs/rules.md §4` reconciliation.

## Context

The workspace holds **87 `pub struct *Config`** types (plus hyperparameter-
bearing builders) across `rlevo-environments`, `rlevo-evolution`, and
`rlevo-reinforcement-learning`. Almost none validate their contents. Invalid
hyperparameters are accepted at construction and only surface as a confusing
panic deep inside training / physics / genetic-operator code — far from the
line that supplied the bad value. Concrete live examples:

- `C51Config { v_min, v_max }` (`c51/c51_config.rs:69`) with `v_min == v_max`
  → degenerate support atoms, divide-by-zero deep in the categorical
  projection.
- `GaConfig::pop_size` / `CmaEsConfig::pop_size` (`ga.rs:68`, `cma_es.rs:75`)
  `== 0` → empty-population panic inside a selection operator.
- `EpConfig::tau` (`ep.rs:48`) or a soft-update `tau` outside `[0, 1]` →
  silently divergent self-adaptation.
- Locomotion `action_clip: (f32, f32)` (`swimmer/config.rs:40` and siblings)
  with `low >= high` → NaN torques inside Rapier.

There is exactly **one** existing precedent, and it diverges from what we want:
`CooperativeCoevolutionParams::validate(&self)`
(`coevolution/cooperative.rs:119`) returns `()` and **panics** via `assert!`.
Left undecided, issue #193 correctly predicts 8+ more divergent local
reimplementations.

### The rule contradiction this ADR must resolve

`docs/rules.md §4` already legislates two clauses that **conflict** for config
validation:

- Line 142 — *"Panics are permitted only for programming errors (index out of
  bounds, dimension mismatch, **invalid builder config**)."*
- Line 143 — *"Never panic in response to user-supplied runtime data; return
  `Err(...)` instead."*

The **Documented Panic Contracts** table further blesses `with_capacity(n)`
panicking on `n == 0` and `with_alpha(x)` panicking on `x ∉ [0, 1]`.

A hyperparameter config is *both* things at once, so the two clauses point in
opposite directions. The tie-breaker is already in the codebase and is not a
matter of taste: **many configs derive `Deserialize`** (`frozen_lake.rs`,
`cartpole.rs`, `pixel_grid.rs`, `blackjack.rs`, `pendulum.rs`, …). A config
loaded from a file or a run manifest **is** user-supplied runtime data, so
clause 143 already governs it — such a config must be *rejected*, never
paniced through. That observation resolves the contradiction without inventing
a new policy: clause 142's "invalid builder config" can only mean the
*localized* single-setter guards, not the *assembled whole config*.

### Relationship to existing seams

- `State`/`Observation` already expose `is_valid(&self) -> bool`
  (`base.rs:78`) — but that is **domain-state legality** with a deliberately
  boolean answer, checked every step. Config validation is a different
  concern: it runs once at construction and must *name the offending field* so
  the user can fix a hyperparameter, which a bare `bool` cannot do. We keep the
  two distinct rather than overloading `is_valid`.
- This mirrors ADR 0011 (`ConstructableEnv`): a cross-cutting capability lives
  in its own standalone trait in `rlevo-core`, not welded onto the type it
  serves. It also mirrors ADR 0023 (`ObjectiveSense`): a small typed primitive
  in a dedicated `rlevo-core` module that many crates reference but none
  duplicate.

## Decision

### 1. The `Validate` trait + `ConfigError` (new `rlevo_core::config` module)

```rust
//! Configuration validation: the [`Validate`] trait and [`ConfigError`].

/// A configuration or hyperparameter-bearing type that can check its own
/// invariants before it is used to construct anything.
///
/// Implement this on every public `*Config` (and any builder that carries
/// tunable hyperparameters). `validate` is the seam a construction chokepoint
/// calls before the config reaches an algorithm, environment, or operator, so
/// an invalid value is rejected with a field-named error instead of surfacing
/// as a panic deep inside training / physics / operator code.
pub trait Validate {
    /// Returns `Ok(())` if every invariant holds, or the **first** violation.
    ///
    /// Fail-fast (one error) is deliberate — see the ADR. Implementations
    /// check cheap, purely-local invariants; they must not run the algorithm.
    fn validate(&self) -> Result<(), ConfigError>;
}

/// A single violated configuration invariant, naming the config, the field,
/// and the constraint kind.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigError {
    /// The config type that failed, e.g. `"C51Config"`.
    pub config: &'static str,
    /// The offending field, e.g. `"v_max"`.
    pub field: &'static str,
    /// The specific invariant that was violated.
    pub kind: ConstraintKind,
}

/// The closed set of config-invariant violations. Structured per rules.md §4
/// (no stringly errors); the `Custom` tail carries a `&'static str`, never an
/// owned `String`, so `ConfigError` allocates nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    /// Value must lie in the closed interval `[lo, hi]`.
    OutOfRange { lo: f64, hi: f64, got: f64 },
    /// Value must be strictly positive (`> 0`).
    NotPositive { got: f64 },
    /// A `(low, high)` pair must satisfy `low < high`.
    NotOrdered { low: f64, high: f64 },
    /// Two values required to differ are equal (e.g. `v_min == v_max`).
    DegenerateInterval { value: f64 },
    /// A count / size / capacity must be non-zero.
    Zero,
    /// A one-off invariant with a static explanation.
    Custom(&'static str),
}
```

`ConfigError` implements `std::error::Error` + `Display` (per §4). `Display`
reads e.g. `C51Config.v_max: value must differ from v_min (both 0)`.

`ConstraintKind` covers every example the review surfaced (`lo > hi`,
`pop_size == 0`, `tau ∉ [0, 1]`, `v_min == v_max`) with `Custom(&'static str)`
for the tail — no `String`, so the type stays cheap and `PartialEq`, and
validation on the `Default` happy path costs nothing.

### 2. Fallibility is `Result`, and where it is enforced

- **Config-consuming construction is fallible.** Any constructor or builder
  `.build()` that receives a *caller-supplied* config calls
  `config.validate()?` and returns `Result<T, ConfigError>` (new code:
  `try_with_config` / a `.build()` that returns `Result`). This is the seam
  that converts "confusing deep panic" into "clear rejection at construction."
- **`Default` construction is exempt but obligated.** Library defaults must
  themselves be valid: every `impl Validate` ships a unit test
  `assert!(Config::default().validate().is_ok())`. This lets the infallible
  `ConstructableEnv::new` / `Default` paths (ADR 0011) stay infallible without
  a `Result`, because a valid default cannot fail.
- **Deserialized configs must be validated by the caller.** Because many
  configs derive `Deserialize`, a loaded config is user-supplied runtime data;
  the loader calls `validate()` and propagates `Err` — it must never panic
  (rules.md §4 line 143).

### 3. Boundary with builder-setter panics (the rule resolution)

We draw one precise line, and codify it in `docs/rules.md §4`:

- **Setter guard → panic (kept).** A single `with_x(v)` builder method whose
  only job is to reject an out-of-domain `v` at the exact call site may panic.
  The panic points at the offending literal; the argument is right there. The
  existing `with_capacity(0)` / `with_alpha(x ∉ [0,1])` rows in the panic table
  stay valid — they are localized setter guards, an *additive* fail-fast
  convenience.
- **Assembled config → `Result` (new).** Validating a config *as a whole* —
  cross-field invariants (`v_min < v_max`, `low < high`), or fields set via
  struct-literal / `Default`-update / **`Deserialize`** where there is no
  guarded setter call site — goes through `validate()` and returns
  `ConfigError`.

The deciding test: **if an invalid value can arrive via `Deserialize` or
struct-update syntax without passing a guarded setter, it MUST be caught by
`validate()` and returned as `Err`** — never paniced — because rules.md §4
line 143 already forbids panicking on such data. Setter guards do not replace
`validate()`; they sit on top of it.

### 4. Reconciliation with `docs/rules.md §4` (applied on acceptance)

On acceptance, `docs/rules.md §4` is reconciled (the same
reconcile-on-acceptance flow as commit "Reconcile rules.md with accepted ADRs
0004–0025"; rules.md is not itself immutable, ADRs supersede it):

- **Reword line 142** — remove "invalid builder config" from the *panic*
  allow-list and narrow it to the localized case:
  > *Panics are permitted only for programming errors (index out of bounds,
  > dimension mismatch, or an out-of-domain argument to a single documented
  > builder setter — see the Panic Contracts table).*
- **Add a "Config Validation Contract" subsection** stating: every public
  `*Config` (and hyperparameter-bearing builder) implements
  `rlevo_core::config::Validate`; construction that consumes a caller-supplied
  or deserialized config returns `Result<_, ConfigError>` and must not panic;
  `Default`s must pass their own `validate()` (unit-tested).
- **Keep** the `with_capacity` / `with_alpha` rows in the Panic Contracts
  table — they are the blessed setter-guard exception, now cross-referenced
  from the new subsection.

### 5. First adopters

- Fold `CooperativeCoevolutionParams::validate()` into `Validate` (return
  `Result`, delete the `assert!`s) — removes the one divergent precedent.
- One reference environment config from #102 (a locomotion box2d config — the
  richest cross-field invariants, `action_clip` ordering + positive masses) as
  the worked example the downstream issues copy.

The remaining ~85 configs are **out of scope here** and land in the per-crate
issues, each routing through `validate()`.

## Consequences

### Positive
- One decision replaces 8+ divergent local reimplementations; invalid
  hyperparameters are rejected at construction with a field-named error
  instead of a deep panic.
- Resolves the standing `rules.md §4` self-contradiction on the codebase's own
  terms (configs are `Deserialize`-able ⇒ user data ⇒ `Result`).
- `ConfigError` is allocation-free and `PartialEq`, so validation is trivially
  unit-testable and cheap on the `Default` path.
- Trait-shaped, so a construction chokepoint can bound generically on
  `C: Validate`; consistent with `ConstructableEnv` (0011) and `ObjectiveSense`
  (0023).

### Negative / costs
- **Adoption is the bulk of the work and is deferred.** This ADR is a small
  trait; wiring ~85 configs through it is the downstream issues' cost.
- Config-consuming constructors that adopt the `Result` return are a
  signature change at those call sites (per-crate, incremental — not a
  workspace-wide flag day like ADR 0011).
- Two validity concepts now coexist (`is_valid -> bool` for domain state,
  `Validate -> Result` for config). Mitigated by the naming split and doc
  cross-references; overloading `is_valid` was rejected (below).

### Neutral
- Purely additive to `rlevo-core` (new module; no change to existing public
  types). No schema change.

## Alternatives Considered

- **Bare `docs/rules.md` convention, no trait.** Cheapest, but gives no
  generic seam (`where C: Validate`), no shared `ConfigError`, and no
  compiler-visible obligation — exactly the drift that produced 87
  un-validated configs. Rejected.
- **A `#[derive(Validate)]` proc-macro.** Attractive for boilerplate but
  premature: the invariants are heterogeneous (range, ordering, degeneracy,
  cross-field) and a macro DSL for all of them is a project of its own. Revisit
  only if the hand-written impls prove repetitive after 2–3 crates adopt.
- **Keep panicking (status quo + the `cooperative.rs` precedent).** Contradicts
  §4 line 143 for the many `Deserialize`-able configs and keeps the
  deep-in-the-algorithm panic that the reviews flagged. Rejected.
- **Overload `State::is_valid` semantics / return `bool`.** A `bool` cannot
  name the offending field, which is the entire point for a user tuning
  hyperparameters; and `is_valid` is a per-step domain-state check with
  different cadence and meaning. Rejected.
- **Accumulate all violations (`Vec<ConfigError>`).** More helpful when several
  hyperparameters are wrong, but complicates every impl and the trait return.
  Deferred: fail-first now; a `validate_all()` can be added additively if
  demand appears.
- **Infallible constructors that `.expect(config.validate())`.** Keeps
  signatures unchanged but reintroduces panic-on-deserialized-data — the exact
  §4 violation this ADR exists to remove. Rejected as the *long-term* shape
  (acceptable only as transitional scaffolding a downstream issue must retire).

## References
- Issue #193 — parent cross-crate issue; unblocks #102, #103, #137, #138,
  #139, #164, #165, #166.
- ADR [0011](0011-lift-construction-off-environment-trait.md) — standalone
  factory trait in `rlevo-core`; the construction chokepoint `Validate` hooks
  into.
- ADR [0023](0023-objective-sense-and-maximize-convention.md) — small typed
  primitive in a dedicated `rlevo-core` module; the shape this ADR follows.
- `docs/rules.md §4` — Error Handling; the reconciled target.
- Code: `crates/rlevo-core/src/base.rs:78` (`is_valid` — the distinct
  concept), `crates/rlevo-evolution/src/coevolution/cooperative.rs:119` (the
  panicking precedent to fold in), `crates/rlevo-reinforcement-learning/src/algorithms/c51/c51_config.rs:69`
  and `crates/rlevo-environments/src/locomotion/swimmer/config.rs:40` (worked
  invariant examples).
