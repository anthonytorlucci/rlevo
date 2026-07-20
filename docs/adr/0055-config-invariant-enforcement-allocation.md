---
project: rlevo
status: active
type: decision
date: 2026-07-20
tags: [adr, decision, config, validation, encapsulation, newtype, non-exhaustive, conventions]
---

# ADR 0055: Config invariant enforcement is allocated, not hidden

## Status

**Accepted (2026-07-20).** Resolves issue #326 ("workspace: all 32 config
structs have `pub` fields, so struct-literal construction bypasses
`validate()`") by **refuting its premise**, not by adopting its remedies.
`docs/rules.md §2` reconciled on acceptance (the "Struct Field Encapsulation"
subsection now states the `*Config` exemption plainly instead of leaving it
inferable from an absence, and adds the `#[non_exhaustive]`-is-for-enums rule).
Extends ADR 0026 (`Validate`/`ConfigError`); depends on ADR 0027 (`Bounds`) and
ADR 0031 (`Probability`/`NonNegativeRate`) for the layer that actually closes
struct-literal holes. Edits neither.

**Chosen shape** — a four-way allocation of *which mechanism enforces which
invariant*:

1. **`*Config` types keep `pub` fields.** They are hyperparameter data bags;
   `Config { lr: 3e-4, ..Default::default() }` is the idiom a research library
   exists to serve.
2. **Validation is the consumer's obligation at construction** (ADR 0026 §2) —
   `config.validate()?` as the first statement of whatever consumes the config
   by value.
3. **`*State` / `*Params` / `*Genome` encapsulate** (`rules.md §2`) — these are
   *structural* types whose invariants (lengths agree, derived tails
   consistent) span fields and outlive construction.
4. **Any invariant that must survive struct-literal construction is encoded in
   a validated newtype** (ADR 0027 / 0031), not enforced by hiding the field.

`#[non_exhaustive]` is reserved for enums.

None of this is new. All four clauses were already the operative practice; this
ADR only writes the allocation down in one place.

## Context

### The rule was real but unreconstructable

Issue #326 is filed by a reviewer who read the code carefully and could not
recover the rule from it. That is the evidence that it needed writing down. The
allocation was distributed across four documents, none of which states it:

- `rules.md §2` "Struct Field Encapsulation" names `*State` / `*Params` /
  `*Genome` and is silent on `*Config`. The exemption is inferable **only by
  noticing what is absent from a list** — the weakest form a rule can take.
- `rules.md §4` "Config Validation Contract" states the consumer obligation but
  never says *why* that is sufficient, i.e. never says the fields stay `pub`.
- ADR 0026 §3 names the struct-literal case explicitly and routes it to
  `validate()`; a reader who has not read 0026 sees only `pub` fields.
- ADR 0027 / 0031 introduce the newtype layer that closes the residual holes,
  but they are framed as *removing `validate()` lines*, not as *the answer to
  struct-literal bypass*.

Four correct documents that do not compose into a legible rule are, from the
outside, indistinguishable from an oversight.

### What #326 observed, and what it inferred

The observation is accurate. The workspace holds **71** `pub struct *Config`
types under `crates/*/src/` — `rlevo-environments` 34, `rlevo-evolution` 20,
`rlevo-reinforcement-learning` 13, `rlevo-benchmarks` 4 — and every one of them
has 100% `pub` fields. (ADR 0026 §Context says 87; the count today is 71. The
drift is consolidation and crate moves since 2026-07-04. 0026 is immutable and
is **not** edited; the current number is recorded here.)

The inference — that this constitutes a *bypass* — is where it goes wrong. ADR
0026 §3 already names this exact case as **in scope and handled**:

> **Assembled config → `Result` (new).** Validating a config *as a whole* —
> cross-field invariants (`v_min < v_max`, `low < high`), or fields set via
> struct-literal / `Default`-update / **`Deserialize`** where there is no
> guarded setter call site — goes through `validate()` and returns
> `ConfigError`.

and its deciding test:

> **if an invalid value can arrive via `Deserialize` or struct-update syntax
> without passing a guarded setter, it MUST be caught by `validate()` and
> returned as `Err`** — never paniced.

Struct-literal construction is not a hole in the design; it is the case the
design was written for. The `pub` fields are the *premise* of ADR 0026, not a
violation of it.

### The chokepoints ADR 0026 promised are built

0026 deferred adoption to per-crate issues. Those issues have largely landed,
and the resulting coverage is what makes consumption-validation load-bearing
rather than aspirational:

- **Environments: 34/34.** Every `with_config` calls `config.validate()?` and
  returns `Result<Self, ConfigError>`.
- **RL agents: 8/8.** `dqn_agent.rs:178`, `c51_agent.rs:165`,
  `qrdqn_agent.rs:165`, `ddpg_agent.rs:227`, `td3_agent.rs:254`,
  `sac_agent.rs:264`, `ppo_agent.rs:255`, `ppg_agent.rs:254` — each
  `config.validate()?` as the first statement.
- **Policy heads: 4/4**, inside `try_init` (`ppo/policies/categorical.rs:52`,
  `ppo/policies/gaussian.rs:217` and siblings) — ADR 0054.
- **Evolution has the generic chokepoint 0026 anticipated.**
  `crates/rlevo-evolution/src/strategy.rs:505-517`:
  ```rust
  pub fn new(
      strategy: S, params: S::Params, fitness_fn: F, seed: u64,
      device: B::Device, max_generations: usize,
  ) -> Result<Self, ConfigError>
  where
      S::Params: Validate,
  {
      params.validate()?;
      ...
  ```
  One bound, one call, every `Strategy<B>` covered — the `where C: Validate`
  seam 0026 §Consequences named as the reason for making `Validate` a trait
  rather than a convention.

### Defence in depth behind that chokepoint

Eight evolution strategies additionally carry, as the first statement of
`Strategy::init`:

```rust
debug_assert!(params.validate().is_ok(), "invalid GaConfig reached init: {params:?}");
```

at `ga.rs:242`, `cma_es.rs:497`, `cmsa_es.rs:334`, `es_classical.rs:359`,
`de.rs:291`, `gp_cgp.rs:382`, `neuroevolution/arch_nas.rs:589`, and
`coevolution/cooperative.rs:281`. Eight sites with an identical shape and an
identical message template is a **convention**, not eight oversights: it asserts
that the `EvolutionaryHarness::new` chokepoint was in fact traversed. It is a
`debug_assert!` deliberately — the *rejection* is the harness's `Result`; this
only catches a caller who constructed a `Strategy` state directly in a test.

### The newtype layer is live, and it is the part #326 missed

The decisive counter-evidence is that **a `pub` field is not necessarily an
unguarded field**. `GaConfig` (`crates/rlevo-evolution/src/algorithms/ga.rs:73-89`):

```rust
pub struct GaConfig {
    pub pop_size: usize,
    pub genome_dim: usize,
    pub bounds: Bounds,                 // ADR 0027 — lo <= hi, no NaN
    pub mutation_sigma: NonNegativeRate, // ADR 0031 — finite, >= 0
    ...
}
```

and its `Validate` impl (`ga.rs:113-117`) says so in a comment:

> `mutation_sigma` is a `NonNegativeRate` (finite, `>= 0`); the `GaCrossover`
> payloads (`alpha: NonNegativeRate`, `p: Probability`) are likewise valid by
> construction — no scalar `in_range` checks here, see ADR 0031.

`GaConfig { mutation_sigma: -1.0, .. }` does not compile. `Bounds` and the rate
newtypes have private fields and fallible constructors, so a struct literal
cannot produce an out-of-range value **at all** — there is nothing left for
`validate()` to catch on those fields, which is why 0027 and 0031 *deleted*
their paired `config::` check lines rather than keeping them. Field privacy on
`GaConfig` would add nothing here that the newtype has not already made
unrepresentable, and it would cost the `..Default::default()` idiom.

This is the general form of the answer: **when an invariant genuinely must
survive struct-literal construction, encode it in the type, not in the
accessor.** Privacy on the *container* is the wrong lever; privacy on the
*value* is the right one.

### `#[non_exhaustive]` house style is already unambiguous

There are **41** `#[non_exhaustive]` attributes under `crates/`. **Zero** are on
a struct. All are on enums — `EnvironmentError`, `RecordError`, `EnvFamily`,
`GridTile`, `CheckpointKind`, `Color`, `EpisodeKind`, and the
`rlevo-benchmarks-report-client` wire types. The attribute is used for its
*match-exhaustiveness* meaning (ADR 0044 notes it "constrains only exhaustive
`match` outside core, not variant construction"), never for its
*struct-literal-forbidding* meaning. The convention was uniform across 41 sites
and undocumented; it is now documented.

### The reframing has already been applied in-tree, citing #326

`crates/rlevo-reinforcement-learning/src/algorithms/c51/c51_config.rs:120-130`
documents `delta_z()` returning `f32::NAN` for a degenerate `num_atoms < 2`:

> [`Validate`] rejects such a config, but a struct literal can bypass the
> builder, so this stays total rather than panicking (see issue #326).

This is the worked example of the correct answer to the one case that
consumption-validation genuinely does not cover: a **read without a
construction**. An accessor on an unvalidated config must be **total** — return
a sentinel, never panic. Not because the config might be invalid in production
(the chokepoint rejects it), but because a total function costs nothing and a
panicking one converts a misuse into a crash. That obligation is on the
*accessor*, and it is discharged by totality, not by privacy.

## Decision

### 1. `*Config` types keep `pub` fields

A `*Config` is a hyperparameter record. Its consumers are researchers tuning
values, and the operation they perform constantly is:

```rust
GaConfig { mutation_sigma: NonNegativeRate::new(0.05), ..Default::default() }
```

Field privacy converts every such site into a builder chain. The library exists
to be tuned; the tuning idiom is not incidental ergonomics.

`*Config` types are therefore bound by the `rules.md §4` Config Validation
Contract **alone**. The §2 encapsulation rule does not apply to them.

### 2. Validation is the consumer's obligation, at construction

Unchanged from ADR 0026 §2, restated because it is the load-bearing half:
whatever consumes a caller-supplied config **by value** calls
`config.validate()?` as its first statement and returns
`Result<_, ConfigError>`. New config-consuming constructors adopt this without
exception.

### 3. `*State` / `*Params` / `*Genome` encapsulate

Unchanged from `rules.md §2`. The distinction from §1 is not stylistic:

| | `*Config` | `*State` / `*Params` / `*Genome` |
|---|---|---|
| Invariant kind | value-range on independent fields | structural, spanning fields (lengths agree, derived tail consistent) |
| Lifetime | checked once, at construction | must hold after every in-place mutation |
| Mutation | none — moved by value into the consumer | operators mutate in place, every generation |
| Consequence of a bad value | rejected at one chokepoint | panics generations later, far from the cause (issue #141) |

A config is *consumed*; a state is *evolved*. Only the second has a window in
which an external write can break an invariant that was already established.

### 4. Struct-literal-surviving invariants go in a newtype

If an invariant must hold for a value regardless of how the containing struct
was built, encode it in a validated newtype following the ADR 0027 / 0031 shape
— private field, `new` (panicking, for literals) / `try_new` (`Result`), a
dedicated `Copy` error, `#[serde(try_from)]`.

This is the correct lever because it is **transitive**: a `Bounds` field is
valid inside a config, inside a `*Params`, after a `Deserialize`, after a
`Clone`-then-mutate, and in a struct literal. Field privacy is valid only at the
one boundary it guards.

The corollary, which is the answer to #326: **a `pub` field of a validated
newtype is safe under struct-literal construction.** Auditing "how many configs
have `pub` fields" measures nothing; auditing "which `pub` fields carry an
invariant a newtype does not enforce and a chokepoint does not check" measures
the real exposure.

### 5. `#[non_exhaustive]` is for enums

`#[non_exhaustive]` marks an enum whose variant set may grow, forcing downstream
`match` to carry a wildcard arm. It is **not** used on structs in this
workspace. Adding it to a struct is a decision requiring its own ADR.

### 6. Accessors on a config must be total

A method on a `*Config` that reads fields without constructing anything (e.g.
`C51Config::delta_z`) must be total for every value the type can represent,
including values `validate()` rejects. Return a documented sentinel (`f32::NAN`
for an undefined spacing) and say so in a `# Returns` section. Do not panic:
`validate()` is the rejection mechanism, and a panicking accessor turns a
misuse into a crash for no gain. `c51_config.rs:120-130` is the reference.

### 7. Validate first, destructure second

A config-consuming constructor calls `validate()` **before** reading any field.
`crates/rlevo-environments/src/box2d/bipedal_walker/env.rs:106-110` currently
reads `config.terrain` to select a terrain generator *before* delegating to
`Self::build(config, terrain)`, which validates. Harmless today —
`BipedalTerrain` is a plain enum with no invariant — but it is a template that
becomes a defect the moment the pre-read field carries one. New code validates
first.

### 8. Reconciliation with `docs/rules.md §2` (applied on acceptance)

Same reconcile-on-acceptance flow as ADR 0026 §4 (rules.md is not itself
immutable; ADRs supersede it). §2 "Struct Field Encapsulation" gains an explicit
`*Config` clause stating the exemption and pointing at §4, a clause stating that
struct-literal-surviving invariants belong in a newtype, and the
`#[non_exhaustive]`-is-for-enums rule — replacing the current situation where the
exemption is inferable only from `*Config`'s absence from a list.

## Consequences

### Positive

- The rule is legible from one document. #326 exists because it was not; that
  cost is now paid once.
- No churn: 71 configs, ~24 cross-crate construction sites, and the published
  user-book examples are unchanged.
- Names the right audit question ("which `pub` field carries an unguarded
  invariant?") in place of the misleading one ("how many `pub` fields?").
- The newtype layer gets an explicit mandate, so the next unguarded scalar has a
  named destination (ADR 0027 / 0031) instead of a privacy debate.

### Negative / accepted costs

These are real. They are accepted with reasons, not waved away.

1. **TOCTOU is real and unfixed.** `pub` fields plus `Clone` permit
   validate → mutate a field → use. Consumption-validation does not close this
   window; only encapsulation or a newtype does. **Accepted** because a config
   moves **by value** into its consumer at construction, so the window requires
   a caller to deliberately keep a mutable handle between the `validate()` call
   and the consuming call — not a shape that arises by accident. Where it
   matters for a specific field, the mitigation is §4: put the invariant in a
   newtype, and the post-validation mutation becomes unrepresentable rather than
   merely unlikely.

2. **Read-before-validate is a footgun template.**
   `bipedal_walker/env.rs:106-110` is the live instance (§7). It is harmless
   today and is left in place rather than churned; the convention is recorded so
   the next constructor copied from it validates first. There is no lint for
   this — it is review-enforced, like `total_cmp` (§3) and the
   `unwrap_or_default` rule (§4).

3. **The `Deserialize` obligation is dormant, not satisfied.** ~22 configs
   derive `Deserialize` plainly — no `#[serde(try_from)]`, no post-decode hook.
   Executed proof:
   `bincode::serde::decode_from_slice::<GoToDoorConfig>` accepts `size: 1`,
   below `MIN_SIZE = 5` (`grids/go_to_door.rs:101`), which `validate()` rejects.
   Nothing violates ADR 0026's loader obligation today **only because no config
   loader exists anywhere in the workspace** — there is no code path that turns
   bytes into a config. The obligation is therefore live and unowned: whoever
   adds the first loader (run manifest, config file, checkpoint restore) must
   call `validate()?` on the decoded value and propagate the `Err`. Recorded
   here explicitly so that person finds it, since neither the type signatures
   nor the tests will tell them.

4. **Conformance-test coverage is scoped to `rlevo-environments`, deliberately.**
   `crates/rlevo-environments/tests/config_validation_chokepoint.rs` pins the
   consumption chokepoint for that crate's configs only. The other three
   config-bearing crates are **out of scope by decision, not by oversight**:
   `rlevo-evolution`'s 20 configs share a single generic chokepoint
   (`crates/rlevo-evolution/src/strategy.rs:505-517`,
   `where S::Params: Validate`), so there are not 20 hand-written call sites to
   regress; `rlevo-reinforcement-learning`'s 13 validate in each agent's `new`;
   and `rlevo-benchmarks`'s 4 have no `impl Validate` at all. The environments
   crate carries the regression risk because its 34 chokepoints are 34 separate
   hand-written `config.validate()?` calls, any one of which a new environment
   can omit silently. Extending coverage to the per-agent RL constructors is
   reasonable follow-up work; the generic evolution bound needs no such test.

### Neutral

- Zero code change. This ADR records an allocation and reconciles `rules.md §2`.
- Two-way door in both directions: any individual config can later adopt a
  builder additively, and any individual field can later become a newtype
  additively. Neither requires revisiting this decision.

## Alternatives Considered

### Option 1 — Private fields + accessors on all 71 configs (#326's primary remedy)

Churns 71 config types and every construction site in the workspace, examples
and user-book chapters included. It destroys the
`GaConfig { pop_size: 64, ..Default::default() }` hyperparameter-override idiom,
which is the single most common thing a user of an evolutionary/RL library does
— replacing a one-line literal with a builder chain per experiment.

And it buys less than it appears to. The invariants it would guard on `GaConfig`
are **already** unrepresentable via `Bounds` / `NonNegativeRate` / `Probability`
(`ga.rs:73-89`, `ga.rs:113-117`); the remaining ones (`pop_size >= 1`,
`genome_dim != 0`) are cross-field-free scalars the chokepoint already rejects
with a field-named `ConfigError`. Rejected.

### Option 2 — `#[non_exhaustive]` on config structs (#326's secondary remedy)

Rejected on four independent grounds, any one of which is sufficient:

- **It breaks 24 cross-crate struct-literal sites across 19 files**, including
  the published user-book example
  `crates/rlevo-examples/examples/book/ch01_sphere_ga.rs:37` and `:87`.
- **It does not preserve the escape hatch people assume it does.**
  `#[non_exhaustive]` on a struct forbids cross-crate **functional-update
  syntax** as well as plain literals — `..Default::default()` stops compiling
  too. The mitigation usually cited for the churn does not exist.
- **It buys zero validation guarantee.** It forces construction through a
  builder whose `build()` still has to call `validate()`. That is *identical
  work to ADR 0026 adoption* — which is already ~95% done — plus a breaking
  change on top.
- **There is no builder to migrate to.** `GaConfig` and the four policy-head
  configs have none, so the change is not "route through the builder" but
  "write five builders first."

Rejected.

### Option 3 — Validate at every point of use, not at construction

#326's objection is that this "scatters" the checks. Empirically it does not,
because of how configs are actually used: **a config is consumed by value at
construction, not read field-by-field at runtime.** There is therefore roughly
**one chokepoint per config type**, and they are already written — environments
34/34, RL agents 8/8, policy heads 4/4, and evolution covered wholesale by the
single generic bound at `strategy.rs:505-517`. The scattering the objection
predicts is not observable in the tree.

The one genuine "read without construction" case is an accessor like
`delta_z()`, and §6 answers it with **totality**, not with a scattered check.
Rejected as a description of the problem, not as a remedy.

### Option 4 — Do nothing, leave the rule implicit

The status quo produced #326. A reviewer competent enough to audit 71 types
across four crates could not reconstruct the rule from four correct documents
that never state it. Rejected — that is precisely the cost this ADR pays down.

### Option 5 — A blanket "every config field becomes a newtype" sweep

Over-corrects. `pop_size: usize` has no interesting sub-`usize` domain beyond
`>= 1`, and a `PopSize` newtype per such field would multiply the type surface
for invariants the chokepoint already reports better (a `ConfigError` naming
`GaConfig.pop_size` beats a bare `PopSizeError`). Newtypes earn their place
where the invariant is (a) violable by a plausible literal and (b) consumed far
from the construction site — which is exactly the `NaN`/`Inf`/out-of-range
*rate* case ADR 0031 already covers. Deferred to case-by-case judgment under §4.

## References

- Issue #326 — "all 32 config structs have `pub` fields, so struct-literal
  construction bypasses `validate()`"; resolved by refutation. (The stated count
  of 32 is also low; the true figure is 71.)
- ADR [0026](0026-shared-config-validation-convention.md) — `Validate` /
  `ConfigError`; §3 already names the struct-literal case and routes it to
  `validate()`. Immutable; its `87` config count is superseded by the `71`
  recorded here.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — `Bounds`; the
  valid-by-construction shape.
- ADR [0031](0031-probability-rate-newtypes.md) — `Probability` /
  `NonNegativeRate`; the same shape for scalar rates, and the precedent for a
  self-validating field *removing* its `validate()` line.
- ADR [0044](0044-post-terminal-step-is-an-error.md) — `#[non_exhaustive]` on
  `EnvironmentError`, with the match-exhaustiveness reading this ADR generalises.
- ADR [0054](0054-policy-head-construction-is-fallible.md) — the most recent
  chokepoint adoption; `try_init` validates before building.
- `docs/rules.md §2` (Struct Field Encapsulation — reconciled here) and `§4`
  (Config Validation Contract — unchanged).
- Code: `crates/rlevo-evolution/src/strategy.rs:505-517` (the generic
  chokepoint), `crates/rlevo-evolution/src/algorithms/ga.rs:73-89` and `:113-117`
  (the newtype layer, with its own rationale in a comment),
  `crates/rlevo-reinforcement-learning/src/algorithms/c51/c51_config.rs:120-130`
  (the totality answer, citing #326),
  `crates/rlevo-environments/src/box2d/bipedal_walker/env.rs:106-110`
  (read-before-validate),
  `crates/rlevo-environments/src/grids/go_to_door.rs:101` (`MIN_SIZE`, the
  dormant `Deserialize` obligation's witness).
