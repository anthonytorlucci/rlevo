---
project: rlevo
status: active
type: decision
date: 2026-08-20
tags: [adr, decision, architecture, rlevo-core, rlevo-benchmarks, rlevo-environments, breaking-change, bench-env]
---

# ADR 0077: Remove `BenchEnv`, `BenchStep`, and `BenchError`

## Status

**Accepted (2026-08-20).**

**Supersedes the remainder of ADR
[0004](0004-move-bench-traits-into-rlevo-core.md)** as it applies to
`crates/rlevo-benchmarks/src/env.rs` → `crates/rlevo-core/src/evaluation.rs`:
that move, its typed `BenchError`, and the `rlevo_benchmarks::env` shim are all
retired, because the moved traits themselves are gone. **ADR 0004's other
decisions remain active** — `fitness.rs`, `util/seed.rs`, the `Metric` /
`MetricsProvider` relocation, the `agent` and `seed` shim modules, and the
dependency-edge rework are untouched.

Completes the arc begun by ADR [0075](0075-bench-env-erases-rank-not-modality.md)
(the rationale was false) and ADR
[0076](0076-trial-seam-splits-machinery-from-trial-shape.md) (the replacement
seams).

## Context

ADR 0075 established that `BenchEnv`'s justification did not hold: the erasure
it advertised was on the rank axis while `Observation`/`Action` — the things
that actually differ between environments — survived it, so it could never have
carried the heterogeneous suite it was named for, and no `dyn BenchEnv` existed
anywhere. It also found that the trait's real job was spanning two disjoint
implementor families.

ADR 0076 gave each family its own seam. Subsequent work drove both to
completion:

- the evolutionary family moved to `GenerationProbe` + `GenerationTrial`, and
  the harness `BenchEnv` impls were dropped;
- the episodic family moved to `Evaluator::run_suite` binding directly on
  `Environment<D, SD, AD, RewardType = ScalarReward>`, with `BenchAdapter`
  deleted.

That left `BenchEnv`, `BenchStep`, and `BenchError` with **zero implementors
and zero users** in the workspace — definitions and re-exports only.

## Decision

**Delete all three, plus the `rlevo_benchmarks::env` shim module and the
crate-root re-export.**

1. `crates/rlevo-core/src/evaluation.rs` keeps only [`GenerationProbe`]. Its
   module docs record what was removed and why, so a reader arriving from ADR
   0004 is not left searching.
2. `crates/rlevo-benchmarks/src/lib.rs` drops `pub mod env` and the flat
   `pub use ... {BenchEnv, BenchError, BenchStep}`; the crate root now
   re-exports `GenerationProbe` instead. The `agent` and `seed` shims stay.
3. `docs/rules.md`'s `rlevo-core` inventory names the `evaluation` drive seam as
   `GenerationProbe` only.
4. `rlevo-core`, `rlevo-benchmarks`, and `rlevo-environments` READMEs updated,
   including the benchmarks quickstart, which taught `impl BenchEnv for MyEnv`
   and now teaches `impl Environment<1, 1, 1> for MyEnv`.

**This is a breaking change to a published surface** and is recorded in the
CHANGELOG's Unreleased *Breaking changes* section with a migration note.

### The rename is moot

ADR 0075 deferred renaming these to `Steppable`/`StepOutcome`/`StepError`, with
the cost noted as depending on whether a later split deleted them. It did.
There is nothing left to rename, and the ~43-file rename that the original
private spec opened with is now unnecessary rather than merely deferred.

## Consequences

**Positive:**

- **A contract crate stops exporting three dead types.** `rlevo-core` is the
  crate every other crate depends on; unused public surface there is the most
  expensive kind.
- **One drive vocabulary per shape.** Episodic work speaks `Environment` /
  `Snapshot`, which already carried strictly more information (`EpisodeStatus`
  vs a flat `done: bool`, `Snapshot::metadata()` for named reward components).
  Self-paced work speaks `GenerationProbe`.
- **The `BenchError` layer disappears.** Its `Reset`/`Step` variant tag was
  redundant with call-site context — the evaluator always knew which lifecycle
  method it had just called — and errors now surface as `EnvironmentError`
  directly.

**Negative / accepted costs:**

- **Breaking for any external implementor of `BenchEnv`.** Migration is to
  implement `Environment` (for an episodic env) or `GenerationProbe` (for a
  self-paced loop). Both are strictly more expressive; neither is a mechanical
  rewrite, because `Environment` requires naming four associated types and a
  `Snapshot`. Mitigated by the project's alpha status and by
  `SnapshotBase<R, Obs, Rew>`, which supplies the snapshot impl.
- **The rank-erasure cost did not vanish, it relocated.** `EpisodicTrial` now
  carries `D`/`SD`/`AD` for exactly the reason `BenchAdapter` did: `Trial` is
  rank-free, so without them the impl is rejected by E0207. One type-parameter
  list in the harness replaces a wrapper type in `rlevo-environments` plus a
  double-wrap at every recording call site — a better trade, but a trade.
- **Const-generic inference now sits on the critical path.** `run_suite` infers
  three parameters from `Suite<E>`, which works only while each type has exactly
  one `Environment` impl. Coherence permits a second at different ranks; the
  failure is E0284 at the call site, not a silent wrong choice, and the fix is a
  turbofish — but adding a second `Environment` impl to an already-benchmarked
  type is a breaking change for its callers.

**Neutral:**

- No behaviour change. Every migrated caller produces the same trials, seeds,
  and reports; the two pre-existing `rlevo-core` `reward.rs` doctest failures
  are unrelated and unchanged throughout.

## Alternatives considered

**Keep them, deprecated.** Rejected. `#[deprecated]` earns its place when
external code plausibly depends on a surface and needs a migration window. This
is a pre-1.0 alpha with no known external implementors, and a deprecated trait
in a contract crate still has to be read, documented, and kept compiling. ADR
0075 already spent one cycle documenting a surface nothing used; a second would
be the same mistake in a quieter form.

**Rename instead of delete** (`Steppable`/`StepOutcome`/`StepError`). Moot — see
above. Renaming a type with no implementors is pure churn.

**Delete `BenchEnv` but keep `BenchStep`/`BenchError` as plain data.** Rejected.
Nothing constructs them once the trait is gone. `BenchStep` was already
displaced by `GenerationStep` on the evolutionary side and by `Snapshot` on the
episodic side, and `BenchError` wrapped an `EnvironmentError` that callers now
receive unwrapped.

## References

- ADR [0004](0004-move-bench-traits-into-rlevo-core.md) — moved these traits
  into core; that portion is now fully superseded.
- ADR [0075](0075-bench-env-erases-rank-not-modality.md) — retired the
  rationale.
- ADR [0076](0076-trial-seam-splits-machinery-from-trial-shape.md) — built the
  replacement seams.
- `crates/rlevo-core/src/evaluation.rs` — what remains.
- `crates/rlevo-benchmarks/src/evaluator.rs` — `Trial`, `EpisodicTrial`,
  `GenerationTrial`, `run_trials`.
