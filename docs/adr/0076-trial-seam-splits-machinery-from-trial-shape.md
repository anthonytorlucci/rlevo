---
project: rlevo
status: active
type: decision
date: 2026-08-20
tags: [adr, decision, architecture, rlevo-core, rlevo-benchmarks, rlevo-evolution, evaluation, trial-seam, generation-probe]
---

# ADR 0076: The `Trial` seam splits evaluator machinery from trial shape, and `GenerationProbe` joins `rlevo-core`

## Status

**Accepted (2026-08-20).** Builds on ADR
[0075](0075-bench-env-erases-rank-not-modality.md), which established that
`BenchEnv` spans two disjoint implementor families — typed environments and
evolutionary generation-steppers — and that its stated rationale did not hold.
This ADR acts on that finding without yet removing anything.

**Supersedes nothing.** `BenchEnv` and its two harness impls remain in place;
every `Evaluator::run_suite` caller is unaffected.

## Context

`Evaluator::run_suite` mixed two concerns. One is expensive and easy to get
wrong: rayon fan-out, per-trial `catch_unwind`, checkpoint load/skip/save,
fail-fast, and the reporter lifecycle. The other is the *shape* of the work —
an episodic environment rollout.

An evolutionary run is a poor fit for that shape, and the repo had already
accumulated three independent workarounds for the same missing abstraction:

1. **`PopulationReporter` is not a `Reporter`.** It implements
   `PopulationObserver` and is wired via `EvolutionaryHarness::with_observer`,
   carrying per-generation data through a side channel that bypasses the
   reporter lifecycle entirely.
2. **`RecordingReporter::without_lifecycle`** exists to switch the episode
   lifecycle *off*, and both `report_*_with_client` examples use it.
3. **`rastrigin_run_suite.rs`'s `collect_best_returns`** reads
   `-e.return_value / steps` off `.last()` — three simultaneous corrections
   (negate for objective sense, divide to un-integrate a summed best-so-far,
   and `.last()` because the episode axis is degenerate).

That degeneracy was measured, not assumed: setting `num_episodes: 2` on the
Rastrigin suite produced bit-identical episode pairs across all 8 trials
(`[-3533.3695335388184, -3533.3695335388184]`, lengths `[80, 80]`), because
`reset` re-seeds from `base_seed`. A second episode is duplicated compute
carrying zero information.

The question was whether the valuable machinery could be shared anyway, or
whether the report and reporter vocabulary is irreducibly episode-shaped. A
survey settled it:

| Surface | Members | Episode-shaped |
|---|---|---|
| `Reporter` | 5 hooks | 1 (`on_episode_end`) |
| `TrialReport` | 9 fields | 1 (`episodes`) |
| `checkpoint.rs` | 5 fns | 0 |

`core_metrics` is the near-miss: episode-*named* (`return/mean`,
`episode/length_mean`) but its signature is `(&[f64], &[usize], f64,
Option<f64>)` — plain slices. Structurally it is already shape-agnostic. And of
the five production `on_episode_end` bodies, one is empty, one fans out, one
forwards to a channel, one logs a line, and only `RecordingReporter` does real
work — behind the off switch noted above.

## Decision

**Introduce a `Trial` seam in `rlevo-benchmarks`, and place `GenerationProbe`
in `rlevo-core` alongside `BenchEnv`.**

1. **`Trial`** (`rlevo-benchmarks::evaluator`) — `fn run(self, cfg, info,
   reporter: &Mutex<&mut dyn Reporter>) -> TrialReport`. `Evaluator::run_trials`
   owns all the machinery and is generic **only** over `T: Trial`; it carries no
   `BenchEnv` or `BenchableAgent` parameter. `run_suite` keeps its exact
   signature and becomes a wrapper that builds one `EpisodicTrial` per key.
   The reporter is threaded as `&mut dyn Reporter` rather than a generic `R` so
   the seam does not spread a type parameter through every impl; `Reporter` was
   already object-safe.
2. **`GenerationProbe`** (`rlevo-core::evaluation`) — `type Metrics; fn
   begin(&mut self); fn advance(&mut self) -> Option<Self::Metrics>`. `advance`
   MUST check the budget before stepping.
3. **`GenerationTrial<P>`** (`rlevo-benchmarks`) — bounded `P::Metrics:
   MetricsProvider`, which had zero impls before this change.
   `EvolutionaryHarness` and `CoEvolutionaryHarness` gain `GenerationProbe`
   impls; `StrategyMetrics` and `CoEAMetrics` gain `MetricsProvider` impls under
   `ea/` and `coea/` name prefixes so they cannot collide with `core_metrics`
   names in the same maps.
4. **A generation trial reports no episodes.** `TrialReport::episodes` is left
   empty and `on_episode_end` never fires. Fabricating one synthetic
   `EpisodeSummary` was rejected: it would preserve exactly the trap that makes
   `num_episodes > 1` multiply cost for no signal today.
5. **`docs/rules.md`'s `rlevo-core` inventory is corrected.** It enumerated
   core's contents but omitted the `evaluation` module entirely, even though ADR
   0004 placed it there and the same section's `rlevo-evolution` bullet
   references `BenchEnv` three lines later. The inventory now names both drive
   seams.

### Why `GenerationProbe` belongs in `rlevo-core`

`rlevo-benchmarks` can only see `rlevo-evolution` behind its `record` feature,
so the `GenerationTrial` impl needs the trait somewhere both crates can reach.
Core is the right home on the ADR 0004 precedent — a drive seam that more than
one crate family needs — and the trait is shape-generic: `begin` plus `advance`
naming no genome, population, or tensor. Per `rules.md`'s contract-crate rule
this is a trait surface, not an algorithm implementation.

### Why `Option` rather than a `done` flag

The budget lives in the probe, which already owns it. A separate boolean
licenses a second, unenforced copy of the same number in the driver's config —
which is exactly what `EvaluatorConfig::max_steps` is today, hand-synced against
`max_generations` at every call site. In `GenerationTrial`, `max_steps` is
demoted to a backstop against a probe with a broken exhaustion check.

## Consequences

**Positive:**

- **The machinery is written once.** A trial shape with no episode axis reuses
  parallelism, panic capture, checkpointing, and fail-fast by implementing one
  method.
- **Typed metrics replace a single `f64`.** `BenchStep` could carry one scalar
  reward; `P::Metrics` carries the full `StrategyMetrics`/`CoEAMetrics` the
  harness already computes and currently discards.
- **The degenerate episode axis is gone on the probe path**, so the
  negate/divide/`.last()` correction has no reason to exist in migrated callers.
- **`MetricsProvider` acquires its first implementors**, four years of being a
  declared-but-unused trait surface ended.

**Negative / accepted costs:**

- **`rlevo-core` grows a second drive-seam trait** while ADR 0075 has an open
  question about whether the first one belongs there. Accepted deliberately: if
  the split completes, `GenerationProbe` plus a slimmed environment path is what
  *replaces* `BenchEnv` in core rather than joining it. The alternative —
  feature-gating — pushes the wrong way (below).
- **Two `Trial` impls now exist with different report vocabularies.** An
  episodic trial fills `episodes` and `return/*`; a generation trial fills
  `ea/*` and leaves `episodes` empty. Report consumers that assume a non-empty
  episode list will see one. That assumption was always unsound for evolution
  runs; it is now visible instead of papered over.

**Neutral:**

- No behaviour change for existing callers. `run_suite`'s signature, seed
  derivation order, and `catch_unwind` placement are untouched.

## Alternatives considered

**An `evolution` feature on `rlevo-benchmarks`** gating `dep:rlevo-evolution`,
leaving `GenerationProbe` in `rlevo-evolution`. Rejected. It makes evolutionary
benchmarking opt-in; `--all-features` runs would mask breakage in the ungated
build; and, decisively, it *deepens* the `rlevo-benchmarks → rlevo-evolution`
edge that ADR 0075 identified as the thing blocking any future relocation of the
trait surface out of core. It entrenches the obstacle to the other fork.

**Promote `rlevo-benchmarks` to a production dependency of the `rlevo` umbrella**
and put `GenerationTrial` there, where both crates are already visible.
Rejected: `rlevo-benchmarks` is currently a dev-dependency of `rlevo`, and
`rules.md` describes the umbrella as aggregating core, RL, evolution, and hybrid
— not the harness. Pulling an evaluation harness into the public aggregator is a
larger change than the one being made.

**Fabricate one `EpisodeSummary` per generation run** so `TrialReport` stays
uniformly shaped. Rejected — see Decision point 4.

**Add an `on_generation` hook to `Reporter`.** Deferred. `PopulationObserver`
already carries per-generation data and is the established channel; adding a
second would give two ways to report the same thing before anything has asked
for it.

## References

- ADR [0075](0075-bench-env-erases-rank-not-modality.md) — established the
  two-disjoint-families finding this ADR acts on.
- ADR [0004](0004-move-bench-traits-into-rlevo-core.md) — the precedent for
  placing a shared drive seam in `rlevo-core`.
- ADR [0005](0005-examples-and-cross-crate-tests-in-umbrella.md) — why the
  cross-crate seam test lives in `crates/rlevo/tests/`.
- `crates/rlevo-core/src/evaluation.rs` — both drive seams.
- `crates/rlevo-benchmarks/src/evaluator.rs` — `Trial`, `EpisodicTrial`,
  `GenerationTrial`, `run_trials`.
- `crates/rlevo/tests/generation_trial_seam.rs` — cross-crate proof that a
  probe runs through the shared machinery with no episode axis.
