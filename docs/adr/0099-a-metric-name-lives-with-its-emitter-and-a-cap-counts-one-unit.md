---
project: rlevo
status: active
type: decision
date: 2026-11-01
tags: [adr, decision, metrics, metric-names, harness, evaluator-config, generation-trial, prelude, rlevo-harness, rlevo-evolution, public-surface, breaking-change]
---

# ADR 0099: A metric's name lives with its emitter, and a cap counts one unit

## Status

**Accepted (2026-11-01).**

- Amends ADR [0085](0085-the-generation-loop-is-named-evolutionary-loop.md)
  Decision 9. A generation trial still warns when the evaluator's cap ends
  it, as that decision states, but the cap is
  `EvaluatorConfig::max_generations`, not `EvaluatorConfig::max_steps`.
- Amends ADR [0004](0004-move-bench-traits-into-rlevo-core.md) Decision 3's
  clause that the harness keeps the `metrics::{core, ea, rl}` aggregators.
  The harness keeps the names and reductions of what it measures itself, in
  `metrics`; `metrics::ea` is deleted and `metrics::core` is merged into
  `metrics`.
- Leaves ADR [0079](0079-harness-metrics-are-a-privileged-absorption-path.md)'s
  reserved set as it is. Its constants and `is_harness_reserved` move from
  `metrics::core` to `metrics`, unchanged.

## Context

Three seams are stamped into the `rlevo-algo` template: `Agent`,
`Environment` and `Strategy`. Before v0.6.0 is published, each was checked
for a public item that means two things or has two homes, because a name
fixed after authors have used it is a breaking release. The `Agent` and
`Environment` seams were clear. The `Strategy` seam was not, on the names of
the metrics a generation loop reports:

- `rlevo_harness::metrics::ea` declared four constants:
  `ea/population_diversity`, `ea/best_fitness`, `ea/generations_to_converge`
  and `ea/fitness_variance`. Nothing in the workspace emits three of them.
  The functions beside them, `population_diversity`, `fitness_variance` and
  `ea_metrics`, compute values from numbers their caller supplies; their one
  caller was the `ga_rastrigin` example, which printed how many metrics it
  had built.
- `EvolutionaryLoop` reports seven names (`ea/generation`,
  `ea/population_size`, `ea/best_fitness`, `ea/best_fitness_ever`,
  `ea/mean_fitness`, `ea/worst_fitness`, `ea/broken_count`) and
  `CoEvolutionaryLoop` eight (`coea/generation`, `coea/binding_fitness`, and
  a best, a mean and a hall-of-fame size per population). One of the
  fifteen had a constant.
- So every reader typed the name. The `strategy` template, the `byoe-2` and
  `byoa-1b` probes, the ask/tell guide and four integration tests spelled
  `"ea/best_fitness_ever"` by hand, and both probes recorded it as a
  finding.

The constants were declared in the wrong crate for a structural reason. The
crates that emit names, `rlevo-evolution` and `rlevo-reinforcement-learning`
today and `rlevo-hybrid`'s drivers next, sit upstream of the harness: no
published crate depends on `rlevo-harness` but the umbrella, and
`rlevo-environments` under its off-by-default `record` feature, which `cargo
xtask guards` checks. At its default features the harness does not depend on
`rlevo-evolution` either. A constant declared in the harness can be read
by a consumer, but never used by the code that emits the name, so the two
copies drift, and they had. The rest of the workspace already followed the
other rule: the harness declares the names it measures (`return/mean`,
`generations`, ...) beside the code that computes them, and the RL agents
build their `rl/` names in their own crate.

A second name in the stamped surface meant two things.
`EvaluatorConfig::max_steps` was steps per episode to an `EpisodicTrial` and
generations to a `GenerationTrial`. Everywhere else in the workspace
`max_steps` is steps per episode: the grid environments' caps, `TimeLimit`,
`evaluator::train`'s argument. The `strategy` template set it to its
generation budget.

## Decision

1. **A metric's name is declared by the code that emits it, as a constant
   that code emits through.** For the two evolution loops the constants are
   associated constants on the summary types, beside the accessor or field
   of the value each one names: `StrategyMetrics::BEST_FITNESS_EVER` is
   `"ea/best_fitness_ever"`, next to `StrategyMetrics::best_fitness_ever()`.
   `StrategyMetrics` has seven, `CoEAMetrics` eight, and
   `rlevo_evolution::probe` builds every metric from them. An author who
   knows the value finds its name on the same rustdoc page.
2. **The strings are wire format and do not change.** A report or a
   recording an earlier binary wrote carries them. A test in
   `rlevo_evolution::probe` pins each summary's names as literal strings,
   in the order they are emitted, so changing a constant fails a test
   rather than silently renaming a metric.
3. **The harness declares only what it measures.** `metrics::ea` is deleted,
   with its four constants and three functions. `metrics::core` is merged
   into `metrics`, so `rlevo::harness::metrics::RETURN_MEAN` is the path,
   and the module no longer reuses the contract crate's name.
   `is_harness_reserved` and the reserved set are unchanged; its docs point
   to `StrategyMetrics` and `CoEAMetrics` for the `ea/` and `coea/` names.
   One table of names stays in the harness's reach: `metrics_registry`, the
   recording stack's list of `tracing` field names, which the next paragraph
   leaves as it is.
4. **A cap counts one unit.** `EvaluatorConfig::max_steps` is steps per
   episode, and a `GenerationTrial` ignores it.
   `EvaluatorConfig::max_generations`, 1000 by default as `max_steps`'s
   default was, caps a generation trial. It is still a backstop that should
   never be the one to end a trial, and the warning ADR 0085 Decision 9
   describes now names `max_generations`. An `EpisodicTrial` ignores it.
5. **`Metric` is in `rlevo::prelude`.** An agent's or a probe's
   `emit_metrics` returns `Vec<Metric>`, so the template's `agent` seam
   imports it from the prelude beside `Agent`, with no path into
   `rlevo::core::fitness`.

Not moved here: the loops also report each generation as a `tracing`
event, whose field names (`best_fitness_ever`, `mean_fitness`, and so on,
with no prefix) the recording stack reads through `rlevo-metrics-registry`'s
table. Those names are kept in step with the table by hand, as the comment
at the event says. They are the recording stack's names, a second channel
beside the metrics a trial reports, and change only with that stack.

Not decided here: whether a generation trial the cap cuts short is marked
in its `TrialReport`, beside the warning. That would be a new reserved
counter, an addition a later release can make without a break. `Metric`'s
module, `rlevo_core::fitness`, stays.

## Consequences

- **Breaking**, for v0.6.0:
  - a path through `rlevo_harness::metrics::core` drops `core`;
  - a use of `metrics::ea` reads the name from `StrategyMetrics` or
    `CoEAMetrics`, and its three functions have no replacement;
  - a config that set `max_steps` for a generation trial sets
    `max_generations`. Such a config still compiles, because `max_steps` is
    still a field, so a generation trial whose budget is above 1000 stops at
    1000, and the warning, which names `max_steps` as the field a generation
    trial ignores, is the only sign;
  - a config written as a struct literal names the new field or ends in
    `..EvaluatorConfig::default()`.
- **What an outside author stops writing.** The `byoe-2` and `byoa-1b`
  probes lose their one finding and read
  `StrategyMetrics::BEST_FITNESS_EVER`. The `strategy` template reads the
  constant and sets `max_generations`, and the `agent` template takes
  `Metric` from the prelude.
- **The rule for the next emitter.** `rlevo-hybrid`'s drivers declare their
  names where they emit them, and can also read `StrategyMetrics`'s
  constants, since the crate depends on `rlevo-evolution`. The RL agents
  name their counters at each call to a crate-private helper that prefixes
  `rl/`. Giving those names public constants applies the same rule, and is
  additive.
- **What a report reader imports.** The harness's own names come from
  `rlevo::harness::metrics`, and a loop's from `rlevo::evo`. There is no
  single module listing every name a report can hold, and there could not
  be one: under ADR 0097 an agent or a strategy reports whatever names it
  chooses. The harness's module docs say where the loops' names are.
- **Public surface.** `rlevo-evolution` gains fifteen associated constants.
  `rlevo-harness` loses four constants, three functions and two module
  paths, and gains one field. `rlevo` gains one prelude item.

## Alternatives considered

- **The harness keeps the names.** `metrics::ea` would declare the seven
  names the loop emits, and a test where both crates are visible (the
  umbrella's tests, or the harness under `record`) would check that the
  loop's strings agree. A report reader would find every evolution name
  next to `return/mean`. Rejected: every name would exist twice, a third
  copy would appear with each emitter that cannot depend on the harness,
  and the module could still not list the RL crate's names or an author's.
- **Free constants in a module of `rlevo-evolution`**, for example
  `rlevo_evolution::probe::names`. As safe against drift as the decision,
  but it separates a value's name from the value. Associated constants put
  both on one page.
- **Keep `max_steps` for both trials and correct its doc.** No migration,
  but the field keeps two units, and the template a generation author
  starts from would keep setting a field called `max_steps` to a number of
  generations.
- **Put the generation cap on `GenerationTrial` instead of the config.**
  `Evaluator::run_trials` builds each trial in a closure, so the cap would
  be set once per seed rather than once per suite, and the config would
  stop being the one place a suite's limits are set.

## References

- ADR [0004](0004-move-bench-traits-into-rlevo-core.md): `Metric` in
  `rlevo_core::fitness`, and the harness's aggregators.
- ADR [0079](0079-harness-metrics-are-a-privileged-absorption-path.md): the
  harness's reserved names and `is_harness_reserved`.
- ADR [0085](0085-the-generation-loop-is-named-evolutionary-loop.md)
  Decision 9: the warning a cut-short generation trial emits.
- ADR [0097](0097-a-trial-reads-its-metrics-from-what-it-drives.md): the one
  rule a metric read follows, and an emitter's freedom to name its metrics.
- `crates/rlevo-evolution/src/strategy.rs` (`StrategyMetrics`),
  `crates/rlevo-evolution/src/coevolution/generation_loop.rs`
  (`CoEAMetrics`), `crates/rlevo-evolution/src/probe.rs` (the emitting
  code and the wire-format test), `crates/rlevo-harness/src/metrics.rs`,
  `crates/rlevo-harness/src/evaluator.rs` (`EvaluatorConfig`,
  `GenerationTrial`).
