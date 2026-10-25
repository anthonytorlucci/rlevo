---
project: rlevo
status: active
type: decision
date: 2026-10-25
tags: [adr, decision, metrics, agent, generation-probe, rlevo-core, rlevo-evolution, rlevo-harness, public-surface, breaking-change]
---

# ADR 0097: A trial reads its metrics from what it drives, under one rule

## Status

**Accepted (2026-10-25).**

Supersedes a clause of each of three ADRs; the rest of each stands.

- ADR [0004](0004-move-bench-traits-into-rlevo-core.md), decision point 3,
  for `MetricsProvider`: the trait is deleted. `Metric` stays in
  `rlevo_core::fitness`.
- ADR [0076](0076-trial-seam-splits-machinery-from-trial-shape.md),
  Decision 3: `GenerationTrial<P>` is bounded `P::Metrics:
  MetricsProvider`, and `StrategyMetrics` and `CoEAMetrics` implement it.
  `GenerationTrial<P>` needs `P: GenerationProbe` alone, and the two loops'
  `GenerationProbe` impls report through `emit_metrics` (Decision 2).
- ADR [0079](0079-harness-metrics-are-a-privileged-absorption-path.md),
  Decision 5's last paragraph: "The fix delegates the emission-cadence
  contract to `MetricsProvider::emit`'s existing rustdoc", the
  drain-and-reset rule that `Metric::Counter`'s docs pointed to. The rule is
  `Metric`'s own (Decision 1). The rest of 0079, the two absorption paths
  and the reserved names, stands.

ADR [0095](0095-the-agent-seam-is-one-agent-trait-with-defaulted-learning-hooks.md)
is unchanged: `Agent::emit_metrics` keeps its signature, its default and
its one read per trial, and "covering the whole trial" is the case of
Decision 1 that the harness reads.

## Context

A trial reports two kinds of number: what the harness measures itself
(returns, lengths, the wall clock, the generation count), and what the thing
it drives knows about its own work (an agent's gradient updates, a
generation loop's best fitness). The second kind reached the report through
two methods with one signature, `fn(&self) -> Vec<Metric>`:

| | `Agent::emit_metrics` | `MetricsProvider::emit` |
|---|---|---|
| implemented by | the agent | a generation's metrics value, `GenerationProbe::Metrics`, not the loop |
| read by | `EpisodicTrial`, once, after the last evaluation episode | `GenerationTrial`, once, on the last generation's value; the others were dropped |
| what a read covers, as written | "the whole trial, training included" | "all metrics accumulated since the last call"; implementations "should drain internal accumulators so that repeated calls do not double-count" |
| what a read covers, as built | the RL agents' counters are totals since the agent was built, which is the trial, since `run_suite` builds one agent per trial | a snapshot of the last generation: per-generation scalars (`ea/best_fitness`) and running ones (`ea/best_fitness_ever`); nothing to drain |

Both results went to the same place, `TrialReport::absorb_metrics`, which
keeps the latest value under a name. So the code followed one rule, a
snapshot of the trial at the end, under two names and two written
contracts. The second contract could not be followed and did harm where it
was cited:

- **`emit(&self)` cannot drain.** A shared reference resets nothing without
  interior mutability, and no implementor had any.
- **A drained value is lost by its reader.** `absorb_metrics` keeps the
  latest value. A provider that drained and was read twice would leave only
  the second interval in the report. ADR 0079 chose to keep the latest
  rather than sum, which is right only if each value already covers the
  whole span.
- **It was the rule an agent's counters were pointed to.** ADR 0079 made
  `Metric::Counter` and `Metric::Histogram` say that the span of a value is
  "`MetricsProvider::emit`'s contract". After ADR 0095 no agent implements
  `MetricsProvider`, and the eight RL agents report running totals, so an
  agent's counters contradicted the docs of the type they return.
- **Two lines went stale with ADR 0095.** `MetricsProvider`'s docs said it is
  "implemented by agents", and the harness's reserved-name docs named it as
  an agent's path.

The rule also has to hold for a reader that reads more than once. A sink
that logs a run as a time series reads at each episode or generation
boundary and logs each value at its point on a timeline. Under a drain
rule every such reader must add values up; under a snapshot rule each
value stands alone.

## Decision

### 1. One rule, stated on `Metric`

A producer's metrics describe the trial so far, from its start to the
read:

- a `Counter` is the total since the trial began;
- a `Scalar` is the value at the read: a running best, a mean so far, the
  latest generation's best;
- a `Histogram` holds every sample since the trial began.

A trial begins when its agent is built, or when its generation loop's
`begin` resets it. A read changes nothing, so it can be repeated. Each read
is complete on its own: the reader has one value per read, the latest is
the current one, and no reader adds two reads together.

The evaluation harness reads once per trial, after the last episode or
generation it counts. A reader that logs a time series reads at each
boundary.

### 2. One name: `emit_metrics` on what a trial drives

`GenerationProbe` gains `fn emit_metrics(&self) -> Vec<Metric>`, with
`Agent::emit_metrics`'s name, rule and default (none). `MetricsProvider` is
deleted. `GenerationTrial` reads `self.probe.emit_metrics()` instead of
keeping the last generation's typed value and converting it.

`EvolutionaryLoop` and `CoEvolutionaryLoop` implement it as their latest
generation's metrics under `ea/` and `coea/`, and none before the first
generation; `StrategyMetrics` and `CoEAMetrics` keep their typed accessors.
`GenerationProbe::Metrics` and `advance`'s return type are unchanged.

### 3. The read comes before the cap check

When a `GenerationTrial` reaches `EvaluatorConfig::max_steps`, it calls
`advance` once more to learn whether the probe had budget left, and drops
that generation (ADR 0085, Decision 9). The trial reads `emit_metrics`
after the last counted generation and before that call, beside the wall
clock, so the dropped generation is in no number the trial reports. Before
this ADR the typed value of the last counted generation was kept for the
same reason.

## Consequences

- **Breaking.** `rlevo_core::fitness::MetricsProvider` and its re-export
  `rlevo_harness::metrics::MetricsProvider` are gone. A generation loop of
  one's own moves its conversion from `impl MetricsProvider for
  MyMetrics` into `GenerationProbe::emit_metrics` on the loop. A bound
  `P::Metrics: MetricsProvider` is deleted.
- **Additive.** `GenerationProbe::emit_metrics` has a default, so a probe
  that reports nothing compiles unchanged.
- **One read for both trials.** A report describes an episodic and a
  generation trial the same way, and a reader written against `Metric`'s
  rule reads both.
- **A time-series reader needs no producer change.** It can read
  `emit_metrics` at every boundary. Checked against one such reader,
  rerun.io's data model (its "Logging and ingestion" documentation, version
  0.40.0): each logged value has a point on a timeline, the viewer draws
  the latest value at a time and plots a range of values, and it never sums
  them. A snapshot per read is what that model stores and draws; a drained
  delta is not. Every metric name emitted in the workspace is also a valid
  entity path there with no escaping (parts of letters, digits, `_`, `-`
  and `.`, joined by `/`, none starting with `__`). Nothing here chooses a
  sink, and `Metric` still imposes no naming scheme.
- **Surface.** `rlevo-core` loses one trait and gains one defaulted method;
  `rlevo-harness` loses one re-export; `rlevo-evolution` loses two trait
  impls and gains two methods in existing impls.

## Alternatives considered

- **Keep both, with one written rule.** Rewrite `MetricsProvider::emit`'s
  docs to the snapshot rule and fix the stale lines. Nothing breaks, but
  two names remain for one job, on two kinds of implementor.
- **Rename `MetricsProvider::emit` to `emit_metrics`.** One name, but still
  a second trait, implemented by a value the harness has to keep, where
  `Agent`'s is implemented by the thing the trial drives.
- **Bound `P::Metrics: Into<Vec<Metric>>`.** No trait of rlevo's own, but the
  rule would live on a `From` impl that nothing documents.
- **Make the drain rule work.** Interior mutability in every producer, and a
  summing reader, for a reader that does not exist and would contradict
  ADR 0079's keep-the-latest rule.

## References

- `crates/rlevo-core/src/fitness.rs`: `Metric` and the rule.
- `crates/rlevo-core/src/evaluation.rs`: `GenerationProbe::emit_metrics`.
- `crates/rlevo-core/src/agent.rs`: `Agent::emit_metrics`.
- `crates/rlevo-evolution/src/probe.rs`: the two loops' `emit_metrics`, with
  tests that a read before the first generation is empty, that a second
  read repeats the first, and that `begin` starts a new run.
- `crates/rlevo-harness/src/evaluator.rs`: `GenerationTrial::run`; the cap
  test `generation_trial_warns_once_when_the_cap_cuts_its_probe_short`
  pins Decision 3.
- rerun.io documentation, "Logging and ingestion" (timelines, entity paths,
  latest-at and range queries, static data), version 0.40.0, read
  2026-10-25.
