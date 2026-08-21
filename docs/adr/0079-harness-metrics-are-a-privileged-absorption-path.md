---
project: rlevo
status: active
type: decision
date: 2026-08-21
tags: [adr, decision, benchmarks, metrics, absorb_metrics, namespace, trial-report, issue-1118]
---

# ADR 0079: Harness metrics are a privileged absorption path; `absorb_metrics` becomes checked, safe-by-default

## Status

**Accepted (2026-08-21).** Resolves issue #1118 (`[benchmarks] an agent metric
can silently overwrite a harness measurement`).

**Supersedes, narrowly, one clause each in two places in ADR
[0078](0078-non-finite-reward-is-counted-and-reported-by-the-benchmarks-harness.md):**
the third bullet of its Decision 4 ("It does not make the counter
unclobberable … a deliberate override is not the hazard being defended
against") and the corresponding clause in its Decision 6 discussion of the
`agent_metrics_are_absorbed_after_the_harness_counter` test. Everything else
in 0078 stays **active and unchanged**. Also resolves 0078's "out-of-scope
observation" on `Metric::Counter`'s doc tension with `absorb_metrics`.

## Context

### 1. ADR 0078 named this exact gap and declined to fix it inline

0078 introduced `RETURN_NON_FINITE_STEPS` under the `return/` prefix
specifically so an agent's unprefixed `non_finite_steps` metric would not
collide by *accident*. Its Decision 4 was explicit that the prefix bought only
that:

> **It does not make the counter unclobberable.** An agent that emits the
> exact key `return/non_finite_steps` still wins, because last-writer-wins
> plus agent-metrics-absorbed-last says so. That is deliberate — a deliberate
> override is not the hazard being defended against.

Its Decision 6 closed with the sentence that reopens it here:

> Inverting this assertion so the harness wins would be a change to
> `absorb_metrics`' contract and needs its own ADR, not a test edit.

This is that ADR. The premise — that an agent naming the exact harness key is
a "deliberate override" and therefore not the hazard — does not survive
contact with `Trial` being a public trait. The party that reaches for
`absorb_metrics` next is not necessarily the party that named
`return/non_finite_steps`; it is whoever writes the *next* `Trial`
implementation. "Deliberate" described the metric's author's intent, not the
caller's. A harness measurement getting silently replaced by an agent that
happened to reuse a name is a defect regardless of whether the agent's author
meant it.

### 2. The defect

`TrialReport::absorb_metrics` had one behaviour: `map.insert(name, value)`,
unconditionally. Both `EpisodicTrial::run` and `GenerationTrial::run`
absorbed their own harness measurements first, then agent/probe output last.
An agent or probe emitting `return/mean`, `wall_clock_seconds`,
`generations`, or any other harness-owned name replaced the harness's own
measurement with no error, no warning, and no record. The resulting
`TrialReport` was byte-for-byte indistinguishable from one where the harness
had measured that value itself.

### 3. The polarity is the decision, not a detail of it

Three shapes were available: (a) leave `absorb_metrics` as last-writer-wins
and add an opt-in checked variant; (b) make `absorb_metrics` checked and add
a privileged unchecked variant for the harness's own use; (c) reject
exact-key collisions outright (error / panic).

**Chosen: (b), with the safe behaviour on the name every future caller
reaches for by default.**

`Trial` is a public trait. Its implementors are not enumerable. Had the fix
taken shape (a), the defect would have closed for exactly the two call sites
this PR touches and stayed armed for every `Trial` implementor written from
today onward. The next implementor reaches for the method with the obvious
name; if the obvious name is the unsafe one, the fix accomplishes nothing.

So `absorb_metrics` — the one with no qualifier — is now the checked path,
and the privileged path is named for what it costs to use it correctly:
`absorb_harness_metrics`. Getting this polarity backwards is the single
easiest way for a well-intentioned future edit to reopen this issue silently.
**A later reader proposing to swap the two names back "for symmetry" is
re-introducing this defect — treat the request as a reopen trigger, not a
simplification.**

Option (c) was rejected because a colliding metric name is runtime data (an
agent's free-form choice), not a programming error, and `absorb_metrics` has
no natural fallible signature without breaking every call site's
fire-and-forget shape.

## Decision

### 1. `absorb_metrics` is the checked path; `absorb_harness_metrics` is new and privileged

`TrialReport::absorb_metrics(&mut self, metrics: Vec<Metric>)` keeps its
signature exactly — no caller needs to change to get the fix — but its
behaviour changes: a metric whose name is harness-owned
(`is_harness_reserved`) is **not** overwritten. It is re-homed to
`agent/<name>` in the same map (scalars stay in `scalars`, counters in
`counters`, histograms in `histograms`), its original key is pushed onto a
new `TrialReport::displaced_metrics: Vec<String>`, and exactly one
`tracing::warn!` fires for the collision, naming the original key and its
new location.

`TrialReport::absorb_harness_metrics(&mut self, metrics: Vec<Metric>)` is
the new, privileged, unchecked path: `map.insert` with no name check.
Calling it asserts that every metric in `metrics` is a measurement the
trial's own harness code made — `core_metrics`, the non-finite-step counter,
the generation count — not something an agent, probe, or environment handed
back. Both `EpisodicTrial` and `GenerationTrial` now call
`absorb_harness_metrics` for their own measurements and `absorb_metrics` for
whatever their agent/probe emits.

The two paths share one private dispatch function (`absorb`, parameterised
by an `Authority` enum) so the checked/unchecked behaviours cannot
independently drift out of sync.

### 2. `agent/` is itself reserved

`is_harness_reserved`'s prefix list includes `agent/` alongside `return/`,
`episode_length/`, and `throughput/`: an agent emitting `agent/foo` directly
is re-homed to `agent/agent/foo`, not inserted at `agent/foo` where it would
be indistinguishable from a genuinely displaced metric.

### 3. Prefix reservation stays broader than the collisions it prevents, and that stays deliberate

`is_harness_reserved` reserves whole prefixes, not just the exact keys
`core_metrics` emits today. This means an agent is forbidden from ever
emitting, say, `return/clipped_mean` — a name `core_metrics` does not use —
purely because it starts with a reserved prefix. That is taken anyway: with
exact-key-only reservation, a *future* harness key would silently begin
colliding with an agent key that was legal the day before. Prefix reservation
converts that failure mode into "the agent's key was already forbidden,"
discoverable at review time. Treat the reserved set as a one-way-ish door:
widening it is cheap; narrowing it re-admits exactly the collision this ADR
closes.

### 4. `#[serde(default)] pub displaced_metrics: Vec<String>` on `TrialReport`

The new field is appended in emission order (a repeated collision on the
same key appears more than once). The `serde(default)` is load-bearing:
`checkpoint.rs` deserializes `BenchmarkReport` from `.ckpt.json` files
written by earlier binaries, and without it every existing checkpoint would
fail to load. Pinned by a test that deserializes a **pre-change** JSON
fixture (no `displaced_metrics` key present at all), so a future edit that
drops `#[serde(default)]` fails the test even if the attribute list still
visually looks complete.

### 5. `Metric::Counter`'s doc no longer claims accumulation

Resolves 0078's recorded tension: `Metric::Counter`'s rustdoc previously
described a count "the harness **may accumulate** across trials," while
`absorb_metrics`/`absorb_harness_metrics` do not accumulate — they insert,
replacing whichever entry occupied that key. The doc was fixed; accumulation
was rejected on four grounds:

- It would fix only `Metric::Counter`. Every displaced key that matters in
  practice — `wall_clock_seconds`, `return/mean`, `generations`,
  `success_rate` — is a `Metric::Scalar`, not a `Counter`.
- It would destroy `absorb_metrics`' idempotence. Nothing requires
  `absorb_metrics` to be called at most once per name per trial; an
  accumulating counter would silently sum across calls that were never meant
  to compose.
- `TrialReport` has no scope wider than one trial for a counter to
  accumulate *across*.
- It would still be silent about the collision itself.

The fix delegates the emission-cadence contract to `MetricsProvider::emit`'s
existing rustdoc ("returns all metrics accumulated since the last call …
drain internal accumulators so repeated calls do not double-count") rather
than restating it on the variant.

### 6. `GenerationTrial` gets the same treatment `EpisodicTrial` had

`GENERATIONS = "generations"` is added to `metrics/core.rs` as a `pub
const`, and `is_harness_reserved`'s exact-key list grows to include it.
`GenerationTrial::run` is rewritten to use the constant instead of the raw
string literal it previously carried, and to call `absorb_harness_metrics`
for its own two measurements and `absorb_metrics` for the probe's. Collision
tests parallel to `EpisodicTrial`'s now cover this site, which had **zero**
test coverage before.

## Consequences

### Positive

- **`absorb_metrics` no longer has a silent-overwrite failure mode** — the
  collision is visible in `displaced_metrics`, in a `tracing::warn!`, and
  the harness's own value is preserved.
- **The safe behaviour is the one every future `Trial` implementor reaches
  for by construction**, not by convention.
- **`GenerationTrial` is no longer a second, undocumented, untested copy of
  the hazard.**
- **`Metric::Counter`'s doc and `absorb`'s actual behaviour now agree.**

### Accepted costs and honest limits — do not soften any of these

- **This is a breaking change on two axes**, given a migration path in the
  CHANGELOG: behavioural (any caller relying on `absorb_metrics` overwriting
  a harness key silently stops working, with no compile error, because the
  signature is unchanged) and API (the new `pub displaced_metrics` field
  breaks external struct-literal construction of `TrialReport`, though no
  in-repo site is affected).
- **Prefix reservation is over-broad by design** (Decision 3) — an agent
  metric can be rejected-by-rename for a name no harness key has ever used.
  This is a deliberate cost, not an oversight to "tighten" later.
- **The degenerate double-collision case drops data.** If an agent emits
  both `<name>` and `agent/<name>` in the same batch, the second write to
  the now-occupied `agent/<name>` slot is dropped rather than overwriting
  the first. Both collisions are still recorded in `displaced_metrics`, so
  the drop is visible even though the value is not recoverable.

### Neutral

- No new production dependency, no new module. One new `pub` method
  (`absorb_harness_metrics`), one new `pub` function (`is_harness_reserved`),
  one new `pub` field (`displaced_metrics`), one new `pub const`
  (`GENERATIONS`), one changed method body (`absorb_metrics`), and doc-only
  changes to `Metric::Counter`.
- `#[non_exhaustive]` on `TrialReport` was considered and declined — the
  type has no `Default` impl and is never struct-literal-constructed outside
  `::new`, so the attribute would buy nothing; the rule in `docs/rules.md`:75
  is discharged by this consideration.
- Test placement is unchanged: in-source `#[cfg(test)] mod tests`, per ADR
  [0012](0012-split-heavy-examples-into-rlevo-examples.md).

## Rejected alternatives

- **(a) Leave `absorb_metrics` last-writer-wins; add an opt-in checked
  variant.** Rejected — see Context, part 3. Closes the defect only for call
  sites an author remembers to opt into.
- **(c) Reject exact-key collisions with an error or a panic.** Rejected —
  see Context, part 3. A colliding metric name is runtime data, not a
  programming error, and `absorb_metrics` has no natural fallible signature.
- **Making `Metric::Counter` actually accumulate across `absorb` calls.**
  Rejected on four grounds — see Decision 5.
- **`#[non_exhaustive]` on `TrialReport`.** Declined — see Consequences →
  Neutral.

## Reopen triggers

Any one of these reopens this ADR:

1. **A proposal to rename `absorb_metrics` and `absorb_harness_metrics`
   back to their original polarity** — "for symmetry," "to keep the common
   path fast," or similar. Context, part 3, is the argument against this.
2. **A third `Trial` implementor adopts the checked and unchecked paths
   inconsistently** — e.g. calls `absorb_harness_metrics` for agent-sourced
   metrics because it is the first method it finds. That would mean the
   naming choice is not actually steering implementors.
3. **`Metric::Counter` genuinely needs cross-trial accumulation** for some
   future consumer. Decision 5's rejection is scoped to "no current need
   exists."
4. **The reserved-prefix set is proposed to narrow** (e.g. to exact-key-only
   reservation). Decision 3's one-way-door argument is the reason to resist
   this.

## References

- Issue **#1118** — "[benchmarks] an agent metric can silently overwrite a
  harness measurement".
- ADR [0078](0078-non-finite-reward-is-counted-and-reported-by-the-benchmarks-harness.md)
  — partially superseded (per Status, above); introduced
  `RETURN_NON_FINITE_STEPS` and the `return/` prefix, which this ADR builds
  on unchanged.
- `docs/rules.md`:75 — the `#[non_exhaustive]`-on-structs rule discharged by
  consideration (Consequences → Neutral).

**Code citations resolve against the working tree at the time of writing.**

- `crates/rlevo-benchmarks/src/report/mod.rs` — `absorb_metrics`,
  `absorb_harness_metrics`, the shared `absorb`/`insert` dispatch, the
  `Authority` enum, and `displaced_metrics` on `TrialReport`. Tests cover
  the namespace split, the `agent/` self-collision, the degenerate
  double-collision drop, and pre-change checkpoint deserialization.
- `crates/rlevo-benchmarks/src/metrics/core.rs` — `is_harness_reserved`,
  `RESERVED_PREFIXES`/`RESERVED_EXACT`, `GENERATIONS`, and the drift test
  that enumerates `core_metrics`' actual return value.
- `crates/rlevo-benchmarks/src/evaluator.rs` — `EpisodicTrial::run` and
  `GenerationTrial::run`; collision coverage for both trial shapes.
- `crates/rlevo-core/src/fitness.rs` — `Metric::Counter`'s corrected
  rustdoc, pointing to `MetricsProvider::emit`'s drain-and-reset contract.
