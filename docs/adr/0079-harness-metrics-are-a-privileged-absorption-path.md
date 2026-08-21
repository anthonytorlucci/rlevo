---
project: rlevo
status: active
type: decision
date: 2026-08-21
tags: [adr, decision, benchmarks, metrics, absorb_metrics, namespace, trial-report, non-exhaustive, issue-1118]
---

# ADR 0079: Harness metrics are a privileged absorption path; `absorb_metrics` becomes checked, safe-by-default

## Status

**Accepted (2026-08-21).** Resolves issue #1118 (`[benchmarks] an agent metric
can silently overwrite a harness measurement`).

**Supersedes, narrowly, one clause each in two places in ADR
[0078](0078-non-finite-reward-is-counted-and-reported-by-the-benchmarks-harness.md):**
the third bullet of its Decision 4 (the sentence beginning "It does not make
the counter unclobberable … a deliberate override is not the hazard being
defended against") and the corresponding clause in its Decision 6 discussion of
the `agent_metrics_are_absorbed_after_the_harness_counter` test. Everything
else in 0078 stays **active and unchanged**: the counting, the
unconditional-including-zero emission, the one-`warn!`-per-trial schedule, and
the no-`rlevo-metrics-registry`-row decision. 0078 itself anticipated this
exact reopening — see Context, part 1, below, which quotes it.

Also resolves 0078's own "out-of-scope observation" on `Metric::Counter`'s doc
tension with `absorb_metrics` (0078, Consequences → Neutral → "An out-of-scope
observation, recorded rather than silently fixed").

## Context

### 1. ADR 0078 named this exact gap and declined to fix it inline

0078 introduced `RETURN_NON_FINITE_STEPS` under the `return/` prefix specifically
so an agent's own, unprefixed `non_finite_steps` metric would not collide with
it by *accident*. It was explicit that the prefix bought only that, in its
Decision 4:

> **It does not make the counter unclobberable.** An agent that emits the
> exact key `return/non_finite_steps` still wins, because last-writer-wins
> plus agent-metrics-absorbed-last says so. That is deliberate — a deliberate
> override is not the hazard being defended against — and both halves are now
> pinned by tests (Decision 6, below): the unprefixed name leaves the counter
> intact, the exact key takes it.

— and its Decision 6, describing the test that pinned the exact-key half of
that claim, closed with the sentence that reopens it here:

> Inverting this assertion so the harness wins would be a change to
> `absorb_metrics`' contract and needs its own ADR, not a test edit.

This is that ADR. The premise — that an agent naming the exact harness key is
a "deliberate override" and therefore not the hazard — does not survive
contact with `Trial` being a public trait. The party that reaches for
`absorb_metrics` next is not necessarily the party that named
`return/non_finite_steps`; it is whoever writes the *next* `Trial`
implementation, and every production call site (`EpisodicTrial`,
`GenerationTrial`) already called it on agent/probe output without knowing —
or needing to know — the harness's own reserved names. "Deliberate" described
the metric's author's intent, not the caller's. A harness measurement getting
silently replaced by an agent that happened to reuse a name is a defect
regardless of whether the agent's author meant it.

### 2. The defect: last-writer-wins, and both trial shapes absorbed agent output last

`TrialReport::absorb_metrics` had one behaviour: `map.insert(name, value)`,
unconditionally, for every metric passed to it. `EpisodicTrial::run` called it
twice, in this order — `core_metrics` first, the per-trial
`RETURN_NON_FINITE_STEPS` counter second, then `self.agent.emit_metrics()`
last. `GenerationTrial::run` (`evaluator.rs`) had the identical shape: its own
`GENERATIONS` / `WALL_CLOCK_SECONDS` scalars first, the probe's metrics last.
An agent or probe emitting `return/mean`, `wall_clock_seconds`, `generations`,
or any other harness-owned name replaced the harness's own measurement with no
error, no warning, and no record. The resulting `TrialReport` was
byte-for-byte indistinguishable from one where the harness had measured that
value itself. A measurement harness whose measurements the thing being
measured can silently overwrite is the wrong default; the silence is what
makes this a bug and not a policy some caller opted into.

`GenerationTrial` is the unnamed second site. Issue #1118 did not mention it —
it used raw string literals (`"generations"`, `"wall_clock_seconds"`) rather
than the `metrics::core` constants, and it had **zero** test coverage for the
collision, unlike `EpisodicTrial`'s already-tested (if insufficiently
defended) counter. It carried the identical hazard and is put on the identical
footing here: a new `GENERATIONS` const in `metrics/core.rs` and collision
tests parallel to `EpisodicTrial`'s.

### 3. The polarity is the decision, not a detail of it

Three shapes were available: (a) leave `absorb_metrics` as last-writer-wins
and add an opt-in checked variant; (b) make `absorb_metrics` checked and add a
privileged unchecked variant for the harness's own use; (c) reject exact-key
collisions outright (return an error / panic).

**Chosen: (b), with the safe behaviour on the name every future caller reaches
for by default.**

`Trial` is a public trait (`crate::evaluator::Trial`). Its implementors are
not enumerable — `EpisodicTrial` and `GenerationTrial` are the two the
workspace ships, not the two that will ever exist. The generator of future
`absorb_metrics` call sites is therefore *external* to this crate. Had the fix
taken shape (a) — leave `absorb_metrics` as the fast, obvious,
last-writer-wins method and introduce a differently-named checked method
beside it — the defect would have closed for exactly the two call sites this
PR touches and stayed armed, unremarked, for every `Trial` implementor written
from today onward. The next implementor reaches for the method with the
obvious name; if the obvious name is the unsafe one, the fix accomplishes
nothing beyond moving the goalposts for this one PR.

So the method named `absorb_metrics` — the one with no qualifier, the one a
skim of the type's public API turns up first — is now the checked path, and
the privileged path is named for what it costs to use it correctly:
`absorb_harness_metrics`, which states in its own name the obligation its
caller is taking on (see Decision 1, below). Getting this polarity backwards
is the single easiest way for a well-intentioned future edit to reopen this
issue silently, which is why it is being written down as the load-bearing
choice rather than left implicit in the diff. **A later reader proposing to
swap the two names back "for symmetry" or "to keep the fast path fast" is
re-introducing this defect — treat the request as a reopen trigger (below),
not a simplification.**

Option (c) — reject the collision outright — was considered and rejected
because a metric name colliding with a harness key is exactly the
runtime-data shape `docs/rules.md`'s Error Handling section reserves for
`Result`, not a panic, and `absorb_metrics` has no natural fallible signature
without breaking every existing call site's fire-and-forget shape (`Vec<Metric>`
in, nothing out). Re-homing plus a warning gives the same visibility a hard
error would, without turning a benign, expected occurrence — a probe or agent
happening to reuse a common word like `mean` — into a trial-ending failure.

## Decision

### 1. `absorb_metrics` is the checked path; `absorb_harness_metrics` is new and privileged

`TrialReport::absorb_metrics(&mut self, metrics: Vec<Metric>)` keeps its
signature exactly — no caller needs to change to get the fix — but its
behaviour changes: a metric whose name is harness-owned
(`is_harness_reserved`, new in this ADR — see Decision 2 and Decision 3,
below) is **not** overwritten. It is
re-homed to `agent/<name>` in the same map (scalars stay in `scalars`,
counters in `counters`, histograms in `histograms`), its original key is
pushed onto a new `TrialReport::displaced_metrics: Vec<String>`, and exactly
one `tracing::warn!` fires for the collision, naming the original key and its
new location.

`TrialReport::absorb_harness_metrics(&mut self, metrics: Vec<Metric>)` is the
new, privileged, unchecked path: `map.insert` with no name check, exactly
`absorb_metrics`' old behaviour. Calling it is an assertion by the caller that
every metric in `metrics` is a measurement the trial's own harness code made —
`core_metrics`, the non-finite-step counter, the generation count — not
something an agent, probe, or environment handed back. Both `EpisodicTrial`
and `GenerationTrial` now call `absorb_harness_metrics` for their own
measurements and `absorb_metrics` for whatever their agent/probe emits.

The two paths share one private dispatch function (`absorb`, parameterised by
an `Authority` enum with `Harness` / `Agent` variants) so the kind-match and
insertion logic exist exactly once; the checked/unchecked behaviours cannot
independently drift out of sync with each other the way two hand-duplicated
methods could.

### 2. `agent/` is itself reserved

`is_harness_reserved`'s prefix list, introduced by this ADR, includes
`agent/` alongside `return/`, `episode_length/`, and `throughput/`: an agent
emitting `agent/foo` directly through the checked path is re-homed to
`agent/agent/foo`, not inserted at `agent/foo` where it would be
indistinguishable from a genuinely displaced metric. `agent/` stays a
harness-managed landing zone that only re-homing itself writes into, never an
agent's own choice of name.

### 3. Prefix reservation stays broader than the collisions it prevents, and that stays deliberate

`is_harness_reserved`, introduced by this ADR, reserves whole prefixes
(`return/`, `episode_length/`, `throughput/`, `agent/`), not just the exact
keys `core_metrics` and the non-finite counter happen to emit today. This
means an agent is forbidden from ever emitting, say, `return/clipped_mean` — a
name `core_metrics` does not use and never has — purely because it starts with
a reserved prefix. That is taken anyway: with exact-key-only reservation, a
*future* harness key (a new `core_metrics` statistic, say) would silently
begin colliding with an agent key that was legal the day before, changing an
existing agent's report with no code change on that agent's side. Prefix
reservation converts that failure mode into "the agent's key was already
forbidden," discoverable at review time rather than at the moment a harness
key happens to be added. Treat the reserved set as a one-way-ish door:
widening it is cheap (worst case, a previously-legal agent name starts
appearing under `agent/<name>`, which this ADR's `displaced_metrics` field
makes visible rather than silent); narrowing it re-admits exactly the
collision this ADR closes.

### 4. `#[serde(default)] pub displaced_metrics: Vec<String>` on `TrialReport`

The new field is `#[cfg_attr(feature = "json", serde(default))] pub
displaced_metrics: Vec<String>`, appended in emission order (a repeated
collision on the same key within one trial appears more than once). The
`serde(default)` is load-bearing, not decorative: `checkpoint.rs` deserializes
`BenchmarkReport` from `.ckpt.json` files written by earlier binaries to
resume an interrupted suite, and without it every existing checkpoint written
before this change would fail to load — a resumed run is a real, expected
operational path, not a hypothetical one. This is pinned by a test that
deserializes a **pre-change** JSON fixture (no `displaced_metrics` key present
at all) rather than merely asserting the attribute's presence in the source,
so a future edit that quietly drops `#[serde(default)]` fails the test even if
the attribute list still visually looks complete.

### 5. `#[non_exhaustive]` was considered for `TrialReport` and rejected

Adding a `pub` field to a struct is exactly the shape the Struct Field
Encapsulation section of `docs/rules.md` (`docs/rules.md`:75) singles out
`#[non_exhaustive]` against being misapplied to: the rule states plainly that
the attribute is for enums, never structs, that on a struct it forbids
cross-crate `..Default::default()` while buying no validation guarantee, and
that applying it to a struct requires its own ADR. This is that
consideration, recorded so the next reader does not have to re-derive it: the
rule's stated rationale for keeping structs exhaustive — preserving the
`Config { ..Default::default() }` tuning idiom — does not engage here, because
`TrialReport` has no `Default` impl and is never struct-literal-constructed
outside `TrialReport::new`; every production and test call site already goes
through the constructor or field access, not a bare struct literal with
`..`. So `#[non_exhaustive]` would cost nothing functionally today. It is
declined anyway: the value it would buy — reserving the right to add a future
field without that being a breaking change for a hypothetical external
struct-literal constructor — is too small to justify carrying an exception
clause on what is otherwise a bug-fix PR, and the door is a fully general one
that any future ADR touching `TrialReport` can still open on its own merits.
This is a considered non-decision, not an oversight; `TrialReport` remains
exhaustive.

### 6. `GenerationTrial` gets the same treatment `EpisodicTrial` had

`GENERATIONS = "generations"` is added to `metrics/core.rs` as a `pub const`
beside `RETURN_NON_FINITE_STEPS` and the rest, and `is_harness_reserved`'s
exact-key list grows to include it (`SUCCESS_RATE`, `WALL_CLOCK_SECONDS`,
`GENERATIONS`). `GenerationTrial::run` is rewritten to use the constant
instead of the raw string literal it previously carried, and to call
`absorb_harness_metrics` for its own two measurements and `absorb_metrics` for
the probe's. Collision tests parallel to `EpisodicTrial`'s
(`agent_metric_cannot_collide_with_the_namespaced_counter` and its sibling)
now cover this site, closing the "zero test coverage" gap noted in Context,
part 2, above, without waiting for issue #1118 to have named it.

### 7. `Metric::Counter`'s doc no longer claims accumulation

Resolves ADR 0078's own recorded tension (its "out-of-scope observation"):
`Metric::Counter`'s rustdoc (`rlevo-core/src/fitness.rs`) previously described
a count "the harness **may accumulate** across trials," while
`absorb_metrics`/`absorb_harness_metrics` do not accumulate anything — they
insert, replacing whichever entry (harness's own or previously re-homed
agent's) occupied that key. Two resolutions were available: make
`absorb_metrics` actually accumulate counters across trials, or fix the doc.
**The doc was fixed.** Accumulation was rejected on four independent grounds,
any one of which is sufficient on its own:

- It would fix only `Metric::Counter`. Every displaced key that matters in
  practice — `wall_clock_seconds`, `return/mean`, `generations`,
  `success_rate` — is a `Metric::Scalar`, not a `Counter`; an accumulating
  `absorb` would still silently overwrite every one of those on collision, so
  the change would not touch the defect this ADR exists to fix.
- It would destroy `absorb_metrics`' idempotence. Nothing in the current
  design requires `absorb_metrics`/`absorb_harness_metrics` to be called at
  most once per name per trial — `core_metrics` and the non-finite counter are
  each absorbed in one call today, but nothing prevents (and the checked path
  now actively relies on) calling `absorb_metrics` more than once, e.g. once
  per agent-emitted batch. An accumulating counter would silently sum across
  calls that were never meant to compose.
- `TrialReport` has no scope wider than one trial for a counter to accumulate
  *across* — the rustdoc's "across trials" language describes a scope that
  does not exist on this type. Accumulation, if it were ever wanted, belongs
  at `BenchmarkReport` (which does span trials), not at `TrialReport`'s
  absorption chokepoint.
- It would still be silent. Even a correctly-scoped accumulating counter gives
  no signal that two writers are contending for the same name; it just
  changes what the wrong-attribution value looks like.

The fix lives at trait level, on `Metric::Counter` itself, and delegates the
actual emission-cadence contract to `MetricsProvider::emit`'s existing
rustdoc ("returns all metrics accumulated since the last call … drain
internal accumulators so repeated calls do not double-count") rather than
restating it a second time on the variant. The first pass at this fix wrote a
new sentence on `Metric::Counter` asserting it represents a "running total to
date" — which directly contradicts `emit`'s drain-and-reset rule four lines
below it in the same file. Two docs stating the emission-cadence contract
independently is exactly how they drifted apart the first time; the corrected
version states it once, at `emit`, and has `Metric::Counter` point there
rather than repeat it.

## Consequences

### Positive

- **`absorb_metrics` no longer has a silent-overwrite failure mode for the
  common case** — the collision is visible in `displaced_metrics`, in a
  `tracing::warn!`, and the harness's own value is preserved rather than
  replaced.
- **The safe behaviour is the one every future `Trial` implementor reaches for
  by construction**, not by convention or documentation — Context, part 3,
  above.
- **`GenerationTrial` is no longer a second, undocumented, untested copy of
  the hazard** this ADR fixes for `EpisodicTrial`.
- **`Metric::Counter`'s doc and `absorb`'s actual behaviour now agree**,
  closing a discrepancy ADR 0078 recorded but explicitly declined to resolve
  inline.

### Accepted costs and honest limits — do not soften any of these

- **This is a breaking change on two independent axes**, detailed and given a
  migration path in the CHANGELOG entry this ADR accompanies: behavioural
  (any caller relying on `absorb_metrics` overwriting a harness key silently
  stops working, with no compile error, because the signature is unchanged)
  and API (the new `pub displaced_metrics` field breaks external struct-literal
  construction of `TrialReport`, though no in-repo site is affected because all
  in-repo construction goes through `TrialReport::new`).
- **Prefix reservation is over-broad by design** (Decision 3, above) — an
  agent metric can be rejected-by-rename for a name no harness key has ever
  used or will ever use, purely for sharing a reserved prefix. This is a
  known, deliberate cost of choosing safety over precision at the
  namespace boundary, not an oversight to "tighten" later without re-deriving
  Decision 3's argument first.
- **The degenerate double-collision case drops data.** If an agent emits both
  `<name>` and `agent/<name>` in the same batch, the second write to the
  now-occupied `agent/<name>` slot is dropped rather than overwriting the
  first — a property of this ADR's re-homing insert (Decision 1, above), not
  inherited from anywhere. Both collisions are still recorded in
  `displaced_metrics`, so the drop is visible even though the value is not
  recoverable.
- **`#[non_exhaustive]` remains off `TrialReport`** (Decision 5) — a future
  field addition to this type is still a breaking change for any external
  struct-literal constructor, exactly as it always has been. This ADR does not
  change that posture; it only records that the option was considered and why
  it was declined here specifically.

### Neutral

- No new production dependency, no new module. One new `pub` method
  (`absorb_harness_metrics`), one new `pub` function (`is_harness_reserved`),
  one new `pub` field (`displaced_metrics`), one new `pub const`
  (`GENERATIONS`), one changed method body (`absorb_metrics`), and doc-only
  changes to `Metric::Counter`.
- Test placement is unchanged: in-source `#[cfg(test)] mod tests`, per ADR
  [0012](0012-split-heavy-examples-into-rlevo-examples.md).

## Rejected alternatives

- **(a) Leave `absorb_metrics` last-writer-wins; add an opt-in checked
  variant beside it.** Rejected as the core decision of this ADR — Context,
  part 3. Closes the defect only for call sites an author remembers to opt
  into, and `Trial`'s public-trait status means the population of future call
  sites is unbounded and unreviewable from inside this crate.
- **(c) Reject exact-key collisions with an error or a panic.** Rejected: a
  colliding metric name is runtime data (an agent's free-form choice of
  metric name), not a programming error, so `docs/rules.md`'s Error Handling
  section reserves this shape for `Result`, and neither `absorb_metrics` nor
  `absorb_harness_metrics` has a natural fallible signature without breaking
  every call site's fire-and-forget shape. Re-homing plus a warning gives
  equivalent visibility without turning an expected, benign name reuse into a
  trial-ending failure.
- **Making `Metric::Counter` actually accumulate across `absorb` calls**, to
  resolve the doc/behaviour tension ADR 0078 flagged. Rejected on four
  grounds — Decision 7, above: it fixes only `Counter` while every collision
  that matters in practice is a `Scalar`; it breaks `absorb`'s idempotence
  under repeated calls; `TrialReport` has no cross-trial scope for
  "across trials" to mean anything; and it would still be silent about the
  collision itself. The doc was corrected instead.
- **`#[non_exhaustive]` on `TrialReport`.** Considered and declined — Decision
  5, above. `docs/rules.md`:75 requires its own ADR for a struct; this
  section is that consideration, and the conclusion is "not now, on this PR."

## Reopen triggers

Any one of these reopens this ADR:

1. **A proposal to rename `absorb_metrics` and `absorb_harness_metrics` back
   to their original polarity** — "for symmetry," "to keep the common path
   fast," or similar. Context, part 3, is the argument against this; treat the
   proposal as a request to re-litigate that argument, not as a style
   preference to accept on its own.
2. **A third `Trial` implementor adopts the checked and unchecked paths
   inconsistently** — e.g. a new trial type that calls `absorb_harness_metrics`
   for agent-sourced metrics because it is the first method it finds, or
   skips the split entirely. That would mean the naming choice in Decision 1
   is not actually steering implementors the way Context, part 3, argues it
   will, and the naming (or the trait's documentation) needs revisiting.
3. **`Metric::Counter` genuinely needs cross-trial accumulation** for some
   future consumer. Decision 7's rejection is scoped to "no current need
   exists"; if one appears, the emission point (likely `BenchmarkReport`, not
   `TrialReport`) needs deciding fresh rather than retrofitted onto
   `absorb`.
4. **The reserved-prefix set is proposed to narrow** (e.g. to exact-key-only
   reservation) for precision. Decision 3's one-way-door argument is the
   reason to resist this without first solving the "future key silently
   collides with an already-legal agent key" problem it exists to prevent.

## References

- Issue **#1118** — "[benchmarks] an agent metric can silently overwrite a
  harness measurement". Resolved by making `absorb_metrics` checked and
  introducing `absorb_harness_metrics` as the privileged path.
- ADR [0078](0078-non-finite-reward-is-counted-and-reported-by-the-benchmarks-harness.md)
  — partially superseded (Decision 4's third bullet and the corresponding
  clause in Decision 6, per Status, above); introduced `RETURN_NON_FINITE_STEPS`
  and the `return/` prefix, which this ADR builds on unchanged. `is_harness_reserved`,
  the reserved prefix/exact-key set, and the re-homing design are new in this
  ADR (Decision 1–3, above) — 0078's `return/` prefix bought only
  accidental-collision avoidance and said explicitly that it did not make the
  counter unclobberable. 0078's own text named this exact gap and deferred it
  to a future ADR (Context, part 1, quotes the passage). Its "out-of-scope
  observation" on `Metric::Counter`'s doc tension is resolved by Decision 7.
- `docs/rules.md`:75 — the `#[non_exhaustive]`-on-structs rule this ADR
  discharges by consideration (Decision 5) rather than by adoption.
- ADR [0055](0055-config-invariant-enforcement-allocation.md) — the struct
  field encapsulation framework the Naming Conventions section of
  `docs/rules.md` codifies, referenced in
  Decision 5's reasoning about why `TrialReport`'s `pub` fields are not
  themselves the concern here (`TrialReport` is not a `*Config`/`*State`/
  `*Params`/`*Genome`, so that framework's allocation does not directly bind
  it either way; the `#[non_exhaustive]` question is separate from field
  privacy).

**Code citations resolve against the working tree at the time of writing.**

- `crates/rlevo-benchmarks/src/report/mod.rs` — `TrialReport::absorb_metrics`
  (checked path), `TrialReport::absorb_harness_metrics` (privileged path), the
  shared private `absorb`/`insert` dispatch, the `Authority` enum, and
  `displaced_metrics` on `TrialReport`. The in-source test module covers the
  namespace split, the `agent/` self-collision, the degenerate
  double-collision drop, and pre-change checkpoint deserialization.
- `crates/rlevo-benchmarks/src/metrics/core.rs` — `is_harness_reserved`, its
  `RESERVED_PREFIXES`/`RESERVED_EXACT` lists (now including `GENERATIONS`),
  and the drift test that enumerates `core_metrics`' actual return value
  rather than a hand-written key list.
- `crates/rlevo-benchmarks/src/evaluator.rs` — `EpisodicTrial::run` and
  `GenerationTrial::run`, both now calling `absorb_harness_metrics` for their
  own measurements and `absorb_metrics` for agent/probe output; the
  in-source test module's collision coverage for both trial shapes.
- `crates/rlevo-core/src/fitness.rs` — `Metric::Counter`'s corrected rustdoc,
  pointing to `MetricsProvider::emit`'s drain-and-reset contract rather than
  restating an emission-cadence rule that contradicted it.
