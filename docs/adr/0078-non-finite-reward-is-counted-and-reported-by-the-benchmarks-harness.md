---
project: rlevo
status: active
type: decision
date: 2026-08-21
tags: [adr, decision, numerical-stability, nan, reward, benchmarks, evaluator, metrics, telemetry, rlevo-core, rlevo-benchmarks, issue-1115]
---

# ADR 0078: A non-finite reward is counted and reported by the benchmarks harness; the `Reward` trait states the obligation and enforces nothing

## Status

**Accepted (2026-08-21).** Resolves issue #1115 (`[core] the Reward trait states
no finiteness obligation`). **Supersedes nothing.** No public item is removed or
renamed; no signature changes; no healthy-step numerics change.

**Scopes ADR [0034](0034-fitness-hygiene-chokepoint-convention.md)'s
`debug_assert!` permission** to the shape that permission was written for — see
this ADR's own Context, part 2, below. 0034 stays `active` and its
sanitize-to-a-sentinel convention in `rlevo-evolution` is untouched.
**Reconciles that permission with ADR
[0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md)'s
by-name rejection of `debug_assert!`**, which claimed to settle the general
question and did not know 0034 had already answered it the other way.
**Extends ADR [0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md)
and ADR
[0070](0070-avg-score-transits-non-finite-scores-the-hardened-mean-is-additive.md)**
to the fourth reward-ingestion site, the one neither of them covers.
**Follows ADR
[0072](0072-loss-skips-are-counted-per-site-and-surfaced-as-a-metric.md) as the
precedent for counting-and-surfacing** — and explicitly *not* as precedent for
the transport (this ADR's own Decision 5, below).

**Chosen shape.** Issue #1115 offered three options for the `debug_assert!` in
`EpisodicTrial::run`: (a) keep it, (b) replace it with a warn plus a counter,
(c) remove it outright. **Option (b).** In `rlevo-core`, documentation only: a
`# Finiteness` section on the `Reward` trait stating the obligation and
enumerating what each layer actually does when it is broken, a pointer on
`ScalarReward`, and a cross-reference on `Environment::step`'s `RewardType`.
**No runtime check is added to `rlevo-core`.** In `rlevo-benchmarks`, the
`debug_assert!` is deleted; accumulation stays raw and unconditional; the trial
carries a `non_finite_steps: u64`; one summary `tracing::warn!` fires per
affected trial; and a `Metric::Counter` keyed `return/non_finite_steps`
(`RETURN_NON_FINITE_STEPS`) is absorbed into the `TrialReport`
**unconditionally, including as `0`**.

## Context

### 1. The assert made the harness's artefact a function of the build profile

This argument stands on its own and depends on no prior ADR. It is the reason
option (a) is not merely disfavoured but *undocumentable*.

The deleted `debug_assert!` sat in `EpisodicTrial::run`'s step loop, immediately
after the reward was read. Its three behaviours:

- **Release.** The assertion compiles away. The trial completes and reports a
  `return/mean` of `NaN` or $\pm\infty$.
- **Debug.** It panics. `run_trials` catches the unwind
  (`evaluator.rs:231`) and the handler at `:237-242` constructs a **fresh**
  `TrialReport` with `errored = true` — discarding **every episode the trial
  had already completed**. A trial 900 episodes into a 1000-episode budget
  reports zero episodes.
- **Debug plus `fail_fast`.** `:246` additionally sets the abort flag, and the
  rest of the suite never runs.

So the same suite, the same seeds, and the same environment produce three
different artefacts depending on how the binary was compiled. **A measurement
harness whose output shape depends on the optimisation profile is broken on its
own terms**, independently of anything one believes about NaN. Worse, the debug
behaviour is not "a stricter version of the release behaviour": it is *lossier*.
Release reports a poisoned number; debug reports nothing at all about episodes
that were measured cleanly before the bad step arrived.

Writing option (a) down as an accepted decision would mean codifying the stance
"in debug builds this harness loses completed measurements and may abort the
suite". That is not a stance anyone would ratify on purpose, which is what
disqualifies it.

Note also what the assert did **not** do: it did not guard anything. The
accumulation `total_reward += reward` ran unconditionally, before and after,
in both profiles. In debug it aborted the trial *after* the poisoned value had
already been added; in release it did nothing whatsoever. It was neither a
tripwire over a sanitiser nor a protection — it was a profile-dependent
detonator.

### 2. Two prior ADRs are in play, not one, and they disagree

Issue #1115 reads the evaluator's original comment — which cited "ADR 0034's
convention" for preferring `debug_assert!` over a hard `assert!` — as a
fabricated or mistaken citation, on the ground that ADR 0065 rejected
`debug_assert!` by name. **The citation was not fabricated.** Both ADRs say what
they say, and reconciling them is this ADR's job.

**ADR 0034:153-157**, in its own Rejected alternatives, permits it verbatim:

> Rejected: `docs/rules.md`'s Error Handling section forbids panicking on
> user-supplied runtime data, and a landscape returning `0/0` *is* runtime
> data. Sanitize to a sentinel instead; a `debug_assert!` tripwire is
> acceptable, a hard `assert!` is not.

**ADR 0065's Rejected alternatives** rejects it by name and claims generality:

> ADR 0056's Runs-unconditionally-in-release decision (Decision 4) already
> settled the general question for this codebase … A release no-op protects
> nothing in the runs where protection matters most.

**The resolution is a scoping distinction, not a reversal.** Read 0034's
sentence with its own first clause attached: *sanitize to a sentinel instead; a
`debug_assert!` tripwire is acceptable.* 0034 grants the permission at a site
where **sanitize-to-a-sentinel is the primary mechanism** and the tripwire is a
redundant second net over it — the value is already made safe in every build,
and the assertion only shortens the distance between a developer and a bug
during development. Remove the assertion there and the sanitiser still holds.

At the evaluator's reward read there was **no sentinel and no sanitiser**. The
tripwire was the *only* behaviour attached to the condition, so removing it
changed the program, and keeping it made release and debug disagree. That is
outside the shape 0034's clause describes, and saying so scopes 0034 without
contradicting it. Stated as a rule a later reader can apply: **0034's
`debug_assert!` permission holds only where the release path is already
correct without it.**

Equally, 0065's supporting argument is narrower than its claim. "The host read
already exists" and "a release no-op protects nothing in the runs where
protection matters most" are **guard** arguments: they assume the construct in
question decides whether a value is admitted. The evaluator's assert was not a
guard (Context, part 1, above) — it protected nothing even in debug. 0065's
conclusion is right here, but by a different route, and pretending its stated
reasoning transfers unchanged would leave the next reader unable to tell which
sites the ruling actually covers.

### 3. Both signs corrupt the report, and neither announces itself

#1115's failure-mode list is `NaN`-only. That understates the problem and gets
one entry backwards. Executing the downstream arithmetic in
`metrics/core.rs` for trial returns $[1, 5, X, 20, 100]$ with
`success_threshold = 10`:

| $X$      | mean   | median | std    | min | max      | success rate |
| -------- | ------ | ------ | ------ | --- | -------- | ------------ |
| finite ($9$) | 27.000 | 9.000  | 41.419 | 1   | 100      | 0.400        |
| `NaN`    | `NaN`  | 20.000 | `NaN`  | 1   | 100      | 0.400        |
| $+\infty$ | $\infty$ | 20.000 | `NaN`  | 1   | $\infty$ | **0.600**    |

Three things this table says that a `NaN`-only account does not:

- **`NaN` is scored as a *failure*** the agent may not have earned, because
  `NaN >= threshold` is false. **$+\infty$ is scored as a *success* the agent
  certainly did not earn**, and poisons `max` besides. The fabricated success
  is exactly the category ADR
  [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md)'s
  no-fabrication rule exists to rule out — the harness reports an outcome that
  did not occur.
- **`min` and `max` survive by accident, not by design.** `f64::min` and
  `f64::max` drop a `NaN` operand, so the order statistics keep reporting
  plausible finite values while the mean is destroyed. In the $+\infty$ row even
  that partial rescue fails on one side, since $+\infty$ orders normally.
- **The median shifts silently, in both rows.** $9.0 \rightarrow 20.0$, with no
  `NaN` and no infinity in the output to hint that anything happened. A report
  can therefore present a poisoned aggregate beside entirely plausible order
  statistics, and no single field is a reliable tell.

That last point is the operative argument for a **counter** rather than a bare
warning: a consumer reading only the report needs one field that says
unambiguously how much of the trial was affected, and it must not be inferable
only by cross-checking five aggregates against each other.

The counter counts **steps, not episodes**, and the distinction is load-bearing.
The episode list records only that an episode's return came out non-finite; a
$+\infty$ step and a $-\infty$ step in the same episode yield a `NaN` return
from two bad steps, and ten bad steps in one episode look identical to one. The
per-step count is not derivable from `TrialReport::episodes`, which is why it
is stored rather than computed.

### 4. What could not be reused, and why

`rlevo-reinforcement-learning`'s `FiniteRewardGuard` (ADR 0065) is the obvious
candidate for reuse and is unavailable on two independent grounds. It is
`pub(crate)` (`shared.rs:719`), and `rlevo-benchmarks` **does not depend on
`rlevo-reinforcement-learning`** at all (`crates/rlevo-benchmarks/Cargo.toml`) —
by design, not by omission: `docs/rules.md:22` defines the crate as the
*paradigm-neutral* evaluation harness, and an edge to the RL crate would make
the harness that measures both paradigms structurally partial to one.

This is worth recording because it forecloses the obvious objection. **ADR
0065's "six inline copies" rejection does not apply here.** That rejection
targets intra-crate copy-paste across a family of sibling call sites, and its
evidence was concrete: two of six `remember` sites had already been created by
pasting an unguarded copy forward. `rlevo-benchmarks` has **exactly one**
reward-ingestion site, in one function, with no siblings to drift from. The
duplication hazard 0065 legislates against does not exist on this side of the
workspace boundary.

## Decision

### 1. `rlevo-core` documents the obligation and enforces nothing

The `Reward` trait gains a `# Finiteness` section stating that implementors
**should** yield finite values, that the framework enforces this nowhere, and —
the part that makes the section worth having — **what each layer actually does
when the obligation is broken**, which is three different things:

- dropped at replay ingestion, counted, warned on a decade schedule (ADR 0065);
- transited deliberately by the reported episode return and `avg_score`
  (ADR 0065's bookkeeping decision, Decision 4, and ADR 0070);
- counted and reported by the benchmarks harness (this ADR).

`ScalarReward` gains a short pointer to that section, noting that neither the
tuple form nor `new` validates. `Environment::step`'s `RewardType` gains a
one-line cross-reference, because that associated type is where an implementor
is standing when the obligation becomes theirs.

**No runtime check is added anywhere in `rlevo-core`, and none is deferred.**
ADR 0065's Rejected alternatives closed validating `ScalarReward::new`
*outright* on three grounds that are unchanged here — the field is `pub` by
documented intent, `new` is `const fn`, and `Reward` is a **trait**, so a check
on one concrete implementor closes nothing for an environment that ships its
own. The `# Finiteness` section is the honest form of that closure: state the
obligation where the trait is, and put enforcement where the value has already
been erased to `f32`.

### 2. Accumulation stays raw and unconditional

`total_reward += reward` runs for every step regardless of finiteness. A
non-finite return is a **true statement** about a run whose environment emitted
a non-finite reward. Dropping the term would manufacture a plausible return the
run never earned — ADR 0061's no-fabrication rule, and the same call ADR 0065's
bookkeeping decision (Decision 4) made for the RL training loop's
`episode_reward`. The two paradigms now agree on this, which matters because a
user comparing an RL run's reported return against a harness trial's must not
have to know which accumulator sanitised.

### 3. One summary `warn!` per affected trial — not the decade schedule

The `warn!` fires **once per trial**, after the episode loop, only when
`non_finite_steps > 0`, carrying the environment name and the total.

ADR 0065's escalating decade schedule (1, 10, 100, …) is deliberately **not**
adopted, and the reason is structural rather than stylistic. That schedule
exists because `remember` is a hot path **with no aggregation point** — there is
no later moment at which the guard could summarise, so the log must throttle
itself in place. `EpisodicTrial::run` *has* an aggregation point: the end of the
trial, where the total is already known exactly. Given that, a decade schedule
would be strictly worse on its own terms: each trial builds its own counter, so
the schedule would **restart at 1 in every trial** and emit *more* lines than
one summary per affected trial, while conveying less (the running partial
totals, rather than the final one).

Log volume is therefore bounded at **one line per affected trial**, and zero
lines on a clean suite.

### 4. A `return/non_finite_steps` counter, emitted unconditionally including zero

`RETURN_NON_FINITE_STEPS = "return/non_finite_steps"` is a `pub const` in
`metrics/core.rs`, beside the other metric-key constants, and the evaluator
absorbs a `Metric::Counter` under it immediately after the `core_metrics` call.

Three properties, each deliberate:

- **Emitted unconditionally, including as `0`.** A consumer must be able to
  distinguish "this trial had no non-finite rewards" from "this harness version
  does not report the quantity". A key that appears only on failure cannot
  express the former, and its absence is then ambiguous forever.
- **`return/`-prefixed — against *accidental* collision only.** `absorb_metrics`
  (`report/mod.rs:105-107`) **overwrites** same-name entries, and agent metrics
  — whose names are free-form by `Metric`'s own contract — are absorbed *after*
  this counter. An unnamespaced `non_finite_steps` is exactly the name an agent
  author would independently reach for, so it would be silently clobberable by
  an agent that never intended to touch a harness metric; the prefix makes that
  accidental collision structurally unlikely, and the two keys then coexist.
  **It does not make the counter unclobberable.** An agent that emits the exact
  key `return/non_finite_steps` still wins, because last-writer-wins plus
  agent-metrics-absorbed-last says so. That is deliberate — a deliberate
  override is not the hazard being defended against — and both halves are now
  pinned by tests (Decision 6, below): the unprefixed name leaves the counter
  intact, the exact key takes it.
- **Not emitted by `core_metrics`.** `core_metrics` sees only per-episode
  aggregates, and the quantity is per **step** (Context, part 3, above). Placing
  it there would require either passing the count in as a parameter, or
  re-deriving it from returns — which is precisely the derivation that does not
  exist.

### 5. No `rlevo-metrics-registry` row — and ADR 0072 is precedent for the counting, not the transport

Stated explicitly so a later reader does not "complete" this change by wiring it
to the wrong pipe.

`rlevo-metrics-registry` classifies **`tracing` field names emitted by
algorithms** (`rlevo-metrics-registry/src/lib.rs:21-27`: "Add one
`MetricDescriptor` row … and emit a matching `tracing::info!` field from the
algorithm"). Those rows drive the live TUI sparklines, the on-disk recording
stream, and the report client's panel grouping — all of which read the tracing
channel. This counter travels the **`TrialReport` channel** instead: it is
absorbed into `TrialReport::counters` by the harness, not emitted as a tracing
field by an algorithm. A registry row would describe a field nothing emits.

ADR 0072 is cited here as precedent for the *shape* of the response — count the
occurrences, surface the count programmatically rather than relying on a log —
and **not** for its transport. 0072's `skipped_updates` is an RL algorithm
metric and correctly took a registry row; this one is a harness observation
about an environment and correctly does not.

### 6. Six profile-independent tests replace the `#[should_panic]` one

The `#[cfg(debug_assertions)] #[should_panic]` test is deleted, not adapted. Its
existence was itself the defect's clearest symptom: a test that can only run in
one profile is an admission that the behaviour differs between them.

Six replacements, none gated. The first two pin the counter's values, the next
two pin its **name** against agent metrics, and the last two pin the log volume
Decision 3 bounds:

- `non_finite_reward_is_counted_not_asserted` — drives a suite whose environment
  emits $+\infty$ for 2 episodes $\times$ 3 steps and pins **values**, not
  merely absence of panic: the trial is not `errored`, both episodes survive
  into the report at their full length, each return is $+\infty$, `return/mean`
  is $+\infty$, and the counter reads exactly **6**. A version that silently
  dropped the term would leave the returns finite; a version that counted
  episodes rather than steps would report 2. `max_steps` bounds the loop, so a
  done-detection defect fails the length assertion rather than hanging.
- `non_finite_counter_is_zero_on_a_clean_trial` — pins the unconditional-`0`
  clause from Decision 4, above.
- `agent_metric_cannot_collide_with_the_namespaced_counter` — an agent emits a
  `Metric::Counter` named `non_finite_steps` (unprefixed, the name a harness
  author would plausibly reach for) with count `999`, against a trial whose own
  count is `8`. Afterwards `return/non_finite_steps` still reads `8` and
  `non_finite_steps` reads `999`: the two keys coexist. This — and only this —
  is the property the `return/` prefix buys, and the test kills the mutant that
  drops the prefix. The three values `0`, `8`, `999` are pairwise distinct so no
  assertion can pass by coincidence.
- `agent_metrics_are_absorbed_after_the_harness_counter` — the complement, and
  the reason the bullet above is worded as narrowly as it is. An agent naming
  the **exact** key `return/non_finite_steps` with `999` wins: the counter reads
  `999`, not `8`. `absorb_metrics` is last-writer-wins and agent metrics are
  absorbed after the harness counter, so this is emission order, pinned. It
  kills the mutant that swaps the two absorb calls. Inverting this assertion so
  the harness wins would be a change to `absorb_metrics`' contract and needs its
  own ADR, not a test edit.
- `non_finite_reward_warns_once_per_trial_with_env_and_total` — captures the
  tracing stream for a poisoned trial (2 episodes $\times$ 3 steps) and asserts
  **exactly one** WARN event, carrying `env = "poisoned-env"` and
  `non_finite_steps = 6`, with the counter reporting the same `6`. One line per
  affected *trial*, not per affected step, and the log total and the report
  total agree.
- `a_clean_trial_emits_no_non_finite_warning` — the other half of Decision 3's
  bound: **zero** WARN events on a clean trial, while the counter is still
  present as `0`.

$+\infty$ rather than `NaN` is used for the poisoned case on purpose: it is the
sign whose harm (a fabricated success, a poisoned `max`) the issue missed, and
an `is_infinite() && is_sign_positive()` assertion is exact where a `NaN`
assertion is satisfied by any of several distinct corruptions.

## Consequences

### Positive

- **The harness's artefact no longer depends on `cfg(debug_assertions)`.** One
  behaviour, one report shape, both profiles.
- **No completed episode is ever discarded by this condition.** The debug path
  that threw away up to a full trial's measurements is gone, as is the
  `fail_fast` suite abort it could trigger.
- **The affected-step count is exact and machine-readable**, and reports the
  quantity that cannot be recovered from the episode list.
- **The `Reward` trait now says what the framework does.** Four sites did four
  different things and none of them was written down at the trait; three of the
  four are now enumerated in one place, with the fourth named as out of scope
  (below) rather than left to be inferred.

### Accepted costs and honest limits — do not soften any of these

- **"Visible in the report" is true for JSON only.** `TrialReport::counters` is
  serialised by `JsonReporter`, which writes the whole `BenchmarkReport`. It is
  **not** rendered by the TUI — `TuiEvent::TrialEnd` is an explicit no-op in
  `tui/runner.rs:277` — and **not** by the static-HTML report, which is built
  from the `record` tracing layer rather than from `TrialReport`. The
  `LoggingReporter` prints `scalars.len()` and does not mention counters at all.
  The TUI's channel for this failure is the `warn!` line, which its log layer
  does capture. Anyone reading this ADR as "the counter shows up in the report"
  should read it as "in the JSON report".
- **This is the first `Metric::Counter` emitted anywhere in the workspace.**
  `Metric::Counter` (`rlevo-core/src/fitness.rs:35-37`) and
  `TrialReport::counters` were declared, plumbed, and entirely unexercised
  surface until now. The plumbing was verified end to end by the tests in
  Decision 6, above, but there is no prior consumer whose expectations this
  counter is known to meet.
- **Coverage is `ScalarReward` only.** `EpisodicTrial` binds
  `RewardType = ScalarReward` (ADR 0077's post-`BenchEnv` shape). An environment
  shipping its own `Reward` implementation is covered by the trait's
  documentation and by nothing else in the harness.
- **The `# Finiteness` contract is deliberately absent from the Core Trait
  Invariants table** (`docs/rules.md:102-116`). That table lists invariants that
  "must never be violated by implementations", and the `Reward` row in it
  (`zero()` is the additive identity) is an algebraic law. Finiteness is a
  SHOULD that the framework explicitly licenses violating — Decision 2, above,
  transits the violation on purpose. Listing it beside the additive-identity row
  would make it read as enforced, which is the misunderstanding this whole ADR
  exists to prevent.
- **The counter's `+=` is unsaturated.** It is a reported statistic, not a
  predicate — the same standing ADR 0072's first reopen trigger gives
  `skipped`, and it inherits the same fence (below).

### Neutral

- No new production dependency, no new module, no new public type. One new
  `pub const` (`RETURN_NON_FINITE_STEPS`) and three doc sections. The one
  dependency added is a **dev**-dependency (below).
- `rlevo-core` gains no code — the diff there is entirely `///` comments.
- Test placement is unchanged: in-source `#[cfg(test)] mod tests`, per ADR
  [0012](0012-split-heavy-examples-into-rlevo-examples.md).
- **The two `warn!` tests carry a tracing-capture constraint, and it is a
  process-global one.** `rlevo-benchmarks` cannot use the workspace helper,
  `rlevo_test_support::capture::FieldCapture`: `rlevo-test-support` pulls in
  `burn` and `rlevo-environments`, which this crate's stub-based test module
  exists precisely to avoid, and `FieldCapture` collects a single integer field,
  so it could not assert the `env` name the `warn!` carries. The tests therefore
  carry a local `CaptureLayer` over a `tracing-subscriber` dev-dependency
  (`registry` + `std` only — the same narrowed feature set the optional
  `[dependencies]` entry already uses, so a test build links nothing a
  `tui`/`record` build does not).

  That layer is installed **process-globally**, once, via `Once` +
  `set_global_default`, writing into a thread-local sink — not scoped with
  `with_default`. The reason is a real defect, not tidiness: `tracing` caches
  each callsite's `Interest` **globally**, computed from whichever dispatcher
  the thread that *first* reaches the callsite happens to have. Three of these
  tests reach this `warn!` from a **rayon worker with no subscriber installed**;
  if one of those registers the callsite first, `Interest::never()` is cached
  for the whole process, a scoped capture then silently observes nothing, and
  `a_clean_trial_emits_no_non_finite_warning` — whose assertion is *absence* of
  a WARN — passes for the wrong reason. This was **observed**, not theorised: 1
  failure in ~26 runs of an earlier `with_default` version, i.e. surfacing as a
  rare false pass. A global default closes it from both sides, since
  `Dispatch::new` recomputes interest for already-registered callsites and
  `get_default` falls back to the global on subscriber-less threads. **Any
  future test that can emit this `warn!` must call `install_capture_subscriber()`
  first**, including tests that assert nothing about logging — otherwise it
  reopens the race for the others.

### An out-of-scope observation, recorded rather than silently fixed

`Metric::Counter`'s own rustdoc (`rlevo-core/src/fitness.rs:35-37`) describes a
count "that the harness **may accumulate** across trials". `absorb_metrics`
(`report/mod.rs:105-107`) does not accumulate: it **overwrites** same-name
entries, and its own doc says so. The two statements are in tension. Nothing in
this change depends on the resolution — this counter is absorbed exactly once
per trial, from a per-trial accumulator, so overwrite and accumulate coincide —
and the discrepancy predates it. It is written down here so the next author of a
counter does not discover it the hard way.

## Out of scope, deliberately

- **The on-policy rollout path** — `ppo/rollout.rs` into `compute_gae` — is
  **untouched and remains unguarded**, exactly as ADR 0065 left it, and is
  tracked as **#1042**. Naming it explicitly matters here because this ADR adds
  a *general* `# Finiteness` section to the `Reward` trait, and a reader could
  reasonably infer from its generality that all four sites are now handled.
  They are not: three are, and this is the fourth. The reason is 0065's, not
  convenience — a rollout is a contiguous positional trajectory, so the
  "drop the tuple" semantic does not transfer, and the real options (truncation
  boundary with a bootstrap value; discard the rollout) are a separate design
  decision.
- **Any runtime finiteness check in `rlevo-core`.** Closed, not deferred, on ADR
  0065's trait-erasure argument (Decision 1, above).
- **Rendering the counter in the TUI or the HTML report.** Those consume a
  different channel (Consequences, above). Bridging `TrialReport` counters into
  the tracing/registry pipeline is a transport decision affecting every metric,
  not a rider on this one.

## Rejected alternatives

- **Option (a): keep the `debug_assert!`.** Rejected as *undocumentable*, not
  merely as suboptimal — Context, part 1, above. The stance it would codify is
  "in debug builds this harness loses completed measurements and may abort the
  suite". Note that this rejection rests on the site, not on `debug_assert!` as
  a construct: ADR 0034's permission survives, scoped (Context, part 2).
- **Option (c): remove the assert with no replacement.** This fixes the
  profile-dependence and nothing else. The report would still present a poisoned
  aggregate beside plausible-looking order statistics with no field that says
  so (Context, part 3), and the shifted median would remain invisible. Removal
  alone trades a loud wrong behaviour for a silent one.
- **A hard `assert!` or `panic!`.** `docs/rules.md`'s Error Handling section:
  panics are for programming errors, never for user-supplied runtime data, and
  an environment's reward is exactly that. Rejected identically by ADR 0034 and
  ADR 0065.
- **Sanitise the step to `0.0`, or filter it from the accumulator.** Both
  fabricate a return the run did not produce, and the filtered version is the
  more dangerous of the two because the result *looks* finite. ADR 0061's
  no-fabrication rule; ADR 0065's identical call for the training loop.
- **Mark the trial `errored`.** Considered, because it is what the debug build
  effectively did. Rejected: `errored` means the trial could not be completed —
  a reset or step failure — and the harness has recovery semantics attached to
  it, including `fail_fast`. A trial that ran every episode to termination and
  measured a non-finite reward *completed*; what it measured is the finding.
  Conflating an observation about the environment with a failure of the harness
  would make `errored` unusable as a signal and would resurrect the suite abort.
- **Reuse `FiniteRewardGuard` from `rlevo-reinforcement-learning`.** Not
  available: `pub(crate)`, and the dependency edge does not exist and must not
  (Context, part 4). ADR 0065's "six inline copies" objection does not transfer
  — one site, no siblings.
- **The escalating decade `warn!` schedule.** Decision 3, above: it would
  restart per trial and emit strictly more lines than one summary, for less
  information, at a site that has the aggregation point the schedule exists to
  substitute for.
- **A `rlevo-metrics-registry` row.** Decision 5, above: wrong channel. The
  registry classifies tracing fields emitted by algorithms.
- **A row in `docs/rules.md`'s Core Trait Invariants table.** Consequences,
  above: it is a SHOULD the framework licenses violating, and the table is for
  invariants that must never be violated.
- **Deriving the count from `TrialReport::episodes` instead of storing it.**
  Impossible, not merely inconvenient: a poisoned return records that an episode
  was affected, never how many of its steps were (Context, part 3).

## Reopen triggers

Any one of these reopens this ADR:

1. **A second consumer of `TrialReport::counters` appears** — the TUI, the HTML
   report, or a `BenchmarkReport`-level aggregation. At that point the
   JSON-only limit above stops being an honest scope statement and becomes a
   transport decision that should be made once, for all counters, rather than
   inherited from this one.
2. **Anything begins to *branch* on `non_finite_steps`** — an abort, a retry, a
   trial re-run, or a suite-level pass/fail gate. The unsaturated `+=` is
   acceptable for a reported statistic and is not acceptable for a control-flow
   predicate; ADR 0072's first reopen trigger draws the same line for
   `skipped`.
3. **The harness gains a second reward-ingestion site**, or `EpisodicTrial`
   stops binding `RewardType = ScalarReward`. Context, part 4's "one site, no
   siblings" argument against a shared guard is exactly what a second site
   falsifies, and ADR 0065's copy-paste-omission evidence would then apply
   in full.
4. **`Metric::Counter`'s accumulate-versus-overwrite discrepancy is resolved in
   favour of accumulation.** This counter is written once per trial from a
   per-trial accumulator, so the two semantics currently coincide; under real
   accumulation across trials they would not, and the emission point would need
   re-deciding.

## References

- Issue **#1115** — "[core] the `Reward` trait states no finiteness
  obligation". Resolved as its option (b).
- ADR [0034](0034-fitness-hygiene-chokepoint-convention.md), **lines 153-157** —
  the `debug_assert!`-is-acceptable clause, **scoped** here to sites where
  sanitize-to-a-sentinel is the primary mechanism (this ADR's own Context, part
  2). Not reversed; `rlevo-evolution`'s four fitness-hygiene chokepoints are
  untouched.
- ADR [0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md) — the
  by-name `debug_assert!` rejection reconciled with 0034 above; the
  `ScalarReward::new` closure adopted unchanged in this ADR's own Decision 1;
  the raw-accumulation call adopted in Decision 2; the decade schedule
  deliberately **not** adopted in Decision 3; the "six inline copies" rejection
  explicitly held not to apply (Context, part 4); and **#1042** carried forward
  as out of scope.
- ADR [0070](0070-avg-score-transits-non-finite-scores-the-hardened-mean-is-additive.md)
  — the RL-side statement that the reported score transits a non-finite value
  rather than sanitising it, which Decision 2 matches on the harness side.
- ADR [0072](0072-loss-skips-are-counted-per-site-and-surfaced-as-a-metric.md) —
  precedent for counting-and-surfacing rather than logging alone, and for
  "1% and 40% must not be byte-identical after the first line". **Not**
  precedent for the transport (this ADR's own Decision 5). Its first reopen
  trigger is the model for this ADR's own second.
- ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) —
  the no-fabrication rule, leaned on twice: against sanitising the step, and to
  name the $+\infty$ success in Context, part 3 for what it is.
- ADR [0077](0077-remove-benchenv.md) — the post-`BenchEnv` evaluator shape in
  which `EpisodicTrial` binds `Environment<…, RewardType = ScalarReward>`
  directly, which is why the counter's coverage is `ScalarReward`-only.
- ADR [0015](0015-shared-typed-metric-registry-crate.md) — the registry whose
  scope Decision 5 declines.

**Code citations resolve against the working tree at the time of writing.**

- `crates/rlevo-benchmarks/src/evaluator.rs`:
  - `:231` — `catch_unwind` around the trial; `:237-242` — the handler that
    builds a **fresh** `TrialReport` with `errored = true`, discarding completed
    episodes; `:246` — the `fail_fast` abort flag. Together, the debug-build
    consequences in Context, part 1.
  - `:384` — the per-trial `non_finite_steps: u64`.
  - `:410-429` — the reward read, the replacement comment recording why the
    `debug_assert!` is gone, and the unconditional `non_finite_steps += 1` at
    `:428` beside the unconditional `total_reward += reward`.
  - `:462-469` — the one-per-trial summary `warn!`, guarded by
    `non_finite_steps > 0`.
  - `:472-477` — `core_metrics` absorbed; `:478-481` — the `Metric::Counter`,
    absorbed unconditionally.
  - The in-source test module — Decision 6's six tests: `:1088`
    `non_finite_reward_is_counted_not_asserted`; `:1143`
    `non_finite_counter_is_zero_on_a_clean_trial`; `:1191`
    `agent_metric_cannot_collide_with_the_namespaced_counter`; `:1219`
    `agent_metrics_are_absorbed_after_the_harness_counter`; `:1236`
    `non_finite_reward_warns_once_per_trial_with_env_and_total`; `:1268`
    `a_clean_trial_emits_no_non_finite_warning`. Their fixtures: `:1159`
    `HARNESS_COUNT = 8` and `:1163` `poisoned_trial_with_agent_metrics`.
  - The tracing-capture scaffolding recorded in Consequences: `:788`
    `CaptureLayer` (and its note on why `FieldCapture` is unavailable); `:830`
    `install_capture_subscriber`, the `Once` + `set_global_default` helper whose
    rustdoc records the observed `Interest`-caching flake; `:841` `capturing`;
    `:857` `EventVisitor`; `:891` `run_trial_capturing_events`.
- `crates/rlevo-benchmarks/src/metrics/core.rs:35-51` —
  `RETURN_NON_FINITE_STEPS` and
  its rustdoc (steps-not-episodes, the `return/` prefix rationale); `mean`,
  `median`, `std_dev`, `min`, `max`, and the `r >= threshold` success-rate
  filter that produce the table in Context, part 3.
- `crates/rlevo-benchmarks/src/report/mod.rs:105-107` — `absorb_metrics`
  overwrites same-name entries; `:75` — `pub counters: BTreeMap<String, u64>`.
- `crates/rlevo-benchmarks/src/reporter/json.rs`, `.../tui.rs`,
  `crates/rlevo-benchmarks/src/tui/runner.rs:277`,
  `crates/rlevo-benchmarks/src/reporter/logging.rs` — the JSON-only reach of
  `counters` recorded in Consequences.
- `crates/rlevo-core/src/base.rs` — the `Reward` trait's `# Finiteness` section;
  `crates/rlevo-core/src/reward.rs` — `ScalarReward`'s pointer;
  `crates/rlevo-core/src/environment.rs` — `RewardType`'s cross-reference.
- `crates/rlevo-core/src/fitness.rs:35-37` — `Metric::Counter` and its
  "may accumulate across trials" doc, the tension recorded above.
- `crates/rlevo-reinforcement-learning/src/algorithms/shared.rs:719` —
  `FiniteRewardGuard` is `pub(crate)`.
- `crates/rlevo-benchmarks/Cargo.toml` — no `rlevo-reinforcement-learning`
  dependency; `docs/rules.md:22` — the paradigm-neutrality that is why. Its
  `[dev-dependencies]` also carries the `tracing-subscriber` entry (`registry`,
  `std`) the capture layer needs, and the comment recording why
  `rlevo-test-support` is not used instead.
- `docs/rules.md:102-116` — the Core Trait Invariants table this contract stays
  out of.
