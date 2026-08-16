---
project: rlevo
status: active
type: decision
date: 2026-08-09
tags: [adr, decision, numerical-stability, nan, loss, telemetry, metrics-registry, ppo, ppg, dqn, c51, qrdqn, sac, ddpg, td3]
---

# ADR 0072: Loss skips are counted per site, warned on a decade schedule, and surfaced as one canonical metric

## Status

**Accepted (2026-08-09).** Resolves the remainder of issue #346 ("[rl] add a
skip counter to `AgentStats` for the finite-loss guard").

**Partially supersedes ADR
[0056](0056-non-finite-loss-skip-and-warn-guard.md): its Decision 1, which set
the guard's type shape, and its Decision 3, which set the surfacing clause and
the one-shot `warn!` latch.** Following the partial-supersession precedent of
ADR [0033](0033-share-splitmix64-mixer-across-core-and-evolution.md) (which
partially supersedes ADR 0004's decision point 6, "keep the local `splitmix64`
mixer") and ADR [0062](0062-grid-layout-fidelity-and-no-dead-rng.md) (which
partially supersedes ADR 0029's dead-`_rng` grid carve-out): the superseded
clauses are restated here in their new form so authority is not split, and ADR
0056 is **not edited** (`docs/rules.md:649`; `docs/adr/README.md:3-6`). ADR 0056
stays `active`. Specifically in force, unchanged:

- **0056's guard-placement decision (Decision 2)** — the guard sits before
  `backward()`, never inside `Slot::step_with`.
- **0056's unconditional-execution decision (Decision 4)** — it runs
  unconditionally in release; no `debug_assert`/config gate. Rejected by name
  for the fifth time (this ADR's own Rejected alternatives section, below).
- **0056's site-count decision (Decision 5)** — 8 agents, one guard per loss
  site, **17 sites**. Re-verified against the tree, site by site, in this
  ADR's own Context, part 3 below.
- **0056's Decision 3 *skip* semantics** — the skip fires on **every**
  non-finite occurrence, and the return value is never gated on the warning
  schedule. Reaffirmed **verbatim** in this ADR's own Decision 1, because that
  is exactly the clause a reader skimming this ADR's headline ("the latch is
  gone") is most likely to break.
- **The whole of 0056's Consequences & honest limits section**, including the
  loss-level-proxy limit and the deferral of finite-loss→NaN-gradient to #328.
  Carried forward and restated in this ADR's own Consequences section, below.

**Builds on ADR [0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md)**
— the decade `warn!` schedule and the `{count, next_threshold, label}` guard
shape are adopted wholesale rather than re-derived. **Complements ADR
[0059](0059-target-update-cadence-counts-gradient-updates.md)'s counter-advance
decision (Decision 4)** — that clause makes the update counters advance
*unconditionally, including on a skip*; this ADR ships the counter that makes
the difference between attempts and applications recoverable, and changes
nothing about 0059.

**Chosen shape.** `FiniteLossGuard` becomes
`{ skipped: u64, next_warn_at: u64, label: &'static str }`. Its `warn!` fires at
skips 1, 10, 100, … carrying the running total. `skipped()` is `pub(crate)` and
**not** test-gated; 0056's Decision 1 `#[cfg(test)] warning_fired()` is
deleted, not reimplemented. Each of the 8 agents exposes one `#[must_use]` public accessor per
loss site — 17 in total, named `skipped_[<site>_]updates` — plus, on the five
multi-site agents, one aggregate `skipped_updates()`. One canonical metric row,
`skipped_updates`, ships in `rlevo-metrics-registry`. **Nothing is added to
`AgentStats`.**

## Context

### 1. The clause being repealed, and why it did not survive contact

ADR 0056's surfacing-and-latch decision (Decision 3) closed with:

> the `warn!`, not a NaN metric, is the surfacing mechanism, matching #184 which
> added no stats field.

That sentence decided two things at once — *the log is the channel* and *there is
no programmatic channel* — and both are wrong for this failure mode, for one
reason: **after the first log line, a run discarding 1% of its updates and a run
discarding 40% of them are byte-identical.** The operator learns that the failure
mode exists and nothing about its rate. A loud divergence quietly becomes a slow
one; #346 was filed to prevent exactly that.

This is not a hindsight objection. **The crate already stated the consequence
against itself**, in `FiniteRewardGuard`'s own rustdoc at
`shared.rs:650-654` (pre-change coordinates, resolving against `6636dc8`):

> …so the run silently loses gradient updates for as long as the poisoned
> transition is resident, and because the loss guard's `warn!` is one-shot, only
> the very first skip is ever logged.

A guard documenting a sibling guard's blind spot, in-tree, in a doc comment
shipped to users, is the strongest possible evidence that the blind spot is real
and was already understood.

### 2. The historical irony, recorded rather than quietly fixed

ADR 0065 gave the *reward* guard a counter and a decade schedule, and — because
it was landing beside a loss guard that latched — had to justify the asymmetry.
It did so in `FiniteRewardGuard`'s rustdoc (`shared.rs:664-669`, pre-change):

> `FiniteLossGuard` latches its warning once, and that is right for a *skipped
> gradient step*: the failure is self-healing, the next minibatch can recover,
> and repeating the line adds nothing.

**That justification is what this ADR overturns.** The error in it is precise and
worth naming, because it is an easy one to make again: "self-healing" is a claim
about the *weights*, and it is true — 0056's guard-placement decision
(Decision 2) guarantees the parameters are never touched, so the next
minibatch genuinely can recover. But the quantity #346 cares about is not
weight health, it is **throughput**. A skipped update is a step of the run's
budget that bought nothing, and skipped throughput does not heal; it
accumulates. 0065's own reward-guard schedule decision (Decision 3) reached
the right general rule one paragraph later — *magnitude, not occurrence, is
the operational fact* — and then failed to apply it to the guard sitting
directly above it in the same file.

The rustdoc is corrected in the same PR as this ADR, because leaving it would
have the crate documenting two contradictory schedules as though both were
current. That is a code fix, not an ADR edit: ADR 0065's Decision 3 (the reward
guard's own schedule) is untouched and remains correct; only its incidental
characterisation of the *loss* guard is now historical.

### 3. Seventeen sites, re-verified rather than inherited

The site enumeration from 0056's site-count decision (Decision 5) was re-derived from the tree
(`rg 'FiniteLossGuard::new' crates/rlevo-reinforcement-learning/src/algorithms/`,
excluding `shared.rs`'s own test fixtures) rather than copied from the ADR:

| agent | sites | labels |
|---|---|---|
| PPO | 2 | `ppo/policy_loss`, `ppo/value_loss` |
| PPG | 4 | `ppg/policy_loss`, `ppg/value_loss`, `ppg/aux_main_value_loss`, `ppg/aux_total_loss` |
| SAC | 3 | `sac/critic_1`, `sac/critic_2`, `sac/actor` |
| TD3 | 3 | `td3/critic_1`, `td3/critic_2`, `td3/actor` |
| DDPG | 2 | `ddpg/critic`, `ddpg/actor` |
| DQN | 1 | `dqn/loss` |
| C51 | 1 | `c51/loss` |
| QR-DQN | 1 | `qrdqn/loss` |

17 total, matching 0056's site-count decision (Decision 5) exactly. **Five
agents are multi-site, three are single-site**, and that split — not the
count — is what this ADR's own per-site/aggregate decision (Decision 3, below)
turns on. SAC's $\alpha$ guard (#184, `sac_alpha.rs`) is a different type in a
different module and is untouched here, exactly as 0056's Consequences &
honest limits section ruled.

### 4. The issue's proposed destination is a scope error, not merely an ADR collision

#346's title asks for the counter on `AgentStats`. It is the natural place to
look — `AgentStats` is where an RL run's numbers live — and it is wrong on
**units**, before any question of ADR collision arises.

`AgentStats::record(entry: T)` is called **once per episode**, at episode end
(`metrics.rs:126-143`). Every accessor on the type is scoped to an episode or a
window of episodes. A loss skip is **per gradient update and per loss site**: a
single episode of PPO can contain dozens of updates across two sites, and a
single update can skip at one site while succeeding at the other. There is no
episode-shaped quantity to record. Putting the counter there would either force
an arbitrary per-episode aggregation the caller cannot undo, or bolt a
per-update counter onto an episode-scoped type and leave the mismatch for a later
reader to trip over.

**And separately**, ADR 0071's twelfth-accessor fence on `AgentStats`
(its first reopen trigger) fences the destination:

> **A twelfth accessor of any kind on `AgentStats`** — deliberately tighter than
> 0070's own first reopen trigger, "a third windowed mean". At eleven … the flat
> surface is at the limit of what naming can carry.

Both facts point the same way, and the order matters: the scope mismatch is the
*reason*, the trigger is the *fence*. This ADR's own per-site/aggregate
decision (Decision 3, below) records the outcome explicitly so that ADR 0071's
trigger is neither fired nor weakened by this change.

### 5. What ADR 0059 left legible-in-principle and illegible-in-practice

ADR 0059's counter-advance decision (Decision 4) requires the target-update counters (`gradient_updates` on the
DQN family, `critic_updates` on SAC/DDPG/TD3) to advance **unconditionally**,
including on a 0056 skip — because gating the cadence on run health would make
the target-update rhythm drift exactly when stability matters most. That is
correct and this ADR does not touch it.

But it leaves a reader holding a counter that means "updates *attempted*" while
the name says "updates". Today there is no way to recover how many of those
attempts moved the weights. After this ADR there is: $\text{applied} = \text{attempts} -
\text{skipped}$, both terms readable from the agent. **The counter is what makes ADR
0059's counter-advance decision (Decision 4) legible rather than a source of
confusion** — the unconditional
advance stops being a silent approximation and becomes one half of a stated
identity.

## Decision

### 1. The decade schedule replaces the one-shot latch — and the skip semantics are reaffirmed verbatim

`FiniteLossGuard` becomes:

```rust
pub(crate) struct FiniteLossGuard {
    skipped: u64,
    next_warn_at: u64,
    label: &'static str,
}
```

The `warn!` fires when `skipped` reaches 1, 10, 100, 1000, …, carrying the
running total and the site label each time. The mechanics of ADR 0065's
reward-guard schedule decision (Decision 3) are adopted **wholesale**, not
re-derived: same field shape, same
`saturating_mul(10)` advance, same "report the total, not just the event"
message. The two guards now sit adjacent in one file with one schedule between
them, which is the second reason to adopt rather than invent — a reader who
learns one has learned both.

Log volume is bounded at **~7 lines per site** over any realistic run (a
10M-skip site emits 8). With 17 sites, the absolute worst case across the whole
workspace is under 140 lines, and no single run holds more than 4 sites.

**Reaffirmed verbatim from ADR 0056's surfacing-and-latch decision (Decision 3),
because this is the clause most at risk from the change above:** the **skip
re-fires on every non-finite
occurrence**. The return value of `check` is **never** gated on `next_warn_at` or
on `skipped`. A run that emits NaN every step must be protected every step;
latching the skip would be a correctness defect, not a cosmetic one.

**And this is exactly why the schedule lives on the log and not on the return.**
The separation is the whole design: `check` computes a *protection decision*
(`loss.is_finite()`, always, unconditionally) and, as a side effect, maintains a
*telemetry channel* whose emission rate is throttled. Fusing the two — "only skip
when we also warn" — is the single-line mutation that turns this guard into a
liability, so the split is stated in the ADR, in the struct's field docs
(`next_warn_at`: "schedules the `warn!` only — never the skip"), and in `check`'s
own `CRITICAL:` rustdoc.

### 2. A count accessor per loss site — on the guard, and on the agent

**On the guard:** `FiniteLossGuard::skipped(&self) -> u64` is `pub(crate)` and
**not** `#[cfg(test)]`. The `#[cfg(test)] warning_fired()` accessor from ADR
0056's type-shape decision (Decision 1) is **deleted, not reimplemented under
a new name**. This mirrors the `dropped()` accessor from ADR 0065's own
Decision 1 (its home-and-shape decision), and for the reason 0065 gave: it is
operationally useful outside tests, being the term each agent's public
accessor reads.

**On the agents:** one `#[must_use]` public accessor per loss site, **17 in
total**, named `skipped_[<site>_]updates` — `skipped_critic_1_updates`,
`skipped_policy_updates`, `skipped_aux_total_updates`, and so on; on the three
single-site agents the site segment is absent and the accessor is plain
`skipped_updates()`. The name says the unit (`updates`) because the value is a
count of updates, not of losses or of steps.

**The deliberate complement to ADR 0059, stated as an identity:**

$$\text{applied} = \text{attempts} - \text{skipped}$$

where `attempts` is 0059's `gradient_updates` / `critic_updates` and `skipped` is
this counter. **The unconditional advance from ADR 0059's counter-advance
decision (Decision 4) is not changed, not weakened, and not to be "fixed" in
light of this counter.** 0059 already rejected gating the cadence on applied
updates, by name, in its own Alternatives Considered section ("it makes the
cadence a function of run health"). This ADR supplies the *second term* so the
first can stay unconditional and still be interpretable — the counter exists
precisely so nobody is tempted to reopen 0059.

### 3. One aggregate per multi-site agent; none on the single-site agents; and it lives on the agent, not on `AgentStats`

`skipped_updates()` — an aggregate over all of that agent's guards — ships on
**DDPG, TD3, SAC, PPO, and PPG**. It does **not** ship on DQN, C51, or QR-DQN:
those have one loss site, so their per-site accessor from this ADR's own
Decision 2 (the count-accessor-per-site decision, above) **already is** the
aggregate, and adding a second name for the same `u64` would be pure
duplication with a synchronisation obligation attached.

The aggregate is the **sole source** of the emitted metric. No train loop sums
guards itself. The failure mode this closes is concrete and has precedent in this
codebase: ADR 0065's Context section recorded that two of six `remember` sites were
missing their guard because an unguarded copy was pasted forward. A future sixth
guard on PPG, added to the agent but forgotten in one of two summation sites,
would produce a metric that under-reports on some runs and not others — the
worst kind of telemetry bug, because the number is plausible. With one aggregate
on the agent, adding a guard field and forgetting to include it is a change in
one file, next to the field, reviewable in one diff.

**Recorded explicitly: the aggregate lives on the agent, and *not* on
`AgentStats`.** Therefore **ADR 0071's twelfth-accessor fence (its first reopen
trigger) does not fire, and is not weakened** — `AgentStats` still carries
exactly eleven accessors after this change, and the next addition to it still
owes the bundled-struct argument ADR 0071 demands.

The reason is the one given in this ADR's own Context, part 4 above, and it is
a *units* reason first: `AgentStats` is episode-scoped (`record(entry: T)` on
episode end), while a loss skip is per-update and per-loss-site. #346's
proposed destination is a **scope error**, not merely an ADR collision — which
matters, because "blocked by a reopen trigger" invites someone to argue the
trigger away, and there is nothing to argue away here. Even with unlimited
accessor budget on `AgentStats`, the counter would not belong there.

### 4. One canonical metric row, `skipped_updates` — repealing 0056's "not a NaN metric" surfacing clause

One row in `crates/rlevo-metrics-registry/src/lib.rs`:

| name | kind | cadence | unit | trend |
|---|---|---|---|---|
| `skipped_updates` | `Rl` | `PerUpdate` | `"updates"` | `LowerIsBetter` |

Placed **immediately after `n_updates`** in the RL region. The two are literal
complements — attempts beside skips — so a report renders their panels adjacent,
and appending at the end of the region leaves every existing row's relative order
unchanged, so panel layout is otherwise byte-identical.

This is the direct repeal of the "the `warn!`, **not a NaN metric**, is the
surfacing mechanism" clause in 0056's surfacing-and-latch decision (Decision
3). Note what the
repealed clause actually conflated: 0056 was rejecting *poisoning an existing
mean with a NaN* — which this ADR agrees with, and which stays rejected
(Decision 3's exclusion of skipped values from epoch-mean accumulators is
untouched). It then over-generalised from
"do not corrupt a metric" to "do not add a metric". A separate, always-finite
counter is not the thing 0056 was ruling out.

**Seventeen rows — one per site — were considered and rejected on four
independent grounds, each sufficient:**

(a) **It breaches the registry's naming convention.** The registry enforces
`names_are_unique`, so 17 rows would have to be agent-prefixed
(`sac_critic_1_skipped_updates`, …). Every existing RL row is named **by role,
not by agent** — `td_loss`, `qf1_loss`, `alpha`, `n_updates` — and that
convention exists for a purpose: **one report template serves all agents**. The
first agent-prefixed family in the table breaks that property for everything
downstream of it.

(b) **Thirteen of seventeen panels would be permanently absent from any given
run.** No run trains more than one algorithm, so at most 4 rows (PPG's) are ever
populated. A table where 76% of a family is structurally dead on every run is not
a metric family; it is a schema for a report nobody renders.

(c) **Per-site attribution already has a strictly better channel.** The `warn!`
carries a structured `site` field *and a diagnostic message* naming likely
causes. For the debugging act that per-site attribution serves — "which loss is
diverging, and what should I try?" — a log line with a label and a remedy beats a
nameless panel that says a number went up.

(d) **This ADR's own per-site accessors (Decision 2, above) are already the
programmatic per-site API.** 17 rows would be a second, weaker copy of a
surface that already exists, kept in sync with it by hand. Two representations
of one fact, one of which the compiler checks and one of which it does not.

**On #184 as precedent.** ADR 0056's surfacing-and-latch decision (Decision 3)
leaned on #184 ("which added no stats field") to justify log-only surfacing.
That analogy is the weaker one here, and saying so is part of the repeal: #184
guarded a **single closed-form scalar step** (SAC's $\alpha$ optimizer) where
occurrence and rate nearly coincide — one site, one hand-rolled update. This
guard spans **17 sites across 8 agents**, and the quantity whose *magnitude* is
the operational fact is the aggregate over them. This is the same reasoning
ADR 0065's reward-guard schedule decision (Decision 3) and its own Literature
section used to reject a latch for the reward guard: where the failure can
recur at a rate, the rate is the observable. #184 remains good precedent for
#184's own guard, which is why it is untouched.

### 5. Cumulative, not per-interval

`skipped_updates` reports a **running total since agent construction**, not the
count since the last emission.

A per-interval value is a function of `log_every` — a **logging** parameter, with
no effect on training. Two runs of the identical config with different
`log_every` would then produce numerically incomparable series for the same
underlying run. A wire contract whose values depend on the observer's sampling
rate is disqualified on that ground alone.

Cumulative also matches the table's existing counters — `n_updates`,
`env_steps_sampled` — so the RL region stays uniform in this respect, and a
consumer does not have to know per-row which convention applies.

The asymmetry is decisive: **first-differencing a cumulative series recovers the
interval view exactly**, at any grouping the consumer chooses. Recovering a total
from interval samples requires knowing every window width and that none were
dropped, which no consumer can establish from the data.

## Consequences

### Positive

- **Rate is observable, in both channels.** The log conveys magnitude at ~7 lines
  per site; the accessors and the metric convey it exactly. 1% and 40% are no
  longer byte-identical after the first line — the defect #346 names.
- **ADR 0059's counter-advance decision (Decision 4) becomes legible.**
  $\text{applied} = \text{attempts} - \text{skipped}$ is recoverable without parsing logs, which
  removes the standing temptation to reopen 0059 and gate the cadence on
  applied updates.
- **The two guards now share one schedule and one shape**, adjacent in one file.
  A reader learns the pattern once.
- **The aggregate has exactly one definition per agent** (this ADR's own
  per-site/aggregate decision, Decision 3, above), so a future sixth guard
  site cannot be silently omitted from one train loop and present in another.

### Negative / accepted costs — do not soften any of these

- **19 agent-side test sites change assertion shape.** Every
  `assert!(guard.warning_fired())` / `assert!(!guard.warning_fired())` becomes an
  exact-value assertion on `skipped()`. This is a **strengthening, not a
  translation**: `!warning_fired()` was satisfiable by a latch that had *already
  fired* on an earlier call in the same test, so the old assertion could pass
  over a guard that had skipped an arbitrary number of updates. `== 0` cannot.
  (The 21 `warning_fired` references in `sac_alpha.rs` belong to #184's separate
  guard and are untouched.)
- **`FiniteRewardGuard`'s rustdoc rationale is now historically wrong** and is
  corrected in the same PR (this ADR's own Context, part 2, above — the
  historical-irony section). Without that, the crate would ship two
  contradictory statements about which schedule the loss guard uses, one of them
  in a doc comment users read. The correction is to code; ADR 0065 is not edited.
- **Per-site log volume rises from $\le 1$ line to $\le {\sim}7$** over a run. Bounded, and the
  point — but it is more output than before, and a run with several diverging
  sites will say so several times.
- **22 new public methods across 8 agents** (17 per-site + 5 aggregate). They are
  additive **inherent** methods; nothing is removed, no signature changes, and
  **no public struct gains a field**. That last constraint is why the metric is
  read *from the agent* at emission time rather than threaded through
  `PpoUpdateStats` / `PpgUpdateStats` — adding a field to a public stats struct
  is a wider break than adding an inherent method, and buys nothing the aggregate
  accessor does not already provide.
- **The guard stays `pub(crate)`.** External users get the agent accessors and
  the metric row, not the guard. If someone wants per-guard access from outside
  the crate, that is a new decision, not an oversight.
- **Honest limit, carried forward verbatim in force from ADR 0056's
  Consequences & honest limits section: this remains a loss-level *proxy*.** It
  counts **skips, not NaN
  origins**. A finite loss producing a NaN gradient is caught one step late, once
  poisoned weights make the *next* loss non-finite, and the counter attributes
  the skip to the site that observed it rather than the one that caused it.
  Gradient-origin NaN and grad-norm handling remain **#328's** territory. A
  rising `skipped_updates` says "updates are being discarded"; it does not say
  why.

### Neutral

- No new dependency, no new public type, no new module.
- `AgentStats` is **byte-for-byte untouched** — no field, no accessor, no change
  to `record`.
- One registry row is added at the end of the RL region, so no existing row's
  index or relative order changes.
- Test placement is unchanged: in-source unit tests under `#[cfg(test)] mod
  tests`, per ADR [0012](0012-split-heavy-examples-into-rlevo-examples.md).

## Rejected alternatives

- **A field/accessor on `AgentStats`** — the issue title's literal proposal.
  Rejected on **units first**: `AgentStats` is episode-scoped, a loss skip is
  per-update and per-loss-site, and there is no episode-shaped quantity to
  record without an aggregation the caller cannot undo (this ADR's own
  Context, part 4, above). ADR 0071's twelfth-accessor fence (its first
  reopen trigger) independently fences the destination. Two reasons, and the
  scope one would stand alone.
- **Seventeen registry rows, one per site.** Rejected on the four grounds
  under this ADR's own Decision 4 (the one-metric-row decision, above):
  agent-prefixed names breaching the table's name-by-role convention;
  13 of 17 panels structurally absent on every run; per-site attribution already
  served better by the `warn!`'s `site` field *and message*; and this ADR's own
  Decision 2 accessors already being the programmatic per-site API, so the rows
  would be a hand-synchronised second copy.
- **Keep the one-shot latch and rely on the counter alone.** Superficially
  attractive — the counter is exact, the log is lossy, why churn the log? —
  and rejected because **the accessor is only reachable programmatically**. An
  operator watching a training run's output, which is the overwhelmingly common
  way a divergence is first noticed, would still see one line and learn nothing
  about the rate. The counter serves the caller who already suspects a problem;
  the schedule serves the one who does not yet.
- **Per-interval emission.** As this ADR's own Decision 5 (the
  cumulative-not-per-interval decision, above) explains: the value would be a
  function of `log_every`, a logging parameter, making two runs of the same
  config incomparable; and first-differencing recovers the interval view from a
  cumulative series while the converse does not hold.
- **Fusing the warning schedule into the return value** (skip only when warning).
  Not seriously proposed, recorded because it is a one-line mutation of `check`
  that compiles, passes any test asserting "a warning appeared", and silently
  reintroduces the exact weight corruption ADR 0056 exists to prevent.
  This ADR's own Decision 1 (above) reaffirms 0056's Decision 3 *skip*
  semantics verbatim for this reason.
- **A `debug_assert!` or config gate on the guard.** Rejected by name for the
  fifth time — ADR 0056's unconditional-execution decision (Decision 4) settled
  it, ADR 0065's own unconditional-execution decision (Decision 5) adopted the
  ruling unchanged, 0070 cited it, 0071 rejected it a fourth time. Unchanged
  here, and this ADR adds nothing to the argument beyond noting that the
  *counter* inherits the same fate: a count that only exists in debug builds
  cannot report on the long release runs that diverge.
- **A bundled per-update health struct** (`update_health() -> UpdateHealth { … }`)
  in place of 22 accessors. Rejected **for now, on thinness**: this ADR adds
  exactly one health quantity, so the struct would have one field and a
  `#[non_exhaustive]` argument longer than the decision it carries — the same
  trade ADR 0070 and 0071 made against bundling on `AgentStats`. It is not
  rejected on principle, and the second reopen trigger below says when to
  re-argue it.

## Reopen triggers

Any one of these reopens this ADR:

1. **Anything in this crate begins to *branch* on a skip count.** `skipped`
   currently uses a plain `+=` without saturation, and that is acceptable
   **because it is a reported statistic, not a predicate** — the same standing
   ADR 0071's own saturation decision (Decision 7) gives `non_finite_episodes`,
   fenced there by its own third reopen trigger. A `u64` incremented once per
   skipped update cannot realistically reach
   its ceiling, but "cannot realistically" is a statistic's standard, not a
   control-flow predicate's. The moment a skip count gates a retry, an abort, an
   LR adjustment, or a cadence, the overflow policy needs re-deciding rather
   than inheriting.
2. **A second `PerUpdate` health counter is proposed** — e.g. a clipped-gradient
   count from #328, or a non-finite-gradient count. At that point the bundled
   health struct rejected above should be **re-argued from scratch**, not
   dismissed by citing this ADR: the thinness ground that defeats it at one
   quantity does not survive at two, and a third parallel accessor family across
   8 agents is a worse outcome than one struct.
3. **The per-site / aggregate split is collapsed** in either direction — an
   aggregate added to the single-site agents, per-site accessors removed in
   favour of the aggregate, or a train loop summing guards itself instead of
   calling the agent's aggregate. This ADR's own Decision 3 (above) is the
   record of why the split exists and what the single aggregate protects;
   collapsing it silently reintroduces the "one train loop forgot the new
   site" failure.
4. **A run is observed where ~7 log lines per site is too few to diagnose it** —
   for example a run whose skip rate changes character over time in a way a
   decade schedule cannot resolve, since the gap between the 6th and 7th lines
   spans nine tenths of the run. The remedy is *not* to unbound the log; it is to
   decide whether the metric's sampling cadence, not the log's schedule, is the
   right channel for that shape of question.

## References

- Issue **#346** — "[rl] add a skip counter to `AgentStats` for the finite-loss
  guard". Resolved, but **not at the destination it names**: the counter ships on
  the guard and on the agents, not on `AgentStats` (this ADR's own Context, part
  4, and its own Decision 3, above).
- Issue **#318** / ADR [0056](0056-non-finite-loss-skip-and-warn-guard.md) —
  **partially superseded**: its Decision 1 (type shape) and Decision 3 (the
  surfacing clause and the one-shot latch). Its Decision 2 (guard placement),
  Decision 4 (unconditional execution), Decision 5 (site count), Decision 3's
  *skip* semantics, and the whole of its Consequences & honest limits section
  remain in force; the skip semantics are reaffirmed verbatim in this ADR's own
  Decision 1 and the honest limit is restated in this ADR's own Consequences
  section.
- ADR [0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md) — the
  decade schedule and `{count, next_threshold, label}` shape adopted wholesale
  in this ADR's own Decision 1, and the `dropped()` non-test-gated accessor
  precedent adopted in this ADR's own Decision 2. 0065's own Decision 3 (the
  reward guard's schedule) is unchanged; only its rustdoc characterisation
  of the *loss* guard is now historical (this ADR's own Context, part 2, above),
  corrected in code.
- ADR [0059](0059-target-update-cadence-counts-gradient-updates.md)'s
  counter-advance decision (Decision 4) —
  the unconditional attempt-counter advance this counter complements and does
  **not** change; $\text{applied} = \text{attempts} - \text{skipped}$ (this ADR's own Decision 2 and
  Context, part 5, above).
- ADR [0067](0067-non-finite-observations-are-dropped-at-replay-ingestion.md) —
  the third guard in the family, whose `dropped_observations()` shares the
  counter-plus-schedule shape adopted here.
- ADR [0071](0071-best-score-latches-plus-infinity-the-finite-best-is-additive-and-counted.md) —
  its twelfth-accessor fence (first reopen trigger) on `AgentStats`, which this
  ADR explicitly does **not** fire and does **not** weaken (this ADR's own
  Decision 3, above). Its own saturation decision (Decision 7) is the model for
  this ADR's own first reopen trigger, above.
- ADR [0033](0033-share-splitmix64-mixer-across-core-and-evolution.md) and ADR
  [0012](0012-split-heavy-examples-into-rlevo-examples.md) — the house precedent
  for partial supersession: name the superseded clauses, restate them in their
  new form here "to avoid split authority", and leave the superseded document
  unedited and otherwise in force. ADR
  [0062](0062-grid-layout-fidelity-and-no-dead-rng.md) applies the same shape to
  ADR 0029 and argues explicitly against superseding wholesale.
- ADR [0015](0015-shared-typed-metric-registry-crate.md) — the typed registry's
  decision on how a single row joins the table, and the `names_are_unique`
  constraint that ground (a) above rests on.
- Issue **#184** (`sac_alpha.rs`) — the precedent ADR 0056's surfacing-and-latch
  decision (Decision 3) leaned on, and the weaker analogy here: one closed-form
  scalar step versus 17 sites across 8 agents (this ADR's own Decision 4,
  above). Its own guard and its 21 `warning_fired` references are untouched.
- Issue **#328** — finite-loss → NaN-gradient and grad-norm handling; still out
  of scope, and the reason the counter is a proxy (this ADR's own Consequences
  section).
- Issue **#352** / ADR 0065 — the reward-ingestion sibling whose rustdoc recorded
  this defect against its own neighbour (this ADR's own Context, part 1, above).

**Code citations. Post-change coordinates resolve against the working tree at
the time of writing; pre-change coordinates are marked and resolve against
`6636dc8`.**

- `crates/rlevo-reinforcement-learning/src/algorithms/shared.rs`:
  - `:602-611` — `FiniteLossGuard`'s new fields, with `next_warn_at`'s
    "schedules the `warn!` only — never the skip" doc.
  - `:576-591` — the "Why a decade schedule and not a one-shot latch" section.
  - `:593-597` — the $\text{applied} = \text{attempts} - \text{skipped}$ composition with ADR 0059's
    counters.
  - `:629-635` — `check`'s `CRITICAL:` rustdoc reaffirming 0056's Decision 3
    skip semantics; `:636-662` — its body, with the unconditional
    `self.skipped += 1` at `:640` and the scheduled `warn!` at `:641-660`.
  - `:673-675` — `skipped()`, `pub(crate)` and not test-gated.
  - `:697-707` — `FiniteRewardGuard`'s corrected "Why a decade schedule" section.
  - **Pre-change** `:650-654` — the loss guard's blind spot documented in-tree
    (this ADR's own Context, part 1, above); **pre-change** `:664-669` — the
    "latches its warning once, and that is right" rationale this ADR overturns
    (this ADR's own Context, part 2, above).
- `crates/rlevo-metrics-registry/src/lib.rs:333-353` — `n_updates` and the
  `skipped_updates` row appended immediately after it, with the placement
  rationale in-comment.
- The 17 `FiniteLossGuard::new` sites, tabulated in this ADR's own Context,
  part 3, above: `ppo_agent.rs`, `ppg_agent.rs`, `sac_agent.rs`, `td3_agent.rs`,
  `ddpg_agent.rs`, `dqn_agent.rs`, `c51_agent.rs`, `qrdqn_agent.rs`.
- `crates/rlevo-reinforcement-learning/src/metrics.rs:126-143` —
  `AgentStats::record`, the episode-scoped entry point that makes #346's proposed
  destination a units mismatch (this ADR's own Context, part 4, above).
- `crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_alpha.rs` —
  #184's separate $\alpha$ guard, untouched, including its own `warning_fired`.
