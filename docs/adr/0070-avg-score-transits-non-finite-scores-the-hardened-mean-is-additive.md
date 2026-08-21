---
project: rlevo
status: active
type: decision
date: 2026-08-09
tags: [adr, decision, numerical-stability, nan, metrics, agent-stats, avg-score, episode-return, rlevo-reinforcement-learning, issue-409]
---

# ADR 0070: `avg_score` transits non-finite scores — the hardened mean is additive, and it ships with a count

## Status

**Accepted (2026-08-09).** Resolves issue #409 (`[rl] AgentStats::avg_score has
no NaN backstop`), which is resolved **not** as filed: the change #409 proposes
is rejected, and the need behind it is met by two new accessors alongside the
untouched one.

**Supersedes nothing. Extends ADR
[0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md)'s bookkeeping
decision (Decision 4)**, in the same "extends, does not supersede" relation
that ADR [0069](0069-sanitized-fitness-is-reduced-in-f64.md)'s Status section
reuses from ADR [0034](0034-fitness-hygiene-chokepoint-convention.md) and ADR
[0023](0023-objective-sense-and-maximize-convention.md). 0065 stays `active`;
its "Episode return is deliberately left poisoned" clause (`0065:162-177`)
remains in force **verbatim and unchanged**, and remains the published contract
of `AgentStats::avg_score`.

**Chosen shape.** `AgentStats::avg_score` is byte-for-byte unchanged. Two
additive accessors ship beside it —

```rust
pub fn finite_avg_score(&self) -> Option<f32>;
pub fn non_finite_recent_len(&self) -> usize;
```

— mirroring the `StrategyMetrics::mean_fitness` / `broken_count` pair
(`crates/rlevo-evolution/src/strategy.rs:357-370`) that ADR 0034's metrics
decision (Decision 4) established for the fitness side of the workspace. The
mean and the count are **one decision, not two**: a hardened mean shipped
without its exclusion count is precisely the silent sanitization 0065
rejects.

Purely additive. No signature change, no behaviour change to any existing
accessor, no numerics change on an all-finite window, no user-book contract
retracted. This ADR also carries a **correction to two citation sites and one
sentence of ADR 0065**; see this ADR's own "Correction to ADR 0065" section,
below.

## Context

### 1. What #409 claimed, and what verification found

#409 is a careful, well-argued issue, and three of its four load-bearing claims
are wrong. Recording which — and why a reader of the code reached them — is most
of this ADR's value.

**Claim: "nothing in the current codebase can inject a NaN into a recorded score
today."** *False.* The path is not merely open, it is **deliberately preserved
and commented as such**. `crates/rlevo-reinforcement-learning/src/algorithms/
dqn/train.rs:129-137` carries a `DELIBERATE:` block whose first line reads
"`reward_f32` is accumulated raw, *including* a non-finite value that `remember`
just refused to store (ADR 0065, #352). Do not 'fix' this to skip a NaN." All
eight training loops accumulate `episode_reward += reward_f32` with no guard:

| agent | site |
|---|---|
| DQN | `algorithms/dqn/train.rs:137` |
| PPO | `algorithms/ppo/train.rs:194` |
| PPG | `algorithms/ppg/train.rs:135` |
| SAC | `algorithms/sac/train.rs:122` |
| TD3 | `algorithms/td3/train.rs:106` |
| DDPG | `algorithms/ddpg/train.rs:121` |
| C51 | `algorithms/c51/train.rs:122` |
| QR-DQN | `algorithms/qrdqn/train.rs:115` |

The `episode_reward` reaching `AgentStats::record` is therefore non-finite
exactly when any step of that episode was, and #409's "backstop, not a live bug"
framing inverts the actual situation: the injection path is the *designed*
behaviour of ADR 0065, not a residue of the two closed defects (#184, #173) the
issue names.

**Claim: "explicitly not an ADR violation … this issue stands on its own as
defensive hardening, not ADR compliance."** *Reversed.* #409 is right that
ADR 0023's RL-exemption sub-decision (RD-4) exempts RL metrics and that
0034's chokepoints are all in `rlevo-evolution` — the ADRs it checked are the
wrong ones. The governing record is 0065, whose bookkeeping decision
(Decision 4) is headed **"Bookkeeping on a drop — stated
explicitly so it is not 'fixed' later"** and which names this accessor by name
(`0065:164-165`). The contract is also *published*:
`docs/user-book/src/part-1-foundations/reinforcement-learning/33-reward.md:149-155`
lists the NaN episode return as one of three independent detection channels, in
prose that opens "This is deliberate, not a gap we missed" and closes "a `NaN`
in your reward curve is telling you the truth." #409 as filed is not hardening;
it is a silent reversal of an accepted decision and a retraction of shipped user
documentation.

**Claim: the `f64` accumulator "also clears the `clippy::cast_precision_loss`
`#[allow]`."** *False, in both directions.* The `#[allow]` is justified by the
`self.recent_history.len() as f32` divisor, and `usize as f64` fires
`cast_precision_loss` just the same; meanwhile the proposed
`(sum / n as f64) as f32` **adds** a `cast_possible_truncation` site that the
current all-`f32` expression does not have. `crates/rlevo-evolution/src/
fitness.rs:378,381` carries exactly these two `#[allow]`s, one each, for exactly
this expression shape. Widening the accumulator costs one more `#[allow]`, not
one fewer.

The claim that survives is the mechanical one, and it is correct: a non-finite
score in the window makes `Iterator::sum::<f32>()` non-finite, so `avg_score`
reports a non-finite mean for up to `window_size` episodes. That is true, it is
the accepted contract, and it is what this ADR's own Decision 2, below, gives
a *sibling* for rather than a *fix*.

### 2. One of eight training loops explains itself

Only DQN carries the `DELIBERATE:` comment. The other seven sites in the table
above are a bare `episode_reward += reward_f32;` with no comment, no ADR
citation, and nothing to stop a reader concluding it is an oversight.

This is a recurrence of a failure mode from ADR 0065's own Context section —
"C51 and QR-DQN were added later by copying an unguarded `remember` from an
earlier agent" — recurring one layer up, in the *documentation* rather than
the code. 0065 answered it for the guard
by mandating a shared struct plus a per-agent test in each of six files; the
comment that records *why* the sibling accumulator is unguarded got no such
treatment, and sits at one site out of eight.

#409 is the evidence that this matters. It was filed by someone who read the
code carefully, cited three ADRs, checked two closed defects, and still
concluded that nothing could inject a NaN — because the one comment that says
otherwise is in a file they had no reason to open.

### 3. #409 and ADR 0065 answer different questions about the same accessor

- #409 asks: *can a non-finite episode blank the reported mean?* Yes,
  transiently, for as long as that episode sits in the window.
- 0065 asks: *must a non-finite episode be visible?* Yes — it is the third of
  three surfacing channels, and the only one that does not share the guard's
  decade `warn!` schedule.

Both questions are legitimate and both answers are correct. They are simply not
questions about the *same accessor*. ADR 0034's metrics decision (Decision 4)
already settled this exact split on the fitness side — `StrategyMetrics`
reports the mean over finite members **and** a `broken_count` of the rest, "so
a single broken individual flags the population without blanking the mean" —
and the shape transfers without modification.

### 4. $\pm\infty$ is reachable with no non-finite reward anywhere

This is the observation that settles the predicate chosen in this ADR's own
Decision 3, below, and it is not in #409.

`episode_reward` is an `f32` accumulating over an episode of unbounded length. A
long episode of **entirely finite** rewards can saturate it to $\pm\infty$ by ordinary
accumulation — no divergence, no NaN, no defect in the environment.
`FiniteRewardGuard` (`crates/rlevo-reinforcement-learning/src/algorithms/
shared.rs:681,715`) never sees this, and correctly so: every individual reward
crossed `remember` finite, was admitted, and is sitting in the replay buffer as
valid data. `dropped_transitions()` reads zero. The `warn!` never fires. The
user-book's three channels report a healthy run.

The score handed to `AgentStats::record` is nonetheless $\pm\infty$, and it poisons the
window exactly as a NaN would. This is a **distinct failure mode from 0065's**,
with a distinct cause and no overlap in telemetry, and it is why the predicate
below is `is_finite()` rather than `!is_nan()`.

It is also a mechanism ADR 0069's Context section describes in another crate —
a finite term joining an `f32` accumulator until the accumulator itself leaves
the finite range — and it is worth naming the kinship without importing
0069's rule (see this ADR's own Decision 6, below).

### 5. The mechanical net that already exists

The workspace's algorithm acceptance gates already assert finiteness on this
exact value. `assert_improves_over_random`
(`crates/rlevo-test-support/src/assert.rs:22-34`) opens with
`assert!(trained.is_finite(), …)`, and `assert_reaches` (`:45-51`) opens with
`assert!(avg.is_finite(), …)`. Both are reached from
`crates/rlevo-test-support/src/macros.rs:62,78`, which feed them
`outcome.avg_score` — so **every** agent's integration suite routes its reported
mean through a finiteness assertion.

That net exists today, it is load-bearing, and this ADR's own Rejected
alternatives section, below, shows that #409's change would disable it.

## Decision

### 1. `avg_score` is unchanged, and its contract moves onto its own rustdoc

No filtering, no widened accumulator, no signature change. The windowed mean
continues to transit whatever the window contains, which is the accepted
contract of ADR 0065's bookkeeping decision (Decision 4).

What *does* change is where that contract is legible. Today the only place in
the crate that states it is a doc comment on a test
(`crates/rlevo-reinforcement-learning/src/metrics.rs:391-394`):

> row 1.2b ("NaN transits `avg_score`") is open and rated Low … `avg_score`
> admitting the NaN is therefore the known, accepted contract — do not "fix" it
> here.

That text is inside `#[cfg(test)] mod tests`. It is not in the rustdoc, not on
docs.rs, not in an IDE tooltip at any call site. **A contract readable only from
a test module is not a contract** — it is a note to whoever next opens the test
file, and #409 is the proof that the people who need it do not. The clause moves
to `avg_score`'s own rustdoc, citing ADR 0065 and naming the sibling accessor
from this ADR's own Decision 2, below.

### 2. Two additive accessors, and the pair is the decision

```rust
/// Mean score over the finite entries of the sliding window.
#[must_use]
pub fn finite_avg_score(&self) -> Option<f32>;

/// Count of window entries excluded from `finite_avg_score`.
#[must_use]
pub fn non_finite_recent_len(&self) -> usize;
```

Both `#[must_use]`, matching the six sibling accessors already on `AgentStats`
(`metrics.rs:140,151,161,172,181,189`).

**The pair is load-bearing; the mean alone is not an acceptable subset of this
decision.** A `finite_avg_score` shipped without `non_finite_recent_len` reports
a clean number over an unknown fraction of the window, and a caller cannot tell
"100 healthy episodes" from "one healthy episode and 99 excluded ones." That is
the silent sanitization ADR 0065's Rejected alternatives section rules out
and ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md)
forbids generally — moved from the reward to the statistic. The mean is only
admissible *because* the count ships with it. This mirrors 0034's metrics
decision (Decision 4) exactly, and the naming deliberately echoes
`broken_count`'s role rather than inventing a new vocabulary.

### 3. The predicate is `is_finite()`, deliberately stronger than #409's `!is_nan()`

Two independent reasons, either sufficient:

- **$\pm\infty$ destroys a mean as thoroughly as `NaN` does**, and it is what this
  workspace's own guard tests for: `FiniteRewardGuard::admit` is
  `if reward.is_finite()` (`shared.rs:715-716`), not an `is_nan` check. A
  `!is_nan()` filter would pass an $+\infty$ entry straight into the sum and produce
  an $+\infty$ mean from a "hardened" accessor — the worst available outcome, because
  it is a hardened name over an unhardened statistic.
- **The path from this ADR's own Context, part 4, above, is unguarded and
  distinct.** An episode whose `f32` return saturates carries no NaN anywhere,
  is invisible to `dropped_transitions()`, and is reachable on a run with no
  defect at all. Only `is_finite()` catches it.

`is_finite()` is also the predicate the acceptance assertions (this ADR's own
Context, part 5, above) and `StrategyMetrics::from_host_fitness`
(`strategy.rs:301-303`) already use, so this is the workspace's existing
convention rather than a new one.

### 4. An all-non-finite window yields `None`

`finite_avg_score()` returns `None` when the window is empty **and** when every
entry in it is non-finite. The two rejected answers, each rejected for its own
reason:

- **`Some(0.0)`** fabricates a score the agent never earned. ADR 0061's
  no-fabrication rule is directly against it, and 0065 invoked the same rule
  twice for the same reason.
- **`Some(f32::NAN)`** would make `finite_avg_score` behave identically to
  `avg_score` at the one input where the distinction is the entire point of the
  accessor.

**A deliberate divergence from `rlevo-evolution`, recorded so it is not read as
an oversight.** ADR 0069's empty-case decision (Decision 2) fixes
`sanitized_mean([]) == f32::NEG_INFINITY`
(`crates/rlevo-evolution/src/fitness.rs:367-384`, the empty branch at
`:374-375`). That sentinel is *not* adopted here. $-\infty$ is meaningful
in `rlevo-evolution` because ADR 0023 makes it the maximise-native worst value,
and `from_host_fitness` already returned it for an all-broken population.
`rlevo-reinforcement-learning` has **no objective-sense convention** — a return
of $-\infty$ there means "unboundedly bad", which is a claim about the agent, not
about the absence of data. Borrowing 0069's number while leaving the convention
that gives it meaning behind would be cargo-culting a constant. `Option` already
expresses "no data" in this API (`avg_score`, `best_score`), so the type that is
already there is the right one.

### 5. The count is derived on each call; no new state is stored

`non_finite_recent_len()` counts the window on demand. `AgentStats` gains **no
field**, and `record` (`metrics.rs:119-131`) is not touched.

- The window invariant `recent_history().len() <= window_size()` — the invariant
  the #407 privatisation exists to protect — is untouched, and so is the
  saturation reasoning settled for the lifetime counters under #408. Neither
  question reopens.
- A stored counter would have to be **decremented on `pop_front`**, which is a
  new invariant coupling a counter to a deque eviction. That is a second
  correctness obligation on the hot path of `record`, and it is precisely the
  shape a later refactor breaks silently: an eviction path that forgets the
  decrement yields a monotonically drifting count with no failing assertion.
- The cost is O(window) per call. `window_size` is bounded by
  `MAX_BUFFER_CAPACITY` at construction (`metrics.rs:93-99`) and is a hard-coded
  `100` at every in-crate call site, and the call site is a once-per-episode
  reporting path. A $\le 100$-element scan there is not a cost worth buying an
  invariant with.

`finite_avg_score` and `non_finite_recent_len` each walk the window
independently. Fusing them into one pass is this ADR's own Rejected
alternatives section's bundled-struct option, below, rejected there on the
same arithmetic.

### 6. The `f64` accumulator, narrowed once — and it does **not** extend ADR 0069 to this crate

`finite_avg_score` accumulates in `f64` and narrows to `f32` once, after the
division, carrying both `#[allow]`s with a justifying comment each. This is
ADR 0069's accumulator-width decision (Decision 1)'s shape, and
`crates/rlevo-evolution/src/fitness.rs:367-384` is the line-for-line precedent
— including `:378` (`cast_precision_loss` on the `usize as f64` divisor) and
`:381` (`cast_possible_truncation` on the single narrowing).

**Stated explicitly, because a future reader will otherwise conclude the
opposite: this does not make ADR 0069 binding on
`rlevo-reinforcement-learning`.** 0069's accumulator-width decision
(Decision 1) scopes its rule to reductions over *sanitized fitness*, defines
"reduction" in terms of fitness magnitudes, and binds transitively through
stored fitness fields. Episode scores are neither sanitized nor fitness;
nothing in this crate routes through `sanitize_fitness`, and
`PerformanceRecord::score` has no hygiene chokepoint upstream of it by
design. The width is adopted here on its own merits — this ADR's own Context,
part 4, above, establishes that the terms have unbounded magnitude, which is
exactly the condition an `f32` accumulator is wrong for — and recording the
scope distinction is what keeps a future reader from either importing 0069's
rule wholesale or, worse, importing `sanitize_fitness` with it.

`avg_score` keeps its `f32` accumulator. Changing it would change reported
numbers in the last ULP for existing runs, for a statistic whose contract is
"report what the window contains."

## Rejected alternatives

- **Filter inside `avg_score`, as #409 filed it.** The primary rejection. Four
  grounds, in descending strength:

  1. **It blinds the workspace's own acceptance gates.** This ADR's own
     Context, part 5, above: every algorithm integration test asserts
     `avg_score.is_finite()` before checking convergence. A self-sanitizing `avg_score` means a run whose episodes went
     NaN would pass that assertion and then be compared against a convergence
     threshold computed over whichever episodes happened to survive the filter —
     possibly one. The suite would report a green convergence check for a run
     that never converged. This is ADR 0061's fabrication failure mode reaching
     the *test suite*, which is strictly worse than reaching a metric: a
     fabricated number a human reads gets questioned, a fabricated number an
     assertion reads gets trusted.
  2. **It supersedes ADR 0065's bookkeeping decision (Decision 4) without
     saying so.** 0065's third surfacing channel is destroyed by this change,
     and #409 proposes it while explicitly stating no ADR is implicated.
     Reversing an accepted decision requires a superseding ADR
     (`docs/rules.md:649`), not a patch.
  3. **It falsifies published user-book prose without editing it.**
     `33-reward.md:149-155` tells users that a `NaN` in their reward curve is
     telling them the truth. After this change it would not be.
  4. **It fabricates.** A mean over the surviving subset, reported under the
     name of a mean over the window, is a number the agent did not earn.

  **The reproducibility suite is indifferent to this change, and that
  indifference is the point.** The `rl_reproducibility_test!` `bits` arm
  (`crates/rlevo-test-support/src/macros.rs:105-118`) runs the same closure twice
  at the same seed and bit-compares the two `avg_score` values through
  `assert_reproducible_bits` (`assert.rs:77-79`). A deterministic `NaN` is
  reproducible; so is a deterministically filtered mean. That arm goes green
  either way, before and after any version of #409's change. It offers **no**
  protection here — which is exactly why the `is_finite()` assertions in
  `assert_improves_over_random` / `assert_reaches` are the only mechanical net
  that exists, and why silencing them is ground (1) rather than a footnote.

- **A `debug_assert!` or a config-gated check.** Rejected by name, and settled
  twice already: ADR 0056's no-debug-gate decision (Decision 4) ("No
  `debug_assert`/config gate: the host read already exists, so a gate would
  strip the guard from exactly the long release runs that diverge") and
  ADR 0065's own no-debug-gate decision (Decision 5), which adopts that
  ruling unchanged. Long release runs are where a saturating episode return
  (this ADR's own Context, part 4, above) is most likely and where a release
  no-op protects nothing. Recorded so it is not re-proposed a third time.

- **`!is_nan()` as the predicate**, as #409 spells it. See this ADR's own
  Decision 3, above: it admits $+\infty$ into a mean advertised as hardened, and it
  misses the path from this ADR's own Context, part 4, entirely.

- **A bundled return, e.g. `finite_score_summary() -> Option<FiniteScoreSummary
  { mean, excluded }>`.** Attractive because it makes the mean and the count
  inseparable at the type level, which is what this ADR's own Decision 2,
  above, argues for in prose. Rejected on three counts. `AgentStats` is a
  **flat accessor surface** after
  #407 — six `#[must_use]` scalar accessors plus `avg_score`, no aggregate
  returns — and a struct would be the only exception. It would need its own
  `#[non_exhaustive]` reasoning as a new public type in a pre-1.0 crate, which is
  a decision with a longer tail than the one being made here. And its one
  concrete advantage, a single pass instead of two, is worth nothing on a
  once-per-episode call over a $\le 100$-entry deque (this ADR's own Decision 5,
  above). `StrategyMetrics` bundles for a reason `AgentStats` does not
  share: it is a snapshot value type
  that crosses crate boundaries (ADR [0015](0015-shared-typed-metric-registry-crate.md)),
  where the fields are read together, serialized together, and versioned
  together. `AgentStats` is a live accumulator read in place.

- **A stored non-finite counter maintained in `record`.** This ADR's own
  Decision 5, above: it buys an O(1) accessor on a once-per-episode path in
  exchange for a new counter/eviction invariant that a future refactor can
  break without failing anything.

- **Sanitizing the score at `record` time**, so nothing non-finite ever enters
  the window. This is the same fabrication moved one layer earlier, and it is
  strictly worse than filtering in `avg_score`: it would additionally poison
  `best_score` (which folds the score at `metrics.rs:124`) and
  `recent_history()`, which is `pub` and hands back the raw `&VecDeque<T>`. With
  the entries themselves rewritten, **no accessor on the type would report what
  actually happened** — 0065's channel would be destroyed rather than merely
  bypassed, and there would be no way to reconstruct it.

## Correction to ADR 0065

ADR 0065 is **not edited** (`docs/rules.md:649`; `docs/adr/README.md:3-6`). The
corrections are carried here, following the precedent ADR 0066's section
titled "…correction is recorded here" (`0066:133-137`) and ADR 0067's
"Correction to ADR 0065" section (`0067:201-215`) set for exactly this
situation. As at `0066:125`, **all line numbers below resolve against the
commit that wrote them**; 0065's own citations resolve against the commit at
which 0065 was accepted.

### (a) Three stale line citations — a citation correction, not a premise correction

0065 cites `metrics.rs` at three sites, all now stale:

| 0065 cites | actually | what is there |
|---|---|---|
| `metrics.rs:71` (at `0065:178`, and again at `0065:377`) | `metrics.rs:124` | the `f32::max` fold in `record` |
| `metrics.rs:92-102` (at `0065:165`, and again at `0065:377`) | `metrics.rs:242-253` | `avg_score`'s body (its rustdoc begins at `:194`) |

Both moved under **#407**, which privatised `AgentStats`' fields and added the
six accessors, landing after 0065 was accepted. (#409's own body cites
`metrics.rs:91-102` and `87-90`, inherited from the same pre-#407 layout.)

`avg_score`'s coordinates moved a **second** time, within the very commit that
writes this ADR: this ADR's own Decision 1's rustdoc block is inserted above
it, pushing the body from `:205-216` to `:242-253`. The number recorded above
is the
post-change one. This is not a footnote — a correction table that goes stale
inside its own diff is the failure mode this section exists to fix, and it very
nearly did.

**0065's reasoning is unaffected.** Both cited mechanisms are intact and
unchanged — `record` still folds with `f32::max`, `avg_score` still sums the
window in `f32` — only their coordinates moved. This is unlike ADR 0066, which
corrected a *premise* 0065 reasoned from; nothing here changes what 0065
concluded or why.

### (b) One 0065 sentence narrows: `best_score` is NaN-immune but **not** $+\infty$-immune

`0065:178-180` reads:

> (`best_score` is **not** poisoned — `metrics.rs:71` uses `f32::max`, which
> ignores a NaN operand, so a single bad episode cannot corrupt the best-ever
> record.)

True of `NaN`, and pinned as true by two tests
(`metrics.rs:402,447`, which cover the argument and receiver positions of the
fold separately). **False of $+\infty$.** `f32::max` has no NaN-suppression story for
infinities — it propagates $+\infty$ correctly, because $+\infty$ *is* the maximum — and
`best_score` is explicitly never evicted by the window (`metrics.rs:159-160`).
So a single $+\infty$ episode, of the kind this ADR's own Context, part 4, shows
is reachable with no non-finite reward anywhere, latches `best_score` at $+\infty$
for the remaining lifetime of the agent. The sentence should read "cannot corrupt the best-ever
record *with a `NaN`*."

**Explicitly out of scope for this ADR**, and filed as a follow-up issue. It
touches `record` rather than a read accessor, which is the one function this ADR
commits to not modifying (this ADR's own Decision 5, above), and the remedy
is a genuine policy question rather than a mechanical fix: $+\infty$ may be the *true* best score of an
episode that really did saturate, in which case latching it is correct
reporting and a `best_finite_score` sibling is the answer; or the latch may be
judged a defect in its own right. That is the same finite/raw pairing this ADR
makes for the mean, and settling it should reuse this ADR's reasoning rather
than pre-empt it — see this ADR's own Reopen trigger 3, below.

## Literature

Deliberately short: this is an API-shape decision, and the numerical
literature is carried by ADR 0065's Literature section and ADR 0069's
Context section, both of which this ADR builds on rather than restates.

- **IEEE 754-2008 `minNum`/`maxNum` — why neither `clamp` nor `f32::max` is
  total, and why the predicate must be explicit.** The standard introduced the
  NaN-suppressing `minNum`/`maxNum` precisely because plain `min`/`max` are not
  NaN-safe, and Rust's `f32::max` follows the NaN-suppressing form. The
  consequence for this ADR is a clean complementarity: **clipping neutralises
  $\pm\infty$ but not `NaN`** (0065's Literature section states this precisely,
  quoting Mnih et al. 2013 on reward clipping as an $\infty$-and-magnitude
  mitigation rather than a NaN one), while **`f32::max` neutralises `NaN`
  but not $+\infty$** (this ADR's own Correction (b), below, is that gap, live in
  this crate). Neither operation's implicit handling is total over the
  non-finite domain. Only an explicit `is_finite` predicate is — which is
  this ADR's own Decision 3, above, and which is also ADR 0066's general
  ruling in a different guise: where correctness depends on what happens to a
  non-finite value, pin it with an explicit test rather than an operation's
  implicit behaviour.

- **The missing-data framing, reused from ADR 0065's Literature section
  rather than re-derived.** Excluding non-finite episodes from a reported mean is
  missing-completely-at-random and statistically inert **if** the cause is a
  policy-independent numerical bug, and missing-not-at-random and systematic
  **if** the non-finite episodes correlate with the states the policy visits —
  the realistic case for a diverging environment, where the agent's own actions
  drive it into the regime that produces the bad value. No property of the
  estimator distinguishes the two regimes; only telemetry from the run itself
  can. That is precisely why `non_finite_recent_len()` is not an ergonomic
  extra but a condition of `finite_avg_score()` existing at all (this ADR's
  own Decision 2, above), and it is the same conclusion 0065's
  count-must-be-observable decision (Decision 3) reached for
  `dropped_transitions()` on operational grounds.

## Consequences

### Positive

- **The accepted contract becomes readable from the API rather than from a
  test.** This ADR's own Decision 1, above, moves it from `#[cfg(test)]`
  prose to `avg_score`'s rustdoc, which is the IDE tooltip at every call site
  and the docs.rs entry for the accessor. #409 is the measured cost of its
  previous location.
- **ADR 0065's third surfacing channel survives intact.** A non-finite episode
  return still reaches the reported mean, still fails the acceptance assertions,
  and still tells the truth about the run.
- **The seven undocumented `episode_reward +=` sites stop being an invitation to
  re-file #409.** The asymmetry from this ADR's own Context, part 2, above,
  is recorded and closed.
- **The hardened statistic exists for the callers who want it**, with the
  exclusion count that makes it honest, and with a predicate (`is_finite`) that
  covers a failure mode (this ADR's own Context, part 4, above) no existing
  guard reaches.
- No public signature change, no behaviour change on an all-finite window, no
  change to `record`, `best_score`, or the window invariant.

### Negative / accepted costs — do not soften these

- **Two means now exist for one quantity, and every reader must work out which
  one they want.** This is the same cost ADR 0069's Consequences section
  records as "two widths now exist for one quantity", in the same shape: a second spelling of a
  familiar statistic is a permanent comprehension tax and a permanent
  opportunity for a call site to pick the wrong one. It is accepted here for the
  same reason — the alternative is a single accessor that lies to one of its two
  audiences. The mitigation is documentation and nothing stronger:
  `avg_score`'s rustdoc names `finite_avg_score`, says what it excludes, and
  says when to reach for it; `finite_avg_score`'s rustdoc points back and states
  that `non_finite_recent_len()` must be read with it.
- **Nothing mechanically prevents a caller from reading `finite_avg_score()` and
  ignoring `non_finite_recent_len()`.** This ADR's own Decision 2, above,
  argues the pair is indivisible; `#[must_use]` on the count does not make it
  so, because a caller
  who never calls it is unaffected. The bundled-struct alternative *would* have
  enforced it and was rejected on other grounds — that trade is real and is not
  being papered over.
- **The count is O(window) and the two accessors walk the window twice.**
  Defensible at `window_size <= 100` on a once-per-episode path (this ADR's
  own Decision 5, above), and not defensible if either of those facts
  changes. Reopen trigger 2.

### Neutral

- No new dependency, no new public type, no new module.
- `AgentStats` derives only `Debug` and `Clone` (`metrics.rs:49`) — no `serde` —
  so there is no wire format, no persisted format, and no
  `rlevo-metrics-registry` entry affected by adding accessors.
- ADR 0065's `FiniteRewardGuard`, its six agent drop tests, and its decade warn
  schedule are byte-for-byte untouched.

## Reopen triggers

Any one of these reopens this ADR:

1. **A third mean accessor over the same window is proposed** — a trimmed mean,
   a median, a per-window standard deviation. At three, the flat-accessor
   argument in this ADR's own Rejected alternatives section, above, inverts
   and the bundled struct earns its price.
2. **A caller needs the finite mean over *lifetime* history rather than the
   window.** The window scoping is what makes the derive-on-call decision cheap;
   a lifetime statistic cannot be derived from state that has been evicted, so
   this ADR's own Decision 5 falls and `record` must maintain state after
   all — with the counter/eviction invariant that decision exists to avoid.
3. **The `best_score` $+\infty$ latch (this ADR's own Correction (b), above) is
   fixed in a way that introduces its own finite/raw pair** — e.g. a
   `best_finite_score` alongside
   `best_score`. At that point `AgentStats` carries two independent finite/raw
   pairings, and they should be unified in naming, predicate, and empty-case
   behaviour rather than each arguing its own case.
4. **Anyone proposes "simplifying" the two means into one.** This ADR is the
   record of why there are two; the simplification is the change #409 asked for,
   under a different name.

## References

- Issue **#409** — "[rl] `AgentStats::avg_score` has no NaN backstop (low
  priority)". Resolved, not as filed: three of its four claims are refuted in
  this ADR's own Context, part 1; its proposed change is rejected in this
  ADR's own Rejected alternatives section, and the need behind it is met
  additively.
- Issue **#407** — the `AgentStats` field privatisation and accessor addition;
  the source of this ADR's own Correction (a)'s stale coordinates and of the
  flat-accessor surface this ADR's own Rejected alternatives section reasons
  from.
- Issue **#408** — the lifetime-counter saturation, settled in `record`; not
  reopened by this ADR's own Decision 5.
- ADR [0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md) —
  **extended and preserved**; its bookkeeping decision (Decision 4)'s
  "Episode return is deliberately left poisoned" clause stands unchanged, and
  its count-must-be-observable decision (Decision 3)'s "the count must be
  observable" reasoning is reused by this ADR's own Decision 2. Its
  `metrics.rs` citations and one `best_score` sentence are corrected here
  without editing it.
- ADR [0034](0034-fitness-hygiene-chokepoint-convention.md) — its metrics
  decision (Decision 4)'s mean-over-finite-members-plus-`broken_count` shape,
  which this ADR's own Decision 2 mirrors.
- ADR [0069](0069-sanitized-fitness-is-reduced-in-f64.md) — the `f64`-accumulator
  shape this ADR's own Decision 6 adopts and the scope that same Decision 6
  declines to extend; the "extends, does not supersede" Status spelling; the
  "two widths now exist" consequence this ADR's own Consequences section
  reuses.
- ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) — the
  no-fabrication rule behind this ADR's own Decision 4's rejection of
  `Some(0.0)` and this ADR's own Rejected alternatives' ground (1) and (4).
- ADR [0056](0056-non-finite-loss-skip-and-warn-guard.md) — its no-debug-gate
  decision (Decision 4), the no-`debug_assert`/config-gate ruling, adopted
  unchanged by 0065's own no-debug-gate decision (Decision 5) and cited a
  third time here.
- ADR [0066](0066-clamp-nan-behavior-is-backend-specific-pin-with-is-nan.md)
  (`:125`, `:133-137`) and ADR
  [0067](0067-non-finite-observations-are-dropped-at-replay-ingestion.md)
  (`:201-215`) — the precedent for recording a correction to an immutable ADR in
  a later one, including the guard that cited line numbers resolve only against
  the commit that wrote them.
- ADR [0023](0023-objective-sense-and-maximize-convention.md) — the
  maximise-native $-\infty$ worst-value convention that `rlevo-reinforcement-learning`
  does **not** have, which is why this ADR's own Decision 4 returns `None`
  rather than 0069's $-\infty$. 0023's RL-exemption sub-decision (RD-4) is the
  clause #409 correctly cites and then draws the wrong conclusion from.
- ADR [0015](0015-shared-typed-metric-registry-crate.md) — why `StrategyMetrics`
  bundles its fields and `AgentStats` need not.
- `crates/rlevo-reinforcement-learning/src/metrics.rs:119-131` (`record`, the
  `f32::max` fold at `:124`), `:194-216` (`avg_score`), `:140,151,161,172,181,189`
  (the six sibling `#[must_use]` accessors), `:159-160` (the never-evicted
  `best_score`), `:391-394` (the `#[cfg(test)]` doc comment carrying the contract
  today).
- `crates/rlevo-evolution/src/strategy.rs:279-332` (`from_host_fitness`, the
  filter-plus-count implementation), `:357-370` (`mean_fitness` /
  `broken_count`, the accessor pair this ADR's own Decision 2 mirrors).
- `crates/rlevo-evolution/src/fitness.rs:367-384` — `sanitized_mean`: the
  `f64`-accumulator-narrowed-once shape (`:378`, `:381` — the two `#[allow]`s
  #409 believed the widening would remove) and the $-\infty$ empty contract at
  `:374-375` that this ADR's own Decision 4 deliberately does not adopt.
- The eight unguarded `episode_reward +=` sites, enumerated in this ADR's own
  Context, part 1; `crates/rlevo-reinforcement-learning/src/algorithms/
  dqn/train.rs:129-137` is the only one carrying the `DELIBERATE:` comment.
- `crates/rlevo-test-support/src/assert.rs:22-34,45-51` — the `is_finite()`
  acceptance assertions #409's change would blind; `:77-79` and
  `crates/rlevo-test-support/src/macros.rs:105-118` — the `bits` reproducibility
  arm that is indifferent to it.
- `crates/rlevo-reinforcement-learning/src/algorithms/shared.rs:681,715` —
  `FiniteRewardGuard` and its `reward.is_finite()` predicate, which this
  ADR's own Context, part 4, shows does not and cannot cover the
  saturating-accumulator path.
- `docs/user-book/src/part-1-foundations/reinforcement-learning/33-reward.md:136-155,349-353`
  — the three published detection channels, the third of which #409's change
  would falsify.
- `docs/rules.md:649` and `docs/adr/README.md:3-6` — the immutability rule that
  makes this ADR's own "Correction to ADR 0065" section the only available
  mechanism.
- IEEE 754-2008 — `minNum`/`maxNum`, the NaN-suppressing variants introduced
  because plain `min`/`max` are not NaN-safe.
