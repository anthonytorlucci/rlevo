---
project: rlevo
status: active
type: decision
date: 2026-08-09
tags: [adr, decision, numerical-stability, infinity, order-statistic, metrics, agent-stats, best-score, episode-return, rlevo-reinforcement-learning, issue-1078]
---

# ADR 0071: `best_score` latches `+∞` permanently — the finite best is additive, and it is counted

## Status

**Accepted (2026-08-09).** Resolves issue #1078 (`[rl] AgentStats::best_score
latches at +∞ permanently, worse than the avg_score case (#409)`).

Unlike ADR [0070](0070-avg-score-transits-non-finite-scores-the-hardened-mean-is-additive.md),
which rejected the remedy #409 proposed, **#1078's analysis is correct
throughout.** Its mechanism is right, its severity ranking against #409 is
right, and it deliberately declines to choose between its own two options —
"latching is correct reporting, document it" and "ship a finite sibling". This
ADR picks the **second**, and records why the first is *necessary but not
sufficient*: documenting a latch tells a reader what happened to a number they
can no longer use, without giving them a number they can.

**Supersedes nothing. Extends ADR 0070's correction to ADR 0065 (b) and its
third reopen trigger, and ADR
[0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md)'s bookkeeping
decision (Decision 4)** — the same "extends, does not supersede" relation 0070
took to 0065 and ADR [0069](0069-sanitized-fitness-is-reduced-in-f64.md) took
to ADR [0034](0034-fitness-hygiene-chokepoint-convention.md). **0065 and 0070
both stay `active`,** and the following clauses of theirs remain in force
verbatim:

- 0065's bookkeeping decision (Decision 4)'s **"Episode return is deliberately
  left poisoned"** clause (`0065:162-177`). Untouched. The eight
  `episode_reward +=` sites still accumulate unconditionally, and this ADR
  depends on that being true (this ADR's own Context, part 4, below).
- 0070's Decision 1 — **`avg_score` is unchanged**, and its contract lives on
  its own rustdoc.
- 0070's Decision 2 — **the pair is the decision**: a hardened statistic
  without its exclusion count is silent sanitization.
- 0070's Decision 3 — **the predicate is `is_finite()`**, not `!is_nan()`.
- 0070's Decision 4 — **`None`, not a fabricated value**, for the empty and
  all-non-finite cases.

**Chosen shape.** `best_score`'s *body* is byte-for-byte unchanged; only its
rustdoc grows a section naming the latch. Two additive accessors ship beside it —

```rust
pub fn finite_best_score(&self) -> Option<f32>;
pub fn non_finite_episodes(&self) -> usize;
```

— the **lifetime** analogue of 0070's **windowed** pair. `AgentStats` now
carries a finite/raw split on both axes:

|          | raw          | hardened            | count of what was excluded |
|----------|--------------|---------------------|----------------------------|
| windowed | `avg_score`  | `finite_avg_score`  | `non_finite_recent_len` — derived on call, **self-heals** |
| lifetime | `best_score` | `finite_best_score` | `non_finite_episodes` — stored, monotone, **never heals** |

**One departure from ADR 0070, flagged here so it is not read as a reversal:
this change *does* modify `record`
(`crates/rlevo-reinforcement-learning/src/metrics.rs:126-143`) and *does* add
two private fields (`:57-60`).** 0070's no-stored-state decision (Decision 5)
committed to touching neither. That commitment was made against a *windowed*
counter and rested on one specific objection — eviction coupling — which does
not exist for a lifetime statistic. This ADR's own Decision 6 (below)
discharges it in full.

This ADR also carries a **correction to ADR 0070**; see this ADR's own
Correction to ADR 0070 section, below.

## Context

### 1. What #1078 got right, and the one thing it left open

Recording this plainly, because it is the opposite of what ADR 0070's Context,
part 1, found for #409: **every load-bearing claim in #1078 verified.**

- `AgentStats::record` folds the best with `f32::max`
  (`metrics.rs:131`), which **discards** a `NaN` operand but **propagates**
  `+∞` — correctly, because `+∞` compares greater than every finite value.
- `best_score` is never evicted by the sliding window (stated in its own
  rustdoc, `metrics.rs:171-172`; pinned by
  `best_score_survives_eviction_while_average_does_not`, `metrics.rs:886`).
- Therefore one `+∞` episode pins `best_score()` at `+∞` for the entire
  remaining lifetime of the agent, with no self-healing of any kind.
- The severity ranking against #409 is correct: this is **strictly worse** than
  the `avg_score` case, which recovers within one `window_size` (this ADR's own
  Context, part 3, below).

The one thing #1078 left open is the remedy, and it left it open *knowingly*,
offering two options and declining to pick. That is the right call for an issue
and the wrong state for a codebase, and this ADR's own Decision 1 (below)
answers it: the latch stays because it is true, and a sibling ships because
"true" is not the same as "usable".

### 2. This is ADR 0070's correction to ADR 0065 (b) coming due

0070 discovered this defect while correcting one sentence of 0065, and filed it
out of scope **by name**, for a reason it stated precisely:

> the remedy is a genuine policy question rather than a mechanical fix: `+∞` may
> be the *true* best score of an episode that really did saturate, in which case
> latching it is correct reporting and a `best_finite_score` sibling is the
> answer; or the latch may be judged a defect in its own right.

0070's third reopen trigger then pre-authorised both this ADR's **existence**
and its **scope**:

> **The `best_score` `+∞` latch (this ADR's own Correction (b), above) is
> fixed in a way that introduces its own finite/raw pair** … At that point
> `AgentStats` carries two independent finite/raw pairings, and they should be
> unified in naming, predicate, and empty-case behaviour rather than each
> arguing its own case.

So the unification is not an ergonomic afterthought here; it is the trigger's
stated deliverable, and this ADR's own Decision 4 (below) discharges it
explicitly. What this ADR adds that 0070 could not is the *policy answer* —
plus one piece of context 0070 did not have (this ADR's own Context, part 4,
below).

### 3. "Not evicted" is what turns a transient into a permanent

The mechanism is one sentence long and it is the whole reason 0070's remedy
cannot be reused.

`avg_score` sums the window (`metrics.rs:429-440`). A non-finite entry poisons
the reported mean for at most `window_size` episodes and then rolls out. The
damage is bounded in time, and — critically — it is bounded *by state that is
still there*: `non_finite_recent_len` can be **derived from the window on every
call** (`metrics.rs:575-580`) precisely because the offending entries are still
resident while they matter.

`best_score` has no window. The episode that set it may have been evicted ten
thousand episodes ago; the *value* survives in a field. There is nothing left to
scan. A latched `+∞` is therefore both permanent and underivable, and the
derive-on-call shape of 0070's no-stored-state decision (Decision 5) — the
obvious thing to reach for by analogy — is not merely inefficient here, it is
**wrong**, in the sense of computing a different quantity. It is pinned as
wrong by `finite_best_score_survives_eviction` (`metrics.rs:1160`), whose doc
comment names that mutant as "the most important mutant of the whole change".

### 4. `+∞` is far more reachable than ADR 0070's Context, part 4, said

This is the substantive new context, and it is what moves this defect from
"theoretically reachable" to "reachable on a correctly configured, fully guarded
agent today".

ADR 0070's Context, part 4, rested on a single path: `f32` saturation of
`episode_reward` over a long episode of entirely finite rewards. That path is
real, and it remains the one no guard can ever see. But there is a **second,
far more reachable path**, and it runs straight through ADR 0065's design:

ADR 0065's `FiniteRewardGuard` refuses to *store* a non-finite per-step reward
(`crates/rlevo-reinforcement-learning/src/algorithms/shared.rs:681,715-716` —
`if reward.is_finite()`). It does **not** stop the reward from reaching the
episode return. All eight training loops accumulate `episode_reward +=
reward_f32` unconditionally **and by design**, because 0065's bookkeeping
decision (Decision 4) requires it — a dropped transition must still surface in
the reported curve:

| agent | `DELIBERATE:` comment | `episode_reward +=` |
|---|---|---|
| DQN | `algorithms/dqn/train.rs:129` | `:137` |
| PPO | `algorithms/ppo/train.rs:194` | `:204` |
| PPG | `algorithms/ppg/train.rs:135` | `:148` |
| SAC | `algorithms/sac/train.rs:122` | `:130` |
| TD3 | `algorithms/td3/train.rs:106` | `:114` |
| DDPG | `algorithms/ddpg/train.rs:121` | `:129` |
| C51 | `algorithms/c51/train.rs:122` | `:130` |
| QR-DQN | `algorithms/qrdqn/train.rs:115` | `:123` |

**Consequence: a single `+∞` environment reward reaches `AgentStats::record` on
an agent with every ingestion guard in place — and `dropped_transitions()` reads
`1`, not zero.** That is the exact inverse of the saturation path's telemetry
signature. The guard fired, the `warn!` fired, the operator was told, the
transition never entered replay — and `best_score()` is nonetheless pinned at
`+∞` for the rest of the run, long after the environment was fixed and the run
recovered. The one accessor that cannot recover is the one no guard protects,
because it is downstream of every guard by design.

Two independent paths, opposite telemetry, same latch:

| path | `dropped_transitions()` | `warn!` | reachable on a guarded agent |
|---|---|---|---|
| non-finite env reward (0065's bookkeeping decision, Decision 4) | `>= 1` | yes (decade schedule) | yes |
| `f32` accumulator saturation (0070's Context, part 4) | `0` | no | yes |

**`−∞` is harmless here, and the asymmetry is worth stating** because it does
not exist on the `avg_score` side. `f32::max` discards `−∞` as the smaller
operand at the first finite episode, so the raw best is unaffected by it. On the
mean side, `−∞` is exactly as destructive as `+∞`. This is the first place in
the workspace where the two infinities behave differently, which is one more
reason the predicate must be spelled out rather than inferred from the operation
(this ADR's own Decision 3, below). It is pinned by
`negative_infinity_is_excluded_from_finite_best_and_counted`
(`metrics.rs:1085`), which exists specifically because raw and hardened agree on
the *value* there and only the *count* separates them.

**One claim from ADR 0070's Context, part 2, is now spent, and must not be
re-asserted.** 0070
recorded that only DQN carried the `DELIBERATE:` comment and the other seven
sites were bare — "an invitation to re-file #409". Commit `acd7b97` closed that:
all eight sites now carry it, as the table above shows. This ADR's own
correction to ADR 0070 (b), below, records this so the table above is not
misread as contradicting 0070.

### 5. Blast radius, and the forward risk

Two reads of `best_score()` exist outside `metrics.rs`, and both are benign
today:

- `crates/rlevo-examples/examples/book/ch03_dqn_cartpole.rs:179` —
  `stats.best_score().unwrap_or(f32::NAN)`, a progress line printed to stdout.
- `crates/rlevo/tests/integration_test.rs:418` —
  `assert_eq!(stats.best_score(), Some(8.0))`, on an all-finite fixture.

Neither changes, and neither would have caught anything. The risk #1078 names is
**forward-looking and precise**: `best_score` is documented on
`PerformanceRecord::score` itself as "the primary scalar metric used for
checkpointing and best-model tracking" (`metrics.rs:11-12`). It is where a
checkpoint-on-best feature *would* read. `+∞` there does not read as "corrupt";
it reads as **"new best, forever"** — every subsequent comparison against it
fails, so the checkpoint freezes at the saturating episode's weights and never
updates again. A latch in a reporting field is a bad number; a latch in a
selection criterion is a silently wrong model. This ADR's own second reopen
trigger (below) puts a fence around that.

### 6. No existing gate would have caught this, and none was blinded

ADR 0070's Context, part 5, turned on a mechanical net: `assert_improves_over_random`
and `assert_reaches` in `rlevo-test-support` both open with an `is_finite()`
assertion, so #409's proposed change would have **blinded a live gate**. That
argument does not transfer, in either direction, and the reason is worth
recording.

Those assertions gate on the **average**, never on the best:

- `crates/rlevo-test-support/src/assert.rs:22` `assert_improves_over_random`,
  with `trained.is_finite()` at `:24` and `random.is_finite()` at `:28`.
- `crates/rlevo-test-support/src/assert.rs:46` `assert_reaches`, with
  `avg.is_finite()` at `:47`.
- Both are fed `outcome.avg_score` from
  `crates/rlevo-test-support/src/macros.rs:62,78`.

`rg 'best_score' crates/rlevo-test-support/` returns nothing. **No acceptance
gate anywhere in the workspace reads `best_score`.** So unlike #409, there was
no net to blind — and, more importantly, no net to rely on. A run whose
`best_score` latched at `+∞` passes the entire suite green: the `avg_score`
assertions see a healthy mean once the window rolls, the reproducibility `bits`
arm bit-compares two deterministically identical latched values, and nothing
looks at the latched field at all.

**That is why this defect could sit in the codebase unnoticed while its milder
sibling was found, filed, and argued over.** The severity ordering and the
detection ordering are inverted, and only the second one had a mechanism behind
it.

## Decision

### 1. `best_score` keeps the latch; its contract moves onto its own rustdoc

No filtering, no predicate, no signature change. The body is byte-for-byte
unchanged (`metrics.rs:217-220`). `+∞` **is** the true maximum score observed —
the episode really did return it — and reporting something smaller under the
name "best score" is exactly the fabrication ADR
[0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) forbids and
0070's Rejected alternatives section ruled out for the mean. #1078's first
option is adopted in full, as far as it goes.

This is the same move as 0070's Decision 1, with one point sharper. 0070 found
the `avg_score` contract stated *only* in a `#[cfg(test)]` doc comment and
argued that "a contract readable only from a test module is not a contract".
Here it was not even that. Before this change, **nothing anywhere said
`best_score` could latch** — not the rustdoc, not a test, not a comment. The
only mention of the mechanism in the whole repository was 0065's sentence
asserting the *opposite* half of it ("`best_score` is **not** poisoned"), which
0070's correction to ADR 0065 (b) had to narrow. A reader following the
documentation was being actively misled.

The clause now lives at `metrics.rs:174-216`, under the heading "`A +∞ score
latches here permanently, by decision`", covering: which operand positions
propagate and which discard, the "never evicted" interaction that makes it
permanent, the comparison to `avg_score`'s self-healing transit, the explicit
"this is not filtered, and must not be", both reachability paths from this
ADR's own Context, part 4, above, the `−∞` asymmetry, and a pointer to the
hardened pair.

### 2. Two additive accessors, and the pair is the decision

```rust
/// Highest finite score observed across all episodes.
#[must_use]
pub fn finite_best_score(&self) -> Option<f32>;   // metrics.rs:285-288

/// Lifetime count of episodes excluded from `finite_best_score`.
#[must_use]
pub fn non_finite_episodes(&self) -> usize;        // metrics.rs:348-351
```

Both `#[must_use]`, matching all nine existing accessors on the type.

The reasoning of 0070's Decision 2 is **inherited, not re-argued**: a hardened
statistic shipped without its exclusion count reports a clean number over an
unknown fraction of the data, which is the silent sanitization 0065 refuses and
0061 forbids generally.

The **local sharpening**, which is stronger than the mean's version of the same
argument: a hardened *maximum* without a count cannot distinguish "one
saturating episode in a million" from "every episode since step 4000". And
unlike a windowed mean, a maximum offers **no other signal at all** from which
to reconstruct the difference — a mean at least moves as the window rolls, so
its trajectory carries information about frequency. A latched maximum never
moves again. Its value is bit-identical in the benign case and the
catastrophic one. The count is the *only* channel that distinguishes them, which
makes it a condition of `finite_best_score` existing rather than an ergonomic
extra (this ADR's own Literature section, third bullet, below).

### 3. The predicate is `is_finite()`

Adopted from 0070's Decision 3 rather than re-argued: `±∞` is as destructive as
`NaN` to the statistics that consume it, `FiniteRewardGuard::admit` is
`if reward.is_finite()` (`shared.rs:715-716`), and the three predicates must
agree about what "bad" means rather than each carrying a private definition.

The argument **unique to the maximum side**, and the one that makes the choice
not merely conventional but forced: `f32::max` already discards `NaN` for free.
A `!is_nan()` predicate here would therefore make `finite_best_score`
observationally identical to `best_score` at **every input except the one — `+∞`
— that it exists for.** It would be pure decoration: a new public accessor, a
new field, a new rustdoc, and a hardened name over a statistic hardened against
nothing that was not already handled. Pinned by
`all_three_non_finite_kinds_are_excluded_and_counted` (`metrics.rs:1131`), which
constructs one history on which the three candidate predicates report three
different counts — `!is_nan()` → 1, `is_infinite()` → 2, `is_finite()` → 3.

### 4. Empty and all-non-finite both yield `None` — and the two pairs are unified

`finite_best_score()` returns `None` when nothing has been recorded **and** when
no finite score has ever been recorded. The rejections from 0070's Decision 4
are inherited unchanged:

- **`Some(0.0)`** fabricates a score the agent never earned (ADR 0061).
- **ADR 0069's `−∞` sentinel** (`crates/rlevo-evolution/src/fitness.rs:367-384`,
  empty branch `:374-375`) is deliberately **not** adopted. `−∞` is meaningful in
  `rlevo-evolution` only because ADR
  [0023](0023-objective-sense-and-maximize-convention.md) makes it the
  maximise-native worst value; `rlevo-reinforcement-learning` has no
  objective-sense convention, so `−∞` there is a claim about the *agent*, not
  about the *absence of data*.

Both are pinned by `all_non_finite_history_has_no_finite_best`
(`metrics.rs:1231`) and `empty_stats_report_no_finite_best_and_no_non_finite_count`
(`metrics.rs:1252`).

**The unification clause, discharging 0070's third reopen trigger by name.**
That trigger required the two finite/raw pairings to "be unified in naming,
predicate, and empty-case behaviour rather than each arguing its own case".
As of this ADR:

| axis | `finite_avg_score` | `finite_best_score` |
|---|---|---|
| prefix | `finite_` | `finite_` |
| predicate | `is_finite()` | `is_finite()` |
| empty | `None` | `None` |
| all-non-finite | `None` | `None` |
| return type | `Option<f32>` | `Option<f32>` |

Agreement on all five. **Trigger 3 is discharged.**

The two **count** accessors deliberately do *not* share a name, and that is not
a gap in the unification — see this ADR's own Decision 5, below.

### 5. The count is a monotone lifetime counter, and the naming makes that audible

`non_finite_episodes` parallels `total_episodes`. `non_finite_recent_len`
parallels `recent_len`. Reading either at a call site tells you its scope
without a trip to the docs, which is the entire justification for two names
where one sounds sufficient.

They are not the same quantity:

- `non_finite_recent_len` is **derived from the window on every call** and
  **self-heals** — once the window rolls past the offending episode it returns
  to zero.
- `non_finite_episodes` is **stored, monotone, and never heals** — it answers
  "did this run *ever* go bad?", not "is it bad *now*?".

**The two will disagree on a recovered run, and that disagreement is the
feature.** It is the same split ADR 0065's home-and-shape decision (Decision 1)
drew for the six agents' `dropped_transitions()`, which is likewise a monotone
lifetime counter
(`dqn_agent.rs:557`, `c51_agent.rs:576`, `qrdqn_agent.rs:537`,
`sac_agent.rs:656`, `td3_agent.rs:660`, `ddpg_agent.rs:608`). Collapsing them
would force a choice between the two questions, and the hardened *maximum* needs
the lifetime answer specifically — a maximum is a lifetime statistic, so pairing
it with a windowed count would produce a hardened value and an exclusion count
that describe different spans of the run.

The disagreement is executable rather than asserted:
`eviction_heals_the_non_finite_count` (`metrics.rs:988`) and
`non_finite_episodes_is_a_lifetime_count_not_a_window_count`
(`metrics.rs:1181`) run the **same history** through the **same eviction** and
expect **opposite answers**. Neither subsumes the other; together they are the
only executable statement of the split.

### 6. `record` gains state, and ADR 0070's no-stored-state decision (Decision 5)'s objection does not transfer

This is the clause a future reader will most want, because on its face this ADR
does the thing 0070 declined to do.

0070's Decision 5 reads: "`AgentStats` gains **no field**, and `record` … is not
touched." It gave one substantive reason, and it was specific:

> A stored counter would have to be **decremented on `pop_front`**, which is a
> new invariant coupling a counter to a deque eviction. That is a second
> correctness obligation on the hot path of `record`, and it is precisely the
> shape a later refactor breaks silently: an eviction path that forgets the
> decrement yields a monotonically drifting count with no failing assertion.

**That objection is entirely about eviction coupling, and a lifetime statistic
has none.** Nothing about `finite_best_score` or `non_finite_episodes` changes
when the window rolls. `record` is **append-only** with respect to both: it
folds or increments on the way in and never revisits. The `pop_front` at
`metrics.rs:139-141` is untouched and touches neither field. There is no
decrement to forget, so the failure mode 0070 was protecting against cannot
occur.

**And the alternative does not exist.** For the windowed count, 0070 had a real
choice: derive-on-call was correct, merely O(window). Here derive-on-call is not
a slower way to get the same answer — it is a *different, wrong* answer, because
a lifetime maximum cannot be reconstructed from a window that evicted the record
which set it (this ADR's own Context, part 3, above). 0070 anticipated exactly
this in its own second reopen trigger: "a lifetime statistic cannot be derived
from state that has been evicted, so this ADR's own Decision 5 falls and
`record` must maintain state after all."

So this is not a reversal of 0070's no-stored-state decision (Decision 5). It
is that decision applied to the case it explicitly scoped itself out of.

The window invariant `recent_history().len() <= window_size()` — the invariant
the #407 privatisation exists to protect — is untouched, and remains something
only `record` maintains.

### 7. The counter saturates

`non_finite_episodes` uses `saturating_add(1)` (`metrics.rs:135`), matching
`total_episodes` (`:127`) and `total_steps` (`:128`), settled under #408. A
long-lived agent is precisely the caller that reaches the ceiling, and for it a
monotone clamped counter beats both a silently rewound total and an "attempt to
add with overflow" panic on the hot path. `record`'s own rustdoc now names all
three counters together (`:117-125`).

The increment sits in the `else` arm of a single exhaustive `if
score.is_finite()` (`:132-136`), so **`non_finite_episodes() <=
total_episodes()` holds by construction** — every episode takes exactly one
branch, and the branch that increments the non-finite count is a subset of the
episodes counted by `total_episodes`. The invariant needs no assertion because
there is no code path that could violate it.

Pinned by `non_finite_episodes_saturates_instead_of_wrapping`
(`metrics.rs:1208`), which asserts on the *value* (`usize::MAX`) rather than on
"it did not panic" — a panic-shaped test passes against an unfixed `+=` under
`cargo test --release`, where the addition wraps silently instead.

### 8. `usize`, not `dropped_transitions()`'s `u64`

Three grounds, each sufficient:

1. **The invariant is against a `usize`.** `non_finite_episodes() <=
   total_episodes()` is the property this ADR's own saturation decision
   (Decision 7, above) establishes by construction, and `total_episodes` is a
   `usize` (`metrics.rs:52`). A `u64` counter would
   make the comparison a cross-width one.
2. **Both counters increment in the same call and must saturate together.** They
   share a `record` invocation; a `u64`/`usize` mismatch would break the
   invariant *at saturation on a 32-bit target*, where `total_episodes` clamps
   at `u32::MAX` while a `u64` non-finite count keeps climbing past it. The
   invariant would then be violated by the saturation policy itself — the worst
   possible place for it to fail, because that is the regime nobody tests.
3. **`dropped_transitions()`'s `u64` is not a workspace convention to follow.**
   It counts *transitions* (`shared.rs:749`, returning
   `FiniteRewardGuard::dropped()`), on a type that has no `usize` sibling to
   stay consistent with. Different quantity, different neighbours, no shared
   invariant. Copying its width here would be cargo-culting a type the way
   0070's Decision 4 declined to cargo-cult 0069's `−∞` constant.

### 9. No `f64`, and no numerics change anywhere

`finite_best_score` carries **no `#[allow]`** — unlike `finite_avg_score`, which
carries two (`metrics.rs:517`, `:521`). Stated explicitly to pre-empt a
symmetry-driven "shouldn't this be `f64` too?", which is the obvious question
after 0070's `f64`-accumulator decision (Decision 6) widened the mean's
accumulator.

**A maximum is a *selection*, not a *reduction*.** ADR 0069's accumulator-width
rule exists because a reduction combines terms into a running value that can
leave the finite range even when every term is inside it. A maximum never
combines anything: it returns one of its inputs, unmodified. There is **no
accumulator**, so there is no width to widen, no precision to lose, and no
intermediate that can overflow. `f64::from(score)` followed by a narrowing would
be provably a no-op — every value `finite_best_score` can return is a value some
episode already reported as an `f32`.

Nothing else in the type changes numerically either: `avg_score`,
`finite_avg_score`, and `non_finite_recent_len` are all untouched by this
change.

## Rejected alternatives

- **Filter inside `best_score` itself** — the one-line "obvious" fix, and the
  primary rejection. Three grounds:
  1. **It fabricates.** `+∞` is the true maximum observed. Returning the
     second-highest value under the name "best score" reports a number the agent
     never earned, which is ADR 0061's rule stated for a metric rather than a
     tensor decode.
  2. **It destroys ADR 0065's second detection channel for `+∞`** with nothing
     shipped alongside to say what was excluded. After the filter, no accessor
     on the type would report that a `+∞` episode ever happened — the raw value
     is gone and there is no count. The `avg_score` channel does not cover it
     either, because it self-heals within one `window_size`.
  3. **It silently changes the meaning of an existing public accessor.** Every
     current caller — including
     `crates/rlevo/tests/integration_test.rs:418` and any downstream user — gets
     a different contract with no signature change to notice. That is the
     mechanism ground (2) of ADR 0070's Rejected alternatives section rules
     out generally: reversing an accepted decision requires a superseding ADR
     (`docs/rules.md:649`), not a patch.

  Pinned as rejected by mutant (a) of
  `positive_infinity_latches_best_score_but_not_finite_best_score`
  (`metrics.rs:1061`), whose first assertion is `assert_eq!(stats.best_score(),
  Some(f32::INFINITY))` — the ADR 0061 ruling made executable rather than prose.

- **A count derived from the window**, i.e. the shape of 0070's no-stored-state
  decision (Decision 5) reused for `non_finite_episodes`. Rejected on two
  independent grounds, either fatal:
  - **It is wrong the moment `total_episodes() > window_size()`**, which is the
    normal case, not an edge case: `window_size` is a hard-coded `100` at every
    in-crate call site and is bounded by `MAX_BUFFER_CAPACITY` at construction
    (`metrics.rs:97-103`). Any run of consequence exceeds it in the first
    minute.
  - **It would make the accessor self-heal**, which is precisely the property
    the defect does not have. A telemetry channel that heals from a latch is
    worse than no channel: it would report zero excluded episodes next to a
    `best_score` still pinned at `+∞`, actively contradicting the value it
    exists to explain. This is the mutant
    `non_finite_episodes_is_a_lifetime_count_not_a_window_count`
    (`metrics.rs:1181`) kills.

- **`!is_nan()` as the predicate.** As this ADR's own Decision 3 (above)
  explains: `f32::max` already handles the `NaN` half, so this predicate
  produces an accessor observationally identical to the one it is meant to
  harden at every input but `+∞`.

- **A bundled metrics struct**, e.g. `finite_score_summary() ->
  FiniteScoreSummary { best, mean, excluded }`. ADR 0070's three grounds carry
  forward unchanged — `AgentStats` is a flat accessor surface, a new public type
  in a pre-1.0 crate needs its own `#[non_exhaustive]` reasoning with a longer
  tail than the decision being made, and `StrategyMetrics` bundles because it is
  a snapshot value type that crosses crate boundaries and is serialized together
  (ADR [0015](0015-shared-typed-metric-registry-crate.md)) while `AgentStats` is
  a live accumulator read in place.

  **And a verdict on 0070's first reopen trigger, since a reader may think this
  change fires it.** It does not: that trigger is scoped to "a third *mean*
  accessor over the same window", and its cost argument was the repeated
  O(window) scan that a bundle would fuse into one pass. **Both accessors added
  here are O(1) field reads** (`metrics.rs:286-288`, `:349-351`). There is no
  scan to fuse. The bundle's one concrete advantage is worth *strictly nothing*
  in this case, so the ADR-0070 trade does not merely fail to fire — it comes
  out further against bundling than it did in 0070. The accessor surface reaches
  eleven, which is a real cost (this ADR's own Consequences section, below),
  and this ADR answers it with a tighter successor trigger rather than a
  struct.

- **Clamping `+∞` to `f32::MAX`**, the shape ADR 0034's metrics decision
  (Decision 4) uses for fitness. Rejected specifically *because* this is a
  maximum. Clamping
  fabricates a score of `3.4e38` that ranks **above every real episode** while
  **looking finite** — so the latch persists in full (every subsequent
  comparison still loses), the corruption is now invisible to a reader eyeballing
  the number, and it is invisible to anyone grepping for `is_finite()` or
  `is_infinite()` while triaging. That is the worst of both branches: the
  behaviour of the unfixed code with the diagnosability of neither. The fitness
  rule earns its place in `rlevo-evolution` because a clamped fitness still
  *ranks* correctly against a population under ADR 0023's convention; there is
  no population here, only a running fold, and a fold has nothing to rank
  against.

- **A `debug_assert!` or config-gated check.** Rejected by name for the fourth
  time — ADR [0056](0056-non-finite-loss-skip-and-warn-guard.md)'s
  unconditional-execution decision (Decision 4) settled it ("the host read
  already exists, so a gate would strip the guard from exactly the long
  release runs that diverge"), 0065's own unconditional-execution decision
  (Decision 5) adopted that ruling unchanged, 0070 cited it a third time. The
  argument is *stronger* here than in either: this ADR's own Context, part 4
  (above)'s saturation path requires a long episode, so long release runs are
  not merely where the failure is likeliest, they are the only place one of
  the two paths exists at all. A release no-op protects nothing.

## Consequences

### Positive

- **The latch is legible from the API rather than absent from it.** Before this
  change, no rustdoc, test, or comment in the repository said `best_score` could
  latch, and 0065's un-narrowed sentence implied it could not. It is now the
  IDE tooltip at every call site (`metrics.rs:174-216`).
- **A caller who needs a usable lifetime maximum has one**, with the exclusion
  count that makes it honest, and with a predicate that covers both reachability
  paths from this ADR's own Context, part 4 (above) — including the one no
  ingestion guard can see.
- **ADR 0065's channels all survive intact.** The raw value still reports what
  happened, `FiniteRewardGuard` is byte-for-byte untouched, the eight
  `episode_reward +=` sites are untouched, and `dropped_transitions()` still
  fires on the env-reward path.
- **0070's third reopen trigger is discharged rather than deferred** (this
  ADR's own Decision 4, above): the two finite/raw pairs agree on prefix,
  predicate, empty case, all-non-finite case, and return type.
- **The windowed/lifetime distinction is executable**, in a mirrored test pair
  (`metrics.rs:988` and `:1181`) rather than in prose alone.
- No public signature change, no behaviour change to any existing accessor, no
  numerics change anywhere, no contract retracted.

### Negative / accepted costs — do not soften any of these

- **`AgentStats` now carries eleven accessors split along two orthogonal axes**
  (finite/raw × windowed/lifetime), four of which encode the split in their
  names: `total_episodes`, `total_steps`, `best_score`, `finite_best_score`,
  `non_finite_episodes`, `recent_history`, `recent_len`, `window_size`,
  `avg_score`, `finite_avg_score`, `non_finite_recent_len`. And the two *count*
  accessors deliberately carry **different names for a quantity that sounds
  identical when spoken**. This is a permanent comprehension tax and a permanent
  opportunity to pick the wrong one — the same shape as 0070's "two means now
  exist for one quantity", one axis larger. Accepted for the same reason: the
  alternative is a single accessor that lies to one of its two audiences. The
  mitigation is documentation and nothing stronger.
- **`record` now carries state that only `record` can maintain**, which 0070's
  no-stored-state decision (Decision 5) explicitly declined to introduce. This
  ADR's own Decision 6 (above) explains why the specific objection does not
  transfer, append-only-ness bounds the risk, and
  three tests pin the behaviour (`metrics.rs:1160`, `:1181`, `:1208`) — but the
  obligation is **real and did not exist before this change**. `record` is now a
  function whose correctness depends on two fields staying in step with the
  history, and that is one more thing a future refactor can get wrong.
- **`AgentStats` derives `Clone` (`metrics.rs:49`); every clone now copies two
  more fields** — an `Option<f32>` and a `usize`. Trivial against a `VecDeque`
  of up to `MAX_BUFFER_CAPACITY` records, and stated rather than omitted.
- **Nothing mechanically forces a caller to read `non_finite_episodes()`
  alongside `finite_best_score()`.** This ADR's own Decision 2 (above) argues
  the pair is indivisible; `#[must_use]` does not make it so, because it says
  nothing to a caller who
  never calls. The bundled-struct alternative *would* have enforced it and was
  rejected on other grounds — that trade is real and is not being papered over.
- **`best_score()` still latches at `+∞`, forever.** This ADR **does not fix the
  reported symptom.** It makes the symptom legible and offers an alternative
  beside it. A user who reads only `best_score()`, or an existing call site that
  is not updated, sees exactly what #1078 reported, in exactly the same
  circumstances. **That is the decision, not an oversight** — filtering the raw
  maximum is rejected in this ADR's own Rejected alternatives section on three
  grounds — but anyone arriving here from the issue expecting the number to
  change should be told plainly that it does not.

### Neutral

- No new dependency, no new public type, no new module.
- `AgentStats` derives only `Debug` and `Clone` (`metrics.rs:49`) — **no
  `serde`** — so there is no wire format and no persisted format, and no
  `rlevo-metrics-registry` entry (ADR 0015) is affected by adding accessors.
- ADR 0065's `FiniteRewardGuard` (`shared.rs:681,715-716`), its decade `warn!`
  schedule, its six agent drop tests, and the eight `episode_reward +=` sites
  enumerated in this ADR's own Context, part 4, above, are all byte-for-byte
  untouched.
- Test placement is unchanged: everything added is an in-source unit test under
  `#[cfg(test)] mod tests`, per ADR [0012](0012-split-heavy-examples-into-rlevo-examples.md).

## Reopen triggers

Any one of these reopens this ADR:

1. **A twelfth accessor of any kind on `AgentStats`** — deliberately tighter
   than 0070's first reopen trigger, "a third windowed mean". At eleven, with two
   orthogonal axes already encoded in the naming and two same-sounding counts
   held apart by prefix alone, the flat surface is at the limit of what naming
   can carry. The next addition should re-argue the bundled struct from scratch
   rather than inherit this ADR's rejection of it.
2. **A checkpoint-on-best-score feature lands.** It must read
   `finite_best_score()`, not `best_score()` — this ADR's own Context, part 5,
   above — and the question of whether a run that produced a `+∞` episode
   should be *checkpointable at all* is a policy question this ADR does not
   answer and should not be assumed to. It deserves its own record.
3. **Anything in this crate begins to branch on `non_finite_episodes`.** The
   saturation rationale in this ADR's own Decision 7 (above) rests explicitly
   on `record`'s own statement that "nothing in this crate branches on the
   counters, they are
   reported statistics" (`metrics.rs:124-125`). A clamped counter is a fine
   statistic and a poor predicate; the moment control flow depends on it, the
   saturation policy needs re-deciding rather than inheriting.
4. **`episode_reward` becomes `f64`, or a per-episode finiteness guard is added
   to the training loops.** The first removes this ADR's own Context, part 4
   (above)'s saturation path (an `f64` accumulator will not saturate over any
   realistic episode); neither removes the env-reward path, which is required
   to stay open by 0065's bookkeeping decision (Decision 4). A change that
   closes one path and is described as closing "the" path is the misreading
   this trigger exists to catch.
5. **Anyone proposes collapsing `non_finite_episodes` and
   `non_finite_recent_len` into one accessor.** This ADR's own Decision 5
   (above) is the record of why there are two; the collapse forces a choice
   between "did this run ever go bad?" and "is it bad now?", and pairs a
   lifetime maximum with a windowed count.

## Correction to ADR 0070

ADR 0070 is **not edited** (`docs/rules.md:649`; `docs/adr/README.md:3-6`). The
corrections are carried here, following the precedent set by ADR 0066's
"Correcting the record" section (`0066:117-142`, the correction itself at
`0066:133-137`) and ADR 0067's own "Correction to ADR 0065" section
(`0067:201-215`) for exactly this situation — and which 0070 itself reused for
0065. As at `0066:125` and in 0070's own correction section, **all line
numbers resolve against the commit that wrote them.**

### (a) Stale citations — and it is the exact failure 0070 warned about one section earlier

0070's correction to ADR 0065 (b) pins the two `best_score` NaN tests at
`metrics.rs:402,447`. Those are **pre-diff coordinates**: at `acd7b97^` the
`fn` lines were 402 and 447, at `acd7b97` they were 587 and 633, and under the
#1078 diff now in the working tree they are:

| 0070 cites | now | what is there |
|---|---|---|
| `metrics.rs:402` | **`metrics.rs:774`** | `fn nan_score_does_not_poison_best_score` (argument position of the fold) |
| `metrics.rs:447` | **`metrics.rs:820`** | `fn nan_as_first_record_does_not_latch_best_score` (receiver position) |

0070's correction to ADR 0065 (a) had *just* written, one section earlier:

> a correction table that goes stale inside its own diff is the failure mode
> this section exists to fix, and it very nearly did.

It did not nearly happen. **It happened two sections later, in the same
document, to the citation supporting the very defect #1078 reports.**

0070's own References section carries the same staleness throughout, all from
the #1078 diff shifting every coordinate below `:56`:

| 0070 cites | now | what is there |
|---|---|---|
| `metrics.rs:119-131` (`record`) | **`:126-143`** | `record` |
| `metrics.rs:124` (the `f32::max` fold) | **`:131`** | `self.best_score = Some(self.best_score.map_or(score, \|b\| b.max(score)))` |
| `metrics.rs:194-216` (`avg_score`) | **`:381-440`** (rustdoc `:381`, body `:429-440`) | `avg_score` |
| `metrics.rs:242-253` (`avg_score` body, from 0070's correction to ADR 0065 (a)) | **`:429-440`** | `avg_score`'s body |
| `metrics.rs:140,151,161,172,181,189` (six `#[must_use]` accessors) | **`:152,163,217,359,368,376`** | `total_episodes`, `total_steps`, `best_score`, `recent_history`, `recent_len`, `window_size` |
| `metrics.rs:159-160` (the never-evicted `best_score`) | **`:171-172`** | "Unlike [`Self::avg_score`], the best score is never evicted by the sliding window." |
| `metrics.rs:391-394` (the `#[cfg(test)]` doc comment carrying the contract) | **`:754-765`** | the row-1.2b passage, now also naming ADR 0070 |
| `metrics.rs:93-99` (the `window_size` ceiling, 0070's no-stored-state decision, Decision 5) | **`:97-103`** | the `MAX_BUFFER_CAPACITY` assertion |
| `metrics.rs:49` (`derive(Debug, Clone)`) | `:49` — **unchanged** | still correct |

**0070's reasoning is entirely unaffected.** Every cited mechanism is intact:
`record` still folds with `f32::max` in the same expression, `avg_score` still
sums the window in `f32`, the two NaN tests still cover the two operand
positions and still pass, and the six original accessors are still six. Only
coordinates moved. This is a citation correction, not a premise correction —
unlike ADR 0066, which corrected a premise 0065 reasoned *from*.

Non-`metrics.rs` citations in 0070's own References section were re-resolved as
well and are **still accurate**: `strategy.rs:357-370` (`mean_fitness`/`broken_count`),
`fitness.rs:367-384` (`sanitized_mean`, `:378`/`:381` `#[allow]`s, `:374-375`
empty branch), `shared.rs:681,715` (`FiniteRewardGuard`), and
`macros.rs:62,78,105-118`. Two ranges in `assert.rs` are off by one line at the
tail — 0070 cites `:22-34,45-51`; the functions are `:22-35`
(`assert_improves_over_random`, `is_finite` at `:24` and `:28`) and `:46-52`
(`assert_reaches`, `is_finite` at `:47`). The assertions 0070 relies on are all
present and unchanged.

### (b) One claim from ADR 0070's Context, part 2, is spent, not wrong

ADR 0070's Context, part 2, is headed "One of eight training loops explains
itself" and records that only DQN carried the `DELIBERATE:` comment while "the
other seven sites in the table above are a bare `episode_reward +=
reward_f32;` with no comment, no ADR citation, and nothing to stop a reader
concluding it is an oversight". Its own Consequences section then lists closing
that gap as future work: "the seven undocumented `episode_reward +=` sites stop
being an invitation to re-file #409".

**Commit `acd7b97` closed it.** All eight sites now carry the `DELIBERATE:`
block, at the line numbers tabulated in this ADR's own Context, part 4, above.
The claim was true when
written and is now spent. It is recorded here so that this ADR's own Context,
part 4 — which enumerates the same eight sites *with* their comments — is not
read as contradicting 0070, and so that the asymmetry is not re-asserted by a
future reader who finds 0070 before finding this.

### No correction to ADR 0065 is needed

Stated explicitly, because 0065 is the record this ADR's mechanism most directly
concerns. 0065's only claim about `best_score` is the parenthetical at
`0065:178-180` ("`best_score` is **not** poisoned — `metrics.rs:71` uses
`f32::max`, which ignores a NaN operand …"), and **0070's correction to ADR
0065 (b) already narrowed it** to "cannot corrupt the best-ever record *with a
`NaN`*", correcting the `metrics.rs:71` coordinate at the same time. That
narrowing is correct and this ADR is the follow-up it anticipated. Nothing
further in 0065 requires amendment; the bookkeeping decision (Decision 4)'s
"Episode return is deliberately left poisoned" clause (`0065:162-177`) is not
merely preserved but **relied upon** by this ADR's own Context, part 4, above.

## Literature

Deliberately short, and built on ADR 0070's own Literature section and ADR
0065's own Literature section rather than restating them.

- **IEEE 754 `maxNum`, and the half-guarantee it gives.** The standard
  introduced the NaN-suppressing `minNum`/`maxNum` forms precisely because plain
  `min`/`max` are not NaN-safe, and Rust's `f32::max` follows that form. That is
  **why the `NaN` half of this defect does not exist** — the fold discards a
  `NaN` in either operand position, which is what `metrics.rs:774` and `:820`
  pin. The standard says nothing about infinities for the simple reason that
  there is nothing to say: **`+∞` *is* the correct maximum**, and propagating it
  is the operation performing exactly to specification. The defect is not in
  `f32::max`; it is in reading a correct maximum as a usable one.

  ADR 0070's own Literature section had already named the complementarity —
  "clipping neutralises `±∞` but not `NaN`; `f32::max` neutralises `NaN` but
  not `+∞`" — and filed the second half as a live gap. **This ADR is that
  sentence coming due.** The general ruling is ADR 0066's, unchanged: where
  correctness depends on a non-finite value's fate, pin it with an explicit
  predicate rather than relying on an operation's implicit behaviour. This
  ADR's own Decision 3 (above) is that ruling applied.

- **Order statistic vs. moment — new to this ADR, and the formal content of
  this ADR's own Decision 6 (above).** A mean is a **moment**: a reduction over
  all terms, in which one
  contaminated term contaminates the result *proportionally* and the
  contamination **dilutes and then vanishes** as the window rolls past it. Its
  breakdown behaviour is bounded in both magnitude and time. A maximum is an
  **order statistic**: it selects a single term, so an extreme value is diluted
  by nothing at all, and once `+∞` is selected **no subsequent observation of any
  magnitude can displace it** — the selection is absorbing. Classical robust
  statistics puts the two at opposite ends of the same scale: the mean has a
  breakdown point of 0 but recovers as data ages out; the maximum has a
  breakdown point of 0 and, over a growing sample with no eviction, never
  recovers.

  This is the precise reason the remedy that sufficed for the mean cannot work
  here, and it is not a matter of degree. 0070 could derive its count from
  resident data because the mean's contamination is *coextensive with the window*
  — while the bad value matters, the bad value is still there to count. The
  maximum's contamination **outlives its own evidence**. Hence a stored field
  (this ADR's own Decision 6, above), a monotone rather than self-healing count
  (this ADR's own Decision 5, above), and a hardened accessor that is a
  genuinely different computation rather than a filtered view of the same one.

- **Missing-not-at-random, reused rather than re-derived** (ADR 0070's own
  Literature section; ADR 0065's reward-guard schedule decision, Decision 3).
  Excluding non-finite episodes is statistically inert if their cause is
  policy-independent numerical noise, and systematic if the excluded episodes
  correlate with the states the policy actually visits — the realistic case,
  where the agent's own improving behaviour drives it into the regime that
  saturates the return. No property of the estimator distinguishes the two
  regimes; only telemetry from the run can. That is the load-bearing reason
  `non_finite_episodes()` is a **condition** of `finite_best_score()` existing
  (this ADR's own Decision 2, above) rather than an ergonomic extra, and it
  binds harder here than for the mean: a hardened maximum, unlike a hardened
  mean, carries no trajectory from which a reader could infer the frequency
  for themselves.

## References

- Issue **#1078** — "[rl] `AgentStats::best_score` latches at `+∞` permanently,
  worse than the `avg_score` case (#409)". Resolved as its *second* option; its
  analysis is adopted in full (this ADR's own Context, part 1, above) and the
  policy choice it declined to make is made in this ADR's own Decision 1
  (above).
- Issue **#409** / ADR
  [0070](0070-avg-score-transits-non-finite-scores-the-hardened-mean-is-additive.md)
  — **extended and corrected**. Its Decisions 1-4 are inherited verbatim; its
  no-stored-state ruling (Decision 5) is scoped rather than reversed (this
  ADR's own Decision 6, above); its third reopen trigger is discharged (this
  ADR's own Decision 4, above) and its first reopen trigger is given a verdict
  (this ADR's own Rejected alternatives section); its stale `metrics.rs`
  citations and one spent claim from its Context, part 2, are corrected here
  without editing it.
- ADR [0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md) — its
  home-and-shape decision (Decision 1)'s monotone `dropped_transitions()`
  counter, mirrored by this ADR's own lifetime/windowed split (Decision 5,
  above); its bookkeeping decision (Decision 4)'s "Episode return is
  deliberately left poisoned" clause (`0065:162-177`), which this ADR's own
  Context, part 4, above, **relies on** and which remains in force verbatim.
  Its `best_score` parenthetical (`0065:178-180`) needs no further correction
  — 0070's correction to ADR 0065 (b) already narrowed it.
- ADR [0034](0034-fitness-hygiene-chokepoint-convention.md)'s metrics decision
  (Decision 4) — the mean-plus-`broken_count` shape 0070's Decision 2 mirrored
  and this ADR inherits (`crates/rlevo-evolution/src/strategy.rs:359,369`), and
  the `f32::MAX` fitness clamp this ADR's own Rejected alternatives section
  declines to reuse for a maximum.
- ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) — the
  no-fabrication rule behind this ADR's own Decision 1, its own Decision 4's
  rejection of `Some(0.0)`, and ground (1) of its own Rejected alternatives
  section.
- ADR [0069](0069-sanitized-fitness-is-reduced-in-f64.md) — the `f64`
  accumulator rule this ADR's own Decision 9 (above) shows has **no purchase**
  on a selection, and the `−∞` empty-case sentinel this ADR's own Decision 4
  (above) again declines to adopt
  (`crates/rlevo-evolution/src/fitness.rs:367-384`, empty branch `:374-375`).
- ADR [0056](0056-non-finite-loss-skip-and-warn-guard.md)'s unconditional-
  execution decision (Decision 4) — the no-`debug_assert`/config-gate ruling,
  adopted by ADR 0065's own unconditional-execution decision (Decision 5), cited by
  0070, and rejected by name a fourth time here.
- ADR [0066](0066-clamp-nan-behavior-is-backend-specific-pin-with-is-nan.md)
  (`:117-142`, correction at `:133-137`) and ADR
  [0067](0067-non-finite-observations-are-dropped-at-replay-ingestion.md)
  (`:201-215`) — the precedent for recording a correction to an immutable ADR in
  a later one, including the guard that cited line numbers resolve only against
  the commit that wrote them.
- ADR [0023](0023-objective-sense-and-maximize-convention.md) — the
  maximise-native `−∞` worst-value convention that
  `rlevo-reinforcement-learning` does **not** have, which is why this ADR's own
  Decision 4 (above) returns `None`.
- ADR [0015](0015-shared-typed-metric-registry-crate.md) — why `StrategyMetrics`
  bundles and `AgentStats` does not, and why no registry entry is affected.
- ADR [0012](0012-split-heavy-examples-into-rlevo-examples.md) — in-source unit tests, where every
  test added by this change lives.
- `docs/rules.md:649` and `docs/adr/README.md:3-6` — the immutability rule that
  makes this ADR's own Correction to ADR 0070 section, above, the only
  available mechanism.

**Code citations, all resolved against the working tree at the time of writing:**

- `crates/rlevo-reinforcement-learning/src/metrics.rs`:
  - `:11-12` — `PerformanceRecord::score`, "the primary scalar metric used for
    checkpointing and best-model tracking" (this ADR's own Context, part 5,
    above).
  - `:49` — `#[derive(Debug, Clone)]`: no `serde`, hence no wire format.
  - `:52,54` — `total_episodes`, `total_steps` fields (the `usize` neighbours of
    this ADR's own Decision 8, above).
  - `:56,58,60` — the `best_score`, `finite_best_score`, and
    `non_finite_episodes` fields; the latter two are the state this ADR's own
    Decision 6 (above) adds.
  - `:97-103` — the `MAX_BUFFER_CAPACITY` ceiling on `window_size`.
  - `:117-125` — `record`'s rustdoc, now naming all three saturating lifetime
    counters; `:124-125` is the "nothing in this crate branches on the counters"
    sentence this ADR's own third reopen trigger (above) fences.
  - `:126-143` — `record`. The `f32::max` fold at `:131`, the guarded finite
    fold at `:132-133`, the saturating increment at `:135`, the untouched
    `pop_front` at `:139-141`.
  - `:171-172` — "the best score is never evicted by the sliding window"
    (this ADR's own Context, part 1, above).
  - `:174-216` — `best_score`'s new latch section (this ADR's own Decision 1,
    above); `:217-220` — its unchanged body.
  - `:222-288` — `finite_best_score`; `:290-351` — `non_finite_episodes`.
  - `:381-440` — `avg_score` (unchanged); `:442-524` — `finite_avg_score`, with
    its two `#[allow]`s at `:517` and `:521`; `:526-580` —
    `non_finite_recent_len`, the derived windowed count.
  - Tests: `:774` and `:820` (the two NaN tests 0070's correction to ADR 0065
    (b) miscites); `:886` (`best_score_survives_eviction_while_average_does_not`);
    `:988` (`eviction_heals_the_non_finite_count`); `:1061`, `:1085`, `:1111`,
    `:1131`, `:1160`, `:1181`, `:1208`, `:1231`, `:1252`, `:1271` (the ten tests
    added by this change).
- `crates/rlevo-reinforcement-learning/src/algorithms/shared.rs:681` —
  `FiniteRewardGuard`; `:715-716` — `admit`'s `if reward.is_finite()`; `:749` —
  `dropped() -> u64`, the different quantity this ADR's own Decision 8 (above)
  declines to match.
- The six agents' `dropped_transitions() -> u64`: `dqn_agent.rs:557`,
  `c51_agent.rs:576`, `qrdqn_agent.rs:537`, `sac_agent.rs:656`,
  `td3_agent.rs:660`, `ddpg_agent.rs:608`.
- The eight `episode_reward +=` sites and their `DELIBERATE:` comments,
  tabulated in this ADR's own Context, part 4, above: `dqn/train.rs:129,137`,
  `ppo/train.rs:194,204`, `ppg/train.rs:135,148`, `sac/train.rs:122,130`,
  `td3/train.rs:106,114`, `ddpg/train.rs:121,129`, `c51/train.rs:122,130`,
  `qrdqn/train.rs:115,123`.
- `crates/rlevo-test-support/src/assert.rs:22-35`
  (`assert_improves_over_random`, `is_finite` at `:24`, `:28`) and `:46-52`
  (`assert_reaches`, `is_finite` at `:47`) — the acceptance gates that read
  `avg_score` and **never** `best_score` (this ADR's own Context, part 6,
  above); fed from `crates/rlevo-test-support/src/macros.rs:62,78`.
- The two external reads of `best_score()`:
  `crates/rlevo-examples/examples/book/ch03_dqn_cartpole.rs:179` and
  `crates/rlevo/tests/integration_test.rs:418` (this ADR's own Context, part 5,
  above).
- `crates/rlevo-evolution/src/strategy.rs:359,369` — `mean_fitness` /
  `broken_count`; `crates/rlevo-evolution/src/fitness.rs:367-384` —
  `sanitized_mean` and its `−∞` empty branch at `:374-375`.
- IEEE 754-2008 — `minNum`/`maxNum`, the NaN-suppressing variants whose
  guarantee covers exactly half of this defect's domain.
