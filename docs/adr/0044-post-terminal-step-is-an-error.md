---
project: rlevo
status: active
type: decision
date: 2026-07-11
tags: [adr, decision, environments, core, episode-lifecycle, error-handling, toy-text, wrappers, issue-105]
---

# ADR 0044: Post-terminal `step()` is an error, not an absorbing transition

## Status

**Accepted (2026-07-11).** Resolves issue #105 ("[env] Post-terminal `step()`
not guarded across `toy_text`"). Supersedes nothing. Follows ADR 0040's
precedent for adding a **generic, structured** `EnvironmentError` variant, and
the ADR 0026/0027/0039 "make invalid states unrepresentable" posture. The
family-by-family rollout it opens is tracked by #289.

## Context

None of the four `toy_text` environments tracked episode termination.
`step()` mutated state unconditionally, so a call made *after* the episode
ended silently resurrected it:

- **`cliff_walking`.** The goal is `(3, 11)`. A `Left` from the goal lands on
  `(3, 10)`, which *is* cliff — so the agent teleports to the start and the
  env emits **−100 on a `Running` snapshot**. A finished episode becomes a
  fresh one carrying a fabricated penalty.
- **`blackjack` (the High finding).** A post-terminal `Hit` keeps pushing
  cards onto `player_hand`, and `hand_value` summed the pips into a **`u8`**.
  After roughly 26 ten-valued cards the accumulator overflows: a **panic in
  debug**, a **silent wraparound in release** — and a wrapped total such as
  260 → 4 lands back inside the "valid" range, so the env emits a
  non-terminal snapshot with a nonsense reward. The overflow is a *symptom*;
  the missing lifecycle guard is the root cause.
- **`frozen_lake`.** After a Hole or Goal, the next step walks *off* the
  terminal tile and returns to `Running`.
- **`taxi`.** After a correct Dropoff, the next step keeps driving.

The survey that grounded this ADR found the gap is **family-wide, not
`toy_text`-specific**: no environment anywhere in the workspace guards a
post-terminal step — none of the ~48 `Environment` impls, and not the
`TimeLimit` wrapper. The four bandits already carry a `done: bool` field that
is **written but never read**, a half-implementation that also violates
`rules.md` §10 (`KArmedBandit::is_done()` is a second source of truth for
done-ness). #105's four files are one instance of that family.

Three facts bounded the blast radius and were verified before deciding:

1. **There is no vectorized / batched env driver in the workspace** (no
   `VecEnv`, no `dyn Environment`). The usual objection to rejecting — "a
   batched rollout keeps stepping every env after one of them finishes" — has
   no in-tree instance to break.
2. **The recording and TUI taps are safe.** `RecordingTap::step` and
   `TuiEnvTap::step` both call `self.inner.step(action)?` first and bump their
   counters only *after* the `?`, so an `Err` propagates with no torn wrapper
   state.
3. **`TimeLimit` was the one real casualty, and the defect is structural.**
   `TimeLimit::step` delegates to `inner.step(action)?` **first**, then upgrades
   `Running → Truncated`. The inner env therefore *never learns it was
   truncated* and its own guard still reads `Running` — so a guard placed below
   the wrapper **cannot, by construction**, reject a step taken after a
   truncation. The inner env would mutate, and the wrapper would re-stamp a
   second, fabricated `Truncated` snapshot.

### What the literature does and does not say

This is a research library, so the grounding matters — and it must not be
overclaimed.

Sutton & Barto's **absorbing state** (§3.4: a state that "transitions only to
itself and generates only rewards of zero") is introduced as part of the
*unified notation* device that lets the return
$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ be written as a single infinite
sum covering both episodic and continuing tasks. Puterman and Bertsekas
formalize the same construct in the stochastic-shortest-path setting
($p_{TT}(a) = 1$, zero cost, absorbing). In both cases the absorbing state is a
statement about the **mathematical process** — it makes the return well-defined
— not an operational specification for what a `step()` function should do when a
caller invokes it out of sequence.

Reference implementations do not fill that gap either. Gymnasium does **not**
reject post-terminal steps: its `OrderEnforcing` wrapper guards only
*reset-before-step*, and `PassiveEnvChecker` has no post-terminal check at all.
`CartPole` emits a one-time `logger.warn` ("...any further steps are undefined
behavior") and then **keeps integrating the physics**, so the observation drifts
onward past termination. Old Gym's docstring said it outright: "further `step()`
calls will return undefined results."

So: **the literature is silent on the API contract.** Rejecting a post-terminal
step is not derived from RL theory, is not contradicted by it, and is strictly
more rigorous than any reference implementation. This ADR records an
**engineering judgment**, and says so plainly.

The decisive engineering argument is asymmetry:

> **Absorbing semantics are recoverable from reject semantics via a wrapper.
> Reject semantics are not recoverable from absorbing semantics.**

A caller who genuinely wants a self-looping, zero-reward absorbing tail can
build a wrapper over a rejecting env. A caller who wants to *find the bug*
cannot un-absorb an env that has already silently appended an unbounded tail of
zero-reward transitions into their replay buffer. Reject is the reversible
choice; absorb is a one-way door into silent data bias. `rules.md` §4 forbids
exactly that class of quiet corruption ("never panic in response to
user-supplied runtime data"; "trait methods that can fail return
`Result<T, DomainError>`"), and Rust has `Result`.

## Decision

### 1. `step()` after the episode ends returns `Err` — it does not absorb

A finished episode is not silently resumed and its terminal snapshot is not
silently repeated. The call sequence is the caller's bug and is reported as
one. Absorbing-state re-emission is **not** the default; if it is ever wanted,
it is an explicit opt-in wrapper over a rejecting environment.

### 2. A generic, structured `EnvironmentError::StepAfterEpisodeEnd`

```rust
/// `step()` was called after the episode already ended.
#[error("step() called after the episode ended ({status:?}); call reset() before stepping again")]
StepAfterEpisodeEnd {
    /// The status that ended the episode (`Terminated` or `Truncated`).
    status: EpisodeStatus,
},
```

It carries the [`EpisodeStatus`] rather than being a unit variant because that
is the **only** thing telling the caller *which layer* ended the episode:
`Terminated` means the MDP ended, `Truncated` means a wrapper's step budget did.
It is `Copy` and one byte. It deliberately carries **no environment-name
`String`** — that is the stringly-typed, single-purpose shape ADR 0040 rejected.

`EnvironmentError` is now **`#[non_exhaustive]`**, so a future variant is not a
breaking change. This is safe for existing code: `#[non_exhaustive]` on an
*enum* constrains only exhaustive `match` outside `rlevo-core`; it does not
block variant construction (that would require `#[non_exhaustive]` on a
*variant*). Every in-tree constructor of `EnvironmentError::InvalidAction(..)`
&c. is unaffected.

### 3. `EpisodeStatus` is the stored state — never a `done: bool`

Per `rules.md` §10, `EpisodeStatus` is the single source of truth for episode
termination. A guarded environment stores an `EpisodeStatus`, not a boolean.
Concretely, **`EpisodeStatus` gains no `NotStarted` variant**: it is the
snapshot's *public* status type, and a fourth variant would ripple through every
env, tap, reporter and record consumer to serve a concern (step-before-reset)
that no snapshot ever needs to express.

### 4. One crate-level `EpisodeGuard`, in `rlevo-environments`

`crates/rlevo-environments/src/episode.rs` (new) holds a one-field state
machine — `new` / `reset` / `check` / `record` / `status`, all `const fn`,
`Debug + Clone + Copy + Default`:

```rust
pub struct EpisodeGuard { status: EpisodeStatus }   // field PRIVATE
```

It lives at **crate level**, not in `toy_text/mod.rs`, because this change alone
has **two consumers in two different module families**: the `toy_text`
environments and the `wrappers::TimeLimit` wrapper. `grids/core/` is the
precedent for *intra*-family helpers; this is cross-family. It cannot live in
`rlevo-core` — that is a contract crate and §1 forbids implementation logic
there (sole exception: `util`).

The `status` field is **private**, and this is load-bearing twice over. It means
(a) an environment cannot hand-write a guard state that the snapshot it emitted
disagrees with, and (b) the representation is a **two-way door**: swapping it to
`Option<EpisodeStatus>` (`None` = never reset) to close the step-before-reset
hole (#296) is a private, non-breaking change.

### 5. Guard placement: first statement, single exit

Every guarded `step()` calls `guard.check()?` as its **first statement** —
before any state mutation **and before any RNG draw**. A rejected step must not
advance the environment's RNG stream (ADR 0029 makes the env's RNG a persistent,
observable part of its state); each guarded env pins this with a test.

Each `step()` was restructured to a **single exit** that `record()`s the status
off the snapshot it actually returns, so the guard and the emitted snapshot
cannot drift apart.

### 6. A wrapper that manufactures a terminal status owns the guard for it

This is the generalizing rule, and it is the clause to cite when the next
wrapper is written.

`TimeLimit` carries its **own** `EpisodeGuard`. It `check()`s **before**
delegating (so a rejected step never reaches, and never mutates, the wrapped
env) and `record()`s `snap.status` **after** the `Running → Truncated` upgrade —
so one mechanism covers both an inner-env `Terminated` and a self-imposed
`Truncated`. `TimeLimit::reset` delegates to the inner reset **first** and clears
the guard only on success: clearing first would re-open a finished episode even
when the inner reset failed, leaving the wrapper willing to step an environment
that never returned to its initial state.

An inner guard is *provably insufficient* here (Context, fact 3). Any future
wrapper that synthesises a terminal status must guard at its own layer.

### 7. Blackjack: widen the accumulator, saturate the return

`hand_value` sums the pips into a **`u16`** and saturates to `u8::MAX` on the way
out. The signature stays `(u8, bool)`, so the **public** `player_sum: u8` fields
on `BlackjackState` and `BlackjackObservation` — serialized and tensor-encoded —
are untouched. Naively widening the return type to `u16` would have been a public
API change smuggled in as a two-line fix.

Saturation is sound for every consumer: `255 > 21`, so a saturated total is still
classified as a **bust**, which is the only meaningful reading of a hand that
large. The usable-ace branch only adds 10 when the total is at most 11, far below
saturation.

With decision 1 in place this path is *unreachable*, so this is defence in depth
— but it is also independently required by `rules.md` §4 ("never panic in
response to user-supplied runtime data"), and it converts a debug panic / release
wraparound into a defined, correctly-classified value.

### 8. The contract is normative in the trait rustdoc, with a disclosed gap

`Environment::step`'s rustdoc now states the contract as **normative**
("implementations **must** return `EnvironmentError::StepAfterEpisodeEnd`..."),
with an explicit **alpha migration note**: only `toy_text` and `TimeLimit`
enforce it today, every other environment's post-terminal behaviour is
**undefined**, and callers must not rely on it — see **#289**. The `# Errors`
section names the new variant (§6).

"Recommended / permitted" was the tempting middle option and is a trap: a
contract nobody can rely on is not a contract, and it gives the author of env #49
permission to skip it. Normative-plus-disclosure declares the target (so
reviewers have a rule to cite and new envs are written correctly) without the
docs lying about the 44 envs that do not yet comply. **Deleting the migration
note is an acceptance criterion on #289.**

### 9. Fixture environments are exempt — by decision, not by accident

Mock / stub / fixture environments (`rlevo-test-support`, the in-crate
`StubEnv`/`MockEnvironment` types) are **not** required to conform. They are test
scaffolding, not environments a policy trains against. The exemption is
deliberate and has a concrete payoff: `TimeLimit`'s test `StubEnv` is left
**unguarded on purpose**, so the wrapper's tests exercise *the wrapper's own*
guard rather than accidentally testing the inner env's. (The one guarded fixture,
`MockGuardedEnv` in `episode.rs`, exists precisely to smoke-test the conformance
helper.)

### 10. Conformance is *checkable*, not *enforced*

`#[cfg(test)] pub(crate) fn assert_rejects_post_terminal_step(env, drive_to_done,
replay_action)` in `episode.rs` is the shared conformance check. It drives the
env to a terminal snapshot, asserts that snapshot is done, then asserts a further
`step` returns `StepAfterEpisodeEnd` **carrying the same status that ended the
episode** — pinning the variant *and* the payload, so a guard that reports
`Truncated` for a termination fails. `replay_action` is a legal action: the guard
must reject on call-sequence grounds alone, never on the action's own validity.

This makes each family's follow-up issue mechanical: "hold an `EpisodeGuard`,
make this assertion pass."

## Consequences

### Positive

- The four reported `toy_text` defects are closed at the root. The blackjack
  `u8` overflow — panic in debug, silent wraparound into the valid range in
  release — is unreachable *and* saturated.
- `TimeLimit`'s post-**truncation** hole, which no inner-env guard could have
  caught, is closed at the layer that created it, under a rule (decision 6) that
  generalizes to every future wrapper.
- **The contract found a latent test defect on day one.** The pre-existing
  `frozen_lake` test `slippery_mean_direction_differs_from_action` hand-places
  the agent and steps 10,000 times; roughly a third of those iterations land on a
  hole, so the harness was **silently stepping past termination** — it was
  relying on the bug. It now calls `env.guard.reset()` alongside its hand
  placement (not `reset()`, which would also move the agent off the tile it is
  about to overwrite). That a guard added for production correctness immediately
  surfaced a real defect in a passing test is evidence the contract is doing
  work, not decoration.
- A caller can now distinguish "you stepped after the MDP terminated" from "you
  stepped after a wrapper truncated you" from the error alone.
- Verified: `cargo test --workspace` 0 failures, `cargo clippy --workspace
  --all-targets --all-features` 0 warnings, `cargo fmt --all --check` clean.

### Neutral

- Correctly-sequenced callers are unaffected. Every production training loop,
  rollout, bench and example in the workspace already checks `is_done()` and
  breaks or resets, so the new error essentially never fires in existing code.
- `EnvironmentError` gaining `#[non_exhaustive]` broke nothing in-tree: the sole
  workspace `match` over it (a `rlevo-core` unit test) is inside the defining
  crate and already carries a wildcard arm.

### Negative / accepted costs

- **44 environments are non-conformant on day one.** Their post-terminal
  behaviour remains undefined. This is disclosed in the trait rustdoc and tracked
  by **#289** (umbrella) with per-family issues #290 classic, #291 grids (12),
  #292 locomotion, #293 box2d, #294 `pixel_grid`, #295 bandits. The migration
  note is deleted when #289 closes.
- **The guard is opt-in. Nothing forces env #49 to call it.** The conformance
  helper is a *mitigation*, not enforcement — it makes compliance checkable, not
  automatic. The only design that would make the contract unbreakable is rejected
  below, and its cost is higher than this residual risk.
- **`rlevo-hybrid/src/rollout_fitness.rs:95` `.expect()`s on `step`**, so a
  caller bug there becomes a **panic** rather than a propagated `Result`. That
  loop breaks on `is_done()` first, so it cannot fire today — and if it ever
  does, a loud panic on a genuine call-sequence bug is the intended outcome, not
  a regression.
- **A future `VecEnv` / batched driver must own auto-reset itself.** It may not
  rely on environments tolerating post-terminal steps to keep a partially-finished
  batch marching in lockstep. Stated explicitly here so it is not re-litigated the
  day someone adds one: the batching layer resets the finished env (Gymnasium's
  own autoreset posture), it does not weaken the environment contract.
- **The step-before-reset hole remains open** (#296) — the sibling
  mis-sequencing defect. Blackjack's pre-reset `player_hand` is empty, so
  `hand_value` returns 0 and a pre-reset step emits nonsense. Deliberately
  deferred. **The intended fix is to change `EpisodeGuard`'s *private*
  representation to `Option<EpisodeStatus>` (`None` = never reset) — a two-way
  door — and explicitly *not* to add a variant to the public `EpisodeStatus`.**

## Alternatives considered

- **Absorbing-state re-emission as the default** (a post-terminal `step()`
  silently re-returns a stable terminal snapshot with zero reward). Rejected: the
  absorbing state is a definitional device for the return (Sutton & Barto §3.4;
  Puterman/Bertsekas SSP), not an API contract, and it does not describe what to
  do about a mis-sequenced *call*. Operationally it lets a buggy trainer push an
  unbounded tail of zero-reward transitions into a replay buffer, **biasing the
  data instead of surfacing the bug**. And it is the irreversible choice: absorb
  is recoverable *from* reject via a wrapper, never the other way round.
- **A template-method split on the trait** — make `step` a provided method that
  guards and then calls a required `step_inner`. Rejected: it is the only design
  that makes the contract *unbreakable*, but it is a **one-way door on the core
  trait**, renaming a method across all ~48 impls, and it is **incompatible with
  the wrapper pre/post pattern** — `TimeLimit`, `RecordingTap` and `TuiEnvTap` all
  need to do work *around* the inner call, which a fixed provided `step` forbids.
  The checkable-conformance helper buys most of the value at a fraction of the
  cost.
- **`EnvironmentError::InvalidAction("step called after termination".into())`.**
  Rejected on two grounds: it is stringly-typed (the exact shape ADR 0040
  rejected), and it **misattributes the fault** — the *action* is perfectly legal;
  the *call sequence* is wrong. `rules.md` §4 requires structured variants.
- **A unit `StepAfterEpisodeEnd` with no payload.** Rejected: the carried
  `EpisodeStatus` is the only signal distinguishing an MDP termination from a
  wrapper-imposed truncation, which is precisely the distinction a caller
  debugging a mis-sequenced loop needs. It costs one `Copy` byte.
- **A `done: bool` field per environment** (completing the bandits'
  half-implementation). Rejected by `rules.md` §10: `EpisodeStatus` is the single
  source of truth for termination and done-ness is never checked by other means.
  A bool cannot carry the `Terminated` / `Truncated` distinction the error needs.
- **An `EpisodeStatus::NotStarted` variant** to also close step-before-reset.
  Rejected: `EpisodeStatus` is the snapshot's public status type; a state no
  snapshot can ever be in does not belong in it, and adding one would ripple
  through every env, tap, reporter and record consumer. See the `Option<_>` path
  in Consequences.
- **The guard in `toy_text/mod.rs`** (a per-family helper, following
  `grids/core/`). Rejected: `wrappers::TimeLimit` must consume it too, and
  `wrappers/` cannot reasonably reach into `toy_text/`. Two consumers in two
  families is a crate-level concern — which is a present requirement, not
  speculative generality.
- **A shared guard in `rlevo-core`.** Rejected by `rules.md` §1: core is a
  contract crate and carries no implementation logic (sole exception: `util`).
  Core owns the *contract* (the error variant, the trait rustdoc);
  `rlevo-environments` owns the *mechanism*.
- **Silence in the trait rustdoc until the family-wide sweep lands.** Rejected:
  it leaves `toy_text` behaving differently from 44 other envs with no documented
  reason, and gives new-env authors no rule to follow. Normative-with-disclosure
  is honest *and* prescriptive.

## References

- Issue #105 — post-terminal `step()` not guarded across `toy_text`.
- Issue #289 — umbrella: guard sweep across the remaining env families
  (**acceptance criteria include deleting the migration note from the
  `Environment::step` rustdoc**). Per-family: #290 classic, #291 grids (12),
  #292 locomotion, #293 box2d, #294 `pixel_grid`.
- Issue #295 — bandits: the dead `done: bool` field, plus the
  `KArmedBandit::is_done()` `rules.md` §10 second-source-of-truth violation.
- Issue #296 — `step()` before `reset()`, the sibling mis-sequencing hole
  (deliberately deferred; intended fix is `EpisodeGuard`'s private
  `Option<EpisodeStatus>`).
- ADR [0040](0040-environment-config-error-and-terrain-output-contract.md) — the
  precedent for adding a **generic, structured** `EnvironmentError` variant and
  rejecting a stringly-typed single-purpose one.
- ADR [0026](0026-shared-config-validation-convention.md) /
  [0027](0027-bounds-newtype-for-closed-ranges.md) /
  [0039](0039-box2d-states-own-markov-dofs.md) — the
  make-invalid-states-unrepresentable posture (private field + accessors) that
  `EpisodeGuard` follows.
- ADR [0029](0029-host-rng-seeding-convention.md) — environments own a persistent
  RNG stream; hence the guard must reject **before** any RNG draw, not merely
  before state mutation.
- ADR [0011](0011-lift-construction-off-environment-trait.md) — construction lives
  on `ConstructableEnv`, which is why the wrappers (`TimeLimit`, taps) exist as
  thin decorators that can carry their own guard.
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), §3.4 —
  the absorbing state ("transitions only to itself and generates only rewards of
  zero") as the **unified-notation device** making $G_t$ well-defined as one
  infinite sum across episodic and continuing tasks. Cited for what it *is*, and
  for what it explicitly is *not*: an operational spec for a `step()` API.
- Puterman, *Markov Decision Processes*; Bertsekas & Tsitsiklis, *Neuro-Dynamic
  Programming* — the SSP formalization of the same construct
  ($p_{TT}(a) = 1$, zero cost, absorbing). Again a statement about the process,
  not a software contract.
- Gymnasium `OrderEnforcing` / `PassiveEnvChecker` / `CartPoleEnv`, and Gym's
  historical `step` docstring ("further `step()` calls will return undefined
  results") — the reference implementations **do not** reject a post-terminal
  step; `OrderEnforcing` guards only reset-before-step, and CartPole warns once
  then keeps integrating. rlevo's contract is strictly more rigorous than, and
  not contradicted by, any of them.
- Code: `crates/rlevo-core/src/environment.rs`
  (`EnvironmentError::StepAfterEpisodeEnd`, `#[non_exhaustive]`, the
  `Environment::step` post-terminal contract);
  `crates/rlevo-environments/src/episode.rs` (`EpisodeGuard`,
  `assert_rejects_post_terminal_step`);
  `crates/rlevo-environments/src/toy_text/{blackjack,cliff_walking,frozen_lake,taxi}.rs`;
  `crates/rlevo-environments/src/wrappers/time_limit.rs` (wrapper-owned guard).
