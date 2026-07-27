---
project: rlevo
status: active
type: decision
date: 2026-07-27
tags: [adr, decision, numerical-stability, nan, reward, replay, dqn, c51, qrdqn, sac, ddpg, td3, issue-352]
---

# ADR 0065: A non-finite reward is dropped at replay ingestion, counted, and warned on a decade schedule

## Status

**Accepted (2026-07-27).** Resolves issue #352. **Counterpart to ADR 0056**,
which deliberately deferred reward finiteness at `remember` by number (0056
§Consequences: "Out of scope, deliberately: reward finiteness at `remember`
(**#352**)"). Purely additive — supersedes nothing, changes no public API
signature, changes no healthy-step numerics.

**Chosen shape.** A `FiniteRewardGuard` in `algorithms/shared.rs`, one
instance per agent, whose `admit(&mut self, reward: f32) -> bool` runs a
host-side `f32::is_finite()` check at the top of each of the six off-policy
agents' `remember`. On a non-finite value the transition is **not pushed**
into the replay buffer; the call is a no-op. The drop fires on **every**
occurrence — a run emitting NaN every step must be protected every step —
but the `warn!` is scheduled rather than fired every time: it escalates at
`dropped == 1, 10, 100, 1000, …`, carrying the running total each time. A
public `dropped_transitions() -> u64` accessor on each agent exposes the
count. The guard runs **unconditionally in release** — by the time `remember`
is called, the reward has already been erased to a plain `f32`, so the check
is a register compare with no device sync.

## Context

### Six sites, not four

`dqn_agent.rs:386`, `ddpg_agent.rs:417`, `td3_agent.rs:466`,
`sac_agent.rs:476`, `c51_agent.rs:405`, `qrdqn_agent.rs:366` all call
`remember` with an unguarded `reward: f32`. Issue #352 names only the first
four; C51 and QR-DQN were added later by copying an unguarded `remember` from
an earlier agent. This is the evidence for two decisions below: a shared
`FiniteRewardGuard` struct over six inline copies (a copy-paste omission is
exactly how two of six sites went missing the first time), and a requirement
that each of the six agent files carry its own drop test rather than trusting
one shared unit test to stand in for all six call sites.

### The harm is not weight corruption — that chain is already broken

The issue's own title ("permanently poisons the target network") is stale and
is corrected here. ADR 0056's `FiniteLossGuard` already breaks the
NaN-reward → NaN-loss → NaN-weights chain before `backward()` runs: a
non-finite reward becomes a non-finite TD target becomes a non-finite loss,
and 0056 skips the backward pass on any non-finite loss, on every
occurrence. Weights do not get corrupted by a NaN reward today.

The residual harm is narrower but real: the poisoned transition **lingers in
the FIFO buffer** until capacity eviction. Every minibatch that resamples it
yields a non-finite loss that 0056 silently skips — silently, because 0056's
`warn!` is a **one-shot latch per loss site**, so only the *first* such skip
is ever logged. The result is unlogged throughput loss (a fraction of every
subsequent minibatch draw is wasted on a transition that can never train
anything) with no way for an operator to see it happening.

### C51 does not escape the loss guard, and the escape route is worth tracing precisely

`c51/projection.rs:97-101` might suggest C51's index-based projection sidesteps
a NaN reward, since C51 does not compute a scalar TD-error the way DQN does.
It does not escape, verified by trace: NaN reward → `tz` NaN
(`projection.rs:141`) → the `clamp` at `:142` propagates NaN rather than
rescuing it (`NaN.clamp(..)` is `NaN`, documented in this file's own comment
at `:97`) → propagates through `:156-157` → `l_idx = NaN.floor().int()`
saturates to `0` under Rust's `as`-cast, but `weight_lower = next_probs *
(u_f - b + mask)` at `:169` still multiplies by the NaN `b`, so the
resulting distributional loss is NaN and 0056's guard fires as designed.

**Open question, recorded rather than claimed away:** the `0`-saturation of
`NaN.floor().int()` under `as` is a Rust language guarantee (`as`-cast
float→int saturates rather than producing an unspecified bit pattern), but
that guarantee is about the *host* numeric cast. On a GPU backend, the
float→int conversion inside a device kernel is not contractually pinned by
the same rule, and an out-of-range scatter index reaching device code would
be a panic on user-supplied runtime data — a `rules.md:232` violation
reachable only through this one hole. This has **not** been tested on a GPU
backend and is **not** claimed to break; it is recorded as an open question
so it is not silently re-discovered. Filed as **#1044** so a maintainer with
GPU hardware can settle it.

### The guard cannot live in the buffer

`ReplayStrategy::push` (`replay/mod.rs:249`) is generic over `T` — the buffer
has no way to know `T` carries a reward field, let alone check it.

### Why not a validating `Transition::try_new`

Every field of `Transition<O, P>` is `pub` (`replay/transition.rs:54-75`), and
the public doctest at `:34-50` constructs one with struct-literal syntax. A
validating constructor placed *alongside* fully public fields is unenforceable
decoration — nothing stops the struct-literal path a public doctest already
exercises. Making the seam real would mean privatising a public type's fields
and rewriting every agent's staging path; that is a materially larger and
differently-scoped change than closing #352.

### The tightest in-repo precedent is `Priority`, not ADR 0060

`Priority::try_new` (`replay/priority.rs:115`) and `from_td_error`
(`replay/priority.rs:142`) already reject NaN/±Inf on a **runtime-derived**
float at a buffer boundary, with a typed error, and
`update_priorities_from_td_errors` (`prioritized.rs:337`) writes nothing when
one is bad. This is structurally the same problem this ADR solves — a
runtime float, computed from live rollout data, being rejected at the exact
point it would otherwise enter buffer state. ADR 0060 is about *config*
values, checked once at construction from source-literal-adjacent data;
`Priority` is about exactly this case, checked continuously against a live
data stream. Lead with `Priority` as precedent, not with 0060.

## Decision

### 1. Home & shape

`FiniteRewardGuard { dropped: u64, next_threshold: u64, label: &'static str }`
in `algorithms/shared.rs`, alongside `FiniteLossGuard` (ADR 0056). `admit(&mut
self, reward: f32) -> bool` returns `true` ⇒ proceed with `push`; `false` ⇒
skip it. `dropped_transitions(&self) -> u64` is a public, non-`#[cfg(test)]`
accessor (unlike `FiniteLossGuard::warning_fired`, this one is operationally
useful outside tests — it is the surfacing channel a caller can poll without
grepping logs). Each of the six agents holds one guard field.

### 2. Guard sits at the top of `remember`, before `push`

The point is to keep a non-finite reward from ever entering
`ReplayStrategy`'s backing store at all, so no later consumer — not the loss
guard, not a future prioritized-replay resampling, not an eventual offline-RL
export — has to re-derive that it might see one.

### 3. Drop semantics: every occurrence, escalating warn

The drop itself **re-fires on every non-finite occurrence** — same rule as
0056's skip, for the same reason: a run that emits NaN every step must be
protected every step, and latching the drop would silently readmit poison
after the first occurrence. What is scheduled is only the `warn!`: it fires
at `dropped_transitions() == 1, 10, 100, 1000, …` (i.e. on powers of ten),
each time reporting the running total. This bounds log volume to
~log₁₀(n) lines — at most about 7 over a run reaching a billion dropped
transitions — while still surfacing magnitude, not just occurrence. See
§Literature for why magnitude specifically, not just an occurrence flag, is
the thing that must be observable.

### 4. Bookkeeping on a drop — stated explicitly so it is not "fixed" later

- **Env-step counter is unaffected.** `on_env_step()` is a separate call
  (`dqn/train.rs:127`) made *after* `remember` (`dqn/train.rs:120`). The
  environment step genuinely happened regardless of whether the reward was
  admitted. ε-decay, `learning_starts`, and ADR 0059's gradient-update-keyed
  target cadence all advance exactly as if the drop had not occurred.
- **`can_learn` is correctly delayed by the drop count.** You cannot learn
  from data you refused to accept, so a dropped transition does not count
  toward buffer fill. The degenerate case is real and intentional: an
  environment that emits NaN on every step yields a buffer that never fills
  and a run that does nothing. That is the *correct* outcome — the
  alternative is training on entirely fabricated or entirely missing reward
  signal — and the escalating `warn!` (§3) is what makes that outcome
  diagnosable instead of a silent hang.
- **Episode return is deliberately left poisoned.** `episode_reward +=
  reward_f32` at `dqn/train.rs:137` still adds the NaN unconditionally, so
  that episode's return is NaN, and `AgentStats::avg_score`
  (`metrics.rs:92-102`) reports NaN for the next `window_size` episodes. This
  is a real tension with 0056 §3 ("skipped values are excluded from
  epoch-mean accumulators so a single NaN cannot re-poison an otherwise-finite
  reported mean") and is addressed here head-on rather than left implicit: a
  loss is an internal training diagnostic, and excluding a skipped one from
  its running mean is bookkeeping hygiene. An episode return is the
  **primary scientific measurement** this whole library exists to produce.
  Silently omitting one step's reward from it would report a return the
  agent never actually earned — exactly the fabrication ADR 0061 rules out
  for tensor decoding, applied here to a metric. The NaN is a true statement
  about what happened in that episode, and it is a second surfacing channel
  that does not share the `warn!`'s decade schedule, which matters precisely
  because the two channels can disagree about *when* they fire.
  (`best_score` is **not** poisoned — `metrics.rs:71` uses `f32::max`, which
  ignores a NaN operand, so a single bad episode cannot corrupt the
  best-ever record.)

### 5. Runs unconditionally in release

No `debug_assert`/config gate. See §Rejected alternatives — this mirrors
0056 §4's ruling and the reasoning transfers unchanged: the host read the
guard needs already exists (the reward is `f32` by the time `remember` sees
it), so a gate would strip the guard from exactly the long release runs
where a diverging environment is most likely to emit the values it exists to
catch.

## Rejected alternatives

- **`debug_assert!`.** This is the C51 code-review's §1.3 recommendation and
  appears by name in this issue's own comment thread, so it is rejected by
  name. ADR 0056 §4 already settled the general question for this codebase
  ("No `debug_assert`/config gate: the host read already exists, so a gate
  would strip the guard from exactly the long release runs that diverge"). A
  release no-op protects nothing in the runs where protection matters most.
- **Sanitize to `0.0` / clamp.** `f32::clamp` propagates NaN
  (`c51/projection.rs:97` already documents this in-tree), so a naive
  `reward.clamp(lo, hi)` is a silent no-op against NaN specifically — see
  §Literature for the ∞-vs-NaN distinction. A *deliberate* NaN→`0.0`
  substitution is worse than doing nothing: it manufactures a reward the
  environment never emitted and is indistinguishable downstream from a
  legitimately-zero step. ADR 0061's no-fabrication rule is directly against
  this. Absence of data is recoverable (you can note it happened and reason
  about it); wrong data that looks like right data is not.
- **`panic!`/`assert!`.** `rules.md:231-232` — panics are for programming
  errors, never for user-supplied runtime data, and an environment's reward
  is exactly that.
- **`remember -> Result`.** This is #317's job — a breaking signature change
  to six public methods, which needs its own ADR justifying the break ahead
  of landing it, not a rider on this one. The shape chosen here is
  forward-compatible with that change: `admit(reward) -> bool` becomes
  `?`-able (`admit(reward).then_some(()).ok_or(..)?` or equivalent) with no
  semantic change when #317 lands.
- **Validating `ScalarReward::new`** (`rlevo-core/src/reward.rs:44`). Closed
  outright, not deferred. Three independent reasons: the tuple field is
  `pub` *by documented intent* (`reward.rs:26-28`), so a validating `new`
  next to a public field is theatre, exactly the same shape rejected for
  `Transition` above; `new` is `const fn`, which cannot run a runtime
  validity check at all; and decisively, `Reward` is a **trait**, and
  `Environment::step`'s `RewardType` bound constrains nothing about which
  concrete type implements it — an environment may ship a reward type that
  never touches `ScalarReward`, so a core-level check would not close the
  hole even where it could run. `remember(reward: f32)` is the universal
  chokepoint precisely because it is the point where every reward type has
  already been erased to `f32`, regardless of which `Reward` impl produced
  it. The guard belongs where the erasure happens, not upstream of it.
- **Six inline copies, one per agent.** The C51/QR-DQN omission (§Context) is
  the evidence against this: it is exactly what already happened once, and a
  shared struct with a per-agent test in each of the six files is how it
  does not happen again.

## Literature

Full citations, quoted passages, and the ecosystem survey in
`docs/.private/research/2026-07-27-issue-352-reward-finiteness-at-ingestion.md`
(companion to `2026-07-21-issue-318-nonfinite-loss-guard.md`, ADR 0056's own
note, which deferred this issue by number).

- **No published precedent and no de-facto convention exists.** Stable-
  Baselines3 (`common/buffers.py`), RLlib (`utils/replay_buffers/
  replay_buffer.py`), Tianshou (`data/buffer/base.py`), OpenAI Baselines
  (`deepq/replay_buffer.py`), and Acme (`adders/reverb/base.py`) were each
  inspected at their `add`/adder path, and **none** validates reward
  finiteness before storing it. Dopamine's `circular_replay_buffer.py` could
  not be retrieved (every mirror 404'd) and is recorded as **unverified**,
  not as a negative result — 5/6 confirmed, 1/6 inaccessible. ADR 0065 is
  therefore *ahead of* established practice, not a port of it, and the
  honest framing is the stronger argument: none of the surveyed
  implementations check, and that absence is itself the gap this ADR closes.
- **Reward clipping does not subsume this guard, stated precisely.**
  `Inf.clamp(-1.0, 1.0)` correctly returns `1.0` — ±∞ orders normally under
  IEEE 754, so clipping *does* neutralize infinities. `NaN.clamp(-1.0, 1.0)`
  returns `NaN`, because IEEE 754 makes every ordering comparison against
  NaN false, which is exactly why IEEE 754-2008 introduced the
  NaN-suppressing `minNum`/`maxNum` as operations distinct from plain
  `min`/`max`. So clipping subsumes an ∞ guard but not a NaN guard — not
  overclaiming this: Mnih et al. (*Playing Atari with Deep Reinforcement
  Learning*, NIPS DL Workshop 2013, arXiv:1312.5602; restated in the 2015
  Nature Methods section) state, verbatim: "we fixed all positive rewards to
  be 1 and all negative rewards to be −1, leaving 0 rewards unchanged,"
  because "clipping the rewards in this manner limits the scale of the error
  derivatives and makes it easier to use the same learning rate across
  multiple games." That is an ∞-and-magnitude mitigation, not a NaN one, and
  rlevo does not clip rewards by default regardless.
- **The AMP/`GradScaler` precedent transfers as analogy, not authority — a
  deliberate departure from how 0056 used it.** Micikevicius et al. (*Mixed
  Precision Training*, ICLR 2018, arXiv:1710.03740, §3.2), verbatim: "One
  option is to skip the weight update when an overflow is detected and
  simply move on to the next iteration." AMP discards a **computation** — one
  optimizer step — because the corruption is an artifact of the fp16
  *procedure*, and the very same training example survives and contributes
  on a later pass once the loss-scale factor re-tunes. ADR 0065 discards a
  **datum**: a transition whose reward is non-finite is evidence the
  transition itself is invalid, and it never comes back. Same defensive
  instinct, different failure mode — cited here as the shape precedent for
  "detect non-finite, skip rather than propagate," not as authority for
  discarding data specifically.
- **The statistical argument, independent of the operational one, and the
  strongest justification for decade escalation specifically.** PER's
  importance-sampling correction (Schaul et al., *Prioritized Experience
  Replay*, ICLR 2016, arXiv:1511.05952) does **not** apply here: it corrects
  graduated, error-correlated *over-sampling* of transitions that are all
  individually valid, whereas a finiteness reject is a hard 0/1 admission
  decision on an event whose true reward has no finite value — there is
  nothing left to re-weight. The question that actually matters is
  missing-data-theoretic, not a re-weighting one: if non-finite rewards come
  from a policy-independent numerical bug, the drop is missing-completely-
  at-random and statistically inert; if they correlate with the states the
  policy visits — the realistic case for a genuinely *diverging*
  environment, where the agent's own actions drive it into the regime that
  emits NaN — the drop is missing-not-at-random and carves a systematic hole
  in the visited-state distribution. **No paper answers which regime a given
  run is in; only telemetry from the run itself can.** This is an
  independent statistical argument for the same conclusion the architecture
  reached on purely operational grounds in §Decision 3: the count must be
  observable and the warn must convey magnitude, not merely occurrence, so
  the operator has a chance of distinguishing the two regimes rather than
  seeing one indistinguishable latched line.

## Consequences

Nothing in the existing test suite caught this, for the same structural
reason 0056 documented for the loss case: the cross-crate tests assert
`*_produces_finite_rewards`, which checks the **environment's** output — the
*input* side of this very boundary, not what the buffer accepted.
Reproducibility tests assert same-seed self-consistency, which a
deterministic NaN satisfies perfectly. **No test in the workspace has ever
asserted anything about replay-buffer contents.** Closing that gap is part of
this change: each of the six agents gets a drop test (transition not pushed,
`dropped_transitions()` increments, `can_learn` correctly delayed) in its own
file, per §Rejected alternatives' "six inline copies" reasoning.

## Out of scope, deliberately

- **PPO/PPG on-policy rollout** (`ppo/rollout.rs:131` →
  `compute_gae:349`). Filed as **#1042**. The reason is not
  convenience — the blast radius per occurrence is *larger* here than in the
  off-policy case: the reverse GAE recursion carries one NaN reward to every
  timestep `t' ≤ t` in the rollout, and `normalize_advantages`
  (`ppo/losses.rs:72-77`) then takes a batch mean and spreads it to every
  advantage in the batch, so one bad reward destroys the whole rollout's
  advantage estimates. But this ADR's semantic — "don't push the tuple" —
  does not transfer: a rollout is a **contiguous positional trajectory**, so
  dropping index `t` either desynchronises the parallel `Vec`s the rollout is
  stored in, or splices two non-adjacent timesteps into one GAE recursion —
  strictly worse than leaving the NaN in place. The real options for the
  on-policy case (treat the bad step as a truncation boundary with a
  bootstrap value; discard the whole rollout) are a separate design
  decision, not a variant of this one. Mitigating the urgency: the rollout
  buffer is cleared every iteration and ADR 0056's `FiniteLossGuard` already
  contains the resulting NaN loss, so the damage does not persist across
  iterations the way a buffered off-policy transition does — that asymmetry
  is what justifies filing this as a separate, later issue rather than
  folding it into #352's scope.
- **Non-finite observations at the same seam.** `obs`/`next_obs` are `O:
  Observation` and can equally carry a NaN into tensor staging with no
  guard today. Structurally identical hole, materially more expensive to
  close — checking an observation is a per-element tensor scan, not one
  `f32` register compare — so it is a different cost/benefit call and is
  left for a separate issue (**#1043**) rather than bundled here.
- **`ScalarReward` / `rlevo-core`.** Closed, not deferred — see §Rejected
  alternatives. There is no follow-up issue for this one; the trait-erasure
  argument there is decisive, not a matter of scope or cost.

## References

- Issue #352 — "[rl] No reward-finiteness guard at replay-buffer ingestion".
- ADR [0056](0056-non-finite-loss-skip-and-warn-guard.md) — the sibling
  guard this one complements: 0056 breaks the poisoned-reward → poisoned-
  weights chain at the loss; this ADR stops the poisoned reward from
  persisting in the buffer in the first place. Explicitly deferred this
  issue by number at its own §Consequences.
- ADR [0060](0060-config-values-must-be-finite.md) — the config-layer
  sibling: "a value must be finite" applied to `*Config` fields checked once
  at construction. Cited for contrast, not as the tightest precedent — see
  `Priority` below for that.
- ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) —
  the no-fabrication rule this ADR leans on twice: once to reject
  sanitize-to-zero, once to justify leaving the episode-return NaN rather
  than silently excluding the step.
- ADR [0059](0059-target-update-cadence-counts-gradient-updates.md) — the
  gradient-update-keyed cadence that this ADR's §Decision 4 confirms
  advances unaffected by a drop.
- `crates/rlevo-reinforcement-learning/src/replay/priority.rs:115,142` —
  `Priority::try_new` / `from_td_error`, the tightest in-repo precedent:
  rejecting NaN/±Inf on a runtime-derived float at a buffer boundary.
- `crates/rlevo-reinforcement-learning/src/replay/transition.rs:34-75` — the
  fully-`pub` `Transition<O, P>` and its struct-literal doctest, the reason a
  validating constructor there is unenforceable.
- `crates/rlevo-reinforcement-learning/src/algorithms/c51/projection.rs:97-101,141-142,156-169`
  — the NaN-propagation trace through C51's distributional projection.
- `crates/rlevo-reinforcement-learning/src/algorithms/dqn/train.rs:120,127,137`
  — `remember` / `on_env_step` / `episode_reward +=` call ordering.
- `crates/rlevo-reinforcement-learning/src/metrics.rs:71,92-102` —
  `best_score`'s NaN-immune `f32::max` versus `avg_score`'s NaN-susceptible
  windowed mean.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. *Playing Atari with Deep
  Reinforcement Learning.* NIPS Deep Learning Workshop, 2013.
  arXiv:1312.5602.
- Micikevicius, P., Narang, S., Alben, J., et al. *Mixed Precision
  Training.* ICLR 2018. arXiv:1710.03740.
- Schaul, T., Quan, J., Antonoglou, I., Silver, D. *Prioritized Experience
  Replay.* ICLR 2016. arXiv:1511.05952.
- IEEE 754-2008 — `minNum`/`maxNum`, the NaN-suppressing variants
  introduced because plain `min`/`max` are not NaN-safe.
- Full ecosystem survey and reconciliation:
  `docs/.private/research/2026-07-27-issue-352-reward-finiteness-at-ingestion.md`.
