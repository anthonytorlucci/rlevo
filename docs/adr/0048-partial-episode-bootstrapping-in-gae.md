---
project: rlevo
status: active
type: decision
date: 2026-07-18
tags: [adr, decision, reinforcement-learning, ppo, ppg, gae, truncation, peb, issue-170]
---

# ADR 0048: Partial-episode bootstrapping in GAE — truncation is not termination

## Status

**Accepted (2026-07-18).** Resolves the on-policy half of issue #170 ("[rl]
Truncation vs. termination bootstrap bias"). Reverses a deliberate tradeoff
previously documented in `algorithms/ppo/rollout.rs`; supersedes no ADR (the
tradeoff was recorded in a doc comment, never in an ADR).

## Context

`rlevo` distinguishes `EpisodeStatus::Terminated` from
`EpisodeStatus::Truncated` at the environment contract level, and `TimeLimit`
(`rlevo-environments/src/wrappers/time_limit.rs`) is an **opt-in wrapper** that
upgrades `Running → Truncated` — the horizon is never intrinsic to an
environment's MDP. That places every `rlevo` environment squarely in Pardo et
al.'s *time-unlimited* case (ii), for which the correct learning target is
**partial-episode bootstrapping** (PEB), their Eq. 6:

> y = { r, at environmental terminations; r + γ v̂_π(s′), otherwise (including timeouts) }

Gymnasium (Towers et al. 2025) states the same rule in boolean form: the
bootstrap mask is `¬terminated`, **never** `¬done`.

### What the code does instead

`compute_gae` (`crates/rlevo-reinforcement-learning/src/algorithms/ppo/
rollout.rs`) collapses `terminated || truncated` into a single
`next_nonterminal` term, so a truncation zeroes the bootstrap exactly like a
termination. The function's own doc comment names this as a deliberate choice:

> Matching CleanRL's default PPO, we accept this small bias on
> truncation-heavy envs. The terminated/truncated arrays are kept separate so
> a future revision can rework this without changing the buffer API.

CleanRL's ORing of the two flags is itself acknowledged upstream
(vwxyzjn/cleanrl #198, #457) as a simplification for a pedagogical reference
implementation, not an oversight. Matching it means knowingly implementing what
Pardo (§3) and Towers (§4.2) both identify as incorrect.

Two of that comment's claims do not survive contact with the evidence:

1. **The bias is not small.** Pardo et al. report *qualitative* failure, not
   numerical drift: on Two-Goal Gridworld (§3.1) the non-PEB agent "always
   tries to go for the closest goal even if there is not enough time"; on
   Hopper/Walker2d (§3.2) standard PPO trained on 300-step partial episodes
   collapses where PEB "significantly outperform[s]" it. The bias grows with γ
   (§2.4), and `rlevo` defaults to γ = 0.99.
2. **The buffer API cannot be preserved.** The comment assumes the fix is a
   *mask* change. It is not — it is a **missing value**. `ppo/train.rs`
   discards `next_snapshot` on a done step before its value is ever computed;
   the value network runs exactly `num_steps + 1` times per rollout and never
   on a truncation continuation state. `V(s_continuation)` does not exist
   anywhere in the process and must be newly computed at record time. Issue
   #170's claim that the infrastructure was "pre-built for this fix" is
   likewise refuted.

### Two masks, not one

At a truncated step the correct treatment splits:

- the **delta** bootstraps from `V(s_continuation)` — the agent's future is
  real, the clock merely stopped;
- the **λ-recursion is cut** — the trajectory genuinely ended and advantage
  must not propagate across the boundary.

A single `next_nonterminal` factor multiplies both terms and therefore cannot
express one without the other. The GAE paper itself (Schulman et al. 2016) does
not address rollout-boundary truncation at all; its §2 setup assumes
trajectories run "until a terminal (absorbing) state is reached." The truncated
form is a retrofit, and Pardo et al. (§3) note the one-step PEB rule is "more
complex" to carry into GAE's exponentially-weighted sum.

### An entangled, independent off-by-one

`push_step` stores `obs[t]` together with the status of the transition *out of*
`obs[t]`, so `terminated[t]` means "transition t ended the episode" — which is
exactly what decides whether `values[t + 1]` is a valid bootstrap. `compute_gae`
reads `terminated[t + 1]`. The `t == n - 1` branch instead consults the
`last_done` parameter, which `ppo/train.rs` and `ppg/train.rs` set from that
step's *own* status — the `[t]` convention. Two conventions coexist in one
loop, and the in-source test asserting the current values records the author
reasoning themselves into the wrong one ("Wait — the convention is…").

This is a **separate defect from the truncation bias** and a strictly larger
one: it mis-times the bootstrap cut for **terminated** episodes too, so it
affects every PPO and PPG user, not only those wrapping an environment in
`TimeLimit`. It is nevertheless not separable *in the code* — see Decision §4.

### Blast radius

PPG shares this code path outright: `ppg_agent.rs` imports
`ppo::rollout::RolloutBuffer` and there is no second GAE implementation in the
workspace. The only other caller of `compute_gae` is
`benches/ppo_bench.rs`. PPG's auxiliary phase draws its value targets from
`RolloutBuffer::returns()`, so it inherits the correction automatically with no
aux-specific change.

## Decision

### 1. Adopt PEB (Pardo Eq. 6) as the on-policy learning target

`rlevo` implements the SB3 / Pardo formulation and **deliberately diverges from
CleanRL's default PPO**. This is a fidelity decision recorded here so that a
reader who diffs `rlevo` against CleanRL and finds a mismatch learns it was
chosen, not missed. The former tradeoff is reversed, not silently patched.

### 2. Express the two masks explicitly inside `compute_gae`, not by folding the correction into the reward

SB3 achieves PEB by adding `gamma * terminal_value` to the stored reward before
GAE runs and then treating the step as done. That is algebraically identical and
was seriously considered (see Alternatives). `rlevo` instead carries the
bootstrap value as its own datum and applies it in the delta term, for three
reasons:

- **The stored `rewards` array must keep meaning "the reward the environment
  emitted."** It happens to be safe to mutate today — `rewards` is a private
  field with no accessor, nothing outside `compute_gae` reads it, and episode
  return metrics accumulate independently in the training loop before the buffer
  ever sees the value. That makes folding *currently harmless*, not *correct*:
  `RolloutBuffer` is `pub`, and the first `rewards()` accessor added for reward
  normalisation, a reward-model diagnostic, or a sequence buffer would silently
  read γ-contaminated numbers with no compile error and no test failure.
- **Testability.** Under the two-mask form the entire truncation rule lives
  inside a pure function of plain slices, unit-testable with a hand-computed
  expected value and no environment, agent, or backend. Under reward folding the
  rule lives in the training loop, which is the one place in this crate that has
  no cheap deterministic test. Folding moves correctness-critical arithmetic out
  of the testable half of the system and into the untestable half.
- **Legibility against the paper.** Eq. 6 says "bootstrap the delta, cut the
  recursion." The two-mask body says that in two adjacent lines. The folded form
  says it in two files.

Neither form is cheaper to plumb: **both** require computing
`V(s_continuation)` in the rollout loop before the reset. That work is identical
either way, so the choice is purely about where the correction is applied.

### 3. `truncated: Vec<bool>` becomes `truncation_value: Vec<Option<f32>>`

A truncation flag without its bootstrap value is a state the type system should
not permit, per the make-invalid-states-unrepresentable posture of ADR
0026/0027/0039/0046. Storing `Some(v)` for a truncated step and `None`
otherwise means `truncated[t] == truncation_value[t].is_some()` **by
construction** — there is no parallel-array skew to keep in sync, and no
zero-filled `&[f32]` that silently reads as "bootstrap from 0.0" when a caller
forgets to populate it.

The recursion-cut mask and the delta bootstrap then read directly off index
`[t]`:

```rust
let ended = terminated[t] || truncation_value[t].is_some();   // cut the λ-recursion
let boot = if terminated[t] {
    0.0
} else {
    truncation_value[t].unwrap_or(next_value)                 // PEB: V(s_continuation)
};
let delta = rewards[t] + gamma * boot - values[t];
last_gae_lam = delta + gamma * gae_lambda * (1.0 - ended as u8 as f32) * last_gae_lam;
```

### 4. The off-by-one is fixed in the same change, as its own commit

The two defects are independent, but they are **not separable in the code**. The
truncation bootstrap belongs to step `t` (it is the continuation of transition
`t`) and can only be read at index `[t]`. Landing PEB therefore forces the `[t]`
convention onto the terminated mask as well; leaving the mask at `[t + 1]` would
put *three* indexing conventions in one loop. Splitting the work across PRs
would also move every seeded PPO/PPG baseline twice instead of once.

They ship as one PR in two commits — the indexing fix first, standalone and
independently revertable, then PEB — which buys the reviewability and
bisectability of a split without the fiction that they can land weeks apart.
The off-by-one is filed as its own issue and closed by that first commit, so
the tracker and the CHANGELOG record it as the distinct, wider-reaching defect
it is.

### 5. `last_done` is deleted, not fixed

Once every step's own status is read at `[t]`, the `last_done` parameter is
redundant: `terminated[n-1] || truncation_value[n-1].is_some()` *is* the final
step's done-ness, recorded by `push_step` like any other. The parameter existed
only to paper over the `[t + 1]` convention running off the end of the arrays,
and it was the site at which the two conventions collided. `last_value` remains,
now used on exactly one path: the final step left the episode `Running`.

### 6. The continuation observation is passed to `record_step`; the agent computes the value

`PpoAgent::record_step` and `PpgAgent::record_step` take the observation the
environment just produced. The agent — which already owns the value network and
already runs it in `finalize_rollout` — computes `V(s_continuation)` **iff the
status is `Truncated`**. The caller passes an observation it already holds; it
never computes a value and cannot forget to. Cost is one extra value forward per
truncation and zero per ordinary step.

This is the load-bearing public API change: `record_step` is the seam a
hand-written training loop calls.

### 7. `compute_gae` stays `pub` and breaks; no deprecated shim

`compute_gae` is currently wrong on two independent counts. A
`#[deprecated]`-but-callable version would communicate "will be removed," not
"produces biased advantages" — the only honest shim would panic, which is
strictly worse than removing it. The workspace is at `0.1.0` alpha with an
established precedent for atomic breaking changes (ADR 0028, 0030, 0038, 0042);
the sole out-of-module caller is an in-tree dev-only bench.

The function remains `pub`. It is the one piece of this crate's mathematics that
is a pure function of plain slices, which is precisely what makes it cheap to
unit-test and to benchmark; hiding it behind `RolloutBuffer::finish` would
trade that away for nothing.

## Consequences

### Positive

- PPO and PPG learn the correct target on any `TimeLimit`-wrapped environment,
  and the correction reaches PPG's auxiliary value distillation for free via
  `returns()`.
- The terminated-episode bootstrap is correctly timed for the first time,
  independent of whether any time limit is in use.
- A truncated step without its bootstrap value becomes unrepresentable in the
  buffer.
- `compute_gae` shrinks by one parameter and gains a single indexing
  convention, both readable against Pardo Eq. 6 line by line.

### Negative / accepted costs

- **Breaking, with no shim**, at three public sites: `compute_gae`'s signature,
  `RolloutBuffer::{push_step, finish}`, and `{Ppo,Ppg}Agent::record_step`.
  Hand-written training loops must pass the continuation observation.
- **Every seeded PPO/PPG baseline moves.** Results are not comparable to
  pre-change runs, and — by design — are no longer comparable to CleanRL's
  default PPO on truncation-heavy environments. Baseline fixtures under
  `rlevo-test-support` must be re-measured, not re-fitted.
- **One extra value-network forward per truncation.** Negligible against a
  rollout, but it is a device round-trip on the collection path, which was
  previously forward-free except for the per-step `act`.
- **The in-source test asserting the current (buggy) values is deleted and
  rewritten**, not adjusted. It encodes the wrong convention in its comments as
  well as its numbers.

### Explicitly out of scope: residual GAE weighting bias

PEB fixes the *terminal bootstrap*. It does **not** fix everything wrong with
GAE at an episode boundary. Doering et al. (2026) identify "a previously
overlooked issue in truncated Generalized Advantage Estimation" — "the geometric
weighting scheme induces infinite mass collapse onto the longest k-step
advantage estimator at episode boundaries" — and report that "a simple weight
correction [yields] substantial improvements in environments with strong
terminal signal, such as Lunar Lander." Jin et al. (2025) and Fan et al. (T-PPO,
2025) pursue related truncation-aware estimators.

So: the *principle* implemented here is settled and uncontested, while the exact
mechanics inside GAE's weighted sum remain an open research question, and a
residual bias persists in `rlevo` even after this change. This is **recorded as
a known, deliberate non-decision**, not an oversight — it is a live research
area, not a bug with an agreed fix, and adopting an unreplicated weight
correction would be a worse trade than carrying a documented residual. It is
filed as its own issue.

## Alternatives considered

- **Keep the CleanRL convention.** Rejected: `rlevo`'s `TimeLimit` is an opt-in
  wrapper, never intrinsic to an MDP, which is exactly Pardo's time-unlimited
  case where the timeout-as-termination target is wrong. CleanRL's own
  maintainers frame the ORing as a simplification appropriate to a
  single-file pedagogical reference — a goal `rlevo` does not share.
- **SB3-style reward folding** (`rewards[t] += gamma * terminal_value`, then
  treat the step as done). Algebraically identical and genuinely tempting: it
  leaves `compute_gae`'s arithmetic almost untouched and matches the most widely
  read production implementation. Rejected on the two grounds in Decision §2 —
  it makes the stored reward array mean something other than "the environment's
  reward," which is safe today only by the accident that no accessor exists;
  and it relocates the rule from a purely unit-testable function into the
  training loop, where it can only be tested end-to-end.
- **Time-aware agents (Pardo Eq. 4), i.e. feeding remaining time into the
  observation.** This is the correct treatment for Pardo's *time-limited* case
  (i), where the horizon is part of the task. Rejected as the wrong case for
  `rlevo`, and it would require an observation-space change in every wrapped
  environment. A future environment whose horizon genuinely is part of its MDP
  should carry the remaining time in its own observation and report
  `Terminated`, not `Truncated` — under which this ADR's rule already does the
  right thing.
- **A parallel `bootstrap_values: &[f32]` argument alongside the existing
  `truncated: &[bool]`.** Rejected in favour of `Vec<Option<f32>>`: two parallel
  arrays can disagree, and the natural "unset" value in a float slice is `0.0`,
  which is indistinguishable from a legitimately-zero bootstrap and reproduces
  the exact bug being fixed.
- **Fix the off-by-one in a separate PR first.** Rejected on the entanglement
  argument in Decision §4 — PEB forces the `[t]` convention — and because it
  would perturb every seeded baseline twice. The two-commit structure preserves
  the reviewability and bisectability that motivated the split.
- **Deprecate-and-shim `compute_gae`.** Rejected: a deprecation warning
  communicates scheduled removal, not numerical incorrectness, and the only
  in-tree caller is a dev-only bench.
- **Adopt Doering et al.'s weight correction in the same change.** Rejected as
  out of scope; see above.

## References

- Issue #170 — "[rl] Truncation vs. termination bootstrap bias." The off-policy
  half (six agents' replay masks) is resolved separately; the research note
  refutes #170's premise that the on-policy infrastructure was pre-built.
- Pardo, Tavakoli, Levdik, Kormushev. *Time Limits in Reinforcement Learning.*
  ICML 2018, PMLR 80. arXiv:1712.00378v4 — <https://arxiv.org/abs/1712.00378>.
  (Eq. 5, Eq. 6; §2.4 γ-dependence; §3.1–3.2 magnitude; §3.4 replay
  interaction.) **Primary source for the decision**: Eq. 6 is the PEB target
  adopted here.
- Schulman, Moritz, Levine, Jordan, Abbeel. *High-Dimensional Continuous
  Control Using Generalized Advantage Estimation.* ICLR 2016.
  arXiv:1506.02438v6 — <https://arxiv.org/abs/1506.02438>. (Eq. 16; §2 assumes
  trajectories run to an absorbing state — rollout-boundary truncation is not
  addressed.) Defines the estimator being corrected; its silence on truncation
  is why the two-mask form is a retrofit rather than a restatement.
- Towers et al. *Gymnasium: A Standardized Interface for Reinforcement Learning
  Environments.* NeurIPS 2025. arXiv:2407.17032v4 —
  <https://arxiv.org/abs/2407.17032>. (Eqs. 1–2, mask is `¬terminated`; §4.2 on
  libraries conflating the two.) Source of the `terminated`/`truncated`
  vocabulary `EpisodeStatus` mirrors, and of the ecosystem-level statement that
  conflating them is a defect.
- Fujimoto, van Hoof, Meger. *Addressing Function Approximation Error in
  Actor-Critic Methods.* ICML 2018. arXiv:1802.09477, **Appendix D** — the
  earliest explicit statement of the rule in the primary literature.
- Raffin et al. *Stable-Baselines3.* JMLR 22(268):1–8, 2021.
  (`on_policy_algorithm.py::collect_rollouts` — the reward-folding form.)
- Huang et al. *CleanRL.* JMLR 23(274):1–18, 2022. (`ppo.py`; upstream issues
  vwxyzjn/cleanrl #198 and #457 acknowledging the simplification.)
- Doering et al. *An Approximate Ascent Approach To Prove Convergence of PPO.*
  2026. arXiv. (Residual GAE weighting bias — explicitly out of scope.)
- Jin et al. *Partial Advantage Estimator for PPO.* IEEE Transactions on Games,
  2025.
- ADR [0026](0026-shared-config-validation-convention.md) /
  [0027](0027-bounds-newtype-for-closed-ranges.md) /
  [0039](0039-box2d-states-own-markov-dofs.md) /
  [0046](0046-slot-newtype-replaces-option-take-around-learn-step.md) — the
  make-invalid-states-unrepresentable posture that `Option<f32>` follows here.
- ADR [0028](0028-tensor-batch-conversion-seam.md) /
  [0030](0030-permutation-tensorgenome-and-population-nonempty-invariant.md) /
  [0038](0038-continuous-action-components-const.md) /
  [0042](0042-snapshotbase-carries-optional-metadata.md) — precedent for atomic
  breaking changes at alpha with no deprecation shim.
- ADR [0044](0044-post-terminal-step-is-an-error.md) — establishes that no
  auto-reset exists in `rlevo-environments`, which is why the continuation
  observation is still in hand at record time.
- Code: `crates/rlevo-reinforcement-learning/src/algorithms/ppo/rollout.rs`
  (`RolloutBuffer`, `compute_gae`, and the doc comment this ADR reverses);
  `.../ppo/{train.rs, ppo_agent.rs}`; `.../ppg/{train.rs, ppg_agent.rs}`;
  `crates/rlevo-reinforcement-learning/benches/ppo_bench.rs`;
  `crates/rlevo-environments/src/wrappers/time_limit.rs`.
