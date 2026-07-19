---
project: rlevo
status: active
type: decision
date: 2026-07-18
tags: [adr, decision, reinforcement-learning, replay, per, her, issue-188, burn]
---

# ADR 0050: A replay-strategy seam; PER is rebuilt, not finished

## Status

**Accepted (2026-07-18).** Resolves issue #188 ("`rl/memory.rs`:
`PrioritizedExperienceReplay` is dead/unfinished but the docs advertise it as the
integration path"). Supersedes nothing. **Carries a correcting note for ADR
[0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md) §Context,
row 1** — see *Correction to ADR 0003* below.

## Correction to ADR 0003

ADR 0003's Context table asserts:

> | `memory.rs` (`PrioritizedExperienceReplay`, `TrainingBatch`, `ReplayBufferError`) | `rlevo-reinforcement-learning` only (8 algorithms — DQN, C51, QR-DQN, DDPG, TD3, SAC, plus PPO/PPG via `AgentStats`) | **Yes** |

**That consumer claim is false, and was false when written.** Verified
2026-07-18 by workspace ripgrep:

- No algorithm in the crate constructs, holds, or samples a
  `PrioritizedExperienceReplay`. The six off-policy agents import **only**
  `crate::memory::ReplayBufferError` from that module.
- `TrainingBatch` has **zero** consumers anywhere in the workspace.
- The `AgentStats` parenthetical belongs to `metrics.rs` (row 3), not to
  `memory.rs`; it is misplaced in row 1 and does not evidence a `memory.rs`
  consumer.
- The *only* live caller of `PrioritizedExperienceReplayBuilder` outside
  `memory.rs`'s own tests is `crates/rlevo/tests/integration_test.rs:304`.

ADR 0003's *decision* — move the three modules out of `rlevo-core` — is
**unaffected**: the modules are RL-only either way, and the correct consumer
count (one integration test) is *further* from the ADR 0002 bar, not closer.
Only the supporting sentence is wrong. ADRs are immutable, so the correction is
recorded here rather than edited into 0003.

## Context

### What is actually in the tree

Six off-policy agents (`dqn`, `c51`, `qrdqn`, `ddpg`, `td3`, `sac`) each
hand-roll a `VecDeque<Transition<O>>` replay buffer with a private, per-file
`Transition` type — six definitions of one concept. The six are byte-identical
except for one field:

| Family | Site | Action field |
|---|---|---|
| Discrete | `dqn_agent.rs:88`, `c51_agent.rs:79`, `qrdqn_agent.rs:80` | `action_idx: usize` |
| Continuous | `ddpg_agent.rs:91`, `td3_agent.rs:100`, `sac_agent.rs:117` | `action: Vec<f32>` |

Every one stores `obs: O`, `reward: f32`, `next_obs: O`, `terminated: bool`, and
every one hand-rolls the same eviction (`if len >= cap { pop_front() }`), the
same `remember`, the same `pub(crate) fn replay_n()`, and the same
with-replacement draw (`(0..batch_size).map(|_| rng.random_range(0..len))`).
The only spelling drift is the capacity config field:
`replay_buffer_capacity` (discrete three) vs `buffer_capacity` (continuous
three).

`PrioritizedExperienceReplay` in `memory.rs` is a **different, unwired data
model**. It stores `ExperienceTuple<D, AD, O, A, R>` — generic over the *typed*
`A: Action<AD>` and `R: Reward`. The agents deliberately erased those types;
`sac_agent.rs:112-115` documents why in the source: storing `Vec<f32>` "so the
buffer can hold them without an additional `Clone + 'static` bound on the
action type."

### Why the existing type cannot simply be "finished"

Four independent defects, any one of which forecloses a drop-in swap:

1. **It is not PER.** There is no `update_priorities`. Priorities are
   write-once at insert (`memory.rs:233-249`), so the TD error never feeds
   back. Schaul's `p_i = |δ_i| + ε` is the entire algorithm; without the
   feedback edge this is a weighted-random buffer wearing PER's name.
2. **It is unseedable.** `sample_batch` calls `rand::rng()` internally
   (`memory.rs:311`). Every agent's `learn_step` takes `rng: &mut R` from the
   caller. A buffer that owns a thread-local RNG can never appear in a
   reproducibility test and violates the spirit of ADR 0029's
   own-a-persistent-stream rule.
3. **Its output shape does not fit its intended consumers.**
   `TrainingBatch::actions` is a float one-hot tensor of rank `BAD`
   (`memory.rs:404`). DQN needs a rank-2 `Int` index tensor for `gather`
   (`dqn_agent.rs`, `q_all.gather(1, action_tensor)`). One `TrainingBatch`
   cannot serve both the discrete and the continuous family.
4. **Its sampling semantics differ from the agents'.** It samples *without*
   replacement via an O(n·k) scan-and-`swap_remove`; all six agents sample
   *with* replacement. Substituting it is a behavioural change, not a
   refactor.

Add the fidelity gaps the literature review (`docs/.private/research/
per-schaul-2016-fidelity.md`) enumerates: no ε floor, no IS weights, no β, no
max-normalization, i.i.d. draws instead of Schaul's stratified one-per-segment
scheme, and no finiteness guard on priorities (a `NaN` priority produces `NaN`
probabilities and silently pins `selected_pos` at `0`).

### Why prioritization is scoped to the value-based three

The research note is decisive and this ADR does not relitigate it. Rainbow's
ablation puts prioritized replay among the **two most crucial** of seven
components (full Rainbow beat the no-priority ablation in 53 of 57 games). On
the continuous-control side, Panahi et al. (RLJ 2024) report that "in control
tasks, none of the prioritized variants consistently outperform uniform
replay," and Saglam et al. (JAIR 2022) give the mechanism: an actor cannot be
effectively trained on high-TD-error transitions, because the policy gradient
under a noisy Q diverges from the gradient under the optimal Q. Every positive
continuous-control PER result in the literature *modifies* PER first.

Turning vanilla PER on for DDPG/TD3/SAC would be a paper-fidelity defect
dressed as a feature.

### The constraint that shapes the seam

Two future strategies must not force a second breaking change:

- **PER** needs `update_priorities(ids, td_errors)` — mutation of buffer
  metadata after insert.
- **HER** needs goal relabeling. Note carefully what Andrychowicz et al. (2017)
  Algorithm 1 actually does: for each visited state it **stores additional
  relabeled transitions** (`store (s_t‖g′, a_t, r′, s_{t+1}‖g′) in R`). It does
  not overwrite the original. HER's real demands on a replay seam are
  (a) ordinary `push` and (b) a goal-conditioned observation type — the latter
  is a change to `O` and to the stored transition, which **no seam shape can
  prevent**. See *Alternatives considered*.

## Decision

### 1. `crate::replay` — a new module, not an evolution of `memory.rs`

A new `crates/rlevo-reinforcement-learning/src/replay/` module hosts the seam.
`memory.rs` is deleted (see §8). `experience.rs` is left **untouched and
out of scope** (see §7).

### 2. One stored transition type, generic over the erased action payload

```rust
/// A stored `(s, a, r, s', terminated)` transition.
///
/// `P` is the *stored action payload*, deliberately erased from the domain
/// `A: Action<AD>`: storing `A` would impose `Clone + 'static` on every
/// action type (see `sac_agent.rs`'s pre-seam note).
pub struct Transition<O, P> {
    pub obs: O,
    pub action: P,
    pub reward: f32,
    pub next_obs: O,
    pub terminated: bool,
}

pub type DiscreteTransition<O>   = Transition<O, usize>;
pub type ContinuousTransition<O> = Transition<O, Vec<f32>>;
```

Six private definitions collapse to one public type and two aliases. The
erasure SAC chose is **preserved, not reversed** — the payload is `usize` /
`Vec<f32>`, never `A`. Reconciling onto `ExperienceTuple`'s typed `A`/`R` is
rejected in *Alternatives considered*.

### 3. `ReplayStrategy<T>` is generic over the stored item and knows nothing about tensors

```rust
pub trait ReplayStrategy<T> {
    fn push(&mut self, item: T);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get(&self, id: TransitionId) -> Option<&T>;
    /// Post-insertion mutation. PER's `update_priorities` and any future
    /// in-place relabeling are instances of this one capability.
    fn get_mut(&mut self, id: TransitionId) -> Option<&mut T>;
    fn iter(&self) -> impl Iterator<Item = &T>;

    /// Draws `batch_size` ids. `beta` is the current IS exponent; it is
    /// ignored by strategies that emit no weights.
    fn sample<R: rand::Rng + ?Sized>(
        &self,
        batch_size: usize,
        beta: f32,
        rng: &mut R,
    ) -> Result<SampledBatch, ReplayBufferError>;

    /// No-op for uniform. Non-finite or non-positive priorities are rejected
    /// by the `Priority` newtype, not silently absorbed.
    fn update_priorities(&mut self, ids: &[TransitionId], priorities: &[Priority]) {}
}
```

The seam draws the line at **which items come back and what weight each
carries**. It never sees a `Tensor`, a `Backend`, or a device. Staging stays in
the agent, which is the only place that knows whether an action becomes an
`Int` index tensor or a float vector tensor. This is why `TrainingBatch` is
retired rather than extended (§5).

`sample` takes `&self` and `rng: &mut R` — matching every agent's existing
`learn_step<R: Rng + ?Sized>(&mut self, rng: &mut R)` signature, and fixing
defect (2).

### 4. Two implementations, not one degenerate one

`UniformReplay<T>` and `PrioritizedReplay<T>` are separate types behind
`ReplayStrategy<T>`.

Schaul Eq. 1 makes uniform the `α = 0` special case *mathematically*. That is
not an implementation mandate, and unifying costs real money on the hottest
loop in the library:

- At `α = 0` a prioritized implementation still walks the whole buffer to build
  the sampling distribution — O(n) per learn step, with `n` = capacity
  (10³–10⁶ in the shipped configs), versus O(k) for `k` `random_range` draws.
- The prioritized path carries a sum-tree (2N floats) and a running max that
  uniform has no use for.
- The two have **different draw semantics**: uniform is i.i.d. with
  replacement; Schaul's is stratified, one draw per equal-mass segment
  (Appendix B.2.1), deliberately *without* the i.i.d. property, "to balance out
  the minibatch." A single α-parameterised implementation must pick one, and
  either choice silently changes the other's behaviour.

The second reason is decisive on its own, independent of cost.

### 5. `UniformReplay`'s draw order is a pinned contract, not an implementation detail

`UniformReplay::sample` **must** issue exactly `batch_size` calls to
`rng.random_range(0..len)`, in order, and return the drawn indices in that
order. This is what makes step 1 of the migration a bit-identical behavioural
no-op against pre-seam `HEAD` (the ADR 0046 verification standard). It is
stated here as a contract, and pinned by a test, precisely so that a later
"optimization" (batched draws, a shuffle, sampling without replacement) cannot
break every seeded baseline in the crate without tripping a test first.

### 6. The index handle is an owned, generation-stamped id — and it is *not* a `Slot`

```rust
/// Absolute, monotone id of a stored transition. Survives FIFO eviction:
/// an evicted id resolves to `None` instead of silently aliasing a
/// different transition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TransitionId(u64);

pub struct SampledBatch {
    ids: Vec<TransitionId>,
    /// `None` for uniform. `Some` carries max-normalized IS weights, one per id.
    weights: Option<Vec<f32>>,
}
```

The buffer keeps a `pushes: u64` counter; slot `i` of a buffer holding `len`
items after `pushes` inserts has absolute id `pushes - len + i`. Resolution is
one subtraction and a bounds check; an id older than the live window returns
`None` and its priority update is dropped.

Why absolute rather than a bare `usize`: today every agent samples and updates
inside a single `learn_step` with no interleaved `remember`, so raw indices
happen to be safe. The roadmap names *distributed replay architectures*, where
a learner's ids outlive the collector's inserts and every raw index silently
shifts by the eviction count. A `u64` counter makes that class of bug
unrepresentable for the cost of one add per insert.

**`SampledBatch` is not a `Slot` (ADR 0046) and must not be folded into one.**
`Slot<M>` exists to bound the window in which a *Burn module* is out of its
field across a by-value `Optimizer::step`. `SampledBatch` carries *sample
identity* across the staging → loss → writeback sequence. They share a shape
("a value that must survive a learn step") and nothing else: `SampledBatch` is
`Copy`-cheap host data with no ownership hazard, no poisoning semantics, and no
interaction with the optimizer. Merging them would give `Slot` a second reason
to change.

### 7. `TrainingBatch` is retired, not extended

`TrainingBatch` has zero consumers, cannot express DQN's `Int` action tensor,
and would need `Option<weights>` + `indices` fields that are meaningless for
half its (hypothetical) callers. It is deleted with `memory.rs`.

IS weights ride on `SampledBatch`, not on a tensor bundle, because **the weight
is consumed at the loss site, not the staging site**, and each agent stages
differently. The agent uploads `weights` as its own `Tensor<B, 1>` at the
moment it needs it.

**There is no persisted-data migration.** Verified: no type in `memory.rs`
derives `Serialize`/`Deserialize`, no replay buffer is written to an
`EpisodeRecord` or a checkpoint, and the record schema has no replay field. A
downstream user's migration is a compile error, not a data loss.

### 8. `PrioritizedExperienceReplay` and its builder are removed, with no `#[deprecated]` shim

They shipped in tagged releases v0.1.0–v0.3.1 and appear in both READMEs'
usage examples. They are nonetheless removed outright in one PR, following the
precedent of ADRs 0028 / 0030 / 0038 / 0042 / 0048 (breaking at alpha, atomic,
no shim), and specifically ADR 0048's reasoning: a `#[deprecated]` shim
"would signal scheduled removal, not numerical incorrectness."

The argument is stronger here. `#[deprecated]` says *this works, but prefer the
new thing*. The type does not work: it is advertised as PER and is not PER
(§Context defect 1). Leaving a compiling, importable, wrong-named path is worse
for a user than a compile error that points at the replacement.

Retiring the builder also retires the `Builder with_alpha(x)` row from
`docs/rules.md` §4's Documented Panic Contracts table (the `with_capacity(n)`
row survives, now naming the replay builders). `rules.md` §3's
`PrioritizedExperienceReplay | priorities.len() == buffer.len()` invariant row
is restated against `PrioritizedReplay`.

`ReplayBufferError` — the one symbol the agents actually import — is kept, moved
into `replay`, and re-exported from `memory`'s former path only if a
compatibility shim is wanted later. It gains no new variants for uniform;
`InsufficientData` already covers the one failure uniform can have.

### 9. `experience.rs` is out of scope

`ExperienceTuple`, `History`, `HistoryRepresentation`, and `SufficientStatistic`
are **not touched**. ADR 0003 §"conservative dead-code policy" deliberately kept
zero-consumer roadmap markers, and `HistoryRepresentation`/`SufficientStatistic`
are structurally coupled to `History` (both take `&History` in their
signatures), so deleting the concrete types drags the traits with them. That is
a separate decision belonging to #190, not to this seam.

What this ADR *does* record for #190: once `memory.rs` is gone, `experience.rs`
is dead outright, not merely dead-by-transitivity. Its module docs gain a
pointer to `crate::replay::Transition` as the live model, so a reader cannot
mistake `ExperienceTuple` for the integration path — which is the actual
complaint in #188.

### 10. `PrioritizedReplay` — the fidelity contract

Follows Schaul et al. 2016, proportional variant, with every deviation named:

- **Priority** `p_i = |δ_i| + ε`, ε = `1e-6`. **Schaul gives no value for ε**
  (it is not in the grid search); `1e-6` is *our* choice and the rustdoc says so.
- **Sampling** `P(i) = p_i^α / Σ p_k^α` via a sum-tree, drawn by Schaul's
  **stratified** scheme: `[0, p_total]` split into `k` equal ranges, one uniform
  draw per range. Not i.i.d. categorical draws.
- **New transitions** get the **running max over priorities seen so far**
  (Algorithm 1 line 6), tracked as an incremental `f32` — not a constant, and
  not a max over live buffer contents.
- **IS weights** `w_i = (1/N · 1/P(i))^β`, **max-normalized over the sampled
  minibatch** (Algorithm 1 line 10), not over a whole-buffer priority bound.
- **`w_i` scales the per-sample loss only.** It must never enter the target
  computation and must never alter `δ` itself. This is the implementer bug class
  the research note names.
- **α ∈ [0, 1]**, **β ∈ [0, 1]**, defaults `α = 0.6`, `β₀ = 0.4 → 1.0` linear
  (Table 3, proportional row).

**Priorities are validated by construction.** A new `Priority` newtype
(`finite && > 0`, `new`/`try_new` + `Copy` error, mirroring `Bounds` /
`Probability` / `NonNegativeRate` — ADR 0027 / 0031) makes the `NaN`-priority
defect unrepresentable rather than guarded. A `NaN` TD error off a diverging
network is *reachable in production*, not theoretical.

### 11. β annealing: schedule on the config, application in the buffer

Appendix B.2.1: "this normalization interacts with annealing on β." The two are
**not independent knobs**, so they must not be settable independently.

- The buffer computes `w_i` at the passed β **and** max-normalizes in the same
  expression. A caller cannot obtain unnormalized weights; there is no knob.
- The **schedule** lives on the agent config
  (`beta_start`, `beta_end`, `beta_anneal_steps`) and the agent passes
  `beta(self.step)` into `sample`.

Rationale for splitting them that way: the buffer has no step counter, and
giving it one duplicates the agent's — a second source of truth, the exact
shape `rules.md` §10 forbids for episode termination. It is also wrong the
moment two learners share one buffer (the distributed-replay roadmap item). The
accepted cost is that a caller can pass a nonsense β; `Validate` on the config
and the three in-crate call sites are the mitigation.

### 12. Naming — `alpha` is not available

`alpha` already means SAC's entropy temperature (`sac_agent.rs:205`). Even
though PER is not wired into SAC, the seam's config type is crate-shared
vocabulary and the collision is in the reader's head, not the compiler's. So:

| Schaul symbol | Field name |
|---|---|
| α (priority exponent) | `priority_exponent` |
| β (IS exponent) | `importance_exponent` / `beta_start`, `beta_end` |
| ε (priority floor) | `priority_epsilon` |

Config type is `PrioritizedReplayConfig`, not `PerConfig` ("per" reads as the
English word at every call site). The Greek letters appear in the rustdoc,
mapped to Schaul's equations, and nowhere else.

### 13. Per-agent priority signal

| Agent | Priority | Provenance |
|---|---|---|
| DQN | `\|δ\|` from the per-sample Huber residual | Schaul §3.3, direct |
| C51 | **KL**, not cross-entropy | Rainbow, verbatim: "prioritize transitions by the KL loss" |
| QR-DQN | per-sample quantile Huber loss | **By analogy only** |

**C51 requires an explicit correction.** `categorical_cross_entropy`
(`c51/loss.rs:26-30`) returns `−Σ target·log pred`. Rainbow specifies
`D_KL(target ‖ pred) = CE − H(target)`. `H(target)` is constant with respect to
θ but **varies across samples**, so using CE as the priority is *not* Rainbow's
priority. The KL priority must subtract `Σ target·log target` explicitly.

**QR-DQN's priority is uncited and ships labelled as such.** Dabney et al.
(2018) explicitly decline the combination: "in our evaluations we compare the
pure versions of C51 and QR-DQN without these additions." Using the quantile
Huber loss as the priority extrapolates Rainbow's stated *principle*
("prioritize by what the algorithm is minimizing") to a case nobody ablated. It
is opt-in and its rustdoc says, in these terms, that it is a design choice by
analogy and not a literature result.

### 14. The loss-site restructure is bit-identical at `w ≡ 1`

Verified against `burn-nn-0.21.0`: `HuberLoss::forward(p, t, Reduction::Mean)`
is *literally* `self.forward_no_reduction(p, t).mean()`
(`burn-nn-0.21.0/src/loss/huber.rs:86-98`). So DQN's restructure
(`dqn_agent.rs:497-500`) to

```rust
let per_sample = huber.forward_no_reduction(q_pred_flat, target);   // [batch]
let loss_tensor = (per_sample * is_weights).mean();
```

is bit-identical to the current code when `is_weights` is all-ones — no
floating-point reassociation, because the reduction is the same `.mean()` on the
same `[batch]` tensor. The same holds for C51 (`loss.rs:26-30` already computes
`per_sample` and then `.mean()`s it; the change is purely to the return type at
`loss.rs:22-25`) and QR-DQN (`quantile_loss.rs:102-112`, same shape, signature
at `:60-65`).

This is what lets the seam migration and the loss-site restructure both be
verified as behavioural no-ops, and it is why they can land as separate PRs
without either one's verification being weakened.

### 15. DDPG / TD3 / SAC get the seam and keep uniform

They move onto `UniformReplay<ContinuousTransition<O>>`. Their loss sites
(`ddpg_agent.rs:508-509`, `td3_agent.rs:585-586`, `sac_agent.rs:609-610`) are
**not** restructured in this work. The twin-critic sites in particular never
name the residual (`(q1_pred - target).powi_scalar(2).mean()`), and SAC carries
a documented Burn autodiff constraint (`sac_agent.rs:598-600`, `:611-614`)
requiring host reads to go via `.inner().into_scalar()`. Restructuring three
loss sites to support a feature the literature says is contested-to-harmful
there is cost with no benefit.

## Consequences

### Positive

- Six hand-rolled buffers collapse to one. The `replay_buffer_capacity` /
  `buffer_capacity` naming split is resolved to one field name at the same time.
- `#188`'s actual complaint — the docs advertise an integration path that no
  algorithm uses — is closed by making the advertised path the real one.
- PER is available where the literature supports it (Rainbow: 53/57 games) and
  absent where it does not, with the reasoning in the rustdoc rather than in a
  vault note.
- The `NaN`-priority defect, the unseedable-RNG defect, and the write-once-
  priority defect are all closed by construction rather than by guard.
- The seam admits HER's `push`-of-relabeled-copies and PER's
  `update_priorities` through the same three methods (`push`, `get_mut`,
  `update_priorities`), so neither forces a second breaking change to the
  trait.

### Negative / accepted costs

- **Breaking removal of a type that shipped in three tagged releases**, with no
  deprecation window. Any downstream user of `PrioritizedExperienceReplay` gets
  a compile error. Accepted per §8; alpha, and the removed type does not do what
  its name says.
- **Two transition models coexist until #190 closes.** `ExperienceTuple` /
  `History` remain in `experience.rs` as ADR 0003 roadmap markers while
  `replay::Transition` is the live one. Mitigated by a module-doc pointer, not
  eliminated.
- **The trait carries `get_mut`, which nothing calls on day one.** PER uses
  `update_priorities`; HER does not exist yet. This is a deliberate, cheap bet
  on the post-insertion-mutation shape — but see *Alternatives* for why the HER
  justification for it is weaker than #188's framing assumes.
- **QR-DQN ships an uncited priority signal.** Opt-in and documented as such,
  but it is an extrapolation the primary literature declines to make.
- **A sum-tree is ~150 lines of new index arithmetic** on a path where an
  off-by-one silently biases sampling rather than crashing. Mitigated by
  landing the O(n) prefix-scan version first, pinning its outputs, and swapping
  the tree in behind identical tests (§Implementation ordering).

### Neutral

- No record `FORMAT_VERSION` bump. Nothing in the replay path is serialized.
- `rlevo-core` is untouched. The seam is entirely inside
  `rlevo-reinforcement-learning`, consistent with ADR 0003's partition.

## Alternatives considered

**Finish `PrioritizedExperienceReplay` in place: add `update_priorities`, IS
weights, and a sum-tree to the existing type.** Rejected. It would still store
`ExperienceTuple<D, AD, O, A, R>`, reintroducing the `A: Action<AD> + Clone +
'static` bound that all six agents deliberately erased — SAC's source comment
is a written record of that decision. It would still own an internal
`rand::rng()`. Its `TrainingBatch` output would still be unable to express
DQN's `Int` gather tensor. Fixing all four amounts to writing the new module
while keeping the old name.

**Reconcile the six `Transition`s onto `ExperienceTuple`'s typed generics.**
Rejected for the same bound. The typed model is strictly more expressive and
strictly less usable here: the agents need `usize` and `Vec<f32>` on the
staging path anyway, so a typed `A` would be converted to the erased form at
every sample. Paying a `Clone + 'static` bound on every downstream action type
to store data that is immediately erased is a cost with no purchaser.

**One `ReplayStrategy` with `α = 0` as the uniform case (Schaul Eq. 1).**
Rejected on two grounds, the second sufficient alone: (a) O(n) distribution
construction per learn step against O(k) draws, at capacities up to 10⁶; (b)
uniform draws i.i.d. with replacement while Schaul's scheme is stratified
without it, so one implementation cannot honour both and either choice silently
changes the other's semantics. Elegance loses to a behavioural difference.

**A single shared `Transition` with an `enum ActionPayload { Index(usize),
Continuous(Vec<f32>) }`.** Rejected: a runtime branch per element on the
staging hot path, plus an unrepresentable-state hole (a DQN agent could be
handed a `Continuous` payload and would have to panic or misbehave). The
generic parameter makes the same distinction at compile time for free.

**Fold `SampledBatch` into ADR 0046's `Slot`.** Rejected — §6. They share a
sentence-level description and no semantics. `Slot`'s single reason to change
is Burn's by-value `Optimizer::step`; giving it a second one dilutes the
guarantee ADR 0046 bought.

**Design an explicit `relabel` hook into the trait for HER now.** Rejected, and
this ADR records the reasoning against the framing in #188. Andrychowicz et al.
(2017) Algorithm 1 **stores additional relabeled transitions** — it does not
overwrite stored ones. HER's binding requirement on a replay seam is therefore
ordinary `push` plus a goal-conditioned `O`, and the `O` change is a data-model
change that no seam shape prevents. A `relabel` hook designed today, against no
implementation, is a good candidate for being the wrong shape. `get_mut` is
kept because **PER's `update_priorities` is a real, present instance of
post-insertion mutation** — that, not HER, is its justification.

**Extend `TrainingBatch` with `Option<weights>` and `indices`.** Rejected —
§7. It institutionalizes a zero-consumer type, and `Option` fields that are
`None` for half the strategies are a smell that the type is being asked to
serve two shapes.

**Wire PER into DDPG/TD3/SAC as well, "for symmetry."** Rejected on the
literature (§Context). Symmetry across algorithm families is not a value the
library holds; fidelity to the algorithm each family implements is.

## Implementation ordering

Each step is independently reviewable and independently verifiable. **Step 1
must land before steps 2–4.** Steps 2, 3, and 4 are mutually independent.

**Step 1 — seam + uniform, behavioural no-op.**
New `replay/{mod,transition,uniform,error}.rs`; `Transition<O, P>` +
two aliases; `ReplayStrategy<T>`; `TransitionId`/`SampledBatch`;
`UniformReplay<T>` with the §5 pinned draw contract. Migrate all six agents'
`buffer` field, `remember` (`dqn:326`, `c51:336`, `qrdqn:303`, `ddpg:376`,
`td3:418`, `sac:443`), `replay_n`, and index sites (`dqn:420`, `c51:460`,
`qrdqn:417`, `ddpg:440`, `td3:489`, `sac:512`); delete the six private
`Transition` definitions (`dqn:88`, `c51:79`, `qrdqn:80`, `ddpg:91`, `td3:100`,
`sac:117`); unify the capacity config field name.
**Acceptance: seeded A/B runs bit-identical to pre-change `HEAD` for all six
agents** (the ADR 0046 standard).

**Step 2 — loss sites emit per-sample, still reduced by the caller.**
`c51/loss.rs:22-25` and `qrdqn/quantile_loss.rs:60-65` change return type to
per-sample `Tensor<B, 1>`; callers (`c51_agent.rs:569`, `qrdqn_agent.rs:512`)
add `.mean()`. `dqn_agent.rs:497-500` switches to `forward_no_reduction` +
`.mean()`. **Acceptance: bit-identical, by the §14 argument** — no IS weight
exists yet.

**Step 3 — `PrioritizedReplay` in isolation.**
`Priority` newtype; `PrioritizedReplayConfig` + `Validate`; sum-tree;
stratified sampling; running max; IS weights with minibatch max-normalization.
Zero agent changes. **Acceptance: unit tests against Schaul's equations —
α = 0 recovers the uniform marginal, weights are max-normalized to 1,
stratification puts exactly one draw per segment, an evicted `TransitionId`
resolves to `None`, `Priority::try_new` rejects `NaN`/`0`/negative.** Land the
O(n) prefix-scan version first if the sum-tree is not review-ready; swap behind
identical tests.

**Step 4 — wire PER into DQN, then C51, then QR-DQN (three PRs).**
Config fields; `SampledBatch` threaded from sample to loss; weight tensor
upload; `update_priorities` writeback. C51's PR carries the §13 CE→KL
correction. QR-DQN's PR carries the "by analogy, not ablated" rustdoc.
**Acceptance: with `priority_exponent = 0.0` and `importance_exponent = 0.0`,
the prioritized path must match the uniform path's learning curve** (not
bit-identical — the draw scheme differs by design — but statistically
indistinguishable over seeds).

**Step 5 — removal + docs.** Delete `memory.rs`
(`PrioritizedExperienceReplay`, its builder, `TrainingBatch`); move
`ReplayBufferError` into `replay`. Update
`crates/rlevo/tests/integration_test.rs:304`, both READMEs
(`rlevo-reinforcement-learning/README.md:348-355`,
`rlevo-core/README.md:228-230` — the latter is additionally stale, still
pointing `rlevo-core` readers at a type ADR 0003 moved out of it),
`docs/contributor-book/src/ch07-adding-an-rl-algorithm.md:13-14,41`,
`docs/rules.md` §3 invariant row and §4 panic-contract row, `CLAUDE.md`'s
Key Files table, and `CHANGELOG.md` under Unreleased/Breaking.

## References

- Issue #188 — "`rl/memory.rs`: `PrioritizedExperienceReplay` is dead/unfinished
  but the docs advertise it as the integration path." Issue #190 (`experience.rs`
  dead by transitivity) is unblocked, not resolved, by this ADR.
- `docs/.private/research/per-schaul-2016-fidelity.md` — the literature
  validation this design is required to honour.
- Schaul, Quan, Antonoglou, Silver (2016). *Prioritized Experience Replay.*
  ICLR 2016, arXiv:1511.05952v4. §3.3, §3.4, Algorithm 1, Appendix B.2.1,
  Table 3.
- Hessel et al. (2018). *Rainbow.* AAAI 2018, arXiv:1710.02298 — KL priority for
  distributional agents; the seven-component ablation.
- Dabney, Rowland, Bellemare, Munos (2018). *Distributional RL with Quantile
  Regression.* AAAI 2018, arXiv:1710.10044 — declines the PER combination.
- Andrychowicz et al. (2017). *Hindsight Experience Replay.* NeurIPS 2017,
  Algorithm 1 — stores *additional* relabeled transitions.
- Panahi et al. (RLJ 2024); Saglam et al. (JAIR 2022) — PER on continuous
  control is contested; actor-gradient mechanism.
- ADR [0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md) —
  corrected above; its conservative dead-code policy scopes `experience.rs` out.
- ADR [0028](0028-tensor-batch-conversion-seam.md) — `stack_to_tensor` was
  introduced with `memory.rs::sample_batch` as its "first consumer"; that
  consumer is retired here, and the helper's remaining users are unaffected.
- ADR [0046](0046-slot-newtype-replaces-option-take-around-learn-step.md) —
  `Slot<M>`; §6 records why `SampledBatch` is not one.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) /
  [0031](0031-probability-rate-newtypes.md) — the validated-newtype shape
  `Priority` follows.
- ADR [0029](0029-host-rng-seeding-convention.md) — the caller-supplied
  `rng: &mut R` convention the seam adopts.
- Code: `crates/rlevo-reinforcement-learning/src/memory.rs` (removed);
  `.../src/experience.rs` (untouched); `.../src/algorithms/{dqn,c51,qrdqn,
  ddpg,td3,sac}/*_agent.rs`; `.../src/algorithms/c51/loss.rs`;
  `.../src/algorithms/qrdqn/quantile_loss.rs`;
  `burn-nn-0.21.0/src/loss/huber.rs:86-98` (`forward(.., Mean)` ≡
  `forward_no_reduction(..).mean()`, the basis of the §14 bit-identity claim).
