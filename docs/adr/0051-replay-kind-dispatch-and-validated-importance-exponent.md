---
project: rlevo
status: active
type: decision
date: 2026-07-18
tags: [adr, decision, reinforcement-learning, replay, per, her, issue-188, numerical-stability]
---

# ADR 0051: `ReplayKind` dispatch, and β is a validated newtype

## Status

**Accepted (2026-07-18).** **Corrects ADR
[0050](0050-replay-strategy-seam.md) §3, §11, and §13** — three statements that
did not survive contact with the implementation. Does not supersede 0050; its
decision (a replay-strategy seam, PER rebuilt rather than finished) stands
unchanged. ADRs are immutable, so the corrections are recorded here rather than
edited into 0050.

Arises during ADR 0050's implementation ordering, between step 3
(`PrioritizedReplay` in isolation) and step 4 (wire PER into the value-based
three).

## Context

ADR 0050 was written before `crate::replay` existed. Its §3 trait sketch and its
§11 accepted cost were both designed against an imagined call site. Building the
real one falsified three claims.

## Decision

### 1. `update_priorities` is not a defaulted trait method; dispatch goes through `ReplayKind<T>`

**ADR 0050 §3's trait sketch is superseded.** It gave:

```rust
pub trait ReplayStrategy<T> {
    /* ... */
    /// No-op for uniform.
    fn update_priorities(&mut self, ids: &[TransitionId], priorities: &[Priority]) {}
}
```

That method is **not** on the trait. It is **inherent on `PrioritizedReplay`**,
and an agent that can be configured either way holds a new
`enum ReplayKind<T> { Uniform(UniformReplay<T>), Prioritized(PrioritizedReplay<T>) }`
which itself implements `ReplayStrategy<T>` by forwarding, and which dispatches
the writeback through an exhaustive `match`.

**Reason: a defaulted no-op is a silent-swallow hazard.** With the default, a
third strategy that forgets to override `update_priorities` compiles, runs, and
silently discards every priority writeback — the *exact* defect ADR 0050
§Context defect 1 removed from the old `PrioritizedExperienceReplay` (priorities
write-once at insert, TD error never fed back). Reintroducing it as a
trait-level default, one PR after deleting it, would be an unforced error. An
exhaustive `match` makes the same case a **compile error**: when HER lands as a
third strategy, `ReplayKind` fails to compile until someone decides, explicitly,
what a hindsight buffer does with a priority writeback. That is the right place
for that question to be asked, and a defaulted no-op answers it by accident.

Two alternatives were rejected:

- **A generic strategy parameter (`DqnAgent<B, S: ReplayStrategy<_>>`).**
  Rejected: it makes PER a **type-level** choice, so `PrioritizedReplay` would
  have to be named in the agent's type to be used. ADR 0050 requires PER to be a
  **config-level opt-in** (§11, §13 — "opt-in and its rustdoc says so"), and a
  type parameter cannot be turned on by a deserialized config field. It also
  infects every downstream signature that mentions the agent.
- **A swapped concrete field.** Rejected: it lands as either an `Option` pair
  (`Option<UniformReplay<T>>` + `Option<PrioritizedReplay<T>>`), which
  reintroduces exactly the unrepresentable-state hole ADR 0046 removed and ADR
  0050 §7 rejected for `TrainingBatch`, or as two `learn_step` bodies that drift
  apart the first time either is touched.

**Accepted cost: `ReplayKind` is a closed set.** No out-of-crate strategy can be
plugged into an agent. This is accepted, not overlooked: no such consumer exists,
`Slot`'s `pub(crate)` visibility (ADR 0046) is in-repo precedent for the same
posture, and it is a two-way door — the enum can grow a variant, or the whole
dispatch can move to a generic parameter later, without breaking a stored format
or a data model. `ReplayStrategy<T>` itself remains public and implementable;
only the *agent's* dispatch is closed.

### 2. `sample` takes `beta: ImportanceExponent`, not `beta: f32`

**ADR 0050 §3's and §11's `beta: f32` are superseded.** The parameter type is
`ImportanceExponent` — a newtype with the invariant
`is_finite() && (0.0..=1.0).contains(&x)`, mirroring `Priority` (ADR 0050 §10)
and through it `Bounds` / `Probability` / `NonNegativeRate` (ADR 0027 / 0031):
`new` (panicking, for literals and const contexts), `try_new -> Result<_,
ImportanceExponentError>` with a `Copy` error, `const fn get`, and a named
`ImportanceExponent::ONE` for the annealed endpoint.

`sample` deliberately does **not** gain a `Result` for a bad β. `UniformReplay`
ignores β entirely, so a fallible signature would force it to carry an error
variant it can never produce — the "field meaningless for half the callers"
shape that retired `TrainingBatch` in ADR 0050 §7. The invariant is carried by
the type instead, so no implementation re-checks it.

### 3. ADR 0050 §11's accepted cost is WITHDRAWN AS UNSOUND

ADR 0050 §11 closes:

> The accepted cost is that a caller can pass a nonsense β; `Validate` on the
> config and the three in-crate call sites are the mitigation.

**That was wrong on the merits, not merely premature.** It is withdrawn.

The mitigation it names cannot work. The config holds schedule *endpoints* —
`beta_start`, `beta_end`, `beta_anneal_steps`. What reaches `powf` is the
*evaluated* interpolation:

```text
beta_start + (beta_end - beta_start) * limit(step as f32 / anneal_steps as f32)
```

With `beta_anneal_steps == 0` the fraction is `0.0 / 0.0` = `NaN` at `step == 0`
and `+∞` thereafter, **while every endpoint is individually valid**. A `Validate`
impl inspecting endpoints has nothing to reject. Validating endpoints cannot
establish a property of the evaluated value.

**The implementation sharpened this argument rather than confirming it as
first stated.** Whether the `NaN` survives the limiter turns on an incidental
IEEE-754 detail: `f32::min` returns the non-`NaN` operand and silently launders
`NaN` into a plausible-looking `1.0`, whereas `f32::clamp` and an ordinary
`if x > 1.0` comparison both propagate it. So with `.min(1.0)` the zero-length
schedule is *masked*, not absent, and a later refactor between three spellings
that read as interchangeable silently decides whether the buffer receives a
`NaN`. Pinned by
`test_importance_exponent_rejects_an_evaluated_zero_anneal_schedule`.

The consequence had β reached `(min_mass / m).powf(beta)` is the same
NaN-propagation shape that made #188 worth filing: `NaN` weights → `NaN`
per-sample loss → `NaN` through `backward()` → every parameter the optimizer
touches next is permanently poisoned, with no panic, no error, and no log line.
Shipping that inside #188's own fix is not acceptable, and the pre-existing
guard was a `debug_assert!` — absent in exactly the release builds where long
training runs happen.

Three things therefore ship together, and none is sufficient alone:

- `beta_anneal_steps` is **`nonzero`-validated** on the agent config, killing the
  `0/0` at its source.
- The schedule evaluator **clamps** before construction, so an in-range value is
  produced by construction rather than by luck.
- `ImportanceExponent` turns any *residual* bug — a fourth spelling, a caller
  bypassing the evaluator, a deserialized β — into a **loud panic at a named
  construction site**, rather than `NaN` weights poisoning gradients twenty steps
  later at a site that names nothing.

That ordering is the point: the newtype is the backstop, not the only guard.

### 4. Correction: ADR 0050:394 names a stale symbol

ADR 0050 §13 writes:

> `categorical_cross_entropy` (`c51/loss.rs:26-30`) returns `−Σ target·log pred`.

The function has since been renamed **`categorical_cross_entropy_per_sample`**
(`c51/loss.rs:37`) by ADR 0050's own implementation step 2, which changed C51's
loss to emit per-sample values reduced by the caller. The §13 **finding is
unaffected** — CE is still not Rainbow's KL priority, and they still differ by
the per-sample-varying `H(target)` — only the symbol name and line reference are
stale. Recorded here; 0050 is not edited.

## Consequences

### Positive

- A future third strategy cannot silently swallow a priority writeback. The
  question "what does this strategy do with `update_priorities`?" is asked by the
  compiler, at the moment the variant is added.
- A non-finite or out-of-range β is unrepresentable at `sample`, closing the
  release-build NaN path that the `debug_assert!` left open.
- PER stays a **config-level** opt-in, as ADR 0050 requires, rather than becoming
  a type-level one.
- `UniformReplay::sample` keeps its infallible-in-β signature and gains no error
  variant it cannot produce.

### Negative / accepted costs

- **`ReplayKind` is closed to out-of-crate strategies** (§1). Two-way door; no
  consumer exists; `Slot`'s `pub(crate)` is precedent.
- **`ReplayKind` adds a forwarding layer.** Every `ReplayStrategy` method on it
  is a two-arm `match`. The branch is per-`learn_step`, not per-transition, so it
  is off the hot path — but it is boilerplate that must be kept in sync with the
  trait.
- **Every `sample` call site is a breaking change** (`f32` → `ImportanceExponent`).
  Alpha; all call sites are in-repo, and the migration is mechanical.
- **A silent mis-anneal becomes a panic.** A caller whose schedule evaluates out
  of range now crashes where it previously produced quiet garbage. That is the
  intended trade — the same one ADR 0049 makes in the opposite direction and for
  the opposite reason — but it is a behaviour change for any caller relying on
  the `debug_assert!` being compiled out.

### Neutral

- The `ImportanceExponent` change is a **strict behavioural no-op**: no RNG draw
  moves and no arithmetic changes. `UniformReplay`'s pinned draw-order contract
  (ADR 0050 §5) is untouched; only the call's β argument is retyped.
- Nothing is serialized. No record `FORMAT_VERSION` bump.

## Alternatives considered

**Keep the `debug_assert!` and document the contract harder.** Rejected — it is
the status quo whose failure mode this ADR exists to close. A guard that
disappears in release builds is not a guard on a code path whose failures take
`~10⁵` steps to appear.

**Return `Result<SampledBatch, _>` with a `BadBeta` variant.** Rejected — ADR
0050 §7's reasoning, applied to an error enum: `UniformReplay` ignores β, so the
variant is unconstructible for half the implementors, and every call site pays a
`?` for a case that cannot arise there.

**Validate β on the agent config only, and keep `f32`.** Rejected — this is
precisely the withdrawn §11 mitigation. Endpoints are not the evaluated value
(§3).

**A defaulted `update_priorities` on the trait, per ADR 0050 §3 as written.**
Rejected — §1. The default's whole purpose is to let an implementor stay silent
about a question that has no safe silent answer.

**A generic strategy parameter on each agent.** Rejected — §1. It converts a
config-level opt-in into a type-level one, which contradicts ADR 0050's own
requirement.

## References

- ADR [0050](0050-replay-strategy-seam.md) — corrected here at §3 (trait sketch,
  `beta: f32`), §11 (`beta: f32`, and the withdrawn accepted cost), and §13 (the
  stale `categorical_cross_entropy` symbol). Its decision is otherwise unaffected.
- ADR [0046](0046-slot-newtype-replaces-option-take-around-learn-step.md) — the
  unrepresentable-state posture the `Option` pair alternative would violate, and
  `Slot`'s `pub(crate)` precedent for a deliberately closed in-crate seam.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) /
  [0031](0031-probability-rate-newtypes.md) — the validated-newtype shape
  `ImportanceExponent` follows, via `Priority`.
- ADR [0026](0026-shared-config-validation-convention.md) — the `Validate`
  convention `beta_anneal_steps`'s `nonzero` check follows, and the documented
  panic exception `ImportanceExponent::new` relies on.
- ADR [0049](0049-ppo-gaussian-log-std-is-bounded.md) — the same NaN-poisoning
  class in PPO's Gaussian head, resolved by bounding rather than by newtype
  because the value there is a learned `Param`, not an argument.
- Issue #188 — the NaN-propagation shape §3 refuses to ship inside its own fix.
- Schaul, Quan, Antonoglou, Silver (2016). *Prioritized Experience Replay.*
  ICLR 2016, arXiv:1511.05952v4. §3.4 (`w_i = (1/N · 1/P(i))^β`), Table 3
  (`β₀ = 0.4 → 1.0`, the `ImportanceExponent::ONE` endpoint).
- Code: `crates/rlevo-reinforcement-learning/src/replay/importance_exponent.rs`
  (the newtype and the zero-anneal test); `.../replay/mod.rs`
  (`ReplayStrategy::sample`); `.../replay/prioritized.rs` (the removed
  `debug_assert!`); `.../algorithms/shared.rs` (`UNIFORM_REPLAY_BETA`);
  `.../algorithms/c51/loss.rs:37`
  (`categorical_cross_entropy_per_sample`, the §4 rename).
