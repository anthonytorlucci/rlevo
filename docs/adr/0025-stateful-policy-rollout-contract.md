---
project: rlevo
status: active
type: decision
date: 2026-06-30
tags: [hybrid, neuroevolution, rollout-fitness, stateful-policy, recurrent-policy, pomdp, module-eval-fn, contract]
---

# ADR 0025: The stateful-policy rollout contract (`StatefulPolicy`)

## Status

Active. Adopted 2026-06-30. Establishes how a policy carries per-episode memory
across an environment rollout in `rlevo-hybrid`, generalising the RL-coupled
fitness seam from reactive (Markov) policies to recurrent (POMDP) ones.
**Contained to `rlevo-hybrid`** (no `rlevo-core` change, no new cross-crate
trait); **not purely additive** — it removes the stateless policy closure and
changes `RolloutFitness::new`'s signature. Implements issues #91 (stateful seam)
and #92 (dedup onto `ModuleEvalFn`) together. Builds on ADR 0023
(`ObjectiveSense`; this file preserves the maximise-native, no-hand-negation
contract `rollout_fitness` adopted there) and the weight-only neuroevolution
pipeline (`ModuleReshaper` / `ModuleEvalFn` / `WeightOnly`).

## Context

`rlevo-hybrid::RolloutFitness<B, M, E>` scores a flat policy-parameter genome by
environment rollout: a `ModuleReshaper` unflattens each population row into a
policy network `M`, the network drives one or more episodes, and mean episode
return is the fitness. The bridge from module to environment action was a
**stateless closure**:

```rust
type PolicyFn<B, M, E> =
    Arc<dyn Fn(&M, &E::ObservationType, &Dev<B>) -> E::ActionType + Send + Sync>;
```

`rollout_once` called it once per step with **no carried state**. So the policy
could only express a reactive map `observation → action`. That was a deliberate
v1 scope (rank-1 classic control — CartPole, MountainCar, Pendulum, Acrobot — all
Markov, where reactive is optimal), but it is **structurally incapable** of the
library's POMDP envs. The Santa Fe ant (`SantaFeAnt`, #68) declares
`MarkovState::is_markov() == false`: its one-bit `food_ahead` percept makes the
optimal policy provably require internal memory to cross the trail's gaps. #69
worked around this by scoring through `rlevo-evolution::ModuleEvalFn` directly —
whose scorer `Fn(&M) -> f32` owns the whole rollout and threads a recurrent hidden
state as a local — keeping the example self-contained but leaving the RL-coupled
fitness path memoryless.

Two facts shape the fix:

- **`RolloutFitness` *is* a `ModuleEvalFn` with a rollout scorer.** Both carried
  the identical slice/unflatten/collect loop; the only difference is the per-row
  score. `rlevo-hybrid` already depends on `rlevo-evolution`, so the duplication
  (#92) is removable by composition.
- **Hidden-state threading lives inside that rollout scorer.** So #91 and #92
  compose: do them together and there is one memory-capable scoring path; a
  future batched-forward fast path is then written once.

The maintainer resolved the open design question (OQ-1) in favour of a **trait**,
not a closure pair: the recurrent contract should be a named, discoverable type
implemented by the policy module (mirroring #69's `AntPolicy`-on-the-module
shape), not an anonymous `reset`/`act` closure tuple.

Constraints:

- Rank-1 (`E: Environment<1,1,1>`) and forward-only (`B: Backend`, not
  `AutodiffBackend`) are unchanged — memory is a runtime value, not a tracked leaf.
- `ModuleEvalFn`'s scorer is `Fn(&R::Module) -> f32 + Send` with **no device
  parameter** (it captures the device, as #69's `make_scorer` does).
- Maximise-native, no hand-negation (ADR 0023) must be preserved.
- Pre-1.0 with API stability as the active focus — the right time to break `new`.

## Decision

**Express the rollout contract as a `StatefulPolicy` trait implemented by the
policy module, and refactor `RolloutFitness` to delegate scoring to an inner
`ModuleEvalFn` whose rollout scorer drives that trait.**

### 1. `StatefulPolicy<B, E>` — the contract

```rust
pub trait StatefulPolicy<B: Backend, E: Environment<1, 1, 1>> {
    type Hidden;                                   // per-episode state; () if reactive
    fn reset(&self, device: &Dev<B>) -> Self::Hidden;
    fn act(&self, hidden: &mut Self::Hidden,
           obs: &E::ObservationType, device: &Dev<B>) -> E::ActionType;
}
```

The evolved module `M` implements it: the reshaper unflattens a genome into `M`,
and the rollout drives `M` through `reset` (once per episode) + `act` (per step,
threading `&mut Hidden`). `act` owns the **full** `&Obs → Action` mapping (decode,
forward, select, encode) so the seam stays **env-generic** — the trait is *not*
the #69 decoded-percept/return-logits shape, which is an example convenience.

### 2. `ReactivePolicy<B, E>` — the `Hidden = ()` convenience

```rust
pub trait ReactivePolicy<B: Backend, E: Environment<1, 1, 1>> {
    fn act(&self, obs: &E::ObservationType, device: &Dev<B>) -> E::ActionType;
}
impl<B, E, P: ReactivePolicy<B, E>> StatefulPolicy<B, E> for P {
    type Hidden = ();
    fn reset(&self, _d: &Dev<B>) {}
    fn act(&self, _h: &mut (), o: &E::ObservationType, d: &Dev<B>) -> E::ActionType {
        ReactivePolicy::act(self, o, d)
    }
}
```

Markov (classic-control) policies impl the one-method `ReactivePolicy` and get
`StatefulPolicy` free; recurrent policies impl `StatefulPolicy` directly. **A
recurrent policy must never impl `ReactivePolicy`** (the blanket would overlap its
direct impl). The blanket's coherence was validated at implementation time
against a concrete recurrent `GruPolicy` (which does not impl `ReactivePolicy`) —
it compiles. If it ever fails for a future module, the fallback is each reactive
module impl-ing `StatefulPolicy` with `type Hidden = ()` directly — identical
contract, only the convenience changes.

### 3. `RolloutFitness` delegates to an inner `ModuleEvalFn` (#92)

`RolloutFitness<B, M, E>` (`M: Module<B> + Sync + StatefulPolicy<B,E>`) holds
`ModuleEvalFn<B, ModuleReshaper<B,M>, Box<dyn Fn(&M)->f32+Send>>`; its
`evaluate_batch` is a one-line delegation and the hand-written loop is **deleted**.
The reshaper moves into the inner `ModuleEvalFn` (single owner). The rollout
scorer is the only caller of the now free-standing `rollout_once`, which
`StatefulPolicy::reset`s once per episode and threads `&mut Hidden` across the
step loop.

> **Deviation from the spec draft (implementation finding).** The spec/draft
> named the scorer type `Arc<dyn Fn(&M)->f32+Send+Sync>`. **`Arc<F>` does not
> implement `Fn`** in std — only `Box<F>` and `&F` do — so it cannot satisfy
> `ModuleEvalFn`'s `F: Fn(&R::Module) -> f32 + Send` bound. The committed code
> uses **`Box<dyn Fn(&M)->f32 + Send>`**: `Box` implements `Fn`, is nameable, and
> `+ Send` matches the `BatchFitnessFn: Send` supertrait (the harness never
> requires the fitness `Sync`, so `Sync` is dropped). The captured `env_factory`
> stays an `Arc<dyn Fn()->E + Send + Sync>` inside the scorer. The contract is
> otherwise exactly as decided.

### 4. `new` loses the closure, gains a device

Because the inner scorer is `Fn(&M) -> f32` (no device param), the device is
**captured at construction**. `RolloutFitness::new(reshaper, env_factory,
episodes, max_steps, device)` — **no policy closure** (the module is the policy)
and **+`device`**. This is the one accepted v1 API break; it matches
`PolicyNeuroevolution::new` (already device-taking) and `ModuleEvalFn`'s
single-device convention. The device later passed to `evaluate_batch` must equal
the captured one.

### 5. Sense unchanged (ADR 0023)

The inner `ModuleEvalFn` is built with the default `ObjectiveSense::Maximize`;
`RolloutFitness::sense()` returns `self.inner.sense()`. Mean episode return flows
through unnegated. No regression to the ADR-0023 contract.

### 6. `PolicyNeuroevolution` follows the trait

Add `M: StatefulPolicy<B,E>` (reactive modules satisfy it via §2). No new
constructor: callers build a reactive or recurrent `RolloutFitness` and pass it to
the existing `new`. Statefulness rides entirely on the module's trait impl.

## Consequences

**Positive**

- **POMDP policies are first-class on the RL-coupled fitness path.** The library's
  POMDP envs (Santa Fe ant today, more later) can be solved through
  `RolloutFitness`/`PolicyNeuroevolution`, not only through an example-local
  `ModuleEvalFn` workaround.
- **One scoring path.** The duplicated unflatten-score loop is gone (#92); a future
  batched-forward fast path is written once.
- **Named, discoverable contract.** `StatefulPolicy` is greppable and
  implementable, unlike an anonymous closure pair; it factors out exactly what #69
  built by hand.
- **Reactive ergonomics preserved.** `ReactivePolicy` keeps Markov policies a
  one-method impl; the `Hidden = ()` restriction is explicit, not a special case in
  the engine.
- **Right time.** Pre-1.0, in the same window ADR 0023 already rewrote this file.

**Negative / accepted costs**

- **`RolloutFitness::new` breaks** (closure removed, `device` added) and reactive
  call sites (`cartpole_smoke.rs`, classic-control examples) migrate to
  `ReactivePolicy`. Mechanical, few sites, acceptable pre-1.0.
- **`Box<dyn Fn>` dispatch** replaces a monomorphized closure — one indirection per
  member per generation, negligible against a 600-step rollout. Revisit only if it
  profiles hot.
- **Device-at-construction** adds a second device reference (captured vs passed)
  that must agree — the existing ModuleEvalFn single-device convention, documented.

**Neutral**

- Rank-1 and forward-only are unchanged; `Hidden` is a runtime value, so
  `ModuleReshaper` flattening (and its `BatchNorm`-running-stats caveat) is
  unaffected.
- No `rlevo-core` or schema change — the contract is local to `rlevo-hybrid`.

## Alternatives considered

**Two closures + a `Hidden` type param (`RolloutFitness<…, H = ()>` with
`reset`/`act` closures).** The lower-friction generalization, continuous with the
v1 closure design, `H = ()` default keeping call sites source-compatible.
**Rejected by maintainer (OQ-1):** an anonymous closure pair is less discoverable
and reusable than a named trait; the trait mirrors #69's working `AntPolicy` and
gives recurrent policies a first-class type.

**A `StatefulRolloutFitness` variant beside the stateless `RolloutFitness`.**
Preserve v1 untouched; add a parallel stateful struct. Rejected: two structs
duplicate the (now `ModuleEvalFn`-delegated) scaffolding the change is trying to
unify, and split the policy-hosting story; the trait unifies both behind one
struct with `Hidden = ()` as the reactive case.

**Keep scoring through `ModuleEvalFn` directly (the #69 workaround), forever.**
Rejected: it leaves the RL-coupled seam permanently memoryless and pushes every
POMDP neuroevolution user to re-hand-roll the rollout the library should own
("resolve, don't just document").

**Extend `ModuleEvalFn`'s scorer to take a device.** Would avoid capturing the
device in `RolloutFitness::new`. Rejected: changes `ModuleEvalFn`'s public scorer
contract and all its callers for one consumer's convenience; capturing the device
is the established #69 pattern.

**Promote `StatefulPolicy` to `rlevo-core`.** Rejected (for now): only
`rlevo-hybrid` consumes it; unlike `ObjectiveSense` (RL + EC + benchmarks all
reason about direction), no second crate needs the rollout-policy contract.
Revisit if an RL-side recurrent driver wants it.

## References

- Spec: vault `specs/2026-06-30-stateful-rollout-fitness/index.md` (KD-1..KD-4,
  API surface, migration, test plan).
- Research: vault `research/santa-fe-ant/recurrent-memory-policy-and-rolloutfitness-gap.md`
  — the gap audit (§2) and the duplication (§2a).
- [ADR 0023](0023-objective-sense-and-maximize-convention.md) — the maximise-native /
  no-hand-negation contract this preserves; rewrote `rollout_fitness` in the same
  window.
- [#91](https://github.com/anthonytorlucci/rlevo/issues/91) /
  [#92](https://github.com/anthonytorlucci/rlevo/issues/92) — the issues this
  decision implements together.
- `crates/rlevo-hybrid/src/policy.rs` — the two traits + blanket impl.
- `crates/rlevo-hybrid/src/rollout_fitness.rs`,
  `.../policy_neuroevolution.rs` — the change surface.
- `crates/rlevo-evolution/src/module_eval_fn.rs` — the inner scorer this delegates
  to (unchanged).
