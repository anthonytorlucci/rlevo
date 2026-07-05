---
project: rlevo
status: active
type: decision
date: 2026-07-05
tags: [adr, decision, rng, seeding, reset, rlevo-environments, rlevo-reinforcement-learning]
---

# ADR 0029: Host-RNG seeding convention (persistent-stream `reset`)

## Status

**Accepted (2026-07-05).** Filed under issue #197 (cross-crate: codify the
host-RNG `seed_stream` convention) and its sibling bug #104 (environments
re-seed on every `reset()`). Records a workspace-wide convention that was already
implemented correctly in `rlevo-evolution` but scattered as tribal per-file
docstrings elsewhere, and fixes the concrete `rlevo-environments` bug it exposes.

**Scope of change.** The environment `reset()` fix is a **behavioural change**:
successive episodes now differ instead of replaying bit-identical noise (a
semver-relevant contract change, acceptable in alpha). The `rlevo-reinforcement-learning`
trait-bound unification and the `docs/rules.md` rule are otherwise non-breaking.
`rlevo-evolution` is untouched — it is the reference implementation.

## Context

Three facts about randomness in `rlevo` were true before this ADR:

1. **`rlevo-evolution` already got it right.** All EA randomness is host-sampled
   through `seed_stream(base, generation, SeedPurpose)`
   (`crates/rlevo-evolution/src/rng.rs`) over the deterministic
   `rlevo_core::util::seed::SeedStream`. `B::seed` + `Tensor::random` is banned
   because Burn's process-wide RNG mutex makes tensor-sampled draws depend on
   thread schedule, not the seed — this raced the parallel test runner (see the
   `*_converges_on_sphere_d10` flakes). ADR 0016 and 0017 already reference this
   as the "host-RNG sampling convention," but it lived only in those ADRs, a
   memory note, and individual docstrings.

2. **`rlevo-environments` violated the spirit of it (#104).** Many stochastic
   environments began `reset()` with
   `self.rng = StdRng::seed_from_u64(self.config.seed);` — re-seeding the RNG
   from stored config on every reset. Because the seed is fixed, every episode
   started from bit-identical noise, silently defeating the episode-to-episode
   diversity a DRL/POMDP agent is meant to generalise over. The canonical
   🔴 High findings were `pixel_grid.rs` and `locomotion/reacher/env.rs`, but the
   pattern was crate-wide.

3. **`rlevo-reinforcement-learning` was inconsistent about RNG trait bounds.**
   DQN/SAC/DDPG/TD3/C51/QRDQN take `rng: &mut R` with `R: Rng + ?Sized`, while
   the PPO/PPG policy trait and agents took `&mut dyn Rng`. The `dyn` form was
   not load-bearing: `PpoPolicy` is already non-object-safe (no-`self`
   associated fns + an associated type) and is never used as a trait object;
   agents hold the policy as a concrete generic `P`; callers already pass
   `&mut impl Rng`. The split was cosmetic drift, not a design constraint.

A convention that lives only in prose is not enforceable. This ADR promotes it to
a `docs/rules.md` rule and brings the two lagging crates into line.

## Decision

### 1. Environments own a persistent RNG; `reset()` advances it

An environment seeds its RNG **once**, in its constructor (`with_config`). The
`Environment::reset()` implementation draws from that persistent stream and lets
it advance — it must **not** re-assign `self.rng` from `config.seed`. Successive
episodes therefore see independent initial states. Run-level reproducibility is
unaffected: a fixed construction seed yields a fixed *sequence* of resets, so a
whole training run still reproduces bit-for-bit.

The fix removes exactly one line per env (`self.rng = StdRng::seed_from_u64(self.config.seed);`),
passing the existing `&mut self.rng` into the world/state builder — mirroring the
environments that were already correct (`toy_text/frozen_lake.rs`,
`toy_text/blackjack.rs`).

### 2. Deterministic replay is an explicit opt-in: `reset_with_seed`

For the rare case a caller wants a *specific* episode to reproduce (e.g.
replaying a failure), each fixed env exposes an inherent method:

```rust
pub fn reset_with_seed(&mut self, seed: u64) -> Result<Self::SnapshotType, EnvironmentError> {
    self.rng = StdRng::seed_from_u64(seed);
    self.reset()
}
```

This is an **inherent** method on the concrete env types, not a new
`Environment`-trait method — adding it to the trait would force an impl on every
environment and constitute a breaking trait change for a rarely-needed capability
(see Alternatives).

### 3. The problem instance is sampled once at construction, not re-seeded

An environment's fixed *problem instance* — a bandit's arm-means / phases /
context, a procedural grid layout — is sampled **once at construction** and
preserved across resets. `reset()` must not re-seed the RNG to reproduce it.
Where an episode mutates the problem (a non-stationary bandit's random-walk
drift), `reset()` **restores** it to the construction baseline (via a stored
`initial_arm_means`) rather than re-sampling — so the baseline is stable while
the drift and reward noise advance independently each episode.

This makes the **bandit family** fully compliant rather than an exception. Their
prior `reset()` re-seeded `self.rng` from `config.seed` (and redundantly
re-sampled the problem), replaying bit-identical reward noise every episode — the
same #104 bug. Post-ADR: `KArmedBandit` / `NonStationaryBandit` /
`ContextualBandit` keep a fixed problem and draw independent per-episode rewards;
`NonStationaryBandit` restores its drift baseline on reset. Reproducibility for
benchmarks comes from constructing with a fixed seed (`with_seed`), which
`rlevo-benchmarks` already does per trial — not from `reset()`.

`AdversarialBandit` is the one **genuinely deterministic** env: its reward is a
pure function of the step index and per-arm phases, so it has no per-episode
randomness at all. Its RNG was construction-only (phase sampling), so the field
is dropped entirely; `reset()` preserves the fixed schedule. Deterministic envs
are allowed, but must still not re-seed on reset.

Environments whose RNG is never drawn from (the deterministic `_rng` grid layouts
— `empty`, `door_key`, `crossing`, `four_rooms`, …) are unaffected: reseeding an
unused RNG is a no-op, not a diversity bug, and they are left untouched.

### 4. Host-sampling and the RNG trait-bound idiom

- Draw randomness on the **host** via `seed_stream` / `SeedStream`, fill a `Vec`,
  and build tensors with `Tensor::from_data`. `B::seed` + `Tensor::random` is
  forbidden in production paths (process-wide RNG mutex; races parallel tests).
- Functions that need randomness take `rng: &mut R` with `R: rand::Rng + ?Sized`.
  `&mut dyn Rng` is reserved for call sites that genuinely require dynamic
  dispatch, and any such use is documented. The PPO/PPG policy trait method
  `sample_with_logprob` and the `PpoAgent`/`PpgAgent::act` methods are converted
  from `&mut dyn Rng` to the generic bound to match the rest of the crate. `?Sized`
  keeps any surviving `&mut dyn Rng` caller compiling.

### 5. The rule lives in `docs/rules.md` §8

The convention is codified as the `### Host-RNG seeding convention (ADR 0029)`
subsection of §8 (Dependency Usage), alongside the existing `rand` / `rand_distr`
rows and the `seed_stream` references from ADR 0016/0017.

## Consequences

### Positive

- The convention is a single referenceable rule, not tribal knowledge; new envs
  and algorithms have one place to look.
- `reset()` now delivers the episode diversity DRL/POMDP training assumes; #104 is
  fixed at its root across all genuinely-stochastic envs.
- RNG trait bounds are uniform across `rlevo-reinforcement-learning`; `Send`/`Sync`
  reasoning on the hot path no longer varies by algorithm.

### Negative / accepted costs

- `reset()` is a behavioural / semver-relevant change: code that (incorrectly)
  relied on identical per-episode initial states must switch to `reset_with_seed`.
  Acceptable in alpha.
- A `reset_with_seed` method is added to ~11 env types (surface growth), and the
  bandit exception means the rule is "persistent unless documented," not absolute.

### Neutral

- No new dependency. `rlevo-evolution` is unchanged. The `_rng` deterministic
  grids are left alone (no behavioural effect either way).

## Alternatives considered

- **`reset_with_seed` on the `Environment` trait.** Uniform, but a breaking
  trait change forcing an impl (or default) on every environment for a
  rarely-used capability. Rejected in favour of an inherent per-env method.
- **Keep the convention as docstrings / fix only the 4 named files.** This is the
  status quo that let the bug spread crate-wide and kept the rule unenforceable.
  Rejected: codify once, sweep all genuinely-stochastic envs.
- **Treat bandits as a documented exception (keep re-seeding on reset).** The
  first cut of this ADR did. Rejected on execution: re-seeding replays
  bit-identical reward noise every episode — the same #104 bug — and the "same
  problem each episode" goal is met more cleanly by sampling the problem once at
  construction and letting the reward stream advance. Bandits are now compliant,
  not exempt.

## References

- Issues: #104 (env reseed-on-reset bug), #197 (codify the convention).
- ADR [0016](0016-memetic-wrapper-and-local-search-seam.md), ADR
  [0017](0017-probability-model-trait-and-eda-strategy.md) — established the
  host-RNG / `seed_stream` sampling convention this ADR generalises.
- `crates/rlevo-evolution/src/rng.rs` (`seed_stream`, `SeedPurpose`),
  `crates/rlevo-core/src/util/seed.rs` (`SeedStream`) — the reference primitives.
- Reference-correct envs: `crates/rlevo-environments/src/toy_text/frozen_lake.rs`,
  `.../toy_text/blackjack.rs`.
- Memory note `project_evolution_host_rng_convention` — the host-RNG convention
  and the `B::seed` / `Tensor::random` / `thread_rng` ban.
- `docs/rules.md` §8 — the codified rule.
