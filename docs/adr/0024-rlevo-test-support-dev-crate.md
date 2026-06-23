---
project: rlevo
status: active
type: decision
date: 2026-06-23
tags:
  - adr
  - decision
  - architecture
  - testing
  - dev-dependency
  - rlevo
---

# ADR 0024: `rlevo-test-support` dev-only crate for the RL integration suite

## Status

Active. Adopted 2026-06-23. Adds one crate (`rlevo-test-support`) to the
workspace, but **only** as a `[dev-dependencies]` / `publish = false` member — it
is not part of the shipped, production-crate closed set the way the crates in
ADR 0001–0023 are. Driven by the pre-release review of the cross-crate RL
integration tests in `crates/rlevo/tests/`.

The crate is the workspace's general home for shared cross-crate test
scaffolding, not an RL-only kit. The **RL** integration suite is simply its
first consumer — it had the most acute duplication — so the initial module set
and fixtures are RL-shaped. The **evolution** (`rlevo-evolution`) and **hybrid**
(`rlevo-hybrid`) cross-crate suites are expected to grow the same shared shapes
(seeded fixtures, determinism preambles, standard assertions, suite macros) and
should land them here additively rather than re-duplicating per file. This ADR
establishes the crate and its boundary; it does **not** pre-build evo/hybrid
fixtures (added when the first such test needs them).

## Context

Each of the eight RL algorithm files in `crates/rlevo/tests/`
(`ddpg_integration.rs`, `td3_integration.rs`, `sac_integration.rs`,
`dqn_integration.rs`, `c51_integration.rs`, `qrdqn_integration.rs`,
`ppo_integration.rs`, `ppg_integration.rs`) compiles to its **own test binary**
(ADR 0012's three-tier placement rule), as do the evolution cross-crate suites
alongside them. Shared code between binaries therefore cannot live as a sibling
`mod`; it has to live in a library crate that every test binary depends on — and
that need is not specific to RL, it is intrinsic to the per-binary layout.

Before this crate existed, the eight RL files independently duplicated:

1. **A ~140-line synthetic `LinearEnv`** — the 1-D continuous tracking fixture
   (`reward = -(a - x)²`) — copied verbatim into each continuous-control test
   (DDPG / TD3 / SAC).
2. **The Burn `Flex` determinism preamble.** `Flex` exposes a *process-global*
   RNG for weight init and dispatches through `rayon` whose float reduction
   order is non-deterministic under multi-threading. Every file repeated the
   same three-part dance: pin `rayon` to one thread, serialise tests within the
   binary against a backend lock, and seed the backend before constructing
   networks.
3. **The acceptance assertions** — finite-reward checks, a "beats the random
   baseline" check, and bit-/sequence-equal reproducibility checks — each
   hand-written per file.
4. **A hard-coded random baseline.** The convergence checks asserted against a
   *commented* random-policy score (`assert_beats_baseline(avg, -1.0, -6.67)`,
   where `-6.67` was a claim in a doc comment, never a measurement). The same
   magic numbers (`-6.67`, `-1500`, `28`) recurred across files and could drift
   from the environment they purported to describe.

Two smaller problems rode along: the per-file test **names had drifted** into
four inconsistent shapes (`*_short_run_produces_finite_rewards` vs
`*_pendulum_smoke`; `*_reproducibility_flex`; `*_solves_linear_1d_continuous` vs
`*_reaches_100`; `ppg_cart_pole_learns_above_random` that was *implemented* as an
absolute `reaches(28)`), and standalone `examples/envs/classic/*_random.rs`
random-policy rollouts existed purely as printed diagnostics that gated nothing.

## Decision

Extract a dev-only crate **`rlevo-test-support`** (`publish = false`) consumed by
`rlevo` through `[dev-dependencies]`, holding the algorithm-agnostic boilerplate
so each test file is left with only its agent/network configuration and its
genuinely algorithm-specific checks.

Its dependency set tracks the fixtures it actually holds. Today those are RL
fixtures, so it depends only on `rlevo-core` and `rlevo-environments`. When
evolution / hybrid fixtures land it gains `rlevo-evolution` / `rlevo-hybrid`
accordingly — feature-gated per domain (e.g. `features = ["rl", "evo"]`) if the
dependency weight or compile cost warrants it, so an RL-only test binary need
not pull the evolution stack. The invariant below holds regardless of which
domains are present.

### Modules (RL-first set)

The modules below are what the RL consumer needed first. New domains add their
own modules additively — e.g. evolution would contribute landscape/strategy
fixtures and, if its host-RNG determinism (seeded `seed_stream`, single-thread
rayon) needs a shared preamble, a sibling to `flex` — without disturbing these.

- **`env`** — the shared `LinearEnv` fixture (state / observation / bounded
  continuous action + all the `rlevo-core` trait impls) and `cartpole_seeded`,
  the canonical discrete fixture.
- **`flex`** — the `FlexAutodiff = Autodiff<Flex>` backend alias, the per-binary
  `flex_guard()` (rayon-pin + backend lock RAII), and `seeded_device()`.
- **`baseline`** — a generic `random_return` rollout plus `uniform_discrete` /
  `uniform_bounded` action samplers (keyed on the `DiscreteAction` /
  `BoundedAction` traits). This **measures** the random-policy baseline a
  learning test must beat instead of hard-coding it; the rollout is fully
  seeded, so the resulting threshold is deterministic, not flaky.
- **`assert`** — `assert_all_finite`, `assert_improves_over_random` (trained >
  measured random + margin), `assert_reaches` (absolute floor), and the
  bit-/sequence-equal reproducibility assertions.
- **`TrainOutcome`** (crate root) — the `{ avg_score, rewards }` a test's single
  `run(seed, total)` function returns; the only algorithm-specific glue the
  macros consume.

### Suite macros

Two `#[macro_export]` declarative macros generate the universal `#[test]`
scaffolding (the `flex_guard` claim, the `run` call, the assertion):

- `rl_learning_test!` with `improves_over_random(margin = …)` (needs a `random =
  <fn(seed) -> f32>` callback that rolls a random policy over the same env) or
  `reaches(threshold)` (absolute).
- `rl_reproducibility_test!` with `bits` (scalar `avg_score`) or `seq` (full
  reward sequence).

Both forward a leading `$(#[$attr:meta])*` so `#[ignore]` is optional per call.
The `rl_` prefix is deliberate: these encode RL-specific assertion semantics
(reward sequences, a `flex_guard` preamble). A future evolution suite would add
its own macro family (e.g. `eo_*`) rather than overload these, sharing only the
domain-agnostic primitives (`baseline`, `assert`, `TrainOutcome`).

### Test naming convention

The categories converge on `<algo>_<env>_<category>` — RL's instantiation of a
general `<subject>_<problem>_<category>` shape that an evolution suite would
mirror (e.g. `<strategy>_<landscape>_<category>`), reusing whichever category
suffixes apply:

| Category | Suffix | Assertion |
|----------|--------|-----------|
| Smoke | `_produces_finite_rewards` | all metric columns finite |
| Reproducibility | `_flex_reproducibility` | two seeded runs match |
| Beats random | `_improves_over_random` | trained > **measured** random + margin |
| Absolute target | `_converges` | `>=` a fixed floor |
| Acceptance | `_acceptance` | the hours-long full-solve floor |

`<env>` earns its place: DDPG/SAC/TD3 smoke-test on `pendulum` but learn and
reproduce on `linear`, so the token disambiguates rather than padding.
Thresholds are **not** baked into names (`_reaches_100` had already gone stale —
PPO's asserted 80, not 100); the numeric bar lives in the `#[ignore]` string.
The `flex` qualifier in `_flex_reproducibility` names the Burn backend whose
global RNG the check pins; a domain without that backend would qualify the
suffix differently (or drop the qualifier).

### What deliberately stays in each test file

The per-algorithm **networks** (`Actor`/`Critic`, `DqnMlp`, `C51Mlp`,
`QrDqnMlp`, the SAC `StochasticActor`, the PPO/PPG `ValueMlp`, the PPG policy
head) stay test-local. They implement *algorithm-specific* Burn model traits
(`ddpg_model::DeterministicPolicy`, `sac_model::SquashedGaussianPolicy`,
`ppo_value::PpoValue`, …) that live in `rlevo-reinforcement-learning`. Hoisting
them into the support crate would force a dev-dependency on
`rlevo-reinforcement-learning` and couple the fixtures to algorithm internals —
the opposite of the boundary below. Algorithm-specific behavioural tests
(`*_act_with_matches_deterministic_act`, `sac_alpha_moves_under_autotune`,
`ppg_aux_phase_actually_runs`, …) likewise stay with their algorithm.

The three `examples/envs/classic/*_random.rs` rollouts are deleted: the
uniform-random baseline they demonstrated now lives in `baseline` where it
actually gates assertions.

## Consequences

- **The dev/prod boundary is one-directional and explicit.** `rlevo-test-support`
  *consumes* production crates (today `rlevo-core` + `rlevo-environments`;
  `rlevo-evolution` / `rlevo-hybrid` as those fixtures land); no production crate
  depends on it, and nothing depends on it outside `[dev-dependencies]`. It
  cannot constrain a shipped type no matter how many domains it grows to serve —
  the same posture as the benchmark/viz isolation rule (`rules.md §1`), applied
  to test scaffolding. It is `publish = false`, so it never reaches crates.io.
- **Determinism lives in one place.** The `Flex` preamble's subtle invariant
  (per-binary lock granularity, seed-before-construct ordering) is implemented
  and documented once in `flex`, not re-derived eight times.
- **Baselines track the environment.** "Beats random" now runs an actual seeded
  random policy over the same env; changing a fixture's dynamics automatically
  moves its baseline. The stale-magic-number failure mode is gone.
- **Adding an algorithm test is smaller.** A new file writes its networks + a
  `run()` + a `random_<env>()` and invokes the macros; the universal three
  checks come for free with conventional names.
- **No production-code or schema change.** This is test-infrastructure only; no
  `FORMAT_VERSION` bump, no change to any shipped trait.

## Alternatives considered

- **A shared `mod` included via `#[path]` into each test file:** rejected — each
  integration test is a separate binary, so an included module is recompiled per
  binary and, worse, a `static` backend lock in it would be per-*inclusion*, not
  per-binary; the library-crate `static` is exactly the right granularity.
- **Hoisting the networks too (a fuller "RL test kit"):** rejected — it would
  pull `rlevo-reinforcement-learning` into the support crate's dependencies and
  couple fixtures to algorithm-trait internals. The line is drawn at
  algorithm-agnostic scaffolding.
- **Keep hard-coded baselines, just deduplicate them into a constant:** rejected
  — a shared `const RANDOM_LINEAR: f32 = -6.67;` is still an unverified claim;
  measuring is barely more code and self-correcting.
- **Make it a normal (publishable) crate:** rejected — it has no value outside
  the test suite and would widen the published API surface for no consumer.

## References

- Pre-release integration-test review, branch `pre-release/v0.3-review-2`.
- ADR 0012 (three-tier test placement — why these are separate binaries).
- The `Flex` + rayon determinism contract the `flex` module encapsulates:
  reproducible training needs both a seeded backend RNG (weight init) and rayon
  pinned to one thread, with tests serialised against a per-binary backend lock.
- `rules.md §1` — production crates must not depend on benchmark/viz crates; this
  ADR applies the same consume-don't-constrain posture to a dev-only crate.
