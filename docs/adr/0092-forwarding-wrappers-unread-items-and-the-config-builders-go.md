---
project: rlevo
status: active
type: decision
date: 2026-10-22
tags: [adr, decision, architecture, rlevo-evolution, rlevo-hybrid, rlevo-harness, rlevo-reinforcement-learning, config, neuroevolution, public-surface, breaking-change]
---

# ADR 0092: Forwarding wrappers, unread items and the config builders go

## Status

**Accepted (2026-10-22).**

The third pruning after ADRs 0090 and 0091, under the test ADR 0081 sets
for every public item: it needs an implementor and a consumer. This one
applies the test to types, fields and constants that add a name and no
behaviour: a wrapper whose every method forwards, a field nothing reads,
keys nothing emits, an empty module, and builders that spell a struct
literal a second way.

Supersedes these clauses. The rest of each ADR stands.

- ADR [0025](0025-stateful-policy-rollout-contract.md) Decision 6,
  "`PolicyNeuroevolution` follows the trait": the type is deleted
  (Decision 1 here).
- ADR [0085](0085-the-generation-loop-is-named-evolutionary-loop.md)
  Decision 7, "`PolicyNeuroevolution` follows": the same.

## Context

Read on `main` at `a699a152`, with `rg -w` for each name and the compiler
for each removal:

- **`WeightOnly<B, S, M>`** (`rlevo-evolution`, 263 lines) is a `Strategy`
  whose `init`, `ask`, `tell` and `best` each pass their arguments to the
  inner strategy and return its result. What it adds is a `ModuleReshaper`
  built from a template, read through `num_params()` and `reshaper()`. Every
  caller also built a second `ModuleReshaper` from the same template for its
  fitness adapter, so a run held two reshapers whose widths agreed by
  convention.
- **`PolicyNeuroevolution<B, S, M, E>`** (`rlevo-hybrid`, 154 lines) holds
  one `EvolutionaryLoop<WeightOnly, RolloutFitness>` and forwards `reset`,
  `step`, `is_done`, `best`, `latest_metrics` and `generation` to it; its
  `run` is `reset`, then `step` until `is_done`. It has no
  `GenerationProbe` impl, so the evaluation harness could not run it, and
  its own documentation told a reader to build the inner loop instead. Its
  constructor carried a TODO about the two reshapers it could not
  collapse. Its one user was a smoke test.
- **`Suite::default_config`** (`rlevo-harness`): `Suite::new(name, config)`
  stored the config, and only `Suite`'s `Debug` read it.
  `Evaluator::run_suite` runs a suite under the evaluator's own
  `EvaluatorConfig`. All 27 calls passed a config that nothing applied, the
  two BYOE probe templates among them, so an author who tuned it saw no
  effect.
- **`metrics::rl`** (`rlevo-harness`): six key constants (`rl/policy_loss`,
  `rl/value_loss`, `rl/approx_kl`, `rl/entropy`, `rl/epsilon`,
  `rl/learning_rate`) that no agent emits; the workspace's one non-default
  `emit_metrics` is a harness test spy. A namespace test read them.
- **The eight RL config builders**, `DqnTrainingConfigBuilder` and its
  siblings for C51, QR-DQN, PPO, PPG, DDPG, TD3 and SAC: 1,228 lines with
  their documentation, 108 setters. Each setter assigns one public field of
  a config that implements `Default` and `Validate` (`prioritized_replay`
  wraps its argument in `Some`; PPG's `with_ppo` maps a closure over the
  inner PPO config), and `build()` is `validate()`, then return. ADR 0055
  already makes struct-update syntax the tuning idiom (Decision 1, `pub`
  fields) and puts validation in the consuming constructor (Decision 2),
  which all eight agents do in their first statement. So each builder was a
  second spelling of the struct literal, and its `build()` a second call of
  the agent's validation.
- **The nine environment config builders**, `CartPoleConfigBuilder` and
  its siblings for Acrobot, LunarLander, BipedalWalker, CarRacing,
  Blackjack, CliffWalking, FrozenLake and Taxi, each with an
  `XConfig::builder()` constructor. Every setter sets the field of its
  name. The three box2d builders' `build()` validated and the six others'
  did not; every environment's `with_config` validates. The four toy-text
  builders kept their own copy of the fields with a derived `Default`, and
  FrozenLake's had drifted: `FrozenLakeConfig::builder().build()` set
  `is_slippery: false`, where `FrozenLakeConfig::default()` sets `true`,
  Gymnasium's default. 25 of the 34 environment configs had no builder, and
  every call to the nine was in the crate's own tests and doc examples.
- **Dependencies no source file names**: `rlevo-hybrid`'s `rand` and
  `serde`, and `rlevo-harness`'s `rand_distr`.
- **`games::connect_four`** (`rlevo-environments`): a module holding only
  its own documentation, 12 lines, no item. It sits beside `games::chess`
  (board state, legal-move generation and the `AlphaZero` action and
  observation encodings, about 1,400 lines and 25 tests, with no
  `Environment` impl yet) under the `#[doc(hidden)]` `games` module.

## Decision

### 1. `WeightOnly` and `PolicyNeuroevolution` are deleted

Weight-only neuroevolution needs no type of its own. A caller builds one
`ModuleReshaper` from the template, sizes the strategy's parameters with
its `num_params()`, moves it into the fitness adapter (`ModuleEvalFn`, or
`rlevo-hybrid`'s `RolloutFitness`), and drives any flat-genome strategy
with `EvolutionaryLoop`:

```rust
let reshaper = ModuleReshaper::new(template);
let params = GaConfig::default_for(64, reshaper.num_params());
let fitness = RolloutFitness::new(reshaper, CartPole::default, 1, 500);
let mut evo_loop = EvolutionaryLoop::new(
    GeneticAlgorithm::new(), params, fitness, seed, device, max_generations,
)?;
```

There is one reshaper, so the widths agree by construction and the TODO
goes, and the loop implements `GenerationProbe`, so the harness runs a
policy search as it runs any other. `run` is not replaced: it was `reset`
and a `while !is_done() { step() }` loop. `rlevo-hybrid` keeps
`StatefulPolicy` and `RolloutFitness`.

### 2. `Suite::new` takes a name

`Suite::default_config` is deleted, and `Suite::new(name)` takes only the
name. How a suite runs is the evaluator's configuration, the one
`run_suite` reads.

### 3. `metrics::rl` is deleted

The `rl/*` namespace stays agent-owned and unreserved
(`is_harness_reserved`), and its test checks two literal names. A key
constant returns with the first agent that emits it.

### 4. The config builders are deleted

An RL config is tuned with struct-update syntax and validated by the agent
that consumes it:

```rust
let config = DqnTrainingConfig {
    learning_rate: 5e-4,
    batch_size: 64,
    ..Default::default()
};
let agent = DqnAgent::new(policy_net, config, device)?; // validates
```

A caller that wants the check earlier calls `config.validate()`. The four
RL builders that had a struct-level example (DQN, DDPG, SAC, TD3) leave it
on their config struct, in this form.

The nine environment builders go the same way: an environment config is a
struct literal over its `Default`, and `with_config` validates it. The
seven doc examples that used a builder show the literal, and FrozenLake has
one default again.

### 5. Dependencies no source file names are dropped

`rlevo-hybrid` drops `rand` and `serde`; `rlevo-harness` drops
`rand_distr`. `Cargo.lock` loses those three edges.

### 6. `games::connect_four` is deleted; chess stays

The empty module goes. Chess stays, at the maintainer's decision:
`AlphaZero`'s chess is what inspired a deep RL library in Burn, and the
board, move generation and encodings are the work toward its
`Environment` impl. It stays hidden (`#[doc(hidden)]`, with its
dead-code allow) until that impl lands, so it is not offered to an
outside author as surface, and ADR 0081's test applies to it then.

## Consequences

### Positive

- 3,891 lines removed and 1,494 added across 86 files before
  documentation; the additions are mostly the 160 rewritten configs, one
  field per line where a builder chain had one setter per line. The Rust
  source of `rlevo-reinforcement-learning` falls from 40,377 lines to
  39,103, `rlevo-environments`' from 62,903 to 62,284, `rlevo-evolution`'s
  from 40,981 to 40,709, `rlevo-hybrid`'s from 445 to 287, and
  `rlevo-harness`'s from 13,157 to 13,108.
- One way to configure an agent or an environment, the one ADR 0055 named;
  one way to run weight-only neuroevolution, in the loop every other
  strategy runs in; one place a suite's run is configured.
- A config has one default: the builder that disagreed with it is gone.
- A policy search runs through the evaluation harness without a wrapper.

### Negative / costs

- Breaking for anyone who named `WeightOnly`, `PolicyNeuroevolution`,
  `Suite::default_config`, the two-argument `Suite::new`, a `metrics::rl`
  constant, an RL or environment config builder, an `XConfig::builder()`
  constructor or the hidden `games::connect_four` module. The changelog
  gives each replacement.
- A caller who relied on `build()` to validate before passing the config on
  now gets the error from the agent's constructor or `with_config`, or calls
  `validate()` itself. The tests that asserted a config validates relied on `build()`,
  and they call `validate()`.

### Neutral

- `MemeticWrapper` is now the one `Strategy` that wraps another. ADR
  0087's argument against a `Strategy` method every wrapper must forward
  holds with one wrapper instead of two.

## Alternatives considered

- **Keep `WeightOnly` as the source of the genome width.** It held a
  reshaper the fitness adapter could not share, so it made a second source,
  not a single one.
- **Give `PolicyNeuroevolution` a `GenerationProbe` impl.** It would forward
  one more trait to the loop it holds, which already implements it.
- **Make `Suite` apply its config.** Two configurations for one run, one
  overriding the other, where one is enough.
- **Keep the builders as an additive choice**, which ADR 0055's Neutral
  section allows. Nothing requires one, each duplicated the struct literal
  and the agent's validation, and the `cargo generate` agent template would
  have stamped a ninth.

## References

- ADR 0081, and ADR 0090 Decision 8 for traits — the test.
- ADR 0055 — struct-update syntax is the tuning idiom; validation happens
  in the consumer.
- ADRs 0025 and 0085 — the superseded clauses above.
- ADR 0087 — the forwarding argument in its alternatives.
