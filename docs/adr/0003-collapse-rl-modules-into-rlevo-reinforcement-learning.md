---
project: rlevo
status: active
type: decision
date: 2026-04-28
tags: [adr, decision, architecture, crates, rl, rlevo-core, rlevo-reinforcement-learning, rlevo-utils]
---

# ADR 0003: Collapse RL-only modules from `rlevo-core` into `rlevo-reinforcement-learning`; fold `rlevo-utils`

## Status

Active. Extends [0002-collapse-evolution-traits-into-rlevo-evolution](0002-collapse-evolution-traits-into-rlevo-evolution.md) by
applying the same "premature centralization" test to the RL side of the
workspace and to the empty `rlevo-utils` crate.

## Context

`rlevo-core` was created to hold "shape-erased vocabulary shared across
downstream crates" — the agent / environment contract. ADR 0002 cut the
EA traits out (`rlevo-evolution` no longer depends on `rlevo-core`). A
review on 2026-04-28 found a symmetric drift on the RL side: three
modules in `rlevo-core` are consumed **only** by `rlevo-reinforcement-learning`, with no
consumer in `rlevo-environments`, `rlevo-evolution`, `rlevo-hybrid`, or
`rlevo-benchmarks`.

Workspace-wide ripgrep of every public symbol in `crates/rlevo-core/src/lib.rs`:

| Module                                                                                         | Consumed by                                                                                                          | RL-specific? |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------ |
| `memory.rs` (`PrioritizedExperienceReplay`, `TrainingBatch`, `ReplayBufferError`)              | `rlevo-reinforcement-learning` only (8 algorithms — DQN, C51, QR-DQN, DDPG, TD3, SAC, plus PPO/PPG via `AgentStats`) | **Yes**      |
| `experience.rs` (`ExperienceTuple`, `History`, `HistoryRepresentation`, `SufficientStatistic`) | `rlevo-reinforcement-learning` only — and only via `memory.rs`                                                       | **Yes**      |
| `metrics.rs` (`PerformanceRecord`, `AgentStats`)                                               | `rlevo-reinforcement-learning` only (every algorithm)                                                                | **Yes**      |

By the ADR 0002 test (**>1 downstream consumer with stable shared
vocabulary**), all three are premature centralizations. They belong in
`rlevo-reinforcement-learning`.

A second drift was visible in the workspace: `rlevo-utils` shipped a
single function (`math::combinations(n, k)`) with **zero** workspace
consumers. The crate had been a placeholder since the project started.

The exploration also confirmed two non-decisions worth recording so we
do not relitigate them later:

- **Do not collapse `rlevo-reinforcement-learning` and `rlevo-environments`.** They have *zero*
  runtime coupling — `rg "use rlevo_(reinforcement-learning|environments)" crates/rlevo-(environments|reinforcement-learning)/src`
  returns nothing. They have disjoint dep cones (`rapier2d`/`rapier3d`/
  `nalgebra` for environments vs `burn` with `train`/`autodiff`/`tui`/`metrics`
  for reinforcement-learning). The boundary is load-bearing.
- **The EA↔RL bridge stays at `rlevo-hybrid`.** `rlevo-evolution`
  parameterises on `Backend` (CubeCL kernels, `Tensor<B, 2>` populations)
  but never touches `burn::module::Module`. Neuroevolution will land as
  a new EA strategy whose `Genome` wraps a Burn `Module`; that drops
  cleanly into the existing layering and does *not* force any item into
  `rlevo-core`.

## Decision

**Move three RL-only modules from `rlevo-core` to `rlevo-reinforcement-learning`, fold `rlevo-utils` into `rlevo-core::util`, and delete `rlevo-utils` from the workspace.**

Concretely:

1. **Move (verbatim, via `git mv` to preserve history)**
   - `crates/rlevo-core/src/memory.rs`     → `crates/rlevo-reinforcement-learning/src/memory.rs`
   - `crates/rlevo-core/src/experience.rs` → `crates/rlevo-rl/src/experience.rs`
   - `crates/rlevo-core/src/metrics.rs`    → `crates/rlevo-reinforcement-learning/src/metrics.rs`
   - `crates/rlevo-core/tests/integration_test.rs` → `crates/rlevo-rl/tests/integration_test.rs`
     (the test exercises `Environment` *plus* the replay buffer and
     `AgentStats`; with two of three components moved, the integration
     test now belongs in `rlevo-reinforcement-learning` where the cross-crate surface is
     visible.)
2. **Inside the moved files**, rewrite `use crate::base::…` and
   `use crate::state::MarkovState` to `use rlevo_core::base::…` /
   `use rlevo_core::state::MarkovState`. `rlevo-reinforcement-learning` already deps on
   `rlevo-core` so no manifest change is needed. `SufficientStatistic`'s
   `MarkovState` supertrait still resolves; `MarkovState` stays in
   `rlevo-core::state` (see "conservative dead-code policy" below).
3. **Inside `rlevo-reinforcement-learning`**, rewrite the 14 import sites that previously
   read `use rlevo_core::{memory, experience, metrics}::…` to use
   `crate::…` paths. Mechanical sweep across
   `crates/rlevo-rl/src/algorithms/{dqn,c51,qrdqn,ddpg,td3,sac,ppo,ppg}/*_agent.rs`.
4. **Drop `pub mod memory; pub mod experience; pub mod metrics;`** from
   `crates/rlevo-core/src/lib.rs`. Add the three modules as siblings
   under `crates/rlevo-reinforcement-learning/src/lib.rs`.
5. **Remove `ringbuffer`** from `crates/rlevo-core/Cargo.toml`. Verified
   via `rg ringbuffer crates/rlevo-core/src` to be unused — the only
   references were in `memory.rs` (now moved) and even there the
   replay buffer used `VecDeque`, not `RingBuffer`. The dep was orphan.
6. **Fold `rlevo-utils::math::combinations`** into a new
   `crates/rlevo-core/src/util.rs`. Delete `crates/rlevo-utils/`. Drop
   the workspace-member entry in the root `Cargo.toml`. Drop the
   `rlevo-utils` dep + `pub use rlevo_utils as utils;` re-export in the
   umbrella `rlevo` crate.

### Scope decision: conservative dead-code policy

This refactor **does not delete** the speculative trait shells in
`rlevo-core` (`UpdateFunction`, `TransitionDynamics`, `MarkovState`,
`BeliefState`, `HiddenState`, `LatentState`, `StateAggregation`,
`MultiDiscreteAction`, `InvalidActionError`, the `Agent` stub).
Phase 1 exploration counted them as zero-consumer, but the project owner
opted to keep them visible as roadmap markers. Cost: continued cargo-doc
surface implying APIs that are not yet wired up. Benefit: the trait
names stay discoverable in the crate's docs page so the planned RL
roadmap (POMDP belief-state agents, latent world-models, etc.) keeps a
visible TODO. Revisitable if/when the cargo-doc noise becomes a
documentation burden.

## Consequences

**Positive:**

- **Tighter `rlevo-core` boundary.** Post-refactor, `rlevo-core` holds
  the agent–environment contract: `Action`, `Observation`, `State`,
  `Reward`, `Environment`, `Snapshot`, `EpisodeStatus`,
  `TensorConvertible`, `ScalarReward`, `EnvironmentError`, `StateError`,
  `Renderer`, plus the speculative trait shells documented above.
  Replay buffers, training batches, and agent stats now live where they
  are consumed.
- **Cleaner dep graph.**
  ```
  rlevo-core         (env contract, util)
     │
     ├──► rlevo-reinforcement-learning (RL algos + replay/experience/metrics)
     ├──► rlevo-environments (concrete envs)
     ├──► rlevo-benchmarks (paradigm-neutral harness)
     │
  rlevo-evolution    (standalone, ADR 0002)
     │
  rlevo-hybrid      ──► rlevo-core, rlevo-reinforcement-learning, rlevo-evolution
  ```
- **Fewer workspace members.** `rlevo-utils` is gone; the workspace
  goes from eight crates to seven. The umbrella `rlevo` crate's prelude
  is unchanged for end users (the prelude never re-exported
  `rlevo-utils`).
- **Drops one transitive dep from `rlevo-core`.** `ringbuffer` was an
  orphan dep — remove it and the `rlevo-core` dep cone shrinks.
- **History preserved.** `git mv` keeps the per-file blame trail across
  the boundary change for the four moved files.

**Negative / accepted costs:**

- **One mechanical refactor across 14 call sites in `rlevo-rl`.**
  Mostly two-line `use rlevo_core::…` → `use crate::…` swaps in each
  agent module. No semantic edits.
- **Speculative trait shells stay in `rlevo-core`** (per the
  conservative dead-code policy). They will continue to appear in
  cargo-doc as items with no consumers. Acceptable trade for keeping the
  roadmap visible; revisit when documentation noise becomes the
  dominant cost.
- **The integration test that lived in `rlevo-core/tests/` moves to
  `rlevo-reinforcement-learning/tests/`.** It exercises the public surface of two crates,
  which is exactly what an integration test in the consuming crate is
  meant to do. Its module-level docstring is updated to reflect the
  cross-crate scope.

**Neutral:**

- The `Renderer` and `NullRenderer` traits in `rlevo-core::render` stay
  put. They have one downstream consumer (`rlevo-environments`) plus a generic
  shape (no RL semantics). Resurrecting them later if a second consumer
  appears is trivial.
- `rlevo-hybrid` is still an empty stub; this refactor neither helps nor
  hurts it. The hybrid crate already deps on `rlevo-core`, `rlevo-reinforcement-learning`,
  and `rlevo-evolution`, so it picks up the new module locations
  naturally when its first hybrid algorithm lands.

## Alternatives considered

**Move only `memory.rs`; leave `experience.rs` and `metrics.rs` in core.**
Rejected: `experience.rs` is consumed *only* by `memory.rs`; if `memory`
moves and `experience` stays, `rlevo-rl` would import `experience` from
`rlevo-core` for a single internal call site, and `rlevo-core` would
host a module that no other crate imports. `metrics.rs` has the same
shape — `AgentStats` is owned and recorded by every RL agent and
nothing else. Splitting the move would leave a worse partition than
the current state.

**Aggressive deletion of zero-consumer abstractions.** Considered as a
counter-proposal: also delete `UpdateFunction`, `TransitionDynamics`,
`MarkovState`/`BeliefState`/`HiddenState`/`LatentState`/`StateAggregation`,
`HistoryRepresentation`/`SufficientStatistic`, `MultiDiscreteAction`,
`InvalidActionError`, the `Agent` stub. Rejected per project owner —
roadmap visibility wins over cargo-doc minimalism for the alpha stage.
Resurrecting via git is cheap (~5 min per trait) when a real consumer
materializes, so the option remains open as an incremental ADR.

**Collapse `rlevo-reinforcement-learning` and `rlevo-environments`.** Rejected. They have zero runtime coupling, disjoint dep cones (envs pulls
`rapier2d`/`rapier3d`/`nalgebra`; rl pulls `burn` with
`train`/`autodiff`/`tui`/`metrics`), and nontrivial size (10.7K + 24.4K
LOC). Merging would force every consumer to compile both cones. The
boundary is load-bearing in the same way ADR 0001 made envs and
benchmarks load-bearing.

**Promote `combinations()` to its own `rlevo-math` crate.** Rejected:
one function is below the bar for a workspace member. If math utilities
grow to a meaningful surface, lifting them out of `rlevo-core::util`
into a dedicated `rlevo-math` crate is a one-commit change.

## References

- Conversation 2026-04-28 (planning + execution).
- ADR 0001 — `keep-environments-and-benchmarks-separate` — established disjoint
  dep cones over speculative shared vocabulary.
- ADR 0002 — `collapse-evolution-traits-into-rlevo-evolution` — applied
  the same principle on the EA side; this ADR mirrors it on the RL side.
- `crates/rlevo-reinforcement-learning/src/{memory,experience,metrics}.rs` — new homes.
- `crates/rlevo-core/src/util.rs` — new home of `combinations()`.
- `crates/rlevo-core/Cargo.toml` — `ringbuffer` dep removed.
- `Cargo.toml` (workspace root) — `rlevo-utils` removed from `members`.
- `crates/rlevo/Cargo.toml`, `crates/rlevo/src/lib.rs` — `rlevo-utils`
  re-export removed.
- Plan file: `~/.claude/plans/in-this-session-we-vast-alpaca.md`.
