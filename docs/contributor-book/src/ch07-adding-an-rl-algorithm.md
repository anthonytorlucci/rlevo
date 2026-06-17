# Adding an RL algorithm

> **Status:** stub — prose and `{{#include}}` anchors coming in a follow-up PR.

**Why this exists.** RL algorithms in `rlevo` live in
`rlevo-reinforcement-learning` and must follow the Burn module conventions and
the crate boundary defined by ADR-0003.

**Key source of truth.** [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md) §8, ADR-0003.

## Crate boundary (ADR-0003)

`rlevo-reinforcement-learning` owns: replay buffers (`PrioritizedExperienceReplay`,
`TrainingBatch`), experience storage (`ExperienceTuple`, `History`), and episode
metrics (`AgentStats`, `PerformanceRecord`). These types must **not** live in
`rlevo-core`.

## Burn module conventions

- `#[derive(Module)]` requires the backend generic to be named `B`.
- `#[derive(Config)]` drives `ConfigBuilder` derivation for training configs.
- `AutodiffBackend` is required for the trainable (forward + backward) path;
  the base `Backend` is sufficient for inference.
- `visit` / `map` from `Module` traverse `BatchNorm` running stats — 4·d leaves
  per norm layer. Account for this in parameter counts.

## New algorithm checklist

- [ ] Configuration struct derives `Config` and `Debug`.
- [ ] Agent struct derives `Module` with backend generic named `B`.
- [ ] Training loop accepts `&mut impl Environment` — does not own the env.
- [ ] No process-global RNG — use seeded `StdRng` threaded through the config.
- [ ] Module lives under `crates/rlevo-reinforcement-learning/src/algorithms/<name>/`.
- [ ] Integration test in `crates/rlevo-reinforcement-learning/tests/` or
      `crates/rlevo/tests/`.

## Outline

1. Anatomy of a DQN implementation — config, model, agent.
2. The `AutodiffBackend` bound — where it appears and why.
3. Replay buffer integration — `PrioritizedExperienceReplay` API.
4. Episode metrics — emitting to `AgentStats`.
5. Testing — smoke test pattern from `crates/rlevo/tests/dqn_integration.rs`.
