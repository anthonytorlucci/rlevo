# Adding an RL algorithm

> **Status:** partial — outline item 2 (the `AutodiffBackend` bound) is
> written below; the rest of the prose and `{{#include}}` anchors are still
> coming in a follow-up PR.

**Why this exists.** RL algorithms in `rlevo` live in
`rlevo-reinforcement-learning` and must follow the Burn module conventions and
the crate boundary defined by ADR-0003.

**Key source of truth.** [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md) §8, ADR-0003.

## Crate boundary (ADR-0003)

`rlevo-reinforcement-learning` owns: the replay-strategy seam (`ReplayStrategy`,
`UniformReplay`, `PrioritizedReplay`; ADR 0050), experience storage
(`ExperienceTuple`, `History`), and episode metrics (`AgentStats`,
`PerformanceRecord`). These types must **not** live in `rlevo-core`.

## Burn module conventions

- `#[derive(Module)]` requires the backend generic to be named `B`.
- `#[derive(Config)]` drives `ConfigBuilder` derivation for training configs.
- `AutodiffBackend` is required for the trainable (forward + backward) path;
  the base `Backend` is sufficient for inference.
- `visit` / `map` from `Module` traverse `BatchNorm` running stats — 4·d leaves
  per norm layer. Account for this in parameter counts.

## The `AutodiffBackend` bound — where it appears and why

`AutodiffBackend` is not a marker for "this code needs gradients today" — it
is the type-level fact that `B` carries a gradient tape at all.
`Autodiff<B>` and `B` are different types, not the same type in two runtime
modes, so the bound has to appear wherever a value crosses from "may be
trained" to "definitely will be trained":

- **Agent structs** (`DqnAgent<B, ...>`, `SacAgent<B, ...>`, …) bound their
  online network's backend generic `B: AutodiffBackend` so `backward()` and
  the optimizer step compile against it.
- **Target networks** run on `B::InnerBackend`, reached via `.valid()` /
  `.inner()` on the online module or its parameters — see
  `crates/rlevo-reinforcement-learning/src/algorithms/td3/target_smoothing.rs`
  for a worked example. Code that only ever sees `B::InnerBackend` cannot
  accidentally backpropagate through it; there is no gradient tape on that
  type to backpropagate through.
- **Inference-only call sites** (`act()` outside training, evaluation loops,
  benchmarks) can bound the plain `Backend` instead of `AutodiffBackend` —
  narrowing the bound is not an optimization, it is documentation: the
  function signature itself states "this cannot produce gradients," which a
  reviewer or the compiler can check without reading the function body.

This is a different mental model from PyTorch's `no_grad()` context manager —
gradients are unrepresentable at the type level here, not suppressed at
runtime. See the
[glossary's "Burn's type-level 'non-autodiff' vs PyTorch's `no_grad`"
section](ch13-glossary.md#burns-type-level-non-autodiff-vs-pytorchs-no_grad)
for the full comparison; don't re-derive it per call site — point new
target-network / inference code at that section instead.

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
2. ~~The `AutodiffBackend` bound — where it appears and why.~~ Written above.
3. Replay buffer integration — the `ReplayStrategy` seam (ADR 0050). Store an
   agent-owned buffer, `UniformReplay` by default; hold an
   `Option<PrioritizedReplaySettings>` config field and select `PrioritizedReplay`
   via `ReplayKind` only when it is `Some`, exactly as DQN/C51/QR-DQN do.
4. Episode metrics — emitting to `AgentStats`.
5. Testing — smoke test pattern from `crates/rlevo/tests/dqn_integration.rs`.
