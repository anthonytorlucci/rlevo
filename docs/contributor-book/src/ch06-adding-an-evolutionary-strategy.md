# Adding an evolutionary strategy

> **Status:** stub — prose and `{{#include}}` anchors coming in a follow-up PR.

**Why this exists.** The `Strategy` ask/tell contract and host-RNG convention
are the two rules most likely to be broken by a new contributor. This chapter
makes them concrete.

**Key source of truth.** [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md) §8, ADR-0002, project memory
`evolution_host_rng_convention`.

## The `Strategy` trait

```rust,no_run
pub trait Strategy<B: Backend>: Send {
    type Genome;
    type Params;

    fn ask(&mut self, params: &Self::Params, device: &B::Device) -> Vec<Self::Genome>;
    fn tell(&mut self, fitnesses: &[f32]);
}
```

`ask` proposes a population; `tell` updates the strategy from scores. The
harness drives this loop — your implementation must not run the loop itself.

## Host-RNG convention (mandatory)

**Never call `B::seed(...)` or `Tensor::random(...)` inside a `Strategy`
implementation.** Use `seed_stream` to derive a deterministic per-purpose stream:

```rust,no_run
use rlevo_evolution::seed_stream::{SeedPurpose, SeedStream};

let mut rng = SeedStream::new(seed, SeedPurpose::Mutation).into_rng();
```

Reason: process-wide RNG mutations race parallel tests and make runs
non-reproducible. The `SeedPurpose` enum names the streams your algorithm uses;
add a new variant if needed.

## `SeedPurpose` streams

| Purpose | When to use |
|---------|-------------|
| `Mutation` | Gene-level perturbation |
| `Crossover` | Recombination operator |
| `Replacement` | Survivor selection when stochastic |
| `Sampling` | Population initialisation |
| `LocalSearch` | Memetic local-search steps |
| `EdaSampling` | EDA model sampling |

## Outline

1. Implementing `Strategy<B>` — minimal skeleton and common mistakes.
2. `SeedPurpose` streams — how to add a new stream and why.
3. `BatchFitnessFn` — when to implement it vs use `FromLandscape` / `FromFitnessEvaluable`.
4. Plugging into `EvolutionaryHarness` — the harness contract from the strategy's side.
5. Testing — determinism test pattern from `crates/rlevo-evolution/tests/determinism.rs`.
