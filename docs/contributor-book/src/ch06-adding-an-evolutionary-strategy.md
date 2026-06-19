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

## Testing operators

Operators are free functions over `&[f32]` and `Tensor<B, 2>`, which makes them
the easiest part of the crate to test directly — no harness, no strategy state.
The examples below are the *actual* in-source tests from
`crates/rlevo-evolution/src/ops/selection.rs`, pulled in by anchor so they cannot
drift from what CI runs. Three patterns recur and are worth copying.

**1. Verify stochastic operators in expectation, with a tolerance band.** A
selection operator is random, so you cannot assert an exact winner. Assert the
*distribution* instead: run many draws and check the win count lands near its
analytic expectation. For a population of size `N` under binary tournaments the
unique best member wins a single draw with probability \\(1 - \left(\frac{N-1}{N}\right)^{2}\\);
for `N = 4` that is \\(7/16 \approx 0.4375\\). The band is deliberately generous so the test
stays green across `rand` version bumps that reshuffle the stream:

```rust
{{#include ../../../crates/rlevo-evolution/src/ops/selection.rs:tournament_expectation_test}}
```

**2. Pin down deterministic operators exactly.** Truncation draws no randomness,
so its test asserts the precise winning indices — no band needed:

```rust
{{#include ../../../crates/rlevo-evolution/src/ops/selection.rs:truncation_ordering_test}}
```

**3. Assert the device-gather shape on the `*_select` wrappers.** The host cores
return `Vec<i32>`; the wrappers gather rows into a tensor. A cheap shape check
confirms `n_winners` and the genome width survive the `Tensor::select`:

```rust
{{#include ../../../crates/rlevo-evolution/src/ops/selection.rs:tournament_select_shape_test}}
```

Test the host core (`*_indices_host`) for *logic* and the wrapper (`*_select`)
for *shape* — splitting the operator into a pure host function and a thin device
gather is what makes the logic testable without a backend in the loop.

## Outline

1. Implementing `Strategy<B>` — minimal skeleton and common mistakes.
2. `SeedPurpose` streams — how to add a new stream and why.
3. `BatchFitnessFn` — when to implement it vs use `FromLandscape` / `FromFitnessEvaluable`.
4. Plugging into `EvolutionaryHarness` — the harness contract from the strategy's side.
5. Whole-run determinism — the seed-reproducibility pattern from `crates/rlevo-evolution/tests/determinism.rs` (complements the operator-level tests above).
