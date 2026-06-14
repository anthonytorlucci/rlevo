# Evolving a network's weights

> **Status:** stub — prose and tested include coming in a follow-up PR.

**The problem.** Same pole as Chapter 2, but the gradients are gone — imagine a
reward you can't differentiate (a black-box simulator, a human rating, a
discontinuous score). Now you must *search* the weights instead of *descending*
them.

**Learning goal.** Neuroevolution as a **drop-in swap**: keep the network, keep
the environment, replace the optimiser with a `Strategy`. This is the chapter
where Chapters 1 and 2 visibly fuse.

## The new seam

`WeightOnly<B, S, M>` — `rlevo_evolution::algorithms::neuroevolution::WeightOnly`.
Wraps an inner `Strategy<B, Genome = Tensor<B, 2>>` and a **template** Burn
`Module`; flattens the module's parameters into a genome and back via an owned
`ModuleReshaper`. Key methods: `WeightOnly::new(inner, template)`,
`.num_params()`, `.reshaper()`.

## Story beat

Reuse the GA from Chapter 1 *unchanged* as the `inner` strategy — the same
`GeneticAlgorithm<B>` now evolves an MLP's weights. The payoff line: "the
searcher didn't change; only what it searches over did."

## Outline

1. Building the template MLP (same architecture as Ch2).
2. Wrapping it in `WeightOnly` with the GA as inner strategy.
3. The fitness function — roll out one episode, return episode return.
4. Running and comparing convergence to the DQN from Ch2.
5. Make it yours — swap the inner `Strategy` (GA → ES) without touching the
   network or environment.

## Example

```bash
cargo run -p rlevo-examples --example ch04_evolve_weights
```

<!-- TODO: {{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch04_evolve_weights.rs}} -->
