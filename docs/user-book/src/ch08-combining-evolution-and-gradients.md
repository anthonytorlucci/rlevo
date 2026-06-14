# Combining evolution and gradients

> **Status:** stub — API-mapping pass over `rlevo-hybrid` required before
> prose can be written. This chapter may be the most in flux.

**The problem.** Evolution explores broadly but slowly; gradients exploit sharply
but get stuck. Real research problems want both.

**Learning goal.** The `rlevo-hybrid` boundary — how a population of agents can
be refined with gradient steps between generations.

## Conceptual seam

The hybrid sits at the intersection of Chapters 4 and 5: use `WeightOnly`
neuroevolution (Ch4) to maintain a population of networks, then apply a gradient
step as the local search inside a `MemeticWrapper` (Ch5). The gradient *is* the
local searcher.

## Outline

1. The `rlevo-hybrid` crate boundary — what it owns, what it delegates.
2. Wiring an RL fine-tuning step as a `LocalSearch` implementor.
3. Running the combined loop on CartPole — evolutionary exploration +
   gradient exploitation.
4. Reading the convergence curve — faster than pure evolution, more robust than
   pure RL.
5. Make it yours — control the refinement budget (how many gradient steps per
   generation).

## Example

```bash
cargo run -p rlevo-examples --example ch08_hybrid
```

<!-- TODO: {{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch08_hybrid.rs}} -->
