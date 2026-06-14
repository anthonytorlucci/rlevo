# Letting the architecture evolve

> **Status:** stub — prose and tested include coming in a follow-up PR.

**The problem.** You don't know how *big* the network should be. Let evolution
choose the architecture, not just the weights.

**Learning goal.** Architecture search: evolving over a **bounded set of
architecture variants**, each with its own weights.

## The new seam

`ArchNasBuilder<B>` / `ArchNasStrategy` / `NasBuilderConfig`
(`rlevo_evolution::ArchNasBuilder`). Register variants with
`.add_variant(module, scorer)`; configure with `NasBuilderConfig`:

```rust,no_run
NasBuilderConfig {
    pop_size: 32,
    arch_mutation_rate: 0.1,
    weight_mutation_std: 0.05,
    weight_init_std: 0.1,
    tournament_size: 3,
    elite_count: 2,
}
```

## Outline

1. Registering architecture variants — shallow, medium, deep MLP.
2. Building the harness — same pattern as Chapter 1.
3. Running the search — which architecture wins and why.
4. Inspecting the population's architecture mix across generations.
5. Make it yours — add a fourth variant; change `arch_mutation_rate`.
6. Forward pointer: the open-ended **NEAT** form (topologies that grow
   node-by-node) is the next step beyond bounded search.

## Example

```bash
cargo run -p rlevo-examples --example ch07_arch_search
```

<!-- TODO: {{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch07_arch_search.rs}} -->
