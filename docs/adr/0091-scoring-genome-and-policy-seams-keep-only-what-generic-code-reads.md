---
project: rlevo
status: active
type: decision
date: 2026-10-22
tags: [adr, decision, architecture, rlevo-evolution, rlevo-hybrid, rlevo-core, fitness, genome, policy, trait-pruning, breaking-change]
---

# ADR 0091: Scoring, genome and policy seams keep only what generic code reads

## Status

**Accepted (2026-10-22).**

The second half of the trait pruning ADR 0090 began, under the same test
(ADR 0090 Decision 8): a public trait stays when generic code reads it, or
when it is the seam an audience implements.

Supersedes, in full, two ADRs whose subject is deleted:

- ADR [0022](0022-tensorgenome-gat-population-storage.md): the
  `TensorGenome` GAT that tied `Population` storage to a genome kind.
- ADR [0030](0030-permutation-tensorgenome-and-population-nonempty-invariant.md):
  `Permutation` as a `TensorGenome`, and `Population`'s non-empty invariant.

Supersedes these clauses. The rest of each ADR stands.

- ADR [0016](0016-memetic-wrapper-and-local-search-seam.md), its
  `&mut dyn FitnessFn<Vec<f32>>` parameter of `LocalSearch::refine` and of
  the budget helper: the parameter is a closure, `&mut dyn FnMut(&[f32]) -> f32`
  (Decision 2 here). The host-side, gradient-free seam and its budget stand.
- ADR [0025](0025-stateful-policy-rollout-contract.md):
  - Decision 1's signature: `StatefulPolicy<B, E>` is `StatefulPolicy<Obs, Act>`,
    and `reset` and `act` take no device (Decision 6 here). The contract,
    `&self` with hidden state threaded by the caller, stands.
  - Decision 2, `ReactivePolicy` and its blanket impl: a memoryless policy
    implements `StatefulPolicy` with `Hidden = ()`.
  - Decision 4's half "gains a device": `RolloutFitness::new` takes no device.
- ADR [0069](0069-sanitized-fitness-is-reduced-in-f64.md) Decision 4, which
  is about `shaping::z_score`: the module is deleted (Decision 4 here). The
  rule the decision illustrated, that a device reduction bounds its terms
  because it cannot widen its accumulator, stands.

Amends, without superseding:

- ADR [0004](0004-move-bench-traits-into-rlevo-core.md)'s inventory of the
  trait surface moved into `rlevo-core`: `FitnessEvaluable` is no longer in it
  (Decision 1 here).
- ADR [0023](0023-objective-sense-and-maximize-convention.md)'s list of the
  cost chokepoint: `FromFitnessEvaluable` is no longer in it, since it no
  longer exists; `FromLandscape` and the loop remain.
- ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md)
  left open "whether the markers stay" for the genome kinds. They go
  (Decision 3 here).

## Context

ADR 0090 pruned the `Environment` seam before the `cargo generate` templates
stamp it. The same audit found the scoring, genome and rollout-policy seams
of the algorithm side in the same state: traits with one implementor, or
none, and nothing generic reading them. Read on `main` at `90e0ffb`, with
`git grep -w` for each name and the compiler for each removal:

- **`FitnessEvaluable`** (`rlevo-core`): two associated types, `Individual`
  and a separate `Landscape` marker, and `evaluate(&self, &Individual,
  &Landscape) -> f64`. 26 impl lines, none in library code: 16 fixtures in
  `#[cfg(test)]` modules, one doc example, eight in tests and one in an
  example. Its one consumer, `FromFitnessEvaluable`, required
  `Individual = Vec<f64>`, and its `BatchFitnessFn` impl was line for line
  `FromLandscape`'s but for one call: `evaluator.evaluate(&x, &landscape)`
  against `landscape.evaluate(&x)`. Both defaulted the sense to `Minimize`.
- **`FitnessFn<G>`** (`rlevo-evolution`): every use was
  `&mut dyn FitnessFn<Vec<f32>>`, the fitness parameter of local search. One
  library implementor, memetic's private `RowFitness`; the rest were test
  fixtures.
- **`Population<B, K>`**: a tensor plus two sizes copied from
  `tensor.dims()`, with constructors that rejected a zero dimension. Outside
  its module only the umbrella prelude named it. The genome markers
  (`GenomeKind`, `TensorGenome`, `Real`, `Binary`, `Integer`, `Tree`,
  `Permutation`) had no use outside `genome.rs` and `population.rs`: every
  `Strategy::Genome` in the workspace is a bare tensor.
- **`shaping`** (`z_score`, `centered_rank`, `ShapingError`, 600 lines): no
  caller.
- **Five traits with one implementor each**: `ParamReshaper`
  (`ModuleReshaper`), `GenotypePhenotypeMap` (`GepDecoder`), `Phenotype`
  (`InterpretedPhenotype`), `PhenotypeBuilder` (`InterpretedBuilder`) and
  `BatchPhenotypeEvaluator` (`DensePaddedEvaluator`). `ModuleEvalFn` was
  written generic over `R: ParamReshaper<B>` and instantiated with
  `ModuleReshaper` at every use.
- **`ReactivePolicy`**: no library implementor; its one consumer was the
  blanket impl that turned it into a `StatefulPolicy`.

Two findings shaped the decisions below:

- `StatefulPolicy`'s `reset` and `act` took `device: &B::Device` besides the
  observation and action, so a trait over `<Obs, Act>` alone has to say where
  a device comes from.
- Merging NEAT's `GraphFitnessFn` into `BatchFitnessFn` was examined and
  rejected for now. `BatchFitnessFn` declares a required `sense()`, which the
  `EvolutionaryLoop` reads at its chokepoint, and NEAT is not driven by the
  loop. A NEAT fitness declaring `Minimize` would be either ignored, the
  maximise-by-omission failure ADR 0023 removed, or canonicalised to `-cost`,
  every value then at most 0; NEAT's species apportionment floors each
  species' adjusted fitness at 0 and splits evenly when the total is 0
  (`species.rs`, GitHub issue 1067), so offspring apportionment would stop
  working.

## Decision

### 1. `FitnessEvaluable` folds into `Landscape`

`FitnessEvaluable` and `FromFitnessEvaluable` are deleted. A test or an
example that scores a candidate with its own formula implements
`Landscape` (`evaluate(&self, x: &[f64]) -> f64`) on a type of its own, and
wraps it in `FromLandscape`. `FromLandscape` is the one host adapter.

### 2. Local-search fitness is a closure

`FitnessFn<G>` is deleted. `LocalSearch::refine` and
`refine_with_known_fitness` take `fitness_fn: &mut dyn FnMut(&[f32]) -> f32`,
a slice as `Landscape::evaluate` takes one. The element type stays `f32`.
Memetic's `RowFitness` keeps its canonicalisation (`sense.to_canonical`) as
an inherent method that the wrapper hands to the searcher as a closure.

### 3. `Population` and the genome markers are deleted

`Population<B, K>`, `GenomeKind`, `TensorGenome`, `Real`, `Binary`,
`Integer`, `Tree` and `Permutation` are deleted with their modules. A caller
names its tensor: `let population: Tensor<B, 2> = ...`. A genome kind
returns with the first example that uses one (the travelling-salesman and
graph-colouring examples planned in GitHub issues 71 to 76), in the shape
that example needs.

### 4. `shaping` is deleted

`z_score`, `centered_rank` and `ShapingError` go, and with `ShapingError`
`rlevo-evolution`'s one use of `thiserror`; the dependency goes too, with
`serde`, which nothing used. A shaping transform returns with the first
strategy that ranks fitness that way.

### 5. Single-implementor traits fold into their types

- `ParamReshaper` into `ModuleReshaper`: `flatten` and `unflatten` are
  inherent, with the trait's contract (leaf order, aliasing, panics) on the
  methods. `ModuleEvalFn<B, R, F>` is `ModuleEvalFn<B, M, F>`, holding a
  `ModuleReshaper<B, M>`. `RolloutFitness` loses the `M: Sync` bound that
  only the trait's impl needed.
- `GenotypePhenotypeMap` and the unit struct `GepDecoder` into
  `ExpressionTree::decode(alphabet, genome)`, a constructor on the phenotype
  the decoder produces. An inherent `decode(&self)` on a fieldless struct
  reads no `self`, and a struct kept only as a namespace adds a name and no
  meaning. The `gep::decode` module, which then holds no public item, is
  private.
- `Phenotype` into `InterpretedPhenotype::forward`.
- `PhenotypeBuilder` with `InterpretedBuilder`: the builder's one method
  returned `InterpretedPhenotype::new(genome)`, so both go and callers use
  that constructor.
- `BatchPhenotypeEvaluator` into `DensePaddedEvaluator::evaluate_population`;
  `BatchGraphFitness<B, E>` is `BatchGraphFitness<B>`.

A second implementor brings its trait back: a sparse batched evaluator
(GitHub issue 56) re-adds the batched-evaluation trait, and a `HyperNEAT`
builder the builder trait.

### 6. `StatefulPolicy<Obs, Act>`, with no device

```rust
pub trait StatefulPolicy<Obs, Act> {
    type Hidden;
    fn reset(&self) -> Self::Hidden;
    fn act(&self, hidden: &mut Self::Hidden, obs: &Obs) -> Act;
}
```

A memoryless policy sets `Hidden = ()`; `ReactivePolicy` is deleted. A
policy that builds a tensor takes the device from its own weights
(`self.layer.weight.device()`), where Burn keeps it. This is the shape of the
planned `Agent::act`, and a step that removes the backend parameter from
Burn's types does not touch the trait. `RolloutFitness::new` loses its
`device` argument, which existed only to reach the policy.

### 7. NEAT keeps `GraphFitnessFn`, without the builder

`GraphFitnessFn` stays NEAT's seam, the one an audience implements, with
`evaluate(&self, population, device) -> Vec<f32>`: the builder argument goes
with `PhenotypeBuilder`. Its merge into `BatchFitnessFn` waits for the change
that makes `NeatStrategy` implement `Strategy`, where the loop's chokepoint
applies the sense; GitHub issue 1067 is fixed there.

## Consequences

### Positive

- 2,546 lines removed and 805 added across 61 files before documentation;
  `rlevo-evolution`'s Rust source falls from 42,520 lines to 40,981. Three
  public modules (`genome`, `population`, `shaping`) are gone, and a
  fourth, `gep::decode`, is private.
- One way to score on the host (`FromLandscape`), one way to hand a searcher
  its objective (a closure), one rollout-policy trait.
- The NEAT and GEP phenotypes are built by a constructor on the phenotype
  type, the same shape on both sides.

### Negative / costs

- Breaking for anyone who implemented `FitnessEvaluable`, `FitnessFn`,
  `LocalSearch` (its `refine` signature), `ReactivePolicy`, `StatefulPolicy`
  (its parameters and methods), `GraphFitnessFn` (its `evaluate` signature)
  or any of the five folded traits, or who named `Population`, a genome
  marker, `shaping`, `GepDecoder` or `InterpretedBuilder`. The changelog
  gives each replacement.
- A policy that builds tensors reads its device from a weight instead of
  receiving it.

### Neutral

- `TopologyGenome`, NEAT's graph genome, is untouched.
- `CoupledFitness` and `BatchFitnessFn` are unchanged.

## Alternatives considered

- **Merge `GraphFitnessFn` into `BatchFitnessFn` now.** Rejected for the
  sense and apportionment interaction in the Context.
- **Keep `StatefulPolicy<B, Obs, Act>` with a device until the backend
  parameter leaves Burn's types.** Smaller now, but the trait would change
  twice in two steps.
- **Keep the genome markers for the planned examples.** The rule is that a
  planned feature is no reason for a public item; the examples define the
  shape they need.
- **Keep `GepDecoder` with an associated `decode`, or a free function.** The
  first keeps a type that is only a namespace; the second adds a name. A
  constructor on `ExpressionTree` adds neither.

## References

- ADR 0090 — the environment half of the pruning, and Decision 8's test.
- ADR 0081 — a public item needs an implementor and a consumer.
- ADRs 0016, 0022, 0023, 0025, 0030, 0069 — superseded or amended above.
- GitHub issues 56, 71 to 76 and 1067 — the planned consumers and the defect
  named above.
