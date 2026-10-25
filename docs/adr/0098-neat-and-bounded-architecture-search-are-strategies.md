---
project: rlevo
status: active
type: decision
date: 2026-10-25
tags: [adr, decision, neuroevolution, neat, arch-nas, strategy, evolutionary-loop, fitness, speciation, rlevo-evolution, public-surface, breaking-change]
---

# ADR 0098: NEAT and bounded architecture search are strategies

## Status

**Accepted (2026-10-25).**

- Supersedes ADR [0085](0085-the-generation-loop-is-named-evolutionary-loop.md)
  Decision 6's clause on `NeatStrategy` and `ArchNasStrategy`: "they do not
  implement `Strategy`, and no loop type drives them". Both now implement
  `Strategy`, and `EvolutionaryLoop` drives them. The rest of Decision 6,
  that "harness" names the evaluation harness only, stands.
- Carries out ADR [0091](0091-scoring-genome-and-policy-seams-keep-only-what-generic-code-reads.md)
  Decision 7, which kept `GraphFitnessFn` "until the change that makes
  `NeatStrategy` implement `Strategy`". This is that change:
  `GraphFitnessFn` is deleted, and NEAT is scored through `BatchFitnessFn`.
- Amends ADR [0034](0034-fitness-hygiene-chokepoint-convention.md), which
  lists `NeatStrategy::tell` and `ArchNasStrategy::tell` as inherent
  drivers outside the loop, each its own sanitisation boundary. They are
  now `Strategy::tell` impls. Under the loop the fitness arrives sanitised,
  and each `tell` still sanitises where it pulls the fitness to the host,
  as ADR 0034 Decision 3 asks of every `tell`.

## Context

`rlevo-evolution` ships two neuroevolution searches that are not
weight-only:

- **NEAT** grows network topology and weights from a minimal seed; its
  genome is a host-side graph, `TopologyGenome`.
- **Bounded architecture search** evolves which of a fixed menu of Burn
  `Module` variants wins a task, alongside each variant's weights. Its
  genome, `NasGenome`, is an architecture id per individual beside a
  zero-padded weight tensor.

Both were named `*Strategy` and had inherent `init`, `ask`, `tell` and
`best` in the shape of `Strategy`, but implemented neither `Strategy` nor
`GenerationProbe`. So no loop drove them and the evaluation harness could
not run them. A user wrote the generation loop by hand, and a run reported
none of the `ea/` metrics. NEAT also had a scoring trait of its own,
`GraphFitnessFn`, beside `BatchFitnessFn`.

The modules gave reasons, and neither held:

- **"The `Strategy` tensor-genome contract does not fit"** (NEAT), and
  **"`Strategy`'s `Genome = Tensor<2>` is a homogeneous float contract"**
  (ArchNAS). `Strategy::Genome` is bounded by `Clone + Send` only. Nothing
  in `EvolutionaryLoop` or its `GenerationProbe` impl bounds it to a
  tensor, and `BatchFitnessFn<G>` is generic in the genome.
- **"`NasGenome` deliberately does not derive `Clone`: Burn tensors alias
  their underlying storage."** On Burn 0.22 a tensor clone copies a handle
  to immutable data, and every operation consumes its input and returns a
  new tensor. Every other strategy state in the crate derives `Clone` over
  tensor fields, and `NasState` already copied its tensors handle by
  handle, in a helper that did what a derive does.

To test this, each type was given a `Strategy` impl that only called its
inherent methods. Under `EvolutionaryLoop` each run was bit-identical to
the hand-written loop it replaced: NEAT over 30 generations, every genome
the same, and ArchNAS over 12, every weight the same. Both then ran through
`Evaluator::run_trials` in a `GenerationTrial` with no change to the
harness, and a second suite repeated the first exactly.

One real interaction remained. It is the one ADR 0091 kept
`GraphFitnessFn` for:

- **Offspring are shared by mean fitness.** NEAT allots each species
  offspring in proportion to its members' mean fitness. `allocate_offspring`
  floors a negative mean at zero, and splits the seats evenly when every
  mean floors.
- **Every cost is negative once canonical.** Through the loop, a `Minimize`
  objective arrives negated, so every species floors and selection between
  species stops for the whole run, silently.
- **The loop makes this one word away.** Implementing `Strategy` turns
  that into a one-word declaration on a fitness function (GitHub issue
  1067). Measured with the delegating impl, before the fix below: XOR
  scored as `Minimize` SSE still solved, at generations 31, 42 and 40,
  against 31, 24 and 47 as `Maximize 4 − SSE` (seeds 42, 7, 2026,
  population 150). The defect degrades a run; it does not stop one.

## Decision

### 1. `ArchNasStrategy` implements `Strategy`, with `Genome = NasGenome`

- `NasGenome` and `NasState` derive `Clone`; the copying helper is deleted.
- `init`, `ask`, `tell` and `best` move from the inherent impl into
  `impl Strategy`; the inherent ones are removed, not kept beside it.
  `tell` returns `StrategyMetrics`. `best` returns the champion as a
  one-row `NasGenome`, as a tensor strategy returns a one-row tensor; the
  state keeps it that way instead of an id and a weight row apart.
- `ArchNasFitnessFn` implements `BatchFitnessFn<NasGenome>`; its inherent
  `evaluate` is removed. Its sense comes from a new field,
  `NasBuilderConfig::sense`, since `BatchFitnessFn::sense` has no default:
  the scorers registered with the builder share one direction, and the
  builder emits the parameters and the fitness function together.
- `NasParams::validate` checks the builder's ranges (the mutation rate in
  `[0, 1]`, both standard deviations finite and not negative), which
  `NasBuilderConfig`'s `Validate` impl checked and nothing called. That
  impl is deleted. `EvolutionaryLoop::new` validates the parameters, so a
  `NaN` standard deviation is refused there instead of panicking in
  `init`.

### 2. `NeatStrategy` implements `Strategy`, with `Genome = Vec<TopologyGenome>`

- `init`, `ask`, `tell` and `best` move into `impl Strategy`.
  - `ask` takes the device it does not use.
  - `tell` takes the fitness as a `Tensor<1>`, pulls it to the host,
    sanitises it there as before, and returns `StrategyMetrics`.
  - `best` returns the champion as a one-genome population.
- `NeatParams` implements `Validate`: the population and the input and
  output counts at least one, every probability and `survival_threshold`
  in `[0, 1]`, and the distance coefficients, the compatibility threshold
  and both standard deviations finite and not negative.
- `GraphFitnessFn` is deleted. A NEAT fitness implements
  `BatchFitnessFn<Vec<TopologyGenome>>` and declares its sense.
  `BatchGraphFitness`, the batched scorer, implements it, and its
  constructor takes the sense of its reducer.
- `NeatState` keeps its public fields. Its documentation now says what a
  clone is: the per-run `InnovationRegistry` is shared through an `Arc`,
  and `ask` allocates innovation and node ids in it, so a clone is not an
  independent snapshot of the run. `init` gives each run a registry of its
  own. No caller clones a state today; holding the registry by value is
  left open (see Consequences).

### 3. A generation with negative fitness is shifted before its offspring are shared

`speciate` computes each species' mean from `f - m` instead of `f`:

- `m` is the generation's smallest finite fitness when that is negative,
  and zero otherwise;
- the worst member therefore counts zero;
- a member sanitised to `-∞` stays at `-∞`, and its species' share at zero,
  as before.

With the shift, a generation that is negative everywhere is shared in
proportion again. Translating such a generation in a way that leaves a
member negative leaves the shares unchanged. A generation with no negative
fitness is not shifted, so every run that was valid before is unchanged,
bit for bit. `allocate_offspring` is unchanged.

A shift of every generation, as `neat-python` normalises, was not taken:

- **It changes positive runs.** It would change NEAT's apportionment where
  fitness is already positive. Two singleton species scoring 1 and 2 share
  three seats `[1, 2]` today, and would share them `[0, 3]`. The species
  at the generation's minimum, often a new one that NEAT's speciation
  exists to protect, would get nothing.
- **It needs more than a shift.** `neat-python` pairs its normalisation
  with a minimum species size, a parameter this crate does not have.

That, or a share by rank, is an algorithm decision, and stays open.

### 4. The names stay

Both types keep their names. The `Strategy` suffix is now true.

## Consequences

### Positive

- **The harness runs NEAT and NAS.** `Evaluator::run_trials` runs them like
  any other strategy, through `EvolutionaryLoop`'s `GenerationProbe`, and
  each trial reports the seven `ea/` metrics. No harness code changed.
- **The loop's guarantees reach both.** A `Minimize` objective is negated
  and fitness is sanitised in one place, the loop. Neither search needs a
  hand-written loop or a hand-negated scorer; the bounded-search
  integration test now scores its MSE as a cost.
- **One scoring trait.** `GraphFitnessFn` is gone; every strategy in the
  crate is scored through `BatchFitnessFn`.
- **No trajectory moved.** Run through the loop, four fingerprinted runs
  (NEAT on XOR at populations 64 and 150, NEAT on a connection count, and
  ArchNAS on XOR declared as a cost) are bit-identical to the hand-written
  loops before the change. The comparison covered every generation's best
  fitness, species count and population, and the innovation and node
  counters.

### Negative / costs

Breaking for a user of 0.5.0's neuroevolution API:

- an implementation of `GraphFitnessFn`;
- calls to the four methods, whose signatures are now the trait's:
  - NEAT's `ask` takes a device;
  - `tell` takes a tensor and returns metrics;
  - `best` returns a population;
- `ArchNasFitnessFn::evaluate`;
- a `NasBuilderConfig` literal, which needs `sense`;
- `BatchGraphFitness::new`, which takes a sense;
- `NasBuilderConfig`'s `Validate` impl.

A `NeatState` clone still shares its registry. The `Strategy` docs say a
state is clonable "so the loop can snapshot before a risky step if needed".
No loop does, and `NeatState` says so. Holding the registry by value
changes `NeatState::registry` and the receivers of
`InnovationRegistry::register_*`, so it is a breaking change of its own,
and is not made here.

### Neutral

The fitness function is still given no seed. A fitness that trains a
network per genome, the direction of architecture search, will need one
that a second `begin` reproduces. Neither search needs it yet.

## Alternatives considered

- **Rename both and drop the `Strategy` suffix.** About 49 mentions in 15
  files. Rejected:
  - it removes the overlap in name only;
  - the harness still cannot run either, and neither reports metrics;
  - it would have been chosen on the modules' account of the trait, which
    was wrong.
- **A loop type of their own**, as `CoEvolutionaryLoop` is for two
  populations. Rejected. One generation of either search has the shape of
  a `Strategy` step, so a second loop would only copy the negation and
  sanitisation the first already does.
- **Generalise `Strategy` for structured genomes.** Not needed: the trait
  is already generic in its genome. The first case that would need a
  change to the trait is fitness with several objectives (accuracy against
  size, for a Pareto front), which neither search has.
- **Shift every generation, or share by rank.** An algorithm change; see
  Decision 3.
- **Keep the inherent methods beside the trait impl.** A method call
  resolves to the type's own method before a trait's, so existing callers
  would keep the old signatures without a warning.

## References

- ADR [0034](0034-fitness-hygiene-chokepoint-convention.md): the
  fitness-hygiene chokepoint.
- ADR [0076](0076-trial-seam-splits-machinery-from-trial-shape.md):
  `GenerationTrial`.
- ADR [0085](0085-the-generation-loop-is-named-evolutionary-loop.md):
  `EvolutionaryLoop` and Decision 6.
- ADR [0091](0091-scoring-genome-and-policy-seams-keep-only-what-generic-code-reads.md):
  Decision 7.
- GitHub issue 1067: negative NEAT fitness disables offspring
  apportionment.
- Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks
  through augmenting topologies. *Evolutionary Computation*, 10(2).
- Galván, E. and Mooney, P. (2020). Neuroevolution in deep neural networks:
  current trends and future challenges. arXiv:2006.05415.
