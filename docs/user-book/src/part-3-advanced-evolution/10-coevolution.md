# Coevolution

Every strategy in Part I optimises one population against a *fixed* objective: the
`BatchFitnessFn` is a function of the genome alone, and two different runs of the
same population see the same landscape. **Coevolution** breaks that assumption.
Here fitness is *coupled* between two populations — each individual is scored by
how it performs in combination with, or in contest against, members of the other
population. The landscape one population climbs is shaped by where the other
population currently stands, and it moves as the other population evolves.

That coupling comes in two flavours, and `rlevo` ships both:

- **Cooperative** coevolution decomposes one hard problem into interacting parts.
  Each population evolves a *piece* of the solution; the pieces are assembled and
  scored together. The populations succeed or fail jointly.
- **Competitive** coevolution pits two populations against each other in an arms
  race — predator and prey, host and parasite. One population's gain is the
  other's loss, and the pressure each exerts drives the other to improve.

Both are *advanced* methods: they inherit every guarantee of the Part I engine —
host-sampled RNG, on-device population tensors, the minimise-cost convention — but
add coupling on top, and with it failure modes (cycling, forgetting) the
single-population loop never meets. Everything here lives under
`rlevo::evo::coevolution`.

## The contract: `CoEvolutionaryAlgorithm` and `CoupledFitness`

A coevolutionary run does not fit the `Strategy` ask/tell shape, because the two
populations cannot be proposed, scored, and updated independently — they must be
evaluated *against each other* in a single coupled step. So coevolution has its
own trait, `CoEvolutionaryAlgorithm<B>`, whose central method advances both
populations at once:

```rust
pub trait CoEvolutionaryAlgorithm<B: Backend>: Send + Sync {
    type Params: Clone + Debug + Send + Sync;
    type State: Clone + Debug + Send;

    fn init(&self, params: &Self::Params, rng: &mut dyn Rng, device: &B::Device)
        -> Self::State;

    /// Advance both populations one simultaneous generation.
    fn step(&self, params: &Self::Params, state: Self::State,
            rng: &mut dyn Rng, device: &B::Device) -> (Self::State, CoEAMetrics);

    fn metrics(&self, state: &Self::State) -> CoEAMetrics;
}
```

The single `step` (rather than separate `ask`/`tell`) is the structural tell: both
populations propose, are evaluated together, and consume their results inside one
call. The shared `CoEAState<StA, StB>` holds each inner strategy's state plus
dual best/mean trackers, and `CoEAMetrics` is the two-population analogue of
`StrategyMetrics` — `best_fitness_a`/`best_fitness_b`, the means, and the
hall-of-fame sizes introduced below.

The coupling itself is isolated behind one trait, `CoupledFitness<B>`:

```rust
pub trait CoupledFitness<B: Backend>: Send + Sync {
    /// `populations[i]` is `(pop_size_i, genome_dim_i)`; returns one fitness
    /// vector per population, each of length `pop_size_i`, lower = better.
    fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>>;
}
```

This is where "how population A scores against population B" lives, and nowhere
else. The trait is deliberately **stateless** — it owns no RNG and no generation
counter; the algorithm and harness own all context. It is written for any number
of populations (the argument is a slice), though both shipped algorithms always
pass exactly two. Like the rest of the engine it **minimises**: wire your
objective as a cost.

Driving a run is the job of `CoEvolutionaryHarness<B, C>`, the coevolutionary
sibling of the `EvolutionaryHarness`. It takes the algorithm, its params, a seed,
a device, and a generation budget; `reset` materialises the joint state and
`step(())` runs one coupled generation. The reward it reports back is
`-min(best_a, best_b)` — the *weaker* population is the binding constraint, so
progress is only credited when the lagging side improves — and it implements the
same benchmark-environment interface as the single-population harness, so the
benchmarking tooling drives the two identically.

## Cooperative coevolution

Cooperative coevolution (Potter and De Jong, 1994)
[[Potter and De Jong, 1994]](#bibliography) attacks a high-dimensional problem by
**splitting its variables across populations**. Population A evolves one subset of
the dimensions, population B the complement; neither holds a complete solution.
To score an individual from A, it is combined with a *representative* drawn from B
to assemble a full-dimensional candidate, which is then evaluated. The two
populations coadapt: each is, in effect, optimising its piece against the other's
current best guess at the rest.

`CooperativeCoEA<B, SA, SB, F>` wraps two inner strategies and a `CoupledFitness`,
and `CooperativeCoEAParams` declares the decomposition:

```rust
let algo = CooperativeCoEA::new(strategy_a, strategy_b, fitness);
let params = CooperativeCoEAParams::new(
    params_a, params_b,
    dims_a,        // global indices assigned to population A
    total_dims,    // full problem dimensionality (B gets the complement)
    RepresentativePolicy::Best,
    evals_per_generation,
);
```

`new` validates the split eagerly — `dims_a` must be non-empty, in range,
duplicate-free, and must leave at least one dimension for B. The key tuning choice
is the **representative policy**, which decides *who* from the other population a
candidate is paired against:

- `Best` — pair against the other population's best-so-far. This is canonical CCGA:
  cheap, greedy, and prone to over-committing to a single collaborator.
- `Random` — pair against a uniformly random member, redrawn each generation. More
  robust to deceptive collaboration, at the cost of noisier fitness.
- `Archive { capacity }` — cycle through a bounded ring of past champions, a middle
  ground between the static `Best` and fully random sampling.

## Competitive coevolution

Competitive coevolution (Hillis, 1990) [[Hillis, 1990]](#bibliography) removes the
decomposition entirely. The two populations are adversaries — Hillis's original
result co-evolved sorting networks against the test cases trying to break them —
and *all* of the coupling lives in the `CoupledFitness`, which scores each
population by how well it does against the current other. Accordingly
`CompetitiveCoEAParams` carries nothing but the two inner strategies' params:

```rust
let algo = CompetitiveCoEA::new(strategy_a, strategy_b, fitness);
let params = CompetitiveCoEAParams { params_a, params_b };
```

Each `step` proposes both populations, runs a single coupled evaluation, and lets
each strategy consume its relative fitness. The arms race is emergent: as one side
gets better, the fitness landscape it presents to the other steepens, and the
pressure ratchets up on both.

## Hall of Fame: defeating forgetting

Competitive coevolution has a notorious failure mode. Because each population is
scored only against the *current* opponent, the two can fall into **cycling** —
chasing each other round a rock-paper-scissors loop, each generation beating the
last opponent while quietly losing the competence to beat an older one. Measured
against the moving opponent, fitness looks healthy; measured against any fixed
yardstick, the population is going in circles and *forgetting* what it once solved.

The classic remedy is a **hall of fame** (Rosin and Belew, 1997)
[[Rosin and Belew, 1997]](#bibliography): keep an archive of past champions and
require the current population to perform well against *them* as well as against
the live opponent, anchoring the landscape so old competencies cannot silently
decay. `rlevo` packages this as `HallOfFameFitness<B, F>`, a transparent wrapper
around any `CoupledFitness`:

```rust
let fitness = HallOfFameFitness::new(inner, num_pops, pop_size, genome_dim, &device)
    .with_blend_weight(0.3);
```

Because it is itself a `CoupledFitness`, the wrapper drops into either algorithm
with no other change. Each generation it scores the population against both the
live opponent and the other population's archived champions, then blends the two:

```math
f_\text{blended} = (1 - w)\, f_\text{current} + w\, f_\text{hof},
```

with the blend weight \\(w\\) defaulting to `0.3` (setting \\(w = 0\\) disables the
mitigation without removing the wrapper). Archives are bounded — capacity defaults
to \\(\max(10, \text{pop\_size}/5)\\), after Rosin and Belew — and are updated
*after* blending, using each champion's current-generation contest fitness so
archive membership reflects genuine performance rather than past archive bias.
`rlevo`'s regression suite exercises exactly this: in a non-stationary coverage
game, a solver without the hall of fame drops coverage of targets the current
opponent stops probing, while the hall-of-fame solver retains broad coverage by
being held accountable to past regimes.

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
