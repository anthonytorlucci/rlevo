# Speciation: Protecting Structural Innovation

The [previous section](42-neat-genome.md) ended on a problem it could only name.
A structural mutation — an add-node or add-connection — almost never helps the
moment it appears: a freshly inserted neuron arrives with untuned weights and
typically *lowers* fitness before it can ever raise it. In a single global
population that genome is out-competed and culled within a generation or two,
long before its new structure has a chance to prove itself. Topology evolution
would stall at the seed.

**Speciation** is NEAT's answer. It partitions the population into clusters of
structurally similar genomes and makes each genome compete for survival *within
its own cluster* rather than against the whole population. A new structure is
then judged against its near-relatives — other genomes carrying the same
innovation — instead of against fully-optimised veterans, buying it the few
generations it needs to tune its weights. This section is about how `rlevo`
realises that: the distance that defines "structurally similar", the per-species
bookkeeping, and the reproduction arithmetic that turns species membership into
offspring counts.

Everything here builds directly on the [innovation numbers](42-neat-genome.md) of
the last section. Speciation is, at bottom, *another* consumer of the same
historical markings: where crossover used them to align genes, speciation uses
them to count how far apart two genomes have drifted.

## Compatibility distance

Two genomes are "the same species" when a scalar **compatibility distance**
between them falls below a threshold. The distance, straight from Stanley and
Miikkulainen (2002) [[Stanley and Miikkulainen, 2002]](../../bibliography.md), is
a weighted sum of three structural disagreements:

```math
\delta = c_1\,\frac{E}{N} + c_2\,\frac{D}{N} + c_3\,\bar{W}
```

where \\(E\\) is the number of **excess** genes (innovations beyond the other
genome's highest), \\(D\\) the number of **disjoint** genes (innovations missing
from one genome but within the other's range), \\(\bar{W}\\) the mean absolute
weight difference over **matching** genes, and \\(N\\) the connection-gene count
of the larger genome. The coefficients tune how much topological difference
(\\(c_1\\), \\(c_2\\)) versus weight difference (\\(c_3\\)) counts;
`NeatParams::default_for` seeds the canonical \\(c_1 = c_2 = 1.0\\),
\\(c_3 = 0.4\\), with a compatibility threshold of \\(3.0\\).

`compatibility_distance` computes this as a single **`O(n)` merge** of the two
innovation-sorted connection lists — exactly the linear scan the
[sorted invariant](42-neat-genome.md) was maintained for. Walking the two lists
in lockstep, an equal-innovation pair is *matching* (accumulate its weight
difference), and a gene present in only one list is classified *excess* if its
innovation exceeds the other genome's maximum and *disjoint* otherwise. The same
`Ordering::Equal/Less/Greater` arms that drove crossover drive the classification
here.

One detail in `N` is easy to miss and load-bearing: when the larger genome has
fewer than 20 genes, `N` is clamped to `1` rather than the gene count. This is
Stanley's own footnote — dividing a one- or two-gene disagreement by a tiny `N`
would inflate the distance and over-fragment small early populations. On a task
like XOR, where genomes stay well under 20 genes for the entire run, `N` is
effectively always `1`, so \\(E\\) and \\(D\\) contribute their raw counts. The
distance tests pin this down: two genomes differing by one excess and two
disjoint genes sit at exactly \\(1\cdot 1 + 1\cdot 2 + 0 = 3.0\\).

## A species and its representative

A species is a cluster plus the bookkeeping that lets it persist across
generations and earn offspring:

```rust
pub struct Species {
    pub id: SpeciesId,
    pub representative: TopologyGenome,   // a CLONED genome, not an index
    pub members: Vec<usize>,              // INDICES into the population Vec
    pub best_fitness: f32,                // best ever seen (maximisation)
    pub last_improved_generation: u64,    // drives stagnation
    pub adjusted_fitness_sum: f32,        // mean raw fitness; drives allocation
}
```

Two representation choices are deliberate. `members` holds **indices** into the
population `Vec`, not genomes, so the population remains the single owner of
genome data and a species is a cheap view over it. The `representative`, by
contrast, is a **cloned** genome rather than an index — because it must survive
the wholesale population turnover between generations, when last generation's
indices no longer mean anything. The representative is the frozen anchor every
genome is compared against during assignment.

## The assignment pass: `speciate`

Each generation re-partitions the population in one pass. Every genome is tested
against the existing species' representatives in order and joins the **first**
species whose representative is within `compat_threshold`; if none matches, the
genome founds a new species with itself as representative. "First match wins",
not "nearest" — the canonical greedy assignment, and cheaper than computing every
distance.

The pass does five things in sequence:

1. **Carry forward** each existing species' representative; clear its membership
   and adjusted-fitness accumulator.
2. **Assign** every genome to the first compatible species, founding a new
   species when none matches.
3. **Drop** any species left empty.
4. **Update** each species' `best_fitness` and `last_improved_generation` (a new
   maximum resets the stagnation clock), and set `adjusted_fitness_sum` to the
   species' **mean raw fitness**.
5. **Re-pick** each survivor's representative at random from its current members,
   seeded — so next generation compares against a fresh, live anchor.

That fourth step is where **fitness sharing** happens, and it is subtler than it
looks. Setting a species' adjusted fitness to the *mean* over its members is
algebraically identical to NEAT's explicit-fitness-sharing formula (divide each
member's raw fitness by the species size, then sum): both yield
\\(\sum_i f_i / |S| = \bar{f}\\). The effect is that a species' claim on the next
generation scales with its *average* quality, not its *total* — so a species
cannot win simply by being large. A lone genome with a brilliant new topology and
a crowded species of mediocre veterans compete on equal per-capita terms.

## From shared fitness to offspring: `allocate_offspring`

Mean-adjusted fitness is only useful if it decides how many children each species
gets. `allocate_offspring` turns the per-species means into exactly `pop_size`
integer offspring counts using the **largest-remainder (Hamilton) method**: give
each species its real-valued proportional share, floor it, then hand the leftover
seats to the species with the largest fractional remainders (ties broken by best
fitness). The counts sum to `pop_size` *exactly* — the
`test_allocate_offspring_sums_to_pop_size` test guards that invariant, and the
reconciliation step reclaims any rare floating-point overshoot from the smallest
remainders.

The degenerate case is handled explicitly: when total adjusted fitness is
non-positive — every genome scored zero, as can happen in the first generations
on a sparse-reward task — proportional allocation is meaningless, so the seats are
split as evenly as possible instead (`10` seats over `3` species → `[4, 3, 3]`).

## Stagnation: pruning the played-out

Sharing fitness protects young species, but it would also keep *dead* ones on
life support forever. `remove_stagnant` culls any species whose `best_fitness`
has not improved for `stagnation_limit` generations (canonical default `15`),
freeing their offspring budget for lineages still making progress. Two guards
keep the cull from going too far:

- The top `STAGNATION_PROTECT_TOP_K` species by best fitness (currently `2`) are
  **shielded unconditionally**. A population-wide plateau — every species
  stagnant at once, common late in a run near a fitness ceiling — would otherwise
  wipe the entire population in a single generation. The shield guarantees the
  best lineages always survive.
- As a final backstop, if the cull would empty the species list entirely, the
  single best species is kept regardless.

`test_remove_stagnant_protects_top_k` exercises exactly this: a high-fitness but
long-stagnant species is protected as a top-K, a recently-improved one survives
on its own merit, and only the low-fitness stagnant species is removed.

## Where it all runs: the `ask`/`tell` split

Speciation and reproduction live on opposite sides of the strategy's
[ask/tell loop](../evolutionary-computation/24-strategy.md), and the split is not
arbitrary:

- **`tell` speciates.** Re-partitioning needs the new population, its freshly
  measured fitness, and the prior species' cloned representatives to all coexist
  consistently — and `tell` is the only point where they do. Member indices stay
  valid against the population just installed, and each species' next
  representative is cloned from a genome that is actually present.
- **`ask` reproduces.** With residents already speciated, producing the next
  generation is pure bookkeeping: prune stagnant species, apportion offspring,
  then fill each species' allotment. A species larger than
  `elitism_min_species_size` (default `5`) copies its **champion** through
  unchanged; the rest of its seats are filled by breeding from the top
  `survival_threshold` fraction (default `0.2`) of its members — intra-species
  crossover, occasional mutation-only — each child then passed through the
  structural and weight mutations of the [genome section](42-neat-genome.md).

So a single generation flows: `ask` breeds offspring from last generation's
species → the harness evaluates them → `tell` installs the scores and re-speciates
for the next round. Speciation is the hinge the whole loop turns on.

## A note on direction

One convention is worth restating because it threads through every formula above.
NEAT **maximises** — higher fitness is better — the same direction as the engine's
canonical-maximise convention, and it is not incidental. Fitness sharing divides
by species size and apportions seats proportionally, both of which assume
**non-negative** fitness; best-fitness tracking and stagnation both treat "higher"
as "better". That non-negativity is a NEAT-specific precondition, orthogonal to
objective sense: a task whose native objective is a cost is reconciled into
canonical (maximise) space by the harness/adapter chokepoint and then shifted
non-negative for the `GraphFitnessFn` seam — you do not hand-negate. This is the
same maximisation stance the [overview](../40-neuroevolution.md) flagged, now
visible in the arithmetic that depends on it.

With the genome, its innovation markings, and the speciation that protects new
structure all in place, the one remaining question is mechanical: how a
`TopologyGenome` actually becomes a network that returns an output. That is the
subject of the [phenotypes section](44-phenotypes.md).

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
