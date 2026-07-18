# Appendix D — Supplementary Material

## Tensor conventions in `rlevo`

`rlevo` uses Burn tensors throughout. Dimensions follow the convention:

- Batch dimension first: a population of \\(\lambda\\) genomes of dimension \\(d\\)
  is a tensor of shape \\([\lambda, d]\\).
- Observations are shaped \\([B, \ldots]\\) where \\(B\\) is batch size during
  training and \\(1\\) during inference.
- The const generic rank parameter \\(R\\) in `State<R>`, `Observation<R>`, and
  `Action<AR>` is the *number of dimensions*, not the size — so a flat vector is
  \\(R = 1\\), a matrix is \\(R = 2\\).

## Glossary

**Ask/Tell.** The interface pattern where a `Strategy` proposes candidates
(`ask`) and receives their scores (`tell`), without owning the evaluation step.
See [The Ask/Tell Contract](ask-tell-contract.md) for the full treatment.

**Backend.** A Burn trait that abstracts over CPU (`Flex`), GPU (`wgpu`), and
other compute targets. Algorithm code is generic over `B: Backend`. `rlevo`
enables `wgpu` and `flex`; other Burn backends exist but are not wired up here.

**Elitism.** Carrying the best \\(k\\) individuals from one generation to the
next unchanged, guaranteeing that the best fitness never regresses.

**Episode.** A single trajectory through an environment from `reset()` to a
terminal `step()`.

**Fitness.** The scalar value assigned to a candidate solution. `rlevo`'s engine
is maximise-native (higher is better); a cost objective declares
`ObjectiveSense::Minimize` and the harness reconciles direction at one chokepoint.

**Genome.** A candidate solution. In `rlevo` a population of genomes lives
on-device as an `f32` tensor (`Tensor<B, 2>` of shape `(pop_size, genome_dim)`),
while the host-side `Landscape` interface scores individual points as `&[f64]`.
A genome may encode raw parameters or the flattened weights of a neural network.

**Harness.** `EvolutionaryHarness` — the standard driver that runs the ask/tell
loop with parallel evaluation, observer callbacks, and convergence checks.

**Landscape.** A static objective function implementing `evaluate(&self, x: &[f64]) -> f64`.

**Population.** The set of candidate solutions evaluated in a single generation.

**Seed stream.** `rlevo`'s mechanism for deriving independent, reproducible
sub-streams of randomness from a root seed, one per algorithmic purpose
(e.g., `SeedPurpose::Selection`, `SeedPurpose::Mutation`).

**Strategy.** A `rlevo::evo` trait that encapsulates the ask/tell loop for
a specific EC algorithm.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
