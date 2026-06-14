# Appendix D — Supplementary Material

## Notation

| Symbol | Meaning |
| ------ | ------- |
| \\(\mathcal{S}\\) | State space |
| \\(\mathcal{A}\\) | Action space |
| \\(s_t, a_t, r_t\\) | State, action, reward at timestep \\(t\\) |
| \\(\pi(a \mid s)\\) | Policy: probability of action \\(a\\) in state \\(s\\) |
| \\(V^\pi(s)\\) | State-value function under policy \\(\pi\\) |
| \\(Q^\pi(s, a)\\) | Action-value function under policy \\(\pi\\) |
| \\(\gamma\\) | Discount factor \\(\in [0, 1)\\) |
| \\(G_t\\) | Discounted return from timestep \\(t\\) |
| \\(\theta\\) | Neural network parameters |
| \\(\mathbf{x}\\) | Genome / candidate solution vector |
| \\(f(\mathbf{x})\\) | Objective / fitness function |
| \\(\mathcal{N}(\mu, \sigma^2)\\) | Normal distribution with mean \\(\mu\\), variance \\(\sigma^2\\) |
| \\(\lambda\\) | Population size (number of offspring) |
| \\(\mu\\) | Number of parents selected |
| \\(\sigma\\) | Mutation step size (evolution strategies) |

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

**Backend.** A Burn trait that abstracts over CPU (`ndarray`), GPU (`wgpu`), and
other compute targets. Algorithm code is generic over `B: Backend`.

**Elitism.** Carrying the best \\(k\\) individuals from one generation to the
next unchanged, guaranteeing that the best fitness never regresses.

**Episode.** A single trajectory through an environment from `reset()` to a
terminal `step()`.

**Fitness.** The scalar value assigned to a candidate solution. `rlevo` uses a
minimisation convention: lower is better.

**Genome.** A candidate solution, typically a `Vec<f64>` for real-valued EC or a
neural network weight vector.

**Harness.** `EvolutionaryHarness` — the standard driver that runs the ask/tell
loop with parallel evaluation, observer callbacks, and convergence checks.

**Landscape.** A static objective function implementing `evaluate(&self, x: &[f64]) -> f64`.

**Population.** The set of candidate solutions evaluated in a single generation.

**Seed stream.** `rlevo`'s mechanism for deriving independent, reproducible
sub-streams of randomness from a root seed, one per algorithmic purpose
(e.g., `SeedPurpose::Selection`, `SeedPurpose::Mutation`).

**Strategy.** A `rlevo-evolution` trait that encapsulates the ask/tell loop for
a specific EC algorithm.
