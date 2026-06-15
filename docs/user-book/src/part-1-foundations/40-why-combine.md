# Why Combine Evolution and Reinforcement Learning?

Evolutionary computation and reinforcement learning both search for good
behaviour — but they search differently, and their failure modes are
complementary.

| | Evolutionary Computation | Reinforcement Learning |
|-|--------------------------|------------------------|
| **Search mechanism** | Population + selection | Gradient ascent on reward |
| **Strengths** | Global exploration, non-differentiable objectives, parallelism | Sample efficiency, fine-grained local improvement |
| **Failure modes** | Slow convergence near optima, expensive fitness evaluations | Sensitivity to reward shaping, local optima in policy space, credit assignment |
| **Parallelism** | Natural — evaluate the whole population at once | Harder — on-policy methods need fresh data |

Combining them lets each side compensate for the other's weaknesses.

## Neuroevolution

The oldest combination is **neuroevolution**: using evolutionary algorithms to
optimise the weights (and sometimes the architecture) of neural networks.

David Moriarty and Risto Miikkulainen (1996) showed that GAs could evolve
network weights for control tasks. Stanley and Miikkulainen's **NEAT**
(NeuroEvolution of Augmenting Topologies, 2002) [[Stanley and Miikkulainen, 2002]](#bibliography) went
further: it co-evolved weights *and* network topology using a historical marking
scheme to enable meaningful crossover between networks of different sizes.

Such et al. (2017) [[Such et al., 2017]](#bibliography) at Uber AI Labs demonstrated that a
simple evolution strategy operating directly on network weights — with no
gradients at all — could match DQN and A3C on many Atari games and train orders
of magnitude faster on modern hardware due to trivial parallelism. This result
was a wake-up call: evolution is a serious competitor to gradient-based RL, not
just a niche alternative.

## OpenAI Evolution Strategies

Salimans et al. (2017) [[Salimans et al., 2017]](#bibliography) framed RL as black-box
optimization over policy parameters and applied a variant of natural evolution
strategies. The gradient estimate is:

<!-- todo! the following equation doesn't render properly; maybe a katex issue? -->
\\[
\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}
  \left[ F(\theta + \sigma \epsilon) \right]
  \approx \frac{1}{n\sigma} \sum_{i=1}^n F(\theta + \sigma \epsilon_i) \epsilon_i
\\]

where \\(F(\theta)\\) is the total undiscounted return of a rollout with
parameters \\(\theta\\). This is a gradient estimate that requires only function
evaluations — no backpropagation. With antithetic sampling (evaluating
\\(\epsilon\\) and \\(-\epsilon\\) in pairs) and virtual batch normalisation,
OpenAI ES matched A3C on MuJoCo locomotion tasks using 1000 CPUs with linear
speedup.

The key lesson: at scale, the bandwidth cost of exchanging gradients across
workers can exceed the communication cost of exchanging scalar returns. ES
becomes competitive.

## Evolutionary Reinforcement Learning (ERL)

Khadka and Tumer (2018) [[Khadka and Tumer, 2018]](#bibliography) introduced a hybrid architecture
that runs a population of policy networks alongside a gradient-based RL agent
(DDPG):

1. Each member of the population is evaluated by acting in the environment and
   accumulating total reward.
2. Their episodes are added to the RL agent's **replay buffer**.
3. The RL agent trains on this shared buffer and periodically **injects** its
   learned policy back into the population.

The population provides diversity and off-policy experience; the RL agent
provides fast local refinement. Together, ERL outperformed both pure DDPG and
pure evolution on continuous control benchmarks.

`rlevo` is designed to support exactly this kind of hybrid: the `Strategy` trait
handles the population loop, the RL algorithms handle the gradient loop, and
the shared `Environment` + `ReplayBuffer` seam connects them.

## Memetic Algorithms

A **memetic algorithm** [[Moscato, 1989]](#bibliography) is any evolutionary algorithm
that includes a local search operator applied to individuals after selection.
The name comes from Dawkins' *meme* — a unit of cultural transmission that allows
acquired improvements to propagate.

In practice: after producing an offspring via crossover and mutation, run a local
optimizer (hill climbing, Nelder-Mead, gradient descent) on that offspring for a
few steps before evaluating it. The population benefits from both global
exploration and local refinement within a single generation.

`rlevo`'s `MemeticWrapper` implements this pattern: it wraps any `Strategy` and
applies a configurable `LocalSearch` operator inside `tell`, with a
`WritebackPolicy` that controls how much of the local improvement is inherited by
the next generation (Lamarckian vs Baldwinian learning).

## Gene Expression Programming

John Koza's Genetic Programming evolved tree-structured programs. **Gene
Expression Programming** (GEP) [[Ferreira, 2002]](#bibliography) uses linear
chromosomes that are decoded into expression trees, separating the genotype (a
fixed-length string that is easy to evolve) from the phenotype (the tree that is
evaluated). This avoids the variable-length crossover problems that plague
standard GP.

`rlevo::evo` includes a GEP implementation for symbolic regression tasks —
evolving mathematical expressions that fit data.

## What rlevo Brings

`rlevo` is not the first library to implement any of these algorithms. Its
contribution is a **unified, type-safe Rust substrate** that:

- Expresses both EA search and gradient RL through the same `Environment` /
  `Strategy` / `ask/tell` vocabulary.
- Enforces dimensional correctness at compile time via const generics.
- Derives reproducible randomness from a single root seed through `seed_stream`,
  making results exactly replicable.
- Runs efficiently on CPU (via `ndarray`) and GPU (via `wgpu`) without changing
  algorithm code.

Part II shows what this looks like in practice.

> **Deeper reading.** Stanley and Miikkulainen (2002), *Evolutionary Computation*
> 10(2), 99–127 for NEAT. Such et al. (2017), arXiv:1712.06567 for deep
> neuroevolution at scale. Salimans et al. (2017), arXiv:1703.03864 for OpenAI
> ES. Khadka and Tumer (2018), arXiv:1805.07917 for ERL. Moscato (1989), Caltech
> Tech Report C3P 826 for memetic algorithms. Ferreira (2002), *Complex Systems*
> 13(2) for GEP.

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Co-Authored-By: Anthropic Claude Sonnet 4.6*\
*Reviewed-By: (Human) Anthony Torlucci*
