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

Combining them lets each side compensate for the other's weaknesses. The
groundwork for this chapter is the [Neuroevolution](40-neuroevolution.md)
chapter, which covers evolving networks with evolution *alone* — weights,
architectures, and topologies. This chapter is about running evolution and
gradient-based RL *together*.

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

> **Deeper reading.** Khadka and Tumer (2018), arXiv:1805.07917 for ERL.
> Moscato (1989), Caltech Tech Report C3P 826 for memetic algorithms. For the
> neuroevolution results this chapter builds on — NEAT, deep neuroevolution at
> scale, and OpenAI ES — see the [Neuroevolution](40-neuroevolution.md) chapter.

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
