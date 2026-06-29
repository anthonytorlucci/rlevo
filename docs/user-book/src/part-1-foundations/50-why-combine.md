# Why Combine Evolution and Reinforcement Learning?

This is the chapter the rest of Part I has been building toward. We have a
population that searches by selection and explores globally, and a gradient agent
that learns from the fine structure of a trajectory and refines locally. The
question now is the one that gives `rlevo` its reason to exist: why run them
*together*?

The short answer is that their strengths and weaknesses line up almost exactly
back-to-front. Where one is strong the other is weak, so each can shore up the
other's failure mode rather than merely averaging with it:

| | Evolutionary Computation | Reinforcement Learning |
|-|--------------------------|------------------------|
| **Search mechanism** | Population + selection | Gradient ascent on reward |
| **Strengths** | Global exploration, non-differentiable objectives, parallelism | Sample efficiency, fine-grained local improvement |
| **Failure modes** | Slow convergence near optima, expensive fitness evaluations | Sensitivity to reward shaping, local optima in policy space, credit assignment |
| **Parallelism** | Natural — evaluate the whole population at once | Harder — on-policy methods need fresh data |

Read the failure-mode row as a shopping list of things the *other* column fixes.
Evolution explores broadly but crawls once it is near an optimum — exactly where
RL's gradient is sharpest. RL is sample-efficient but gets stuck in policy-space
local optima and leans hard on reward shaping — exactly the deceptive,
sparse-reward terrain a population shrugs off. Combine them and each side
compensates for the other; the only real question is *how tightly* you couple the
two loops, and the two patterns below sit at different points on that spectrum.

The groundwork is the [Neuroevolution](40-neuroevolution.md) chapter, which
covered evolving networks with evolution *alone* — weights, architectures, and
topologies. This chapter is about letting that population and a gradient learner
share the work.

## Evolutionary Reinforcement Learning (ERL)

The looser coupling keeps two distinct learners running side by side and lets them
trade what each produces cheaply. Khadka and Tumer (2018)
[[Khadka and Tumer, 2018]](#bibliography) introduced exactly this: a population of
policy networks running *alongside* a gradient-based RL agent (DDPG), with a
two-way exchange between them:

1. Each member of the population acts in the environment and accumulates total
   reward — an ordinary evolutionary evaluation.
2. The episodes those rollouts generate are poured into the RL agent's **replay
   buffer**.
3. The RL agent trains on that shared buffer and periodically **injects** its
   learned policy back into the population as a new member.

The elegance is in what each side contributes through the buffer. The population
is a diversity engine — it fills the replay buffer with varied, off-policy
experience that a lone agent exploring on-policy would rarely reach, softening
RL's exploration problem. The RL agent is a refinement engine — it distils that
experience with gradients and hands a sharpened policy back, softening evolution's
slow-near-the-optimum problem. Neither could produce the other's contribution
alone, which is why ERL beat both pure DDPG and pure evolution on continuous
control. `rlevo` is built to support precisely this shape: the `Strategy` trait
runs the population loop, the RL algorithms run the gradient loop, and the shared
`Environment` + `ReplayBuffer` seam is the channel between them.

## Memetic Algorithms

The tighter coupling folds the local learner *inside* the evolutionary loop, so
refinement happens to every individual within a single generation rather than
through a periodic exchange. A **memetic algorithm**
[[Moscato, 1989]](#bibliography) is any evolutionary algorithm that applies a
local-search operator to individuals after selection — the name nods to Dawkins'
*meme*, a unit of cultural transmission, because here an individual's *acquired*
improvements can propagate, not just its inherited genes.

In practice: after crossover and mutation produce an offspring, run a local
optimiser — hill climbing, Nelder–Mead, gradient descent — on it for a few steps
before you evaluate it. Global exploration and local refinement now happen within
the *same* generation instead of being divided between two cooperating loops.
`rlevo`'s `MemeticWrapper` implements the pattern: it wraps any `Strategy` and
applies a configurable `LocalSearch` operator inside `tell`, governed by a
`WritebackPolicy` that decides how much of the local improvement the next
generation inherits. That policy is the crux, and it encodes a genuine question
from biology — does an individual's lifetime learning rewrite its genes
(*Lamarckian*: write the improved solution back into the genome) or merely bias
which individuals survive to reproduce (*Baldwinian*: keep the improved *score*
but leave the genome untouched)?

## What `rlevo` Brings

`rlevo` is not the first library to implement any one of these algorithms. Its
contribution is the **unified, type-safe Rust substrate** that lets them
interlock without friction — the reason ERL and memetic search are a few seams
apart rather than separate codebases:

- Both EA search and gradient RL speak the same `Environment` / `Strategy` /
  `ask`/`tell` vocabulary, so wiring one into the other is composition, not
  translation.
- Const generics enforce dimensional correctness at compile time, so a
  population and an agent sharing a network cannot disagree about its shape three
  hours into a run.
- Reproducible randomness derives from a single root seed through `seed_stream`,
  making a hybrid run — with all its interacting stochastic parts — exactly
  replicable.
- The same algorithm code runs on CPU (via `ndarray`) and GPU (via `wgpu`)
  without change.

Part II puts all of this to work, and shows what the combination looks like in
practice.

> **Deeper reading.** Khadka and Tumer (2018), arXiv:1805.07917 for ERL.
> Moscato (1989), Caltech Tech Report C3P 826 for memetic algorithms. For the
> neuroevolution results this chapter builds on — NEAT, deep neuroevolution at
> scale, and OpenAI ES — see the [Neuroevolution](40-neuroevolution.md) chapter.

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
