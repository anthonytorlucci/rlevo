# Part I — Foundations: The "How it Works" Bit

Welcome to the engine room.

Before you start spawning populations of neural networks or tuning PPO
hyperparameters, it helps to know how `rlevo` thinks about the world. If you've
used Python RL libraries, most of this will feel familiar — we just make a few
different choices, mostly because Rust's type system and Burn's backend
flexibility let us.

At its core, `rlevo` is a marriage of two philosophies of "getting better at
something."

### 1. The Gradient Path (Deep RL)

Algorithms like **PPO**, **SAC**, and **DQN** take a single agent (or a small
batch) and use calculus to nudge network weights toward higher reward. It's fast
and sample-efficient — and, as every RL practitioner learns the hard way, it's
easy to settle into a local optimum: a little hill that feels like the summit
right up until you find the real one.

**In `rlevo`, this looks like:**
- **Policies & value functions:** neural networks built with Burn.
- **Experience replay:** buffers that store transitions so the agent can learn
  from its past instead of forgetting it.
- **The gradient loop:** standard backpropagation guided by the reward signal.

### 2. The Population Path (Evolutionary Computation)

Evolution takes the opposite bet. Instead of staking everything on one candidate
and one trajectory, it keeps a whole *population* of candidate solutions and lets
selection, recombination, and mutation do the searching. No derivatives, trivial
to parallelise, and hard to trap: while a gradient method carefully walks down a
single hill, a population is probing thousands at once. That is exactly what you
want on rugged, non-differentiable, or discrete landscapes — the ones where you
don't yet know what a good solution even looks like.

**In `rlevo`, this looks like:**
- **The population:** a collection of candidates, often packed into one large
  tensor for raw throughput.
- **Fitness:** usually the *episodic return* (the total score over an episode)
  rather than a step-by-step reward.
- **Genetic operators:** crossover and mutation that shuffle genes around
  without ever computing a derivative.

### 3. The Hybrid Space: Where It Gets Interesting

The point of `rlevo` is that you don't have to choose. Hybrid strategies let a
population explore the map while gradients climb the most promising peak — use
evolution to find the right mountain, gradients to reach its top.
[Why Combine Them?](40-why-combine.md) makes the case in detail.

### 4. The Rust Layer (The Safety Net)

You'll see a lot of `State<R>`, `Observation<R>`, and `Action<AR>` in this guide.
The `R` and `AR` are **const generics** — compile-time integers that record the
*rank* (the number of axes) of each space: `1` for a flat vector, `2` for a
matrix, `3` for a height × width × channels image.

In most RL libraries you discover your observation tensor is the wrong shape when
a training run panics three hours in. The usual workaround — flatten everything
into one long vector — throws away the structure of the problem along the way. By
encoding the rank in the type, `rlevo` turns a class of runtime shape mismatches
into compile errors: if a layer expects a rank-3 image and the environment hands
back a rank-1 vector, the code simply does not build.

It can feel like arguing with the type system at first. The payoff is that once
it compiles, the plumbing is correct — the bugs you have left are about your
algorithm, not your wiring.

> **Aside: PettingZoo chess.** The
> [chess environment](https://pettingzoo.farama.org/environments/classic/chess/)
> in PettingZoo has an `(8, 8, 111)` observation — a stack of planes encoding the
> board and its recent history. Flatten that to a 7,104-vector and you've thrown
> away exactly the spatial structure a convolutional policy was built to exploit.

---

Where an algorithm or derivation deserves more than a summary, a callout box points to the relevant appendix.

**You do not have to read this part before Part II.** If you learn better by doing, start with the tour and come back here when a term needs grounding. The cross-references work in both directions.

## What is covered

| Section | Core idea |
| ------- | --------- |
| [What Is Optimization?](10-optimization.md) | Fitness landscapes, the exploitation–exploration trade-off, and why gradient descent is not always the answer |
| [Evolutionary Computation](20-evolutionary-computation.md) | Populations, selection, variation operators, and the family of algorithms that descend from Holland's genetic algorithm |
| [Reinforcement Learning](30-reinforcement-learning.md) | The agent–environment loop, Markov decision processes, value functions, and the road from Q-learning to deep RL |
| [Why Combine Them?](40-why-combine.md) | Neuroevolution, evolutionary RL, and what happens when you let evolution drive gradient-based agents |

---

*Co-Authored-By: Anthropic Claude Opus 4.8*
*Reviewed-By: (Human) Anthony Torlucci*
