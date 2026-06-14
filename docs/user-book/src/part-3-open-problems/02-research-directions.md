# Research Directions

`rlevo` is intended as a research substrate, not just a toolkit. The following
are open problems where the library's architecture creates genuine opportunities
for new experiments.

## Evolutionary Reinforcement Learning at scale

Khadka and Tumer (2018) showed ERL outperforming pure RL on MuJoCo benchmarks,
but their experiments used DDPG — an algorithm known to be sensitive to
hyperparameters and reward shaping. Open questions:

- Does ERL's advantage persist with SAC (a more stable off-policy algorithm)?
- How does the population size / gradient-update ratio affect the speed of
  convergence? Is there an optimal exchange rate?
- Can the replay buffer sharing be made asynchronous so the EA and RL agent
  run on different hardware simultaneously?

`rlevo`'s `ask`/`tell` / `Environment` seam is the natural place to implement
this experiment.

## Neuroevolution of policy architectures

NEAT evolves both weights and topology. Modern deep RL uses fixed architectures
and learns only weights. The question is whether co-evolving architecture and
weights — using Burn's dynamic graph to evaluate arbitrary topologies — can find
more sample-efficient policies than tuning a fixed architecture manually.

`rlevo`'s Gene Expression Programming implementation provides one angle on this:
GEP genomes could encode not just numerical weights but connectivity patterns.

## Sparse reward and the exploration problem

CartPole and Sphere are easy because reward is dense — the agent receives a
signal at every step. Many real problems (robotic manipulation, drug discovery,
circuit design) have sparse reward: you succeed or you fail, and most episodes
fail. Standard RL collapses; pure evolution is slow; the combination may help.

Specifically: a population of random policies will occasionally stumble into a
sparse reward by chance. Those policies can be injected into the RL agent's
replay buffer, bootstrapping the value function before gradient updates begin.
This is the ERL intuition applied to exploration.

## Quality-Diversity algorithms

Novelty Search (Lehman & Stanley, 2011) [[LS11]](#bibliography) showed that
searching for *novel* behaviours — without regard to fitness — can find solutions
that pure fitness maximisation cannot, because novelty avoids deceptive local
optima.

Quality-Diversity algorithms (MAP-Elites, AURORA) maintain an archive of diverse,
high-quality solutions rather than converging to a single optimum. This is
valuable for robotics (a repertoire of gaits for different terrains) and for
generating training data for RL (diverse starting states).

`rlevo` does not yet implement novelty search or MAP-Elites. The `Strategy` trait
would need an archive alongside the population to support these — a well-scoped
contribution.

## Non-stationary environments

Most RL benchmarks have fixed dynamics. Real environments change: the wind
changes, the rules change, the opponent adapts. Evolutionary methods maintain
population diversity that may make them more robust to non-stationarity, but this
has not been systematically studied in the evolutionary RL context.

The `k_armed_bandit` environment in `rlevo-environments` includes a non-stationary
variant where reward distributions shift during the episode — a starting point for
this investigation.

## Benchmarking against Python baselines

`rlevo` needs rigorous comparisons against `stable-baselines3` and `DEAP` on
shared benchmarks. This requires:
- matching random seeds across languages (non-trivial),
- shared benchmark definitions (Sphere, Rastrigin, CartPole, LunarLander),
- statistical significance testing over multiple seeds.

This is a contribution that requires no new algorithms — only careful
experimental design and reporting.

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*
