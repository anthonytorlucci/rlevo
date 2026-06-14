# Where rlevo Stands Today

## What is implemented

### Evolutionary algorithms (`rlevo::evo`)

| Algorithm | Status | Notes |
| --------- | ------ | ----- |
| Genetic Algorithm (GA) | Stable | Tournament, BLX-α, SBX, Gaussian mutation, elitism |
| Evolution Strategy (ES) | Stable | (1+1)-ES, (μ+λ)-ES, self-adaptive σ |
| CMA-ES | Planned | Tracked in issue backlog |
| PBIL | Stable | Univariate Bernoulli EDA |
| UMDA | Stable | Univariate Gaussian EDA |
| cGA | Stable | Compact Genetic Algorithm |
| MIMIC | Stable | Pairwise dependency chain EDA |
| BOA | Stable | Bayesian network EDA; Concatenated Trap benchmark |
| Memetic wrapper | Stable | Hill climbing, Nelder-Mead, SA, random restart; Lamarckian/Baldwinian write-back |
| Gene Expression Programming | Stable | Symbolic regression |

### Reinforcement learning (`rlevo::rl`)

| Algorithm | Status | Notes |
| --------- | ------ | ----- |
| DQN | In progress | Discrete action spaces; experience replay, target network |
| Double DQN | Planned | |
| PPO | Planned | |
| SAC | Planned | |

### Environments (`rlevo::envs`)

| Family | Environments |
| ------ | ------------ |
| Classic control | CartPole, Acrobot, MountainCar |
| Grid worlds | FrozenLake, GridWorld |
| K-Armed Bandit | Stationary and non-stationary variants |
| Landscapes | Sphere, Rastrigin, Rosenbrock, Ackley, Concatenated Trap |

## Known gaps and limitations

**DQN lunar lander.** A DQN agent trained on LunarLander for 150,000 steps
performed worse than random. Root cause: the hyperparameters and training budget
were insufficient; the implementation is believed correct but has not been tuned.
This benchmark is deferred pending a proper tuning budget.

**Multi-objective optimisation.** `rlevo` does not implement NSGA-II or any
Pareto-based selection. Single-objective only.

**Continuous RL.** PPO and SAC (required for MuJoCo-style locomotion tasks) are
not yet implemented. DQN handles discrete actions only.

**Distributed evaluation.** Population evaluation is parallelised locally via
`rayon`. There is no built-in support for distributing evaluation across machines.

**Reproducibility with GPU backends.** The `wgpu` backend uses GPU kernels whose
non-determinism can break exact reproducibility even with a fixed seed. The
`ndarray` backend is fully deterministic.
