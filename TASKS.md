# Tasks

- [ ] EXAMPLES.md

## Milestone 1 & 2: evo-opt-algos (specs; **!! complete the research docs first before starting plan !!**)
- [✔] classical-evolutionary-algorithms (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/classical-evolutionary-algorithms -> sessions/2026-04-17-classical-ea-implementation)
- [✔] swarm-intelligence-algorithms (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/swarm-intelligence-algorithms -> projects/burn-evorl/sessions/2026-04-17-swarm-intelligence-implementation)
- [ ] advanced-hybrid-specialized-ea (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/advanced-hybrid-specialized-ea)
  - [ ] memetic-algorithms (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/memetic-algorithms)
  - [ ] estimation-of-distribution-algorithms (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/estimation-of-distribution-algorithms)
  - [ ] co-evolutionary-algorithms (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/co-evolutionary-algorithms)
  - [ ] neuroevolution-weight-only (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/neuroevolution-weight-only)
  - [ ] neuroevolution-neat (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/neuroevolution-neat)
  - [ ] gene-expression-programming (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/gene-expression-programming)
- [ ] multi-objective-evolutionary-algorithms (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/multi-objective-evolutionary-algorithms)
  - [ ] nsga (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/nsga)
  - [ ] moea-d (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/moea-d)
  - [ ] spea (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/spea)
  - [ ] mopso (projects/burn-evorl/specs/2026-04-17-evo-opt-algos/mopso)

## Milestone 3: rl-envs (`projects/burn-evorl/specs/2026-04-17-rl-envs/rl-envs-overview`, `projects/burn-evorl/sessions/2026-04-17-rl-envs-specs`)
- [✔] classic-control (`projects/burn-evorl/specs/2026-04-17-rl-envs/classic-control`)
- [ ] box2d (`projects/burn-evorl/specs/2026-04-17-rl-envs/box2d`)
- [ ] toy-text (`projects/burn-evorl/specs/2026-04-17-rl-envs/toy-text`)
- [ ] mujoco-locomotion (`projects/burn-evorl/specs/2026-04-17-rl-envs/mujoco-locomotion`)

---

- [ ] Check out and consider implementing RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning [arXiv](https://arxiv.org/pdf/1611.02779) user implemented [github](https://github.com/fatcatZF/RL2-Torch)
- [ ] Bayesian inference for RL (?)

## Search with `ripgrep`

```zsh
rg "TODO|FIXME|HACK|todo|fixme|hack"
```

**2026-02-13**

```
crates/evorl-core/src/agent.rs
2:// todo! distinguish between RL agents and EA agents

crates/evorl-core/src/base.rs
49:    fn shape() -> [usize; D]; // todo! why static method here?
149:    // todo! fn from_tensor(tensor: &Tensor<B, D>) -> Result<Self, TensorConversionError>;

crates/evorl-rl/src/algorithms/dqn/dqn_agent.rs
17:// todo! implement From
392://         todo!("perform a single training step.")
418://         todo!()
440://         todo!("Implement the learning for the agent.")
462://         todo!()
467://         todo!()
471://         todo!("Clear the AgentStats")
506://         todo!("Implement soft_update method which is a required method for the trait DqnModel.")
516://         todo!()
569://         todo!()
605://         self.stats.clone() // todo! would it better to return a reference? Would this pass the ownership to the caller?

crates/evorl-core/src/memory.rs
9:// todo! RolloutBuffer for on-policy algorithms)

crates/evorl-core/examples/grid_position.rs
62:        todo!("Implement conversion to tensor")

crates/evorl-envs/src/lib.rs
50:// todo! pub mod benchmarks;
54:    // todo! pub mod mountain_car;

crates/evorl-envs/src/games/chess/moves.rs
86://! todo!
90://! assert!(todo!);  // assert the number of dimensions in indices == 3 i.e., the rank of the action space.

crates/evorl-envs/src/classic/ten_armed_bandit.rs
363:// todo! use evorl_core::dynamics::Reward
```
