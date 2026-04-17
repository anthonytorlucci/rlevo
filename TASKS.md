# Tasks

evo-opt-algos (specs; **!! complete the research docs first before starting plan !!**)
- [✔] classical-evolutionary-algorithms (sessions/2026-04-17-classical-ea-implementation)
- [ ] swarm-intelligence-algorithms
- [ ] advanced-hybrid-specialized-ea
  - [ ] memetic-algorithms
  - [ ] estimation-of-distribution-algorithms
  - [ ] co-evolutionary-algorithms
  - [ ] neuroevolution-weight-only
  - [ ] neuroevolution-neat
  - [ ] gene-expression-programming
- [ ] multi-objective-evolutionary-algorithms
  - [ ] nsga
  - [ ] moea-d
  - [ ] spea
  - [ ] mopso

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
