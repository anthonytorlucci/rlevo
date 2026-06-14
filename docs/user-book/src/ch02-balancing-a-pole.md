# Balancing a pole: gradient RL

> **Status:** stub — prose and tested include coming in a follow-up PR.

**The problem.** A cart must keep a pole upright by pushing left or right. The
canonical control task — recognisable, low-dimensional, solvable in seconds.

**Learning goal.** A real `Environment` (state → action → observation + reward),
the `TensorConvertible` bridge to Burn tensors, and training a **Deep Q-Network**
agent. This is the "other half" of the library from Chapter 1: gradients, not
populations.

## The new seams

- `Environment<R, SR, AR>` with associated `StateType`, `ObservationType`,
  `ActionType`, `RewardType`, `SnapshotType` and methods `reset` / `step`
  (`rlevo-core/src/environment.rs`).
- `ConstructableEnv::new(render: bool)` factory trait (ADR-0011).
- `TensorConvertible<R, B>` — `to_tensor` / `from_tensor`
  (`rlevo-core/src/base.rs`).
- `DqnAgent` + `DqnTrainingConfig` + the `DqnModel<B, DB>` trait
  (`rlevo-reinforcement-learning/src/algorithms/dqn/`).

## Outline

1. Introducing CartPole — obs = 4 floats (cart pos/vel, pole angle/ang-vel);
   actions = `Left` / `Right`.
2. Defining a small `DqnMlp` — three `Linear` layers implementing
   `DqnModel<B, 2>`.
3. Building and running the training loop with `DqnAgent`.
4. Reading the convergence curve — episode return climbing toward 200.
5. Make it yours — `double_q`, episode budget, MLP width.

## Example

```bash
cargo run -p rlevo-examples --example ch02_cartpole_dqn
```

<!-- TODO: {{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch02_cartpole_dqn.rs}} -->
