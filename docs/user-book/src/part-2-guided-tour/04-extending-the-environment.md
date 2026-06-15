# Bring Your Own Environment

Every algorithm in Part II ran against a built-in environment. This section shows
how to implement `Environment` for your own domain so that any `rlevo` algorithm
can use it.

> This section is a stub. Full content will be added in a future release.
> The canonical reference implementation is
> `crates/rlevo-environments/src/classic/k_armed_bandit.rs` — it includes
> complete trait implementations and extensive tests illustrating the expected
> contract.

## What you need to implement

The `Environment` trait requires three associated types and two methods:

```rust,no_run
impl Environment<STATE_DIM, STATE_DIM, ACTION_DIM> for MyEnv {
    type StateType       = MyState;
    type ObservationType = MyObservation;
    type ActionType      = MyAction;
    type RewardType      = ScalarReward;
    type SnapshotType    = SnapshotBase<STATE_DIM, MyObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> { ... }
    fn step(&mut self, action: Self::ActionType)
        -> Result<Self::SnapshotType, EnvironmentError> { ... }
}
```

The const generics `STATE_DIM` and `ACTION_DIM` are the tensor ranks of your
observation and action spaces. For CartPole both are `1` (flat vectors). For a
2D image observation you would use `3` (height × width × channels).

## The contract your implementation must satisfy

- `reset()` initialises the episode and returns the first observation. It must
  always succeed or return an `EnvironmentError`.
- `step(action)` advances the simulation by one timestep. It must mark the
  snapshot as terminal when the episode ends. After a terminal step, the caller
  will call `reset()` before the next `step()`.
- Neither method should panic on valid input. Use `Result` for recoverable errors
  and document invariants that callers must uphold.

## Testing your environment

Follow the pattern in `k_armed_bandit.rs`:

1. Test `reset()` returns a non-terminal snapshot with a valid observation.
2. Test `step()` with a known action and verify the observation and reward.
3. Test that a terminal condition is reached and flagged correctly.
4. Test `InvalidAction` is returned for out-of-bounds actions.

Once your environment passes these tests, every `rlevo` algorithm that takes an
`Environment` will work with it — including the evolutionary wrappers and the DQN
agent from the previous section.

---

*Co-Authored-By: Anthropic Claude Opus 4.8*
*Reviewed-By: (Human) Anthony Torlucci*
