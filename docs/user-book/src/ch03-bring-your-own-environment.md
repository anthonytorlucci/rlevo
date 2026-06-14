# Bring your own environment

> **Status:** stub — prose and tested include coming in a follow-up PR.

**The payoff.** Everything so far used built-in environments. Here you wrap
**your own** problem so every `Strategy` and agent in the book applies to it.

**Learning goal.** Implement `Environment` end to end:

1. `State<SR>`, `Observation<R>`, `Action<AR>`, and a `Snapshot` (usually
   `SnapshotBase`).
2. `Environment<R, SR, AR>`: `reset`, `step`.
3. `ConstructableEnv::new(render)`.
4. `TensorConvertible<R, B>` for the observation (needed for the neural paths
   in Chapters 4–8).

## Const-generic guardrails

`R`, `SR`, and `AR` must line up across `State`, `Observation`, `Snapshot`, and
`Action`; `shape()` lengths must match. Three-point checklist from the CLAUDE.md:

1. The const generic parameters match across all trait bounds.
2. `shape()` implementations return arrays of correct length.
3. Tensor conversions preserve dimensionality.

## The new seams

- `State<SR>` / `Observation<R>` / `Action<AR>` / `SnapshotBase`
  (`rlevo-core/src/state.rs`, `action.rs`, `environment.rs`, `base.rs`).
- `Environment<R, SR, AR>`: `reset`, `step` returning `Result<SnapshotType, EnvironmentError>`.
- `ConstructableEnv::new(render: bool)` (ADR-0011).
- `TensorConvertible<R, B>`: `to_tensor`, `from_tensor`.

## Reference implementations to study

- `rlevo-environments::classic::CartPole` — full worked example with
  `TensorConvertible` and discrete actions.
- `rlevo-environments::classic::KArmedBandit` — the minimal environment, heavily
  tested.
- `crates/rlevo-core/examples/grid_position.rs` — State/Action patterns.

## Outline

1. Picking your state and observation representation.
2. Defining actions — discrete vs. continuous.
3. Implementing `reset` and `step`.
4. Wiring `TensorConvertible` so your env can talk to neural strategies.
5. Writing the tests — happy path, invalid actions, terminal states.

## Example

```bash
cargo run -p rlevo-examples --example ch03_custom_env
```

<!-- TODO: {{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch03_custom_env.rs}} -->
