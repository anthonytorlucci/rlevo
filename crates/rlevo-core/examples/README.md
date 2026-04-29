# burn-evorl Examples

Example implementations for the `burn-evorl` library demonstrating core trait patterns.

## Directory Structure

```
examples/
├── README.md (this file)
├── continuous_state_with_constraints.rs
└── grid_agent.rs
```

## Examples

### `continuous_state_with_constraints.rs`

Implements `RobotPose`, a 2D robot state bounded to a 1000 mm × 1000 mm workspace with
orientation clamped to [−180°, 180°]. Demonstrates the builder/validator pattern
(`RobotPose::new` returns `Option`), `State` and `Observation` trait wiring, and
reward-shaping helpers (`distance_to`, `normalize_orientation`).

```bash
cargo run -p rlevo-core --example continuous_state_with_constraints --release
```

### `grid_agent.rs`

Minimal egocentric grid agent mirroring the Minigrid-style environments in `rlevo-environments`.
Shows how to wire `State`, `Observation`, and `TensorConvertible` together, implement
`DiscreteAction` with index round-trips, and compose independent sub-dimensions with
`MultiDiscreteAction`.

```bash
cargo run -p rlevo-core --example grid_agent
```
