# rlevo-environments Examples

Standalone examples demonstrating `rlevo-environments` environments. Each example runs independently and serves as a smoke test for the corresponding environment.

## Running Examples

```bash
# From the workspace root
cargo run -p rlevo-environments --example <name>

# Box2D and locomotion examples require feature flags (enabled by default)
cargo run -p rlevo-environments --example bipedal_walker_random --features box2d
cargo run -p rlevo-environments --example swimmer_random --features locomotion
```

## Available Examples

### Classic Control

| Example | Description |
|---------|-------------|
| `cartpole_random` | Random-policy rollout over `CartPole` |
| `cartpole_timelimit` | `CartPole` with `TimeLimit` wrapper — truncation vs. termination |
| `mountain_car_continuous_random` | Random-policy rollout over `MountainCarContinuous` |
| `pendulum_random` | Random-policy rollout over `Pendulum` |

### Grids

| Example | Description |
|---------|-------------|
| `grid_door_key_scripted` | Scripted rollout of `DoorKeyEnv` with ASCII trace per step |

> Several former `*_random` examples are now random-vs-DQN **benches** —
> the random policy is the baseline a learned policy is compared against
> (reward/success quality plus per-step throughput). Shared DQN scaffolding
> lives in `benches/support/dqn.rs`. Run with `cargo bench -p rlevo --bench <name>`:
>
> | Bench | Was example | Env |
> |-------|-------------|-----|
> | `grid_empty_dqn` | `grid_empty_random` | `EmptyEnv` |
> | `grid_memory_dqn` | `grid_memory_random` | `MemoryEnv` (memoryless DQN ≈ chance — the point) |
> | `acrobot_dqn` | `acrobot_random` | `Acrobot` |
> | `mountain_car_dqn` | `mountain_car_random` | `MountainCar` |

### Box2D (`--features box2d`)

| Example | Description |
|---------|-------------|
| `bipedal_walker_random` | Random-agent smoke test for `BipedalWalker` |
| `lunar_lander_discrete_random` | Random-agent smoke test for `LunarLander` (discrete) |
| `lunar_lander_continuous_random` | Random-agent smoke test for `LunarLander` (continuous) |
| `car_racing_random` | Random-agent smoke test for `CarRacing` |

### Locomotion (`--features locomotion`, Rapier3D-backed)

| Example | Description |
|---------|-------------|
| `reacher_random` | Random-agent smoke test for `Reacher` |
| `inverted_double_pendulum_random` | Random-agent smoke test for `InvertedDoublePendulum` |
| `swimmer_random` | Random-agent smoke test for `Swimmer` |
