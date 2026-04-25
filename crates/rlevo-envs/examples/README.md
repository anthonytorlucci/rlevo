# rlevo-envs Examples

Standalone examples demonstrating `rlevo-envs` environments. Each example runs independently and serves as a smoke test for the corresponding environment.

## Running Examples

```bash
# From the workspace root
cargo run -p rlevo-envs --example <name>

# Box2D and locomotion examples require feature flags (enabled by default)
cargo run -p rlevo-envs --example bipedal_walker_random --features box2d
cargo run -p rlevo-envs --example swimmer_random --features locomotion
```

## Available Examples

### Bandit

| Example | Description |
|---------|-------------|
| `ten_armed_bandit_training` | ε-greedy training on the ten-armed bandit |

### Classic Control

| Example | Description |
|---------|-------------|
| `cartpole_random` | Random-policy rollout over `CartPole` |
| `cartpole_timelimit` | `CartPole` with `TimeLimit` wrapper — truncation vs. termination |
| `mountain_car_random` | Random-policy rollout over `MountainCar` |
| `mountain_car_continuous_random` | Random-policy rollout over `MountainCarContinuous` |
| `pendulum_random` | Random-policy rollout over `Pendulum` |
| `acrobot_random` | Random-policy rollout over `Acrobot` |

### Grids

| Example | Description |
|---------|-------------|
| `grid_empty_random` | Random-policy rollout over `EmptyEnv` |
| `grid_door_key_scripted` | Scripted rollout of `DoorKeyEnv` with ASCII trace per step |
| `grid_memory_random` | Random-policy rollout over `MemoryEnv` |

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
