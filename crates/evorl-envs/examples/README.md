# evorl-envs Examples

This directory contains standalone examples demonstrating the `evorl-envs` crate's functionality. These examples serve as both learning resources and integration tests for the library.

## Table of Contents

- [Overview](#overview)
- [Running Examples](#running-examples)
- [Available Examples](#available-examples)
- [Example Structure](#example-structure)
- [Writing Your Own Examples](#writing-your-own-examples)
- [Testing Examples](#testing-examples)
- [Best Practices](#best-practices)

## Overview

Examples in this project are designed to be:

- **Self-contained**: Each example can run independently without external setup
- **Educational**: Clear documentation explains algorithms, patterns, and API usage
- **Tested**: Built-in unit and integration tests ensure correctness
- **Production-ready**: Follow best practices from Rust API guidelines and Microsoft Rust Guidelines

Examples demonstrate:
- Environment creation and interaction
- Agent training and evaluation
- Physics simulations and state management
- Deep Q-Learning (DQN) implementation
- Experience replay and neural network training

## Running Examples

### Prerequisites

Ensure you have Rust installed (1.70+) and are in the project root:

```bash
cd burn-evorl
```

### Running All Examples

```bash
# Run with default settings (uses NdArray CPU backend)
cargo run --example cartpole_training

# Run with optimizations (recommended for training)
cargo run --example cartpole_training --release

# Run specific example with output
cargo run --example cartpole_training --release 2>&1 | tee output.log
```

### Running Example Tests

```bash
# Run all tests in examples
cargo test --examples

# Run tests with output
cargo test --examples -- --nocapture

# Run a specific example's tests
cargo test --examples cartpole_training
```

### Benchmarking Examples

```bash
# Profile with release optimizations
cargo run --example cartpole_training --release

# Generate flamegraph for performance analysis
cargo install flamegraph
cargo flamegraph --example cartpole_training
```

## Available Examples

### 1. CartPole Training (`cartpole_training.rs`)

**Purpose**: Train a Deep Q-Network (DQN) agent to balance a pole on a moving cart.

**What It Demonstrates**:
- Environment implementation with physics simulation
- Agent creation and action selection using epsilon-greedy exploration
- Experience replay buffer for storing and sampling transitions
- Q-learning updates and neural network training
- Evaluation metrics and performance tracking

**Key Concepts**:
- **State Space**: `[position, velocity, angle, angular_velocity]` (4D)
- **Action Space**: `[0: push left, 1: push right]` (discrete)
- **Reward**: +1 for each timestep the pole remains balanced
- **Goal**: Maximize episode length (max 500 steps)
- **Success**: Agent should achieve reward > 400 after ~300 episodes

**Algorithm**: Deep Q-Networks (DQN)

The Bellman equation for Q-learning:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) - Q(s,a)]
```

Where:
- `α` = learning rate (controls how much new information overrides old)
- `r` = immediate reward
- `γ` = discount factor (future reward weight)
- `max Q(s',·)` = best Q-value in next state

**Expected Output**:

```
Training DQN Agent on CartPole
==============================
Episodes: 500
Max Steps: 500
Learning Rate: 0.001
Discount Factor: 0.99
Initial Epsilon: 1.0

Episode 50: Reward = 45.0, Avg = 32.5, Epsilon = 0.7788
Episode 100: Reward = 98.0, Avg = 67.3, Epsilon = 0.6065
Episode 150: Reward = 185.0, Avg = 145.2, Epsilon = 0.4724
...
Episode 500: Reward = 495.0, Avg = 475.3, Epsilon = 0.0100

Training Complete!
Final 10 Episode Average: 485.5

Evaluating Agent (10 episodes, no exploration)
==============================================
  Evaluation Episode Reward: 500.0
  Evaluation Episode Reward: 500.0
  ...

Average Evaluation Reward: 498.5

==============================
Training Summary
==============================
Maximum Episode Reward: 500.0
Minimum Episode Reward: 12.0
Average Episode Reward: 285.4
Final Epsilon: 0.0100
```

**Modifying the Example**:

Adjust hyperparameters in `main()`:

```rust
let mut config = DqnConfig::default();
config.num_episodes = 1000;           // More training
config.learning_rate = 0.01;          // Faster learning
config.discount_factor = 0.95;        // Less future discount
config.epsilon_decay = 0.99;          // Slower exploration decay
config.batch_size = 64;               // Larger training batches
```

## Example Structure

All examples follow this structure:

```
example_name.rs
├── Module Documentation (//!)
│   ├── Purpose and overview
│   ├── Algorithm explanation with math
│   └── Running instructions
├── Configuration Struct
│   └── Hyperparameters with documentation
├── Data Structures
│   ├── Environment
│   ├── Agent
│   └── Supporting types
├── Core Functions
│   ├── train_agent()
│   ├── evaluate_agent()
│   └── main()
└── Tests
    ├── Unit tests for components
    ├── Integration tests
    └── Property-based tests
```

### Documentation Sections

Every example includes:

1. **Module-level docs** (`//!`): Overview, algorithm explanation, running instructions
2. **Type documentation**: Each struct documents purpose and fields
3. **Function documentation**: Parameters, return values, examples
4. **Inline comments**: Explain non-obvious logic
5. **Constants**: Clearly named hyperparameters with units

## Writing Your Own Examples

### Step 1: Create the File

```bash
touch crates/evorl-envs/examples/my_example.rs
```

### Step 2: Add Module Documentation

```rust
//! Brief description of what this example demonstrates.
//!
//! # Overview
//!
//! Detailed explanation of the problem, algorithm, and learning objectives.
//!
//! # Algorithm
//!
//! Mathematical formulation if applicable.
//!
//! # Running
//!
//! ```sh
//! cargo run --example my_example --release
//! ```
//!
//! # Expected Output
//!
//! Describe what successful output looks like.
```

### Step 3: Implement Core Logic

Follow the pattern:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize environment and agent
    let mut env = MyEnv::new()?;
    let mut agent = MyAgent::new()?;

    // 2. Training loop
    for episode in 0..NUM_EPISODES {
        let mut state = env.reset()?;
        
        loop {
            let action = agent.select_action(&state);
            let (next_state, reward, done) = env.step(action)?;
            agent.learn(&state, action, reward, &next_state, done)?;
            state = next_state;
            
            if done {
                break;
            }
        }
    }

    // 3. Evaluation
    evaluate_agent(&mut env, &agent)?;

    Ok(())
}
```

### Step 4: Add Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let env = MyEnv::new().unwrap();
        // assertions...
    }

    #[test]
    fn test_integration() -> Result<(), Box<dyn std::error::Error>> {
        let mut env = MyEnv::new()?;
        env.reset()?;
        // assertions...
        Ok(())
    }
}
```

### Step 5: Register in Cargo.toml

```toml
[[example]]
name = "my_example"
path = "examples/my_example.rs"
```

## Testing Examples

### Running Example Tests

```bash
# All example tests
cargo test --examples

# Specific example
cargo test --examples my_example

# With output
cargo test --examples -- --nocapture --test-threads=1
```

### Test Coverage

Examples should include:

- **Unit Tests**: Individual functions and components
- **Integration Tests**: Full training loops
- **Property Tests**: Invariants (e.g., rewards > 0)

### Example Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Unit tests
    #[test]
    fn test_environment_reset() {
        let mut env = CartPoleEnv::new(500);
        let state = env.reset();
        assert!(state.len() == 4);
    }

    // Integration tests
    #[test]
    fn test_training_loop() -> Result<(), Box<dyn std::error::Error>> {
        let config = DqnConfig::default();
        let rewards = train_agent(&config)?;
        assert!(!rewards.is_empty());
        Ok(())
    }

    // Property tests
    #[test]
    fn test_episode_rewards_positive() {
        let mut env = CartPoleEnv::new(500);
        env.reset();
        let (_, reward, _) = env.step(0);
        assert!(reward > 0.0);
    }
}
```

## Best Practices

### 1. Error Handling

Use `Result` with the `?` operator:

```rust
// ✅ Good
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = CartPoleEnv::new()?;
    // ...
    Ok(())
}

// ❌ Avoid
fn main() {
    let env = CartPoleEnv::new().unwrap(); // Panics on error
}
```

### 2. Documentation

Include examples in docstrings:

```rust
/// Trains the agent on CartPole.
///
/// # Examples
///
/// ```no_run
/// let config = DqnConfig::default();
/// train_agent(&config)?;
/// ```
fn train_agent(config: &DqnConfig) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // ...
}
```

### 3. Reproducibility

Use deterministic initialization where possible:

```rust
// ✅ Good - seeded randomness
let seed = 42;
let mut rng = StdRng::seed_from_u64(seed);

// ✅ Good - explicit hyperparameters
let config = DqnConfig {
    learning_rate: 0.001,
    // ...
};
```

### 4. Performance

- Use `--release` for training: `cargo run --example ... --release`
- Profile hot paths with flamegraph
- Document time complexity

### 5. Output

Print progress regularly:

```rust
if (episode + 1) % 50 == 0 {
    println!("Episode {}: Reward = {:.1}", episode + 1, reward);
}
```

### 6. Backend Specification

For computational examples, explicitly specify backends:

```rust
use burn::backend::NdArray;

type Backend = NdArray;
let model = MyModel::<Backend>::new(&device)?;
```

## Troubleshooting

### Example Takes Too Long

```bash
# Run with optimizations
cargo run --example cartpole_training --release

# Or reduce episodes in the code:
config.num_episodes = 100;
```

### Tests Fail

```bash
# Run with verbose output
cargo test --examples -- --nocapture

# Run specific failing test
cargo test --examples test_name -- --nocapture
```

### Memory Issues

- Reduce `replay_buffer_size` in config
- Use smaller batch sizes
- Run in release mode: `--release`

## Contributing Examples

When submitting new examples:

1. Include comprehensive module documentation
2. Add unit and integration tests
3. Document expected output
4. Include troubleshooting tips
5. Follow the structure in `cartpole_training.rs`
6. Run `cargo fmt` and `cargo clippy`
7. Update this README with the new example

## References

- [Rust API Guidelines - Examples](https://rust-lang.github.io/api-guidelines/documentation.html#examples-are-provided-c-example)
- [Deep Q-Networks (DQN) Paper](https://www.nature.com/articles/nature14236)
- [CartPole Gym Environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)
- [burn Documentation](https://docs.rs/burn/)

---

**Last Updated**: 2024
**Maintainer**: burn-evorl Team