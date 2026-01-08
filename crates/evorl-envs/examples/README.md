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

## Available Examples

### 1. 10-Armed Bandit Training (`ten_armed_bandit_training.rs`)

**Purpose**: Demonstrate three classical approaches to the exploration-exploitation trade-off in reinforcement learning.

**What It Demonstrates**:
- Multi-armed bandit problem formulation
- Three distinct algorithms selectable via feature flags:
  - **Epsilon-Greedy** (default): Random exploration with probability ε
  - **UCB** (feature="ucb"): Upper Confidence Bound algorithm
  - **Thompson Sampling** (feature="thompson"): Bayesian posterior sampling
- Incremental action-value estimation
- Performance metrics and evaluation

**Key Concepts**:
- **State Space**: Stateless (bandit problem)
- **Action Space**: 10 discrete actions (arms 0-9)
- **Reward Distribution**: Each arm returns N(q*(a), 1) where q*(a) ~ N(0, 1)
- **Goal**: Maximize total reward over 1000 steps by learning optimal arm
- **Success Metrics**: 
  - Average reward per step
  - Percentage of optimal actions taken

**Running**:
```bash
# Default epsilon-greedy (ε=0.1)
cargo run --example ten_armed_bandit_training

# UCB algorithm (c=2.0) - best performance
cargo run --example ten_armed_bandit_training --features ucb

# Thompson Sampling - Bayesian approach
cargo run --example ten_armed_bandit_training --features thompson
```

**Expected Performance** (after 1000 steps):
- Epsilon-Greedy: ~1.15 avg reward, ~80% optimal actions
- UCB: ~1.33 avg reward, ~91% optimal actions
- Thompson Sampling: ~1.22 avg reward, ~84% optimal actions

**Reference**: Sutton & Barto (2018), Chapter 2

**Detailed Documentation**: See [README_ten_armed_bandit.md](README_ten_armed_bandit.md)

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
touch crates/evorl-envs/examples/my-example/my_example.rs
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
path = "examples//my-example/my_example.rs"
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

---

**Last Updated**: 2026
**Maintainer**: burn-evorl Team
