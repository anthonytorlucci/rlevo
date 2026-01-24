# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

burn-evorl is an **Evolutionary Deep Reinforcement Learning** library written in Rust, using the [Burn](https://github.com/tracel-ai/burn) framework for tensor operations and neural network architectures. The project is currently in **alpha stage** with a focus on establishing core architecture and design patterns.

## Development Commands

### Building
```bash
# Build the entire workspace
cargo build

# Build a specific crate
cargo build -p evorl-core
cargo build -p evorl-rl
cargo build -p evorl-envs
cargo build -p evorl-evolution
cargo build -p evorl-hybrid
cargo build -p evorl-utils

# Build in release mode
cargo build --release
```

### Testing
```bash
# Run all tests in the workspace
cargo test

# Run tests for a specific crate
cargo test -p evorl-core
cargo test -p evorl-envs

# Run a test for a specific file
cargo test -p evorl-core -- action

# Run a specific test by name
cargo test test_environment_reset

# Run tests with verbose output
cargo test -- --nocapture
```

### Running Examples
```bash
# List all examples
cargo run --example

# Run a specific example
cargo run -p evorl-core --example grid_position
cargo run -p evorl-core --example combat_action
```

### Linting
```bash
# Check for clippy warnings (workspace has extensive lints configured)
cargo clippy --all-targets --all-features

# Auto-fix warnings where possible
cargo clippy --fix
```

### Documentation
```bash
# Generate and open documentation
cargo doc --open

# Generate docs for all workspace members
cargo doc --workspace --no-deps
```

## Architecture Overview

### Crate Structure

The project follows a **workspace architecture** with six primary crates:

1. **evorl-core**: Foundation layer providing generic traits and core abstractions
   - Defines trait-based interfaces for: `State`, `Observation`, `Action`, `Environment`, `Agent`, `Reward`
   - Implements memory/replay buffers (`ReplayBuffer`)
   - Provides tensor conversion utilities via `TensorConvertible` trait
   - No algorithm implementations - purely abstract interfaces

2. **evorl-envs**: Concrete environment implementations
   - `classic/`: Classic control problems (e.g., `ten_armed_bandit.rs`)
   - `games/`: Board games (chess, connect4)
   - `benchmarks/`: Standard RL benchmarks
   
3. **evorl-rl**: Reinforcement learning algorithms using deep neural networks
   - `algorithms/dqn/`: Deep Q-Network variants
   - Depends on `evorl-core` for trait definitions
   - Integrates with Burn for neural network architectures

4. **evorl-evolution**: Evolutionary optimization engine
   - Population management
   - Selection strategies (tournament, elite preservation)
   - Mutation operators (Gaussian noise, crossover)

5. **evorl-hybrid**: Combines RL and evolutionary approaches
   - Genome representation (genotype)
   - Agent phenotype implementations
   - Fitness evaluators bridging environment rewards to evolution
   - Population orchestrator with parallel dispatch

6. **evorl-utils**: Shared utilities
   - Math operations, logging, validation, serialization

### Core Design Pattern: Trait-Based Architecture

The library follows a **trait-based composition model** rather than inheritance:

```
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│   Agent     │────→│ Environment  │────→│  Model   │
│ (Learner)   │     │ (Problem)    │     │ (Policy) │
└─────────────┘     └──────────────┘     └──────────┘
```

**Key traits** (all defined in `evorl-core/src/`):
- `State<const D: usize>`: Complete Markovian state representation
- `Observation<const D: usize>`: What the agent perceives (may be partial)
- `Action<const D: usize>`: Valid agent actions (with `DiscreteAction` and `ContinuousAction` specializations)
- `Environment<const D: usize, const SD: usize, const AD: usize>`: Defines `reset()` and `step(action)` protocol
- `Snapshot`: Result of `reset()`/`step()` containing `(observation, reward, done)`
- `Reward`: Scalar reward type (implements `Add`, `From<f32>`)

### Dimensionality Encoding

States, observations, and actions use **const generics** to encode dimensionality:
- `D` = dimension of observation space
- `SD` = dimension of state space  
- `AD` = dimension of action space

Example: `Environment<1, 1, 1>` represents 1D observation, 1D state, 1D action.

### Tensor Conversion

The `TensorConvertible<const D: usize, B: Backend>` trait bridges domain types to Burn tensors:
```rust
trait TensorConvertible<const D: usize, B: Backend> {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, D>;
}
```

This enables seamless integration between:
- Environment states → Neural network inputs
- Neural network outputs → Actions
- Memory buffers → Training batches

### Evolutionary RL Concepts

**Hybrid approach** combines two paradigms:

1. **Genome (Genotype)**: Raw data (`Vec<f64>`) encoding agent parameters
2. **Agent (Phenotype)**: Living instance that interacts with environment
3. **Fitness Evaluator**: Converts environment rewards to scalar fitness for evolution
4. **Population Orchestrator**: Manages generation lifecycle with parallel evaluation (via Rayon)

## Working with the Codebase

### Adding a New Environment

1. Implement required traits in `evorl-core`:
   - `State<D>` for full state representation
   - `Observation<D>` for agent perception
   - `Action<D>` for valid actions
   - `Snapshot<D>` for step results (usually use `SnapshotBase`)

2. Implement `Environment<D, SD, AD>` trait:
   ```rust
   impl Environment<1, 1, 1> for MyEnv {
       type StateType = MyState;
       type ObservationType = MyObservation;
       type ActionType = MyAction;
       type RewardType = ScalarReward;
       type SnapshotType = SnapshotBase<1, MyObservation, ScalarReward>;
       
       fn new(render: bool) -> Self { ... }
       fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> { ... }
       fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> { ... }
   }
   ```

3. Add to `evorl-envs` in the appropriate module (`classic/`, `games/`, `benchmarks/`)

4. Implement `TensorConvertible` for state/action types if using neural networks

5. Write tests following existing patterns (see `ten_armed_bandit.rs` for examples)

### Implementing a New RL Algorithm

1. Create module under `evorl-rl/src/algorithms/`
2. Define configuration struct (e.g., `DqnConfig`)
3. Implement agent struct (e.g., `DqnAgent`)
4. Define neural network model (e.g., `DqnModel`)
5. Integrate with `evorl-core` traits for environment interaction
6. Use Burn for tensor operations and gradient computation

### Const Generics and Type Inference

When working with const generic dimensions:
- State dimension `D` must match across `State`, `Observation`, and `Snapshot`
- Action dimension `AD` must be consistent with `Action<AD>`
- Environment's `D`, `SD`, `AD` parameters create a type-level constraint system

If you encounter dimension mismatch errors, verify:
1. The const generic parameters match across all trait bounds
2. `shape()` implementations return arrays of correct length
3. Tensor conversions preserve dimensionality

## Key Files and Patterns

### Critical Trait Definitions
- `crates/evorl-core/src/state.rs`: State and Observation traits
- `crates/evorl-core/src/action.rs`: Action trait hierarchy
- `crates/evorl-core/src/environment.rs`: Environment and Snapshot traits
- `crates/evorl-core/src/dynamics.rs`: Reward, transition dynamics
- `crates/evorl-core/src/memory.rs`: Replay buffer implementations

### Example Reference Implementations
- `crates/evorl-envs/src/classic/ten_armed_bandit.rs`: Complete environment example with extensive tests
- `crates/evorl-core/examples/grid_position.rs`: State/Action implementation patterns
- `crates/evorl-core/src/environment.rs`: MockEnvironment in test module demonstrates trait usage

### Error Handling Patterns
- `EnvironmentError`: For environment operations (InvalidAction, RenderFailed, IoError)
- `StateError`: For state validation (InvalidShape, InvalidData, InvalidSize)
- Use `Result<T, ErrorType>` for fallible operations
- Implement `std::error::Error` and `Display` for custom error types

## Dependencies and Workspace Configuration

The workspace uses shared dependencies defined in root `Cargo.toml`:
- **burn**: Version 0.19.0 with features `["wgpu", "train", "tui", "metrics", "ndarray"]`
- **rand**: 0.9.2 for randomness
- **serde**: 1.0 with `["derive", "rc"]` for serialization
- **tracing**: 0.1 for logging

Workspace lints are configured for:
- Rust: `ambiguous_negative_literals`, `missing_debug_implementations`, `redundant_imports`, `unsafe_op_in_unsafe_fn`
- Clippy: All categories at warn level (`cargo`, `complexity`, `correctness`, `pedantic`, `perf`, `style`, `suspicious`)

## Testing Philosophy

- Write comprehensive unit tests in `#[cfg(test)]` modules within each source file
- Test happy paths AND error conditions (see `EnvironmentError` tests)
- Verify trait implementations with mock types (see `MockEnvironment` tests)
- Use `approx` crate for floating-point comparisons
- Test custom trait implementations thoroughly (see `CustomSnapshot` tests)

## Agent Workspace Organization

When working on tasks in this project, please follow these file organization rules:

### Working Files Directory Structure
- Create all task-specific markdown files (checklists, refactor plans, summaries, thinking documents, etc.) in the `agent-space/` directory at the project root
- For each distinct task or problem, create a new subdirectory within `agent-space/` using a descriptive name
- Use ISO date prefixes for task directories to maintain chronological order: `YYYY-MM-DD-task-description/`

### Examples
```
agent-space/
├── 2024-01-15-chess-state-refactor/
│   ├── CHECKLIST.md
│   ├── CHESS_STATE_REFACTOR.md
│   └── IMPLEMENTATION_NOTES.md
├── 2024-01-16-api-redesign/
│   ├── API_CHANGES.md
│   └── MIGRATION_PLAN.md
└── 2024-01-17-performance-optimization/
    └── PROFILING_RESULTS.md
```

### What Goes in `agent-space/`
- ✅ Task checklists and TODO lists
- ✅ Refactoring plans and strategies
- ✅ Architectural decision documents
- ✅ Implementation summaries
- ✅ Debugging notes and analysis
- ✅ Research and exploration documents

### What Stays in the Main Project
- ❌ Permanent documentation (README, API docs, user guides)
- ❌ Source code files
- ❌ Configuration files
- ❌ Test files

### Cleanup
The `agent-space/` directory is considered temporary working space. Task directories can be archived or deleted once the work is complete and any relevant information has been incorporated into permanent documentation.

## Note to All LLMs
This guideline applies to any AI assistant working on this project through Zed or other development tools. Always create your working files in `agent-space/` with appropriate task subdirectories.

## Current Development Focus

The project is in early alpha with emphasis on:
1. **Trait design and API stability**: Core abstractions are being refined
2. **Type-level safety**: Using const generics to enforce dimensional correctness
3. **Integration patterns**: Establishing conventions for Burn tensor conversion
4. **Documentation**: Inline docs are extensive - maintain this standard

The main branch for PRs is `production`, not `main`.
