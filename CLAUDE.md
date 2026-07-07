# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and human
contributors when working with code in this repository.

## Project Overview

`rlevo` is an **Evolutionary Deep Reinforcement Learning** library written in
Rust, using the [Burn](https://github.com/tracel-ai/burn) framework for tensor
operations and neural network architectures. The project is currently in
**alpha stage** with a focus on establishing core architecture and design
patterns.

## Project docs

| Doc | Purpose |
|-----|---------|
| [`docs/rules.md`](docs/rules.md) | Hard constraints and conventions — **read before implementing anything** |
| [`docs/adr/`](docs/adr/) | Immutable architectural decision records ([annotated index](docs/adr/README.md)) |
| [`docs/user-book/`](docs/user-book/) | User / researcher guide (mdBook) |
| [`docs/contributor-book/`](docs/contributor-book/) | Developer / contributor guide (mdBook) |

## Constraints

See [`docs/rules.md`](docs/rules.md) for the full constraint list. Read it before
making implementation decisions. The single most important rule: **production
crates must not depend on `rlevo-benchmarks` or any visualization crate** — the
benchmark/viz layer consumes production types, never constrains them.

## Architectural Decisions

Immutable architectural decision records live in [`docs/adr/`](docs/adr/); the
[annotated index](docs/adr/README.md) summarizes each one. Read them for the
*why* behind crate boundaries and trait design. When you make an architectural
decision, add a new numbered ADR there — do not edit an accepted one; supersede
it.

## Development Commands

### Building
```bash
# Build the entire workspace
cargo build

# Build a specific crate
cargo build -p rlevo-core
cargo build -p rlevo-reinforcement-learning
cargo build -p rlevo-environments
cargo build -p rlevo-evolution
cargo build -p rlevo-hybrid
cargo build -p rlevo-benchmarks
cargo build -p rlevo-benchmarks-report-client

# Build in release mode
cargo build --release
```

### Testing
```bash
# Run all tests in the workspace
cargo test --workspace

# Run tests for a specific crate
cargo test -p rlevo-core
cargo test -p rlevo-environments

# Run a test for a specific file
cargo test -p rlevo-core -- action

# Run a specific test by name
cargo test test_environment_reset

# Run tests with verbose output
cargo test -- --nocapture
```

### Running Examples
```bash
# Run a specific example
cargo run -p rlevo-core --example grid_agent
```

### Formatting
```bash
# Format the whole workspace (config: rustfmt.toml, stable-only)
cargo fmt --all

# Check formatting the way CI does — fails on any drift (fmt.yml gate)
cargo fmt --all --check
```
Formatting is enforced in CI. The toolchain is pinned in `rust-toolchain.toml`
so local and CI rustfmt agree; run `cargo fmt --all` before pushing.

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

## Working with the Codebase

### Adding a New Environment

1. Implement required traits in `rlevo-core`:
   - `State<D>` for full state representation
   - `Observation<D>` for agent perception
   - `Action<D>` for valid actions
   - `Snapshot<D>` for step results (usually use `SnapshotBase`)

2. Implement `Environment<R, SR, AR>` and `ConstructableEnv`:
   ```rust
   impl Environment<1, 1, 1> for MyEnv {
       type StateType = MyState;
       type ObservationType = MyObservation;
       type ActionType = MyAction;
       type RewardType = ScalarReward;
       type SnapshotType = SnapshotBase<1, MyObservation, ScalarReward>;

       fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> { ... }
       fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> { ... }
   }
   ```
   Construction lives on the standalone `ConstructableEnv` factory trait, not on
   `Environment` (ADR 0011).

3. Add to `rlevo-environments` in the appropriate module (`classic/`, `games/`,
   `landscapes/`, `locomotion/`, `grids/`).

4. Implement `TensorConvertible` for state/action types if using neural networks.

5. Write tests following existing patterns.

### Implementing a New RL Algorithm

1. Create module under `rlevo-reinforcement-learning/src/algorithms/`
2. Define configuration struct (e.g., `DqnConfig`)
3. Implement agent struct (e.g., `DqnAgent`)
4. Define neural network model (e.g., `DqnModel`)
5. Integrate with `rlevo-core` traits for environment interaction
6. Use Burn for tensor operations and gradient computation

### Const Generics and Type Inference

When working with const generic dimensions:
- State order `SR` must match across `State`, its `Observation`, and `Snapshot`
  for the same-modality case; `Environment<R, SR, AR>` permits `R != SR` for
  modality-changing POMDPs (see `Observable<OR>`, ADR 0019).
- Action order `AR` must be consistent with `Action<AR>`.
- Environment's `R`, `SR`, `AR` parameters create a type-level constraint system.

If you encounter dimension mismatch errors, verify:
1. The const generic parameters match across all trait bounds
2. `shape()` implementations return arrays of correct length
3. Tensor conversions preserve dimensionality

## Key Files and Patterns

### Critical Trait Definitions
- `crates/rlevo-core/src/state.rs`: State, Observation, and POMDP seams (`Observable`, `BeliefState`, ...)
- `crates/rlevo-core/src/action.rs`: Action trait hierarchy
- `crates/rlevo-core/src/environment.rs`: Environment and Snapshot traits
- `crates/rlevo-core/src/base.rs`: Reward, TensorConvertible, transition dynamics
- `crates/rlevo-reinforcement-learning/src/memory.rs`: RL replay buffer (`PrioritizedExperienceReplay`, `TrainingBatch`) — moved out of core in ADR 0003
- `crates/rlevo-reinforcement-learning/src/experience.rs`: `ExperienceTuple`, `History` — RL-only trajectory storage
- `crates/rlevo-reinforcement-learning/src/metrics.rs`: `AgentStats`, `PerformanceRecord` — RL-only episode tracking

### Example Reference Implementations
- `crates/rlevo-environments/src/classic/bandit/k_armed.rs`: Complete environment example with extensive tests
- `crates/rlevo-environments/src/pixel_grid.rs`: `Observable<OR>` modality-changing env (ADR 0020)
- `crates/rlevo-core/examples/grid_position.rs`: State/Action implementation patterns
- `crates/rlevo-core/src/environment.rs`: MockEnvironment in test module demonstrates trait usage

### Error Handling Patterns
- `EnvironmentError`: For environment operations (InvalidAction, RenderFailed, IoError)
- `StateError`: For state validation (InvalidShape, InvalidData, InvalidSize)
- Use `Result<T, ErrorType>` for fallible operations
- Implement `std::error::Error` and `Display` for custom error types

## Dependencies and Workspace Configuration

The workspace uses shared dependencies defined in root `Cargo.toml`:
- **burn**: Version 0.21.0 with features `["wgpu", "train", "tui", "metrics", "ndarray"]`
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

Test placement (ADR 0012): unit tests in-source → single-crate integration tests
in `crate/tests/` → cross-crate integration tests in `crates/rlevo/tests/`.

## Current Development Focus

The project is in early alpha with emphasis on:
1. **Trait design and API stability**: Core abstractions are being refined
2. **Type-level safety**: Using const generics to enforce dimensional correctness
3. **Integration patterns**: Establishing conventions for Burn tensor conversion
4. **Documentation**: Inline docs are extensive — maintain this standard
5. **User Guide**: the user/researcher guide (`docs/user-book`)
6. **Contributor Guide**: the developer/contributor guide (`docs/contributor-book`)

## Commit Messages

Act as a Senior Software Engineer writing professional Git commits.

**Format**: Conventional Commits — `<type>(<scope>): <description>`

- **Types**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`
- **Scope**: The specific Rust crate, module, or struct being modified (e.g., `rlevo-core`, `evolution`, `memetic`)

**Rules**:

1. Subject line: max 50 characters, imperative mood, capitalized, no trailing punctuation
2. Separate subject from body with a blank line
3. Body (optional): explain *why*, not what; wrap at 72 characters; omit if not useful
4. Append a `CRITICAL:` line flagging the most complex or risky part of the diff (e.g., a lifetime adjustment, unsafe block, lock ordering, or a subtle invariant)
5. End with `Co-Authored-By: <model> <noreply@anthropic.com>`, where
   `<model>` is the Claude model that authored the change — one of
   `Claude Fable 5`, `Claude Opus 4.8`, or `Claude Sonnet 4.6`

**Output**: raw commit message text only — no markdown fences, no preamble, no meta-commentary.
