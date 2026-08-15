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

Standard cargo invocations. Formatting is enforced in CI (`cargo fmt --all --check`,
`.github/workflows/fmt.yml`) and the toolchain is pinned in `rust-toolchain.toml` so
local and CI rustfmt agree — run `cargo fmt --all` before pushing.

## Working with the Codebase

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
- `crates/rlevo-reinforcement-learning/src/replay/`: RL replay-strategy seam (`ReplayStrategy`, `UniformReplay`, `PrioritizedReplay`, `PrioritizedReplaySettings`) — moved out of core in ADR 0003, reshaped as a strategy seam in ADR 0050
- `crates/rlevo-reinforcement-learning/src/experience.rs`: `ExperienceTuple`, `History` — RL-only trajectory storage
- `crates/rlevo-reinforcement-learning/src/metrics.rs`: `AgentStats`, `PerformanceRecord` — RL-only episode tracking

### Example Reference Implementations
- `crates/rlevo-environments/src/classic/bandit/k_armed.rs`: Complete environment example with extensive tests
- `crates/rlevo-environments/src/pixel_grid.rs`: `Observable<OR>` modality-changing env (ADR 0020)
- `crates/rlevo-core/examples/grid_position.rs`: State/Action implementation patterns
- `crates/rlevo-core/src/environment.rs`: MockEnvironment in test module demonstrates trait usage

## Testing Philosophy

- Property/invariant tests use `proptest` (see ADR 0036), a `rlevo-evolution`-only
  dev-dependency. proptest generates **host config only** (`λ`, `D`, structural
  sizes, a `seed: u64`); the test body drives all algorithm randomness through
  `seed_stream(seed, generation, SeedPurpose::_)` per ADR 0029 — proptest's own
  PRNG never touches Burn. Use proptest for **input-space invariants** (roundtrips,
  shape/length invariants, monotonicity, "no panic / no NaN across the valid
  domain", `Validate` accept/reject boundaries); keep seeded-`StdRng` example
  tests for **specific-scenario / known-answer** cases (e.g.
  `*_converges_on_sphere_d10`). Do not rewrite passing example tests into
  properties.

Test placement (ADR 0012): unit tests in-source → single-crate integration tests
in `crate/tests/` → cross-crate integration tests in `crates/rlevo/tests/`.

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
5. End with a `Claude-Session:` trailer linking the session that authored the
   change — e.g. `Claude-Session: https://claude.ai/code/session_<id>`. The
   session link is the useful provenance: it reaches the reasoning behind the
   diff, which a model name alone does not. Omit the trailer entirely for
   commits authored without Claude.

**Output**: raw commit message text only — no markdown fences, no preamble, no meta-commentary.
