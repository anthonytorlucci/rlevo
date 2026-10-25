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
| [`docs/ask-tell-contract.md`](docs/ask-tell-contract.md) | How an evolutionary run is driven: the ask/tell protocol, `EvolutionaryLoop`, and many seeds through `rlevo::harness` |

## Constraints

See [`docs/rules.md`](docs/rules.md) for the full constraint list. Read it before
making implementation decisions. The single most important rule: **production
crates must not depend on any visualization crate** (`ratatui`, `leptos`, chart
crates, ...); `rlevo-harness` is the sole exception, and only behind its
`report` / `record` feature gates. The umbrella `rlevo` carries
`rlevo-harness` at its default features, which link no visualization
crate, and re-exports it as `rlevo::harness` (ADR 0084). No other
production crate depends on it, except `rlevo-environments` under its
off-by-default `record` feature. The benchmark/viz layer consumes production
types, never constrains them.

The second: **library crates select no Burn backend** (ADR 0080). The
workspace `burn` entry is `default-features = false, features = ["std"]`;
`rlevo-reinforcement-learning` adds `optim`, which brings `autodiff`, and
neither selects a backend (ADR 0093). On Burn 0.22 there is no backend type
parameter: library code takes the `burn::tensor::Device` it is given and never
constructs one (`Device::flex()`, `Device::wgpu(..)`). An RL agent switches
autodiff on itself, from the device its caller passes. The leaf picks the
backend: the `rlevo` umbrella's `flex` (default) and `wgpu` features for a
consumer, `[dev-dependencies]` for a crate's own tests, benches and doctests,
which name their device (`Device::flex()`, never `Device::default()`, which is
wgpu in a build that compiles it; `tools/no-default-device.sh` checks). A workspace-wide build hides a
violation, so check a library alone: `cargo check -p <crate> --lib`.
`rlevo-core` goes further and depends on Burn only under its `burn` feature
(ADR 0089): an environment author implements `HostRow` and writes no Burn
code, and `TensorConvertible` comes with the feature, for every `HostRow`.

## Architectural Decisions

Immutable architectural decision records live in [`docs/adr/`](docs/adr/); the
[annotated index](docs/adr/README.md) summarizes each one. Read them for the
*why* behind crate boundaries and trait design. When you make an architectural
decision, add a new numbered ADR there — do not edit an accepted one; supersede
it.

An accepted ADR keeps the names of its day. Three renames to carry when
reading one: the harness crate was `rlevo-benchmarks` before ADR 0088 and is
`rlevo-harness`; the generation loop was `EvolutionaryHarness` before ADR
0085 and is `EvolutionaryLoop`; and before ADR 0093 (Burn 0.22) every tensor,
module and trait carried a backend parameter (`Tensor<B, D>`, `Linear<B>`,
`Strategy<B>`, `B::Device`, `B: AutodiffBackend`), which is now gone: read
`Tensor<D>`, `Linear`, `Strategy`, `Device`, and autodiff as device state.

## Development Commands

Standard cargo invocations. Formatting is enforced in CI (`cargo fmt --all --check`,
`.github/workflows/fmt.yml`) and the toolchain is pinned in `rust-toolchain.toml` so
local and CI rustfmt agree — run `cargo fmt --all` before pushing.

Tests are split into three tiers, each an `xtask` command that owns the list of
what it runs. Together they partition the workspace's tests, and a `#[ignore]`d
test claimed by no tier fails the command that notices — so a new heavy test
cannot quietly end up run by nothing.

| Command | Runs | Where |
|---------|------|-------|
| `cargo xtask test-fast` | every crate's default test set, with the features its test targets need | `crate-tests.yml`, the pull-request gate |
| `cargo xtask test-heavy` | the `#[ignore]`d long tests a CPU can run | `weekly-tests.yml` |
| `cargo xtask test-gpu` | the `#[ignore]`d tests needing a wgpu adapter | local only — never CI |

Both CI workflows build their matrices from the tier tables (`--list`), so adding
a crate or a heavy test target is a one-line change in `xtask/src/commands/`, not
a YAML edit. `test-gpu` aborts rather than fails on a host with no adapter, which
is why no workflow calls it.

The tiers build inside the workspace, where siblings resolve by path. An
**outside-in probe** builds a small crate against the workspace *as packaged*,
from a directory outside the checkout, so it sees what a `cargo add rlevo`
user sees. `cargo xtask probe <name>` runs one; `--list` prints them, and
`crate-tests.yml` runs each on every pull request. The table is `PROBES` in
`xtask/src/commands/probe/mod.rs`, and each probe's template is
`xtask/probes/<name>/`. A change to the API a template uses changes the
template in the same pull request: its diff is the measure of what an outside
author stops or starts writing.

| Probe | What it is |
|-------|------------|
| `byoe-1-thermostat` | an environment author's walk: implement `Environment`, hold the post-terminal contract, write a policy as an `Agent` that implements only `act`, run it through `Evaluator::run_suite` |
| `byoe-1-tank-farm` | the same walk over a rank-2 observation, a bounded continuous action that can be invalid, and an episode that an overflow terminates or `TimeLimit` truncates; the harness runs it through the wrapper |
| `byoe-2` | a landscape author's walk: implement `Landscape` for an invented cost surface, wrap it with `FromLandscape`, search it with a shipped strategy in an `EvolutionaryLoop`, and run that through `Evaluator::run_trials`; the best fitness must fall |
| `learner-1` | a learner's walk: the README's Quick Start pasted into a new crate as written, built and run to the line it prints when its episode ends. Every `rlevo` dependency line in the README and the umbrella's crate docs must name the release the checkout builds |
| `byoa-1a` | an algorithm author's walk: implement `Agent` for a tabular Q-learner, then train and evaluate it on FrozenLake with one `Evaluator::run_suite` call and `EvaluatorConfig::train_steps`; the trial must pass every step's status to the learner unchanged, with both endings in the run, the learner must beat a random agent under the same configuration, and a second run must repeat the first bit for bit |
| `byoa-1b` | an optimiser author's walk: implement `Strategy` for a hill-climber and for a control that never moves, and run both on a shipped landscape through `Evaluator::run_trials`; only the hill-climber may improve, and a second run must repeat the first bit for bit |
| `consumer-core` | that author's environment file built against `rlevo-core` alone, with its package count, its Burn package count and a cold build time |
| `consumer-umbrella`, `consumer-ppo` | the same three numbers for `rlevo` at default features, and for `rlevo-core` with `rlevo-reinforcement-learning`; `consumer-umbrella` also builds a network derived through `rlevo::burn` beside a default one, which no doctest can check |

The layering rules no compiler checks are checked by **`cargo xtask guards`**,
on every pull request (`crate-tests.yml`, `layering guards`):
`rlevo-evolution` names no path into the environment contract (ADR 0002); no
published crate depends on `rlevo-harness` but the umbrella, and
`rlevo-environments` only under `record`; the umbrella's default features turn
on no viz, recording or `rerun` feature; no published crate reaches a viz
package at its default features, in its build or its tests; no workspace
manifest names `ratatui` or `crossterm` (ADR 0096); and the five
`katex-header.html` copies are byte-identical. A failed guard says what
breaks the rule: a file and line, a crate and its feature, the copy that
differs, or, for a viz package, the `cargo tree -i` command that prints the
path to it. The guards are in `xtask/src/commands/guards.rs`.

A change to a crate's public surface reports its **surface delta** (ADR 0081):
`tools/public-surface.sh <crate>` lists the public items, one per line, and
the delta is the diff of the listing at the base commit and on the branch. A
`pub` item needs an implementor and a consumer; an addition says what it
replaces.

## Working with the Codebase

### Common Generic Types

| Symbol | Type | Description |
| --- | --- | --- |
| `OR` | `Observation` | rank of the representative observation tensor |
| `AR` | `Action` | rank of the representative action tensor |
| `E` | Environment | - |
| `BS` | Batch size | - |

### Const Generics and Type Inference

When working with const generic dimensions:
- `Environment<OR, AR>` has two associated types, `ObservationType:
  Observation<OR>` and `ActionType: Action<AR>`; `reset` and `step` return
  `Snapshot<Self::ObservationType>`. An environment's state is its own type,
  with no rank in the contract (ADR 0090), so a modality-changing POMDP
  needs nothing special.
- Action order `AR` must be consistent with `Action<AR>`.
- Environment's `OR` and `AR` parameters create a type-level constraint system.

If you encounter dimension mismatch errors, verify:
1. The const generic parameters match across all trait bounds
2. `shape()` implementations return arrays of correct length
3. Tensor conversions preserve dimensionality

### Mathematical Notation in Docs

Five crates (`rlevo-core`, `rlevo-environments`, `rlevo-evolution`,
`rlevo-harness`, `rlevo-reinforcement-learning`) ship a
`katex-header.html` (`crates/<crate>/katex-header.html`, wired via
`[package.metadata.docs.rs] rustdoc-args`) that renders KaTeX in `cargo doc`
output; the five copies stay byte-identical, which `cargo xtask guards`
checks. Rustdoc and ADR/markdown docs need **different** math syntax — see
[`docs/rules.md`](docs/rules.md)'s Documentation section ("Mathematical
notation" subsection) before writing or editing either. Short version: rustdoc
math is backtick-wrapped (`` `$...$` `` inline, ` ```math ` fenced for
display); ADR/markdown math is bare (`$...$` / `$$...$$`). In both, convert
only genuine formula content — never touch ASCII diagrams, real code fences,
verbatim quotes, or a plain `→` used as flow/rename shorthand.

## Key Files and Patterns

### Critical Trait Definitions
- `crates/rlevo-core/src/action.rs`: Action trait hierarchy
- `crates/rlevo-core/src/environment.rs`: the `Environment` trait, the `Snapshot<O>` struct every `reset` and `step` returns (read through its fields), `EpisodeStatus`, `EnvironmentError`; the emission model and construction are documented there as patterns, not traits (ADR 0090)
- `crates/rlevo-core/src/base.rs`: `Observation`, `Action`, `HostRow` (the row a tensor is built from), `TensorConvertible` (behind the `burn` feature)
- `crates/rlevo-core/src/reward.rs`: `ScalarReward`, the one reward type, and its finiteness obligation
- `crates/rlevo-core/src/agent.rs`: `Agent`, the one agent seam: `act`, and four hooks that default to doing nothing (`on_transition`, `on_episode_end`, `set_training`, `emit_metrics`), so a frozen policy is one method long; and `Transition`, the step a learner is shown and the replay stores (ADR 0095)
- `crates/rlevo-harness/src/evaluator.rs`: `EpisodicTrial` and `train`, the one loop that drives an `Agent`. Its private `play_episode` holds the status rule for both phases: a transition carries the snapshot's status unchanged, and an episode cut at `max_steps` ends `Truncated` for the agent while its last transition stays `Running`
- `crates/rlevo-reinforcement-learning/src/replay/`: the off-policy agents' replay buffers — moved out of core in ADR 0003, reshaped as a strategy seam in ADR 0050, crate-private but for the `PrioritizedReplaySettings` opt-in since ADR 0095
- `crates/rlevo-evolution/src/strategy.rs`: `Strategy`, the stateless ask/tell contract, and `EvolutionaryLoop`, which holds a run's state and drives one with `reset()` / `step()` / `is_done()`. `step()` returns nothing; the status is the separate predicate (ADR 0085). For a `Strategy`, the loop is where a `Minimize` objective is negated and fitness is sanitised
- `crates/rlevo-evolution/src/fitness.rs`: `BatchFitnessFn`, which declares its `ObjectiveSense`, and the fitness-hygiene rule as two public functions, `sanitize_fitness` / `sanitize_fitness_tensor`. A `Strategy`'s `tell` and a hand-written ask/tell loop call them, on canonical values, after any negation; a run through `EvolutionaryLoop` calls neither (ADR 0086)
- `crates/rlevo-core/src/evaluation.rs`: `GenerationProbe` (`begin` / `advance` / `emit_metrics`), the seam the evaluation harness drives a generation loop through. `advance` past exhaustion returns `None` and does nothing; the harness calls it once at its `max_steps` cap to learn whether a trial was cut short, and warns if it was. It reads `emit_metrics` before that call, so the generation the call may run is in no reported number
- `crates/rlevo-core/src/fitness.rs`: `Metric`, and the one rule `Agent::emit_metrics` and `GenerationProbe::emit_metrics` follow (ADR 0097): a read describes the trial so far (a counter is a total since the trial began, a scalar the value at the read) and changes nothing, so a reader may read at every boundary and keeps the latest; nothing adds two reads together
- `crates/rlevo-evolution/src/probe.rs`: the `GenerationProbe` impls for the two loops, built from the three calls above, and the recipe a new driver follows. A run is driven by hand or by the harness; the harness resets the loop itself

### Example Reference Implementations
- `crates/rlevo-environments/src/classic/bandit/k_armed.rs`: Complete environment example with extensive tests
- `crates/rlevo-environments/src/pixel_grid.rs`: modality-changing env, a rank-1 latent observed as a rank-3 image through `PixelGridState::project` (ADR 0020)
- `xtask/probes/byoe-1-thermostat/src/thermostat.rs`: Environment/Action/Observation implementation patterns, as an outside author writes them; its probe builds it on every pull request
- `xtask/probes/byoe-1-tank-farm/src/tank_farm.rs`: the same for a `ContinuousAction` + `BoundedAction` command, a rank-2 observation and an environment with no horizon of its own
- `xtask/probes/byoa-1a/src/q_learner.rs`: an `Agent` that learns, implemented from outside the workspace: ε-greedy from the driver's RNG, a TD target that bootstraps through `Truncated` and not through `Terminated`, `set_training` switching exploration off, and its own counters through `emit_metrics`
- `xtask/probes/byoa-1b/src/hill_climber.rs`: a `Strategy` implemented from outside the workspace: parameters with `Validate`, state threaded through `init` / `ask` / `tell`, draws from the loop's RNG, fitness sanitised where it reaches the host
- `crates/rlevo-core/src/environment.rs`: MockEnvironment in test module demonstrates trait usage

## Testing Philosophy

- Property/invariant tests use `proptest` (see ADR 0036), a `rlevo-evolution`-only
  dev-dependency. proptest generates **host config only** ($\lambda$, `D`, structural
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

**Output**: raw commit message text only — no markdown fences, no preamble, no meta-commentary.
