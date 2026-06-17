---
project: rlevo
status: superseded
type: decision
date: 2026-04-28
tags: [adr, decision, architecture, crates, testing, examples, rlevo]
---

# ADR 0005: Examples and cross-crate integration tests live in the umbrella `rlevo` crate

## Status

Superseded by [0012-split-heavy-examples-into-rlevo-examples](0012-split-heavy-examples-into-rlevo-examples.md). ADR 0012 refines the examples split (lightweight examples stay in `rlevo`; heavy viz/record/report examples move to `rlevo-examples`) and canonicalizes the three-tier test placement rule. Read ADR 0012 as the authoritative source; this document is retained for historical context.

Previously: Active. Companion to [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md). Together they re-shape the workspace so every internal dep edge points strictly up-stack with no cycles even through dev-dependencies.

## Context

Pre-refactor, each subcrate (`rlevo-environments`, `rlevo-evolution`, `rlevo-reinforcement-learning`, `rlevo-benchmarks`) carried its own `examples/` and `tests/` directories. Each subcrate's `Cargo.toml` description reads *"internal crate — use `rlevo` for the full API"*, yet the examples that demonstrate that API lived in the internal crates rather than the umbrella.

This produced two structural problems:

1. **Cross-crate integration tests created dev-dep edges that mirrored prod-dep edges**, in some cases creating dependency cycles. Most notably:
   - `rlevo-benchmarks/tests/evaluator_smoke.rs` exercised the harness against real `rlevo-environments` envs. To compile, `rlevo-benchmarks` dev-depended on `rlevo-environments` *with the `bench` feature on*. But `rlevo-environments[bench]` already depends on `rlevo-benchmarks` (the optional prod dep wired in `bench/{adapter, suites}.rs`). Cargo tolerates the cycle through dev-dependencies, but it confuses feature resolution and human readers alike.
   - `rlevo-evolution/tests/{rastrigin_run_suite, swarm_rastrigin_suite}.rs` and `rlevo-reinforcement-learning/tests/*_integration.rs` similarly required `rlevo-environments` as a dev-dep on top of their already-substantial prod dep cone.

2. **Per-subcrate examples scattered the user-facing demo surface.** A user inspecting `crates/rlevo` for examples found none — they had to look in four sibling crates. Naming was already designed as if the examples would consolidate (every example name is unique workspace-wide). The convention was just never codified.

## Decision

**Examples and cross-crate integration tests live in `crates/rlevo/`. Crate-local unit and integration tests stay in their owning crate.**

The rules:

| Target type                                                              | Lives in                                                |
| ------------------------------------------------------------------------ | ------------------------------------------------------- |
| `#[cfg(test)]` unit tests                                                | The owning crate (unchanged)                            |
| Integration tests that exercise *one* crate's public surface             | The owning crate's `tests/`                             |
| Integration tests that cross *two or more* crates' public surfaces       | `crates/rlevo/tests/`                                   |
| Examples (any kind, any audience)                                        | `crates/rlevo/examples/`, organized by area             |
| Criterion `[[bench]]` entries                                            | The owning crate (measure single-crate hot paths)       |

Concretely:

1. **`git mv` examples to the umbrella.** All examples from `rlevo-environments`, `rlevo-evolution`, `rlevo-reinforcement-learning`, and `rlevo-benchmarks` move under `crates/rlevo/examples/{envs, evo, rl, benchmarks}/`. The umbrella `Cargo.toml` enumerates `[[example]]` entries with the new paths and the `required-features` flags rewritten through the umbrella's existing `box2d` / `locomotion` features.
2. **`git mv` cross-crate integration tests to the umbrella.** Specifically:
   - `rlevo-reinforcement-learning/tests/*_integration.rs` (8 algo tests) + `tests/integration_test.rs` + `tests/baselines/`
   - `rlevo-evolution/tests/{rastrigin_run_suite, swarm_rastrigin_suite}.rs`
   - `rlevo-benchmarks/tests/evaluator_smoke.rs`

   All flatten directly under `crates/rlevo/tests/` (cargo only auto-discovers top-level files in a `tests/` directory).
3. **Crate-internal tests stay put.** `rlevo-environments/tests/grids_solvable.rs` (env-internal) and `rlevo-evolution/tests/{backend_parity, determinism}.rs` (evo-internal, no envs dep) remain in their owning crates.
4. **Drop now-orphan dev-deps from the source crates:**
   - `rlevo-benchmarks/Cargo.toml`: drop `rlevo-environments` and `tracing-subscriber` dev-deps. Only `approx` remains.
   - `rlevo-evolution/Cargo.toml`: drop `rlevo-environments` dev-dep. `criterion` remains for the `operators` bench. (No `rlevo-environments` dev-dep was strictly required after the refactor — the one remaining unit-test reference to `rlevo_environments::landscapes::sphere::Sphere` was rewritten to use a 5-line local `Landscape` impl.)
   - `rlevo-reinforcement-learning/Cargo.toml`: drop `tracing-subscriber` dev-dep and the `[[example]]` block. `rlevo-environments` stays as a dev-dep because `[[bench]]` entries (`dqn_bench`, `sac_bench`, etc.) construct CartPole/Pendulum envs.
5. **Add umbrella dev-deps.** `crates/rlevo/Cargo.toml` gains a `[dev-dependencies]` block: `rlevo-benchmarks`, `rlevo-environments` (with `features = ["bench"]`), `approx`, `burn`, `rand`, `rand_distr`, `serde`, `tracing-subscriber`. Plus all the relocated `[[example]]` and the implicit `[[test]]` entries.

### Scope decision: flatten test layout

Cargo's integration-test discovery looks at top-level `.rs` files in `tests/`; subdirectories are treated as helper modules unless paired with explicit `[[test]]` entries. The plan initially proposed `tests/{rl, evo, benchmarks}/` subdirectories. We flattened in execution because:

- Test names are already unique workspace-wide (`dqn_integration`, `evaluator_smoke`, etc.).
- Avoiding `[[test]]` Cargo.toml entries keeps the manifest simpler.
- `tests/baselines/` (a CSV-only fixture directory used by the DQN bench) is preserved as a non-test data directory under `crates/rlevo/tests/`.

## Consequences

**Positive:**

- **Strict-DAG dev-dep graph.** No subcrate dev-deps another subcrate. Combined with [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md), the workspace has zero cycles in either prod or dev edges.
- **Single discovery point for users.** `cargo run -p rlevo --example <name>` is the canonical way to run any demo, matching the *"use `rlevo` for the full API"* contract on every internal crate.
- **Cross-crate tests verify the umbrella's re-export surface.** Tests written against `rlevo`'s deps automatically catch regressions in the prelude or module re-exports.
- **Tighter dev-dep cones in subcrates.** `rlevo-benchmarks`'s `[dev-dependencies]` shrinks to just `approx`. `rlevo-evolution`'s drops `rlevo-environments`. `rlevo-reinforcement-learning`'s drops `tracing-subscriber`.
- **Names unchanged.** Every relocated example and test keeps its name; CI scripts and READMEs that reference `--example dqn_cart_pole` continue to work after the `-p` flag is updated to `rlevo`.

**Negative / accepted costs:**

- **Subcrates lose stand-alone runnability.** `cargo run -p rlevo-environments --example cartpole_random` no longer works. Acceptable per the *"internal crate"* descriptions.
- **The umbrella `Cargo.toml` grows.** ~30 `[[example]]` entries and a substantial `[dev-dependencies]` block now live in `crates/rlevo/Cargo.toml`. Localized to one file; not a maintenance burden.
- **One in-crate unit test in `rlevo-evolution::fitness` was rewritten** to drop its `rlevo_environments::landscapes::sphere::Sphere` import. The test now uses a 5-line local `Landscape` impl. Coverage is unchanged; the removed import was incidental.

**Neutral:**

- `[[bench]]` entries stay in their owning crates. They measure single-crate hot paths; centralizing them would obscure the boundary. The RL benches' `rlevo-environments` dev-dep is the only non-`approx`/`criterion` dev-dep that survives in any subcrate post-refactor — appropriate, since benchmark setup needs concrete envs.
- `tests/baselines/dqn_cartpole.csv` moved to `crates/rlevo/tests/baselines/`. Currently referenced only in a doc comment in `dqn_bench.rs`; no test code reads it. If a future bench wants to load it, the path is updated then.

## Alternatives considered

**Keep examples per-crate, only move cross-crate tests.** Rejected. Half-fixes the discovery problem (users still hunt in four crates for envs/rl/evo demos) and leaves the `rlevo-environments` dev-dep edges in `rlevo-evolution` and `rlevo-reinforcement-learning` intact. The cycle would still be broken (only `rlevo-benchmarks/tests/evaluator_smoke.rs` was load-bearing for the cycle), but the manifest tangles remain.

**Move *all* tests, including crate-internal ones, to the umbrella.** Rejected. Tests like `rlevo-environments/tests/grids_solvable.rs` exercise one crate's invariants. Putting them in the umbrella would force `cargo test -p rlevo-environments` to skip them, and would break the *"each crate's tests live next to its source"* convention that ADR 0003 already established for `rlevo-reinforcement-learning/tests/integration_test.rs` (since relocated under this ADR, but only because it crosses crates).

**Use `[[test]]` Cargo.toml entries with subdirectory paths** (`tests/rl/dqn_integration.rs`, etc.). Rejected for execution simplicity — flat layout works without manifest entries, and naming is already unique. Revisit if the `tests/` directory grows to a size where grouping helps navigation.

## References

- ADR 0001 — `keep-environments-and-benchmarks-separate` — load-bearing prod-dep boundary preserved here.
- ADR 0002 — `collapse-evolution-traits-into-rlevo-evolution`.
- ADR 0003 — `collapse-rl-modules-into-rlevo-reinforcement-learning` — established the *"each crate's tests live with its source"* default that this ADR refines for the cross-crate case.
- [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md) — companion ADR; together they close the dependency-graph cleanup.
- `crates/rlevo/examples/{envs, evo, rl, benchmarks}/` — new home for examples.
- `crates/rlevo/tests/` — new home for cross-crate integration tests.
- `crates/rlevo/Cargo.toml` — `[dev-dependencies]`, `[[example]]` listings.
- Plan file: `~/.claude/plans/yes-please-draft-the-crispy-newt.md`.
