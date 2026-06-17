---
project: rlevo
status: active
type: decision
date: 2026-04-25
tags: [adr, decision, architecture, crates]
---

# ADR 0001: Keep `rlevo-environments` and `rlevo-benchmarks` as separate crates

## Status

Active.

## Context

`rlevo-environments` (a library of concrete RL/optimization environments implementing
`rlevo_core::Environment`) and `rlevo-benchmarks` (a deterministic, parallelisable
evaluation harness with rayon, reporters, checkpointing, and seed splitting) were
built independently. The natural question was whether they should be merged into
a single crate. We considered three structural options:

1. **Merge into one crate.**
2. **Keep separate, with mutual `[dev-dependencies]`** so each could pull
   utilities from the other.
3. **Factor a third "glue" crate** to host adapters between the two.

Two related substructure decisions came up while answering the merge question:

- A `BenchAdapter` is needed somewhere so users do not hand-roll a `BenchEnv`
  wrapper for every concrete `Environment` (see
  `crates/rlevo-benchmarks/examples/tabular_bandit.rs` pre-cleanup, which
  wrote a 30-line wrapper just to thread `TenArmedBandit` into the harness).
- The harness's `BenchEnv::{reset, step}` methods were originally infallible,
  forcing the adapter to either panic on `EnvironmentError` or stringify it
  via a `BenchStep::error: Option<…>` field.

## Decision

**Keep `rlevo-environments` and `rlevo-benchmarks` as separate crates.**

To make the boundary practical, ship a feature-gated `bench` module inside
`rlevo-environments` that contains a `BenchAdapter<E, D, SD, AD>` and a small set of
preset `Suite` factories. The feature is **off by default** — base `rlevo-environments`
users do not pay the harness's dep cone, and base `rlevo-benchmarks` users do
not pay the physics dep cone. The umbrella `rlevo` crate already excludes
`rlevo-benchmarks`; this is preserved.

To keep the layering clean, `BenchEnv::{reset, step}` were migrated to return
`Result<_, BenchError>` where `BenchError` is a stringly-typed error enum local
to `rlevo-benchmarks`. The adapter converts upstream `EnvironmentError` via
`Display` so `rlevo-benchmarks` does **not** gain a runtime dep on `rlevo-core`.

## Consequences

**Positive:**

- **Disjoint dep cones.** `rlevo-environments` pulls `rapier2d` / `rapier3d` / `nalgebra`
  (heavy physics). `rlevo-benchmarks` pulls `rayon`, `tracing`, optional
  `serde_json` / `ratatui`. Each consumer pays only for what they use; this
  matters most for users who wire `rlevo-benchmarks` against a non-`rlevo-environments`
  `BenchEnv` impl.
- **Different abstractions stay aligned with their audiences.** `rlevo-environments`
  exposes the const-generic-rich `Environment<D, SD, AD>` trait for compile-time
  dimensional correctness. `rlevo-benchmarks` deliberately keeps `BenchEnv`
  object-safe and free of const generics so trial-level rayon parallelism stays
  ergonomic.
- **Truthful error semantics.** `EnvironmentError` is recoverable in
  `rlevo-core`. The harness preserves that distinction instead of escalating to
  a panic, and reporters can now surface env errors separately from genuine
  programming-bug panics (which the existing `catch_unwind` boundary in
  `Evaluator::run_one_trial` still catches).
- **Mutual dev-dep tangle is avoided.** With the `bench` feature, `rlevo-environments`
  optionally depends on `rlevo-benchmarks` as a regular (not dev) dep — no
  mutual cycle.
- **The umbrella crate stays unchanged.** `rlevo` keeps publishing
  envs+rl+evolution+hybrid+utils without ever touching the benchmarking harness.

**Negative / accepted costs:**

- **Mixed-shape `Suite`s are deferred.** `Suite<E>` is monomorphic, so a
  cross-env suite (e.g. "all classic-control discrete-action") needs either a
  `Box<dyn BenchEnv<…>>` wrapper with a normalized obs/action shape, or a
  runner-level change to accept heterogeneous env vecs. Deferred to a follow-up
  ADR — v0.1 ships per-env suite factories
  (`cartpole_suite`, `pendulum_suite`, `ten_armed_bandit_suite`, …).
- **`BenchAdapter` carries `(D, SD, AD)` const generics on the struct.** Rust's
  E0207 (unconstrained generic params) requires this. Inference picks them up
  from the wrapped env at construction, so end-user code reads
  `BenchAdapter::new(env)` without turbofish — paid once in adapter
  implementation complexity, not in caller ergonomics.
- **Trait migration breakage was wider than scoped.** Every `BenchEnv` impl
  in the workspace (16 sites in `rlevo-evolution` algorithm code, plus the in-
  tree examples and tests) needed an `Ok(...)` wrap, and every direct
  `harness.step(()).done` / `harness.reset()` call site needed `.unwrap()`.
  The migration was mechanical but touched ~25 files outside the originally
  scoped `rlevo-envs` / `rlevo-benchmarks`.

**Neutral:**

- The optimization fitness landscapes (`Sphere`, `Ackley`, `Rastrigin`) live
  in `rlevo-environments::landscapes` (formerly `::benchmarks` — renamed to remove
  collision with the `rlevo-benchmarks` crate name). They remain in
  `rlevo-environments` because they implement the `FitnessEvaluable`-shaped contract
  and ship alongside the envs they conceptually substitute for in
  optimizer-on-landscape comparisons.

## Alternatives considered

**Merge into one crate.** Rejected: would either force `BenchEnv` to thread
const generics through (defeating its object-safety design) or force every
consumer to compile both dep cones. The umbrella `rlevo` crate would also have
to either pull `rlevo-benchmarks` (forcing rayon onto every `rlevo` consumer) or
selectively re-export, which becomes awkward.

**Mutual `[dev-dependencies]`.** Rejected: works for cargo (dev-deps don't form
a normal-build cycle) but is fragile — `cargo doc`, IDE indexing, and
`cargo test -p rlevo-environments` all start pulling the harness for examples that just
want a logger. A small `rlevo-utils::log` helper or directly calling
`tracing_subscriber::fmt().init()` covers the actual ergonomics need without
the cycle smell.

**Third `rlevo-bench-glue` crate.** Rejected for v0.1: a feature-gated module
inside `rlevo-environments` adds zero new package metadata to maintain and the same
dep-isolation property. Revisitable if the adapter+suites surface grows large
enough to warrant its own README and versioning story.

**`BenchEnv` stays infallible; adapter panics on `EnvironmentError`.** Rejected:
loses the structured-error fidelity that `rlevo-core` deliberately exposes,
fails under `panic = "abort"` build profiles, and prevents `BenchEnv` wrappers
(e.g. a future `TimeLimit`-style harness wrapper) from inspecting recoverable
errors. The migration cost (one `match` arm in `run_one_trial`, `Ok(...)`
boilerplate per impl) was small enough to absorb up-front rather than
litigate later.

## References

- Conversation 2026-04-25 (planning + execution).
- `crates/rlevo-environments/Cargo.toml` — physics dep cone (`rapier2d` /
  `rapier3d` / `nalgebra`) and the new `bench` feature.
- `crates/rlevo-benchmarks/Cargo.toml` — harness dep cone (`rayon`,
  `tracing`, optional `serde_json` / `ratatui`).
- `crates/rlevo-benchmarks/src/env.rs` — `BenchEnv` trait + `BenchError`.
- `crates/rlevo-environments/src/bench/{mod,adapter,suites}.rs` — feature-gated
  glue.
- `crates/rlevo/Cargo.toml` — umbrella crate that excludes
  `rlevo-benchmarks`.
- Plan file: `~/.claude/plans/create-a-plan-to-steady-narwhal.md`
  (build artefact; not load-bearing for the decision).
