---
project: rlevo
status: active
type: decision
date: 2026-04-28
tags: [adr, decision, architecture, crates, rlevo-core, rlevo-benchmarks, rlevo-evolution, rlevo-environments]
---

# ADR 0004: Move bench trait surface into `rlevo-core`

## Status

Active. Partially reverses [0002-collapse-evolution-traits-into-rlevo-evolution](0002-collapse-evolution-traits-into-rlevo-evolution.md) by reintroducing the `rlevo-evolution → rlevo-core` edge — but for a different trait surface than the one ADR 0002 cut.

**Update (2026-07-06):** decision point #6's "keep the local splitmix64 mixer" and the corresponding "Neutral" consequence (line 95) are superseded by ADR [0033](0033-share-splitmix64-mixer-across-core-and-evolution.md), which dedupes the mixer into a single `pub` `rlevo_core::util::seed::splitmix64`. All other decisions in this ADR remain active.

## Context

A 2026-04-28 audit of the workspace dependency graph found two related defects:

1. **A circular edge through dev-dependencies.** `rlevo-environments` had an optional prod dep on `rlevo-benchmarks` (the `bench` feature, used by `src/bench/{adapter,landscape}.rs`). `rlevo-benchmarks` then dev-depended on `rlevo-environments` *with that feature on*, exclusively to power `tests/evaluator_smoke.rs`. Cargo tolerates the cycle through dev-dependencies but it complicates feature resolution and makes the graph confusing for readers.

2. **A misplaced production dep.** `rlevo-evolution` had a non-optional prod dep on `rlevo-benchmarks`. The actual usage was a tiny trait surface (~184 lines):

   | Source (`rlevo-benchmarks/src/`)         | Symbols                                                     |
   | ---------------------------------------- | ----------------------------------------------------------- |
   | `agent.rs`  (61 lines)                   | `BenchableAgent`, `FitnessEvaluable`, `Landscape`           |
   | `env.rs`    (42 lines)                   | `BenchEnv`, `BenchError`, `BenchStep`                       |
   | `seed.rs`   (81 lines)                   | `SeedStream`, splitmix64 fan-out                            |

   The harness machinery (~550 lines: `evaluator`, `suite`, `storage`, `report`, `checkpoint`, `metrics`, `reporter` and the `rayon` + optional `ratatui`/`serde_json` cone) was unused by `rlevo-evolution`. Worse, `rlevo-evolution/src/rng.rs` re-implemented splitmix64 locally with a comment claiming this avoided the `rlevo-benchmarks` dep — yet the dep was already declared.

The ADR 0002 test was *"≥1 stable downstream consumer with shared vocabulary"*. ADR 0002 removed `rlevo-core` from `rlevo-evolution` because the EA traits in core (`Fitness`, `MultiFitness`, `GenomeKind`) had zero consumers. The bench traits do not have that problem: they have two production consumers (`rlevo-evolution`, `rlevo-environments[bench]`) and one harness consumer (`rlevo-benchmarks` itself). They pass the ADR 0002 test cleanly.

A third drift was visible: `BenchError` was stringly-typed (`Reset(String)`, `Step(String)`) with a doc comment justifying the design as *"`rlevo-benchmarks` does not depend on `rlevo-core`, so a typed bridge from `rlevo_core::EnvironmentError` would invert the dependency direction."* The justification was already stale — `rlevo-benchmarks/Cargo.toml` declares a prod dep on `rlevo-core` — and irrelevant once the trait moves.

## Decision

**Move the three trait modules from `rlevo-benchmarks` to `rlevo-core` and switch `BenchError` to wrap `EnvironmentError` typedly.**

Concretely:

1. **Move (verbatim, via `git mv` to preserve history):**
   - `crates/rlevo-benchmarks/src/env.rs`   → `crates/rlevo-core/src/evaluation.rs`
   - `crates/rlevo-benchmarks/src/agent.rs` → `crates/rlevo-core/src/fitness.rs`
   - `crates/rlevo-benchmarks/src/seed.rs`  → `crates/rlevo-core/src/util/seed.rs`
2. **Promote** `crates/rlevo-core/src/util.rs` → `crates/rlevo-core/src/util/mod.rs` so `util::seed` can sit alongside the existing `combinations()` helper.
3. **Move the `Metric` enum and `MetricsProvider` trait** from `rlevo-benchmarks/src/metrics/mod.rs` into `rlevo-core::fitness` (alongside `BenchableAgent`, which returns `Vec<Metric>` from its `emit_metrics` hook). The harness keeps the `metrics::{core, ea, rl}` aggregators and re-exports `Metric` / `MetricsProvider` from `rlevo-core::fitness` so internal harness code paths keep working unchanged.
4. **Switch `BenchError` to typed `EnvironmentError`:**
   ```rust
   pub enum BenchError {
       Reset(#[source] EnvironmentError),
       Step(#[source] EnvironmentError),
   }
   ```
   `Clone` is dropped from `BenchError` (`EnvironmentError` is not `Clone` because of `IoError(std::io::Error)`). A grep confirmed no consumer cloned `BenchError` — it was only constructed in `rlevo-environments::bench::adapter`.
5. **Backward-compat shim modules** in `rlevo-benchmarks/src/lib.rs`: `pub mod env`, `pub mod agent`, `pub mod seed` re-export the relocated symbols from core. External consumers (existing examples, in-tree tests) continue to compile against `rlevo_benchmarks::env::*`, etc., without import churn.
6. **`rlevo-evolution`**: drop the `rlevo-benchmarks` prod dep, add the `rlevo-core` prod dep. Rewrite ~14 import sites from `rlevo_benchmarks::{agent, env, seed}` to `rlevo_core::{fitness, evaluation, util::seed}`. Keep `src/rng.rs`'s local splitmix64 mixer (its API — `(base, generation, SeedPurpose) → StdRng` — is distinct from `SeedStream::trial_seed`'s `(env_idx, trial_idx) → u64`); rewrite the docstring to drop the "lock-step with rlevo-benchmarks" framing. The two implementations now share an algorithm, not a crate.
7. **`rlevo-environments`**: rewrite `src/bench/{adapter,landscape,mod}.rs` to import the trait surface from `rlevo-core` instead of `rlevo-benchmarks`. The optional `rlevo-benchmarks` dep stays — `src/bench/suites.rs` needs `Suite` and `EvaluatorConfig`, which remain in the harness crate. Update the adapter's `BenchError` construction from `BenchError::Reset(e.to_string())` to `BenchError::Reset(e)`.

### Scope decision: keep the `rlevo-environments[bench] → rlevo-benchmarks` edge

The plan briefly considered eliminating the optional `rlevo-benchmarks` dep from `rlevo-environments`. That would require relocating `bench/suites.rs` (which constructs `Suite` factories tying `BenchAdapter`-wrapped envs to `EvaluatorConfig`). Suites belong adjacent to the env definitions they wrap; moving them elsewhere is a separate concern. The cycle between environments and benchmarks is killed by [0005-examples-and-cross-crate-tests-in-umbrella](0005-examples-and-cross-crate-tests-in-umbrella.md) (which removes the `rlevo-benchmarks → rlevo-environments` dev-dep), not by removing the optional prod-dep direction. Both directions are evaluated independently, and this ADR keeps the optional prod-dep direction intact.

## Consequences

**Positive:**

- **Strict-DAG dependency graph**, with all internal edges pointing up-stack:
  ```
  rlevo-core
     │
     ├──► rlevo-benchmarks
     ├──► rlevo-environments    [optional → rlevo-benchmarks for bench/suites.rs]
     ├──► rlevo-reinforcement-learning
     │
     └──► rlevo-evolution
              │
              └──► (re-uses rlevo-core)
                                │
  rlevo-hybrid ──► rlevo-core, rlevo-reinforcement-learning, rlevo-evolution
  ```
- **Typed errors at the harness boundary.** `BenchError::Reset(EnvironmentError)` preserves the typed upstream error; reporters that want to introspect `InvalidAction` vs `IoError` no longer have to parse a Display string.
- **No dep-cone bloat for `rlevo-evolution`.** Evolution now pulls just `rlevo-core` (which it already needed conceptually) instead of the harness's `rayon` + optional `ratatui`/`serde_json` cone.
- **Backward-compatible call sites.** `rlevo-benchmarks::{env, agent, seed}` continue to resolve via shim modules — no break for downstream code that imported through the harness crate.
- **Eliminates a stale comment.** The doc-comment justification *"rlevo-benchmarks does not depend on rlevo-core"* was already false; removing it removes future confusion.

**Negative / accepted costs:**

- **Mechanical refactor of ~14 import sites in `rlevo-evolution`.** Pure path swaps; no semantic edits.
- **Reverses ADR 0002's "no `rlevo-core` dep in `rlevo-evolution`" stance.** The reversal is principled: ADR 0002's dead traits were genuinely dead. The bench traits are not.
- **`Clone` removed from `BenchError`.** Acceptable: nothing cloned it. Adapters may need to refactor if they want to fan a single error to multiple reporters, but no current consumer does.

**Neutral:**

- The harness still owns `evaluator`, `suite`, `storage`, `report`, `checkpoint`, `metrics::{core, ea, rl}`, `reporter`. The boundary between *trait surface* (now in core) and *runner* (still in benchmarks) is now the load-bearing boundary; ADR 0001's envs-vs-benchmarks separation continues to hold.
- The local splitmix64 mixer in `rlevo-evolution/src/rng.rs` stays. Its API is distinct from `SeedStream` (different inputs, different output type). They share an algorithm — a consequence of using a well-known mixer — not a dependency.

## Alternatives considered

**Create a new `rlevo-bench-traits` crate.** Considered as a counter-proposal: extract the ~184 lines of trait surface into a separate workspace member with its own minimal dep cone (`rand` + `thiserror`). Rejected:

- **No dep cone to protect.** The traits already pull only `rand` + `thiserror`; `rlevo-core` already carries both.
- **Conceptual fit.** `BenchEnv` is a narrower `Environment`; `FitnessEvaluable`/`Landscape` are reward-shaped; `SeedStream` is a generic RNG utility. They sit at the same conceptual level as items already in core (`Environment`, `Reward`, `util`).
- **Workspace hygiene.** Tiny trait crates earn their place when they prevent cycles or carry distinct semantics. Here they would just add a 7th internal crate to maintain. Subcrate descriptions already say *"internal crate — use `rlevo` for the full API"* — there is no third-party publication concern.
- **Lower friction.** One new module triplet under `rlevo-core` is simpler than a new workspace member with its own `Cargo.toml`, `lib.rs`, lints config, and CI lane.

**Keep `BenchError` stringly-typed.** Rejected. The justification was already stale; the `Clone` derive on `BenchError` was unused; consumers that want structured error data are better served by typed errors. The migration cost was negligible (one `e.to_string()` site).

**Move `bench/suites.rs` out of `rlevo-environments` to eliminate the optional `rlevo-benchmarks` prod dep entirely.** Deferred. Suites are tightly coupled to the env constructors they wrap; relocating them would either (a) push them into the harness, inverting the natural data flow (the harness should not know about specific envs), or (b) push them into the umbrella `rlevo` crate, where they would be one more thing to export. Neither is clearly better than the current placement, and the cycle is already broken by ADR 0005 alone.

## References

- ADR 0001 — `keep-environments-and-benchmarks-separate` — established the envs ↔ benchmarks boundary.
- ADR 0002 — `collapse-evolution-traits-into-rlevo-evolution` — removed the dead EA traits from `rlevo-core` and severed `rlevo-evolution → rlevo-core`. This ADR partially reverses that severance for a different (live) trait surface.
- ADR 0003 — `collapse-rl-modules-into-rlevo-reinforcement-learning` — established the conservative dead-code policy retained here.
- [0005-examples-and-cross-crate-tests-in-umbrella](0005-examples-and-cross-crate-tests-in-umbrella.md) — companion ADR; together they break the `benchmarks ↔ environments` cycle.
- `crates/rlevo-core/src/{evaluation,fitness}.rs`, `crates/rlevo-core/src/util/seed.rs` — new homes.
- `crates/rlevo-benchmarks/src/lib.rs` — backward-compat shim modules.
- Plan file: `~/.claude/plans/yes-please-draft-the-crispy-newt.md`.
