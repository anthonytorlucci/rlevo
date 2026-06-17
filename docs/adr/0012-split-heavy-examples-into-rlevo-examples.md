---
project: rlevo
status: active
type: decision
date: 2026-05-31
tags: [adr, decision, architecture, crates, examples, visualisation, recording, reporting, rlevo-examples, rlevo-benchmarks, testing]
---

# ADR 0012: Split non-library examples into `rlevo-examples`; formalize the three-tier test placement rule

## Status

Active. Supersedes the *examples* portion of [0005-examples-and-cross-crate-tests-in-umbrella](0005-examples-and-cross-crate-tests-in-umbrella.md). The *test placement* rules in ADR 0005 are carried forward here with explicit wording; ADR 0005's test rules are considered superseded by this document to avoid split authority.

## Context

ADR 0005 consolidated all examples into `crates/rlevo/` to fix dependency cycles and create a single discovery point. That solved the structural problem. The cost, accepted at the time, was that `crates/rlevo/Cargo.toml` accumulated ~60 `[[example]]` entries, a substantial `[dev-dependencies]` block, and a mix of very different example audiences — a beginner looking at `cartpole_random` and a developer instrumenting a training run with `record_evolution_sphere_with_client` are not the same reader.

Two subsequent developments sharpened the problem:

1. **Visualisation examples are heavyweight in a different category.** The viz/record/report examples in `examples/viz/` require features `viz-tui`, `viz-record`, and `viz-report`, which pull in `ratatui`, `axum`, `leptos`, `parking_lot`, and the full `rlevo-benchmarks` reporting stack. These are *application-tier* programs — they embed servers, spawn threads, and write to disk — not library demonstrations. Grouping them with `cartpole_random.rs` blurs that distinction.

2. **Harness examples belong in the same category.** `examples/benchmarks/ga_rastrigin.rs` and `examples/benchmarks/tabular_bandit.rs` import `rlevo-benchmarks` directly — a crate outside the five library sub-crates (`rlevo-core`, `rlevo-environments`, `rlevo-evolution`, `rlevo-reinforcement-learning`, `rlevo-hybrid`). An example that reaches beyond the library tier is not a library demonstration; it is an application built on top of the library.

3. **The `[dev-dependencies]` block in `crates/rlevo/Cargo.toml` is the wrong home for application deps.** `parking_lot`, `tracing-subscriber`, `rand_distr`, and `criterion` are there solely because the viz examples need them. They inflate the development build surface of the library crate.

The fix is to add one crate above `rlevo` in the DAG: `crates/rlevo-examples`. It is a pure consumer with no library surface of its own. The umbrella `crates/rlevo/` retains only examples that demonstrate the five library sub-crates exclusively, keeping its manifest clean.

### Scope rule: what belongs in `rlevo` vs `rlevo-examples`

An example belongs in `crates/rlevo/examples/` if and only if it imports exclusively from the five library sub-crates:

| Crate | Allowed in `rlevo/examples/`? |
|---|---|
| `rlevo-core` | Yes |
| `rlevo-environments` | Yes |
| `rlevo-evolution` | Yes |
| `rlevo-reinforcement-learning` | Yes |
| `rlevo-hybrid` | Yes |
| `rlevo-benchmarks` | **No → `rlevo-examples`** |
| Any viz/record/report feature dep | **No → `rlevo-examples`** |

If an example imports from `rlevo-benchmarks` for *any* reason — harness invocation, suite construction, agent trait, recording, reporting — it belongs in `rlevo-examples`.

### Opportunity: codify the three-tier test placement rule

ADR 0005 established test placement implicitly through the table of decisions. That rule is now referenced across multiple ADRs and session logs without a canonical single definition. This ADR makes it explicit so future contributors have one authoritative source.

## Decision

### 1. Create `crates/rlevo-examples` as a new workspace member

`rlevo-examples` is a thin crate — no public library surface of its own. Its sole purpose is to host examples that reach beyond the five library sub-crates: anything that imports `rlevo-benchmarks` or the viz/record/report feature stack.

Examples are organised by domain (the subject of the example), not by viz tier:

```
crates/rlevo-examples/
  Cargo.toml
  src/lib.rs        ← doc-only; satisfies Cargo's "at least one target" requirement
  examples/
    rl/             ← tui_ppo_cartpole, record_ppo_cartpole, report_ppo_cartpole_with_client
    evolution/      ← record_sphere_landscape, report_sphere_landscape, record_evolution_sphere
    grids/          ← record_grids, report_grids_with_client
    toy_text/       ← record_toy_text, report_toy_text_with_client
    locomotion/     ← record_inverted_pendulum, report_inverted_pendulum_with_client
    box2d/          ← record_lunar_lander, report_lunar_lander_with_client
    harness/        ← ga_rastrigin, tabular_bandit  (rlevo-benchmarks harness demos)
    common/         ← shared helper modules (ppo_cartpole.rs)
```

`crates/rlevo-examples/Cargo.toml` dependencies:

| Section           | Crates                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------- |
| `[dependencies]`  | `rlevo` (path dep, with `features = ["viz-tui", "viz-record", "viz-report", ...]`)      |
| `[dev-dependencies]` | `parking_lot`, `tracing-subscriber`, `rand`, `rand_distr`                            |

`rlevo-examples` does **not** appear as a dep of any other crate. It is a leaf node in the workspace DAG.

### 2. `crates/rlevo/` retains lightweight examples, tests, and benches

The umbrella crate keeps `examples/`, `tests/`, and `benches/`, but scoped strictly to:

| Directory        | Contents                                                                         |
| ---------------- | -------------------------------------------------------------------------------- |
| `examples/envs/` | Single-env random/scripted rollouts — demonstrate `rlevo-environments` API       |
| `examples/evo/`  | Single-landscape showcases — demonstrate `rlevo-evolution` API                   |
| `examples/rl/`   | Single-algorithm training loops — demonstrate `rlevo-reinforcement-learning` API |
| `tests/`         | Cross-crate integration tests (see rule below)                                   |
| `benches/`       | Cross-crate throughput benchmarks (currently `cartpole_record.rs`)               |

`examples/benchmarks/` and `examples/viz/` are removed entirely from `crates/rlevo/`. Any example that imports `rlevo-benchmarks` — whether for harness invocation or viz/record/report — lives in `rlevo-examples` instead.

The `[dev-dependencies]` block in `crates/rlevo/Cargo.toml` is pruned to only what the retained examples and tests actually require. `parking_lot`, `tracing-subscriber`, and `rand_distr` are removed if no remaining file imports them.

### 3. Three-tier test placement rule (canonical definition)

| Test kind                                                              | Lives in                                           | Discovered by                      |
| ---------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------- |
| Unit tests (`#[cfg(test)]` modules inside source files)                | The owning source file, in the owning crate        | `cargo test -p <crate>`            |
| Single-crate integration tests (exercises one crate's public surface)  | `<crate>/tests/` in the owning crate               | `cargo test -p <crate>`            |
| Cross-crate integration tests (exercises two or more crates together)  | `crates/rlevo/tests/` (flat, no subdirectories)    | `cargo test -p rlevo`              |

Supplementary rules:

- **Unit tests stay in-source.** A `#[cfg(test)]` block in `state.rs` is not moved to `tests/state.rs` just because a file grows large. In-source placement keeps the test adjacent to the invariant it checks.
- **Single-crate integration tests belong to their crate.** `rlevo-environments/tests/grids_solvable.rs` exercises only `rlevo-environments` internals and must not migrate to the umbrella. `cargo test -p rlevo-environments` must run it without pulling the full umbrella cone.
- **Cross-crate tests are flat in `crates/rlevo/tests/`.** Cargo discovers top-level `.rs` files; subdirectories are helper modules. Do not add subdirectory structure unless paired with explicit `[[test]]` Cargo.toml entries, and prefer not to — test names are unique workspace-wide.
- **`[[bench]]` entries stay in their owning crate** when measuring single-crate hot paths. Cross-crate throughput benchmarks (e.g. `cartpole_record.rs`) live in `crates/rlevo/benches/`.

### 4. Workspace `Cargo.toml` membership

`crates/rlevo-examples` is added to the workspace `[workspace] members` list. It is **not** a member of the `[workspace] default-members` list, so `cargo build` and `cargo test` without `-p` do not build it by default. Developers opt in with `cargo run -p rlevo-examples --example <name>`.

## Consequences

**Positive:**

- **`crates/rlevo/Cargo.toml` is dramatically smaller.** The ~14 viz `[[example]]` entries and their required-features chains move to `rlevo-examples`. The `[dev-dependencies]` block loses all application-tier deps.
- **`rlevo-examples` can evolve independently.** Adding a new viz demo, upgrading `axum`, or changing `tracing-subscriber` configuration touches only `rlevo-examples/Cargo.toml`, not the library manifest.
- **Audience separation is explicit.** A user looking for an env demo runs `-p rlevo`; a developer instrumenting a full training pipeline runs `-p rlevo-examples`. The naming communicates intent.
- **DAG remains a strict DAG.** `rlevo-examples` → `rlevo` → sub-crates. No new cycles are introduced.
- **Test placement rule is authoritative.** A single document answers where a new test file goes. Removes the ADR 0003 / ADR 0005 split-authority ambiguity.

**Negative / accepted costs:**

- **New crate in the workspace.** Workspace member count increases by one. Acceptable — `rlevo-examples` is simple (no `src/lib.rs`, no public API, no version constraint concerns).
- **`cargo run -p rlevo --example tui_ppo_cartpole` stops working.** The canonical command for viz examples becomes `-p rlevo-examples`. CI scripts, READMEs, and session notes that reference viz example invocations need updating. All other examples (`-p rlevo --example dqn_cart_pole`, etc.) are unaffected.
- **`examples/viz/common/` must move.** The shared helper modules used by viz examples currently live at `crates/rlevo/examples/viz/common/`. They move to `crates/rlevo-examples/examples/common/` and their `mod` declarations inside each example file are updated to the new relative path. No logic changes.

**Neutral:**

- Example *names* are unchanged. `record_ppo_cartpole`, `report_grids_with_client`, etc. keep their names; only the `-p` flag changes.
- The `cartpole_record` benchmark in `crates/rlevo/benches/` stays. It is a throughput benchmark on the recording tier that belongs in the umbrella, not in `rlevo-examples`, because it measures library performance rather than demonstrating application patterns.
- `[[test]]` names in `crates/rlevo/tests/` are unchanged. `cartpole_report_smoke` stays in the umbrella because it tests the library's report pipeline, not an application built on top of it.

## Alternatives considered

**Keep all examples in `crates/rlevo/`, annotated by audience.** Rejected. Comments in `Cargo.toml` do not enforce the distinction at compile or discovery time. The `[dev-dependencies]` bloat is structural, not cosmetic.

**Move *all* examples to `rlevo-examples`, make `rlevo` example-free.** Rejected. Lightweight examples like `cartpole_random.rs` are documentation for the library's public API. They compile as part of `cargo test -p rlevo --examples`, providing a low-cost regression check on the re-export surface. Stripping them from the umbrella removes that check without corresponding benefit.

**`rlevo-examples` as a workspace member in `default-members`.** Rejected. Including it in default builds forces everyone to compile the full viz stack (axum, leptos, ratatui) on every `cargo build`. The viz stack is opt-in by design (ADR 0008).

**Use `[[example]]` path entries with `../rlevo/examples/viz/` paths from `rlevo-examples`.** Rejected. Cross-crate path references in `[[example]]` entries are not supported by Cargo.

## References

- [0005-examples-and-cross-crate-tests-in-umbrella](0005-examples-and-cross-crate-tests-in-umbrella.md) — superseded (examples portion); test-placement rules carried forward here.
- [0008-three-tier-visualisation-ratatui-live-static-report](0008-three-tier-visualisation-ratatui-live-static-report.md) — defines the viz feature flags and the TUI / record / report tiers that motivate the `rlevo-examples` split.
- [0010-unify-on-parking-lot-across-viz-stack](0010-unify-on-parking-lot-across-viz-stack.md) — the `parking_lot` dep that currently lives in `crates/rlevo/[dev-dependencies]` migrates to `rlevo-examples/[dev-dependencies]`.
- `crates/rlevo-examples/` — new home for viz/record/report examples.
- `crates/rlevo/examples/{envs,evo,rl,benchmarks}/` — retained lightweight examples.
- `crates/rlevo/tests/` — canonical home for cross-crate integration tests.
