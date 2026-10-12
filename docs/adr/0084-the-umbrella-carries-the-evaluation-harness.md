---
project: rlevo
status: active
type: decision
date: 2026-10-12
tags: [adr, decision, architecture, crate-boundaries, umbrella, benchmarks, environments, examples, breaking-change]
---

# ADR 0084: The umbrella carries the evaluation harness

## Status

**Accepted (2026-10-12).**

Supersedes three things in ADR
[0001](0001-keep-environments-and-benchmarks-separate.md): the sentence
"The umbrella `rlevo` crate already excludes `rlevo-benchmarks`; this is
preserved", the consequence "The umbrella crate stays unchanged", and the
preset `Suite` factories of the feature-gated `bench` module. The decision
ADR 0001 is named for, that `rlevo-environments` and `rlevo-benchmarks` are
separate crates, stands.

Supersedes the `rlevo-benchmarks` row of the example scope rule in ADR
[0012](0012-split-heavy-examples-into-rlevo-examples.md), "If an example
imports from `rlevo-benchmarks` for any reason ... it belongs in
`rlevo-examples`". The crate `rlevo-examples` and the three-tier test
placement rule stand.

## Context

`rlevo-benchmarks` is the evaluation harness: `Suite`, `Evaluator`, the
metrics and the reporters. It is how an agent is run over an environment
and a result is reported, for a shipped environment and for one a user
wrote. The umbrella crate did not depend on it. `rlevo` listed the harness
under `[dev-dependencies]`, so the workspace's own tests and examples
reached it and a consumer of `rlevo` did not. The harness's own description
on crates.io reads "internal crate — use `rlevo` for the full API", which
sent a reader to a crate that did not offer it.

Three measurements, taken with the outside-in probes, which build a crate
against the workspace as packaged from a directory outside the checkout:

- `byoe-1-thermostat` plays an author who brings an environment and runs it
  through `Evaluator::run_suite`. Its manifest had to name
  `rlevo-benchmarks` beside `rlevo`. The probe's step 2 reported
  `rlevo alone is insufficient; 2 additional crate(s) required by hand:
  rand, rlevo-benchmarks`.
- `consumer-umbrella` depends on `rlevo` at default features. It reported
  two published crates its graph did not reach: `rlevo-benchmarks` and
  `rlevo-metrics-registry`.
- The umbrella's `viz-tui` feature is `rlevo-benchmarks/tui`. A feature
  that names a dev-dependency does nothing for a consumer:
  `cargo tree -e normal -p rlevo --features viz-tui` had no
  `rlevo-benchmarks` line, while `--features viz-report` had one, reached
  through `rlevo-environments/record`. Two features documented as a pair
  worked by different mechanisms, and one of them not at all outside the
  workspace.

ADR 0001 kept the harness out of the umbrella for its dependency cone:
pulling it would mean "forcing rayon onto every `rlevo` consumer". That
does not hold now. `rayon`, `tracing`, `thiserror`, `serde`,
`serde_json`, `rand` and `rand_distr`, which are the harness's
dependencies at its default features, are all in `rlevo`'s normal
dependency tree without it.

ADR 0001 also shipped preset `Suite` factories in `rlevo-environments`
behind a `bench` feature, with an adapter type. ADRs
[0076](0076-trial-seam-splits-machinery-from-trial-shape.md) and
[0077](0077-remove-benchenv.md) removed the adapter: `run_suite` binds
on `Environment`, and an environment is registered as itself. What was left
of the presets was three functions, each one expression:

```rust
Suite::new("ten-armed-bandit", cfg)
    .with_env("ten-armed-bandit-default", TenArmedBandit::with_seed)
```

`cartpole_suite` and `pendulum_suite` had no caller anywhere in the
workspace. `ten_armed_bandit_suite` had one, an example. With them gone the
`bench` module holds one thing, the `RecordedEnvFamily` impls, which are
already behind the `record` feature. A `bench` feature that turns on
nothing but a dependency is a second name for `record`.

## Decision

1. **`rlevo` depends on `rlevo-benchmarks` unconditionally and re-exports
   it as `rlevo::harness`.** The dependency is at the harness's default
   features. It is not behind a feature of the umbrella.
2. **The re-export is named `harness`**, for what the crate is. The crate
   is still named `rlevo-benchmarks` when this is decided, so the line
   reads `pub use rlevo_benchmarks as harness;`. The path a consumer writes
   is chosen once, before any release carries it; renaming the crate to
   match is a separate decision and changes no consumer path. The names
   not taken: `benchmarks` repeats a crate name that already needs a
   warning in `docs/rules.md` against reading it as cargo's `[[bench]]`;
   `bench`, as a plain intra-doc link, fails the documentation build with
   "`bench` is both a crate and an attribute macro", the macro being the
   built-in `#[bench]`; and `evaluation` is a module of `rlevo-core`.
3. **The three preset `Suite` functions are deleted**, and the module
   `rlevo_environments::bench::suites` with them. An environment is
   registered with `Suite::with_env`, which takes a constructor from a
   seed.
4. **The `bench` feature of `rlevo-environments` is removed.** `record` is
   the one feature under which `rlevo-environments` depends on
   `rlevo-benchmarks`. The module path `rlevo_environments::bench::family`
   and everything in it are unchanged.
5. **One exception to the layering remains and is named.** A production
   crate other than the umbrella does not depend on the harness, except
   `rlevo-environments` under its off-by-default `record` feature. That
   edge exists for the recording stack and goes when that stack does, which
   is a later decision and not this one.
6. **An example's home is decided by what it needs, not by whether it
   names the harness.** An example lives in `crates/rlevo/examples/` when
   everything it uses is reachable through `rlevo` at default features. It
   lives in `crates/rlevo-examples/` when it needs one of the harness's
   `tui`, `record` or `report` features, or a crate that is not published.
   The two harness examples, `ga_rastrigin` and `tabular_bandit`, move to
   `crates/rlevo/examples/harness/` and spell every path through `rlevo`,
   as a consumer does.
7. **`docs/rules.md` follows.** Section 1's inventory lists what the
   umbrella includes; section 5's placement table and section 11's example
   scope rule carry Decision 6; section 10's rule on visualization
   dependencies says what the umbrella may link.

## Consequences

**Positive.**

- `cargo add rlevo` reaches `Suite` and `Evaluator`. The thermostat
  probe's manifest loses its `rlevo-benchmarks` line, its harness file
  names `rlevo::harness`, and its step 2 reports one hand-added crate,
  `rand`, where it reported two.
- A consumer of `rlevo` reaches all eight published crates.
  `consumer-umbrella` no longer lists any as unreached.
- `viz-tui` reaches a consumer. `cargo tree -e normal -p rlevo --features
  viz-tui` now has a `rlevo-benchmarks` line, as `--features viz-report`
  does, and the two features work the same way.
- Less surface. From `tools/public-surface.sh`, all features on:
  `rlevo-environments` loses three functions and one module; `rlevo` gains
  one re-export. `rlevo-environments` loses a feature.

**Costs.**

- "Harness" names two things until the evolution crate's generation
  drivers are renamed. `EvolutionaryHarness` and `CoEvolutionaryHarness`
  drive a strategy's generations and are not this crate. Their rename is
  its own decision; until it lands, `rlevo::harness` and
  `rlevo::evo::strategy::EvolutionaryHarness` sit side by side.
- Every consumer of `rlevo` compiles the harness. The umbrella's dependency
  cone goes from 207 packages to 209 (`cargo tree -e normal -p rlevo
  --prefix none`, unique names, less the crate itself), and the two are
  `rlevo-benchmarks` and `rlevo-metrics-registry`. No third-party package
  enters. A cold `cargo build` of a crate that depends on `rlevo` took
  34.3 s before and 34.0 s after (the `consumer-umbrella` probe, one run
  each, `dev` profile, rustc 1.99.0, Apple M2 Pro, 12 cores), which is no
  difference this measurement can see.
- A breaking change in `rlevo-environments`. A manifest that asks for
  `features = ["bench"]` no longer resolves; `record` is the feature that
  remains, and a build that only wanted the presets needs no feature.
  `cartpole_suite`, `pendulum_suite` and `ten_armed_bandit_suite` are gone;
  the replacement is the `Suite::with_env` call each one made.
- The umbrella is a production crate that depends on `rlevo-benchmarks`,
  which is the one crate allowed visualization dependencies. They stay
  behind the harness's `tui`, `report` and `record` features, none of
  which the umbrella turns on by default. `cargo tree -p rlevo -e normal`
  has no line for `ratatui`, `crossterm`, `leptos`, `axum` or
  `wasm-bindgen`. That check is on the resolved graph, as `docs/rules.md`
  section 10 requires, and it is what keeps the rule true from here on.

**Unchanged.**

- The publish order. `rlevo-benchmarks` still precedes
  `rlevo-environments`, whose optional dependency on it carries a version,
  and both precede `rlevo`.
- The harness's own features and API. The two examples print the same
  result lines after the move as before it.

**How it was checked.** The four outside-in probes were run on the branch;
their output is quoted above. The workspace compiles with all targets and
all features, and each library compiles alone. The fast test tier passes.
Naming the re-export `bench` and building the umbrella's documentation
reproduces the error in Decision 2.

## Alternatives considered

**Make the harness an optional dependency behind an umbrella feature.**
Rejected. `rlevo::harness` would exist or not depending on a feature,
and the compile error for a missing path names neither the feature nor the
crate. Evaluating an environment without recording it, the common case,
would still be unreachable by default. The saving would be two first-party
packages.

**Keep the presets, and move them into the harness behind a `fixtures`
feature** with an optional dependency from `rlevo-benchmarks` on
`rlevo-environments`. Rejected. It reverses an edge and the publish order,
and it needs a lockstep the manifests cannot express: the umbrella's
`box2d` feature would have to turn on a matching feature in the harness,
because a crate cannot condition on another crate's feature. All of that
would carry three one-line functions, two of them unused.

**Move the `RecordedEnvFamily` impls out of `rlevo-environments` now**, so
that no configuration of it depends on the harness. Not done here. They
serve the recording stack, whose future is being decided separately, and
moving them before that decision moves them twice.

**Leave the harness examples in `rlevo-examples`.** Rejected. They would
keep teaching `rlevo_benchmarks::` paths and a second manifest entry to a
reader who needs neither.

## References

- ADR [0001](0001-keep-environments-and-benchmarks-separate.md), the
  separation that stands and the clauses superseded.
- ADR [0012](0012-split-heavy-examples-into-rlevo-examples.md), the example
  scope rule amended by Decision 6.
- ADR [0013](0013-metrics-only-live-tui.md), the two visualization
  products the umbrella's `viz-tui` and `viz-report` features forward.
- ADRs [0076](0076-trial-seam-splits-machinery-from-trial-shape.md) and
  [0077](0077-remove-benchenv.md), which let an environment be
  registered as itself.
- ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md),
  the test the presets fail.
- `docs/rules.md` sections 1, 5, 10 and 11.
- Code: `crates/rlevo/Cargo.toml` and `crates/rlevo/src/lib.rs` (the
  dependency and the re-export); `crates/rlevo-environments/Cargo.toml`
  (the `record` feature); `crates/rlevo/examples/harness/`;
  `xtask/probes/byoe-1-thermostat/` (the probe whose manifest is the
  measurement).
