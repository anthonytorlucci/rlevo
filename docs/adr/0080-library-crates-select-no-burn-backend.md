---
project: rlevo
status: active
type: decision
date: 2026-10-10
tags: [adr, decision, architecture, burn, backend, features, dependency-graph, breaking-change, rlevo-evolution]
---

# ADR 0080: Library crates select no Burn backend; the leaf does

## Status

**Accepted (2026-10-10).**

Supersedes no ADR. It makes one existing invariant true where it had only been
written down: `docs/rules.md` and ADR
[0013](0013-metrics-only-live-tui.md) say no production crate depends on a
visualisation crate, and until this change every production crate did, through
a feature of Burn.

## Context

The workspace declared Burn once, and every crate inherited the line:

```toml
burn = { version = "0.21.0", features = ["wgpu", "train", "tui", "metrics", "flex"] }
```

Each library crate therefore built Burn with two backends, Burn's training-loop
crate, and Burn's default features. Four facts about that line follow. Package
counts are unique package names in `cargo tree -e normal -p <crate>`, less the
crate itself, on an Apple M2 Pro (12 cores, 32 GiB) with rustc 1.99.0.

**Every consumer paid for all of it.** `rlevo-core`, the contract crate an
environment author implements against, resolved 352 packages. A cold
`cargo build -p rlevo-core` took 43 to 45 seconds and about 295 CPU-seconds (three runs). The crate's own source is a
rounding error in that figure; the rest is Burn's GPU path (`wgpu`, `naga`,
`cubecl`), its CPU backend, and what `train` brings: `burn-train` and
`burn-dataset`, and through them an image and AV1 stack, SQLite, system
metrics, and a terminal UI.

**Nothing in the workspace used most of it.** No `.rs` file under `crates/` or
`xtask/` names `burn::train`, `burn::tui`, `burn::lr_scheduler`, a Burn
dataloader or a Burn metric. No library names a backend type either: library
code is generic over `B: Backend`, and `Flex`, `Wgpu` and `Autodiff<_>` appear
only in tests, benches, doctests, examples and the unpublished
`rlevo-test-support`. That is established by build, not by search: each
library compiles alone with Burn at `default-features = false, features =
["std"]`.

**The lever is `train`.** In Burn 0.21, `train = ["burn-train", "autodiff",
"dataset"]`, `tui = ["burn-train?/tui"]` and `metrics =
["burn-train?/sys-metrics"]`; Burn's own `default` includes
`burn-train?/default`, and `burn-train`'s default is `["sys-metrics", "tui",
"rl"]`. So with `train` on and defaults on, the `tui` and `metrics` strings in
the workspace line were redundant, and removing them alone would have changed
nothing.

**A written invariant was false in the resolved graph.** `ratatui` and
`crossterm` were in the normal dependency graph of all five library crates, of
`rlevo-benchmarks` with its `tui` feature off, and of `rlevo`. No manifest in
the workspace named them outside `rlevo-benchmarks`' feature gate, so every
manifest audit passed.

Separately, `rlevo-evolution` had a `custom-kernels` feature, on by default
there and in `rlevo`. It gated no kernel. Its modules
(`ops::kernels`, `algorithms::metaheuristic::kernels` and two children) were
214 lines of design notes with no items. It did two things: its `cubecl`
dependency carried `features = ["wgpu"]`, which pulls the GPU stack whatever
Burn is asked for; and it switched off `FireflyConfig`'s swarm cap, so that
with the feature on `validate` accepted any `pop_size`, a debug build then
panicked in `init`, and a release build ran an `(N, N, D)` tensor at whatever
size was asked.

## Decision

**1. No published library crate selects a Burn backend.** The workspace entry
becomes

```toml
burn = { version = "0.21.0", default-features = false, features = ["std"] }
```

and `rlevo-core`, `rlevo-environments`, `rlevo-evolution`,
`rlevo-reinforcement-learning` and `rlevo-hybrid` inherit it unchanged in
`[dependencies]`. `std` is the whole list each of them needs on Burn 0.21. In
particular the RL crate needs no `autodiff` feature: it is written against the
`AutodiffBackend` trait, which Burn always compiles, and the feature only adds
the `Autodiff<B>` type a caller instantiates it with.

**2. The leaf selects.** Three kinds of leaf exist:

- *A consumer of `rlevo`.* The umbrella gains two features, `flex` (CPU, on by
  default) and `wgpu` (GPU), and re-exports Burn as `rlevo::burn`, so a
  consumer never writes Burn's version. Each feature is
  `["burn/default", "burn/<backend>", "burn/autodiff"]`.
- *A crate's own tests, benches and doctests.* Each crate adds a `burn` entry
  to `[dev-dependencies]` with the features its test targets use. Cargo does
  not apply a dev-dependency's features to a library built for a consumer.
- *The unpublished crates.* `rlevo-test-support` and `rlevo-examples` name
  backends in their own code and select them in `[dependencies]`.

A consumer of a single library crate is a fourth, and selects in its own
manifest. That is Burn's own convention: every backend in Burn's facade is an
optional feature the application turns on.

**3. The leaf keeps Burn's defaults for the backend it selects.** This is what
`burn/default` in item 2 is for, and it is a measured choice, not a habit.
With the facade's defaults off, Burn's Flex backend loses its own defaults,
`simd` and `rayon`. The facade has a `simd` feature but none that reaches
`burn-flex/rayon`, so `burn/default` is the only way to get the multithreaded
backend without naming a Burn sub-crate. Three settings of the leaf, the first
being what every build used before this change:

| Workload on Flex | Burn's defaults (`simd`, `rayon`) | `simd` only | Neither |
|---|---:|---:|---:|
| DQN update, MLP 4-64-64-2, batch 64 | 123 µs | 123 µs | 136 µs |
| DQN update, MLP 4-64-64-2, batch 256 | 253 µs | 253 µs | 292 µs |
| DQN update, MLP 4-512-512-512-2, batch 64 | 3.97 ms | 4.47 ms | 4.77 ms |
| DQN update, MLP 4-512-512-512-2, batch 256 | 6.15 ms | 13.39 ms | 14.09 ms |
| DQN update, conv net on pixel observations, batch 64 | 22.3 ms | 28.5 ms | 28.6 ms |
| DQN update, conv net on pixel observations, batch 256 | 84.4 ms | 115.3 ms | 115.3 ms |
| DE generation, population 1024, 10 genes | 125 µs | 125 µs | 131 µs |
| Tournament selection, population 1024 | 15.1 µs | 15.2 µs | 15.1 µs |
| Packages, `rlevo --no-default-features --features flex` | 171 | 149 | 142 |

Criterion medians, one run per cell, on the machine above (8 performance and
4 efficiency cores), from `learn_step_bench` (the `host_row` arm) and
`operators` in the RL and evolution crates. One update is stage, forward,
loss, backward and optimizer step.

`simd` is worth 10 to 16 percent on the small network and 5 percent on a DE
generation at population 1024. `rayon` is worth nothing on the small network
and a factor of 2.2 on a 512-wide one at batch 256. A leaf without Burn's
defaults would have slowed every network larger than a toy, to save 29
packages.

So the umbrella's `flex` feature and every test build compile `burn-flex` with
exactly the features it had before: `[default, rayon, simd, std]`, read from
`cargo tree -e normal -f '{p} [{f}]'`. No numeric result can have moved, and
none did.

**4. `custom-kernels` is deleted, not disconnected.** The feature, the
`cubecl` dependency and the four placeholder modules go. `FireflyConfig`'s cap
becomes one contract: `validate` rejects `pop_size >
FIREFLY_PURE_TENSOR_CAP` (128) in every build, through the same
`config::at_most` helper every other config uses.

**5. The property is checked on one crate at a time.** A workspace build
unifies features across members, so a test target's `burn/flex` reaches the
library under test and a library that names `burn::backend::Flex` compiles
everywhere in CI. The `libraries-select-no-backend` job in `crate-tests.yml`
derives the list of publishable crates that depend on Burn and, for each one
except the umbrella, asserts the resolved `burn` feature set is exactly `std`,
runs `cargo check -p <crate> --lib`, and runs `cargo doc -p <crate> --no-deps`.
`docs/rules.md` gains the matching rule for dependency audits: check `cargo
tree -p <crate> -e normal | grep -c ratatui`, never the manifests.

## Consequences

**Positive:**

- Package counts, same method and machine:

  | Crate | Before | After |
  |---|---:|---:|
  | `rlevo-core` | 352 | 107 |
  | `rlevo-reinforcement-learning` | 353 | 108 |
  | `rlevo-evolution` | 353 | 114 |
  | `rlevo-benchmarks` | 354 | 115 |
  | `rlevo-hybrid` | 355 | 116 |
  | `rlevo-environments` | 376 | 149 |
  | `rlevo` (default features) | 380 | 207 |
  | `rlevo` with `wgpu` added | | 253 |
  | `rlevo --no-default-features` | | 118 |

- Cold builds, three runs each into an empty target directory, `dev` profile:

  | Build | Before | After |
  |---|---|---|
  | `cargo build -p rlevo-core` | 42.9 to 45.0 s, 295 CPU-s, 383 units | 11.7 to 12.1 s, 58 CPU-s, 113 units |
  | `cargo build -p rlevo` | 43.5 to 45.2 s, 331 CPU-s, 413 units | 34.1 to 34.6 s, 172 CPU-s, 224 units |

  CPU-seconds are user plus system time of the middle run. The umbrella's
  wall time falls less than its CPU time because its longest chain of
  dependent units, through the CPU backend, is still there.

- `cargo tree -p <crate> -e normal | grep -c ratatui` prints 0 for every
  production crate and for `rlevo` (it printed 9 for `rlevo`). The invariant
  now holds in the graph that is built.
- A consumer of `rlevo` compiles no GPU shader stack unless it asks for
  `wgpu`.
- Firefly has one swarm-size contract instead of three that depended on a
  feature and a build profile.

**Negative / accepted costs:**

- **Breaking for a direct consumer of a library crate.** They add `burn` with
  a backend feature themselves; without one `burn::backend::Flex` does not
  exist and the compiler says so. A consumer of `rlevo` at default features
  is unaffected on CPU.
- **Breaking for a consumer of `rlevo` who used the GPU backend.** `wgpu` was
  compiled for everyone; it is now a feature to request.
- **Breaking for a release build that ran Firefly above 128.** The default
  feature set used to let it through. `validate` now refuses it. The cap is a
  constant on the swarm and takes no account of the genome dimension, which is
  a known weakness and is filed as its own issue.
- **Eight manifests repeat a backend feature list.** Cargo has no way to name
  a feature set once for dev-dependencies. The lists differ by need
  (`rlevo-core` asks for `flex`; the RL crate for `flex`, `autodiff` and
  `wgpu`), so they are not copies of one another.
- **`burn/default` at the leaf costs 29 packages** over a bare `flex` plus
  `autodiff` (171 against 142 for `rlevo --no-default-features --features
  flex`). Among them is `cubecl` and its runtime, without its wgpu backend:
  `burn-backend` depends on it once Burn's defaults are on. Item 3 is why that
  price is paid.
- **Test builds of `rlevo-evolution` and the RL crate still compile the GPU
  stack**, because their backend-parity tests and benches name `Wgpu`. That is
  the maintainer's cost, not a consumer's.

**Neutral:**

- `rlevo-core` still depends on Burn, at `std`. A core with no Burn in it is a
  separate decision with its own design (tensor conversion has to move behind
  a feature), and this ADR does not make it.
- `rlevo`'s default features still include `box2d` and `locomotion`, so a
  default build still compiles two physics engines. Which features the
  umbrella turns on by default is likewise not decided here.
- The Burn version does not change. Burn 0.22 makes `burn-optim` an optional
  feature, so on 0.22 the RL crate's list will not be `std` alone. Decision 1
  is "no backend", which survives that; the feature list is re-measured when
  the workspace moves.

## Alternatives considered

**Trim the feature list and leave defaults on** (`["wgpu", "flex",
"autodiff"]`). It removes the `train` stack and keeps both backends in every
library. Rejected: an environment author still compiles a GPU shader stack to
implement a trait, and the library crates would still be choosing backends
for their consumers.

**Keep one backend in the libraries** (`["std", "flex", "autodiff"]`).
Simpler manifests, and tests need no dev-dependency line. Rejected for the
same reason at a smaller scale: `rlevo-core` would resolve 131 packages
instead of 107 to compile a backend it never names, and a GPU-only consumer
would compile the CPU backend regardless.

**Turn Burn's defaults off at the leaf as well.** The smallest graph.
Rejected on the measurements in item 3 of the decision, and because it would
have put every test in the workspace on a differently compiled backend from
the one their known answers were taken on.

**Depend on `burn-flex` directly to enable `rayon` without `burn/default`.**
Rejected: it puts a Burn sub-crate and its version in the umbrella's
manifest, which is what going through the facade exists to avoid.

**Keep `custom-kernels` and drop `wgpu` from its `cubecl` entry.** Rejected:
the feature would still gate nothing but a validation check, and it would
still be on by default.

## References

- `Cargo.toml` (`[workspace.dependencies]`), `crates/rlevo/Cargo.toml`
  (`flex`, `wgpu`), and the `[dev-dependencies]` of each library crate.
- `.github/workflows/crate-tests.yml`, job `libraries-select-no-backend`.
- `docs/rules.md`: the `burn` row under Dependency Usage, and "Audit
  dependencies on the resolved graph" under Architecture Invariants.
- `crates/rlevo-evolution/src/algorithms/metaheuristic/firefly.rs`:
  `FIREFLY_PURE_TENSOR_CAP` and `FireflyConfig::validate`.
- Burn 0.21.0's `Cargo.toml` for the feature definitions quoted in Context,
  and `burn-flex` 0.21.0's for its defaults.
- ADR [0013](0013-metrics-only-live-tui.md), whose "no production crate
  depends on viz deps" invariant this change makes true in the resolved graph.
