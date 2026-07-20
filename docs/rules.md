---
project: rlevo
status: active
type: reference
date: 2026-05-31
tags: [rules, conventions, constraints, architecture]
---

# rlevo Rules

Hard constraints and conventions that apply to every contribution. Read this before making any implementation decisions.

---

## 1. Workspace Structure
- The workspace contains exactly these crates, each with a single, bounded responsibility:
  - `rlevo-core` — agent / environment contract: `State`, `Observation`, `Action`, `Reward`, `Environment`, `Snapshot`, `TensorConvertible`, the `util::combinations` math helper, and the **render trait surface** (`Renderer`, `NullRenderer`, `AsciiRenderable` [optional debug helper — not a mandatory env invariant, per ADR 0013], `AsciiRenderer`, `StyledFrame`, `StyledLine`, `StyledSpan`, `SpanStyle`, `Color`, `Modifier`, `palette`). No algorithm or environment implementations. (ADR 0009)
  - `rlevo-environments` — environment implementations (depends on `rlevo-core`). `rlevo-environments::render` is a re-export shim that forwards all render types from `rlevo-core::render`; do not add new types there.
  - `rlevo-reinforcement-learning` — gradient-based RL algorithms + the RL-only `memory` (replay buffer), `experience` (trajectory storage), and `metrics` (`AgentStats`) modules. Depends on `rlevo-core`. `rlevo-environments` is a dev-dep only (used by examples / benches / integration tests).
  - `rlevo-evolution` — evolutionary strategy algorithms. Depends on `rlevo-core` (for `BenchEnv`, `FitnessEvaluable`, `Landscape`, `SeedStream` per ADR 0004; ADR 0002's zero-dep stance was partially reversed by ADR 0004). Depends on `parking_lot` for `SharedPopulationObserver`.
  - `rlevo-hybrid` — hybrid RL + EA approaches. Hosts `StatefulPolicy` / `ReactivePolicy` (the rollout-policy contract), `RolloutFitness`, and `PolicyNeuroevolution` (ADR 0025). Depends on `rlevo-core`, `rlevo-reinforcement-learning`, and `rlevo-evolution`.
  - `rlevo-benchmarks` — paradigm-neutral evaluation harness (light-touch dep on `rlevo-core`, per ADR 0001). Has optional `tui` (ratatui live dashboard), `report` (static-HTML Leptos viewer), and `record` (EpisodeRecord file writer) features; this is the **only** production crate permitted to carry viz deps, and only behind those feature gates.
  - `rlevo-benchmarks-report-client` — WASM report client; mirrors `rlevo-benchmarks`'s wire types for browser-side rendering. **Not in `default-members`**. Depends on `rlevo-metrics-registry`. (ADR 0014, 0015)
  - `rlevo-metrics-registry` — `#![no_std]`, zero-dependency leaf crate holding the typed `MetricDescriptor` table (name, kind, cadence, title, unit). Shared by `rlevo-benchmarks` and `rlevo-benchmarks-report-client` as a single source of truth; eliminates the forked-metric-list footgun. (ADR 0015)
  - `rlevo` — public API re-export aggregator (excludes `rlevo-benchmarks`); exposes `viz-tui` and `viz-report` features that forward to `rlevo-benchmarks`.
  - `rlevo-examples` — heavyweight example programs that import `rlevo-benchmarks` or any viz/record/report feature dep. **Not in `default-members`** — opt in with `-p rlevo-examples`. Has no public library surface of its own. (ADR 0012)
  - `rlevo-test-support` — `publish = false`, dev-only crate holding shared cross-crate test scaffolding (`LinearEnv`, Flex backend helpers, seeded baseline runners, assertion macros). Never a prod dep of any shipped crate. **Not in `default-members`**. (ADR 0024)
- Never add implementation logic to `rlevo-core`; it is a contract crate. The `util` module is the single exception (a math helper folded in from the deleted `rlevo-utils` crate per ADR 0003).
- All shared dependencies must be declared in the root `Cargo.toml` `[workspace.dependencies]` table. Crates inherit via `{ workspace = true }`.
- Workspace resolver must remain `resolver = "3"`.
- All crates share version `0.1.0` and license `MIT OR Apache-2.0` via workspace inheritance.

## 2. Naming Conventions

### Files and Modules
- File and module names: `snake_case` (e.g., `frozen_lake.rs`, `k_armed.rs`).
- Concept modules: singular noun (`state.rs`, `action.rs`, `environment.rs`).
- Collection modules (folders): plural noun (`grids/`, `landscapes/`, `toy_text/`).
- Maximum module nesting depth: 3 levels (e.g., `envs/toy_text/frozen_lake.rs`).

### Example target names

The filename of an example **is** its `cargo run --example <name>` target (Cargo auto-discovery — there is no `[[example]]` typo-check for `examples/*.rs`, so a mistyped name is unrecoverable except via Cargo's "similar name" hint). Keep names short and unambiguous:

- `snake_case`, **≤ 24 characters**, no abbreviations that drop letters (`constraints`, never `contraints`).
- Lead with the **subject**, not a filler verb: `state_constraints.rs`, not `continuous_state_with_constraints.rs`; `grid_agent.rs`, not `egocentric_grid_agent_demo.rs`. Drop `_demo` / `_example` / `_showcase` suffixes — the `examples/` directory already says it is one. *(Pre-existing longer names are grandfathered; apply this to new examples.)*
- For heavy `rlevo-examples` targets, prefix by product tier so the required feature is obvious from the name: `tui_<subject>` (needs `viz-tui`), `report_<subject>` (needs `viz-report`).
- Each example is owned by the crate whose `examples/` dir holds it; run it with that crate's `-p` (see §11). Prefer the `justfile` recipe over typing the full `-p … --example …` line.

### Types and Traits
- Types and traits: `PascalCase` (e.g., `ScalarReward`, `SnapshotBase`, `DiscreteAction`).
- Error types: `PascalCase` enum or struct ending in `Error` (e.g., `StateError`, `EnvironmentError`, `TensorConversionError`).
- Config types: `PascalCase` ending in `Config` (e.g., `FrozenLakeConfig`).
- Builder types: `PascalCase` ending in `Builder` (e.g., `DqnTrainingConfigBuilder`).

### Functions and Methods
- All functions and methods: `snake_case`.
- Constructors: `new()` (default), `with_config(config)` (alternate), `from_*()` (conversion).
- Accessors returning references: `observation()`, `reward()`, `status()`.
- Predicate methods: `is_valid()`, `is_done()`, `is_terminated()`, `is_truncated()`.
- Dimension methods: `shape()`, `numel()`.
- Mutation methods: `update()`, `reset()`, `step()`, `add()`, `record()`.
- Builder chain methods: `.with_*()` (fluent), `.build()` (finalize).
- Identity/utility methods: `zero()`, `enumerate()`, `aggregate()`, `encode()`, `decode()`.

### Struct Field Encapsulation (ADR 0055)

The workspace allocates invariant enforcement across three mechanisms. Which one
applies is decided by the *kind* of struct, not case by case:

- **`*Config` types keep `pub` fields.** A config is a hyperparameter record;
  `GaConfig { pop_size: 64, ..Default::default() }` is the tuning idiom the
  library exists to serve. Config types are bound by the **Config Validation
  Contract in §4 alone** — the consumer calls `config.validate()?` at
  construction — and are **exempt** from the encapsulation rule below. A struct
  literal reaching an invalid config is the case ADR 0026 §3 was written for,
  not a bypass of it.
- **State / params / genome structs do not expose `pub` data-bag fields.**
  A `*State`, `*Params`, or `*Genome` type keeps its fields private (or
  `pub(crate)` when a family of in-crate operators mutates it in place) and
  exposes `#[must_use]` read accessors named after the field
  (`fitness()`, `best_fitness()`, …). This prevents external — or careless
  internal — code from struct-literal-constructing a structurally invalid
  value (mismatched lengths, out-of-range probabilities, a broken derived
  tail) that only panics generations later. (issue #141; mirrors `Population`
  and the ADR 0022/0030 accessor style.)
- **Guarded construction.** A struct with a checkable invariant offers a
  validating `try_new(...) -> Result<Self, ConfigError>` (structural checks:
  lengths agree, counts non-zero, `σ > 0`) as its public entry point; the
  trusted in-module construction path may keep a struct literal. A struct
  with no invariant offers an infallible `new`. Hyperparameter-bearing
  `*Params` follow the Config Validation Contract below and expose their
  overrides through validating `with_*` setters, not `pub` fields.
- **An invariant that must survive struct-literal construction goes in a
  validated newtype, not behind a private field.** Follow the ADR 0027 /
  0031 shape (`Bounds`, `Probability`, `NonNegativeRate`): private inner field,
  `new` (panicking, for literals) / `try_new` (`Result`), a dedicated `Copy`
  error, `#[serde(try_from)]`. A newtype is **transitive** — it holds inside a
  config, inside a `*Params`, after `Deserialize`, and after a
  `Clone`-then-mutate — whereas field privacy guards exactly one boundary. A
  `pub` field of a validated newtype is therefore safe under struct-literal
  construction; `GaConfig`'s `bounds: Bounds` / `mutation_sigma: NonNegativeRate`
  are the reference (`algorithms/ga.rs:73-89`). Such a field **removes** its
  paired `config::` check from `validate()` rather than duplicating it.
- **`#[non_exhaustive]` is for enums, never structs.** It is used here for its
  match-exhaustiveness meaning (`EnvironmentError`, `RecordError`, `EnvFamily`,
  …), not to forbid struct literals. On a struct it also forbids cross-crate
  `..Default::default()`, breaking the tuning idiom above while buying no
  validation guarantee (a builder's `build()` must still call `validate()`).
  Applying it to a struct requires its own ADR.
- **Validate first, destructure second.** A config-consuming constructor calls
  `validate()` before reading any field, so no field is acted on ahead of its
  own check.
- **An accessor on a config must be total.** A method that reads config fields
  without constructing anything must return a documented sentinel for values
  `validate()` rejects, never panic — `C51Config::delta_z` returning `f32::NAN`
  for `num_atoms < 2` is the reference (`c51/c51_config.rs:120-130`).

### Constants and Associated Constants
- Constants: `UPPER_SNAKE_CASE` (e.g., `ACTION_COUNT`, `MAX_STEPS`, `GOAL_STATE`).
- Const generic parameters: `D` (observation/state rank), `AD` (action rank), `SD` (state rank).
- Batch ranks are always `BD = D + 1`, `BAD = AD + 1` — never any other offset.
- Reified const generic: `const DIM: usize = D` (reify inside trait body for ergonomic access).

### Generic Parameters
- Burn backend: `B: Backend` (always single letter `B`).
- Domain bounds: `O: Observation<D>`, `A: Action<AD>`, `S: State<SD>`, `R: Reward`.
- Const generic ordering in signatures: `<const D: usize, const SD: usize, const AD: usize>`.
- Place all non-trivial bounds in a `where` clause, not inline.

---

## 3. Trait Design Constraints

- Every public trait must have a doc comment explaining its purpose and when to implement it.
- Associated types use singular nouns matching the concept (`type Observation`, `type StateType`, `type ActionType`).
- Provided default method implementations must be conservative and correct for all conforming types; document any assumptions.
- Trait methods that can fail return `Result<T, DomainError>` — never `Option` for error signalling.
- Marker traits (`MarkovState`, `GenomeKind`) carry `const` or zero methods; they must not grow methods unless the concept fundamentally requires it.
- Do not add blanket impls without a documented invariant justifying them.

### Core Trait Invariants (these must never be violated by implementations)

| Trait | Invariant |
|-------|-----------|
| `State<D>` | `shape().iter().product::<usize>() == numel()` |
| `Observation<D>` | `shape().iter().product::<usize>() == DIM` |
| `DiscreteAction<D>` | `from_index(a.to_index()) == a` and `from_index(x).to_index() == x` for all valid `x` |
| `ContinuousAction<D>` | `as_slice().len() == COMPONENTS` always, and `from_slice` accepts exactly `COMPONENTS` values — `D` is the tensor rank, never the component count (ADR 0038) |
| `BoundedAction<D>` | `low().len() == high().len() == COMPONENTS` (**not** `D`), and `low()[i] < high()[i]` for all `i` (ADR 0053) |
| `HostRow<D>` | `write_host_row` pushes exactly `row_shape().iter().product()` plain `f32` values, appended (never clearing `buf`); a type implements `HostRow` at exactly **one** rank `D` |
| `TensorConvertible<D, B>` | `from_tensor(x.to_tensor(device)) == Ok(x)` for all valid `x`. The row-writer half of the contract lives on the backend-independent `HostRow<D>` supertrait; `to_tensor` is derived from it and must not be overridden |
| `Reward` | `zero()` is the additive identity: `r + zero() == r` |
| `History` | `buffer.len()` never exceeds `capacity` field; use explicit eviction |
| `ReplayStrategy<T>` | `len()` never exceeds capacity; every freshly sampled id resolves via `get()` until evicted (ADR 0050) |

### Optimisation direction — maximise-native (ADR 0023)

The engine is **maximise-native**: *higher fitness is always better*. This is the
single internal convention across the whole library (RL, evolutionary, NEAT).

- Every `Strategy`, operator, shaping rule, local searcher, and metric
  aggregation in `rlevo-evolution` works purely in **canonical (maximise)**
  space and is **sense-unaware** — it never sees an `ObjectiveSense`. Best is the
  largest value; the worst-value sentinel is `f32::NEG_INFINITY`. Fitness hygiene
  is one rule (ADR 0034): `NaN → −inf` (worst), `+∞ → f32::MAX` (ranks top but
  finite, so it cannot blow a `mean`/reward up), `−inf` passes through.
- An objective declares its natural direction with
  `rlevo_core::objective::ObjectiveSense { Minimize, Maximize }`. Fitness
  functions and landscape adapters return **natural** values — **never
  hand-negate** a cost into a reward.
- Cost objectives are reconciled into canonical space at **exactly one
  chokepoint**: the `EvolutionaryHarness` (and the `FromLandscape` /
  `FromFitnessEvaluable` adapters that carry the sense). The harness negates a
  `Minimize` objective before `tell` and maps metrics back to the declared sense
  for reporting (`best_fitness` reads as the natural cost — Sphere → 0).
- `BatchFitnessFn::sense()` is **required, no default** — a reward/accuracy
  objective must declare `Maximize` explicitly so it cannot be optimised
  backwards by omission. Only `Landscape::sense()` defaults (to `Minimize`, since
  a landscape is a cost surface by definition).
- **Compare floats with `total_cmp`, never `partial_cmp`.** `f32`/`f64`
  comparisons in sorts, `max_by`/`min_by`, and argmax use `total_cmp` — a
  deterministic total order. `partial_cmp(..).unwrap()` panics on any `NaN`
  operand, and `partial_cmp(..).unwrap_or(Equal)` places `NaN` non-
  deterministically. There is no clippy lint for this footgun
  (`clippy::unwrap_used` is too broad to enable workspace-wide), so it is a
  review-enforced convention.
- **Sanitise fitness before comparing it.** `total_cmp` alone is *not* enough for
  fitness ordering: Rust's `f32::NAN` is a *positive* NaN, so `total_cmp` ranks it
  as the **maximum** — a raw `fit[b].total_cmp(&fit[a])` descending sort would put
  a `NaN`-fitness member *first* (as the best). Under maximise-native `NaN` is the
  **worst**, so map it to `−inf` first with `sanitize_fitness` (the single rule of
  §Optimisation direction) and then `total_cmp`:
  ```rust
  let sane: Vec<f32> = fitness.iter().map(|&f| sanitize_fitness(f)).collect();
  order.sort_by(|&a, &b| sane[b].total_cmp(&sane[a])); // best first; NaN last
  ```
  Non-fitness float comparisons (geometry, eigenvalues, argmax over logits) use
  plain `total_cmp` — only *fitness* needs the sanitise step. This per-site rule is
  the **correctness floor** and stays even where a driver chokepoint pre-sanitizes
  (ADR 0034): direct (non-harness) callers — unit tests, custom drivers — bypass
  the chokepoint, so every ordering/argmax/champion-write site sanitizes locally.
  For a whole `Tensor<B, 1>` on the hot path use `sanitize_fitness_tensor`
  (one device op) instead of a host round-trip.

---

## 4. Error Handling

- All error types must implement `std::error::Error` + `std::fmt::Display` + `std::fmt::Debug`.
- Prefer enum error types with structured variants over string-based errors:
  ```rust
  pub enum StateError {
      InvalidShape { expected: Vec<usize>, got: Vec<usize> },
      InvalidData(String),
      InvalidSize { expected: usize, got: usize },
  }
  ```
- Return `Result<T, SpecificError>` — never erase error types with `Box<dyn Error>` in library-facing APIs.
- Domain boundaries: `EnvironmentError` for environment ops, `TensorConversionError` for tensor ops, `ReplayBufferError` for buffer ops.
- **Never `.unwrap_or_default()` a tensor host-read.** `from_tensor` and any
  other host-read of a tensor (`into_vec`, `as_slice`, `to_vec` on
  `TensorData`) must propagate the transfer error — via
  `map_err(|e| TensorConversionError { .. })?` in production code, or
  `.expect("<named invariant>")` where the read cannot fail by construction
  (e.g. a tensor the same function just built). `.unwrap_or_default()`
  substitutes an **empty buffer** for a real dtype/device failure, which
  surfaces later as a misleading out-of-bounds panic far from the cause.
  There is no clippy lint narrow enough for this footgun, so it is a
  review-enforced convention (see #136 for the historical sweep).
- **Panics are permitted only for programming errors** (index out of bounds, dimension mismatch, or an out-of-domain argument to a single documented builder setter — see the Panic Contracts table and the Config Validation Contract below). Document every panic site in a `# Panics` doc section.
- Never panic in response to user-supplied runtime data; return `Err(...)` instead. A `Deserialize`-able config *is* user-supplied runtime data.

### Documented Panic Contracts

| Site | Condition |
|------|-----------|
| `DiscreteAction::from_index(i)` | `i >= ACTION_COUNT` |
| `ContinuousAction::from_slice(v)` | `v.len() != COMPONENTS` (**not** `D`, the tensor rank — ADR 0038/0053) |
| `MultiDiscreteAction::from_indices(arr)` | any `arr[i] >= space[i]` |
| Builder `with_capacity(n)` | `n == 0` |
| Builder `with_alpha(x)` | `x ∉ [0.0, 1.0]` |
| `SimulatedAnnealingParams::with_max_iters` | `max_iters == 0` |
| `SimulatedAnnealingParams::with_initial_temp` | value not finite or not `> 0` |
| `SimulatedAnnealingParams::with_cooling` | `Geometric` `factor ∉ (0, 1)`; `Linear` `delta` not finite or not `> 0` |
| `SimulatedAnnealingParams::with_min_temp` | `min_temp` not finite or `< 0` |
| `SimulatedAnnealingParams::with_step_size` | `step_size` not finite or not `> 0` |
| Batch rank assertion | `BD != D + 1` or `BAD != AD + 1` |
| `ops::selection::tournament_*` | `fitness.is_empty()` or `tournament_size < 2` |
| `ops::selection::truncation_*` | `fitness.is_empty()` or `top_k > fitness.len()` |
| `ops::selection::{tournament_select, truncation_select}` | `population.dims()[0] != fitness.len()` |
| `ops::crossover::{blx_alpha, uniform_crossover, binary_uniform_crossover}` | parents differ in shape |
| `ops::mutation::gaussian_mutation_per_row` | `sigmas.len() != population N` |
| `ops::replacement::elitist` | `k > pop_size` or `pop_size − k > offspring count` |
| `ops::replacement::mu_plus_lambda` | `mu > parent count + offspring count` |
| `ops::replacement::mu_comma_lambda` | `mu > offspring count` (i.e. `λ < μ`) |
| `local_search::simulated_annealing::{refine, refine_with_known_fitness}` (via `refine_impl`) | `max_iters == 0` |

The `Builder with_capacity(n)` / `with_alpha(x)` and
`SimulatedAnnealingParams::with_*` rows are the blessed **setter-guard**
exception: a single `with_*` method may panic on an out-of-domain argument
because the panic points at the offending call site. They do **not** replace
whole-config validation — see below.

### Config Validation Contract (ADR 0026, 0055)

- Every public `*Config` (and any hyperparameter-bearing builder) implements
  `rlevo_core::config::Validate` — `fn validate(&self) -> Result<(), ConfigError>`.
  `ConfigError` names the config, the field, and the violated `ConstraintKind`
  (structured, allocation-free).
- Construction that consumes a **caller-supplied or deserialized** config calls
  `config.validate()?` and returns `Result<_, ConfigError>`. It **must not
  panic** — a config can arrive via `Deserialize` or struct-update syntax with
  no guarded setter, and such data falls under the "never panic on
  user-supplied runtime data" rule above.
- `Default` construction stays infallible, but every `impl Validate` unit-tests
  that its own `Default` passes `validate()` (a library default must be valid).
- The line: a **setter guard** (`with_x(v)`) may panic on one out-of-domain
  argument at its call site; validating an **assembled config as a whole**
  (cross-field invariants, or fields set without a guarded setter) returns
  `ConfigError`. If an invalid value can arrive via `Deserialize`, it must be an
  `Err`, never a panic.
- Config fields stay **`pub`** (§2, ADR 0055). This contract — not
  encapsulation — is the whole of a config's enforcement, so a new
  config-consuming constructor adopting `validate()?` is not optional. Any
  invariant that must hold independently of that chokepoint belongs in a
  validated newtype (§2, ADR 0027/0031).
- **No config loader exists in the workspace yet.** The first one added (run
  manifest, config file, checkpoint restore) owns the `Deserialize` half of this
  contract: call `validate()?` on the decoded value and propagate the `Err`.
  ~22 configs derive `Deserialize` plainly, so a decoder will otherwise accept
  values `validate()` rejects (ADR 0055 §Consequences).

---

## 5. Tests, Benches, and Examples — Placement

This section is the **single decision authority** for where any test, bench, or example
goes. §11 holds the detailed dependency boundary for examples; everything else is here.
When adding any of the three, classify by **purpose first, scope second**.

### Step 1 — Classify by purpose (these three are never interchangeable)

| If the artifact… | it is a… | purpose |
|---|---|---|
| asserts a correctness property and **fails the build** when violated | **test** | verification |
| measures runtime / throughput / allocations under `criterion` | **bench** | performance measurement |
| is a runnable program demonstrating real usage, **never asserted against** | **example** | demonstration |

If one file does two of these (e.g. an example that also asserts), **split it** — an
example is not a test, and a bench is not an example with timing prints.

> **Terminology trap — "benchmark" means two different things in this workspace:**
> - **`rlevo-benchmarks`** (the *crate*) is the paradigm-neutral **evaluation harness** —
>   it runs agents against environments and reports fitness / performance / metrics. It is
>   **not** a home for `cargo bench` targets.
> - **benches** (`benches/` dirs, `[[bench]]` targets, `criterion`) are **performance**
>   micro/throughput benchmarks and may live in any crate.
>
> Never add a `[[bench]]` target to `rlevo-benchmarks` "to match the name", and never put evaluation-harness logic in a `benches/` dir.

### Step 2 — Classify by scope, then place

| Artifact         | Scope (what it touches)                                     | Lives in                                       | Run with                                       |
| ---------------- | ----------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| Unit test        | invariants inside **one source file**                       | in-source `#[cfg(test)]`, owning crate         | `cargo test -p <crate>`                        |
| Integration test | **one crate's** public surface                              | `<crate>/tests/`                               | `cargo test -p <crate>`                        |
| Integration test | **≥2 crates** together                                      | `crates/rlevo/tests/` (flat)                   | `cargo test -p rlevo`                          |
| Bench            | **one crate's** hot path                                    | `<crate>/benches/` + `[[bench]]` in that crate | `cargo bench -p <crate>`                       |
| Bench            | **cross-crate** throughput (e.g. `cartpole_record.rs`)      | `crates/rlevo/benches/`                        | `cargo bench -p rlevo`                         |
| Example          | imports **only the 5 library crates**                       | `crates/rlevo/examples/`                       | `cargo run -p rlevo --example <name>`          |
| Example          | imports `rlevo-benchmarks` or any **tui/record/report** dep | `crates/rlevo-examples/examples/`              | `cargo run -p rlevo-examples --example <name>` |

The single rule that unifies the table: **single-crate scope stays in that crate;
multi-crate scope moves up to the umbrella `crates/rlevo/`** — except heavy examples,
which move *sideways* to `crates/rlevo-examples/` (see §11 for the exact dep test).

### Step 3 — Decision procedure (apply in order)

1. **Purpose** — assert correctness → *test*; measure performance → *bench*; demonstrate
   usage → *example*. Pick exactly one.
2. **Scope** — does it exercise one crate or several? One → owning crate. Several → umbrella.
3. **Heaviness (examples only)** — does it import `rlevo-benchmarks` or any
   viz/record/report feature dep? Yes → `crates/rlevo-examples/`; No → `crates/rlevo/examples/`.
   This test is dependency-based, not scope-based — see §11.

### Three-tier test placement rule (canonical — ADR 0012)

| Test kind | Lives in | Discovered by |
|---|---|---|
| Unit tests (`#[cfg(test)]` modules inside source files) | The owning source file, in the owning crate | `cargo test -p <crate>` |
| Single-crate integration tests (exercises one crate's public surface) | `<crate>/tests/` in the owning crate | `cargo test -p <crate>` |
| Cross-crate integration tests (exercises two or more crates together) | `crates/rlevo/tests/` (flat, no subdirectories) | `cargo test -p rlevo` |

Supplementary rules:
- **Unit tests stay in-source.** Do not move a `#[cfg(test)]` block to `tests/` just because a file grows large; in-source placement keeps the test adjacent to the invariant it checks.
- **Single-crate integration tests belong to their crate.** `cargo test -p rlevo-environments` must run them without pulling the full umbrella cone.
- **Cross-crate tests are flat in `crates/rlevo/tests/`.** Do not add subdirectory structure unless paired with explicit `[[test]]` Cargo.toml entries — prefer flat; test names are unique workspace-wide.

### Bench placement rule

- **Scope of `benches` (cargo bench).** A `[[bench]]` target exists to **micro-benchmark core logic** and must **strictly measure execution time or resource allocation** (CPU time, throughput, allocations/peak memory) under `criterion`. It is not a correctness test, not a demo, and not an evaluation run of an agent against an environment (that is `rlevo-benchmarks`). If a "bench" asserts a result or prints a story instead of producing timing/allocation numbers, it is misfiled — move it to a test or an example.
- **`[[bench]]` entries stay in their owning crate** when measuring a single-crate hot path; declare `criterion` as a `dev-dependency` there.
- **Cross-crate throughput benches** (those that drive an env + agent + record sink together, e.g. `cartpole_record.rs`) live in `crates/rlevo/benches/`.
- A bench that needs `rlevo-benchmarks` evaluation-harness machinery is **not** a `[[bench]]` target — it is either a `rlevo-examples` program or a harness-internal helper. Do not smuggle it into a `benches/` dir.
- `harness = true` (libtest) benches are forbidden; all benches use the `criterion` harness (`harness = false`).

### Unit tests live inside `/src`

- **Unit tests live in the source file they verify**, inside an in-source `#[cfg(test)] mod tests { ... }` block at the bottom of the `.rs` file under `<crate>/src/`. They are never relocated to a `tests/` directory — that directory is reserved for integration tests against the crate's *public* surface.
- Because they compile inside the module, unit tests may exercise **private** items (non-`pub` functions, fields, and helpers); integration tests in `tests/` cannot. Use the in-source location precisely when you need to assert an internal invariant that is not part of the public API.
- Gate the module with `#[cfg(test)]` so it is excluded from release builds and adds zero cost to the shipped crate.
- One `mod tests` per source file; keep each test adjacent to the code it checks rather than collecting tests for several files into one module.

### Unit-test conventions

- Test names follow `test_[type_or_trait]_[behavior]_[condition]` (e.g., `test_discrete_action_roundtrip`, `test_environment_reset_clears_state`).
- Every test module must include mock/test types prefixed with `Test`, `Mock`, or `Simple`.
- Test types must derive `Debug` and `Clone`; implement only the minimum traits required.
- Coverage requirements per trait:
  - All required methods
  - All provided default methods that have non-trivial logic
  - Round-trip conversions (to/from index, to/from tensor, to/from slice)
  - Boundary values, empty/full collections, zero, infinity, NaN where applicable
  - All documented panic conditions via `#[should_panic(expected = "...")]`
- Floating-point comparisons use `(a - b).abs() < 1e-6` or the `approx` crate; never `==`.
- Always include an assertion message explaining what the assertion verifies.
- `MockEnvironment` inside `rlevo-core` tests is the canonical reference for correct `Environment` trait usage.

---

## 6. Documentation

- Every public item must have a doc comment. Zero-exception policy.
- Modules use `//!` inner doc comments at the top of `lib.rs` or `mod.rs`.
- Traits, structs, enums: `///` outer doc comment, minimum structure:
  1. One-line summary (fits in IDE tooltip)
  2. Blank line + extended explanation (design rationale, constraints)
  3. `# Examples` section with runnable or `ignore`-annotated code
  4. `# Panics` section if any method panics
  5. `# Errors` section if any method returns `Result`
- Methods: one-line summary + `# Arguments`, `# Returns`, `# Errors`, `# Panics` as applicable.
- Enum variants and public struct fields require at least a one-line `///` comment.
- Use backticks for inline code: `` `Type` ``, `` `method()` ``.
- Use `[`Type`]` syntax for intra-doc links to types in scope; use fully-qualified paths for cross-crate links.
- Document all trait invariants in the trait's doc comment under an `# Invariants` heading.

---

## 7. Const Generics and Type-Level Constraints

- The const generic `D` always represents the **tensor rank** of an observation or state, not the element count.
- Batch ranks `BD` and `BAD` must be validated at the call site with `assert_eq!` when the caller supplies them, not silently inferred.
- Never use `D` for the number of elements; use `numel()` or `shape().iter().product()` for that.
- Associated type bounds cascade: if `Environment<D, SD, AD>` is parameterised, all five associated types must be consistent with those parameters.
- `PhantomData<(O, A, R)>` is the approved pattern to carry erased generic parameters in builder/wrapper types.

---

## 8. Dependency Usage

| Dependency | Approved Usage | Forbidden |
|------------|---------------|-----------|
| `burn` | `TensorConvertible`, neural network models, backends (`wgpu`, `flex`) | Importing Burn in `rlevo-core` beyond trait bounds |
| `rand` | `rand::rng()` for thread-local RNG; `rng.random_range(0..n)` | `rand::thread_rng()` (deprecated) |
| `rand_distr` | Advanced sampling distributions | Inline manual rejection sampling when a distribution exists |
| `serde` | Derive `Serialize + Deserialize` on all domain types | Manual `impl Serialize` unless unavoidable |
| `tracing` | Structured logging in training loops and benchmarks | `println!` / `eprintln!` in library code |
| `approx` | Float comparisons in tests only | Float `==` comparisons anywhere |
| `rapier2d` / `rapier3d` | Physics-based environment simulation | Direct FFI to Box2D or MuJoCo C libraries |
| `criterion` | Micro-benchmarks as `dev-dependencies` in any crate; integration benchmark suites in `rlevo-benchmarks` | Production dependencies; `harness = true` benches |
| `parking_lot` | Shared observer/sink state in `rlevo-evolution` and `rlevo-benchmarks`; `SharedPopulationObserver = Arc<parking_lot::Mutex<dyn PopulationObserver>>` | `std::sync::Mutex` for observer or record-sink types (lock-type split is a known footgun — ADR 0010) |
| `ratatui` + `crossterm` | Live **metrics-only** TUI dashboard (no env panel, per ADR 0013) in `rlevo-benchmarks` behind the `tui` feature | Any crate other than `rlevo-benchmarks`; ungated imports; rendering env state in the live TUI |
| `leptos` | Static-HTML report tier in `rlevo-benchmarks` behind the `report` feature (compiled to WASM, no runtime server) | Embedded live server; any production crate without the feature gate |

- All crate features must be enabled at the workspace level; individual crates must not re-declare features unless they are crate-local.
- Do not add new workspace dependencies without updating `CLAUDE.md` and writing a decision record in `decisions/`.

### Host-RNG seeding convention (ADR 0029)

Stochastic environments and algorithms **own a persistent RNG that advances
across calls**. Seed it once at construction; never re-seed from stored config
inside `reset()` / `step()` / a per-generation loop. Re-seeding each call replays
bit-identical randomness and silently destroys the episode-to-episode (or
generation-to-generation) diversity the caller relies on.

- **Environments.** `reset()` draws from the persistent stream and lets it
  advance, so successive episodes see independent noise. A fixed *problem
  instance* (bandit arm-means / phases / context, procedural layout, …) is
  sampled **once at construction** and preserved across resets — restored to its
  baseline where an episode mutates it (e.g. a non-stationary bandit's drift),
  never re-seeded to reproduce it. Run-level reproducibility is already
  guaranteed by the construction seed (a fixed seed yields a fixed *sequence* of
  resets). For deterministic replay of one specific episode, expose an inherent
  `reset_with_seed(&mut self, seed: u64)`; `reset()` itself must not reseed.
  Genuinely deterministic environments (e.g. `AdversarialBandit`, whose reward is
  a pure function of the step index) have no per-episode randomness and simply
  preserve their fixed schedule — they still must not re-seed on reset.
- **Host sampling (evolution / RL).** Draw randomness on the host via
  `rlevo_evolution::rng::seed_stream(base, generation, SeedPurpose::_)` (or an
  `rlevo_core::util::seed::SeedStream`), fill a `Vec`, then build the tensor with
  `Tensor::from_data`. **Never** `B::seed(..)` + `Tensor::random(..)` in a
  production path: Burn's process-wide RNG mutex makes results depend on thread
  schedule (→ core count) rather than the seed, which races the parallel test
  runner. `rlevo-evolution` is the reference implementation. (ADR 0016, 0017)
- **RNG trait bounds.** Functions that need randomness take `rng: &mut R` with
  `R: rand::Rng + ?Sized`. Use `&mut dyn Rng` only where a call site genuinely
  requires dynamic dispatch (an object-safe trait method); document any such
  exception.

---

## 9. Linting and Formatting

- **Formatting is enforced.** The workspace is `rustfmt`-clean and CI runs `cargo fmt --all --check` (`.github/workflows/fmt.yml`); a PR that leaves the tree unformatted fails. Run `cargo fmt --all` before pushing. Because the tree is clean, a scoped `cargo fmt` no longer reflows unrelated files.
- Formatting config is stable-only: `rustfmt.toml` at the workspace root (`edition = "2024"`, `max_width = 100`, `newline_style = "Unix"`). Do not add nightly-only keys (`imports_granularity`, `group_imports`, `wrap_comments`, …) unless CI is switched to nightly rustfmt.
- The toolchain is pinned in `rust-toolchain.toml` (`channel = "1.94.1"`, components `rustfmt`, `clippy`) so local and CI rustfmt cannot diverge. Bumping Rust is a deliberate edit to that file.
- The one-time normalization commit is recorded in `.git-blame-ignore-revs`; enable blame-skip locally with `git config blame.ignoreRevsFile .git-blame-ignore-revs`.
- Workspace clippy lint groups (`cargo`, `complexity`, `correctness`, `pedantic`, `perf`, `style`, `suspicious`) are all set to `warn` at priority `-1`. This is the baseline and must not be lowered.
- Workspace Rust lints (`ambiguous_negative_literals`, `missing_debug_implementations`, `redundant_imports`, `redundant_lifetimes`, `trivial_numeric_casts`, `unsafe_op_in_unsafe_fn`, `unused_lifetimes`) are set to `warn`.
- Item-level `#[allow(...)]` attributes are permitted only when:
  1. The lint is a false positive in that specific context, **and**
  2. A comment directly above the attribute explains why.
- Global `#![allow(...)]` in any crate is forbidden.
- `#[allow(dead_code)]` is only permitted on stubs explicitly planned for future implementation; add a `// TODO:` comment with the tracking issue (see §12 — the issue must exist *before* the stub lands).
- `unsafe` blocks require a `// SAFETY:` comment explaining the invariant maintained.

---

## 10. Architecture Invariants

- `rlevo-core` must have zero knowledge of any RL algorithm or environment implementation — it defines the contract only.
- **Construction is separated from the `Environment` behaviour contract.** `Environment` has **no** `new(render: bool)` method (removed by ADR 0011, accepted PR2 2026-06-03). Constructing an environment goes through the standalone `ConstructableEnv` factory trait, which is **not** a supertrait of `Environment`. The `render: bool` parameter survives on `ConstructableEnv::new` as a hint to the `record` feature for frame capture; the live TUI no longer uses it (ADR 0013 removed the env panel). Environments must not require a global runtime or external process at construction time. Do not add `new(render: bool)` to environments or to decorator/wrapper types such as `RecordingTap` or `TuiEnvTap`. (ADR 0011)
- `EpisodeStatus` is the single source of truth for episode termination; never check done-ness by any other means.
- **`SnapshotBase` is the only `Snapshot` implementation in the workspace.** It carries optional `SnapshotMetadata` via a fluent `with_metadata(self, SnapshotMetadata) -> Self` builder and overrides `Snapshot::metadata()` to expose it, so needing named reward components or positions is **not** a reason to hand-roll a `Snapshot` impl. A bespoke `SnapshotType` forfeits composition with `TimeLimit` (`wrappers/time_limit.rs` binds its inner env's `SnapshotType = SnapshotBase<D, Obs, Rew>` to mutate `.status` in place) — and every future wrapper bound to `SnapshotBase` the same way. When a family of environments needs a named shape for its snapshot, define a local type alias (`pub type LocomotionSnapshot<O> = SnapshotBase<1, O, ScalarReward>;`), not a new `Snapshot` impl. (ADR 0042)
- Replay buffers must maintain `priorities.len() == buffer.len()` as a hard invariant and enforce capacity via explicit `pop_front()` eviction — never rely on `VecDeque`'s internal capacity.
- All environment `step()` implementations must be deterministic given the same initial state and action sequence. Recording/visualisation taps (`RecordingTap`, `TuiEnvTap`) must never alter env dynamics — observing an environment must not change its trajectory. (ADR 0011/0013)
- Tensor conversion round-trips (`to_tensor` → `from_tensor`) must be lossless for all valid instances. If lossless round-trip is impossible for a type, that type must not implement `TensorConvertible`.
- **Visualisation is two products, not a library invariant.** (1) a live **metrics-only** `ratatui` TUI (no env panel), and (2) post-run replay driven by an `EpisodeRecord`, rendered by the static-HTML report from structured per-family env state. The `EpisodeRecord` seam is the canonical path for env playback. (ADR 0013, supersedes 0008)
- **`AsciiRenderable` is demoted to an optional debug helper.** It is no longer a library-level rendering contract: env families are **not** required to implement it, and there is no `Visualize` supertrait of `Environment`. Implement it only as an ad-hoc debugging aid. (ADR 0013, supersedes 0008)
- **The render type vocabulary lives in `rlevo-core::render`.** `StyledFrame`, `StyledLine`, `StyledSpan`, `SpanStyle`, `Color`, `Modifier`, `palette`, `AsciiRenderable`, and `AsciiRenderer` are all defined there. `rlevo-environments::render` is a re-export shim only — do not define new render types in `rlevo-environments`. (ADR 0009)
- **No production crate depends on viz deps.** `rlevo-core`, `rlevo-environments`, `rlevo-reinforcement-learning`, `rlevo-evolution`, and `rlevo-hybrid` must not depend (production *or* dev) on `ratatui`, `crossterm`, `leptos`, `axum`, `wasm-bindgen`, or any chart crate. `rlevo-benchmarks` is the sole exception, and only behind its `tui` / `report` / `record` feature gates. (ADR 0008)
- **`parking_lot::Mutex` is the workspace-standard lock for viz/observer shared state.** `SharedPopulationObserver` is `Arc<parking_lot::Mutex<dyn PopulationObserver>>`. Do not mix `std::sync::Mutex` into observer or record-sink types — a split lock type across sibling subsystems is a footgun (see ADR 0010). (ADR 0010)

---

## 11. Example Scope Boundary (ADR 0012)

An example belongs in `crates/rlevo/examples/` **if and only if** it imports exclusively from the five library sub-crates:

| Crate                                 | Allowed in `rlevo/examples/`? |
| ------------------------------------- | ----------------------------- |
| `rlevo-core`                          | Yes                           |
| `rlevo-environments`                  | Yes                           |
| `rlevo-evolution`                     | Yes                           |
| `rlevo-reinforcement-learning`        | Yes                           |
| `rlevo-hybrid`                        | Yes                           |
| `rlevo-benchmarks`                    | **No → `rlevo-examples`**     |
| Any tui / record / report feature dep | **No → `rlevo-examples`**     |

If an example imports from `rlevo-benchmarks` for any reason — harness invocation, suite construction, recording, or reporting — it belongs in `crates/rlevo-examples/examples/`, not in the umbrella.

`crates/rlevo-examples` is not in `default-members`. Developers opt in with `cargo run -p rlevo-examples --example <name>`.

---

## 12. Deferred Work Gets a GitHub Issue

Every deferral is filed as a GitHub issue **at the moment it is decided**, before
the change that defers it lands. This applies equally to human contributors and
coding agents. A deferral is any of: an implementation postponed, a known gap or
bug left unfixed, a follow-up split off a PR, a TODO/FIXME/stub, a QA-flagged
non-blocking finding, or an option consciously not taken.

- **File first, then defer.** Open the issue with `gh issue create` (or the web
  UI) *before* merging the code that leaves the gap. "I'll file it later" is how
  follow-ups vanish.
- **Point the code at the issue.** Any `// TODO:` / `// FIXME:` / `#[allow(dead_code)]`
  stub must name the issue number (`// TODO(#231): …`) so the deferral is
  discoverable from the source, not only from the tracker. This is the same rule
  §9 states for `dead_code` stubs.
- **A vault note, session log, or code comment is not a substitute for an issue.**
  Working notes in `docs/.private/` and session logs live only on one maintainer's
  machine (the vault is gitignored) — a follow-up recorded solely there is lost to
  every other contributor and to CI. The issue tracker is the single shared,
  durable backlog. Record context in the vault *in addition to*, never *instead
  of*, the issue.
- **Prefer fixing over filing when the fix is in scope and cheap.** An issue is
  for work genuinely deferred, not an excuse to punt a one-line fix that belongs
  in the current change. If it can be resolved now, resolve it now; file an issue
  only for what actually carries over.
- **Give the issue enough to act on later:** what is deferred, why it was deferred
  now, the affected files/sites, and the acceptance check that closes it. Label
  it (`enhancement`, `bug`, `chore`, …) and link the originating PR or ADR.

## 13. Vault and Session Protocol

- Read `rules.md` (this file) before making any implementation decision.
- Read `roadmap.md` when planning new work to ensure alignment with current milestones.
- Read relevant files in `memory/` before starting work in their domain.
- Consult `decisions/` for architectural history; never reverse a decision without writing a superseding ADR.
- Write a session log to `sessions/` at the end of every session covering: decisions made, files changed, open questions, and next steps.
- All vault writes must include the required frontmatter schema:
  ```yaml
  ---
  project: rlevo
  status: active | draft | superseded
  type: research | reference | specification | decision | memory
  date: YYYY-MM-DD
  tags: []
  ---
  ```
