# Architectural Decision Records

Immutable records of the architectural decisions behind `rlevo`. Once accepted,
an ADR is not edited — a later decision supersedes it, and the superseded record
is annotated rather than deleted. Read these for the *why* behind the crate
boundaries and trait design before proposing structural changes.

When you make an architectural decision, add a new numbered file here following
the existing format (`Status` / `Context` / `Decision` / `Consequences` /
`Alternatives considered` / `References`).

> Drafts and in-flight proposals live in the maintainer's working notes; an ADR
> lands here only once accepted. The repo copy is canonical.

| # | Decision |
|---|----------|
| [0001](0001-keep-environments-and-benchmarks-separate.md) | Keep `rlevo-environments` and `rlevo-benchmarks` as separate crates; use a feature-gated `bench` adapter inside `rlevo-environments` instead of merging. Decides to keep `rlevo-environments` and `rlevo-benchmarks` as separate crates rather than merging them, preserving each crate's disjoint dependency cone (physics for environments, `rayon`/`tracing` for benchmarks). To make the boundary practical, it ships a feature-gated `bench` module in `rlevo-environments` providing a `BenchAdapter` and preset `Suite` factories, plus a `Result`-based `BenchEnv` API for truthful error semantics. |
| [0002](0002-collapse-evolution-traits-into-rlevo-evolution.md) | Collapses `rlevo-core::evolution`'s EA traits into `rlevo-evolution` and removes that crate dependency. Only `GenomeKind` had a single real consumer; `Fitness` and `MultiFitness` were dead, aspirational placeholders with no usage. The refactor moves `GenomeKind` into `rlevo-evolution`, deletes the two dead traits, and removes the `rlevo-core` dep edge, leaving core as RL-only and evolution self-contained. RL/EA composition now happens at the `rlevo-hybrid` layer. |
| [0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md) | Moves three RL-specific modules (`memory`, `experience`, `metrics`) out of `rlevo-core` into `rlevo-reinforcement-learning`, since all are consumed only there — premature centralization by the ADR 0002 test. It also folds the empty `rlevo-utils` crate (a single orphan `combinations()` function) into `rlevo-core::util` and deletes the crate, shrinking the workspace from eight crates to seven. The move uses `git mv` for history, rewrites import paths, drops the orphan `ringbuffer` dep, and relocates the integration test. Two non-decisions are recorded: `rlevo-reinforcement-learning` and `rlevo-environments` stay separate (zero runtime coupling), and `rlevo-hybrid` remains the EA↔RL bridge. Speculative zero-consumer trait shells stay in `rlevo-core` as roadmap markers rather than being deleted. |
| [0004](0004-move-bench-traits-into-rlevo-core.md) | Relocates the ~184-line benchmark trait surface (the `env`, `agent`, and `seed` modules) from `rlevo-benchmarks` into `rlevo-core`, alongside the `Metric`/`MetricsProvider` types. It also converts `BenchError` from stringly-typed errors to typed wrappers around `EnvironmentError`, and rewrites import sites in `rlevo-evolution` and `rlevo-environments` accordingly. The goal is a strict-DAG dependency graph with no dep-cone bloat in evolution, while backward-compat shim modules keep existing callers compiling. Alternatives like creating a separate `rlevo-bench-traits` crate were rejected. A later note supersedes the local splitmix64-mixer decision in favor of ADR 0033's shared mixer. |
| [0005](0005-examples-and-cross-crate-tests-in-umbrella.md) | Moved all examples and cross-crate integration tests into the umbrella `rlevo` crate, keeping only crate-local unit and single-crate integration tests in their owning subcrates. This eliminated dependency cycles caused by dev-dependency edges that mirrored prod-dependency edges, notably `rlevo-benchmarks` dev-depending on `rlevo-environments[bench]`. Combined with ADR 0004, it yields a strictly acyclic workspace and a single discovery point via `cargo run -p rlevo --example <name>`. Accepted costs include subcrates losing standalone runnability and a larger umbrella `Cargo.toml`. The ADR is now superseded by ADR 0012, which refines the examples split and canonicalizes the test-placement rule. |
| [0006](0006-leptos-first-visualisation-defer-bevy.md) | (superseded) Decided to build rlevo's visualisation layer as a **Leptos web client** served by an embedded `axum` server, and to **defer adopting Bevy** for 3D this milestone. Only the `locomotion` family genuinely needs 3D, so it gets a temporary 2D sagittal-plane skeleton projection instead. The decision favoured shareable URLs, a richer plot ecosystem, and lower author activation energy over Bevy's native 3D strength. It retained a transport-agnostic `rlevo-viz-core` so a future 3D/native viewer could reuse the core unchanged. Its conclusions were later replaced by ADR 0008 and 0013, though the "no Bevy" and production-crate-isolation principles carried forward. |
| [0007](0007-visualisation-crates-isolated-from-production-crates.md) | Keeps visualisation crates (`rlevo-viz-core` and `rlevo-viz-web`) out of the dependency graph of all production crates. `rlevo-viz-core` may depend only on `rlevo-core`, while `rlevo-viz-web` may additionally depend on `rlevo-environments` via feature-gated adapters; neither may be a prod or dev dependency of any production crate. The `Visualize` trait lives in `rlevo-viz-core`, environments opt in with adapters located in `rlevo-viz-web`, and algorithm crates stay visualisation-agnostic, emitting metrics through existing callback seams. This avoids bloated dependency cones, recompilation, and WASM contamination across the workspace. The ADR is now superseded, though its isolation principle persists in later ADRs. |

| [0008](0008-three-tier-visualisation-ratatui-live-static-report.md) | Adopts a three-tier visualisation architecture for `rlevo`, splitting visualisation into the library tier (`AsciiRenderable`), a live `ratatui` TUI, and a static-HTML Leptos report — superseding the single Leptos-server model of ADRs 0006/0007. The live tier wraps benchmark runs in a terminal dashboard (feature-gated in `rlevo-benchmarks`), while the report tier renders post-run playback from a recorded `EpisodeRecord` file into one self-contained HTML. A core principle is that no production crate depends on viz crates; all viz lives behind features in the leaf `rlevo-benchmarks` crate. The ADR is **superseded by 0013**, which collapses the tiers to a live metrics TUI plus post-run records/report, removes the live env panel, and demotes `AsciiRenderable` to an optional debug helper. The `EpisodeRecord` seam and production-crate isolation rules remain in force. |

| [0009](0009-move-styled-render-into-rlevo-core.md) | Moves the styled-output render surface (`StyledFrame`, `StyledLine`, `StyledSpan`, `SpanStyle`, `Color`, `Modifier`, the `palette` module, and `AsciiRenderable`/`AsciiRenderer`) from `rlevo-environments::render` into `rlevo-core::render`. This was driven by a Cargo cyclic-dependency failure: `rlevo-benchmarks` (for the live ratatui TUI) needed the `AsciiRenderable` bound and `StyledFrame`, but `rlevo-environments` already optionally depends on `rlevo-benchmarks`. Since the trait and types are structural and env-agnostic, they pass the ADR 0004 "shared vocabulary" test for living in core. `rlevo-environments::render` is preserved as a pure re-export shim so all per-env impls compile unchanged. Consequence: the cycle is eliminated, the workspace stays a strict DAG with viz deps confined to `rlevo-benchmarks`, and the `StyledFrame → ratatui` conversion ships as free functions instead of a `From` impl due to orphan rules. |

| [0010](0010-unify-on-parking-lot-across-viz-stack.md) | Unifies the visualisation stack on `parking_lot::Mutex` after a repair pass found ~10 of 17 `rlevo/examples/viz/` examples silently failing to compile (they were `required-features`-gated and never built/tested). The root cause was a lock-type split: record-sink producers use `Arc<parking_lot::Mutex>`, while the `SharedPopulationObserver` alias uses `std::sync::Mutex`, forcing one example to import both types. The ADR redefines the observer alias over `parking_lot::Mutex`, removes the dual-import smell, converges the remaining sphere examples onto the same type, and drops `.lock().unwrap()` calls. It accepts the cost of a public API change and loss of lock poisoning, since observer state is non-critical telemetry. An example build+clippy CI job is tracked separately as the higher-leverage companion fix. |

| [0011](0011-lift-construction-off-environment-trait.md) | Removes the `new(render: bool)` constructor from the behaviour-focused `Environment` trait, which previously forced viz decorators like `RecordingTap` and `TuiEnvTap` to implement degenerate, silent-failure stubs. Construction is lifted into a standalone `ConstructableEnv` trait in `rlevo-core` (not a supertrait of `Environment`), keeping behaviour and construction concerns decoupled. All ~31 concrete env `new` bodies in `rlevo-environments` moved into `impl ConstructableEnv` blocks, and the `NullSink`/dead-channel stubs in `rlevo-benchmarks` were deleted. This aligns `Environment` with the clean `BenchEnv` shape and reinforces ADR 0007 (rendering stays off the production env surface). |

| [0012](0012-split-heavy-examples-into-rlevo-examples.md) | Creates a new workspace crate, `rlevo-examples`, to host application-tier examples — any that import `rlevo-benchmarks` or the viz/record/report feature stack (which pull in server and dependency-heavy deps like ratatui and axum). The umbrella `crates/rlevo/` keeps only lightweight examples that import exclusively from the five library sub-crates, and its `[dev-dependencies]` block is pruned accordingly. It also codifies the canonical three-tier test placement rule: in-source unit tests, per-crate `tests/` for single-crate integration tests, and flat `crates/rlevo/tests/` for cross-crate tests. `rlevo-examples` is added to workspace members but not default-members, so builds opt in via `cargo run -p rlevo-examples`. The main breaking change is that viz example commands change from `-p rlevo` to `-p rlevo-examples`. |

| [0013](0013-metrics-only-live-tui.md) | Reframes rlevo's visualisation from three tiers into two products, superseding ADR 0008: a live metrics-only `ratatui` TUI and a post-run `EpisodeRecord`-driven HTML report. The live TUI drops its env panel entirely, answering only "is it learning?" via learning curves, while the report answers "what did it learn?" with publication-quality per-family SVG/canvas playback from structured state. `AsciiRenderable` is demoted from a mandatory library-wide invariant to an optional debug-helper trait, since neither product consumes it. ADR 0008's `EpisodeRecord` seam and the rule that production crates never depend on viz (confined to `rlevo-benchmarks`) are preserved. Reversal criteria keep a cheap opt-in live env panel available if a mid-run debugging need arises. |

| [0014](0014-record-schema-v6-single-agent-richness-and-provenance.md) | Bumps the on-disk record format to `FORMAT_VERSION = 6`, extending (not superseding) ADR 0013. It fixes three gaps found by surveying RLlib, CleanRL, SB3, Tianshou, Gymnasium, and reproducibility literature: thin metric coverage, untyped/incomplete provenance, and no eval-vs-training distinction. The decision lands single-agent RL metric richness (e.g. `explained_variance`, per-iteration returns, DQN/SAC losses), typed provenance fields on `RunManifest`, an `EpisodeKind` train/eval tag, an `EpisodeRecordHeader`/wall-clock seam, and a learner-checkpoint reference seam. Multi-agent, variable-topology neuroevolution, and Bayesian-network records are deliberately deferred to future additive bumps. All changes follow binding invariants: additive-only `Option`/non-exhaustive enum fields, `BTreeMap`, typed fields over the string map, and mirrored consts/wire types enforced at compile time. Accepted cost: v5 files no longer decode under v6, and producer wiring for the new metrics is deferred. |

| [0015](0015-shared-typed-metric-registry-crate.md) | Introduces a new leaf crate `rlevo-metrics-registry` to solve two problems: (1) the canonical metric list was duplicated between `rlevo-benchmarks` and the WASM report client with no drift guard, causing silent demotion of new metrics; (2) the flat string list couldn't carry semantics, forcing the client to maintain three hardcoded shadow taxonomies for grouping, cadence, and titles. The solution extracts a `#![no_std]`, zero-dependency crate containing a typed `MetricDescriptor` table with `MetricKind` (Rl/Eo/Shared), `Cadence`, title, and unit. Both crates now depend on this single source of truth—the benchmarks crate becomes a thin re-export, and the report client deletes its fork, deriving all UI logic from registry lookups. No `FORMAT_VERSION` bump is needed; production crates remain isolated (they still emit only `tracing` field names). |

| [0016](0016-memetic-wrapper-and-local-search-seam.md) | Introduces a **memetic wrapper** and **host-side local-search seam** that enables hybridizing population-level evolutionary search with per-individual local refinement. The design adopts a `MemeticWrapper<B, S, L, F>` that implements `Strategy<B>` by composing an inner strategy `S` with a `LocalSearch<B>` trait and a fitness function `F` held behind a `parking_lot::Mutex<F>` (since `Strategy::tell` is `&self` but `F` requires `&mut self`). The wrapper drives refinement inside its `tell`, using a two-stream RNG scheme (`seed_stream` with `SeedPurpose::LocalSearch` and `SeedPurpose::Replacement`) so writeback policy (`Lamarckian`, `Baldwinian`, `Partial(p)`) is bit-identically overlay-compatible. Four searchers ship v1 (`HillClimbing`, `NelderMead`, `SimulatedAnnealing`, `RandomRestart`), all monotone-non-worsening over host `Vec<f32>` genomes. The `EvolutionaryHarness` stays memetic-unaware and holds a second `F` instance; this additive design preserves zero blast radius on the frozen public surface. Reversal criteria address the two-fitness-instance edge, per-refine re-eval cost, and calibration margins. |

| [0017](0017-probability-model-trait-and-eda-strategy.md) | Introduces a `ProbabilityModel<B>` trait and generic `EdaStrategy<B, M>` to enable Estimation-of-Distribution Algorithms (EDAs) as first-class `Strategy<B>` implementations in the evolutionary computation crate. The trait defines `fit` (incremental model fitting with optional prior state for algorithms like PBIL/cGA and future CMA-ES) and `sample` (host-side RNG sampling via `rand_distr`, not Burn's seedless PRNG kernels) with a dedicated `SeedPurpose::EdaSampling` stream for reproducibility. Four concrete univariate/chain models ship (UnivariateGaussian/UMDA, UnivariateBernoulli/PBIL, CompactGenetic/cGA, DependencyChain/MIMIC), while BOA is deferred. The design is purely additive—zero changes to the frozen `Strategy` surface, harness, or manifests—and reuses ADR 0016's NaN chokepoint and deterministic truncation selection. The `prev: Option<&State>` seam in `fit` deliberately supports incremental models and CMA-ES evolution-path state without a third `init_state` method, and `EdaStrategy` remains pure/lock-free because it consumes pre-evaluated populations in `tell` without interior mutability. |

| [0018](0018-boa-bayesian-network-and-concatenated-trap.md) | Introduces the **BOA (Bayesian Optimization Algorithm)** as a new Estimation-of-Distribution Algorithm (EDA) model in `rlevo-evolution`, extending the frozen `ProbabilityModel<B>` trait from ADR 0017 without any trait changes. BOA learns a **Bayesian network (DAG)** over genes each generation to capture higher-order dependencies that univariate (UMDA, PBIL, cGA) and first-order chain (MIMIC) models miss — critical for deceptive landscapes where per-gene statistics drive the population toward the deceptive basin. The decision ships `ConcatenatedTrap`, the canonical Deb & Goldberg deceptive trap function (summed over contiguous `block_size`-bit blocks), as the workspace's first discrete landscape and the discriminating benchmark. Structure learning uses **BIC scoring on raw MLE counts** (no smoothing), with a greedy lexicographic edge-addition search bounded by `max_parents = 3`; the BIC penalty ($\frac{1}{2} \cdot \ln N \cdot 2^{|Pa|}$) replaces the ad-hoc significance filter used in MIMIC with a principled complexity guard. Smoothing (`s = 1` default) applies only to CPT estimation for sampling, never to structure selection. `fit()` is non-incremental (relearns from scratch), matching canonical BOA. The convergence gate is empirically pinned at `pop = 2000`, selection ratio `0.3`, 60 generations — at this budget BOA solves trap-5 10/10 while UMDA and MIMIC stall (median cost $\ge 2$). The trait held without modification, validating the ADR 0017 surface for learned-structure models; accepted costs include host-side $O(D^2 \cdot N \cdot \kappa)$ fit time and the order-of-magnitude larger population requirement, both consistent with literature. |

| [0019](0019-observable-projection-trait.md) | Introduces the `Observable<OR>` trait to `rlevo-core::state` as a standalone, additive projection trait for modality-changing POMDPs where observation tensor order differs from state tensor order (e.g., compact RAM state → pixel observations). The trait decouples observation order `OR` from state order `SR` by providing an infallible `project(&self) -> Self::Observation` method, distinct from `State::observe()` which preserves order. `OR` is a const generic (not an associated const) to drive the `Observation<OR>` bound, and the trait deliberately avoids a blanket impl over `State` to prevent coherence lock-in and method ambiguity. The `Environment` contract already supports `R != SR`; this trait simply gives the projection a typed, tested home. A cross-crate test validates `Environment<2, 1, 1>` end-to-end, and real modality-changing environments will consume the seam. |

| [0020](0020-synthetic-pixel-over-grid-env.md) | Introduces the first real consumer of the `Observable<OR>` trait (from ADR 0019): a synthetic **pixel-over-grid** navigation environment implemented as `Environment<3, 1, 1>` in `crates/rlevo-environments/src/pixel_grid.rs`. The environment uses a rank-3 RGB observation (`[20, 20, 3]`) projected from a rank-1 latent state (`[2]`), exercising the exact `R != SR` modality-changing path that a future Atari/ALE backend would use (which is deferred behind a feature gate due to C++ FFI and ROM toolchain concerns). The `PixelGridState` implements both `State<1>` (trivial `observe()`) and `Observable<3>` (rendering `project()` → `PixelObservation`), with snapshots built exclusively via `project()`. The observation round-trips through `TensorConvertible<3, B>` so Burn policies can consume the projected image directly. Dynamics are 4-way Cartesian with wall clamping, terminated on goal with a decaying success reward, truncated at max steps. Fixed dimensions (`GRID_SIDE=5`, `CELL_PX=4`, `CHANNELS=3`) are used for v1; configurable parameters are deferred. The module lives as a standalone concept module (`pixel_grid.rs`), deliberately decoupled from the `grids/` family (which is egocentric, `R == SR`, and shares no core). |

| [0021](0021-cma-es-placement-and-self-contained-strategy.md) | Finalizes the placement and design of CMA-ES and CMSA-ES as two flat, self-contained `Strategy<B>` implementations in `crates/rlevo-evolution/src/algorithms/{cma_es,cmsa_es}.rs` rather than an `es_advanced/` submodule, following the project's convention that single-file algorithm families remain flat siblings. Both algorithms implement `Strategy<B>` directly and deliberately do **not** instantiate `ProbabilityModel<B>`, refining ADR 0017's anticipation that CMA-ES would reuse the trait; the authors argue that CMA-ES's identity lies in evolution-path machinery (CSA, rank-1/$\mu$ covariance updates) and CMSA-ES's in per-individual log-normal $\sigma$ self-adaptation—neither of which fits the `fit`/`sample` density-estimation seam, and forcing them through it would leak strategy state into the model. A hand-rolled cyclic Jacobi eigensolver handles host-side decomposition and sampling (no new dependency), `SeedPurpose::CmaSampling = 11` isolates the multivariate draw, and CMSA-ES reuses the classical ES $\sigma$ rule ($\tau = 1/\sqrt{2D}$). The change is purely additive with zero impact on the frozen API surface. |

| [0022](0022-tensorgenome-gat-population-storage.md) |  |

| [0023](0023-objective-sense-and-maximize-convention.md) |  |

| [0024](0024-rlevo-test-support-dev-crate.md) |   |

| [0025](0025-stateful-policy-rollout-contract.md) |   |

| [0026](0026-shared-config-validation-convention.md) |   |

| [0027](0027-bounds-newtype-for-closed-ranges.md) |   |

| [0028](0028-tensor-batch-conversion-seam.md) |   |

| [0029](0029-host-rng-seeding-convention.md) |   |

| [0030](0030-permutation-tensorgenome-and-population-nonempty-invariant.md) |   |

| [0031](0031-probability-rate-newtypes.md) |   |

| [0032](0032-neat-opaque-id-newtypes.md) |   |

| [0033](0033-share-splitmix64-mixer-across-core-and-evolution.md) |   |

| [0034](0034-fitness-hygiene-chokepoint-convention.md) |   |

| [0035](0035-coupled-fitness-sense-and-coevolution-canonicalization.md) |   |

| [0036](0036-adopt-proptest-for-property-tests.md) |   |

| [0037](0037-external-force-lifetime-and-substep-actuation.md) |   |

| [0038](0038-continuous-action-components-const.md) |   |

| [0039](0039-box2d-states-own-markov-dofs.md) |   |

| [0040](0040-environment-config-error-and-terrain-output-contract.md) |   |

| [0041](0041-rapier3d-joint-actuation-and-contact-wrench-semantics.md) |   |

| [0042](0042-snapshotbase-carries-optional-metadata.md) |   |

| [0043](0043-grid-observation-contract.md) |   |

| [0044](0044-post-terminal-step-is-an-error.md) |   |

| [0045](0045-landscape-bounds-is-a-search-box.md) |   |

| [0046](0046-slot-newtype-replaces-option-take-around-learn-step.md) |   |

| [0047](0047-sensor-relocates-emission-model-to-environment.md) |   |

| [0048](0048-partial-episode-bootstrapping-in-gae.md) |  |

| [0049](0049-ppo-gaussian-log-std-is-bounded.md) |  |

| [0050](0050-replay-strategy-seam.md) |  |

| [0051](0051-replay-kind-dispatch-and-validated-importance-exponent.md) |  |

| [0052](0052-hostrow-supertrait-splits-layout-from-backend.md) |  |

| [0053](0053-bounded-action-per-component-bounds.md) |  |

| [0054](0054-policy-head-construction-is-fallible.md) |   |

| [0055](0055-config-invariant-enforcement-allocation.md) |  |

| [0056](0056-non-finite-loss-skip-and-warn-guard.md) |  |

| [0057](0057-target-soft-update-path-is-fallible.md) |  |

| [0058](0058-target-update-type-unifies-cadence-and-tau.md) |   |

| [0059](0059-target-update-cadence-counts-gradient-updates.md) |   |

| [0060](0060-config-values-must-be-finite.md) |   |

| [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) |   |

| [0062](0062-grid-layout-fidelity-and-no-dead-rng.md) |   |

| [0063](0063-grid-visibility-occlusion.md) |   |

| [0064](0064-observation-carries-no-serde-supertrait.md) |   |

| [0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md) |   |

| [0066](0066-clamp-nan-behavior-is-backend-specific-pin-with-is-nan.md) |   |

| [0067](0067-non-finite-observations-are-dropped-at-replay-ingestion.md) |   |

| [0068](0068-bounds-strictness-enforcement-is-crate-asymmetric.md) |   |

| [0069](0069-sanitized-fitness-is-reduced-in-f64.md) |   |

| [0070](0070-avg-score-transits-non-finite-scores-the-hardened-mean-is-additive.md) |   |

| [0071](0071-best-score-latches-plus-infinity-the-finite-best-is-additive-and-counted.md) |  |

| [0072](0072-loss-skips-are-counted-per-site-and-surfaced-as-a-metric.md) |  |

| [0073](0073-carried-item-is-stamped-into-the-agents-own-view-cell.md) |  |

| [0074](0074-panic-contract-table-is-mechanically-checked.md) |  |
