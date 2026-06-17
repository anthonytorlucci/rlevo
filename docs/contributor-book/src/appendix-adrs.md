# Appendix: Architectural Decision Records

All accepted ADRs for `rlevo`, in order. Records are immutable once accepted;
superseded records are noted. When you author a new ADR, add an entry here.

| # | Title | Status |
|---|-------|--------|
| 0001 | Keep `rlevo-environments` and `rlevo-benchmarks` as separate crates; use a feature-gated `bench` adapter inside `rlevo-environments` | active |
| 0002 | Move `GenomeKind` into `rlevo-evolution`; delete dead `Fitness`/`MultiFitness` traits; drop `rlevo-core` dep from `rlevo-evolution` | active |
| 0003 | Move `memory`/`experience`/`metrics` from `rlevo-core` to `rlevo-reinforcement-learning`; fold `combinations` into `rlevo-core::util`; delete `rlevo-utils` | active |
| 0004 | Move `BenchEnv`/`BenchError`/`BenchStep`, `BenchableAgent`/`FitnessEvaluable`/`Landscape`, `Metric`/`MetricsProvider`, `SeedStream` from `rlevo-benchmarks` into `rlevo-core` | active |
| 0005 | Examples and cross-crate tests in umbrella crate | **superseded by 0012** |
| 0006 | Leptos-first visualisation; defer Bevy and native 3D | **superseded by 0008** |
| 0007 | Visualisation crates isolated from production crates; `Visualize` is not a supertrait of `Environment` | active |
| 0008 | Three-tier visualisation: `AsciiRenderable`, `ratatui` TUI, static-HTML Leptos viewer | **superseded by 0013** |
| 0009 | Hoist `StyledFrame`/`StyledLine`/`StyledSpan`/`SpanStyle`/`Color`/`Modifier`, palette, `AsciiRenderable`/`AsciiRenderer` from `rlevo-environments::render` to `rlevo-core::render` | active |
| 0010 | Unify on `parking_lot::Mutex` across viz stack; redefine `SharedPopulationObserver` over `parking_lot::Mutex` | active |
| 0011 | Remove `fn new(render: bool)` from `Environment`; add standalone `ConstructableEnv` factory trait | active |
| 0012 | Heavy viz/record/report examples move to `crates/rlevo-examples`; canonicalize three-tier test placement rule | active |
| 0013 | Collapse visualisation to two products: live metrics-only `ratatui` TUI + post-run `EpisodeRecord` static-HTML report | active |
| 0014 | Bump `FORMAT_VERSION` 5→6: expand canonical metrics, typed run-provenance fields, `EpisodeKind`, wall-clock metric, deep-RL checkpoints seam | active |
| 0015 | Extract `rlevo-metrics-registry` — `#![no_std]` zero-dep leaf crate holding the canonical metric table | active |
| 0016 | Memetic wrapper and local-search seam: `LocalSearch<B>` trait, `MemeticWrapper<B, S, L, F>`, four gradient-free searchers, `WritebackPolicy` | active |
| 0017 | `ProbabilityModel<B>` trait and EDA strategy: `EdaStrategy<B, M>`, four concrete models (UMDA/PBIL/cGA/MIMIC) | active |
| 0018 | `BayesianNetwork` (BOA) fifth probability model; `ConcatenatedTrap` deceptive landscape | active |
| 0019 | Standalone `Observable<OR>` projection trait for modality-changing POMDPs (observation tensor order ≠ state order) | active |
| 0020 | First production `Observable<OR>` consumer: synthetic pixel-over-grid env `PixelGridEnv` (`Environment<3,1,1>`) | active |

The full text of each record lives in [`docs/adr/`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/README.md).

## How to read an ADR

Each ADR in [`docs/adr/`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/README.md) follows the same structure:

- **Problem** — what situation triggered the decision.
- **Decision** — what was chosen.
- **Alternatives considered** — what was rejected and why.
- **Consequences** — what changes and what new constraints apply.

ADRs are immutable once accepted. If a decision must change, author a new ADR
that supersedes the old one — never edit the body of an accepted record.
