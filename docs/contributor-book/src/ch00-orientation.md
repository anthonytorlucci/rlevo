# Welcome and orientation

> **Status:** stub — prose, Mermaid crate-dependency diagram, and placement
> decision tree coming in a follow-up PR.

**Why this exists.** You want to add something to `rlevo` — a new environment,
a novel algorithm, better tooling — and you need a map before you touch the code.
This chapter gives you that map: where things live, why they live there, and what
you must not cross.

**Key source of truth.** `rules.md §1`, `CLAUDE.md`.

## Crate map

<!-- TODO: replace with Mermaid dependency diagram generated from crate list -->

| Crate | Role |
|-------|------|
| `rlevo-core` | Shared trait surface: `Environment`, `Strategy`, `State`, `Action`, `Observation`, `Landscape`, `BenchEnv` |
| `rlevo-environments` | Built-in environments (classic, games, landscapes, locomotion) |
| `rlevo-evolution` | Evolutionary algorithms and strategies |
| `rlevo-reinforcement-learning` | RL agents, replay buffers, experience storage |
| `rlevo-hybrid` | Algorithms that bridge evolution and gradient RL |
| `rlevo-benchmarks` | Harness, TUI, recording, report infrastructure |
| `rlevo-benchmarks-report-client` | Static-HTML report renderer |
| `rlevo-metrics-registry` | Zero-dep canonical metric table (ADR-0015) |
| `rlevo` | Umbrella re-export crate; cross-crate integration tests and benches |
| `rlevo-examples` | Heavy viz/record/report examples (ADR-0012) |

## Dependency rules

The critical rule: **production crates must not depend on `rlevo-benchmarks`
or any viz crate** (ADR-0007, ADR-0013). The benchmark/viz layer is a consumer
of production types, never a constraint on them.

## Placement decision tree

Before writing a single line of code, answer these questions in order:

1. **Is it a shared trait or error type?** → `rlevo-core`.
2. **Is it an environment implementation?** → `rlevo-environments`.
3. **Is it an evolutionary algorithm or operator?** → `rlevo-evolution`.
4. **Is it an RL algorithm?** → `rlevo-reinforcement-learning`.
5. **Does it bridge evolution and RL?** → `rlevo-hybrid`.
6. **Is it a benchmark harness, TUI, or recording primitive?** → `rlevo-benchmarks`.
7. **Is it a heavyweight example with recording or visualization?** → `rlevo-examples`.
8. **Is it a cross-crate integration test or bench?** → `crates/rlevo/tests/` or
   `crates/rlevo/benches/`.

If you can't answer any of these, open a discussion before writing code.

## Outline

1. Full Mermaid dependency diagram (auto-generated reference + narrative).
2. What "alpha stage" means — API stability expectations.
3. The vault protocol (preview — full chapter follows).
4. How to read an ADR — structure, status values, immutability rule.
