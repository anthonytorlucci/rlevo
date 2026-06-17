# Dependency discipline

> **Status:** stub — prose and approved dependency table coming in a follow-up PR.

**Why this exists.** Every new dependency is a build-time cost, a maintenance
burden, and a potential supply-chain risk. The approved table and the ADR
requirement exist to make that cost visible and intentional.

**Key source of truth.** [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md) §8.

## The approved dependency table

<!-- TODO: \{{#include ../../rules.md:anchor-approved-deps}} -->

Key workspace dependencies and their purpose:

| Crate | Version | Purpose |
|-------|---------|---------|
| `burn` | 0.21.0 | Tensor operations, neural networks, training |
| `rand` | 0.9.2 | Host-side RNG (all evolutionary operators) |
| `serde` | 1.0 | Serialization for records and configs |
| `tracing` | 0.1 | Structured logging |
| `parking_lot` | — | `Mutex` for shared observer state (ADR-0010) |
| `approx` | — | Floating-point assertions in tests |

## Adding a new dependency

1. Check if an existing approved dep can solve the problem.
2. If not, author an ADR explaining the need, the alternative considered, and
   the maintenance posture.
3. Add the dep to `[workspace.dependencies]` in the root `Cargo.toml`.
4. Reference it from the owning crate without repeating the version.

## The `parking_lot` convention (ADR-0010)

Use `parking_lot::Mutex` everywhere in `rlevo-evolution` shared observer
state. Do not mix `std::sync::Mutex` and `parking_lot::Mutex` in the same
lock-ordering group.

## Outline

1. Why dependency minimalism matters for a Rust library.
2. The full approved dependency table with rationale.
3. How to propose a new dependency — the ADR template.
4. How to audit a dep for security and maintenance status.
