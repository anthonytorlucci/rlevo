# Trait design

> **Status:** stub — prose and `{{#include}}` anchors for `rules.md` coming in
> a follow-up PR.

**Why this exists.** `rlevo`'s public API is almost entirely trait-based. The
rules for designing and evolving traits are stricter here than in a typical
project because breaking a trait breaks every downstream implementation.

**Key source of truth.** [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md) §3–4, §7.

## Invariants that cannot be broken

1. Const generic parameters (`R`, `SR`, `AR`) encode dimensional correctness at
   the type level. Never relax these to runtime checks.
2. `shape()` must return an array whose length equals the const generic rank.
3. Trait objects (`dyn Trait`) are only permitted where the trait is explicitly
   object-safe and the callsite genuinely needs dynamic dispatch.

## Error handling policy

- `EnvironmentError` for all environment operations.
- `StateError` for state validation.
- Use `Result<T, ErrorType>` for fallible operations; never `panic!` in library
  code for recoverable errors.
- Implement `std::error::Error` and `Display` for all custom error types.

## Adding a method to an existing trait

Adding a required method is a breaking change. Always provide a default
implementation or add a new trait. Open an ADR before breaking any public trait.

## Outline

<!-- TODO: \{{#include ../../rules.md:anchor-trait-design}} -->

1. The const-generic constraint system — how `R`, `SR`, `AR` flow through
   `State`, `Observation`, `Action`, `Snapshot`.
2. The `Send + Sync` requirement — why it's required and what it rules out.
3. Object safety — which traits are intentionally non-object-safe and why.
4. Error hierarchy — `EnvironmentError`, `StateError`, and the extension pattern.
5. Stability guarantees — what "alpha" means for trait evolution.
