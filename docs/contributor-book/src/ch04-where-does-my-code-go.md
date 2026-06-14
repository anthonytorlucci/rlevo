# Where does my code go?

> **Status:** stub — prose and decision-tree diagram coming in a follow-up PR.

**Why this exists.** The three-tier test and example placement rule prevents
duplication and confusion about where canonical behaviour lives.

**Key source of truth.** `rules.md §5`, ADR-0012.

## Three-tier test placement rule

| Test type | Location |
|-----------|----------|
| Unit test for a single item | `#[cfg(test)]` module in the same `.rs` file |
| Integration test for one crate's API surface | `crates/<name>/tests/` |
| Cross-crate integration test | `crates/rlevo/tests/` |

## Example placement (ADR-0012)

| Example type | Location |
|-------------|----------|
| Lightweight (no viz, no recording) | `crates/rlevo/examples/` |
| Heavy viz / record / report | `crates/rlevo-examples/examples/` |
| Book example (compiled, test-guarded) | `crates/rlevo-examples/examples/book/` |

## Benchmark placement

`[[bench]]` entries stay in their owning crate (`crates/<name>/benches/`). No
cross-crate benchmark orchestration.

## Outline

<!-- TODO: {{#include ../../../../rules.md:anchor-placement}} -->

1. The full placement decision tree (flowchart).
2. When to promote a unit test to an integration test.
3. Book examples — the `{{#rustdoc_include}}` contract and CI guard.
4. The `publish = false` guard on `rlevo-examples`.
