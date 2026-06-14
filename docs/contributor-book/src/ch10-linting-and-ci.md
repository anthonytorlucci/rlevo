# Linting and CI

> **Status:** stub — prose and `{{#include}}` anchors for workspace lint config
> coming in a follow-up PR.

**Why this exists.** The workspace lint configuration is strict by design —
`clippy::pedantic` and several custom groups are enabled at warn level. New
contributors routinely underestimate how many warnings they'll need to address.

**Key source of truth.** `rules.md §9`, root `Cargo.toml [workspace.lints]`.

## Workspace lint groups enabled

```toml
[workspace.lints.rust]
ambiguous_negative_literals = "warn"
missing_debug_implementations = "warn"
redundant_imports = "warn"
unsafe_op_in_unsafe_fn = "warn"

[workspace.lints.clippy]
cargo = "warn"
complexity = "warn"
correctness = "warn"
pedantic = "warn"
perf = "warn"
style = "warn"
suspicious = "warn"
```

## The `#[allow]` policy

`#[allow(clippy::...)]` is permitted only when:

1. The lint is a false positive in the specific location.
2. The suppression includes a comment explaining why.

Never suppress a lint workspace-wide without an ADR.

## `unsafe` contracts

Any `unsafe` block must be preceded by a `// SAFETY:` comment that states the
invariant that makes the block safe. `unsafe_op_in_unsafe_fn` is enabled, so
inner unsafe operations inside an `unsafe fn` must also carry `// SAFETY:`
comments.

## Running the lints locally

```bash
cargo clippy --all-targets --all-features
```

CI runs this command on every PR. A PR with lint warnings will not be merged.

## Outline

<!-- TODO: {{#include ../../../../rules.md:anchor-linting}} -->

1. The full workspace lint configuration walk-through.
2. Common pedantic lints and how to satisfy them.
3. `#[allow]` — when it's legitimate and how to document it.
4. `unsafe` — the `// SAFETY:` contract.
5. How to add a new lint or change a lint level (requires discussion).
