# Documentation standards

> **Status:** stub — prose and `{{#include}}` anchors for `rules.md §6` coming
> in a follow-up PR.

**Why this exists.** `rlevo` enforces a zero-exception public documentation
policy: every public item must have a doc comment. This chapter makes the
required structure concrete so you don't have to infer it from existing code.

**Key source of truth.** `rules.md §6`.

## Zero-exception policy

Every `pub` item — struct, enum, trait, function, type alias, constant — must
have a `///` doc comment. CI will fail a PR that adds a public item without
documentation.

## Doc-comment structure

```rust,no_run
/// One-line summary sentence — what this *is*, in imperative mood.
///
/// Optional extended description. Explain *why* the item exists, any
/// non-obvious invariants, and what callers should watch out for.
///
/// # Panics
///
/// List any panic conditions. Omit the section if the item cannot panic.
///
/// # Errors
///
/// List error variants for `Result`-returning items.
///
/// # Examples
///
/// ```rust
/// // A short, runnable example.
/// ```
```

## What not to document

- Implementation details that are obvious from the code.
- The current task, PR, or issue number ("added for issue #42").
- Caller names ("used by `foo`").

These belong in commit messages and PR descriptions, not in source comments.

## Module-level documentation

Every `mod.rs` or `lib.rs` must have a `//!` module doc at the top. It should
explain the module's role in the crate and list the primary types or traits it
exports.

## Generating and checking docs locally

```bash
cargo doc --workspace --no-deps --open
```

Check for broken intra-doc links:

```bash
RUSTDOCFLAGS="-D rustdoc::broken-intra-doc-links" cargo doc --workspace --no-deps
```

## Outline

<!-- TODO: {{#include ../../../../rules.md:anchor-documentation}} -->

1. The zero-exception policy — what it covers and why.
2. Summary sentences — imperative mood, one line.
3. The canonical sections — Panics, Errors, Examples.
4. Module docs — `//!` structure.
5. Intra-doc links — `[Type]`, `[method]`, `[crate::module::Item]`.
6. Generating docs and the broken-link check.
