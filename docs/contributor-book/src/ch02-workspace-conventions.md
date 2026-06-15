# Workspace conventions

> **Status:** stub — prose and `{{#include}}` anchors for `rules.md` coming in
> a follow-up PR.

**Why this exists.** The workspace has a consistent set of naming, visibility,
and dependency-inheritance rules. Following them means your code looks and builds
like the rest of the project without special-casing.

**Key source of truth.** `rules.md §2`, root `Cargo.toml`.

## Naming conventions

- Crate names: `rlevo-<domain>` (kebab-case).
- Module names: `snake_case`.
- Types and traits: `PascalCase`.
- Constants and statics: `SCREAMING_SNAKE_CASE`.

## Visibility rules

- Public API items that cross crate boundaries: `pub`.
- Items used only within a module or crate: `pub(crate)` or no qualifier.
- Never `pub(super)` except in test helpers.

## Cargo inheritance

All `[package]` metadata fields (`version`, `authors`, `edition`, `license`,
`repository`) are inherited from `[workspace.package]` via `.workspace = true`.
All shared dependencies are declared in `[workspace.dependencies]` and
referenced in crate `Cargo.toml` files without repeating the version.

## Outline

<!-- TODO: \{{#include ../../../../rules.md:anchor-workspace-conventions}} -->

1. Full naming convention table.
2. Visibility policy — when `pub(crate)` and when `pub`.
3. How to add a new crate to the workspace.
4. Cargo feature naming — how to name and gate optional functionality.
5. The `publish = false` rule for internal crates.
