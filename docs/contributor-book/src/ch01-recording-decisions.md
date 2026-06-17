# Recording decisions

> **Status:** stub — worked ADR example and authoring walk-through coming in a
> follow-up PR.

**Why this exists.** `rlevo` records the *why* behind its structural choices —
crate boundaries, trait surfaces, dependency directions — as **Architectural
Decision Records (ADRs)**. Before you change one of those boundaries, read the
record that established it; when you make a new structural decision, leave a
record so the next contributor inherits the reasoning, not just the result.

**Key source of truth.** [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md) (the hard
constraints) and [`docs/adr/`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/README.md) (the decisions).

## Two repo-resident sources of truth

| What | Where | Nature |
|------|-------|--------|
| Hard constraints and conventions | [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md) | Living — read before every change |
| Architectural decisions | [`docs/adr/`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/README.md) | Immutable once accepted |

Everything you need to contribute lives in the repository. (The maintainer keeps
private research and planning notes elsewhere; you never need them to land a PR.)

## What an ADR is

Each ADR captures a single architectural decision and follows the same shape:

- **Context** — the situation that forced the decision.
- **Decision** — what was chosen.
- **Alternatives considered** — what was rejected, and why.
- **Consequences** — what changes, and what new constraints now apply.

ADRs are **immutable once accepted.** If a decision must change, you do not edit
the old record — you author a new ADR that supersedes it, and mark the old one's
status `superseded by NNNN`. This keeps the history of *why the project is the
way it is* intact and auditable.

## When to write one

Write an ADR when a change alters the project's structure or contracts, e.g.:

- adding, removing, or re-scoping a crate;
- changing a core trait surface (`Environment`, `State`, `Strategy`, …);
- changing the dependency direction between crates;
- introducing a new cross-cutting convention.

A bug fix, a new environment that follows existing patterns, or a localized
refactor does **not** need an ADR — the existing rules and patterns already cover
it. When in doubt, open a GitHub issue or discussion to propose the design before
you write code; that conversation often becomes the ADR's *Context* section.

## How to add one

1. Author it in [`docs/adr/`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/README.md) as `NNNN-short-slug.md`, using
   the next sequential number and the structure above.
2. Add a row to the index, [`docs/adr/README.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/README.md).
3. Add a row to this book's [ADR appendix](appendix-adrs.md).
4. If the decision changes a contributor-facing constraint, update
   [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md) too.

Cross-reference related ADRs by relative link (e.g. `[0011](0011-...md)`) so the
record set stays navigable.

## Outline

1. A worked example: reading ADR-0011 end to end.
2. How to write each ADR section well.
3. Choosing the next number and slug; the supersession workflow.
