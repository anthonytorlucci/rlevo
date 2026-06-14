# The vault protocol

> **Status:** stub — prose and session-log template coming in a follow-up PR.

**Why this exists.** `rlevo`'s design decisions, specifications, and session logs
live in an Obsidian vault, not in the repository. This chapter explains the
protocol so you never write a decision without leaving a record.

**Key source of truth.** `rules.md §12`, `decisions/`.

## The vault layout

```
projects/rlevo/
  CONTEXT.md        ← symlinked from repo CLAUDE.md
  rules.md          ← hard constraints (read before touching crates/)
  roadmap.md        ← current milestones and priorities
  memory/           ← persistent project understanding
  decisions/        ← immutable ADRs (one file per decision)
  research/         ← active investigation notes
  specs/            ← forward-looking feature specifications
  sessions/         ← per-session logs
  drafts/           ← pre-publication book drafts
```

## The spec-first rule

**On `main`, author a vault spec before editing code; branch before touching
`crates/`.** This is not a suggestion — it is the order in which work happens.

The spec answers: what is the problem, what is the design, what are the
alternatives considered, and what are the open questions. Once the spec is
accepted you branch and implement.

## Frontmatter schema

Every vault write requires this frontmatter:

```yaml
---
project: rlevo
status: active | draft | superseded
type: research | reference | specification | decision | session-log | memory
date: YYYY-MM-DD
tags: []
---
```

## Architectural Decision Records

ADRs live in `decisions/` and are **immutable once accepted**. If a decision is
superseded, its status becomes `superseded` and the new ADR references it. Never
edit the body of an accepted ADR.

The ADR appendix lists all current records with one-line summaries. When you
write a new ADR:

1. Author it in `decisions/` with the next sequential number.
2. Add a one-line pointer in the CLAUDE.md `## Decisions` section.
3. Update the ADR appendix in this book.

## Session logs

Write a session log to `sessions/` at the end of every working session covering:
decisions made, files changed, open questions, and next steps.

## Outline

1. Full vault directory walk-through with examples.
2. How to write a spec — the five required sections.
3. How to write an ADR — problem, decision, alternatives, consequences.
4. Session log template — fill in, don't skip.
5. The wikilink convention and how the preprocessor resolves it in this book.
