# Recording decisions

> **Status:** partial — the worked ADR example below is complete; the
> section-by-section authoring guidance and supersession walk-through are still
> outlined.

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

## A worked example: reading an ADR end to end

The fastest way to learn the format is to read a real one. We'll walk through
[ADR-0011, *Lift construction (`new`) off the `Environment` trait*](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0011-lift-construction-off-environment-trait.md),
section by section. It's a good first read: the decision is small enough to hold
in your head, but it still exercises every part of the format — including the one
part beginners always miss, where the *implemented* shape diverged from the
*proposed* one.

### Status — what state is this decision in?

Read this first, always. ADR-0011 is `Accepted`, with the date it landed and the
PR that implemented it. An ADR's status tells you whether you're reading a
binding constraint (`accepted`), a proposal still up for debate (`draft`), or a
decision that's been replaced (`superseded by NNNN`). If it's superseded, stop
and go read the ADR that replaced it — this one is now history, not policy.

Notice the status block also names the *chosen shape* up front ("Option 1, with
one refinement"). A reader in a hurry can get the verdict here and read the rest
only if they need the reasoning.

### Context — what forced the decision?

Context is not background colour; it's the set of facts that made the status quo
untenable. ADR-0011 shows the actual offending code: `Environment` declared
`fn new(render: bool) -> Self`, and the author names precisely *why* that's
wrong — it conflates two unrelated concerns, **behaviour** (`reset`/`step`) and
**construction**. Then it shows the cost: decorators like `RecordingTap` are
forced to write a degenerate `new` that builds a `NullSink` and silently drops
every frame.

The lesson for your own ADRs: make the Context concrete enough that a reader who
disagrees with the Decision has to argue with *facts*, not vibes. A good Context
section quotes the code, names the files, and states the pain.

### Decision — what was chosen?

ADR-0011's Decision section lays out two real options:

- **Option 1** — a separate `ConstructableEnv` factory trait.
- **Option 2** — drop the `bool` and standardise on config constructors.

This is the heart of the record. Note that it doesn't just assert "we picked
Option 1" — it describes each option concretely enough that you could implement
either. When you write a Decision, give the reader enough to understand the
fork, not just the branch you took.

### Consequences — what changes, and what's now constrained?

ADR-0011 splits consequences into **Positive** and **Negative / why deferred**.
The positives are the payoff (two silent-failure stubs gone; `Environment` now
matches the cleaner `BenchEnv` shape). The negatives are the honest costs — and
here they're load-bearing: "Wide blast radius. Every `impl Environment` ...
must change." That cost is *why the ADR was initially deferred*.

This is the section beginners under-write. A Consequences section that lists only
benefits is a sales pitch, not a decision record. State the costs you're
accepting; the next contributor inherits them.

### Implementation — where the proposal met reality

ADR-0011 has an `Implementation` section that most ADRs won't, and it teaches the
single most important thing about this format: **the decision you propose and the
decision you implement can differ, and the ADR must record the gap.**

Option 1 was *sketched* as `ConstructableEnv: Environment` (a supertrait). The
implementation deliberately did **not** do that — `ConstructableEnv` shipped as a
standalone trait, because the supertrait bound "bought nothing and would have
re-coupled the two concerns" the ADR set out to separate. The author didn't
quietly edit the Decision to match what they built; they recorded the divergence
and the reason for it.

When your implementation departs from your proposal, do the same: leave the
proposal intact and note what changed and why. That delta is often the most
useful thing in the whole record.

### References — how it connects to the rest

ADR-0011 closes by linking the ADRs it builds on — [0007](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0007-visualisation-crates-isolated-from-production-crates.md)
(render concerns stay off the production surface) and [0008](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0008-three-tier-visualisation-ratatui-live-static-report.md)
(the decorator architecture that surfaced the smell) — plus the exact source
files a reader should open next. Cross-references are what turn a pile of ADRs
into a navigable history. Always link the decisions yours extends, reverses, or
depends on.

### What to take away

Read an ADR top-to-bottom the way you'd read this one: **Status** to know if it
still binds, **Context** for the forcing facts, **Decision** for the fork,
**Consequences** for the costs you're inheriting, and — when present —
**Implementation** for where reality bent the plan. When you write your own,
aim for the same: concrete Context, a real fork in the Decision, honest costs,
and a recorded gap whenever the build diverges from the proposal.

## Outline

1. How to write each ADR section well.
2. Choosing the next number and slug; the supersession workflow.
