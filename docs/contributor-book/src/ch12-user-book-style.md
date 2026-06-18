# User-book style

**Why this exists.** This chapter is the canonical style guide for the
[user / researcher guide](https://github.com/anthonytorlucci/rlevo/tree/main/docs/user-book)
(`docs/user-book/`). It governs *prose* — voice, structure, notation, spelling
— and is distinct from [Documentation standards](ch11-documentation-standards.md),
which governs rustdoc (`///`) comments in source. When the two touch — for
example a doc-comment that mirrors a user-book explanation — keep them in sync.

## Authorial voice

Write as a lead architect of `rlevo` with deep academic grounding in
evolutionary computation, deep learning, and reinforcement learning. The tone
is **authoritative, precise, and structured, with pedagogical intent**:
technical and exact, systematic and methodical, instructional and explanatory,
objective and clear. State conventions as facts; avoid conversational openers
("a word of warning if you come from…"), hedging, and second-person filler.

## The three tiers

Every page belongs to one tier, and the tier sets how much code the page
carries and how deep the prose goes.

| Tier | Role | Code density |
| ---- | ---- | ------------ |
| **Chapter** | High-level overview and orientation | Sparse — only enough to give context (trait method signatures, generic arguments). |
| **Section** | Technical, focused treatment of one topic | Verbose — real, codebase-faithful signatures and runnable snippets. |
| **Appendix** | Very technical deep dive | Maximal — full update equations, pseudocode, derivations; written for the reader who wants to *understand*, not copy-paste. |

Match the page's depth and code budget to its tier. A chapter that drowns in
code, or an appendix that hand-waves the mathematics, is mis-tiered.

## `rlevo` is the subject

This is a guide *to `rlevo`*, so `rlevo` stays front and centre: frame every
concept in terms of the crate's types, traits, and modules rather than as
generic theory.

**Comments are for side notes, never for `rlevo` explanations.** Anything
load-bearing about how `rlevo` works belongs in prose. Reserve code comments
for material *outside* the document's scope — where to find more information,
or a deliberate deviation from the literature.

## Notation: prefer KaTeX over raw Unicode

The user-book renders mathematics with KaTeX (wired in
`docs/user-book/theme/head.hbs`). Use `\\( … \\)` for inline math and ```` ```math ````
fences for display equations. **Do not** enable mdBook's `mathjax-support`.

Prefer KaTeX over raw Unicode for mathematical notation:

- **Math symbols and expressions → KaTeX.** Greek scalars and parameters
  (\\(\sigma\\), \\(\alpha\\), \\(\mu\\), \\(\lambda\\), \\(\tau\\), \\(k\\),
  \\(p\\)), ES variant tuples (\\((\mu+\lambda)\\), \\((\mu,\lambda)\\),
  \\((1+1)\\)), and inline expressions (\\(1 - x\\), \\(\sigma \cdot
  \mathcal{N}(0,1)\\)). KaTeX renders correctly inside table cells too.
- **Code identifiers → backticks.** Field and parameter names as they appear in
  source (`tournament_size`, `top_k`, `pop_size`) and tensor shapes
  (`(N, D)`, `(N,)`).
- **Proper operator names → plain text.** A named operator such as BLX-α is a
  label, not an equation; leave it as plain text.

## British English

Use British spelling throughout: *organise*, *colour*, *behaviour*,
*optimise*, *minimise*, *maximise*, *specialise*, *normalise*. This applies to
prose; it does not override identifiers fixed by the source code.

## Keep the book and the codebase in sync

When prose describes a signature, field, default, or behaviour, it must match
the source. If editing the book reveals that a doc-comment or code comment is
stale or wrong, fix the source in the same change — do not let the two drift.
Conversely, a code change that invalidates a user-book claim should update the
book.

## Verify rendering with a build

Never assert how a page renders — build it and check the output:

```bash
cd docs/user-book && mdbook build
```

Then grep the generated `book/` HTML to confirm anchors resolve, intra-book
links point at real targets, and `\\( … \\)` math delimiters survived markdown
(table parsing in particular). KaTeX renders client-side, so the build HTML
carries the raw `\\( … \\)` source, not pre-rendered spans — verify the
delimiters are intact rather than looking for rendered output.

## Related sources of truth

- [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md) — hard project constraints.
- [Documentation standards](ch11-documentation-standards.md) — rustdoc (`///`) policy.
