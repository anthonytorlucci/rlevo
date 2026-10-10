---
project: rlevo
status: active
type: decision
date: 2026-10-10
tags: [adr, decision, architecture, public-surface, traits, dead-code, breaking-change, rlevo-core, rlevo-reinforcement-learning, rlevo-evolution, rlevo-benchmarks]
---

# ADR 0081: A public item needs an implementor and a consumer

## Status

**Accepted (2026-10-10).**

Supersedes one clause of ADR
[0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md), its
"Scope decision: conservative dead-code policy", for the items listed under
Decision. Amends the "stay as-is" notes on `Renderer`, `NullRenderer` and
`AsciiRenderer` in ADRs
[0006](0006-leptos-first-visualisation-defer-bevy.md),
[0008](0008-three-tier-visualisation-ratatui-live-static-report.md),
[0009](0009-move-styled-render-into-rlevo-core.md) and
[0013](0013-metrics-only-live-tui.md), and section 9 of ADR
[0050](0050-replay-strategy-seam.md). The Amendments section says what
changes in each.

## Context

ADR 0003 counted a set of traits in `rlevo-core` as having no consumer and
kept them on purpose: "the project owner opted to keep them visible as
roadmap markers", with the cost named as "continued cargo-doc surface
implying APIs that are not yet wired up" and the decision marked
"revisitable if/when the cargo-doc noise becomes a documentation burden". It
rejected the counter-proposal to delete them because "roadmap visibility
wins over cargo-doc minimalism for the alpha stage".

Two things have changed since.

**The surface is about to be offered to outside authors.** The next release
invites people to bring their own environment and their own algorithm. A
`pub trait` in the contract crate reads to such an author as something to
implement or to bound on. A trait that nothing implements and nothing bounds
on is a promise the library does not keep.

**Several of the markers do not fit the code they were sketched for.** They
are not unfinished seams waiting for a user; their signatures rule the
intended user out:

- `LatentState::encode(observation: &Self::Observation) -> Self` is an
  associated function with no `self`. A learned encoder has parameters, so it
  cannot implement it, and `predict_next` and `decode` inherit the defect.
- `HiddenState::update(&mut self, observation)` takes no action, so a
  recurrent state-space model step cannot implement it, and it takes no
  parameters, so a GRU could implement it only by storing its weights inside
  the hidden state. The mechanism that works is `StatefulPolicy::Hidden` in
  `rlevo-hybrid`, where the policy holds the parameters and the hidden state
  is separate.
- `Landscape2DPayloadSource` was to be implemented by an environment so that
  a recording tap could wrap it. A landscape is a `Landscape`, not an
  `Environment`, so no tap can wrap one.

What was measured, on the public surface as rustdoc sees it (`--lib`, all
features, non-blanket implementations in library code, test modules
excluded), with every reference found by `git grep -w` over all tracked
files:

| Item | Crate | Implementors | Consumers |
|---|---|---|---|
| `UpdateFunction<Input, Output>` | `rlevo-core` | 0 | none |
| `LatentState<R, AR>` | `rlevo-core` | 0 | none |
| `HiddenState<R>` | `rlevo-core` | 0 | none |
| `MarkovState` | `rlevo-core` | 1: `SantaFeAntState`, returning `false` | a supertrait bound on `SufficientStatistic`, and one unit test asserting the `false` |
| `HistoryRepresentation` | `rlevo-reinforcement-learning` | 0 | a supertrait bound on `SufficientStatistic` |
| `SufficientStatistic` | `rlevo-reinforcement-learning` | 0 | none |
| `Landscape2DPayloadSource` | `rlevo-core` | 0 in library code; one test stub | one bound, on `RecordingTap::with_landscape_payload` in `rlevo-benchmarks`, which no type could satisfy |
| `Renderer<E>` | `rlevo-core` | 2, both in `rlevo-core` (`NullRenderer`, `AsciiRenderer`) | no bound and no call anywhere |
| `NullRenderer` | `rlevo-core` | | named in documentation only |
| `AsciiRenderer` | `rlevo-core` | | one doctest, and a re-export from `rlevo_environments::render` |

ADR 0003 recorded `Renderer` as having "one downstream consumer
(`rlevo-environments`)". That consumer is the re-export in the last row.
Nothing bounds on the trait. ADR 0003 also said that "resurrecting them later
if a second consumer appears is trivial"; none appeared.

One more published item promised something it did not do.
`rlevo_evolution::algorithms::metaheuristic::aco_perm::AntColonyPermutation`
implemented `Strategy` with `init`, `ask` and `tell` each a `todo!()`, and
three of its seven tests pinned the panic. A published `Strategy` that panics
on `ask` breaks the workspace rule that a panic marks a programming error.

## Decision

1. **A `pub trait` in a published crate stays only if one of these holds.**
   - It has at least two production implementors that differ, and at least
     one generic consumer that bounds on it.
   - It is the seam an audience is asked to implement: an environment, an
     agent, an optimisation strategy.
   - It exists for coherence (a blanket implementation has to hang on
     something) or to carry associated constants and types that a function
     cannot.

   Otherwise the concept is a type, an enum, a closure or a documented
   convention. A trait with one legal implementor is a type with extra
   steps. A trait implemented everywhere and bounded on nowhere is
   documentation with a compile cost.

2. **A public struct, enum or function is public because a user writes its
   name.** One that nothing outside its module constructs, matches on or
   calls is private.

3. **The roadmap lives in documentation, never in a `pub` item.** An item
   whose only justification is a feature not yet written is deleted, and
   returns in the change that writes the feature, with the signature that
   feature needs. This replaces ADR 0003's conservative dead-code policy.

4. **These items are deleted.**

   | Removed | What to use instead |
   |---|---|
   | `rlevo_core::base::UpdateFunction` | a closure |
   | `rlevo_core::state::LatentState` | nothing yet; a world-model seam returns with the first world model, shaped by it |
   | `rlevo_core::state::HiddenState` | `rlevo_hybrid::policy::StatefulPolicy`, whose `Hidden` associated type is the recurrent state |
   | `rlevo_core::state::MarkovState` | the environment's documentation; `SantaFeAnt`'s says its percept is not Markov |
   | `rlevo_reinforcement_learning::experience::HistoryRepresentation`, `SufficientStatistic` | nothing yet; a history summary is an observation the agent is given, so it belongs to an environment wrapper, not to the experience buffer |
   | `rlevo_core::render::Landscape2DPayloadSource`, and `rlevo_benchmarks::record::RecordingTap::with_landscape_payload`, the constructor bounded on it | build a `Landscape2DSnapshot` directly |
   | `rlevo_core::render::Renderer`, `NullRenderer`, `AsciiRenderer`, and the re-export `rlevo_environments::render::AsciiRenderer` | `AsciiRenderable::render_ascii`, which `AsciiRenderer` only forwarded to |
   | `rlevo_evolution::algorithms::metaheuristic::aco_perm` (`AntColonyPermutation`, `AcoPermConfig`, `AcoPermState`) | nothing; it returns with an implementation |

5. **What is deliberately not deleted here.**
   - The payload types `Landscape2DSnapshot` in `rlevo-core` and
     `Landscape2DPayload` in `rlevo-benchmarks`. The record format reads
     them; only the source trait is gone.
   - `ExperienceTuple` and `History`, which sat beside the two history
     traits. No shipped agent uses them. Whether they stay is decided with
     the agent seam, which has to settle what one stored step of experience
     is.
   - `rlevo_evolution::genome::Permutation`. It does not panic, and ADR
     [0030](0030-permutation-tensorgenome-and-population-nonempty-invariant.md)
     made it a `TensorGenome` on its own merits. With the stub gone it has
     no consumer, and neither do the other genome-kind markers: every
     `Strategy::Genome` in the workspace is a bare tensor. Whether the
     markers stay is a question about all five and is not answered here.
   - The other markers ADR 0003 lists: `TransitionDynamics`, `BeliefState`,
     `StateAggregation` and `MultiDiscreteAction`. They fail item 1 and item
     3 today. Each is decided with the pruning of the environment contract,
     not here.

6. **The public surface is measured, not read.** `tools/public-surface.sh
   <crate>` lists a crate's public items from rustdoc's JSON output, one per
   line. The diff of two listings is a change's surface delta. A pull
   request that changes the public surface reports its delta from that
   command, and a change whose delta adds items says what each addition
   replaces.

## Consequences

**Positive.**

- The workspace has 67 public traits where it had 75, and `rlevo-core` has
  30 where it had 36.
- A reader of `rlevo-core`'s documentation no longer meets five traits for
  partial observability and world models that no code in the workspace can
  use.
- `rlevo-evolution` ships no strategy that panics when driven.

Surface delta, from `tools/public-surface.sh` at the parent commit and at
this change, all features on:

| Crate | Lines before | Lines after | Removed |
|---|---:|---:|---|
| `rlevo-core` | 445 | 433 | six traits, two associated types, two structs, two re-exports |
| `rlevo-evolution` | 1,226 | 1,215 | one module, three structs, five fields, two methods |
| `rlevo-reinforcement-learning` | 1,316 | 1,312 | two traits, two trait methods |
| `rlevo-benchmarks` | 632 | 631 | one constructor |
| `rlevo-environments` | 1,788 | 1,787 | one re-export |

Nothing was added.

**Negative.**

- A breaking change for anyone who named one of these items. None of them
  could be used for anything, with two exceptions: `AsciiRenderer`, whose
  one method call becomes `env.render_ascii()`, and a downstream type that
  implemented `MarkovState` or `Renderer` for its own purposes, which now
  owns that trait itself.
- `SantaFeAnt` no longer states at the type level that its percept is not
  Markov. Nothing read that statement; the module documentation still makes
  it.
- The roadmap for partial observability and world models is no longer
  visible in `cargo doc`. It is where a roadmap belongs.

**How the deletions were checked.** Each group was deleted and the workspace
compiled with `cargo check --workspace --all-targets --all-features`, so the
compiler's error list is the consumer list. Deleting the six trait shells
needed the one `MarkovState` implementation and its unit test removed and
nothing else. Deleting `Landscape2DPayloadSource` needed the tap constructor
and one line of a test stub removed. Deleting `Renderer` and its two
implementors needed the re-export removed. Deleting `aco_perm` needed
nothing outside its module.

## Amendments

- **ADR 0003, superseded clause.** "This refactor does not delete the
  speculative trait shells in `rlevo-core` [...] the project owner opted to
  keep them visible as roadmap markers." Superseded by Decision 3 for
  `UpdateFunction`, `MarkovState`, `HiddenState` and `LatentState`, and for
  `HistoryRepresentation` and `SufficientStatistic`, which its rejected
  counter-proposal named. Its neutral note that `Renderer` and
  `NullRenderer` "stay put" with "one downstream consumer" is superseded by
  Decision 4; the consumer was a re-export. The rest of ADR 0003 stands.
- **ADRs 0006, 0008 and 0013.** Each records that `Renderer<E>`,
  `NullRenderer` and `AsciiRenderer` are preserved, "stay as-is" or are
  "untouched". Those notes described scope, not a reason to keep the items;
  they are superseded by Decision 4. `AsciiRenderable` is unaffected.
- **ADR 0009.** Its move of `AsciiRenderable` and the styled-frame types
  into `rlevo-core::render`, with a re-export shim in
  `rlevo-environments::render`, stands. `AsciiRenderer` is removed from both
  places.
- **ADR 0050, section 9.** It left `experience.rs` untouched because the two
  history traits "are structurally coupled to `History`". The traits are now
  gone and the coupling with them. `ExperienceTuple` and `History` remain
  out of that ADR's scope and are covered by Decision 5.
- **ADR 0030.** Its note that the permutation ACO "remains a `todo!()` stub"
  is out of date: the stub is removed. Its decision, that `Permutation`
  implements `TensorGenome`, stands.
- **ADRs 0019, 0025 and 0047** name `HiddenState`, `LatentState` or
  `MarkovState` in passing, as a naming precedent or as a fact about
  `SantaFeAnt`. No decision in them depends on the traits existing.

## Alternatives considered

**Keep the markers and document them as unimplemented.** This is what the
workspace did. `experience.rs` carried a module-level warning that its items
had no consumer and should not be built against, and `rlevo-core`'s README
listed the state traits without saying nothing implemented them. A warning
beside a public trait is weaker than the trait's absence, and three of the
traits did not fit the code they were sketched for.

**Fix the signatures instead of deleting.** A `LatentState` with a `&self`
encoder, a `HiddenState` that sees parameters and actions. Rejected: a
signature designed without its first implementor is how these traits came to
be unimplementable. Decision 3 has the seam return with the code that needs
it.

**Hide the items with `#[doc(hidden)]` or a feature gate.** Rejected. A
hidden public item is still public for semver purposes and still has to be
kept compiling, and a feature that gates only dead items is the pattern ADR
[0080](0080-library-crates-select-no-burn-backend.md) removed from
`rlevo-evolution`.

**Delete every marker ADR 0003 lists, now.** `TransitionDynamics`,
`BeliefState`, `StateAggregation` and `MultiDiscreteAction` have the same
measurements as the items deleted here. They are left for the contract
pruning because that change decides what the environment contract is, and
each of the four has a planned first implementor whose needs should shape
what comes back.

**Keep `aco_perm` so the `Permutation` genome kind has a consumer.** The
stub's own documentation gave this as its reason to exist. A consumer that
panics is not one.

## References

- ADR [0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md),
  the policy this supersedes in part.
- ADR [0019](0019-observable-projection-trait.md) and ADR
  [0047](0047-sensor-relocates-emission-model-to-environment.md), which
  placed observation production on the environment.
- ADR [0025](0025-stateful-policy-rollout-contract.md), the recurrent-state
  mechanism that replaces `HiddenState`.
- ADR [0080](0080-library-crates-select-no-burn-backend.md), the previous
  step of the same simplification.
- `tools/public-surface.sh`, the surface listing.
