---
project: rlevo
status: active
type: decision
date: 2026-10-10
tags: [adr, decision, architecture, public-surface, dead-code, breaking-change, rlevo-reinforcement-learning, replay]
---

# ADR 0082: `ExperienceTuple` and `History` are deleted

## Status

**Accepted (2026-10-10).**

Supersedes one bullet of ADR
[0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md), Decision
5, which left these two types in place "for the agent seam". Completes, for
the `experience` module, the supersession of ADR
[0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md)'s
conservative dead-code policy that ADR 0081 began. Section 9 of ADR
[0050](0050-replay-strategy-seam.md), "`experience.rs` is out of scope", no
longer has a subject.

## Context

`rlevo_reinforcement_learning::experience` held two types. `ExperienceTuple<D,
AD, O, A, R>` is one step of experience: `observation`, `action`, `reward`,
`next_observation`, `terminated`. `History<D, AD, O, A, R>` is a
fixed-capacity FIFO of them.

ADR 0003 moved the module out of `rlevo-core` and kept it as a roadmap
marker. ADR 0050 built the replay seam beside it and left it alone because
two history traits were coupled to `History`. ADR 0081 deleted those traits
and left the two types, on the reasoning that the agent seam "has to settle
what one stored step of experience is".

That reasoning does not hold up. What the agent seam has to settle is the
shape of `replay::Transition<O, P>`, the type every off-policy agent stores
today. `ExperienceTuple` is not a candidate for that decision. It is a
second definition of the same five fields, differing in that it stores the
typed action and a generic reward, which ADR 0050 considered and rejected
for the stored form: storing `A` imposes `Clone + 'static` on every action
type so that a buffer can hold it, and the agents erase the action to
`usize` or `Vec<f32>` on the way in regardless. Whatever the seam decides
about `Transition`, it will not be to adopt the type ADR 0050 already turned
down.

What was measured, at the merge of ADR 0081:

- Non-blanket implementations and constructions in library code: none. No
  agent, replay strategy, example or bench names either type.
- With the whole module switched off (`#[cfg(any())]` on `pub mod
  experience`), `cargo check --workspace --all-targets --all-features`
  reports no error in any crate.
- `cargo doc` then reports one broken link, in `replay/uniform.rs`.
- Two things a compile cannot see: `docs/rules.md` carries three rows for
  `History` (one invariant, two panic contracts), and the guard test
  `crates/rlevo/tests/panic_contract_table_guard.rs` used `experience.rs` as
  its live example of an `impl` header that wraps onto a second line.

## Decision

1. **The module `rlevo_reinforcement_learning::experience` is deleted**, with
   `ExperienceTuple` and `History`.
2. **`replay::Transition` is the workspace's one stored-transition type
   for replay.** A change that needs a different shape changes that type or
   says, under ADR 0081, what its new type replaces.
3. **The three `docs/rules.md` rows for `History` are removed** with the
   type.
4. **The guard test's wrapped-header example moves to `EpisodicTrial`** in
   `rlevo-benchmarks/src/evaluator.rs`, both of whose `impl` headers wrap.
   The test pins the same property as before: a type whose name appears only
   on the `for T` continuation line of an `impl` header is still recognised.

## Consequences

**Positive.**

- One concept, one definition. A reader looking for how experience is stored
  finds `replay` and nothing beside it.
- The RL crate's README loses a section that existed to say "not used by any
  algorithm".

Surface delta, from `tools/public-surface.sh rlevo-reinforcement-learning`
before and after, all features on: 1,312 lines to 1,286. Removed: one
module, two structs, five public fields and ten methods. Nothing is added,
and no other crate's listing changes.

**Negative.**

- A breaking change for anyone who named either type. No code in the
  workspace did.
- A typed-action, generic-reward trajectory buffer is no longer available
  off the shelf. If one is needed it returns with the code that needs it,
  as ADR 0081 Decision 3 has it.

**How it was checked.** The module was deleted and the workspace compiled
with all targets and all features. The repointed guard test was then checked
against a mutant: with the `for T` alternative removed from the guard's
`mentions_impl_of`, the test fails; with it restored, the test passes. So
the test still pins what it pinned.

## Alternatives considered

**Leave them for the agent seam**, as ADR 0081 did. Rejected for the reason
in Context: the seam's question is about `Transition`, and these types do
not bear on the answer. Leaving them also keeps a module whose documentation
tells readers not to build against it.

**Make them private.** Nothing inside the crate uses them either, so a
private copy would be dead code with a `dead_code` warning.

**Fold `History` into `UniformReplay`.** They are both bounded FIFOs, but
`UniformReplay` already is the bounded FIFO the agents use, with stable ids
and sampling. There is nothing in `History` to fold in.

**Keep `experience.rs` because a guard test reads it.** The test needs a
wrapped `impl` header, not this one. Two other types in the workspace have
headers that all wrap, `SnapshotBase` and `EpisodicTrial`; either would do.

## References

- ADR [0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md),
  which moved the module and kept it.
- ADR [0050](0050-replay-strategy-seam.md), the replay seam and its
  rejection of a typed-action stored form.
- ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md), the
  test these types fail and the bullet this supersedes.
