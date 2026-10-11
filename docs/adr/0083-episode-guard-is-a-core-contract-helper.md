---
project: rlevo
status: active
type: decision
date: 2026-10-11
tags: [adr, decision, architecture, core, environments, episode-lifecycle, crate-boundaries, breaking-change]
---

# ADR 0083: `EpisodeGuard` is a core contract helper

## Status

**Accepted (2026-10-11).**

Supersedes the placement in Decision 4 of ADR
[0044](0044-post-terminal-step-is-an-error.md), "One crate-level
`EpisodeGuard`, in `rlevo-environments`", and its sentence "It cannot live
in `rlevo-core`". The type, its private field, its five methods and every
other decision of ADR 0044 stand.

## Context

ADR 0044 put a contract on every implementor of `Environment`: a `step`
taken after a done snapshot returns
`EnvironmentError::StepAfterEpisodeEnd`, carrying the status that ended the
episode, and changes nothing. The contract is written in `rlevo-core`, on
`Environment::step`. The state machine that implements it,
`EpisodeGuard`, was placed in `rlevo-environments`.

ADR 0044 gave one reason for keeping it out of core: core "is a contract
crate and `rules.md`'s Workspace Structure section forbids implementation
logic there (sole exception: `util`)". That rule was withdrawn from
`rules.md` on 2026-08-11, because it was false of core at six sites
(`Bounds`, `Probability`, `NonNegativeRate` and the config constraint
helpers among them). What the section says now is that core holds no
algorithm or environment implementation and no second grab-bag beside
`util`, and that "validation and invariant-enforcement logic owned by a
type defined here" is part of the contract, not an exception to it. A type
whose one job is to enforce an invariant of `Environment::step` is in that
category. The reason no longer exists; the placement outlived it.

The placement has a cost. `rlevo-environments` is the crate of shipped
environments: 35 source files there implement `Environment`, and it brings
two physics engines by default. An author writing an environment of their
own needs the contract, not the fixtures, and can depend on `rlevo-core`
alone. That author owes the post-terminal contract like everyone else and
could not reach the guard without the fixture crate. A `done: bool` is not
enough to hand-roll it, because the error carries the `EpisodeStatus` that
ended the episode, so each such author writes the same one-field state
machine again.

What was measured, at the merge of ADR 0082:

- The guard's code imports one thing,
  `rlevo_core::environment::{EnvironmentError, EpisodeStatus}`. Moving it
  adds no dependency to core.
- Every one of the 35 files under `crates/rlevo-environments/src` that
  implements `Environment` holds a guard, the `TimeLimit` wrapper among
  them (`rg -l '^impl.*\bEnvironment<'`, each checked for `EpisodeGuard`).
- No file outside `rlevo-environments` names the guard
  (`git grep -l EpisodeGuard -- crates xtask`).
- Core's documentation of the contract could not link to its
  implementation, which lived in a crate that depends on core.

## Decision

1. **`EpisodeGuard` is `rlevo_core::episode::EpisodeGuard`.** The struct,
   its derives, its private `status` field and its five `const fn` methods
   (`new`, `reset`, `check`, `record`, `status`) move unchanged, with the
   guard's six unit tests and its doctest.
2. **`rlevo-environments` does not re-export it.** The module
   `rlevo_environments::episode` is no longer public. A re-export would
   keep a second public path that no file outside the crate ever wrote,
   and ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md)
   has a public item be public because a user writes its name. The 35
   files import the guard from core, so a shipped environment shows the
   import an outside author writes.
3. **The conformance check stays where it is.**
   `assert_rejects_post_terminal_step` and the `MockGuardedEnv` that
   smoke-tests it remain in `rlevo-environments`, compiled for tests only
   and private to the crate, as ADR 0044 Decision 10 left them.
4. **`TimeLimit` stays in `rlevo-environments`.** It implements
   `Environment`, and core holds no environment implementation.
5. **`docs/rules.md` section 1 names the guard** in core's inventory and
   among the invariant-enforcing types its contract-crate paragraph lists.
6. **The contract's documentation points at the guard.**
   `Environment::step`'s post-terminal section links to it.

## Consequences

**Positive.**

- An environment written against `rlevo-core` alone holds the same guard
  as the shipped ones. The guard's doctest names only `rlevo_core` paths,
  which is the check that it is reachable from core by itself.
- One definition and one path. A reader of the contract follows a link to
  the type that satisfies it.

Surface delta, from `tools/public-surface.sh` before and after, all
features on. `rlevo-core`: 433 lines to 443, one module, one struct and
five methods added (ten lines, because the listing carries three of the
methods a second time by path). `rlevo-environments`: 1,787 lines to 1,777,
the same ten removed. The addition replaces the removal; no other crate's
listing changes. Core's dependency cone is unchanged at 107 packages
(`cargo tree -e normal -p rlevo-core --prefix none`, unique names, less
the crate itself).

**Negative.**

- A breaking change for anyone who imported
  `rlevo_environments::episode::EpisodeGuard`, or
  `rlevo::envs::episode::EpisodeGuard` through the umbrella. The new paths
  are `rlevo_core::episode::EpisodeGuard` and
  `rlevo::core::episode::EpisodeGuard`. No code in the workspace outside
  `rlevo-environments` did.
- Core gains a module and a concrete type.
- The conformance check is still out of an outside author's reach. See
  Alternatives.

**How it was checked.** The workspace compiles with all targets and all
features, and `rlevo-core` compiles alone (`cargo check -p rlevo-core
--lib`). The fast test tier passes with its count unchanged: the six unit
tests and the doctest left `rlevo-environments` and run in `rlevo-core`.

## Alternatives considered

**Leave the guard in `rlevo-environments`.** Rejected: the one reason
recorded for it cites a rule that has been withdrawn, and the cost falls
on exactly the author who depends on the contract crate alone.

**Move it and keep a public re-export in `rlevo-environments`.** Rejected
under Decision 2. It spares a break that no code in the workspace would
feel, at the price of two public names for one type.

**Move it and keep the 35 imports unchanged with a `pub(crate) use`.**
Rejected: the shipped environments are what an author copies from, and
they would show an import that does not exist outside the crate.

**Put it in `rlevo_core::environment`**, beside `EpisodeStatus` and
`EnvironmentError`. A fair alternative: one import line fewer for an
author. Not taken because that module is what an author implements, and
the guard is something an author holds, with usage notes of its own; a
separate module keeps the trait contract readable. Either placement
satisfies this ADR's reason.

**Move the conformance check to core as well.** It would be a public test
helper in a published crate, generic over `Environment`, with no consumer
outside `rlevo-environments` today. Under ADR 0081 it returns with the
code that needs it. Until then an outside author writes the assertion by
hand: drive to a done snapshot, step once more, match
`StepAfterEpisodeEnd` and its status.

**Move `TimeLimit` to core.** Rejected under Decision 4.

## References

- ADR [0044](0044-post-terminal-step-is-an-error.md), the contract, the
  guard's design, and the placement this supersedes.
- ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md),
  the test a public path must pass.
- `docs/rules.md` section 1, the contract-crate paragraph.
- Code: `crates/rlevo-core/src/episode.rs` (`EpisodeGuard`);
  `crates/rlevo-core/src/environment.rs` (the post-terminal contract);
  `crates/rlevo-environments/src/episode.rs`
  (`assert_rejects_post_terminal_step`, test-only).
