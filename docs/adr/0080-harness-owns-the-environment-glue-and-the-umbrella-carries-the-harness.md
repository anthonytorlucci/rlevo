---
project: rlevo
status: active
type: decision
date: 2026-08-22
tags: [adr, decision, architecture, crates, rlevo, rlevo-benchmarks, rlevo-environments, breaking-change, byoe]
---

# ADR 0080: The harness owns the environment glue, and the umbrella carries the harness

## Status

**Accepted (2026-08-22).**

**Supersedes two clauses of ADR
[0001](0001-keep-environments-and-benchmarks-separate.md)**:

1. "ship a feature-gated `bench` module inside `rlevo-environments`" — the glue
   moves to `rlevo-benchmarks::fixtures`, and `rlevo-environments`'s `bench` and
   `record` features are removed along with its dependency on the harness.
2. "The umbrella `rlevo` crate already excludes `rlevo-benchmarks`; this is
   preserved" — `rlevo` now takes it as a normal dependency, re-exported as
   `rlevo::benchmarks`.

**The decision ADR 0001 exists for is unaffected**: the two crates stay
separate, with disjoint dependency cones and a feature gate between them. Only
the *direction* of the gated edge and the umbrella's relationship to the harness
change. Every "Consequences" bullet in ADR 0001 that motivated separation
survives; see Consequences below for the point-by-point.

## Context

Two BYOE-1 blockers and one standing `docs/rules.md` violation turned out to be
the same fact seen from three sides.

**The edge pointed the wrong way.** `rlevo-environments` carried an optional
dependency on `rlevo-benchmarks` behind a `bench` feature, so that
`src/bench/{suites,family}.rs` could define preset `Suite` factories and
`RecordedEnvFamily` impls. Measured before the move: 713 lines behind that edge,
naming exactly three harness items — `EvaluatorConfig`, `Suite`, and
`{EnvFamily, RecordedEnvFamily}` — and **zero** `pub(crate)` accesses into
`rlevo-environments`. Every `crate::` path in those files was a public module
path. The glue was, in other words, already writable from the other side; only
its street address said otherwise.

The direction is not cosmetic. `rlevo-environments` is a library of
environments. Nothing about `Rastrigin` or `CartPole` depends on how they are
evaluated, and a consumer who wants a `CartPole` should not have a feature flag
available to them whose only effect is to acquire the harness. The inverse is
not symmetric: a harness that ships preset suites over named environments
genuinely does depend on those environments, and says so.

**B2 — the harness was unreachable from the umbrella.** `rlevo-benchmarks` was a
`[dev-dependencies]` entry in `crates/rlevo/Cargo.toml`. Inside the workspace
that resolves, so nothing in CI noticed. Outside it, a researcher who ran the
`cargo add rlevo` the README tells them to run got no harness. To evaluate
anything they had to discover a second crate whose own package description reads
"internal crate — use `rlevo` for the full API", and pin its version by hand.
The BYOE-1 probe's manifest carried that second entry as a recorded finding.

**B10 — the umbrella's viz features were inert.** `viz-tui` and `viz-report`
forwarded into `rlevo-benchmarks/tui` and `/report` — i.e. into a
dev-dependency. Measured: `cargo tree -p rlevo -e normal --features viz-tui`
reported **zero** ratatui nodes. The features were advertised in the manifest,
in the crate docs, and in `rlevo-examples`'s README, and did nothing for anyone
outside this repository. `docs/rules.md` already recorded this in prose
("Advertising a feature that does nothing is worse than not offering one")
without a decision attached.

B2 and B10 are one question — *does the umbrella carry the harness?* — and the
edge flip forces it, because after the flip the preset suites live in
`rlevo-benchmarks`. Leaving the umbrella's dependency as a dev-dep would have
made `cartpole_suite` unreachable from `rlevo` for the first time.

## Decision

**Three changes, one decision.**

1. **Move the glue to the harness.** `rlevo-environments/src/bench/` becomes
   `rlevo-benchmarks/src/fixtures/`, behind a new `fixtures` feature (off by
   default). `rlevo-environments` loses its `bench` and `record` features and
   its dependency on `rlevo-benchmarks` entirely, in every configuration. The
   orphan rule permits this: `rlevo-benchmarks` owns `Suite`, `EvaluatorConfig`,
   and `RecordedEnvFamily`, so it may implement them for foreign types.

2. **The umbrella takes the harness as a normal dependency**, re-exported as
   `rlevo::benchmarks`, with `rlevo-benchmarks/fixtures` enabled unconditionally.
   `viz-tui` and `viz-report` keep their names and now reach an external
   consumer.

3. **The physics feature pairs travel together.** A crate cannot `cfg` on
   another crate's feature, so the box2d and locomotion `RecordedEnvFamily`
   impls are gated on `rlevo-benchmarks`'s own `fixtures-box2d` /
   `fixtures-locomotion` passthroughs. `rlevo/box2d` and `rlevo/locomotion` each
   enable both halves.

`rlevo-environments` no longer has any feature that could re-hide the
`Landscape` impls or re-acquire the harness. Most of that is enforced by cargo
itself and not by us: because `rlevo-benchmarks` now names `rlevo-environments`
as a normal (if optional) dependency, re-adding the reverse edge — optional or
not — is a **resolution** failure, `"cyclic package dependency"`, before
anything compiles. `crates/rlevo-environments/tests/no_harness_dependency.rs`
covers the two gaps cargo leaves: a `[dev-dependencies]` entry, which forms no
cycle and is the arrangement ADR 0001 rejected by name as fragile, and the fact
that cargo's protection is a *side effect* of `fixtures` existing rather than a
stated rule.

## Consequences

**Positive:**

- **`cargo add rlevo` is sufficient to run a suite.** The BYOE-1 probe's
  hand-added dependency count drops from four to three, and the three that
  remain (`parking_lot`, `rand`, `serde`) are a different problem — API types
  leaking third-party crates — not a discoverability one.
- **The viz features do what they say.** `cargo tree -p rlevo -e normal
  --features viz-tui` goes from 0 ratatui nodes to 3; the default build stays at
  0.
- **Measured cost to every `rlevo` consumer: two first-party crates, zero
  third-party.** `cargo tree -p rlevo -e normal` on default features goes 254 →
  256 unique crates: `rlevo-benchmarks` itself and `rlevo-metrics-registry`, a
  `#![no_std]` zero-dependency leaf. rayon, tracing, thiserror, serde,
  serde_json, and bincode were all already in the cone via `burn` and
  `rlevo-evolution`. The genuinely new weight — ratatui, crossterm, base64,
  toml, time — stays behind `tui` / `record` / `report`.
- **ADR 0001's separation properties all survive.** Disjoint dep cones: a
  `rlevo-environments` consumer now pays *strictly less* than before, since the
  harness is not reachable from it at all; a `rlevo-benchmarks` consumer pays
  the physics cone only under `fixtures-*`, exactly as they paid it under
  `bench` before, in the other crate. No mutual cycle: the edge is now single
  and one-directional, which is a stronger version of the property ADR 0001
  wanted than the optional-reverse-edge it settled for. Truthful error
  semantics and audience-aligned abstractions are untouched — no trait changed.

**Negative / accepted costs:**

- **Two feature-name pairs must be kept in lockstep by hand.** Enabling
  `rlevo-environments/box2d` without `rlevo-benchmarks/fixtures-box2d` compiles
  a `BipedalWalker` with no `RecordedEnvFamily` impl, so
  `RecordingConfig::for_env::<BipedalWalker>` stops resolving. The `rlevo` and
  `rlevo-examples` manifests bind each pair; a consumer wiring
  `rlevo-benchmarks` directly must enable both halves, and the module docs say
  so. This is the price of the direction flip — the old arrangement had the
  gates inside one crate, where cargo's feature unification kept them
  consistent for free.
- **Breaking change.** `rlevo_environments::bench::suites::*` and the `bench` /
  `record` features are gone. The replacement path is
  `rlevo_benchmarks::fixtures::suites::*` (or `rlevo::benchmarks::fixtures::suites::*`),
  and `rlevo/viz-report` no longer needs a companion `rlevo-environments/record`.
- **`rlevo` gains a dependency it can never drop.** `pub use rlevo_benchmarks as
  bench` is a public API commitment. Making it optional was rejected — see
  Alternatives.
- **The publish boundary needed a fix to stay testable.** `cargo package`
  resolves internal `path + version` dependencies against the crates.io index
  whenever a satisfying version is already published, so packaging `rlevo` at an
  already-published version number aborted with "`rlevo-benchmarks` does not
  have that feature", naming the *published* crate's feature list. This is the
  blind spot the maintainer's publishing guide has documented across three
  release cycles; it had never been fatal before because no release had renamed
  a feature that a sibling forwards to. `xtask/byoe/run.sh` now threads a
  `[patch.crates-io]` entry per staged crate through the packaging loop, which
  is in dependency order for exactly this reason.

**Neutral:**

- The `Landscape` impls stay in `rlevo-environments::landscapes::fitness`, where
  the preceding ungating commit put them. They name nothing from the harness and
  are unaffected by this move in either direction.
- `EnvFamily::Landscapes` still has no producer on the environments side, for
  the reason `fixtures/family.rs` gives: no landscape implements `Environment`.

## Alternatives considered

**Leave the glue in `rlevo-environments`, fix only B2/B10.** Rejected: it
resolves the two blockers while leaving the `docs/rules.md` violation standing,
and it makes the violation worse in shape rather than better — `rlevo` would
then depend on `rlevo-benchmarks`, which optionally depends on
`rlevo-environments`, which optionally depends back on `rlevo-benchmarks`. Legal
for cargo, unreadable as a diagram.

**A third `rlevo-bench-glue` crate.** ADR 0001 considered and deferred this,
noting it would be revisitable "if the adapter+suites surface grows large enough
to warrant its own README and versioning story". It has not: after the
`BenchAdapter` deletion (ADR 0076) and the `Landscape` ungating, the surface is
736 lines — over half of them the `family.rs` doc comments and its in-source
test module — and three public functions. A new published crate, a new version to
keep in step, and a new README is a lot of ceremony for a module that has one
correct home now that the orphan rule permits it.

**Make `rlevo-benchmarks` an *optional* dependency of `rlevo`, enabled by
`viz-tui` / `viz-report`.** Rejected. It fixes B10 and leaves B2 exactly as it
was for the researcher who wants to evaluate an environment without recording
one — which is the common case, and the case BYOE-1 walks through. It also
re-creates the failure mode being removed: `rlevo::benchmarks` would exist or not
depending on a feature named after a *visualisation* product, and the compile
error for the missing path names neither.

**Delete `viz-tui` / `viz-report` from the umbrella and document the direct
route.** This was the honest minimal answer to B10 alone, and would have been
correct if the umbrella were staying out of the harness's way. Once the edge
flips it is strictly worse: the features would be deleted from a crate that now
*does* carry the harness, so the documented "direct route" would be a second
dependency on a crate the umbrella already links.

**Naming the re-export `rlevo::bench`.** Rejected on a measurement: `bench` is
the built-in `#[bench]` attribute macro, so the path is ambiguous to rustdoc.
`cargo doc -p rlevo` reports two `ambiguous link` errors under that name and
zero under `benchmarks`, `harness`, or `evaluation`; every intra-doc link would
need a `mod@` disambiguator, in this crate **and in every downstream crate that
links to it**. `harness` was the runner-up and was rejected because the word is
already taken in this workspace for something else — `evo::coevolution::harness`,
`EvolutionaryHarness`, `CoEvolutionaryHarness`, which per ADR 0075's implementor
census step *generations*, not transitions. `evaluation` collides with
`rlevo-core`'s `evaluation` module one level down. `benchmarks` maps 1:1 to the
crate name, so there is no mapping to learn, and it introduces no ambiguity that
the crate name did not already carry.

**Make `rlevo-environments` a non-optional dependency of `rlevo-benchmarks`,
avoiding the `fixtures-*` passthroughs entirely.** Rejected: it would force
every harness consumer to compile rapier2d and rapier3d to evaluate their own
environment, which is precisely the disjoint-dep-cone property ADR 0001 exists
to protect and precisely what BYOE-1 is testing for.

## References

- ADR [0001](0001-keep-environments-and-benchmarks-separate.md) — the two
  superseded clauses; its separation decision stands.
- ADR [0007](0007-visualisation-crates-isolated-from-production-crates.md),
  ADR [0013](0013-metrics-only-live-tui.md) — the two-product viz model
  `viz-tui` / `viz-report` forward to.
- ADR [0076](0076-trial-seam-splits-machinery-from-trial-shape.md),
  ADR [0077](0077-remove-benchenv.md) — removed the adapter that made the glue
  large enough to argue about.
- `crates/rlevo-benchmarks/src/fixtures/` — the moved glue.
- `crates/rlevo-environments/tests/no_harness_dependency.rs` — pins the arrow.
- `crates/rlevo/tests/umbrella_reaches_harness.rs` — pins B2 and B10.
- `xtask/byoe/run.sh`, `xtask/byoe/probe/Cargo.toml.template` — the acceptance
  test that measures B2, and the packaging fix that keeps it runnable.
