---
project: rlevo
status: active
type: decision
date: 2026-08-20
tags: [adr, decision, architecture, rlevo-core, rlevo-benchmarks, rlevo-environments, rlevo-evolution, bench-env, object-safety]
---

# ADR 0075: `BenchEnv` erases rank, not modality — the heterogeneous-suite rationale is unsound

## Status

**Accepted (2026-08-20).**

**Supersedes the object-safety rationale** in ADR
[0004](0004-move-bench-traits-into-rlevo-core.md) and ADR
[0001](0001-keep-environments-and-benchmarks-separate.md) — specifically ADR
0001's "object-safe and free of const generics so trial-level rayon parallelism
stays [possible]", its proposed `Box<dyn BenchEnv<…>>` wrapper "with a
normalized obs/action shape", and its "threading const generics through
(defeating its object-safety design)"; and ADR 0004's "Conceptual fit" claim
that "`BenchEnv` is a narrower `Environment`" so far as that narrowing is
justified by dynamic dispatch.

Citations here are by quoted phrase, not line number, deliberately: annotating
an immutable record shifts every line below the annotation, and both of these
ADRs' line numbers moved when this supersession was appended. ADR 0004's
existing 2026-07-06 annotation cites "line 95" for a consequence now at line 99
— left as found, since the record is immutable and the quoted text still
locates it.

**The relocation decisions in ADR 0004 are untouched.** Moving the trait surface
into `rlevo-core`, the typed `BenchError`, the shim modules, the dep-edge
rework — all remain active. Only the *reason given for `BenchEnv`'s shape* is
superseded. This ADR does not decide `BenchEnv`'s fate; it removes a false
premise so that decision can be made on accurate grounds.

## Context

`BenchEnv`'s own doc comment justified the trait as an object-safety erasure
layer: it "strips the const-generic dimensionality of `Environment` so
benchmarking harnesses and evolutionary outer loops can work with a plain trait
object (`dyn BenchEnv`) rather than threading dimension parameters through their
own type signatures." `rlevo-benchmarks/README.md` stated the payoff directly:
"so that heterogeneous environments can be boxed and dispatched at runtime."
`rlevo-environments/src/bench/suites.rs:6` deferred the feature that payoff
describes — an "all of classic control" suite — pending "a `Box<dyn
BenchEnv<…>>` design."

Three findings, each executed rather than reasoned about:

1. **The trait is object-safe.** A probe test constructing
   `Box<dyn BenchEnv<Observation = u32, Action = ()>>` and a `Vec` of them
   compiles and runs. Any claim that associated types make `dyn BenchEnv`
   unusable is **false** — they must be named, but naming them is legal. (The
   probe was temporary and is not retained; see *Alternatives considered*.)

2. **The erasure is on the wrong axis.** `BenchEnv` erases `D`/`SD`/`AD` — the
   const-generic *ranks* — while **preserving** `Observation` and `Action` as
   associated types. But rank is not what differs between environments;
   modality is. Every classic-control env declares a distinct pair:

   | Env | `ObservationType` | `ActionType` |
   |---|---|---|
   | `CartPole` | `CartPoleObservation` | `CartPoleAction` |
   | `Pendulum` | `PendulumObservation` | `PendulumAction` |
   | `MountainCar` | `MountainCarObservation` | `MountainCarAction` |
   | `Acrobot` | `AcrobotObservation` | `AcrobotAction` |
   | `TenArmedBandit` | `KArmedBanditObservation` | `KArmedBanditAction<K>` |

   A single `Box<dyn BenchEnv<Observation = X, Action = Y>>` can therefore hold
   only envs sharing one obs/action pair — homogeneous in exactly the dimension
   the deferred feature needs heterogeneous. **The "all of classic control"
   suite cannot be built on `BenchEnv` as designed.** ADR 0001's boxed-wrapper
   plan is unsound as written; its "with a normalized obs/action shape" is
   where all the unspecified work actually lives, and that normalization, not
   object-safety, is the real prerequisite.

3. **The erasure is unexercised.** Zero `dyn BenchEnv` values are constructed
   anywhere in the workspace. `Suite<E>` is monomorphic; `run_suite<E: BenchEnv
   + Send>` is a static generic bound. Monomorphization already does the only
   erasure that happens.

Separately, the implementor census contradicts ADR 0004's "narrower
`Environment`" framing. `rg "BenchEnv for"` returns seven impls:

- `BenchAdapter` — the only `Environment`-shaped one;
- `EvolutionaryHarness`, `CoEvolutionaryHarness` — **production, and genuinely
  not `Environment`s**: they step generations, not transitions;
- four test/example stubs.

`BenchEnv` is not a narrowing of `Environment`. It is a union bound over two
disjoint families, one of which has no environment semantics at all.

## Decision

**Record that `BenchEnv`'s object-safety rationale does not hold, and correct
every doc comment that asserts it. Change no code.**

Concretely:

1. **`crates/rlevo-core/src/evaluation.rs`** — module and trait docs restated:
   the trait is object-safe but its erasure is rank-only, no `dyn` site exists,
   and its actual job is spanning two disjoint implementor families.
2. **`crates/rlevo-core/src/lib.rs`** — doc-table row and `evaluation` module
   doc corrected to match.
3. **`crates/rlevo-benchmarks/README.md`** — the "heterogeneous environments can
   be boxed and dispatched at runtime" claim is removed. It is the sharpest
   false statement in the tree, because it names a capability the design cannot
   deliver.
4. **`crates/rlevo-core/README.md`** — trait-table rows restated.
5. **`crates/rlevo-environments/src/bench/adapter.rs`** — "Object-safe wrapper"
   corrected; the adapter's job is rank erasure, which is what its three phantom
   const params are for.
6. **`crates/rlevo-environments/src/bench/suites.rs`** — the deferred-feature
   comment records that obs/action normalization, not a `Box<dyn BenchEnv<…>>`
   design, is the blocking prerequisite.

**No trait, type, signature, or test changes.** This ADR deliberately stops at
the documentation boundary.

## Consequences

**Positive:**

- **A false capability claim is removed from a published README.** A reader
  planning a heterogeneous suite on the strength of
  `rlevo-benchmarks/README.md:38` would have built against a design that cannot
  support it.
- **The deferred feature gets an accurate blocker.** "Needs a `Box<dyn
  BenchEnv<…>>` design" pointed at the wrong axis; "needs an obs/action
  normalization" is the actual work, and it is a much larger and more
  interesting question (what *is* the normalized shape?).
- **`BenchEnv`'s fate can now be decided on accurate grounds.** Every option —
  keep, rename, relocate, delete, split by caller — was previously being argued
  against a rationale that does not survive contact with a compiler.
- **The two-disjoint-families finding is recorded.** It is the single most
  decision-relevant fact about this trait and it appeared in no prior document.

**Negative / accepted costs:**

- **ADR 0004 now has two superseding annotations** (0033 for the splitmix64
  mixer, this one for the object-safety rationale) while its core relocation
  decision stays active. Readers must partition it. Unavoidable under the
  immutability rule, and preferable to editing an accepted record.
- **A known-false rationale is corrected without deciding the follow-on.**
  `BenchEnv` keeps a name derived from its sole consumer crate, and keeps a
  shape justified by nothing in particular. That is deliberate: conflating the
  correction with the redesign is how the original error got written down as
  settled in the first place.

**Neutral:**

- No behaviour, API, or dependency change. `cargo build`/`test` output is
  identical before and after.
- The object-safety *property* is real and retained — nothing here forecloses a
  future `dyn BenchEnv`. What is retracted is the claim that the property buys
  the heterogeneous suite.

## Alternatives considered

**Edit ADR 0004 and 0001 in place.** Rejected — `docs/adr/README.md` and
`CLAUDE.md` both make ADRs immutable once accepted. Superseding is the
prescribed mechanism, and the annotation trail is itself informative: it shows
the rationale drifted rather than being wrong on day one.

**Retain the object-safety probe as a permanent test** (a `trybuild` or plain
integration test asserting `Box<dyn BenchEnv<…>>` compiles). Rejected. It would
pin a property nothing depends on, and the property was never the disputed part
— the *payoff* was. A test asserting the trait is object-safe would, if
anything, re-entrench the idea that object-safety is the point.

**Fold the rename into this ADR.** Deferred, but the target is decided:
`BenchEnv`/`BenchStep`/`BenchError` become
**`Steppable`/`StepOutcome`/`StepError`** whenever the rename lands. All three
drop `Env`, which is the census finding — the evolutionary harnesses are not
environments — and `Steppable` names the one operation both families share
without promising episodes, environments, or dynamic dispatch.

It is deferred rather than done because its cost depends on a still-open
question. If the two callers are later split (`GenerationProbe` for evolution,
`Environment` for the env path), the env-side trait is deleted outright and a
rename today is churn across ~43 files on a doomed symbol. If the traits are
instead relocated out of `rlevo-core`, the rename rides along with the move at
no extra import churn. Only if nothing further happens does the rename stand
alone — and even then it is now the *lesser* complaint, since this ADR replaces
"the name is wrong" with "the shape is unjustified".

Doing both at once would also make this record's factual findings hostage to a
naming debate. The rename is a separate decision with this ADR as its input.

**Delete `BenchEnv` now, per the private spec.** Out of scope, and this ADR is
an input to that decision rather than a substitute for it. The spec's own
premise — that the erasure is never exercised — is confirmed here, but its
inference (that the trait is therefore redundant) is refuted by the census: the
two evolutionary harnesses are not `Environment`s, and re-pointing them at
`Environment` would require fabricating null state, observation, and action
types.

## References

- ADR [0001](0001-keep-environments-and-benchmarks-separate.md) — its
  *Consequences* and *Alternatives considered* sections carry the superseded
  object-safety framing.
- ADR [0004](0004-move-bench-traits-into-rlevo-core.md) — relocation decisions
  remain active; its "Conceptual fit" claim is narrowed here.
- ADR [0033](0033-share-splitmix64-mixer-across-core-and-evolution.md) — the
  other partial supersession of ADR 0004.
- `crates/rlevo-core/src/evaluation.rs` — the trait and its corrected docs.
- `crates/rlevo-environments/src/bench/suites.rs` — the deferred heterogeneous
  suite and its corrected blocker.
- `crates/rlevo-evolution/src/strategy.rs`,
  `crates/rlevo-evolution/src/coevolution/harness.rs` — the two non-`Environment`
  production implementors.
