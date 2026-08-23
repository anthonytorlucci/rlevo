---
project: rlevo
status: active
type: decision
date: 2026-08-23
tags: [adr, decision, architecture, crates, record-schema, wire-format, rlevo-core, rlevo-benchmarks, rlevo-scene, wasm, breaking-change]
---

# ADR 0081: The record format is a leaf crate

## Status

**Accepted (2026-08-23). Amended the same day, during implementation** — see
"Amendments" below. The core decision is unchanged and `rlevo-scene` exists; four
specifics in it were wrong and are corrected in place rather than left standing.

**The deferred verification is discharged.** Decision 1 depended on `Bounds`'s
`#[serde(try_from, into)]` validation behaving as expected under **bincode**
(issue #942's second caveat), with "validate at construction instead" as the
fallback. Executed: validation **does** run under bincode, and the encoded form
is byte-identical to a bare `(f32, f32)`. The fallback is not needed, and any
validated range ADR 0082 adds to the wire can use the same mechanism at no wire
cost. Guard: `crates/rlevo-benchmarks/tests/bounds_bincode_validation.rs`.

### Amendments (2026-08-23)

Normally an accepted ADR is superseded rather than edited. These landed within
hours of acceptance, from executing the very first slice, and a successor ADR
correcting four clauses of an unimplemented decision would be harder to read
than the correction itself. The original claims are quoted at each site so the
error stays visible.

| # | Original claim | What measurement showed |
|---|---|---|
| 1 | dependencies are "`serde` and `bincode`. Nothing else" | `BoundsError` and `DecodeError` derive `thiserror::Error`. The measurement behind the clause grepped `use` lines, which a fully-qualified derive path never appears on. |
| 2 | the leaf owns `RunManifest` | The manifest crosses to the client as **JSON**, not bincode. It stays in `rlevo-benchmarks`. |
| 3 | the leaf owns `Bounds` | It is a config-validation primitive named in **47 files** across evolution, RL, and environments, and **nothing on today's wire uses it**. It stays in `rlevo-core`. |
| 4 | "`wire.rs` is **deleted** — 839 lines" | Nearly all of it goes; the `RunManifest` / `ObjectiveSense` JSON view survives, deliberately. |

Amendments 2 and 3 share a root cause worth naming: **this ADR drew the boundary
around a crate diagram when the real boundary is a transport.** See decision 2.

**Sequenced before ADR
[0082](0082-the-scene-is-the-payload-and-identity-is-open.md)**, which rewrites
every file this ADR moves. Extracting afterwards means writing the scene types
three times and then merging them; extracting first means 0082's migration happens
once, in the crate that should own it. The two are independently valuable — this
one deletes an existing fork whether or not 0082 is accepted — but the order
matters and is not arbitrary, and the ADR numbers follow it.

Follows the precedent of ADR [0015](0015-shared-typed-metric-registry-crate.md)
exactly, at roughly five times the size and with one dependency difference
(`alloc`, not zero-dep). It supersedes nothing.

## Context

### Three copies of the same struct

Every rich payload shape currently exists three times:

| # | Location | Role |
|---|---|---|
| 1 | `rlevo-core::render::payload::*Snapshot` | producer-facing; what an env returns |
| 2 | `rlevo-benchmarks::record::schema::*Payload` | *"Bincode-stable mirror"*, with a `From` impl |
| 3 | `rlevo-benchmarks-report-client::wire::*Payload` | WASM-side mirror |

Copy 2 is not a guess about intent — six structs in `schema.rs` carry the doc
comment *"Bincode-stable mirror of `<T>`"* verbatim: `Landscape2DSnapshot`,
`Box2dSnapshot`, `Locomotion2DSnapshot`, `GridSnapshot`, `TabularSnapshot`,
`Classic2DSnapshot`.

**The boundary between copies 1 and 2 is incoherent.** `schema.rs` *imports*
`Point2`, `RigidBody2D`, `Classic2DBody`, `GridTile`, `GridAgentMarker`,
`TabularLayout`, and `StyledFrame` from `rlevo-core::render` while duplicating the
structs that contain them. Some types cross the crate line; the ones wrapping them
do not. No dependency forces this — `rlevo-benchmarks` already depends on
`rlevo-core`. The duplication buys insulation against producer-type churn, which
is a real concern, but a dedicated wire crate serves it better than a hand-synced
copy does.

### Copy 3 is forced, and only by a dependency the types do not have

`rlevo-benchmarks-report-client` cannot depend on `rlevo-core`, and its own
manifest records why:

> `rlevo-core` pulls in burn (→ rand → getrandom) which doesn't WASM-build without
> extra feature gating.

That is true of `rlevo-core`. It is **not** true of the module being mirrored.
Measured, `rlevo-core/src/render/` imports exactly two things: `serde` and
`std::ops`. No burn, no rand, no rand_distr, no thiserror, no tracing — all five
of which `rlevo-core` carries.

So the fork exists because of the module's *street address*, not its contents.
This is the same shape ADR 0080 found in `rlevo-environments/src/bench/`: glue
that was already writable from the other side, with only its location saying
otherwise.

### The fork is 42% of the client's wire module

`wire.rs` is 839 lines and sections itself explicitly:

| Lines | Content |
|---|---|
| 20-91 | `// ---- Mirror of rlevo_core::render::styled` |
| 93-377 | `// ---- Mirror of rlevo_core::render::payload` |
| 379-839 | mirror of `rlevo-benchmarks::record::schema`, plus the codec |

355 lines mirror `rlevo-core::render`. The remaining 460 mirror the schema, which
is copy 2 of the same information. A 495-line drift-guard test
(`wire_format_compat.rs`) exists to keep the halves honest.

### This is exactly why `rlevo-metrics-registry` exists

ADR 0015 extracted a `#![no_std]` zero-dependency leaf crate because the report
client had forked the canonical metric list with no guard. The argument here is
the same and stronger on every axis: the forked surface is larger, it is
versioned, there are three copies rather than two, and the guard that exists is a
round-trip test over a hand-written fixture — so a type added on one side and
forgotten on the other is caught only if someone remembers to extend the fixture.

The BYOE spec's design point 3 predicted this in as many words: *"The record schema
wants to be a leaf crate."*

## Decision

### 1. Extract `rlevo-scene`

A new leaf crate, `#![no_std]` with `extern crate alloc`, holding the **data and
the codec** of everything that crosses the boundary as bincode:

- the styled-text types — `StyledFrame`, `StyledLine`, `StyledSpan`, `SpanStyle`,
  `Color`, `Modifier`
- `render::palette`'s constants, which are pure data over those types
- the payload types — every `*Snapshot`, and under ADR 0082 the scene types
- the record schema — `EnvFamily`, `FamilyPayload`, `EpisodeRecord`,
  `MetricSample`, `FrameRecord`, `RecordChunk`, and the rest
- `FORMAT_VERSION`, `MIN_SUPPORTED_VERSION`, `bincode_config()`
- `decode_episode_record`, its chunk reader, and `DecodeError`

> **Amended.** The original list also named **`RunManifest`** and **`Bounds`**.
> Neither belongs here; see decision 2 for the rule that excludes them.

Dependencies: `serde`, `bincode`, and `thiserror`, **all three taken from
`[workspace.dependencies]`**. The crate must build for `wasm32-unknown-unknown`
with no feature gating, and that is a CI target rather than an assumption.

> **Amended.** The original clause read *"`serde` and `bincode`, both with
> `alloc` features. Nothing else."* Two of the three claims were wrong.
>
> **`thiserror` is required.** `BoundsError` and `DecodeError` both derive
> `thiserror::Error`. The measurement behind "nothing else" came from grepping
> `use` lines, and a fully-qualified derive path never appears on one. thiserror
> 2.0 is `no_std`-capable, is a proc-macro plus `core`, and already ships in the
> report client's wasm32 cone, so it costs the leaf nothing it was avoiding.
>
> **Workspace entries, not pinned locally, and no `default-features = false`.**
> Taking the versions from the workspace is what stops this crate drifting from
> the writer and the reader it sits between — which matters most for `bincode`,
> because it is the codec and a version skew across the three crates is a silent
> wire break, the same defect class this crate closes for the *types*. That
> promoted `bincode` to `[workspace.dependencies]`; it had been declared **four**
> separate times. Forcing `default-features = false` on shared entries was
> dropped: cargo unifies features across the graph, so serde resolves with `std`
> anyway in any build containing `rlevo-core`, and the override would buy a
> guarantee the workspace cannot keep while costing a second version to maintain.
> The crate is `#![no_std]` in its own right and builds for wasm32, which is the
> property actually being asked for.

**Not zero-dep, unlike the ADR 0015 precedent.** These types need `String` and
`Vec`, so `no_std` + `alloc` is the honest ceiling. The property that matters is
unchanged: no burn, no rand, no getrandom, no transitive path to any of them.
Verified — the resolved cone is `serde`, `bincode`, `thiserror`, and nothing else.

**On the name.** `rlevo-scene` is broader than its contents strictly warrant
today — the crate also holds styled text and the codec, neither of which is a
scene. It is named for the part that dominates it in size, changes most, and
third parties actually reach for, and it reads as a vocabulary rather than a
mechanism.

> **Amended, in the name's favour.** The original sentence listed *"styled text,
> the run manifest, and the codec"* as the parts the name did not fit. The
> manifest is no longer in the crate, and `ObjectiveSense` never enters it, so
> the gap between the name and the contents is narrower than when the name was
> chosen. The question was reopened on that basis and `rlevo-scene` was kept. Two rejected candidates and why: `rlevo-record` collides with
`rlevo-benchmarks::record`, the module that will re-export it, so every doc
sentence would need to say which one it meant; `rlevo-wire` names the transport
rather than the thing transported, and the crate's job is to define shapes, not to
move bytes. Under ADR 0082 the scene payload absorbs four of the six payload
variants, so the name gets more accurate rather than less.

### 2. The boundary is the transport, and core keeps the traits

`rlevo-core::render` stays, and keeps every trait: `AsciiRenderable`,
`AsciiRenderer`, `Renderer`, and the `*PayloadSource` family. Those are
producer-side obligations on environment types and belong beside `Environment`.

It re-exports the leaf's types, so existing paths like
`rlevo_core::render::Point2` keep resolving and no environment changes an import.
`rlevo-core` gains `rlevo-scene` as a normal dependency.

So the first half of the split is: **the leaf owns what goes on the wire; core
owns what an environment must implement to put it there.**

> **Amended — the second half was missing, and it is the one that decides the
> hard cases.** "What goes on the wire" is not one thing. Two formats cross this
> boundary with opposite failure modes:
>
> | Crosses as | Keyed by | Drift behaviour | Treatment |
> |---|---|---|---|
> | bincode | position / variant tag | **silent corruption** | one definition, in the leaf |
> | JSON | field name | graceful — unknown ignored, missing become `None` | mirroring is safe |
>
> Measured: the report emitter does `serde_json::to_string(run.manifest())` and
> the client reads it from a `<script type="application/json">` block, while
> episode records travel as length-prefixed bincode. The drift guard and the
> three-way fork this ADR exists to kill are entirely a **bincode** problem.
>
> **Mirror what is self-describing; share what is positional.**
>
> That rule excludes `RunManifest` from the leaf, and excluding it also keeps
> `ObjectiveSense` in `rlevo-core` — which is where an optimisation concept
> belongs, and which the original list would have dragged into a record crate as
> a side effect of a boundary drawn one level too coarse.
>
> It excludes **`Bounds`** for a second, independent reason: nothing on today's
> wire uses it (`Landscape2DSnapshot` carries bare `(f32, f32)`,
> `Box2dSnapshot::world_bounds` a `(Point2, Point2)`), while **47 files** across
> `rlevo-evolution`, `rlevo-reinforcement-learning`, and `rlevo-environments`
> name it as a config-validation primitive — 173 mentions in the first two alone.
> The original argument for moving it was purely to pre-satisfy ADR 0082, and
> relocating a workspace-wide primitive into a record crate to serve a type that
> does not exist yet puts it in the wrong place permanently. ADR 0082's viewport
> range is a wire concern and gets its own type in the leaf; that a config bound
> and a viewport extent share an invariant does not make them the same type.

### 3. Copies 2 and 3 are deleted

- The six `*Payload` "bincode-stable mirror" structs in `schema.rs` and their
  `From` impls **go away**. The producer-facing type is the wire type.
- `rlevo-benchmarks::record` becomes a thin re-export module over `rlevo-scene`,
  exactly as `rlevo-benchmarks::metrics_registry` is today over
  `rlevo-metrics-registry` (ADR 0015's shape).
- `rlevo-benchmarks-report-client/src/wire.rs` is **all but deleted** and
  replaced by `use rlevo_scene::…`. The client takes `rlevo-scene` as a normal
  dependency.

  > **Amended** from *"is **deleted** — 839 lines"*. What survives is the
  > `RunManifest` view and the `ObjectiveSense` it carries — roughly 70 of the
  > 839 lines — because the manifest arrives as JSON and decision 2 says a
  > self-describing mirror is safe. This is a mirror kept **on purpose**, with a
  > stated reason, rather than one nobody got round to deleting; it should carry
  > a comment saying so, or a later reader will "finish the job" and re-couple
  > the client to `rlevo-core`.
- `crates/rlevo-benchmarks/tests/wire_format_compat.rs` is **deleted** — 495
  lines. A drift guard between two definitions is unnecessary when there is one
  definition. This is the load-bearing win: the test could only ever catch drift
  it had been taught to look for.

The client's `styled.rs` (Leptos `StyledFrame` → HTML) stays in the client. It is
rendering, not data.

### 4. The new crate denies `missing_docs` from day one

**#945** measures roughly a third of `payload.rs`'s public surface as
undocumented, against a rules "zero-exception policy": `Point2.x`/`.y`, every
variant of `BodyKind`, `GridDir`, `GridColor`, and `GridDoorState`, the
`TabularLayout` discriminant, and all five `CardTable` fields — while
`GridSnapshot` and `Locomotion2DSnapshot` document every field, so the file is
inconsistent with itself.

**This extraction is the moment that debt becomes public-API debt**, because the
code stops being a module inside a crate and becomes a published crate with its
own docs.rs page. #945 suggests enabling `missing_docs` at `warn`; for a *new*
crate there is nothing to grandfather, so it goes in at `deny` and the class is
self-policing from the first commit. The debt is paid once, during a move that
touches every line anyway.

Two of #945's items evaporate rather than getting fixed: `Point2` becomes
`Point3` and `BodyKind` becomes an open `role` key, both under ADR 0082.

### 5. Wire types declare `Send + Sync`

**#991** (with **#697**) reports that `LocomotionSnapshot`'s `Send + Sync` is
accidental — true only because every type it happens to contain is — with no
declaration and no compile-time signal if that changes. Record payloads cross
rayon worker threads on every run: the sink is
`Arc<parking_lot::Mutex<dyn RecordSink>>` and `RecordSink: Send + 'static`.

A new crate is the cheap moment to make that a stated contract rather than a
coincidence, and a leaf with two dependencies is the easiest place in the
workspace to keep it true.

### 6. Publishing and cone

`rlevo-scene` is published, and must publish **before** `rlevo-core`,
`rlevo-benchmarks`, and the client. It joins the `PUBLISHABLE` list in
`xtask/common/stage.sh`'s callers, ahead of `rlevo-core`.

Consumer cost: one additional first-party crate in every `rlevo` consumer's graph,
and **zero** additional third-party crates — `serde` and `bincode` are already in
the cone. ADR 0080 measured the analogous addition at 254 → 256 nodes.

### 7. Non-decisions

This ADR does not change any type's *contents*, any wire tag, or
`FORMAT_VERSION`. It is a pure relocation plus two deletions. ADR 0082 changes the
contents, and does so afterwards.

## Consequences

**Roughly 1,250 lines deleted outright** — about 770 of `wire.rs`'s 839, plus the
495-line drift guard — against a new crate that is mostly moved code plus a
manifest. The six mirror structs and their `From` impls go too.

> **Amended** from "roughly 1,300". The ~70-line `RunManifest` JSON view stays in
> the client by decision 2.

**One definition, so drift stops being a category.** Today a field added to
`Locomotion2DSnapshot` must be propagated to `Locomotion2DPayload` and then to the
client's copy, with a round-trip test over a hand-written fixture as the only net.
After this, it is one edit.

**A new failure mode, named honestly.** With producer and wire types unified, a
change to a producer-facing type *is* a wire change, and the insulation copy 2
provided is gone. That insulation was the stated reason for the mirror, so
removing it needs a replacement: `FORMAT_VERSION` and the packaging probes are it,
plus the rule that `rlevo-scene` is the crate where "does this break the format?"
is always the first review question. A reviewer should push on whether that is
enough.

**`rlevo-core` gets smaller and more honest.** The one module in it that had no
business needing burn stops living behind burn.

**Breaking, but shallowly.** Type paths move; `rlevo-core::render` re-exports keep
first-party call sites compiling. External consumers naming
`rlevo_benchmarks::record::schema::Box2dPayload` must move to
`rlevo_scene::Box2dSnapshot` — the type both merges and renames. CHANGELOG entry
with a migration table required.

**It is a topology change**, which the BYOE spec defers to Phase 4. The deferral
was about *umbrella membership and per-family splits decided from prose*; this is
an extraction forced by a measured three-way fork, and it adds a leaf rather than
moving a boundary. Recording the tension rather than pretending it is absent: if a
reviewer thinks Phase 4 should own this, the counter-argument is that ADR 0082
cannot land cleanly until it is settled.

**Sequencing risk.** If this lands and 0082 does not, the workspace keeps six
payload shapes in a crate named for a scene vocabulary it has not yet adopted.
Still strictly better than three copies, and the extraction stands alone — but the
name would be writing a cheque 0082 has to cash.

## Alternatives considered

**1. Extract only `rlevo-core::render`.** Smaller, and it is the module the
question was asked about. Rejected: it kills 42% of the mirror and leaves 55%
alive, and it preserves the incoherent boundary — the payload types would live in
a leaf while the schema types wrapping them stay in `rlevo-benchmarks`, still
duplicated in the client. The fork is one fork; splitting the fix in half fixes
half of it.

**2. Two leaf crates — `rlevo-render` and `rlevo-scene-schema`.** Cleaner
separation of concerns on paper. Rejected: the concerns are not separate. The
schema's payload variants *are* the render module's types, which is why copy 2
exists at all. Two crates would need one to depend on the other, and the split
point would be arbitrary.

**3. Keep the mirrors; strengthen the drift guard.** ADR 0015 considered and
rejected the analogous "mirror-and-guard" option for the metric table, and the
argument is stronger here. A guard over a hand-written fixture catches drift in
types it already knows about; the failure mode that actually bites is a *new* type
added on one side.

**4. Make `rlevo-core` WASM-buildable instead**, by feature-gating burn. Then the
client depends on `rlevo-core` directly and no extraction is needed. Rejected:
burn is not incidental to `rlevo-core` — it is `TensorConvertible`'s whole reason
to exist — and a `no-burn` feature on the workspace's foundational crate would be
a large, permanently load-bearing cfg surface to avoid moving 1,200 lines of
already-independent code. It also would not touch copy 2.

**5. Do it after ADR 0082.** Rejected on sequencing: 0082 rewrites all four
geometric payloads plus the scene types, and doing that before the extraction
means writing the new types three times and merging them afterwards.

## References

- ADR [0015](0015-shared-typed-metric-registry-crate.md) — the precedent; a forked table became a `no_std` leaf crate
- ADR [0080](0080-harness-owns-the-environment-glue-and-the-umbrella-carries-the-harness.md) — the same "already writable from the other side" finding, applied to the fixtures glue
- ADR [0082](0082-the-scene-is-the-payload-and-identity-is-open.md) — sequenced after this; its scene types land in `rlevo-scene`
- ADR [0007](0007-visualisation-crates-isolated-from-production-crates.md) — the viz-dependency prohibition `rlevo-scene` must satisfy
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — the `Bounds` newtype, which **stays in `rlevo-core`** (amended; see decision 2)
- Issues reviewed before drafting: **#942** (`Bounds` for payload ranges; its bincode caveat is discharged in Status, with the opposite consequence to the one assumed), **#945** (a third of `payload.rs`'s public surface undocumented — hence `deny(missing_docs)`; 29 items paid off on the first commit), **#991** / **#697** (accidental `Send + Sync`)
- Spec: `docs/.private/specs/2026-08-21-byoe-first-class-citizen/` (design point 3, "The record schema wants to be a leaf crate")
