---
project: rlevo
status: active
type: decision
date: 2026-08-23
tags: [adr, decision, architecture, record-schema, wire-format, rendering, rlevo-benchmarks, rlevo-core, rlevo-metrics-registry, byoe, byoa, breaking-change]
---

# ADR 0082: The scene is the payload, and identity is open

## Status

**Accepted (2026-08-23). Decision 4 implemented; decisions 1, 2, 3, 5, 6, 7
outstanding.** Sequence steps 1–3 have landed: the golden-frame tests, the scene
types in `rlevo-scene`, `ScenePayloadSource`, `RecordSink::on_scene`,
`FamilyPayload::Scene`, and the single `FORMAT_VERSION` bump 8 → 9. No producer
has migrated yet, so every existing family payload is still live and the scene
path has one synthetic producer — the test fixture — behind it.

Six clauses of this ADR were contradicted by implementing it; they are tabulated
under *Amendments from implementation* rather than edited away.

**Accepted with #276 open by design** (decision 8). If that fork later resolves to
"plumb `SnapshotMetadata` to the record tier", it costs a second `FORMAT_VERSION`
bump. That was weighed and accepted rather than overlooked.

**This ADR is not discharged on acceptance.** BYOE-1 step 8 stays red until its
probe emits a scene (Consequences), and the golden-frame prerequisite in decision
6 gates the migration.

**Amended 2026-08-23**, when ADR 0081's first implementation slice landed and
corrected where `Bounds` lives. This ADR specified `SceneDescriptor::bounds` as a
`Bounds` triple on the assumption that ADR 0081 would move `Bounds` into the leaf
crate; it does not, so the field gets a leaf-native range type instead. See
decision 4's viewport-range subsection, retitled and rewritten. The
*intent* — a validated, per-axis viewport range rather than four bare tuples — is
unchanged, and #942's argument is still adopted in full.

**Sequenced after ADR
[0081](0081-the-record-format-is-a-leaf-crate.md)**, which extracts the
`rlevo-scene` leaf crate this ADR's new types land in. Not a hard dependency —
see Consequences — but taking 0081 second would mean writing the scene types three
times and merging them afterwards. The ADR numbers follow the implementation
order.

Supersedes no ADR. Amends the *use* of two:

- **ADR 0015** established `rlevo-metrics-registry` as the single source of truth
  for canonical metric names. This ADR keeps the table and removes one of its two
  jobs: it stops being an ingestion allowlist and becomes a descriptor table only.
- **ADR 0014** owns the record schema. This ADR changes what `EnvFamily` is, what
  dispatches on it, and adds a payload variant, which is wire-format work and
  therefore lands here rather than in a topology ADR.

Discharges the spec's requirement that the 3-D `LocomotionPayload` migration land
in the same scope as the extension-point decision: under this ADR the two are the
same change, because locomotion migrating to the scene payload **is** the 3-D
migration.

## Context

Blocker B3 in the BYOE spec was stated as *"a third-party environment cannot name
itself."* Two acceptance probes have since executed against it, and the executed
result is narrower and stranger than the statement.

### The vocabulary problem has three axes, and only two are in scope

| Axis | Question | State | Scope |
|---|---|---|---|
| **Producer** | can a foreign env *emit* a rich payload? | **already open** — `RecordingTap::with_payload_extractor` takes `Fn(&E) -> FamilyPayload`, and all six `*PayloadSource` traits live in `rlevo-core/src/render/payload.rs` | done |
| **Vocabulary** | can it *name* a shape the report dispatches on, and is there a shape that fits? | **closed on both counts** | **this ADR** |
| **Consumer** | can a foreign *renderer* reach the report? | whole-client swap only, via `ReportOptions::client_assets` | BYOE-3 |

The consumer axis moved to BYOE-3 on 2026-08-23. A caller-supplied *renderer*
needs caller-supplied drawing code; a caller-supplied *scene* does not, and that
distinction is what makes this ADR tractable while BYOE-3 stays deferred.

### The metrics axis already solved the naming half, then broke it

`rlevo-metrics-registry` answers the identical question — *how do a producer and a
consumer in different crates agree on a vocabulary when a third party may
introduce a term neither knows?* — on the metrics axis, and gave the **opposite**
answer from `EnvFamily`: a `&'static str` key, a closed descriptor table that
enriches known names, and per-field fallback for unknown ones (`title_for` yields
the raw name, `trend_for` yields `Diagnostic`, `hint_for` yields `""`).

That contract is doctested in the leaf crate and independently asserted in the
report client, and **it does not work**, because the producer half contradicts it:

```rust
fn record_canonical(&mut self, field: &Field, value: f64) {
    if is_canonical_metric(field.name()) {
        self.metrics.push((field.name().to_string(), value));
    }
}
```

Every `tracing` field whose name is absent from `CANONICAL_METRICS` is discarded
before it reaches the sink. **BYOA-1 step 8 measured it**: a submitted algorithm
emitting `qlearn_mean_abs_td` and `qlearn_updates` produces a record containing
`episode_length`, `episode_return`, and `episode_wall_clock_secs` and nothing
else. The client's `title_for("my_custom_metric") == "my_custom_metric"` test is
exercising a path no producer can reach.

One table, two contradictory contracts: **allowlist at ingest, graceful fallback
at render.** The filter is not gratuitous — its own unit test asserts
`!is_canonical_metric("batch_size")`, so the intent was to keep hyperparameter
noise out of the chart list. The defect is that "reject noise" was implemented as
"reject everything unrecognised", which also rejects signal.

The metrics axis is therefore not the counter-example to `EnvFamily`; it is **the
same defect at a site where the fix is small**, which makes it the right place to
establish the rule this ADR then applies twice more.

### `EnvFamily` is doing two jobs and is load-bearing for one

Measured across the workspace, `EnvFamily` has exactly three consumers:

| Consumer | Use |
|---|---|
| `adapters::render(family, frame)` | picks the report adapter |
| `default_frame_stride(family)` | `Locomotion => 6`, `Box2d => 4`, `_ => 1` |
| `app.rs` meta row, `html.rs` `<dt>env family</dt>` | displayed as run metadata |

The first is **redundant**. Every one of the six family adapters opens with a
match on the *payload* variant and delegates to
`fallback::render(.., UnsupportedPayload)` on anything else — one payload variant
each. The outer dispatch on family selects an adapter that then re-derives its
decision from the payload. Two layers, one real predicate.

The client already half-knows this. `FallbackReason` was split because the two
concerns diverge: `NoFamilyAdapter` (whose own doc calls it *"structurally
unreachable today"*) versus `UnsupportedPayload`, which fires six times out of
six. `default_frame_stride` keys off exactly the two families whose payloads are
`Locomotion2D` and `Box2dBodies` — a payload property wearing a family's name.

Only the third job needs `EnvFamily` to exist at all.

### The naming defect and the shape defect are different defects

The six payload shapes cover the six first-party families. That yields two
third-party situations, and conflating them is why B3 read as one problem:

1. **A foreign env whose natural view is one of the six.** It can already emit the
   rich payload — but must declare somebody else's `EnvFamily` to record at all,
   and the report then labels the run with that lie. A **naming** defect.
2. **A foreign env whose natural view is none of the six** — BYOE-1's thermostat.
   It gets `FamilyPayload::Ascii`. **No naming scheme fixes this**, because the gap
   is a missing shape, not a missing label. This is what BYOE-1 step 8 measures.

An earlier draft of this ADR fixed only the naming defect and left step 8 red.
That was rejected on review: the shape defect is the one the acceptance test
actually encodes.

### The four geometric payloads are already the same shape

Decomposed, the continuous-geometry payloads differ in vocabulary, not structure:

| Payload | Geometry | Pose | Style key | Viewport | Transients |
|---|---|---|---|---|---|
| `Classic2DBody` | world-space polyline + `closed` | pre-applied | `Classic2DRole` (6 variants) | `bounds` | — |
| `RigidBody2D` | local-frame polygon | `position` + `rotation_rad` | `BodyKind` (7 variants) | `world_bounds` | `contacts` |
| `Locomotion2DSnapshot` | `joints` + `bones` | pre-applied | implicit | none | `ground_y`, `com`, `contacts` |
| `Landscape2DSnapshot` | named heatmap | — | — | `bounds_x`/`bounds_y` | `current`, `best`, `trail` |

Three of the four are *posed primitives with a semantic style key*. Both style
keys are closed enums of six or seven values driving CSS class selection — the
same closure as `EnvFamily`, one level down.

`Grid` and `TabularText` are **not** in this family and must not be folded into
it. They are discrete semantic structures whose renderers know what a door, a
key, and a dealt card are. Generalizing them to geometry would replace meaning
with coordinates. That boundary is a decision, not an omission.

**#846 is the evidence, not just the intuition.** It reports that `grid_snapshot`
silently relocates an off-grid agent to the origin — where the ASCII renderer
simply does not draw it — and its *preferred* fix is to "make `GridAgentMarker`
carry `Option` position / an in-bounds flag so an off-grid agent is representable
rather than relocated". That is a semantic type change. A mesh-and-transform scene
has no vocabulary for "this marker is nowhere"; it would draw the agent at the
origin forever. Migrating `Grid` would foreclose #846's fix.

**#873** confirms the exclusion is free on the producer side: `empty.rs` diverges
from the other grid envs on `SnapshotType`, but the issue notes both "still
implement `GridPayloadSource`, so the report path is unaffected either way".

## Decision

### 1. The registry is a descriptor table, never a gate

`rlevo-metrics-registry` keeps `CANONICAL_METRICS` and every lookup function
unchanged. `RecordingLayer` stops using `is_canonical_metric` as an admission
test. The new ingest predicate is:

> Record a numeric `tracing` field if its name is canonical **or** carries the
> reserved `agent.` prefix.

Canonical names keep working verbatim, so no first-party algorithm changes. A
third party opts in by prefixing: `tracing::info!(agent.my_algo_kl = 0.3)`.
Unprefixed unrecognised fields are still dropped, so `batch_size` stays out of the
chart list and the filter's original intent is preserved.

`agent.` rather than a new word because **ADR 0079 already reserved the `agent/`
prefix** for displaced agent-supplied metrics in `TrialReport`. One reserved
namespace, two transports; `.` instead of `/` because `tracing` field names admit
dotted paths and not slashes. The record stores the name as written, prefix
included, so the transports stay distinguishable downstream.

### 2. The report dispatches on `FamilyPayload`, not `EnvFamily`

`adapters::render` takes the frame alone and selects the adapter from the payload
variant. The six per-adapter payload matches collapse into one.
`FallbackReason::NoFamilyAdapter` becomes genuinely unreachable and is **removed**,
along with the `#[allow(unreachable_patterns)]` on the old outer match.
`UnsupportedPayload` remains.

### 3. `EnvFamily` becomes open identity

`EnvFamily` stops being an enum. It becomes a newtype over a string key with a
closed descriptor table beside it — the metrics-registry shape, now that the
metrics-registry shape is honest.

- The six existing names become table entries with display titles. Constructors
  named for the six are retained, so first-party call sites and
  `RecordedEnvFamily` impls do not churn.
- Any other key is accepted, displayed verbatim, enriched with nothing. A
  thermostat names itself `thermostat`.
- `RecordedEnvFamily::FAMILY` and `RunManifest::env_family` both carry it, so both
  closed sites open together.

### 4. `FamilyPayload::Scene` — one shape for continuous geometry

A new variant, 3-D native, with static geometry separated from per-frame pose.
The types land in **`rlevo-scene`**, the leaf crate ADR 0081 extracts, alongside
the payload shapes they replace — not in `rlevo-core`, which cannot reach the
report client. Sketch:

```rust
/// 3-D point. 2-D producers set `z = 0`; the renderer projects.
pub struct Point3 { pub x: f32, pub y: f32, pub z: f32 }

/// Translation plus unit quaternion, scalar-first — matching the physics
/// layer's own `Pose`, so locomotion stops discarding orientation.
pub struct Transform3 { pub position: Point3, pub orientation: [f32; 4] }

/// Stable handle for a node, assigned by the producer. See "Node identity".
pub struct NodeId(pub u32);

/// Endpoints of one bone. A named struct, not a bare `(u32, u32)` — #944's
/// companion item, which is only worth doing alongside a `FORMAT_VERSION`
/// bump, and this is that bump.
pub struct Bone { pub from: u32, pub to: u32 }

/// What a node is made of, in its own local frame.
pub enum Geometry {
    /// Ordered points; `closed` fills the polygon.
    Polyline { points: Vec<Point3>, closed: bool },
    /// Shared vertex pool plus bones — joints and links without duplication.
    Skeleton { vertices: Vec<Point3>, bones: Vec<Bone> },
    /// A single point, drawn as a marker.
    Marker,
    /// Analytic solids, so a 3-D renderer never needs a mesh on the wire.
    Box { half_extents: Point3 },
    Sphere { radius: f32 },
    Capsule { half_height: f32, radius: f32 },
}

/// One drawable. Recorded once per episode.
pub struct SceneNode {
    pub id: NodeId,
    pub geometry: Geometry,
    /// **Open** style key, table-enriched. Unknown keys get a default style.
    pub role: String,
    /// Set when the node never moves; `None` means a per-frame pose supplies it.
    pub static_transform: Option<Transform3>,
}

/// Recorded once, at episode start.
pub struct SceneDescriptor {
    pub nodes: Vec<SceneNode>,
    /// Viewport the renderer fits to, one `Extent` per axis. A validated
    /// newtype rather than a bare tuple — see below.
    pub bounds: (Extent, Extent, Extent),
    /// Named background — a landscape heatmap, a ground plane. `None` for neither.
    pub background: Option<String>,
}

/// Recorded per frame. This is what `FamilyPayload::Scene` carries.
pub struct ScenePose {
    /// Keyed by node id, not positional. A node absent from this map keeps
    /// its previous pose.
    pub transforms: Vec<(NodeId, Transform3)>,
    /// Transient points — contacts, centre of mass, trail — tagged by role.
    pub markers: Vec<(String, Point3)>,
}
```

Three properties are load-bearing:

- **Geometry is recorded once, poses per frame.** Naive per-frame geometry would
  multiply a 3-D record's size by the frame count for zero information gain. This
  is the spec's design point 1, honoured rather than deferred — and it now also
  answers **#671**, which flags `box2d/physics.rs::snapshot()` allocating a fresh
  `Vec<BodyRecord>` per call and defers the fix until "a hot caller shows up".
  Recording geometry per frame would have been that caller.
- **`role` is a `String`, not an enum.** Same argument as decision 3, one level
  down. `BodyKind` and `Classic2DRole` become table entries; a third party writes
  `"thermometer"` and gets a default style rather than a compile error.
- **Poses are keyed, not positional.** See below.

#### Node identity — a defect this ADR introduced and #944 caught

An earlier draft declared `ScenePose::transforms: Vec<Transform3>` as *"one
transform per moving node, in `SceneDescriptor::nodes` order, skipping nodes that
carry a `static_transform`."* That is an implicit parallel-array correspondence
**spanning two separately-recorded structures** — one written at episode start,
one per frame — with the skip rule as an extra unwritten precondition.

**#944** names exactly this defect class in the payloads being replaced:
`GridSnapshot`'s `tiles.len() == width * height`, `TabularGrid`'s equivalent,
`Locomotion2DSnapshot`'s `bones` indexing into `joints`, and `RigidBody2D`'s
documented-but-unchecked winding. Its summary is the reason to take this
seriously: a buggy producer *"ships a malformed payload that fails (or silently
mis-renders) only in the WASM report client — the worst place to debug it."*

The positional version was **worse than any of those**, because #944's proposed
remedy — a per-snapshot `is_consistent()` the recording layer `debug_assert!`s at
capture — cannot check it. Neither struct alone holds enough information; the
descriptor is long gone by the time a pose is written.

Keying transforms by `NodeId` removes the coupling rather than documenting it. An
unknown id is ignorable, a missing id means "unchanged", and a reordered
descriptor is harmless.

#### #944's remedy applies to what remains

`Skeleton` still carries a bone-index-into-`vertices` invariant, and `Polyline`
still has winding. Both ship with `#[must_use] pub fn is_consistent(&self) -> bool`
per #944's suggested fix, `debug_assert!`ed by the recording tier at capture time.
`bones: Vec<Bone>` rather than `Vec<(u32, u32)>` adopts #944's "Related, lower
priority" companion, which it scopes to a `FORMAT_VERSION` bump — this is that
bump, and doing it later would need another one.

#### The viewport range is a validated newtype — a leaf-native one

```rust
/// An inclusive `[lo, hi]` range over one axis, valid by construction.
/// Serializes as `(f32, f32)` via `#[serde(try_from, into)]`.
pub struct Extent { lo: f32, hi: f32 }
```

`SceneDescriptor::bounds` uses a validated range newtype per **#942**, which flags
`Landscape2DSnapshot`'s `bounds_x`/`bounds_y` as *"a straggler from that
migration"* — an inverted or NaN range sailing through serialization into the
client, "whose viewport-fitting math then divides by a negative or NaN span".

> **Amended.** This subsection originally said the field *"uses `Bounds`
> (ADR 0027)"*, because ADR 0081 was going to move `Bounds` into `rlevo-scene`.
> It does not: `Bounds` is a config-validation primitive named in 47 files across
> evolution, RL, and environments, and nothing on today's wire uses it, so it
> stays in `rlevo-core` — which the leaf cannot reach.
>
> The field therefore takes a **leaf-native** range type carrying the same
> invariant (`lo <= hi`, so NaN is excluded) and the same
> `#[serde(try_from, into)]` mechanism. That mechanism is **verified** rather
> than assumed: see ADR 0081's Status. Validation runs under bincode and the
> encoded form is byte-identical to a bare `(f32, f32)`, so this costs no wire
> change.
>
> Two types with one invariant is the right outcome, not a duplication to
> apologise for. A config bound constrains a hyperparameter at construction; a
> viewport extent describes what a renderer should fit. They share an invariant
> and nothing else, and collapsing them would put a workspace-wide config
> primitive in a record crate permanently to save one small type.

The scope is wider than #942's title suggests. `payload.rs` has **four** bare
bounds sites — `Landscape2DSnapshot`'s two, `Box2dSnapshot::world_bounds`,
`Classic2DSnapshot::bounds` — and all four collapse into this one field. Deleting
`Landscape2D` (decision 6) moots #942's literal target while generalizing its
argument; getting the type right once fixes what four sites had wrong.

Per #942, the newtype serializes as `(f32, f32)` through
`#[serde(try_from, into)]`, so the wire shape is unchanged and deserialization
simply begins rejecting invalid ranges. **#942's second caveat is discharged**:
`try_from`-validated deserialization was asserted to behave under bincode, and it
does — the same bytes that decode cleanly as a `(f32, f32)` are rejected when
decoded as a validated range. The "validate at construction instead" fallback is
not needed. Guard: `crates/rlevo-benchmarks/tests/bounds_bincode_validation.rs`.

A per-axis `Extent` triple also makes a non-square domain representable, which is
the blocker **#304** is explicitly waiting on.

`SceneDescriptor` reaches the sink through a **new defaulted** method,
`RecordSink::on_scene(&mut self, _descriptor: SceneDescriptor) {}`, rather than by
changing `on_episode_start`'s signature. `RecordSink` is a public trait with
unbounded implementors, and ADR 0079's polarity lesson applies: additive and
defaulted, so no existing implementor breaks and no implementor is forced to care.

A `ScenePayloadSource` trait joins the surviving `*PayloadSource` traits in
`rlevo-core::render`. Per ADR 0081's split, **the leaf owns the data and core owns
the trait**: `ScenePayloadSource::scene_pose` returns `rlevo-scene`'s types, and
lives beside `Environment` because it is an obligation on an environment type.

### 5. Locomotion migrates to `Scene`, and that is the 3-D migration

`Locomotion2DPayload` is the worked case, chosen because it is the one payload
whose lossiness is already documented: the physics layer computes full 3-D `Pose`,
`Twist`, and a 6-DOF contact wrench, and the record keeps a sagittal projection.
`Skeleton` plus `Transform3` carries what was being discarded. The sagittal 2-D
view becomes a projection performed at render time rather than a fact baked into
the wire.

There is no separate `LocomotionPayload` 3-D variant. Adding one *and* a scene
payload would be the same wire change twice.

### 6. All four geometric payloads migrate; `Landscape2D` is deleted outright

`Classic2D` and `Box2dBodies` migrate to `Scene` in this ADR's scope, and their
variants, payload structs, producer traits, tap constructors, and client adapters
are removed. No deprecation window: a shape nobody may emit and nobody renders is
not worth keeping compilable.

**`Landscape2D` is deleted rather than migrated, because it has no producer.**
Measured: zero environments implement `Landscape2DPayloadSource` anywhere in the
workspace. The only impls are the `TimeLimit` forwarder — whose own doc comment
says *"No shipped environment implements this"* — and two test fixtures. A payload
struct, a producer trait, a 207-line client adapter, and a wire mirror exist for a
shape nothing emits. Migrating it would port untested, unreached code onto a new
foundation and re-verify nothing.

The persona that would need it is **BYOE-2**, the optimizer sibling walking a
caller-supplied `Landscape`, which has not started. A landscape view returns then,
as a scene, with a real producer to test it against. Carrying an untested adapter
against a future need is how this one arrived.

**A precision #304 forces.** The payload is not orphaned by accident — it was
built *ahead* of a planned producer: `landscapes/render.rs` anticipates
candidate-overlay markers "once `rlevo-evolution` emits `FrameRecord`s", and #304
cites `Landscape2DSnapshot { bounds_x, bounds_y }` as the anchor it is waiting on,
"already per-axis, so it needs no schema change". Deleting it therefore has a
consequence beyond removing dead code, and #304 must be re-triaged rather than
left pointing at a type that no longer exists.

The deletion is still the right call, and it arguably *helps* #304: that issue is
blocked because the ASCII tier would draw a true-domain rectangle while the report
tier draws a search box, so "the same landscape produces two disagreeing frames".
Removing the report-tier rectangle removes one of the two conventions, and
decision 4's per-axis `Extent` triple makes the surviving one representable.

`Grid` and `TabularText` are **not** deprecated and will not migrate. See Context.

**The migration is gated on a test net that does not yet exist.** The four
adapters being replaced carry **one test between them** (locomotion); classic,
box2d, and landscape have zero, and there are no snapshot or SVG assertions
anywhere in the adapter tree. Replacing 818 lines of untested view code with ~350
lines of untested view code is a visual regression surface with no automated
detection. Therefore:

> **Golden-frame tests for `classic`, `box2d`, and `locomotion` must land against
> the *current* adapters before any producer migrates.** They assert structural
> facts about the emitted markup — node count, roles present, bounds, key
> coordinates — not pixels. Written first so they are red-green against the
> migration rather than green-green against whatever the new code happens to
> produce.

This is a prerequisite, not a follow-up. Without it the migration's only
verification is opening four reports and looking.

### 6a. Sequence

1. Golden-frame tests for `classic`, `box2d`, `locomotion`, against current adapters.
2. Scene types in `rlevo-scene` (ADR 0081); `ScenePayloadSource` in
   `rlevo-core::render`; `RecordSink::on_scene`.
3. `FamilyPayload::Scene`, one `FORMAT_VERSION` bump, wire mirror, `wire_format_compat`.

> **Amended 2026-08-23**, when steps 1–3 landed. Steps 1 and 3 are as written;
> step 2's address is wrong and its trait is under-specified. Corrections are
> tabulated in *Amendments from implementation* below.
4. **Locomotion first** — it is the 3-D case and the only adapter with prior coverage.
5. `Classic2D` and `Box2d`; delete four adapters, four tap constructors, the
   `TimeLimit` and `TuiEnvTap` forwardings, and shrink
   `payload_forwarding_completeness` from six traits to three.
6. `Landscape2D` deleted; note the return path in BYOE-2's scope.

### 7. `default_frame_stride` moves into the family descriptor table

The previous draft keyed the stride off the payload variant, which decision 6
breaks: with `Locomotion2D` and `Box2dBodies` both becoming `Scene`, a
payload-keyed stride would silently change box2d from 4 to 6.

It cannot move to `SceneDescriptor` either, tempting as that is — the descriptor
arrives at `on_episode_start`, but the stride is needed when `RecordingConfig` is
built, which is earlier. A producer-supplied stride would arrive after the writer
has already been configured with one.

So the stride becomes a **column in decision 3's family descriptor table**:
`locomotion => 6`, `box2d => 4`, the other four `=> 1`, unknown keys `=> 1`.
Existing values are preserved exactly and no first-party recording rate changes.

**This is the one behavioural use of `EnvFamily` that survives**, and the ADR
should be honest that it is a real one rather than claim family became pure
metadata. The justification is that recording rate is a property of how fast an
environment's visual state changes — genuinely per-environment, not per-payload —
so the family table is the right home for it, and always was. A third party gets
1, which over-records rather than under-records; that is the safe direction and
`RecordingConfig::frame_stride` already exists as the per-run override.

Minor API cost: `default_frame_stride` is currently `pub const fn` and a
string-keyed table lookup drops the `const`. Nothing in the workspace calls it in
a const context.

### 8. The bump is a coordination point, not just a compatibility one

Both this ADR and ADR 0081 argue a `FORMAT_VERSION` bump is cheap because
`MIN_SUPPORTED_VERSION == FORMAT_VERSION`, so there is no window to preserve.
That is true about **compatibility** and false about **coordination**: filed work
is queued behind the next bump, and this is the next bump.

| Issue | What it wants from a bump | Ruling here |
|---|---|---|
| **#944** companion — `bones` as a named `Bone` struct rather than `(u32, u32)` | scoped by the issue to "alongside a `FORMAT_VERSION` bump per ADR 0014's precedent" | **adopted**, decision 4 |
| **#942** — a validated newtype for payload ranges | wire shape unchanged, but the type changes | **adopted** as `Extent`, decision 4 |
| **#276** — `SnapshotMetadata`'s fate: plumb to the record tier, or remove | option (a) "needs a record `FORMAT_VERSION` bump" | **explicitly not decided** — see below |

**#276 is deliberately left open, and that is a cost this ADR accepts.** It asks
whether `SnapshotMetadata` — which today has *zero production consumers*, is
populated by six environments, and is dropped before `EpisodeRecord` — should be
plumbed through to per-component reward curves in the report, or removed. Its own
instruction is *"Do not implement either until the fork is decided,"* and deciding
it here would mean answering a reward-reporting question inside a rendering ADR by
whoever happened to be writing this one.

The honest consequence: if #276 later resolves to (a), it costs a second bump.
That is acceptable — `MIN_SUPPORTED_VERSION == FORMAT_VERSION` makes each bump
individually cheap, and #127 (per-step `BTreeMap` allocation for the same
metadata) suggests the representation should change before it is plumbed anywhere.
What is **not** acceptable is bumping silently and letting #276 be decided by
default.

### 9. Explicit non-decisions

This ADR does **not**: add a renderer-registration or plugin seam (BYOE-3); name a
3-D engine; migrate `Grid` or `TabularText`; or resolve **#276**.

## Amendments from implementation

Added 2026-08-23, when decision 4's types landed. Each row quotes what this ADR
said and states what is true, so the original decision stays readable and the
correction is not buried in a rewrite.

| Clause | This ADR said | What landed | Why |
|---|---|---|---|
| `ScenePayloadSource`'s address | *"joins the surviving `*PayloadSource` traits in `rlevo-core::render`"* | `rlevo-scene::scene` | `rlevo-core::render` no longer exists. ADR 0081's implementation moved the whole module — including all six `*PayloadSource` traits — into the leaf, because a trait whose only method returns a wire type cannot live apart from that type without inverting the dependency. This clause was written against 0081's *pre-amendment* split, which the same amendment already reversed. |
| `ScenePayloadSource`'s methods | names only `scene_pose` | `scene_descriptor` **and** `scene_pose` | The two halves are recorded at different rates, which is decision 4's central claim; a trait supplying only the per-frame half leaves the once-per-episode half with no producer. One trait, not two: a pose addresses nodes only a descriptor can declare, so an env implementing one without the other has no coherent meaning. |
| `Extent`'s invariant | *"`lo <= hi`, so NaN is excluded"* | also requires both endpoints finite | `lo <= hi` does exclude NaN, but admits `(-inf, inf)`. That span's reciprocal is zero, so every projected coordinate collapses onto the origin — the same silent-mis-render class the newtype exists to stop, and free to exclude at construction. |
| `Polyline::is_consistent` | *"`Polyline` still has winding"*, listed as a checked invariant | checks arity and finiteness; **winding is not checked** | Winding has no observable effect under SVG's default `nonzero` fill rule for a simple polygon, and these points are 3-D, where "counter-clockwise" is undefined without naming a viewing direction. The check would have asserted nothing. |
| How the descriptor reaches disk | not specified beyond `RecordSink::on_scene` | `RecordChunk::Scene` at bincode tag 3, plus `EpisodeRecord::scene` | The sink method needed a carrier. A chunk rather than a header field, because the header is written when the episode file opens — before the environment has been asked for anything. Decoders keep the first descriptor and ignore any later one: poses already written key against the first, so keeping it is what stays renderable. |
| Drift guards | *"The mirror in `wire.rs` and `tests/wire_format_compat.rs` make drift a test failure … both grow by the scene types"* | neither exists; nothing grew | ADR 0081's implementation deleted both. There is one definition now, so drift is not a category and the scene types needed no mirror. |

Two things the ADR specified were checked rather than assumed, and held:
`Transform3`'s scalar-first `[w, x, y, z]` does match
`rlevo-environments::locomotion::backend::Pose` exactly, and `#[serde(try_from,
into)]` does validate on the bincode decode path while encoding byte-identically
to the bare `(f32, f32)` it replaces — asserted directly against both formats in
`Extent`'s own tests rather than inherited from the `Bounds` guard.

## Consequences

**Wire format.** `FORMAT_VERSION` bumps once. `MIN_SUPPORTED_VERSION ==
FORMAT_VERSION` always holds, so there is no compatibility window to preserve and
no migration to write — the reasoning that made #306's bump cheap. `FamilyPayload`
variants append, preserving existing bincode tags. The mirror in
`rlevo-benchmarks-report-client/src/wire.rs` and `tests/wire_format_compat.rs`
make drift a test failure rather than a silent divergence; both grow by the scene
types.

> **Amended.** The last sentence is void: ADR 0081's implementation deleted both
> the mirror and `wire_format_compat`, so neither grew. Drift is not a category
> any more — there is one definition, in `rlevo-scene`. `FORMAT_VERSION` is the
> only enforcement, and it did bump exactly once, 8 → 9.

**BYOA-1 step 8 goes green** on decision 1 alone.

**BYOE-1 step 8 goes green only after the probe is updated.** The capability gap
closes here — a thermostat can emit a `Polyline` for the temperature trace, a
`Marker` for the target, and a `role` of `"thermometer"` — but the probe currently
emits `FamilyPayload::Ascii` and will keep doing so until rewritten. That rewrite
is legitimate rather than gaming the test: the probe models a researcher who wants
a view of their own domain, and handing them primitives means they draw one.
**The ADR is not discharged until that probe change lands and step 8 flips.**

**This is the largest wire change since the schema was written, and it makes the
codebase smaller.** Measured inventory:

| Site | Now | After |
|---|---|---|
| Producers migrated | 11 envs — 4 classic, 3 box2d, 4 locomotion | 11 rewritten |
| Producers for `Landscape2D` | **0** | deleted |
| Client adapters | classic 156 + box2d 205 + landscape 207 + locomotion 250 = **818** | one `scene.rs`, ~350 |
| Payload types (in `rlevo-scene` per ADR 0081) | ~330 of 551 lines | ~150 |
| `FamilyPayload` variants | 7 | 4 — `Ascii`, `Grid`, `TabularText`, `Scene` |
| Tap constructors | `with_{landscape,box2d,locomotion,classic2d}_payload` | one `with_scene_payload` |
| Forwarding impls | `TimeLimit` ×4, `TuiEnvTap` ×4 | ×1 each |
| Payload-source traits | 6 | 3 |

Roughly 1,200 lines removed against roughly 600 written. **The blast radius is
wide but shallow**: producers are mechanical (`*_snapshot()` becomes a
`SceneDescriptor` plus a `ScenePose`), and the adapters are net deletion. An
earlier draft priced this migration as "four producer rewrites plus four adapter
rewrites" and deferred it on that basis, without noticing the adapters are
replaced by fewer lines than they occupy. That was the wrong number and it drove
the wrong call.

Untouched: **21** `GridPayloadSource` and `TabularPayloadSource` impls, which is
the Context exclusion paying for itself.

**The risk is not size, it is the absent test net.** One test exists across the
four adapters being replaced. Decision 6's golden-frame prerequisite exists
entirely because of this, and it is the item most likely to be dropped under
schedule pressure and most costly to drop.

**#709 looks resolved by the open `role` key and is not.** It asks for
lunar-lander leg-contact state to be visible, and calls a new `Bodyish` variant or
contact glyph "a shared-surface decision, not a local fix", note-only pending an
ADR. An open role key does mean the *report* tier never needs a shared-surface
decision for a case like this again — `role: "leg-grounded"` needs nobody's
permission. But #709 is about `box2d/render.rs`'s **ASCII/styled** tier, and it
says plainly that the contact flags "are surfaced in the report-tier
`box2d_snapshot`" already. The tier that lacks them is the one this ADR does not
touch. #709 stays open.

**Semantics move from the enum to a table.** With `role` open, an env that writes
`"whee1"` gets a default style and no error. The closed enums caught that at
compile time for first-party code. The trade is deliberate and is the same one
decisions 1 and 3 make; the mitigation is that the descriptor table is the single
place to add a role, and first-party roles should be `const`s rather than string
literals at call sites.

**A confidently wrong picture replaces a fallback banner.** With dispatch on
payload, an env that emits a shape whose semantics do not match its data gets
drawn rather than falling back. `test_recorded_env_family_agrees_with_emitted_payload`
in `fixtures/family.rs` exists because that pairing was never type-enforced; it
should be kept and re-pointed rather than deleted with the enum.

**Breaking.** `EnvFamily::Classic` as a *pattern* stops compiling; as a
*constructor* it is retained. Exhaustive external matches were already forbidden
by `#[non_exhaustive]`, so external breakage should be limited to construction
sites, which keep working — **this needs checking against a real downstream
before the ADR is accepted, not assumed.** `FallbackReason::NoFamilyAdapter` and
`Locomotion2DPayload` leave the public surface. CHANGELOG entry with a migration
table required.

**The client gets smaller in one place and larger in another.** One dispatch layer
and one `FallbackReason` variant removed, six adapters lose an outer match arm; one
scene adapter added, and it is the most complex adapter in the client because it
must project 3-D to 2-D SVG.

**Not a topology change on its own**, but it depends on one. No crate moves and no
dependency edge changes *here*; the new leaf crate is ADR 0081's decision, taken
separately and landing first. If 0081 is rejected, this ADR still works — the
scene types go in `rlevo-core::render` and the client mirrors them by hand, as it
mirrors the six shapes they replace. That is strictly worse and is the reason 0081
exists, but it is not a blocker.

## Alternatives considered

**1. `EnvFamily::Custom { name: String }`.** The obvious move. It is decision 3
with the descriptor table and the degradation contract removed — the two parts
that do the work. It also leaves dispatch keyed on family, so a `Custom` env lands
in `NoFamilyAdapter` and renders nothing, having gained only a better banner.
Rejected.

**2. Fix naming only; leave the shape gap.** This was the previous draft of this
ADR. It closes BYOA-1 step 8, removes the masquerade, simplifies the client, and
leaves **BYOE-1 step 8 red** because the thermostat still has no shape to emit.
Rejected on review: an acceptance test the design declines to satisfy is a design
choosing its own scoreboard.

**3. A 2-D scene now, 3-D later.** Smaller, and every family except locomotion is
natively 2-D. Rejected: it guarantees a second wire bump for the one migration the
spec already named, and `z = 0` plus an identity quaternion costs a producer
nothing.

**4. Freeze the other three geometric payloads instead of migrating them.**
Deprecate-on-arrival: `Classic2D`, `Box2dBodies`, and `Landscape2D` keep working,
no new rich payload may use them, migration is tracked follow-up. This was the
previous draft's decision 6, rejected on measurement. It was justified by a
landing-risk estimate — "four producers, four adapters, the locomotion 3-D change,
and a `RecordSink` addition in one reviewable change" — that counted rewrites
without counting lines. The adapters are replaced by fewer lines than they occupy,
one of the three has no producer at all, and the producer changes are mechanical.
Freezing buys an unbounded window in which two ways to draw a polygon both
compile, in exchange for avoiding work that is mostly deletion.

**4a. Migrate `Landscape2D` rather than deleting it.** Rejected: zero producers.
The migration would port untested, unreached code and verify nothing. BYOE-2 is
the persona that needs a landscape view and it can add one as a scene, against a
real producer. Keeping it costs a wire type, a trait, and a 207-line adapter for
a shape no test exercises end to end.

**5. A closed `SceneRole` enum instead of an open `String`.** Type-safe, and CSS
class selection is genuinely finite. Rejected: it reintroduces `EnvFamily`'s exact
defect one level down, and a third party's `"robot_arm"` would land in `Other`
alongside everything else it cannot describe.

**6. Trait-supplied family descriptor.** `RecordedEnvFamily` grows methods
supplying title, stride, and style hints. Rejected: it puts the descriptor on the
*type*, but the report reads a serialized record rather than a Rust type. The
information would have to be serialized into every frame to be usable — a table
with extra steps and worse locality.

**7. Adapter registration seam.** The client exposes a registry a third party adds
to. Rejected for Phase 3: in a self-contained HTML report with a prebuilt WASM
blob, "register" means relinking the client, which requires the client crate to be
publishable (it is `publish = false`) and is the BYOE-3 problem.

**8. Leave the metrics filter alone and open only the env axis.** Rejected. The
filter is a measured, reproducible defect that silently discards a submitter's
data, and decisions 3 and 4 both argue *"do what the metrics axis does"* — not an
argument worth making while the metrics axis does not do it.

## References

- ADR [0014](0014-record-schema-v6-single-agent-richness-and-provenance.md) — record schema ownership
- ADR [0015](0015-shared-typed-metric-registry-crate.md) — the registry this ADR amends
- ADR [0079](0079-harness-metrics-are-a-privileged-absorption-path.md) — the `agent/` prefix reservation, and the additive-defaulted-method polarity reused for `on_scene`
- ADR [0080](0080-harness-owns-the-environment-glue-and-the-umbrella-carries-the-harness.md) — `rlevo-benchmarks::fixtures`, where the `RecordedEnvFamily` impls now live
- ADR [0081](0081-the-record-format-is-a-leaf-crate.md) — sequenced before this; extracts `rlevo-scene`, where this ADR's types land
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — the `Bounds` newtype whose *approach* `Extent` reuses. `Bounds` itself stays in `rlevo-core` (amended; see ADR 0081 decision 2)
- Issues that **changed** this draft: **#944** (unenforced cross-field invariants — caught the positional `ScenePose` defect; its `Bone` companion is adopted), **#942** (a validated newtype for payload ranges), **#276** (`SnapshotMetadata`'s fate, queued behind a `FORMAT_VERSION` bump — deliberately left open, decision 8)
- Issues that **confirmed** it: **#671** (per-call `Vec<BodyRecord>` allocation — the static/per-frame split is its "hot caller"), **#846** / **#873** (why `Grid` must not migrate), **#304** (what deleting `Landscape2D` unblocks and what must be re-triaged)
- Issue **not** resolved by this ADR, contrary to first appearance: **#709** — see Consequences
- `xtask/byoe/README.md`, `xtask/byoa/README.md` — the two probes; both currently fail at step 8 of 9, for different reasons
- Spec: `docs/.private/specs/2026-08-21-byoe-first-class-citizen/` (B3, Phase 3, and the 3-D forward constraint)
