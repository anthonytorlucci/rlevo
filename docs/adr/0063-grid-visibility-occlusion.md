---
project: rlevo
status: active
type: decision
date: 2026-07-26
tags: [adr, decision, environments, grids, observation, sensor, occlusion, minigrid, pomdp, issue-281]
---

# ADR 0063: Grid occlusion is env-side `Sensor`; unseen is `Option<Entity>`, not a tile

## Status

**Accepted (2026-07-26).** Resolves issue #281 (the grid family's
`see_through_walls` gap).

**Supersedes [ADR 0043](0043-grid-observation-contract.md)'s own
Decision 3** ("The occlusion gap is recorded as a stated non-decision").
That decision named its own reopening condition — "that change is
ADR-worthy in its own right and should supersede this section, not edit
it" — and this is that ADR. ADR 0043's other decisions (Invariant M's
derivation, the mission-by-channel precedent for `GoToDoorEnv`) are
untouched. Decision 3's blast-radius argument for deferring occlusion
("it invalidates every existing grid benchmark baseline") and the matching
Alternatives-considered entry are corrected, not merely superseded: the
claim was false when written (Context, "What #281 and ADR 0043's Decision 3
got wrong").

**Amends [ADR 0047](0047-sensor-relocates-emission-model-to-environment.md)
Decision 5** (the grid family's exemption from `Sensor`, justified there as
"the emission model is shared across eleven envs and is state-pure"). The
premise fails once visibility is per-environment, so the exemption is
withdrawn; the grid family stops being the one env family in the workspace
that does not implement `Sensor`. ADR 0047's other decisions — the trait
shape, `Observable`'s demotion, `pixel_grid` as the `Sensor`-delegates-to-
`Observable` reference — are unaffected; `Observable<3>` remains in
`rlevo-core` and remains implemented by `PixelGridState`.

**Breaking** (alpha): `Entity::type_u8`'s byte mapping changes for all 12 grid
envs (Decision 4); `GridObservation`/`GoToDoorObservation`'s channel-0
encoding changes to match; `egocentric_view` becomes `pub(crate)`
(Decision 5); `impl Observable<3> for GridState` no longer exists (Decision 1,
already removed on this branch — see Context).

## Context

### The gap ADR 0043 recorded and deferred

ADR 0043's own Decision 3 stated the fact plainly: rlevo's `egocentric_view` read every
cell of the rotated `7×7` window unconditionally — `see_through_walls` was
effectively `true` for all 12 environments, opposite of canonical Minigrid's
own default. It deferred a fix because "the blast radius is all 12 grid
envs' observation semantics at once" and because a correct fix "needs a
per-env `see_through_walls` knob." It also, deliberately, phrased Invariant M
(`MemoryEnv`'s `size >= 11` floor) to survive an eventual occlusion change
untouched: "occlusion merely makes the invariant satisfiable at smaller
sizes; it does not change what the invariant states." That framing turned out
to matter for the opposite of the reason it was written: Decision 6 below is
the case where occlusion changed *nothing* about what is satisfiable, and
ADR 0043's own wording is exactly what lets that outcome be recorded as a
successful test rather than a contradiction.

### Canonical formulation, reconciled against current source

The full reconciliation, fetched live from `Farama-Foundation/Minigrid`
`master` on 2026-07-26, lives in
`docs/.private/research/2026-07-26-issue-281-occlusion-canonical.md`
(gitignored; this ADR reproduces its load-bearing findings and citations
rather than deferring to it, following the ADR 0062 precedent for private
research notes). The mechanism:

- `Grid.process_vis` (`minigrid/core/grid.py`) is a forward **flood fill**,
  not a ray cast, seeded at the agent's own cell: two horizontal sub-passes
  per row, propagating visibility sideways before pushing into the next row
  back. An occluder is itself visible but propagates nothing — the agent sees
  the wall that blocks it, never the wall's own cell as unseen. This
  distinction is load-bearing, not academic: a *single* isolated wall cell
  casts essentially no shadow, because light routes diagonally around it from
  the neighbouring columns in the same row (pinned in rlevo's port by
  `process_vis_lone_wall_cell_casts_no_shadow`,
  `crates/rlevo-environments/src/grids/core/grid.rs:500`). Durable shadows
  need an occluder that also blocks the sideways spread — a wall *run*, or the
  view-window edge — not a lone cell. An earlier reading of this algorithm as
  a ray cast (treating a single wall as sufficient to hide whatever is behind
  it) is wrong and is what Decision 6 corrects.
- `see_behind()` (`minigrid/core/world_object.py`) is `True` by default;
  `Wall` overrides it `False`, `Door` returns `self.is_open`. Every other
  object type — `Goal`, `Floor`, `Lava`, `Key`, `Ball`, `Box` — is
  transparent. An agent sees straight past lava, and past a key or ball lying
  on the floor; only walls and shut doors occlude.
- `gen_obs_grid` (`minigrid/minigrid_env.py`) invokes `process_vis` at the
  view-local, post-rotation agent position `(agent_view_size // 2,
  agent_view_size - 1)` — column-centre of the bottom row at the default
  `agent_view_size = 7` — only when `see_through_walls` is `False`; otherwise
  it uses an all-visible mask.
- `Grid.encode` (`minigrid/core/grid.py`, `minigrid/core/constants.py`)
  writes `[OBJECT_TO_IDX["unseen"], 0, 0]` for a masked cell, and
  `OBJECT_TO_IDX` reserves `0` for `"unseen"` with `"empty"` at `1` — the two
  are deliberately distinct indices, not a synonym pair.
- `MiniGridEnv.__init__` declares `see_through_walls: bool = False` —
  occlusion is **on by default**; an environment opts out, not in.

rlevo's existing geometry matches the parts that are free to reuse: the
agent sits at `view[VIEW_SIZE - 1][VIEW_SIZE / 2]` looking toward row `0`
(`AGENT_VIEW_ROW`/`AGENT_VIEW_COL`, `grid.rs:108,111`), which is the same
frame as canonical's post-rotation `(col 3, row 6)` at `VIEW_SIZE = 7`.
`rotate_view_offset` (`grid.rs:152-163`) needs no change; occlusion ports as
a post-pass over the array `egocentric_view` already produces.

### What #281 and ADR 0043's Decision 3 got wrong

Three claims used to justify deferral, or carried forward from it, do not
hold under inspection or measurement.

- **"Invalidates every existing grid benchmark baseline" is false — there are
  no grid baselines.** `crates/rlevo/tests/baselines/` holds exactly one
  file, `dqn_cartpole.csv`; nothing under `crates/rlevo-benchmarks`
  instantiates a grid environment (its handful of `grid`/`Grid` hits are the
  wire-format `GridTile`/`GridSnapshot` payload types used generically by the
  record/report tiers, not a grid env under evaluation). The change this ADR
  makes is materially cheaper than ADR 0043 recorded.
- **"Relax `MemoryEnv::MIN_SIZE` from 11 to 7" was recorded as a near-certainty
  ("a one-line change"); it is not merely a hypothesis, it is refuted.**
  ADR 0043's Decision 3 and issue #281 both implicitly assumed the
  mechanism is a wall standing between the hallway and the cue's row
  acting as a ray-cast occluder. It is not: `process_vis` is the flood
  fill above, and light
  reaches the cue by routing *around* the corridor walls through the open
  start room. The executed sweep (Decision 6) confirms `MIN_SIZE` does not
  move.
- **Whether upstream `MiniGrid-MemoryS7-v0`/`S9-v0` genuinely hide the cue is
  unknown, not confirmed.** rlevo's `process_vis` is a faithful transcription
  of the canonical Python source, verified line-by-line against
  `minigrid/core/grid.py`, but no live Python Minigrid run has been executed
  to check this ADR's port against ground truth. If canonical really does
  hide the cue at S7/S9 where this port does not, the divergence is in the
  port or in rlevo's `MemoryEnv` layout, not necessarily in the algorithm —
  and Decision 6's sweep is exactly where that divergence would surface if
  someone runs the comparison later.

## Decision

### 1. Visibility is an emission-model policy: the grid family adopts env-side `Sensor`

ADR 0047's own Decision 5 kept the grid family off `Sensor` on the stated
grounds that "the emission model is shared across eleven envs and is
state-pure." Canonical Minigrid sets `see_through_walls` **per
environment** — of the twelve rlevo implements, eight are occluded and
four are see-through (Decision 2's table) — so the premise that one
state-pure projection covers the family is false, and the exemption
ADR 0047's Decision 5 carved out lapses on its own terms, not because
this ADR overrules it.

`impl Observable<3> for GridState` is removed. `GridState::project` would
need to read a value that does not live on the state — an env's visibility
policy — so `Observable`'s `&self`-only signature is structurally the wrong
seam, exactly the category error ADR 0047 diagnosed for world-derived
sensors. `Observable` itself is untouched everywhere else in the workspace:
it stays in `rlevo-core` and stays the reference projection for
`PixelGridState` (ADR 0047's own Decision 4, "`Observable<OR>` is demoted,
not deleted"). `crates/rlevo-environments/src/grids/core/state.rs`
now documents the absence directly rather than leaving a reader to wonder why
`GridState` — alone among the family's data types — has no `Observable` impl.

Each grid environment implements `Sensor<OR, AR, SR>` for its own struct and
declares its policy as an inherent `const VISIBILITY: Visibility` — the same
shape `GoToDoorEnv` already carried
(`crates/rlevo-environments/src/grids/go_to_door.rs:759`,
`impl Sensor<3, 1, 3> for GoToDoorEnv`) since ADR 0043, now landed for the
other eleven. `grep -rn "const VISIBILITY"` over `grids/` is the audit
surface Decision 2's table is checked against; each site's doc comment names
the canonical file the value was read from
(`crates/rlevo-environments/src/grids/core/mod.rs:33-37`). Every `Sensor`
impl's `observe`/`observe_reset` body is a one-line forward to the shared
free function `observe_grid(state, Self::VISIBILITY)`
(`crates/rlevo-environments/src/grids/core/mod.rs:159-162`), which in turn
calls the crate-private `mask_view` (`mod.rs:174-186`) to cut the window and
dispatch the `Visibility` policy, and `grid::process_vis`
(`crates/rlevo-environments/src/grids/core/grid.rs:204-261`) to actually run
the shadow cast when the policy is `Occluded`. This is deliberately not
eleven duplicated occlusion algorithms — ADR 0047's Alternatives already
rejected "per-env `Sensor` for all eleven grid envs" on exactly that
duplication concern, and that reasoning still holds: what is per-env is a
one-line trait impl and a `Visibility` constant, not the shadow-cast logic
itself, which lives in exactly one place (`grid::process_vis`). `GoToDoorEnv`
needs the same masked view but a wider encoder (its fourth channel carries
the mission, Decision 3), so it calls `mask_view` directly and feeds
`GoToDoorObservation::from_masked_view` rather than `observe_grid` — keeping
the `Visibility` dispatch itself in one `match`, not two. `build_snapshot`
(`mod.rs::build_snapshot`) is repointed to take an already-produced
`GridObservation` rather than projecting one itself, so the `Visibility`
decision stays entirely with the caller.

Be honest about what this buys: `Sensor` has **zero generic consumers**
anywhere in the workspace today (no `<S: Sensor<..>>` bound, no `dyn Sensor`)
— it is a convention every other env family happens to follow, not a trait
object or a generic algorithm depends on. The value of closing the exemption
is narrower and real anyway: it removes the one documented, load-bearing
inconsistency ADR 0047's own Consequences section called out (its
"Negative / accepted costs" bullet: "one documented inconsistency: the
grid family builds snapshots via `Observable`... rather than a per-env
`Sensor`"), so a reader auditing "does every environment implement
`Sensor`" gets a uniform yes.

### 2. The per-env value is a compile-time associated const, not a config field

A `Deserialize`-able config is user-supplied runtime data (rules.md's Error
Handling section: "Never panic in response to user-supplied runtime data;
return `Err(...)` instead. A `Deserialize`-able config *is* user-supplied
runtime data."). A `see_through_walls` config field would let a deserialized
run manifest disable occlusion on `MemoryEnv` and silently re-break
Invariant M at small sizes — a POMDP-correctness invariant would depend on a
field a caller can flip. Each environment instead carries `Visibility` as a
compile-time constant: an inherent `const VISIBILITY: Visibility` on the env
struct, read by that environment's own `Sensor` impl (Decision 1), mirroring
the `Visibility` enum
(`crates/rlevo-environments/src/grids/core/mod.rs:124-137`). Const → config
field is purely additive later, if a genuine need for a runtime toggle
appears; the reverse is not, which is why the const is the conservative
choice now.

The accepted cost: rlevo cannot currently A/B occlusion as an experimental
ablation on a fixed environment, which canonical Minigrid permits since
`see_through_walls` is a constructor keyword argument there. Nothing in the
workspace exercises that ablation today, so this is a real but currently
unpaid cost.

The twelve values are canonical Minigrid fidelity, not a free choice per
environment:

| rlevo env | canonical file | value | source |
|---|---|---|---|
| `empty` | `empty.py` | `see_through_walls=True`, explicit | SeeThrough |
| `dist_shift` | `distshift.py` | `True`, explicit | SeeThrough |
| `dynamic_obstacles` | `dynamicobstacles.py` | `True`, explicit | SeeThrough |
| `go_to_door` | `gotodoor.py` | `True`, explicit | SeeThrough |
| `crossing` | `crossing.py` | `False`, explicit | Occluded |
| `lava_gap` | `lavagap.py` | `False`, explicit | Occluded |
| `memory` | `memory.py` | `False`, explicit | Occluded |
| `four_rooms` | `fourrooms.py` | omitted — `MiniGridEnv` default `False` | Occluded |
| `door_key` | `doorkey.py` | omitted — default `False` | Occluded |
| `multi_room` | `multiroom.py` | omitted — default `False` | Occluded |
| `unlock` | via `RoomGrid.__init__` | passes `see_through_walls=False` | Occluded |
| `unlock_pickup` | via `RoomGrid.__init__` | same | Occluded |

`MiniGridEnv.__init__`'s own default is `False` — occlusion is **on** by
default upstream, and an environment must opt out to be see-through. Eight
of twelve keep the default; four opt out. This is why the `Visibility` enum
(`mod.rs:105-108`) deliberately carries no `Default` impl: a default would be
wrong for a third of the family, so every environment states its own value
at its call site rather than inheriting one silently.

### 3. An unseen cell is `Option<Entity>::None`, never an `Entity` variant

Enforced by the type, not by prose. `render.rs::entity_to_tile`
(`crates/rlevo-environments/src/grids/core/render.rs:51-63`) is an exhaustive
match over `Entity` producing `GridTile`, the versioned wire type the WASM
report client (`rlevo-benchmarks-report-client`) mirrors for browser-side
rendering. Adding `Entity::Unseen` would propagate a value into that wire
format and into every downstream consumer's exhaustive match — including
`entity_char`/`glyph_for_entity` (`render.rs:193-234`) — that can never
legitimately appear in a *recorded* world grid: a completed episode's
recorded state is fully known, occlusion is a property of what the *agent*
observed at that instant, not of the world. `Entity::Unseen` would also force
`is_passable()`/`is_pickable()` (`entity.rs:66,78`) to answer a question that
has no meaningful answer for a cell nobody has looked at.

`Option<Entity>` is already the workspace's absence idiom for exactly this
shape of question — `AgentState::carrying: Option<Entity>`
(read at `render.rs:46`, `agent.carrying.map(entity_to_tile)`) and
`GridObservation::agent_direction: Option<Direction>` (ADR 0061) both encode
"this specific thing may not be known/present" the same way. Canonical
Minigrid encodes the identical distinction identically: `process_vis` does
`self.set(i, j, None)` for a masked cell, not a sentinel object.

rules.md's Trait Design Constraints section's `TensorConvertible`
invariant, clause 2, forbids the alternative directly: "any field
`write_host_row` does not write must decode
to a value *representing absence* (`None`, or a dedicated 'unknown'
variant), never to a plausible in-domain value." Collapsing an occluded cell
into `Entity::Empty` is exactly a plausible in-domain value — it asserts
"confirmed floor" where the truth is "unknown" — so it is forbidden by the
same clause ADR 0061 wrote to close the `agent_direction` fabrication, not
merely inadvisable by convention.

The mechanical consequence: `egocentric_view` itself keeps returning
`[[Entity; VIEW_SIZE]; VIEW_SIZE]` — it stays the raw, unmasked read
(Decision 5) — and `grid::process_vis` is the function that turns that into
`[[Option<Entity>; VIEW_SIZE]; VIEW_SIZE]`. Both `GridObservation` and
`GoToDoorObservation` gained a `from_masked_view` constructor
(`observation.rs:135`; `go_to_door.rs:269`) that accepts the optional form
directly; the pre-existing `from_entity_view` constructors are retained as
thin wrappers that map every cell to `Some` and delegate
(`go_to_door.rs:295-308`), so a caller that genuinely has a fully-visible
view (tests, `Visibility::SeeThrough`) is not forced to wrap it by hand. A
`None` cell encodes as the reserved unseen byte in channels 0-2
(`UNSEEN_TYPE`, Decision 4) rather than calling `Entity::type_u8` on a value
that was never observed.

**One deliberate exception: `GoToDoorObservation`'s mission channel is never
masked.** `from_masked_view` stamps `mission.to_u8()` into channel 3 of
*every* cell, seen or not (`go_to_door.rs:262-264`, verbatim: "the mission is
the agent's own goal, not something it perceives through the grid, so
occlusion must not hide it"). This matters mechanically, not just
philosophically: `mission_color_u8()` reads the mission byte back out of the
agent's own cell, `view[0][0][MISSION_CHANNEL]`
(`go_to_door.rs:314-315`), and the agent's own cell is always visible
(`process_vis` seeds the mask there) — but masking the mission channel
generally, rather than carving it out, would make that accessor silently
return `0` the moment any occlusion geometry reached cell `(0, 0)`. Canonical
Minigrid has no analogue to check this against — it has no mission channel
inside the image at all (ADR 0043) — so there is no upstream answer here;
this is an rlevo-only decision, recorded because a future contributor
"fixing" `from_masked_view` to mask channel 3 like the other three would
silently break `mission_color_u8` for every occluded environment. It is
currently inert: `GoToDoorEnv::VISIBILITY` is `SeeThrough` (Decision 2's
table), so no cell of its observation is ever masked in practice today.

### 4. The observation type-byte table adopts canonical Minigrid indices

`Entity::type_u8` (`entity.rs:89-101`) currently numbers `Empty=0, Wall=1,
Floor=2, Goal=3, Lava=4, Door=5, Key=6, Ball=7, Box=8` — a numbering that
happens to look canonical-adjacent but is not. It becomes `Empty=1, Wall=2,
Floor=3, Door=4, Key=5, Ball=6, Box=7, Goal=8, Lava=9`, matching
`OBJECT_TO_IDX` in `minigrid/core/constants.py` exactly for every variant
`Entity` has, with `0` reserved for unseen (Decision 3). This is a breaking
change to channel 0 of every grid observation across all 12 environments.

Two reasons, not one:

- **Parity.** ADR 0043 already treats Minigrid as the authoritative
  reference implementation for these environments. A numbering that visually
  resembles canonical's but silently disagrees with it (rlevo's current
  table) is worse than either full parity or an numbering that makes no
  claim to match — it invites a reader to assume equivalence that is not
  there.
- **Zero should mean unknown.** Under canonical numbering, an all-zero
  observation tensor decodes as "every cell unseen" — the correct reading
  for a zero-padded or attention-masked sequence. Under rlevo's current
  numbering, the same all-zero tensor decodes as "every cell confirmed empty
  floor," a materially different and false claim. These twelve environments
  exist specifically to train recurrent POMDP policies, and zero-padding a
  variable-length sequence to a fixed window is precisely the workload where
  this distinction is load-bearing, not cosmetic.

There are zero stored grid observation baselines anywhere in the workspace
(Context, "what #281 and ADR 0043's Decision 3 got wrong") to invalidate,
which is why this renumbering is cheap now — no recorded episode, checkpoint, or
regression fixture encodes the old byte values — and would be expensive
after the first one is recorded.

**This is not parity polish; it is a live correctness defect, measured.**
Because `UNSEEN_TYPE == 0 == Entity::Empty.type_u8()` under rlevo's current
numbering, a masked empty cell and a seen empty cell encode identically —
the two-index distinction canonical numbering exists to draw (Context) is
partially collapsed *today*, not just at risk in principle. Measured on
`MemoryEnv`'s default board: of 9,053 masked cells produced across its
occlusion sweep, 2,928 are `Entity::Empty`, and every one of those loses its
occlusion signal on the wire. The worst case is not incidental: facing West
from the fork junction — the pose at which the agent must actually answer —
the shadow cast masks 20 cells, and the resulting encoded observation is
**byte-identical** to the unoccluded one. Pinned directly in-source
(`crates/rlevo-environments/src/grids/memory.rs:1326`, doc: *"The
`UNSEEN_TYPE` / `Entity::Empty` byte collision, pinned at the pose where it
costs the most"*), with the test's own final assertion currently an
`assert_eq!` that this ADR's renumbering flips to `assert_ne!`. So the
renumbering is not optional finish work on top of Decision 5 — until it
lands, occlusion is unobservable on the wire at exactly the pose it matters
most, which is also the concrete case rules.md's Trait Design Constraints
section's no-fabrication clause was written to forbid: encoding an
occluded cell as `Empty` asserts "confirmed floor" about a cell the agent
cannot see.

**This renumbering is deliberately deferred out of the behavioural change,
not abandoned.** It did not land in the same set of commits as Decisions 1,
2, 3, and 5, so that a byte-encoding break and a behavioural (masking) change
are two reviewable diffs instead of one
(`crates/rlevo-environments/src/grids/core/observation.rs:61-65`, the
`UNSEEN_TYPE` doc's own words: *"deferred out of the visibility change so
that a behaviour change and an encoding break do not land in one diff"*). It
remains tracked under issue #281 and — per the measurement above — **must
land before any grid observation baseline is ever recorded**, which Decision
4 states as a requirement rather than a suggestion.

### 5. `egocentric_view` stays policy-free; occlusion is a post-pass; the function becomes crate-private

Adding a visibility parameter to `egocentric_view` itself would weld a
sensor policy onto what is otherwise a pure geometric window extractor, and
would make the raw, unoccluded window unobtainable for a caller that
legitimately wants it (a debug renderer, a test asserting the underlying
geometry, or an environment that has genuinely opted into
`Visibility::SeeThrough`). Occlusion is instead a post-pass:
`egocentric_view` (`grid.rs:130-147`) keeps returning the raw
`[[Entity; VIEW_SIZE]; VIEW_SIZE]` window unconditionally, and the
crate-private `grid::process_vis` (`grid.rs:204-261`) — a direct port of
`Grid.process_vis` — turns that into `[[Option<Entity>; VIEW_SIZE];
VIEW_SIZE]` by flood-filling from the agent's cell outward: two horizontal
sub-passes per row, seeded at the agent, masking behind any cell whose
`see_behind()` analogue is false (walls, and closed doors). `mask_view`
(`mod.rs:174-186`) is the one place that dispatches on `Visibility` — the
`SeeThrough` arm wraps every cell in `Some` and skips `process_vis` entirely,
the `Occluded` arm calls it — so there is exactly one `match Visibility` in
the family, not one per consumer.

Because rlevo's view is already in the same frame as canonical's
post-rotation window (Context), the port needed no change to
`rotate_view_offset` or to `egocentric_view`'s indexing — it is purely an
additional pass over the array `egocentric_view` already produces. One index
transposition was load-bearing during the port and is documented at the
call site: canonical indexes its mask `mask[i, j]` with `i` the column and
`j` the row, while rlevo's view array is `view[row][col]`, so every index
pair in `process_vis` is deliberately swapped relative to the Python source
(`grid.rs:190-197`).

`egocentric_view` is `pub(crate)`, not `pub`. A raw, unoccluded view is
semantically wrong for eight of the twelve environments once Decision 2's
table takes effect, and exposing it as public API would invite an external
caller to build an observation that silently disagrees with the
environment's own emission model. It is not re-exported from
`grids::core` at all (`mod.rs:60-64` documents the omission directly);
`observe_grid` is the crate-*and*-public entry point, so the visibility
policy cannot be bypassed even from within the crate without going through
`mask_view` explicitly, which `GoToDoorEnv` is the one place that does
(Decision 3).

### 6. `MemoryEnv`'s `MIN_SIZE` relaxation is refuted; `MIN_SIZE` stays 11

ADR 0043's own Decision 3 called the 11 → 7 relaxation "a one-line
change." **It is wrong, and this is no longer a hypothesis — it has been
tested and it failed.** The sweep ADR 0043's Decision 3 deferred to a
future occlusion ADR has been executed: every decision-region cell × all
four facings, at every size the
question is meaningful for, comparing `Visibility::Occluded` against
`Visibility::SeeThrough` on the identical board:

| size | occluded violations | see-through violations | verdict |
|---|---|---|---|
| 7 | 5 | 5, same poses | fails Invariant M |
| 9 | 5 | 5, same poses | fails Invariant M |
| 11 | 0 | 0 | holds |
| 13 | 0 | 0 | holds |

At every size tested, occlusion and see-through produce **the identical
violation set** — not merely the same count, the same `(cell, facing)`
pairs. At size 9 the failing poses include `(7, 3)` and `(7, 5)`, the two
decision cells immediately adjacent to the fork objects, so this is not an
artifact of a generously-drawn decision region catching cells that do not
matter; the violations sit exactly where the agent is about to answer.
Occlusion changes **nothing** about whether Invariant M holds at any size
rlevo tests. The reason is the mechanism correction in Context: `process_vis`
is a flood fill, and light reaches the cue at `(1, mid - 1)` by routing
around the corridor's walls, out through the open mouth at `x = 4`, and
sideways across the start room, which has no wall separating it from the
cue's row. A single wall standing between the hallway and the cue casts no
shadow (the lone-wall finding, Context) — there is no wall configuration in
this layout that would.

`MIN_SIZE` therefore **stays 11**, and its derivation is unchanged: it is
still a pure **distance** bound (`(size - 2) - VIEW_REACH > 1`), because the
shadow cast never enters the arithmetic — occlusion buys this environment
nothing at any size the sweep covers. The compile-time assertion tying
`MIN_SIZE` to `VIEW_REACH`,

```rust
const _: () = assert!(
    MIN_SIZE > VIEW_REACH + 3,
    "MIN_SIZE no longer hides the cue from the fork decision cell (Invariant M)"
);
```

(`memory.rs:261` `MIN_SIZE`, `:277` `VIEW_REACH`, `:291-294` the assertion —
`VIEW_REACH = VIEW_SIZE - 1 = 6`, so the assertion reads `MIN_SIZE > 9`, and
the smallest odd size satisfying it is `MIN_SIZE`'s actual value of `11`)
needs **no revision**: it was already the right derivation and the sweep
confirms it, rather than obsoleting it the way Decision 6 originally
anticipated it might. Canonical `MiniGrid-MemoryS7-v0`/`S9-v0` remain
unreproducible in rlevo.

This is pinned mechanically, not just narrated:
`test_memory_env_occlusion_does_not_relax_min_size`
(`crates/rlevo-environments/src/grids/memory.rs:1268`) asserts, for sizes 7
and 9, that the occluded violation set is non-empty *and* identical to the
see-through violation set — so the test fails loudly (not silently passes)
if a future change to sight computation ever does close the gap, which is
the correct trigger to reopen this section rather than edit it. Its own doc
comment states the reopening condition verbatim: *"If this test fails... that
is the good outcome, not a regression: reopen ADR 0063 Decision 6, lower
`MIN_SIZE`, and replace the `MIN_SIZE > VIEW_REACH + 3` compile-time
assertion with one derived from the occluding geometry."*

**Say plainly what this means for ADR 0043:** its "a one-line change"
prediction was wrong, tested against measurement rather than argument, and
refuted. ADR 0043 Decision 1 — Invariant M's derivation and the `size >= 11`
floor — stands **exactly as originally written**, which is the outcome its
own "phrased to survive a future occlusion change" wording anticipated as
one of the two possible results, not a guarantee of the other.

One honest caveat, carried from Context: this refutation is a statement
about **rlevo's port**, verified against the canonical Python source
line-by-line but not against a live run of it. If upstream
`MiniGrid-MemoryS7-v0` genuinely does hide its cue where this port does not,
the gap is in the port or in rlevo's `MemoryEnv` layout — and this same
sweep, re-run after fixing whichever it turns out to be, is where that would
surface.

## Consequences

### Positive

- **The grid family stops being the one env family in the workspace without
  a `Sensor` impl.** Every environment — grids included — now owns its
  emission model on the `Environment<..>`, not the `State<..>`.
- **The occlusion gap ADR 0043's own Decision 3 recorded is closed**, not
  merely re-recorded: eight of twelve environments genuinely hide information
  behind walls and shut doors for the first time, matching what their
  canonical Minigrid counterparts guarantee.
- **The record is corrected at the same time it is closed.** ADR 0043's
  Decision 3's "invalidates every benchmark baseline" claim is retired
  with evidence (Context), not repeated into a second ADR.
- **The byte encoding stops looking canonical while silently disagreeing
  with it** (Decision 4), and does so while the cost of changing it is
  lowest it will ever be — zero recorded baselines exist to break.
- **Invariant M's derivation survives untouched** (ADR 0043 Decision 1's own
  design goal). Decision 6's sweep found occlusion changes what is
  satisfiable for `MemoryEnv` not at all — a stronger and more surprising
  form of "survives" than ADR 0043 anticipated, and one only measurement
  could have established.

### Negative / accepted costs

- **`grid_memory_rl`'s DQN numbers move and are not comparable
  across this change.** `crates/rlevo/benches/grid_empty_rl.rs` and
  `grid_memory_rl.rs` compute their baselines live rather than asserting
  against a stored fixture, so nothing fails — but a before/after comparison
  of either bench's reported numbers is comparing two different tasks, not
  two runs of the same one.
- **Eight of twelve environments switch from see-through to occluded, but the
  effect on the room-based ones is weaker than layout alone suggests, and is
  now stated per-pose and measured rather than asserted from the floor
  plan.** `FourRooms` was measured directly: its four cross openings let the
  flood fill spread sideways into a neighbouring room, so from *most* agent
  poses the goal in an unentered room stays visible exactly as before —
  across 12 seeds, the goal was maskable from any pose in only 2 of them.
  `test_four_rooms_occlusion_hides_the_goal_from_an_adjacent_room`
  (`crates/rlevo-environments/src/grids/four_rooms.rs:1339`) is deliberately
  a one-pose regression test, not a general "walls hide the room" property,
  because that general property does not hold — its own doc comment says so
  directly: *"Note how weak the occlusion turns out to be... this test
  therefore names one concrete pose rather than asserting a general 'walls
  hide the goal' property that does not hold."* `MultiRoom`, `DoorKey`, and
  `UnlockPickup` were not swept the same way for this ADR, but they share the
  same doorway-as-flood-fill-conduit geometry `FourRooms` does, so the same
  tempering effect should be expected there rather than the naive "occlusion
  ⇒ proportionally harder" reading. Any policy or hyperparameter tuned
  against the prior see-through behaviour is still tuned against a somewhat
  different problem — the direction of the change is real — but the
  magnitude is measured smaller than the layout alone predicts.
- **A known, separate defect is deliberately not fixed here.** Canonical
  Minigrid writes the agent's carried object into the agent's own view cell
  *after* occlusion, so a carried item is never itself masked
  (`gen_obs_grid`: slice → rotate → occlude → then stamp the carried object).
  rlevo shows the world cell the agent stands on instead, and
  `AgentState.carrying` never reaches the observation at all. This is
  tracked separately as **#1027** (not #281) and should land before any grid
  baseline is ever recorded — the encoding-break window this ADR opens
  (Decision 4) is free to extend now and costly once a baseline exists.
  Confirmed a real POMDP defect rather than a cosmetic one: `dynamics.rs:126`
  gates door-unlocking on `agent.carrying`, and `unlock_pickup.rs:569` gates
  success on it, so the agent must infer its own hand state from history.
- **Rlevo cannot currently A/B occlusion as a runtime ablation** (Decision
  2's accepted cost) — the const-generic choice trades that capability for
  ruling out a config-driven regression of Invariant M.
- **Breaking observation encoding, alpha-scoped.** `Entity::type_u8`'s byte
  values change for every grid environment (Decision 4); any external code
  comparing raw channel-0 bytes against the old table must be updated.

### Neutral

- **`Sensor`'s trait shape, `Observable`'s demotion, and `pixel_grid`'s
  `Sensor`-delegates-to-`Observable` reference are unchanged** — this ADR
  amends only ADR 0047's own Decision 5's grid-specific exemption.
- **Rank and snapshot type stay put.** `GridSnapshot = SnapshotBase<3,
  GridObservation, ScalarReward>` and `GoToDoorSnapshot`'s `[7,7,4]` shape
  are unaffected; only the meaning of the bytes each channel carries
  changes.
- **`MemoryEnv`'s `MIN_SIZE` stays 11** (Decision 6) — settled by measurement,
  not left open.
- **`build_snapshot` survives Decision 1's `Observable` removal essentially
  unchanged, and it is worth saying why it survives at all.** With the
  observation now passed in rather than projected, the function's remaining
  body — `if done { SnapshotBase::terminated(..) } else {
  SnapshotBase::running(..) }` — looks like it could be inlined at every call
  site as boilerplate. It is kept as a named chokepoint because that
  `done -> EpisodeStatus` mapping is itself wrong, identically, in all twelve
  environments: every env's `step` computes `done = self.steps >=
  self.config.max_steps` and folds that into the same boolean that also
  signals a genuine terminal transition (goal reached, lava stepped on),
  so a step-limit cutoff is reported as `EpisodeStatus::Terminated` rather
  than `EpisodeStatus::Truncated` — the distinction `EpisodeStatus` exists to
  let an RL algorithm draw (`rlevo-core/src/environment.rs:18-20`, doc:
  *"Separating `Terminated` from `Truncated` allows RL algorithms to
  correctly bootstrap the value function: a truncated episode still has
  future value, whereas a terminated one does not"*). This ADR does not fix
  that defect — it is
  unrelated to visibility — but having every environment's mapping pass
  through one function is exactly what will make the eventual fix a one-site
  change instead of a twelve-site one. Filed as **#1028**, separately from
  #281. Confirmed grid-family-specific rather than workspace-wide: the
  time-limit wrapper, the locomotion and box2d families, and `pixel_grid` all
  emit `Truncated` correctly; `grep -rn "EpisodeStatus::Truncated" grids/`
  returns zero hits across all twelve envs.

## Alternatives considered

- **A `see_through_walls` field on `GridState`.** Rejected: re-commits the
  exact category error ADR 0047 diagnosed for world-derived sensors —
  visibility is a property of the observing environment's policy, not of a
  point in state space, and a `&self`-only projection cannot read
  environment-level config regardless of where the field sits.
- **A field on `Grid`.** Rejected for the same reason, one layer down:
  `Grid` is the world map, and visibility is even less a property of the map
  than it is of the agent's own state.
- **A mixed design: keep `Observable` for eleven envs, `Sensor` only for
  `go_to_door`.** Rejected: two mechanisms for one concept, and they would
  disagree about where visibility lives for two environments making the
  *same* choice — `empty` and `go_to_door` are both `SeeThrough` (Decision
  2's table), yet one would express that fact on the state and the other on
  the environment.
- **An `Entity::Unseen` variant.** Rejected per Decision 3: it propagates a
  value into the versioned `GridTile` wire type and the WASM report client
  that can never appear in a recorded world grid, and gives
  `is_passable()`/`is_pickable()` a question with no answer.
- **Keep rlevo's current type-byte numbering, append unseen at index 9.**
  Rejected per Decision 4: it preserves a table that looks canonical-adjacent
  while disagreeing with it on every other index, and an all-zero tensor
  would still decode as "confirmed empty" rather than "all unseen" — the
  opposite of what a zero-padded POMDP sequence needs.
- **Assert the `MemoryEnv` `MIN_SIZE` relaxation on the strength of ADR
  0043's prediction, without running the sweep.** This is what an earlier
  draft of this ADR did (stating the relaxation conditionally rather than
  asserting it outright), reserving judgment until the sweep ran. The sweep
  has since run and refuted the prediction (Decision 6); had this ADR instead
  asserted the relaxation outright, it would have repeated exactly the
  unverified-claim mistake in ADR 0043's Decision 3 that Context corrects,
  only with a wrong answer instead of an unstated one. This is recorded
  because it is the
  clearest demonstration in this ADR of why a decision record should state a
  measured result rather than a plausible one.

## References

- Issue #281 — the grid family's `see_through_walls` gap.
- ADR [0043](0043-grid-observation-contract.md) — superseded at Decision 3 only;
  Invariant M's derivation and the mission-by-channel precedent stand.
- ADR [0047](0047-sensor-relocates-emission-model-to-environment.md) —
  amended at Decision 5 only; the `Sensor` trait shape, `Observable`'s
  demotion, and the `pixel_grid` reference are unchanged.
- ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) —
  the `Option<Direction>`/no-fabrication precedent this ADR's Decision 3
  extends to `Option<Entity>`, and the `TensorConvertible` clause-2 text
  quoted from rules.md's Trait Design Constraints section.
- ADR [0062](0062-grid-layout-fidelity-and-no-dead-rng.md) — the sibling
  grid-family fidelity ADR this one follows in structure and in its use of a
  gitignored private research note as provenance.
- `docs/rules.md`'s Trait Design Constraints section (`TensorConvertible`
  two-clause invariant) and its Error Handling section (`Deserialize`-able
  config is user-supplied runtime data — the basis for Decision 2's
  const-not-config choice).
- Code: `crates/rlevo-environments/src/grids/core/grid.rs:130-147`
  (`egocentric_view`, `pub(crate)`), `:152-163` (`rotate_view_offset`),
  `:204-261` (`process_vis`, the `Grid.process_vis` port), `:500`
  (`process_vis_lone_wall_cell_casts_no_shadow`),
  `crates/rlevo-environments/src/grids/core/entity.rs:89-101`
  (`Entity::type_u8`), `crates/rlevo-environments/src/grids/core/render.rs:51-63`
  (`entity_to_tile`, the exhaustive match onto the `GridTile` wire type),
  `crates/rlevo-environments/src/grids/core/observation.rs:36-72`
  (`UNSEEN_TYPE` and its documented collision with `Entity::Empty`), `:135`
  (`GridObservation::from_masked_view`),
  `crates/rlevo-environments/src/grids/core/state.rs` (`GridState`, the
  documented absence of `Observable`),
  `crates/rlevo-environments/src/grids/core/mod.rs:159-162`
  (`observe_grid`), `:174-186` (`mask_view`, the single `Visibility` dispatch
  point), `:199-205` (`build_snapshot`),
  `crates/rlevo-environments/src/grids/go_to_door.rs:262-264`
  (`from_masked_view`'s unmasked mission channel), `:269-293`
  (`from_masked_view`), `:314-315` (`mission_color_u8`), `:759`
  (`impl Sensor<3, 1, 3> for GoToDoorEnv`, the reference `Sensor` impl),
  `crates/rlevo-environments/src/grids/memory.rs:1-160` (module docs
  reconciling the leak zone, Invariant M, and the flood-fill mechanism),
  `:261` (`MIN_SIZE`), `:277` (`VIEW_REACH`), `:291-294` (the compile-time
  Invariant-M assertion), `:1268`
  (`test_memory_env_occlusion_does_not_relax_min_size`), `:1326`
  (the `UNSEEN_TYPE`/`Entity::Empty` collision test),
  `crates/rlevo-environments/src/grids/four_rooms.rs:1339`
  (`test_four_rooms_occlusion_hides_the_goal_from_an_adjacent_room`),
  `crates/rlevo-core/src/environment.rs:16-29` (`Sensor` trait definition and
  `EpisodeStatus`, including the `Terminated`/`Truncated` doc quoted in
  Consequences), `crates/rlevo-core/src/state.rs` (`Observable`, unaffected).
- Reference implementation (authoritative for this ADR): Farama-Foundation/Minigrid,
  `master`, fetched 2026-07-26 —
  [`minigrid/core/grid.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/core/grid.py)
  (`process_vis`, `encode`),
  [`minigrid/minigrid_env.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/minigrid_env.py)
  (`gen_obs_grid`, `get_view_exts`, the `see_through_walls=False` default),
  [`minigrid/core/world_object.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/core/world_object.py)
  (`see_behind`),
  [`minigrid/core/constants.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/core/constants.py)
  (`OBJECT_TO_IDX`),
  [`minigrid/envs/memory.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/envs/memory.py),
  [`minigrid/envs/gotodoor.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/envs/gotodoor.py).
- Chevalier-Boisvert et al., "Minigrid & Miniworld: Modular & Customizable RL
  Environments for Goal-Oriented Tasks," NeurIPS 2023 D&B,
  [arXiv:2306.13831](https://arxiv.org/abs/2306.13831) — general
  POMDP-benchmark-suite framing only.
- Morad et al., "POPGym: Benchmarking Partially Observable Reinforcement
  Learning," [arXiv:2303.01859](https://arxiv.org/abs/2303.01859) — treats
  occlusion as a source of partial observability distinct from limited
  sensor range, motivating why occlusion (not merely view size) is the
  mechanism that forces genuine temporal integration.
- "Benchmarking Partial Observability in RL with a Suite of
  Memory-Improvable Domains,"
  [arXiv:2508.00046](https://arxiv.org/abs/2508.00046) — corroborates that
  occlusion-induced (rather than range-induced) partial observability is
  what forces memory reliance; reached by web search rather than Consensus,
  cited as corroborating context only, per the research note's own caveat.
- Provenance: `docs/.private/research/2026-07-26-issue-281-occlusion-canonical.md`
  — the full canonical-vs-current reconciliation and citation trail. Gitignored,
  so this ADR reproduces its load-bearing findings (Context, Decision 2's
  table) directly rather than deferring to it.
