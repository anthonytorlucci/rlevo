---
project: rlevo
status: active
type: decision
date: 2026-08-10
tags: [adr, decision, environments, grids, observation, sensor, pomdp, minigrid, issue-1027]
---

# ADR 0073: The carried item is stamped into the agent's own view cell, after occlusion

## Status

**Accepted (2026-08-10).** Resolves issue #1027 (`AgentState::carrying`
never reached the observation).

**Implements work ADR [0063](0063-grid-visibility-occlusion.md) deliberately
deferred**, named there by number in its own "Negative / accepted costs"
section:

> **A known, separate defect is deliberately not fixed here.** Canonical
> Minigrid writes the agent's carried object into the agent's own view cell
> *after* occlusion, so a carried item is never itself masked
> (`gen_obs_grid`: slice → rotate → occlude → then stamp the carried object).
> rlevo shows the world cell the agent stands on instead, and
> `AgentState.carrying` never reaches the observation at all. This is
> tracked separately as **#1027** (not #281) and should land before any grid
> baseline is ever recorded...

ADR 0063 is `status: active`, superseded by nothing, and this ADR does
**not** supersede it — 0063's own text already named this exact fix and
deferred it for review-diff-size reasons, not because it was wrong. Nothing
in 0063's Decisions 1-6 changes; this ADR closes the one item its
Consequences left open.

## Context

### The defect was a real POMDP break, not a cosmetic omission

`AgentState.carrying: Option<Entity>` (`grids/core/agent.rs`) tracked what
the agent held, but no emission path ever wrote it into a `GridObservation`
or `GoToDoorObservation`. The agent's own view cell instead reported
whatever world entity it was standing on — before this fix, `egocentric_view`
read the grid unconditionally at every cell, including the agent's, and
neither `process_vis` nor the `SeeThrough` identity pass touched what that
cell *contained*, only whether it was masked.

This is not merely a missing feature. Two call sites gate transition
dynamics and episode success on `carrying` directly:

- `dynamics.rs`'s `toggle`, on a locked door:
  `if agent.carrying == Some(Entity::Key(color)) { ... }` — unlocking is
  conditioned on hand state.
- `unlock_pickup.rs`'s `has_target`:
  `self.state.agent.carrying == Some(self.layout.target)` — episode success
  is `carrying == target`.

`door_key`, `unlock`, and `unlock_pickup` therefore had reward and
transition dynamics that depended on a piece of state the policy's own
observation never carried in any form. A policy could not condition on
"do I have the key" or "do I have the box" from its observation — it had to
infer hand state from the action history alone, which is a strictly harder
and, for a purely reactive policy, an impossible inference. This is exactly
the shape of POMDP defect `docs/rules.md` §3's `TensorConvertible` clause 2
("no fabrication") and ADR 0061 exist to rule out at the tensor boundary —
except this was a stronger failure than fabrication: the field never
reached the tensor boundary to be fabricated *or* faithfully encoded at all.

### Canonical formulation

`MiniGridEnv.gen_obs_grid` (`minigrid/minigrid_env.py`, `master`, lines
597-632) runs, in order: slice the window → `rotate_left(agent_dir + 1)` →
`process_vis` (masks the window) → **then**, transcribed into Rust:

```rust
let agent_pos = (grid.width() / 2, grid.height() - 1);
match self.carrying {
    Some(item) => grid.set(agent_pos, Some(item)),
    None => grid.set(agent_pos, None),
}
```

`Grid.encode` (`minigrid/core/grid.py`, lines 260-263) turns canonical's
`None` (no object in this cell) into `[OBJECT_TO_IDX["empty"], 0, 0] ==
[1, 0, 0]`. `process_vis` seeds `mask[agent_pos] = True` unconditionally,
before any cell is read (`grid.py`, line 294) — so the stamp, running after
masking, is on a cell that is always visible under either occlusion policy.
A carried item is therefore unmaskable by construction, in both canonical
and this port.

## Decision

### 1. The stamp site is `mask_view`, not `egocentric_view`, `from_masked_view`, or `process_vis`

`grid::stamp_carried` (`crates/rlevo-environments/src/grids/core/grid.rs`,
near line 161) is called from `mask_view`
(`crates/rlevo-environments/src/grids/core/mod.rs`), immediately after the
`match visibility` that dispatches `Occluded`/`SeeThrough`:

```rust
pub(super) fn mask_view(
    grid: &Grid,
    agent: &AgentState,
    visibility: Visibility,
) -> [[Option<Entity>; VIEW_SIZE]; VIEW_SIZE] {
    let view = egocentric_view(grid, agent);
    let masked = match visibility {
        Visibility::Occluded => grid::process_vis(view),
        Visibility::SeeThrough => view.map(|row| row.map(Some)),
    };
    stamp_carried(masked, agent)
}
```

`mask_view` is the single chokepoint ADR 0063 Decision 1 built: eleven
environments reach it via the shared `observe_grid` free function, and
`GoToDoorEnv` — the one environment on the wider 4-channel encoder — calls
`mask_view` directly (`grids/go_to_door.rs`, around line 822) rather than
going through `observe_grid`. Placing the stamp inside `mask_view` covers
all twelve environments for free, with no per-env call site to add or
forget.

Stamping after the `match` also makes the behaviour policy-independent: the
`Occluded` arm's `process_vis` output and the `SeeThrough` arm's all-`Some`
identity map both pass through the same `stamp_carried` call, so there is
exactly one place that writes the hand, regardless of which `Visibility`
constant an environment declares.

Three alternative stamp sites were rejected, each for a reason specific to
what the function is for (§Alternatives considered).

### 2. The empty hand is `Some(Entity::Empty)`, never `None`

`stamp_carried` writes `view[AGENT_VIEW_ROW][AGENT_VIEW_COL] =
Some(agent.carrying.unwrap_or(Entity::Empty))` — never `None`, even when
`agent.carrying` is itself `None`.

Canonical's `None` in this branch means *no object in this cell*, and
`Grid.encode` turns it into `[1, 0, 0]`, the same encoding a genuinely empty
tile anywhere else on the board gets. rlevo's `Option<Entity>` in a masked
view means something different: `None` is *masked/unseen*, and
`GridObservation::from_masked_view` encodes it as `[UNSEEN_TYPE, 0, 0] ==
[0, 0, 0]` (ADR 0063 Decisions 3 and 4). Writing `None` here would be the
faithful-looking but wrong transcription of canonical's own `None` — it
would tell the policy that its own cell, the one cell it always sees under
either `Visibility` policy, is unobserved. This is recorded as the single
likeliest future mis-"simplification" of this function, and the in-source
rustdoc on `stamp_carried` and its own unit test
(`stamp_carried_writes_empty_for_an_empty_hand`) both assert `Some(Entity::Empty)`
and separately assert `!= None`, so a regression here fails loudly rather
than by drift.

### 3. Terminal frames change observed bytes even with an empty hand

`Entity::Goal` and `Entity::Lava` are both `is_passable()`
(`grids/core/entity.rs`), and `dynamics.rs`'s `step_forward` moves the agent
onto the front cell before matching on what was there
(`agent.x = fx; agent.y = fy; match front { Entity::Goal => ...,
Entity::Lava => ..., ... }`). On the step that reaches either, the agent's
own cell *is* `Goal` or `Lava` at the moment `mask_view` runs — and
`stamp_carried` unconditionally overwrites that cell with the hand (or
`Entity::Empty`), so the terminal-frame observation's own-cell byte changes
from `[8, 0, 0]` (`Goal`) or `[9, 0, 0]` (`Lava`) to `[1, 0, 0]` (`Empty`,
empty-handed) on every affected environment. This is a real behaviour
change even when nothing is being carried, and it matches canonical: the
agent's own cell is never a channel the policy reads the world through, by
design (§Context).

Verified against the current source, the affected environments are:

- **Goal-reaching**: `Empty`, `Crossing`, `DistShift`, `DoorKey`,
  `DynamicObstacles`, `FourRooms`, `LavaGap`, `MultiRoom`.
- **Lava** (as a `Lava`-kind hazard): `Crossing` (its `Lava` obstacle kind),
  `DistShift`, `LavaGap`.

`Entity::Floor` never appears on a real board — it is used only in
`all_floor_like`-style test probes (`grids/unlock.rs`, `grids/lava_gap.rs`,
`grids/crossing.rs`) rather than in any environment's generated layout — so
it is a no-op for this decision in practice. Recorded here so a future
reader does not re-derive the same conclusion from scratch.

### 4. Four rejected alternatives

- **A `carrying` field on `GridObservation`, outside the tensor.** Rejected:
  `docs/rules.md` §3's `TensorConvertible` clause 2 requires every field
  `write_host_row` does not write to decode to absence, and ADR 0061 closed
  exactly this shape of gap for `agent_direction` on the principle that
  anything a policy must read has to live in the tensor a `Strategy` or
  replay buffer actually round-trips. An out-of-tensor field would decode to
  `None`/a default on replay, hiding from the policy the one fact it needs —
  the inverse of what this ADR fixes.
- **A fourth encoder channel.** Rejected on ADR 0043's own ground
  (lines 206-208): widening the shared `GridObservation` from 3 to 4
  channels changes shape from `147` to `196` elements across all twelve
  environments for the benefit of encoding one already-typed fact that fits
  in the existing channel-0 byte at the agent's own cell.
- **Stamping inside `from_masked_view`.** Rejected: `from_masked_view` must
  stay a pure encoder from `[[Option<Entity>; VIEW_SIZE]; VIEW_SIZE]` to the
  tensor row shape — its own tests call it directly with hand-constructed
  views (e.g. `Entity::Goal` placed at `view[6][3]`) and assert the encoding
  of exactly the cells they set. Reaching into `AgentState` from inside the
  encoder would require threading the agent through a function whose
  contract is "encode this array," not "read the world."
- **Stamping inside `process_vis`.** Rejected: `process_vis` runs only under
  `Visibility::Occluded` — `mask_view`'s `SeeThrough` arm skips it entirely
  — so stamping there would leave every `SeeThrough` environment unfixed.
  `grid.rs`'s own `process_vis_agent_cell_is_always_visible` test asserts on
  the agent's cell staying `Some(..)` post-mask, a narrower guarantee than
  "carries the hand," and conflating the two would overload what that
  function is responsible for proving.

## Consequences

### Positive

- **`door_key`, `unlock`, and `unlock_pickup` stop being POMDP-broken.** A
  policy can now condition its action on whether it is holding the key or
  the target box, the same fact `dynamics.rs`'s `toggle` and
  `unlock_pickup.rs`'s `has_target` already gated reward and transition on.
- **One chokepoint, twelve environments.** The fix is a call inside
  `mask_view`; no per-environment `Sensor::observe` implementation changed.
- **Landed before any grid baseline exists.** ADR 0063 said this should land
  before any grid observation baseline is ever recorded. Re-verified for
  this ADR: `crates/rlevo/tests/baselines/` holds exactly one file,
  `dqn_cartpole.csv`, and nothing under `rlevo-benchmarks` instantiates a
  grid environment. There is still zero recorded grid baseline to invalidate.

### Negative / accepted costs

- **Observed bytes change for eleven environments on every terminal frame**,
  independent of whether anything is carried (§Decision 3's list). Any
  hand-inspected fixture, doctest, or external comparison pinned against a
  pre-fix terminal observation is now stale.
- **No shape change, and ADR 0061's clauses are untouched.**
  `GridObservation` stays `[7, 7, 3]`, `GoToDoorObservation` stays
  `[7, 7, 4]`, `GridState::shape()` is unchanged, no struct gains a field,
  and the serde wire format is unaffected. This is deliberately listed as a
  cost-that-isn't: the stamp writes a real observed fact into an existing
  byte the encoder already owned, rather than decoding an absence into a
  plausible value, so ADR 0061's no-fabrication clause was never at risk
  here — recorded so a reader does not go looking for a fabrication
  argument that does not apply to this change.

### Neutral

- **ADR 0063 is not superseded.** Its Decisions 1-6 stand exactly as
  written; this ADR discharges the one item its own Consequences section
  named and deferred by issue number.

## Alternatives considered

See §Decision 4 for the four rejected implementation sites; there was no
live disagreement about *whether* to fix this, only *where* the stamp
belongs and what the empty-handed encoding should be.

## References

- Issue #1027 — `AgentState.carrying` never reaches the observation.
- ADR [0063](0063-grid-visibility-occlusion.md) — names #1027 explicitly in
  its "Negative / accepted costs" section and defers it; this ADR implements
  that deferred item and supersedes nothing in 0063.
- ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) —
  the `TensorConvertible` no-fabrication clause and the "anything a policy
  must read goes in the tensor" precedent this ADR's rejected
  out-of-tensor-field alternative is measured against.
- ADR [0043](0043-grid-observation-contract.md), lines 206-208 — rejects
  widening `GridObservation` to a fourth channel, the same ground this ADR's
  rejected fourth-channel alternative rests on.
- `docs/rules.md` §3 — the `TensorConvertible` two-clause invariant table.
- Code: `crates/rlevo-environments/src/grids/core/grid.rs` (`stamp_carried`,
  `AGENT_VIEW_ROW`, `AGENT_VIEW_COL`), `crates/rlevo-environments/src/grids/core/mod.rs`
  (`mask_view`, `observe_grid`), `crates/rlevo-environments/src/grids/go_to_door.rs`
  (`mask_view` called directly), `crates/rlevo-environments/src/grids/core/dynamics.rs`
  (`toggle`'s door-unlock gate on `agent.carrying`, `step_forward`'s
  passable-then-match ordering that leaves `Goal`/`Lava` under the agent),
  `crates/rlevo-environments/src/grids/unlock_pickup.rs` (`has_target`'s
  success gate on `agent.carrying`), `crates/rlevo-environments/src/grids/core/entity.rs`
  (`Entity::is_passable`, `Entity::type_u8`).
- Reference implementation (authoritative for this ADR):
  Farama-Foundation/Minigrid, `master` —
  [`minigrid/minigrid_env.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/minigrid_env.py)
  (`gen_obs_grid`, lines 597-632),
  [`minigrid/core/grid.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/core/grid.py)
  (`process_vis`'s `mask[agent_pos] = True` seed at line 294, `Grid.encode`
  at lines 260-263).
