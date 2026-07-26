---
project: rlevo
status: active
type: decision
date: 2026-07-26
tags: [adr, decision, environments, grids, observation, tensor-convertible, minigrid, issue-286, issue-844]
---

# ADR 0061: Grid facing is `Option<Direction>`; `TensorConvertible` gains a no-fabrication clause

## Status

**Accepted (2026-07-26).** Resolves issue #286 (`GridObservation::agent_direction`
is dropped by `to_tensor` and **fabricated** by `from_tensor`) and issue #844
(`agent_direction` should be a typed `Direction`, not a raw byte). Extends ADR
[0052](0052-hostrow-supertrait-splits-layout-from-backend.md) — the contract
text this ADR amends lives on the `TensorConvertible` trait 0052 last touched
— and closes the loop opened by ADR
[0043](0043-grid-observation-contract.md), whose mission-by-channel precedent
this ADR's "anything a policy must read goes in the tensor" half restates as a
general rule. Supersedes nothing.

## Context

`GridObservation` (11 grid envs) and `GoToDoorObservation` (`GoToDoorEnv`,
ADR 0043) both carry `agent_direction: u8`. `write_host_row` serializes only
the `view` — `row_shape()` is `[7, 7, 3]` / `[7, 7, 4]`, with no slot for the
facing (`grids/core/observation.rs:87-95` before this change; the identical
shape at `grids/go_to_door.rs:224,338`). `from_tensor` on both types
hard-coded `agent_direction: Direction::North.to_u8()` on decode — a value
that looks like a real measurement but is not one. A test on each type,
`view_round_trips_through_tensor`, asserted this lossiness directly
(`assert_eq!(round_tripped.agent_direction, Direction::North.to_u8())`), so it
could never flag the defect it was sitting on top of.

This mattered against the trait's own text. `rlevo-core/src/base.rs`'s
`TensorConvertible` doc required *"Implementors must round-trip:
`from_tensor(x.to_tensor(device))` equals `Ok(x)` for any valid `x`. Strategies
and replay buffers rely on this invariant."* — and `docs/rules.md §10`
independently stated the same rule with a sharper edge:

> Tensor conversion round-trips (`to_tensor` → `from_tensor`) must be lossless
> for all valid instances. If lossless round-trip is impossible for a type, that
> type must not implement `TensorConvertible`.

Read literally, both impls were non-conforming, and §10 went further than mere
non-conformance: because a lossless round trip *is* impossible for these types
by design, the rule as written **forbade the `TensorConvertible` impl
outright**. The tree therefore contained two impls that its own rules said must
not exist. That is why this ADR has to amend the contract rather than only fix
the two call sites — leaving §10 untouched would have left the fixed impls
still formally illegal (see Decision §2 for the replacement wording).

**`GoToDoorObservation` has the identical defect and was unfiled.** A search
for it under #286's title returns nothing; #286 as filed named only
`GridObservation`. Any fix scoped to one type leaves the other half of the
grid family broken, and `GoToDoorObservation` is exactly the ADR-0043 sibling
type the mission-channel precedent was built on — so this ADR treats both as
one subject.

**The blast radius is narrower than the contract text implies.** Nothing in
production round-trips an observation through a tensor.
`ExperienceTuple<..>` stores `observation: O` **by value**
(`rlevo-reinforcement-learning/src/experience.rs:38`), and the replay modules
contain no `to_tensor`/`from_tensor` call at all. Every `from_tensor` call
site on a grid observation is a test or a doc example — including the six
off-policy agents' benches (`crates/rlevo/benches/grid_empty_rl.rs`,
`grid_memory_rl.rs`) that train `DqnAgent` directly on `GridObservation`: they
call `to_tensor`/`write_host_row` to stage the network input, never
`from_tensor` to decode one back. This drops the severity from "silently
corrupting training runs" to "a public field that lies, plus a trait impl
that does not conform to its own contract" — real, but not the replay-buffer
corruption scenario the contract's prose warned about.

**The view is egocentric *and already rotated* into the agent's frame.**
`egocentric_view` (`grids/core/grid.rs:108-124`) calls `rotate_view_offset`
keyed on the agent's current facing (`:129-142`, one match arm per
direction), so what a policy sees is already expressed relative to "forward".
This is the fact the whole decision turns on, and it is not an implementation
accident rlevo introduced — it reproduces canonical Minigrid's own
`gen_obs_grid`, which slices the local window and then calls
`grid.rotate_left()` exactly `agent_dir + 1` times
(`minigrid/minigrid_env.py`, Farama-Foundation/Minigrid `master`).

Canonical Minigrid's observation is a `spaces.Dict` with `"image"`,
`"direction"`, and `"mission"` as three **separate** entries
(`minigrid/minigrid_env.py`); `direction` is never folded into the image.
rlevo's structure already matches this — `view` is the image, `agent_direction`
is the dict's `direction` entry — so the defect was never the field's
existence, only what `from_tensor` invented for it. And canonical baselines
that consume the dict drop `direction` before it reaches a network:
`ImgObsWrapper` (`minigrid/wrappers.py`, docstring verbatim: *"Use the image as
the only observation output, no language/mission"*) discards it outright;
`rl-starter-files`' reference `ACModel` (`lcswillems/rl-starter-files`,
`model.py`) references only `obs.image` and `obs.text`, never `obs.direction`;
RIDE's and NovelD's shared `MinigridPolicyNet`
(`facebookresearch/impact-driven-exploration/src/models.py`; Raileanu &
Rocktäschel, "RIDE: Rewarding Impact-Driven Exploration for
Procedurally-Generated Environments," ICLR 2020, arXiv:2002.12292; Zhang et
al., "NovelD: A Simple yet Effective Exploration Criterion," NeurIPS 2021)
consumes only `partial_obs`, with no direction embedding anywhere in the
inverse/forward dynamics nets either. The platform paper itself — Chevalier-
Boisvert, Bahdanau, Lahlou, Willems, Saharia, Nguyen, Bengio, "BabyAI: A
Platform to Study the Sample Efficiency of Grounded Language Learning," ICLR
2019, arXiv:1810.08272v4 — describes its own baseline model's inputs, §4.1, as
"a 7x7x3 symbolic observation $x_t$ ... and a variable length instruction $c$"
with no third term, and reports 100% success on single-room levels (Table 2)
without one. Its Baby Language BNF (Figure 2, §3.2) expresses only
agent-relative locations — `on your left | on your right | in front of you |
behind you` — with no absolute compass term anywhere in the grammar, so an
absolute heading carries nothing the rotated image and the relative mission
do not already encode.

**The honest counterexample exists, and is why this decision is scoped.**
`FullyObsWrapper` (`minigrid/wrappers.py`) encodes the whole grid in absolute,
unrotated coordinates, and writes `env.agent_dir` into the agent's own cell —
`full_grid[pos] = [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]`
— because there, orientation has no other carrier. Direction is redundant
**precisely because** the image is rotated; turn rotation off and it becomes
the sole signal of facing. rlevo implements only the rotated branch today.

**`Direction::default()` is `East`, not `North`, and there was no decoder.**
`direction.rs:11-13` marks `East` `#[default]`; `to_u8` maps `East→0,
South→1, West→2, North→3` (`:64-71` after this change). There has never been
a `from_u8`. So the fabricated `North` was neither the type's `Default` nor
recoverable from the byte it invented — the North-out asymmetry an earlier
code review flagged was real, but "fix" options that make the fabricated
value match `Default` carry their own hazard (Alternatives considered).

## Decision

### 1. `agent_direction` becomes `Option<Direction>` on both grid observation types

`GridObservation::agent_direction` and `GoToDoorObservation::agent_direction`
are retyped from `u8` to `Option<Direction>`. `from_entity_view` (the
constructor every environment's `project()`/`observe` calls) always yields
`Some(direction)` — the real facing is known at construction time.
`from_tensor` yields `None` — the tensor never carried a facing, so decoding
one must say so rather than invent one. `Direction` itself is unchanged
(`East` `#[default]`, `to_u8` the canonical Minigrid byte order); no
`from_u8` is added, because nothing needs one — the only place a byte would
be decoded back into a `Direction` is exactly the fabrication path this ADR
removes.

### 2. `TensorConvertible`'s contract becomes two clauses, not one

The prior text — *"round-trip: `from_tensor(x.to_tensor(device))` equals
`Ok(x)` for any valid `x`"* — assumed every implementor's row covers every
field. It does not name what an implementor with a partial row is supposed
to do, which is exactly the gap `GridObservation`/`GoToDoorObservation` fell
into. `crates/rlevo-core/src/base.rs`'s `TensorConvertible` rustdoc, and the
matching row in `docs/rules.md`'s Core Trait Invariants table and §10, now
state two clauses, both required of every implementor, for any valid `x` and
device `d`:

1. **Tensor-image fidelity.** `from_tensor(x.to_tensor(d))?.to_tensor(d) ==
   x.to_tensor(d)` — decode-then-re-encode is a no-op *on the tensor*.
   Everything the tensor carries survives a decode unchanged.
2. **No fabrication.** Any field `write_host_row` does not write must decode
   to a value that *represents absence* — `Option::None`, or a dedicated
   "unknown" variant — never to a plausible in-domain value.

A type whose `write_host_row` covers every field gets the stronger identity
`from_tensor(x.to_tensor(d)) == Ok(x)` for free, and clause 2 is satisfied
vacuously — that is the expected case and what nearly every other
`TensorConvertible` impl in the workspace already does, unmodified by this
ADR. Clause 1 is a **floor**, not a licence to be partial: it is what stops a
partial encoding from also being a dishonest one. `GridObservation` and
`GoToDoorObservation` now satisfy clause 1 in its weaker form (the `view`
half round-trips totally) and clause 2 by construction (the omitted facing
decodes to `None`, never a `Direction` value).

### 3. Scope: the rotated branch only — this is the reopening condition

This decision is scoped to environments whose observation is rotated into
the agent's frame, which is every grid env in the workspace today. The
`FullyObsWrapper` counterexample above is structural, not incidental: a
future unrotated or fully-observable grid variant — one that does not call
`rotate_view_offset` — would have no other carrier for orientation and would
need the direction **encoded inside the tensor**, the same way canonical
Minigrid's `FullyObsWrapper` stamps `agent_dir` into the agent's own cell.
Should that variant ever land, it reopens this ADR's Decision §1 for that
variant specifically; it does not invalidate it for the rotated envs this
ADR covers.

## Consequences

### Positive

- **The fabrication is gone, and the type system now names the honest
  state.** `agent_direction: Option<Direction>` cannot silently disagree with
  reality the way `u8` could: `Some(dir)` is a real measurement,
  `None` is an explicit "unknown," and there is no third value a caller could
  mistake for either.
- **Also resolves #844.** `agent_direction: Direction` rather than a raw byte
  makes an illegal encoding (any value outside `0..=3`) unrepresentable;
  #844 can be closed by this work.
- **The `TensorConvertible` contract now states a rule that generalizes**,
  rather than one this workspace's own impls already violated. Clause 2 is a
  standing check for any future partial `HostRow`/`TensorConvertible` impl,
  not a one-off patch to two types.
- **Closes the loop with ADR 0043 into one rule.** Anything a policy must
  read goes in the tensor — 0043's mission channel, still the precedent for
  goal-conditioned grid envs. Anything else must not be fabricated on
  decode — this ADR. `GoToDoorObservation`'s mission byte lives *inside* the
  tensor at `MISSION_CHANNEL` and round-trips totally under clause 1's
  stronger form; 0043 is vindicated by this ADR, not amended by it.
- **`Direction::to_u8` stays public despite being callerless in-tree.** It
  encodes the canonical Minigrid byte order (`East=0, South=1, West=2,
  North=3`) for callers that need the wire byte — logging, interop, or a
  bespoke encoding of their own — independent of whether anything inside the
  crate currently calls it.

### Negative / accepted costs

- **Breaking public API on a struct 12 environments share.** Every caller
  comparing `obs.agent_direction` against a `u8` (via `Direction::to_u8()`
  or a literal) must migrate to comparing against `Some(Direction::_)` /
  `None`. See the `CHANGELOG.md` entry for the exact migration.
- **Clause 2 is not machine-checkable.** Nothing stops a future
  `HostRow`/`TensorConvertible` impl from writing a partial row and then
  fabricating a plausible value on decode anyway — the compiler cannot see
  "this looks like real data." This is review-enforced, the same class of
  convention as `docs/rules.md`'s "compare floats with `total_cmp`, never
  `partial_cmp`" rule: a real invariant with no total, mechanical guard. That
  is exactly why it is recorded in `rules.md`'s Core Trait Invariants table
  and §10, not left to live only in this ADR.
- **This decision is scoped, not general** (Decision §3). A future
  unrotated/fully-observable grid variant reopens Decision §1 for that
  variant; this is recorded so a future contributor does not read "direction
  stays out of the tensor" as a crate-wide law rather than a consequence of
  rotation.

### Neutral

- **No other `TensorConvertible` impl in the workspace changes.** The ~33
  other impls already write every field their type declares, so clause 1's
  stronger form and clause 2's vacuous case already held for them; this ADR
  amends prose and adds a compile-time-unenforced but now-explicit rule, not
  a signature change.
- **No rank or shape change.** `GridObservation` stays `[7, 7, 3]` /
  `Environment<3, 3, 1>`; `GoToDoorObservation` stays `[7, 7, 4]`. Neither
  `row_shape()` nor `HostRow`/`TensorConvertible`'s method signatures move.
- **No performance change.** `write_host_row` pushes the same bytes it
  always did; only the struct field type and the decode path's constructed
  value change.
- **No persisted-data format changes in-tree** — see the `CHANGELOG.md`
  entry for the externally-serialized wire-form caveat (`3` → `"North"` /
  `null`), which is real but affects no in-tree consumer.

## Alternatives Considered

- **Reset the fabricated value to `Direction::default()` instead of an
  explicit absence.** Rejected, and the sharpest reason: `Direction::default()`
  is `East` — also the most common real facing in these envs (agents start
  facing `East` in every env that doesn't otherwise sample a start facing) —
  so a round-trip test comparing the decoded facing against the *expected*
  facing would **pass by coincidence** whenever the agent happened to be
  facing `East`, exactly the kind of accidental-pass the fabricated `North`
  at least avoided by being conspicuously wrong. This option fixes the
  aesthetic North-in/East-out asymmetry a code review flagged while making
  the underlying defect *harder* to detect, which is strictly worse.
- **Encode the facing as a fourth channel plane.** Rejected on two
  independent grounds: it deviates from canonical, which never puts
  `direction` inside the image; and it spends 33% more input features on all
  11 `GridObservation` envs (`[7,7,3]` → `[7,7,4]`, 147 → 196 elements per
  observation) and 25% more on `GoToDoorObservation` (`[7,7,4]` → `[7,7,5]`)
  to deliver a signal every inspected published baseline discards before the
  network sees it.
- **Remove the field entirely.** Rejected: canonical Minigrid *does* ship
  `direction` in its observation dict, so removing it would diverge from
  Minigrid in the opposite direction from the 4-channel option; it would
  also delete a field `GridState::project()` deliberately threads through
  and that `MemoryEnv`'s cue-invariance assertion
  (`grids/memory.rs:1311`) reads directly to assert two episodes differ only
  in cue type, not in facing.
- **An associated const `ROUND_TRIP_IS_TOTAL: bool` on `TensorConvertible`.**
  Rejected: nothing in the workspace can branch on it. The six off-policy
  agents bind `O: TensorConvertible<..>` but only ever call
  `to_tensor`/`write_host_row` — no production code decodes an observation —
  so the const would be write-only metadata on a core trait: unenforceable
  (a wrongly-set `true` still compiles), and it declares *that* a type is
  partial without declaring *which field*. `Option<Direction>` discharges
  the same motivation at the one place it can actually be checked: the use
  site, where a caller must handle `None`.
- **Split the observation into a tensor-only payload type plus a
  facing-carrying wrapper.** Rejected on a hard constraint, not a style
  preference: `Environment::ObservationType` is what the off-policy agents
  bind, and their bounds require `TensorConvertible` on the observation type
  itself — `crates/rlevo/benches/grid_empty_rl.rs` and `grid_memory_rl.rs`
  train `DqnAgent` directly on `GridObservation`. Splitting either breaks
  every off-policy agent's generic bound, or keeps a delegating impl on the
  wrapper with the identical partiality this ADR fixes, plus one more type
  to maintain.
- **Make `from_tensor` return `Err` instead of `Ok` with `agent_direction:
  None`.** Rejected: it discards the half of the decode that *is* useful —
  view decoding is exercised by tests, debugging, and visualization — and
  would make issue #860's proposed proptest round-trip property unstatable,
  since there would be no successful decode to assert properties about.

## References

- Issue #286 — `GridObservation::agent_direction` dropped by `to_tensor`,
  fabricated by `from_tensor`.
- Issue #844 — `agent_direction` should be a typed `Direction`, not a raw
  byte; closed by Decision §1.
- Issue #860 — proptest round-trip over arbitrary valid views; the
  `Err`-on-decode alternative above would have made it unstatable.
- Issue #841 — `GridState::project` doc cross-reference to the direction
  round-trip; unaffected by this ADR but adjacent.
- ADR [0043](0043-grid-observation-contract.md) — mission-by-channel
  precedent (`GoToDoorObservation`'s `MISSION_CHANNEL`); this ADR's
  "anything a policy must read goes in the tensor" clause is 0043 restated
  as a general rule, and 0043 is the reason `GoToDoorObservation` exists at
  all.
- ADR [0052](0052-hostrow-supertrait-splits-layout-from-backend.md) —
  the `HostRow`/`TensorConvertible` split whose contract text this ADR
  amends; every 0052 decision (the supertrait split, the derived
  `to_tensor`, hand-written `from_tensor`) is unchanged by this ADR.
- Provenance and full citation trail:
  `docs/.private/research/2026-07-26-issue-286-grid-observation-direction.md`
  (code reconciliation against current source; Minigrid/BabyAI literature
  review, including the evidentiary-strength caveats it records honestly).
- Code: `crates/rlevo-environments/src/grids/core/observation.rs`
  (`GridObservation`), `crates/rlevo-environments/src/grids/go_to_door.rs`
  (`GoToDoorObservation`, `MISSION_CHANNEL`),
  `crates/rlevo-environments/src/direction.rs` (`Direction`, `to_u8`),
  `crates/rlevo-environments/src/grids/core/grid.rs:108-142`
  (`egocentric_view`, `rotate_view_offset`),
  `crates/rlevo-environments/src/grids/memory.rs:1311` (cue-invariance
  assertion reading `agent_direction`), `crates/rlevo-core/src/base.rs`
  (`HostRow`, `TensorConvertible`), `crates/rlevo-reinforcement-learning/src/experience.rs:38`
  (`ExperienceTuple` stores `observation: O` by value),
  `crates/rlevo/benches/{grid_empty_rl.rs,grid_memory_rl.rs}` (the only
  production-adjacent `TensorConvertible` consumers of `GridObservation`,
  both encode-only).
- Reference implementation: Farama-Foundation/Minigrid, `master` —
  [`minigrid/minigrid_env.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/minigrid_env.py)
  (`gen_obs_grid`, the observation `Dict`),
  [`minigrid/wrappers.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/wrappers.py)
  (`ImgObsWrapper`, `FullyObsWrapper`).
- Chevalier-Boisvert, Bahdanau, Lahlou, Willems, Saharia, Nguyen, Bengio.
  "BabyAI: A Platform to Study the Sample Efficiency of Grounded Language
  Learning." ICLR 2019. [arXiv:1810.08272v4](https://openreview.net/pdf?id=rJeXCo0cYX).
  Appendix B.4 (observation spec), §4.1 (model inputs, no direction term),
  Figure 2 (Baby Language BNF, agent-relative locations only), Table 2
  (100% single-room success without `direction`).
- Raileanu & Rocktäschel. "RIDE: Rewarding Impact-Driven Exploration for
  Procedurally-Generated Environments." ICLR 2020. arXiv:2002.12292.
  Code: `facebookresearch/impact-driven-exploration/src/models.py`
  (`MinigridPolicyNet`).
- Zhang et al. "NovelD: A Simple yet Effective Exploration Criterion."
  NeurIPS 2021 — builds on the same `MinigridPolicyNet`, direction-free.
- lcswillems/rl-starter-files, `model.py` (`ACModel`, references only
  `obs.image`/`obs.text`).
