---
project: rlevo
status: active
type: decision
date: 2026-07-26
tags: [adr, decision, rng, seeding, reset, environments, grids, minigrid, fidelity, issue-282, issue-108]
---

# ADR 0062: Grid layouts reproduce upstream `_gen_grid`; a dead RNG field is not a resting state

## Status

**Accepted (2026-07-26).** Resolves issue #282 (nine grid envs hold a dead
`_rng` and re-seed it inside `reset()`) and issue #108 (the same family's
`seed` field is inert while its docs imply it drives layout randomization).

**Partially supersedes ADR [0029](0029-host-rng-seeding-convention.md)** —
specifically the dead-`_rng` grid carve-out in its Decision §3 (lines 117-119)
and the matching "Neutral" consequence (line 161). **Every other ADR 0029
decision remains in force and ADR 0029 stays `active`**: the persistent-stream
`reset()` rule, `reset_with_seed` as the inherent replay hatch, the
sample-the-problem-once rule, the bandit-family treatment, the host-sampling
ban on `B::seed` + `Tensor::random`, and the `R: Rng + ?Sized` trait-bound
idiom are all unchanged and this ADR depends on them.

This ADR does not reverse 0029; it removes an exception that contradicted
0029's own rule. See Context §1.

## Context

### 1. ADR 0029 states the rule and then carves these grids out of it

Two paragraphs of ADR 0029 §3 sit six lines apart and disagree. The first
states the general rule (line 114-115):

> Deterministic envs are allowed, but must still not re-seed on reset.

The second exempts the grid family from it (lines 117-119):

> Environments whose RNG is never drawn from (the deterministic `_rng` grid
> layouts — `empty`, `door_key`, `crossing`, `four_rooms`, …) are unaffected:
> reseeding an unused RNG is a no-op, not a diversity bug, and they are left
> untouched.

and the Consequences → Neutral section repeats it (line 161): "The `_rng`
deterministic grids are left alone (no behavioural effect either way)."

The carve-out's reasoning is sound *given its premise* — reseeding an RNG
nothing reads really is a no-op. The premise is what fails. It classifies nine
environments as "deterministic" on the evidence that their code does not sample,
without checking whether they *should*. Seven of the nine should. Under this
ADR the general rule at line 115 stands unqualified and the exception is gone,
so the two paragraphs no longer disagree.

### 2. The premise rests on a reading of the Minigrid paper that conflates two things

arXiv:2306.13831 §2.2 describes the Minigrid environments as deterministic —
no randomness in the **transition function**. That is a statement about `step`:
given a state and an action, the successor is fixed, with no slip probability
and no stochastic dynamics. rlevo's grid `step` path
(`grids/core/dynamics.rs::apply_action`) correctly reproduces this and is not
at issue.

It is not a statement about `reset`. The same paper's §2.4 and Figure 4 present
`_gen_grid` as *the* world-generation seam — the method a user overrides to
define a new task — and the shipped `_gen_grid` implementations sample. A
transition-deterministic MDP whose initial-state distribution is a procedural
generator is exactly the standard Minigrid setup, and it is the setup the whole
procedurally-generated-environment generalization literature is built on.

ADR 0029's "deterministic no-ops" read appears to rest on that conflation:
transition determinism was taken to license a fixed board. Saying so plainly
is the point of this section — the carve-out was not careless, it was a
category error, and naming it is what stops the next reviewer from making it
again.

### 3. What the nine envs actually do, reconciled against upstream

`rg 'self\._rng\.'` over `crates/rlevo-environments/src/grids/` returns zero
hits. All nine store an RNG, re-seed it in `reset()` via
`self._rng = StdRng::seed_from_u64(self.config.seed);`, and never read it.
Their `build(config)` functions take no RNG parameter at all, so they are
structurally incapable of sampling. Three siblings — `memory.rs`,
`go_to_door.rs` (both fixed under #109), and `dynamic_obstacles.rs` (correct
from the start) — already carry the intended seam and are the in-tree
reference.

Reconciled against Farama-Foundation/Minigrid `master` (`minigrid/envs/*.py`,
`minigrid/core/roomgrid.py`, `minigrid/minigrid_env.py`) as of 2026-07-26:

| env | upstream `_gen_grid` draws | rlevo today | verdict |
|-----|---------------------------|-------------|---------|
| `Empty` | **None.** `MiniGrid-Empty-5x5/6x6/8x8/16x16` fix the agent at `(1,1)` dir `0`; only the separately-registered `-Random-` ids call `place_agent`. | agent `(1,1,East)`, goal `(size-2,size-2)` | **Faithful.** Genuinely deterministic. |
| `DistShift` | **None** for the shipped `DistShift1/2-v0`. The lava row is a registration constant, not a draw — determinism is the experimental point of a train/test distributional-shift probe over two fixed, known boards. | variant-selected lava row, all else literal | **Faithful.** Genuinely deterministic. |
| `LavaGap` | wall column `_rand_int(2, w-2)`, gap row `_rand_int(1, h-1)` | both pinned to `size/2` | Deviation (small). |
| `Crossing` | `shuffle(rivers)` → take `num_crossings`; `shuffle(path)`; a `choice` per opening | exactly two strips at `mid±1` sharing one gap at `mid` | Deviation (small draws, ordering-critical — see §4). |
| `DoorKey` | split column `_rand_int(2, w-2)`, door row `_rand_int(1, h-2)`, agent pose via `place_agent`, key via `place_obj` | split `size/2`, door on the diagonal at `(split, split)`, key `(1,1)`, agent `(1,2,North)` | Deviation (small). |
| `FourRooms` | four inter-room doorway offsets; agent pose and goal via `place_agent`/`place_obj` (the 2×2 quadrant partition itself is fixed) | doorways pinned at `mid±3`; agent and goal fixed | Deviation (small). |
| `UnlockPickup` | door row, door colour, box colour, key and box positions in room, agent pose | all constants | Deviation (small). |
| `Unlock` | door y on the shared **interior** wall, `_rand_color()`, key colour tied to door colour, key pos via `place_in_room` with `reject_next_to`, agent pose | door at `(1, 0)` — **inside the top perimeter wall** — key `(2,1)`, single room | Deviation, **plus an independent placement defect**. |
| `MultiRoom` | room **count**; each room's size and position via recursive `_placeRoom` with backtracking; each door's wall side, position, and colour | uniform horizontal strip of equal-width rooms, every door at `height/2`, all one colour | Deviation, **and a different size of problem** — procedural generation, not a handful of draws. |

Two of nine are faithful. Seven deviate, undocumented and unjustified.

`unlock.rs:293-305` is a confirmed defect, not a suspicion: `build` calls
`grid.draw_walls()` and then `grid.set(1, 0, Entity::Door(DOOR_COLOR, Locked))`
— row `0` is the perimeter it just drew. Upstream `Unlock` is a two-room task
with the door on the interior wall. A door in the perimeter is not the Unlock
task.

### 4. The hazard is the combination, not either half

The re-seed line is harmless **only because** the RNG is dead. Add sampling
without deleting the line and every episode draws an identical layout while a
"we sample from the RNG now" test passes — the env is provably calling `rng`,
and the observations are provably identical every reset, and both facts are
consistent with a green test suite. This is the trap #109 nearly fell into,
and it is why this ADR pairs the fidelity rule with a structural rule that
makes the dangerous intermediate state unrepresentable.

### 5. There is no shared placement machinery

`grids/core/` offers `Grid::{new, in_bounds, get, set, draw_walls}`,
`AgentState::new`, and nothing else — no free-cell predicate, no uniform
position sampler, no random `Direction`. The only sampling code in the family
is `GoToDoorEnv::sample_door_colors` (`go_to_door.rs:682`), a hand-rolled
rejection loop. Randomizing six envs without landing a shared sampler first
means six hand-rolled rejection loops with six independent chances to get the
free-cell predicate wrong. `CHANGELOG.md:2004` already acknowledges this gap
in prose ("no-merge guarantee Farama Minigrid gets from its `place_obj`
rejection loop") without closing it.

## Decision

### 1. Grid layouts reproduce the draws their upstream `_gen_grid` makes

A grid environment in `rlevo-environments` reproduces the random draws its
published Minigrid counterpart makes in `_gen_grid`, for the registered
environment id it claims to implement. Faithfulness is to the *algorithm* — the
set of quantities drawn and the order they are drawn in — not to bit-identical
output, which is impossible across `numpy`'s and `rand`'s generators anyway.

Any deviation is **documented in the config's `seed` field doc**, naming the
upstream id it departs from and why. An undocumented deviation is a defect
under the "grounded in the literature" rule, independent of whether any test
catches it.

Two corollaries that are not obvious and cost real time when rediscovered:

- **Draw order is part of the algorithm.** Upstream `Unlock`/`UnlockPickup`
  call `add_object` with a `reject_next_to` filter that reads
  `self.agent_pos`, but `place_agent` runs *after* — so the filter evaluates
  against `RoomGrid`'s deterministic centre placement, not the final agent
  position. A port that reproduces the draws but not the order reproduces a
  different distribution.
- **The registered id must be named.** `MiniGrid-MultiRoom-N4-S5-v0` is
  misregistered upstream with `minNumRooms == maxNumRooms == 6`; `-v1`
  corrects it to 4. "Implements MultiRoom" is not a specification.

### 2. A dead RNG field is not an acceptable resting state

Every environment in `rlevo-environments` is exactly one of two things. There
is no third option.

**(a) It samples.** Then it owns the full seam ADR 0029 §1-2 defines and #109
established in `go_to_door.rs`:

- a persistent `rng` field (not `_rng`), seeded **once** in `with_config`;
- `build(config, rng: &mut R)` with `R: Rng + ?Sized` per `docs/rules.md` §8 —
  not `&mut StdRng`, which the three already-migrated envs concretize by
  accident and which must not propagate;
- `reset()` draws from the persistent stream and lets it advance; **no
  re-seed**;
- an inherent `reset_with_seed(&mut self, seed: u64)` as the explicit replay
  hatch;
- a `seed` doc stating that the stream advances across resets, so a fixed seed
  reproduces a fixed *sequence* of episodes rather than one repeated episode.
  `go_to_door.rs:395-401` is the wording of record; new envs copy it rather
  than paraphrase it.

**(b) It is honestly deterministic.** Then it has **no RNG field at all**, and
its `seed` doc says the environment makes no random draws, why that is correct,
and — where the config field is retained for surface uniformity across the
family — that the stored value therefore never affects behaviour. "Reserved for
future stochastic variants" is not an acceptable `seed` doc: it is a promise
with no owner, and it is precisely the wording that let this family sit
mis-documented through two reviews.

The `seed` config field itself is retained in both cases. It is
`Serialize`/`Deserialize` on `Copy` config structs with 20+ construction sites;
removing it is a persisted-data break bought for nothing.

This binary is what makes §4's guard enforceable with zero exceptions. An env
that samples has its `seed_from_u64` in a constructor or in `reset_with_seed`;
an env that does not sample has no `seed_from_u64` anywhere. A third "has an
RNG but does not use it yet" state would need an allowlist, and an allowlist is
where the next dead field hides.

### 3. Placement is a shared, fallible, materialized sampler

A new `grids/core/placement.rs` holds free functions — not an extension trait
on `Grid`, not a borrowing `Placer` struct — taking `(&mut Grid, &mut R, …)`
with `R: Rng + ?Sized`. `Grid` is a `Clone` data type on the
observation-extraction path and must not acquire an RNG; the six consumers
interleave placement with direct `grid.set` calls and reads of already-placed
cells, so a struct holding `&mut Grid` across the whole `build` body buys a
borrow-checker fight and amortizes nothing. Region and predicate vary per env
(a sub-rectangle for DoorKey's left room and UnlockPickup's right room; a
"reject if adjacent to the agent" filter for `Unlock`-family keys), so both are
parameters: a rectangle, and a rejection closure.

Two sub-decisions are recorded because they are architectural and will
otherwise be re-litigated at every call site.

**(a) Exhaustion returns `Result`, propagated with `?` — never `.expect()`.**
The construction chokepoint (ADR 0026) validates *config*; exhaustion depends
on config **and** the draws already made this episode. `DoorKey` at
`MIN_SIZE = 5` has a 3×3 interior: place the agent, then place the key under a
`reject_next_to` filter, and the residual free set can be empty on some draws
from a config that is entirely valid. That is an ordinary config, not a
pathological one, so "config validation makes this unreachable" is false.

The path is already paved and already documented for exactly this case.
`EnvironmentError` is `#[non_exhaustive]` and its
`Config(#[from] ConfigError)` variant's rustdoc reads: *"A `reset()` may re-run
construction-time work (e.g. rebuilding a procedural world), so a
config-domain invariant — a `ConfigError` — can surface at reset, not only at
construction."* `with_config` already returns `Result<_, ConfigError>` and
`reset` already returns `Result<_, EnvironmentError>`, so both call sites
propagate with `?` and no signature moves. `Result` at the sampler with
`.expect()` at the call sites is a panic with extra steps and is forbidden.

**(b) The sampler materializes the candidate set and draws a uniform index.**
It does **not** port upstream's unbounded rejection loop, and it takes no
`max_tries` parameter. `dynamic_obstacles.rs:389-402` already does the right
thing in-tree: collect the free cells into a `Vec`, `rng.random_range(0..len)`,
`swap_remove`. Same uniform distribution over free cells; exhaustion is exactly
`candidates.is_empty()` rather than a tries budget that conflates "unlucky"
with "impossible"; O(area) once rather than unbounded. Upstream's loop is
unbounded because it declines to materialize the free set — an implementation
detail of a different language's memory posture, not a semantic. On boards of
at most 19×19 the allocation is noise. This deviation is deliberate and is
documented at the sampler so a future contributor does not "restore fidelity"
by reintroducing an unbounded loop in `reset()`.

### 4. The rule is mechanically guarded, and the guard's limits are stated

A source-text guard test in `crates/rlevo-environments/tests/`, modelled on the
existing `crates/rlevo-environments/tests/landscape_dim_guards.rs` (which reads
`src/landscapes/` from disk at test time via `CARGO_MANIFEST_DIR` and fails
unless every module is classified), scans the **whole crate's** `src/` — not
just `grids/` — resolves each `seed_from_u64` occurrence to its enclosing `fn`,
and requires that function to be on an allowlist of constructors and explicit
replay hatches (`with_config`, `new`, `with_seed`, `reset_with_seed`, …). An
occurrence inside `reset` fails with a message citing this ADR, ADR 0029, and
`docs/rules.md` §8. Like the landscape guard, the check runs both ways: an
allowlisted function that no longer exists is a stale row and also fails.

Crate-wide scope is deliberate. `toy_text/`, `classic/`, `locomotion/`, and
`pixel_grid.rs` are under the same rule and were all touched by #104. A guard
scoped to `grids/` guarantees the next recurrence lands somewhere else.

The allowlist is not optional bookkeeping: `reset_with_seed` contains
`self.rng = StdRng::seed_from_u64(seed);` three lines above `self.reset()`, so
a naive grep flags the very method ADR 0029 mandates.

**Known limits, recorded rather than discovered later.** The guard reads source
text: it is brittle against reformatting, and it is defeatable by aliasing
(`use rand::SeedableRng as S; S::seed_from_u64(..)`). It catches the accident,
not the adversary. That is the correct threat model — the failure this ADR
exists to prevent is a contributor adding sampling to an env whose `reset()`
still re-seeds, which is a mistake, not an evasion. Deleting the guard later
costs nothing, so this is a reversible decision taken cheaply.

### 5. Deliberate, documented non-conformance is allowed; silent non-conformance is not

Two environments remain knowingly non-conformant with §1 when this ADR lands:

- **`MultiRoom`** (#1021) keeps its fixed equal-width strip. Upstream's
  recursive `_placeRoom` with backtracking is procedural generation, not a
  handful of draws, and bundling it would hold five tractable envs hostage.
- **`Unlock`** (#1020) keeps its perimeter-wall door pending the two-room
  topology change, which moves `MIN_SIZE`, the layout, and the solvability
  oracle together.

Both are tracked by issue and **documented in-file** under §2(b)'s wording rule
— naming the upstream id they depart from and pointing at the issue — so the
deviation is visible at the call site, not only in a tracker. Issue #282
therefore remains open on a 7/9 checklist rather than being closed against
work that was not done. Issue #108 closes: its acceptance ("wire the seed into
layout randomization per env, or correct the doc to state the layout is fixed")
is satisfied for all seven files it names.

## Consequences

### Positive

- **ADR 0029's own rule now applies without exception.** The line-115 rule and
  the line-117 carve-out no longer contradict each other, and "deterministic"
  now means "verified against upstream and documented," not "does not happen to
  call `rng`."
- **The dangerous intermediate state is unrepresentable.** Under §2 there is no
  env holding an unread RNG for someone to start sampling from while the
  re-seed line survives. The trap #282 names cannot be re-entered without
  deleting a guard test that names it.
- **Six hand-rolled rejection loops become one reviewed sampler** (§3), with
  one free-cell predicate and one exhaustion semantics to get right.
- **The `seed` field stops lying.** `crossing.rs:136-138`,
  `door_key.rs:103-105`, and `four_rooms.rs:129-131` currently say "Using the
  same seed always produces the same episode layout" on envs whose seed does
  nothing at all; `empty.rs:90-92`, `lava_gap.rs:85`, `multi_room.rs:102`,
  `unlock.rs:85`, and `unlock_pickup.rs:90-91` promise "reserved for future
  stochastic variants." Both classes are replaced by statements that are true.
- **Test coverage strengthens where it looks like it weakens.** The per-env
  `build_places_*` assertions being replaced pinned exact cells on a fixed
  board; they never tested whether a generated board is *solvable*, which is
  the actual failure mode of procedural generation. The seed-loop oracles that
  replace them do.

### Negative / accepted costs

- **Behavioural and semver-relevant, in the same class as ADR 0029's own
  `reset()` change.** Five environments (`LavaGap`, `Crossing`, `DoorKey`,
  `FourRooms`, `UnlockPickup`) produce a different layout per episode where
  they previously produced one fixed board. Any consumer relying on a fixed
  layout must move to `reset_with_seed`. Acceptable in alpha; recorded in
  `CHANGELOG.md`.
- **Scripted rollouts against those five stop being expressible as fixed action
  lists.** The existing seed-loop oracles work because their randomness is a
  small discrete choice (one of four walls; one of two fork arms), which a
  script table can index. A randomized `DoorKey` draws split column, door row,
  agent pose, and key position — no script table indexes that. Those oracles
  need a planner over the shared dynamics, which is new test machinery this ADR
  obliges but does not itself provide.
- **`Empty` and `DistShift` keep a `seed` config field that provably does
  nothing.** Retained for config-surface uniformity and to avoid a
  persisted-data break; the cost is a field a reader must be told is inert,
  which §2(b) requires the doc to do explicitly.
- **The guard test is source-text, not semantic** (§4) — brittle to
  reformatting, defeatable by aliasing.
- **`MultiRoom` and `Unlock` ship non-conformant with the rule this ADR sets**
  (§5). That is a real, named gap, not an oversight, and #282 stays open
  because of it.
- **`four_rooms.rs:90-97`'s `const _: ()` assertion**, which ties `MIN_SIZE` to
  the fixed `±3` doorway offsets and is deliberately written to break the build
  if `MIN_SIZE` drops, loses its stated justification once the offsets are
  sampled. It must be *replaced* with an assertion that the sampling range is
  non-empty at `MIN_SIZE`, not deleted.

### Neutral

- **No new dependency, and no proptest.** `proptest` is declared in
  `[workspace.dependencies]` but consumed only by `rlevo-evolution` (ADR 0036);
  `rlevo-environments`' dev-dependencies are `approx` and `bincode`. The
  invariants here quantify over *seeds*, not an input space, so the existing
  `for seed in 0..ORACLE_SEEDS` shape in
  `crates/rlevo-environments/tests/grids_solvable.rs` is the natural form and
  a second PRNG would be the thing ADR 0036 warns against.
- **No trait, rank, or observation-shape change.** `Environment<3, 3, 1>`,
  `GridObservation`, `GridSnapshot`, and the ADR 0043 contract are untouched;
  `reset_with_seed` remains an inherent method per ADR 0029's own rejection of
  putting it on the trait.
- **`dynamic_obstacles.rs`, `memory.rs`, and `go_to_door.rs` are unchanged in
  behaviour.** They already implement §2(a); they are the reference, and the
  only edit they may attract is the `&mut StdRng` → `R: Rng + ?Sized` bound
  correction, which is source-compatible at every call site.
- **The existing per-env `reset_is_deterministic` tests survive unmodified.**
  They construct two envs from the same config and compare one `reset()` each,
  which still holds under the persistent-stream model — and becomes a
  meaningful assertion rather than a tautology.

## Alternatives considered

- **Delete the dead `_rng` and the re-seed line, and stop there.** The
  minimal reading of #282, and it satisfies the letter of two of its four
  acceptance bullets. Rejected: it leaves seven envs silently deviating from
  the algorithm they claim to implement, leaves the `seed` docs saying things
  that are not true, and — because #108 would still be open against the same
  files — guarantees the family is reopened a third time. Hygiene without the
  fidelity rule is what produced this state.
- **Randomize all seven deviating envs in one change.** Rejected on risk
  shape, not on principle. `MultiRoom` is recursive room generation with
  backtracking and a connectivity guarantee; `Unlock` is a topology change that
  moves `MIN_SIZE` and the oracle. Bundling either with five envs that need a
  handful of `_rand_int` draws holds the tractable work hostage to the hard
  work and makes the resulting change unreviewable. Split, tracked, and
  documented in-file (§5).
- **Amend ADR 0029 in place.** Forbidden by the repository's own convention
  (`docs/adr/README.md`: "Once accepted, an ADR is not edited — a later
  decision supersedes it"), and wrong on the merits: 0029's carve-out was a
  reasoned position on the evidence available, and the record of *why* it was
  reasonable is what stops the same reasoning recurring. Partial supersession
  follows the ADR 0033 → 0004 precedent.
- **Supersede ADR 0029 wholesale.** Rejected: every other 0029 decision is
  correct, in force, and load-bearing for this one. A wholesale supersession
  would put the persistent-stream rule, the `reset_with_seed` hatch, and the
  `B::seed` ban into a superseded document, which is exactly how a live
  convention gets read as historical.
- **Make placement exhaustion a panic, justified by the config chokepoint.**
  Rejected: exhaustion is a function of config *and* prior draws, so a valid
  config can exhaust (§3(a)); `EnvironmentError`'s `Config` variant already
  documents the reset-time procedural-rebuild case; and both call sites already
  return `Result`, making the error path free.
- **Port upstream's unbounded `place_obj` rejection loop verbatim, for
  fidelity.** Rejected: it is fidelity to a memory-posture detail rather than
  to a distribution, and its failure mode is a non-terminating `reset()` —
  strictly worse than a returned error. A bounded `max_tries` variant was also
  rejected, as it conflates "unlucky draws" with "no free cell" and requires a
  magic number no one can justify. Materializing the candidate set (§3(b))
  gives the same distribution with neither problem.
- **Enforce the convention by review alone, as ADR 0061's clause 2 does.**
  Rejected here, and the difference is that this rule *is* mechanically
  checkable at useful fidelity. 0061's "no fabrication" clause requires knowing
  whether a decoded value looks like real data, which a compiler and a grep
  both cannot see; "`seed_from_u64` does not appear inside `fn reset`" is a
  syntactic property, and `landscape_dim_guards.rs` already establishes that
  this crate accepts source-text guards for exactly this shape of rule.
- **Scope the guard test to `grids/`.** Rejected: #104 was crate-wide, and a
  grids-only guard guarantees the recurrence lands in `locomotion/` or
  `toy_text/` instead.

## References

- Issues: **#282** (dead `_rng` in nine grid envs + the ADR-0029-violating
  re-seed; remains open on a 7/9 checklist per §5), **#108** (dead/misleading
  `seed`/`_rng`, non-gameplay-breaking cases; closed by this work), **#109**
  (grid gameplay bugs — established the seam this ADR generalizes), **#104**
  (RNG reseeded on every `reset()`), **#197** (codify the host-RNG convention),
  **#1020** (`Unlock` door-in-perimeter / two-room topology, deferred),
  **#1021** (`MultiRoom` procedural generation, deferred — its acceptance
  chooses between the misregistered `-v0` and the corrected `-v1`).
- ADR [0029](0029-host-rng-seeding-convention.md) — the persistent-stream
  convention this ADR depends on and whose §3 grid carve-out (lines 117-119)
  and Neutral consequence (line 161) it supersedes. Everything else in 0029
  stands.
- ADR [0026](0026-shared-config-validation-convention.md) — the construction
  chokepoint whose scope §3(a) delimits (config, not per-episode draws).
- ADR [0036](0036-adopt-proptest-for-property-tests.md) — why the seed-loop
  shape, not proptest, is correct for these invariants.
- ADR [0043](0043-grid-observation-contract.md), ADR
  [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) — the
  grid observation contract, unchanged by this ADR.
- ADR [0033](0033-share-splitmix64-mixer-across-core-and-evolution.md) — the
  in-repo precedent for partial supersession (of ADR 0004).
- `docs/rules.md` §8, "Host-RNG seeding convention (ADR 0029)" — gains this
  ADR's §1/§2 binary and the `R: Rng + ?Sized` restatement.
- Code — the seam: `crates/rlevo-environments/src/grids/go_to_door.rs`
  (`rng` field `:536`, `build(&config, &mut rng)` `:648-676`,
  `reset_with_seed` `:583-597`, `seed` doc of record `:395-401`, the
  hand-rolled `sample_door_colors` `:682` this ADR's §3 replaces),
  `crates/rlevo-environments/src/grids/memory.rs`,
  `crates/rlevo-environments/src/grids/dynamic_obstacles.rs:389-402`
  (the materialized-candidate idiom §3(b) adopts).
- Code — the subjects: `crates/rlevo-environments/src/grids/`
  `{crossing,dist_shift,door_key,empty,four_rooms,lava_gap,multi_room,unlock,unlock_pickup}.rs`;
  `unlock.rs:293-305` (the perimeter-wall door);
  `four_rooms.rs:90-97` (the `const _: ()` tied to the fixed offsets).
- Code — the seams this ADR leans on:
  `crates/rlevo-environments/src/grids/core/grid.rs` (`Grid`, `Clone`),
  `crates/rlevo-environments/src/grids/core/dynamics.rs` (`apply_action` — the
  transition function, deterministic and unchanged),
  `crates/rlevo-core/src/environment.rs` (`EnvironmentError`,
  `#[non_exhaustive]`, `Config(#[from] ConfigError)` and its reset-time
  rustdoc).
- Tests: `crates/rlevo-environments/tests/landscape_dim_guards.rs` (the
  source-scanning guard pattern §4 copies),
  `crates/rlevo-environments/tests/grids_solvable.rs` (`ORACLE_SEEDS`, and the
  `GoToDoorEnv` `:269-296` / `MemoryEnv` `:382-412` seed-loop oracles with
  their non-degeneracy guards),
  `crates/rlevo-environments/tests/config_validation_chokepoint.rs`,
  `crates/rlevo-environments/tests/render_coverage.rs:139-158`.
- Other affected consumers: `crates/rlevo/benches/grid_empty_rl.rs` (DQN policy
  quality on `EmptyEnv` — depends on `Empty` staying deterministic, which §1
  confirms it should),
  `crates/rlevo/examples/envs/grids/grid_door_key_scripted.rs`,
  `crates/rlevo-environments/src/bench/family.rs` (feature-gated `bench`/
  `record` — must be checked under those features, not only the default set),
  `crates/rlevo-examples/examples/grids/report_grids_with_client.rs`.
- Chevalier-Boisvert, M., Dai, B., Towers, M., de Lazcano, R., Willems, L.,
  Lahlou, S., Pal, S., Castro, P. S., Terry, J. *Minigrid & Miniworld: Modular
  & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks.*
  arXiv:2306.13831 (2023). §2.2 (transition determinism — the claim Context §2
  shows was misread as layout determinism), §2.4 and Figure 4 (`_gen_grid` as
  the world-generation seam).
- Farama-Foundation/Minigrid, `master` @ 2026-07-26:
  `minigrid/envs/{empty,distshift,lavagap,crossing,doorkey,fourrooms,unlock,unlockpickup,multiroom}.py`,
  `minigrid/core/roomgrid.py` (`place_in_room`, `add_object`,
  `reject_next_to`), `minigrid/minigrid_env.py` (the `place_obj`/`place_agent`
  rejection-sampling contract), `minigrid/__init__.py` (registered kwargs,
  including the `MultiRoom-N4-S5-v0` misregistration).
- Provenance: `docs/.private/research/2026-07-26-issue-282-grid-rng-minigrid-reconciliation.md`
  — the full per-env reconciliation and citation trail. That file is
  **gitignored**, so this ADR reproduces its load-bearing verdicts (Context §3)
  and citations rather than deferring to it.
