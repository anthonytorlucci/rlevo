---
project: rlevo
status: active
type: decision
date: 2026-07-11
tags: [adr, decision, environments, grids, memory, gotodoor, observation, pomdp, issue-109]
---

# ADR 0043: Grid observation contract — cue-hiding by geometry, mission by channel

## Status

**Accepted (2026-07-11).** Resolves issue #109 ("Grid gameplay bugs: memory
task defeated and static instruction-conditioned mission"). Supersedes
nothing.

## Context

Two `rlevo-environments` grid envs claimed properties they did not have. Both
findings are verified against current source and the reference implementation
(Farama-Foundation/Minigrid, `master`).

**`MemoryEnv` is not a POMDP recall task.** The cue was a compile-time
constant (`Key(Yellow)`), the stored `_rng` was never read, and the reward
was keyed to a coordinate independent of the cue — a reactive feedforward
policy solves it. Object colors also leaked the answer (cue/match Yellow,
distractor Red); canonical Minigrid makes all three objects green so only
object *type* carries signal.

A second, deeper defect was missed by the original code review: rlevo's
`egocentric_view` (`crates/rlevo-environments/src/grids/core/grid.rs:108`)
applies **no occlusion** — it reads every cell of the rotated `7×7` window
straight from the grid; walls never block line of sight. Canonical Minigrid
sets `see_through_walls=False` for `MemoryEnv` and runs
`Grid.process_vis` shadow-casting every observation. Canonical guarantees the
recall property by **two** mechanisms: (a) occlusion plus placing the cue off
the corridor centerline behind a doorway wall, and (b) at larger sizes, raw
distance — the backward view reach from the fork is exactly 6 cells, and for
S11+ the cue at `x=1` falls outside it. rlevo's fixed `7×5` layout put the
cue *on* the centerline only 4 cells from the fork, so even after sampling
the cue, an agent could turn and simply re-read it — sampling alone is
necessary but not sufficient to restore the recall property.

**`GoToDoorEnv` advertised "instruction-conditioned policies" but the
mission never reached the policy.** `build_snapshot` emits only
`GridObservation` (the shared `7×7×3` egocentric view,
`crates/rlevo-environments/src/grids/core/observation.rs`, channels
`[type, color, state]` per `Entity::color_u8`,
`crates/rlevo-environments/src/grids/core/entity.rs:106`) plus a non-tensor
direction byte; `mission()` had zero callers workspace-wide. Sampling the
target color — the fix the issue literally prescribed — would therefore have
made the env information-theoretically unsolvable: all four target
hypotheses yield byte-identical observation sequences, capping every policy
at 25%.

This ADR records how hidden information enters and leaves a grid
observation — one subject spanning both envs.

## Decision

### 1. Cue-hiding is a stated geometric invariant, not a magic number

**Invariant M:** *for every cell in the decision region and every facing
direction, the egocentric view must not contain the cue cell.*

`MemoryEnv` becomes size-configurable on the canonical layout — square, odd
`size`; `hallway_end = size - 3`; fork column `size - 2`; cue at
`(1, height / 2 - 1)`, off the corridor centerline — and its `Validate`
impl (ADR 0026) **rejects `size < 11`**.

Derivation of the 11: the agent sits at view cell `[6][3]` (`VIEW_SIZE = 7`,
so `agent_row = VIEW_SIZE - 1 = 6`, `agent_col = VIEW_SIZE / 2 = 3`) looking
toward row 0, giving a forward/backward reach of 6 and a lateral reach of
±3. The cue is at `x = 1`; the fork column is `size - 2`. Requiring
`(size - 2) - 6 > 1` gives `size >= 11` (rounded up to the next odd value).

rlevo therefore ships a **strict subset** of canonical configs: S11 and S13
reproduce `MiniGrid-MemoryS11-v0` / `S13`; S7 and S9 do not exist here,
because canonical relies on occlusion at those sizes and rlevo has none.
`size >= 11` is a **consequence** of Invariant M under rlevo's no-occlusion
view, not an independent design choice — adding occlusion later relaxes the
bound to 7, a one-line change. This is a two-way door, and the reason it is
recorded: without this ADR a future contributor "fixes" the minimum size
back to 7 and silently un-fixes #109.

### 2. Goal/instruction information reaches a policy via a bespoke per-env observation channel

`GoToDoorEnv` gets its own `GoToDoorObservation`, shape `[7, 7, 4]`: channels
0..2 are the existing entity encoding (identical to `GridObservation`),
channel 3 broadcasts the mission's color byte to every cell.

The broadcast is an **ordinal byte matching `Entity::color_u8`'s encoding**,
not a one-hot vector. `Entity::color_u8` already encodes perceived door
colors as an ordinal byte in channel 1 of the shared entity encoding; putting
the mission in the *same* encoding is what lets a network learn *equality*
between the mission channel and a perceived door's color channel. A one-hot
mission against ordinal door colors would be an encoding mismatch on the two
sides of the very comparison the task exists to test.

Rank stays 3, so `Environment<3, 3, 1>` is unchanged and the snapshot
remains a `pub type GoToDoorSnapshot = SnapshotBase<3, GoToDoorObservation,
ScalarReward>;` alias, following the local-type-alias convention of ADR 0042
(`GridSnapshot = SnapshotBase<3, GridObservation, ScalarReward>` at
`crates/rlevo-environments/src/grids/core/mod.rs:45` is the precedent).

Rejected alternatives for carrying the mission (see Alternatives considered
below): widening the shared `GridObservation` to 4 channels; the
`SnapshotMetadata` seam from ADR 0042; an out-of-band `env.mission()`
accessor plus a wrapper env.

This decision is the **precedent** for how any future goal-conditioned or
instruction-conditioned env in rlevo surfaces its goal to a policy — a
bespoke per-env observation type that folds the goal into the existing
per-cell encoding, not a side channel. That precedent-setting is the
one-way-door part of this ADR and the main reason it needs to be recorded.

### 3. The occlusion gap is recorded as a stated non-decision

rlevo applies no visibility masking anywhere; `see_through_walls` is
effectively always `true` crate-wide (§Context). This is a real fidelity gap
versus Minigrid, whose default is `false`.

This is **not fixed** by this ADR. The blast radius is all 12 grid envs'
observation semantics at once, it invalidates every existing grid benchmark
baseline, and a correct fix needs a per-env `see_through_walls` knob
(canonical `GoToDoor` sets it `true`; canonical `Memory` sets it `false`).
It is recorded explicitly here rather than left as folklore. Invariant M
(Decision 1) is deliberately phrased so that it **survives** a future
occlusion change untouched — occlusion merely makes the invariant
satisfiable at smaller sizes; it does not change what the invariant states.

Tracked in **issue #281**. When it lands, `MemoryEnv`'s `MIN_SIZE` relaxes
from `11` to `7` and canonical S7/S9 become reproducible; that change is
ADR-worthy in its own right and should supersede this section, not edit it.

The related **dead-RNG sweep** (nine other grid envs still store an unread
`_rng` and still carry the ADR-0029-violating re-seed inside `reset()`) is
tracked in **issue #282**.

## Consequences

### Positive

- `MemoryEnv` genuinely tests recall: no reactive policy can solve it (see
  the acceptance-test note below), and rlevo's supported sizes (S11, S13)
  are real subsets of canonical Minigrid configs rather than a fixed,
  solvable-by-inspection layout.
- `GoToDoorEnv`'s mission is information-theoretically recoverable by a
  policy for the first time, without capping achievable accuracy at 25%.
- Both fixes are geometric/type-level (`Validate` rejection, a new
  observation shape), not magic constants — a later contributor cannot
  silently regress either without touching a documented invariant.
- Establishes a reusable pattern (bespoke per-env observation type, ordinal
  mission channel matched to the existing entity encoding) for future
  goal-conditioned grid envs.

### Negative / accepted costs

- **The grid family now has two observation shapes**: `GridObservation`
  `[7,7,3]` (the other 11 envs) and `GoToDoorObservation` `[7,7,4]`
  (`GoToDoorEnv` only). Any future "one model across all grid envs" harness
  must handle both shapes. There is no such consumer today, and the split is
  reversible.
- **rlevo's `MemoryEnv` is a strict subset of canonical Minigrid.** S7/S9
  configs do not exist and cannot be added without first closing the
  occlusion gap (Decision 3).
- **The occlusion gap remains crate-wide.** `see_through_walls` stays
  effectively `true` everywhere; this ADR records the gap but does not close
  it (Decision 3).

### ADR 0029 was being violated by both files

Both `MemoryEnv` and `GoToDoorEnv` re-seeded `self._rng` from `config.seed`
inside `reset()` — the exact anti-pattern ADR 0029 (persistent-stream
`reset`) forbids. This was harmless only because the RNG was never read; the
moment cue/mission sampling is added without deleting that line, every
episode draws an identical cue/mission and the #109 bug survives behind a
passing test. Both re-seed lines are deleted; both envs gain an inherent
`reset_with_seed` per the ADR 0029 pattern. Nine other grid envs still carry
a dead `_rng` and the same latent re-seed hazard; that is a known gap
tracked separately from this ADR's scope (rules.md §12 governs filing it).

### Config surface: two fields removed

`MemoryConfig::swap_fork` and `GoToDoorConfig::target_color` are removed.
Both are config fields that pin a quantity the environment is supposed to
sample per episode; both defaults would silently re-create the exact bug
this ADR fixes. Determinism for tests is served by ADR 0029's
`reset_with_seed`, which is strictly better than a pinning config field
because it exercises the real (sampling) environment rather than a special
non-sampling mode.

### Acceptance test shape

`MemoryEnv`'s acceptance test is an adversarial observation-equality pair:
two episodes differing *only* in cue type, with the same fork order, must
produce byte-identical observations at the decision cell while requiring
*different* correct actions. This is a mechanical proof that no reactive
policy can beat chance on the env. It only passes if all three objects are
green — the all-green rule (§Context) is load-bearing, not cosmetic;
non-green objects would let the test (and a real policy) shortcut through
color instead of type.

## Alternatives considered

- **Widen the shared `GridObservation` to 4 channels.** Rejected: changes
  all 12 grid envs' observation shape from `147` to `196` elements and
  breaks every existing grid benchmark baseline, for the benefit of one env.
- **Carry the mission on `SnapshotMetadata` (ADR 0042).** Rejected:
  `SnapshotMetadata` is a `BTreeMap<&str, f32>` that never reaches a
  policy — `BenchStep` drops it entirely (ADR 0042, "Neutral" consequences)
  — and encoding a color as an `f32` is abuse of a debug/analysis field, not
  a policy input.
- **An out-of-band `env.mission()` accessor plus a wrapper env.** Rejected:
  no rollout loop in the repository threads a side-channel accessor into a
  policy's observation; this is exactly the gap that let `mission()` sit
  with zero callers workspace-wide in the first place.
- **Fix cue-hiding with a fixed larger layout instead of a `Validate`
  invariant.** Rejected: a fixed size hard-codes the *consequence* (11)
  without recording the *invariant* (Decision 1) that produced it, which is
  precisely the kind of magic number a future contributor could
  "simplify" back to a smaller, exploitable size.
- **Close the occlusion gap now instead of deferring it.** Rejected for this
  issue: the blast radius is crate-wide (all 12 envs, every benchmark
  baseline) and needs a per-env `see_through_walls` knob design of its own;
  recorded as Decision 3 instead of folded into #109's fix.

## References

- Issue #109 — "Grid gameplay bugs: memory task defeated and static
  instruction-conditioned mission."
- ADR [0026](0026-shared-config-validation-convention.md) — the `Validate`
  / `ConfigError` convention `MemoryConfig` uses to reject `size < 11`.
- ADR [0029](0029-host-rng-seeding-convention.md) — the persistent-stream
  `reset()` convention both envs were violating; `reset_with_seed` is this
  ADR's prescribed remediation pattern.
- ADR [0042](0042-snapshotbase-carries-optional-metadata.md) — the
  `SnapshotBase` local-type-alias convention `GoToDoorSnapshot` follows, and
  the `SnapshotMetadata`/`BenchStep` behaviour cited in Alternatives
  considered.
- Code: `crates/rlevo-environments/src/grids/core/grid.rs:108`
  (`egocentric_view`, the un-occluded view read), `crates/rlevo-environments/src/grids/core/observation.rs`
  (`GridObservation`, `OBS_CHANNELS`, `VIEW_SIZE`), `crates/rlevo-environments/src/grids/core/entity.rs:106`
  (`Entity::color_u8`), `crates/rlevo-environments/src/grids/core/mod.rs:45`
  (`GridSnapshot` alias precedent), `crates/rlevo-environments/src/grids/memory.rs`,
  `crates/rlevo-environments/src/grids/go_to_door.rs`.
- Reference implementation (authoritative for these two envs):
  Farama-Foundation/Minigrid, `master` —
  [`minigrid/envs/memory.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/envs/memory.py),
  [`minigrid/envs/gotodoor.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/envs/gotodoor.py),
  [`minigrid/core/grid.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/core/grid.py)
  (`process_vis`),
  [`minigrid/minigrid_env.py`](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid/minigrid_env.py)
  (`get_view_exts`, `gen_obs_grid`, `_reward`).
- Chevalier-Boisvert et al., "Minigrid & Miniworld: Modular & Customizable RL
  Environments for Goal-Oriented Tasks," NeurIPS 2023 D&B,
  [arXiv:2306.13831](https://arxiv.org/abs/2306.13831). Cited **only** for
  the general POMDP-benchmark-suite framing — this paper does not mention
  `MemoryEnv`, `GoToDoorEnv`, or recall anywhere, so no env-specific design
  rationale in this ADR is attributed to it.
- Morad et al., "POPGym: Benchmarking Partially Observable Reinforcement
  Learning," [arXiv:2303.01859](https://arxiv.org/abs/2303.01859) — critique
  of nominally-POMDP benchmarks solvable by reactive / short-memory policies
  via design shortcuts; motivates Invariant M.
- Osband et al., "Behaviour Suite for Reinforcement Learning,"
  [arXiv:1908.03568](https://arxiv.org/abs/1908.03568), and
  `bsuite/environments/memory_chain.py` — the canonical minimal *valid*
  memory benchmark (context emitted exactly once, all intermediate
  observations uninformative), the design target Invariant M approximates
  within rlevo's geometric constraints.
- Bakker (2002), "Reinforcement Learning with Long Short-Term Memory,"
  NeurIPS — origin of the T-maze recall task family `MemoryEnv` descends
  from. Cited from the NeurIPS proceedings listing; the primary PDF did not
  text-extract, so no text is quoted from it here.
- [Recurrent Action Transformer with Memory](https://consensus.app/papers/details/68507af9cdd55a95be4bb13ed7f0785d/)
  (Cherepanov et al., 2023) — uses Minigrid-Memory as a memory-intensive
  benchmark, motivating why the recall property must actually hold.
- [Probing Dec-POMDP Reasoning in Cooperative MARL](https://consensus.app/papers/details/3165d3bb87845894a6cdfb6470592c81/)
  (Tessera et al., 2026) — finds reactive policies match memory-based agents
  in over half the scenarios of popular partially-observable benchmarks,
  the general failure mode Decision 1 closes for `MemoryEnv`.
