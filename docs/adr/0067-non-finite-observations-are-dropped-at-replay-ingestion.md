---
project: rlevo
status: active
type: decision
date: 2026-07-28
tags: [adr, decision, numerical-stability, nan, observation, replay, hostrow, dqn, c51, qrdqn, sac, ddpg, td3]
---

# ADR 0067: A non-finite observation is dropped at replay ingestion, counted, and reported at `act`

## Status

**Accepted (2026-07-28).** Resolves issue #1043. **Counterpart to ADR 0065**,
which deliberately deferred observation finiteness by number (0065's own "Out
of scope, deliberately" section: "left for a separate issue (**#1043**)").
Supersedes nothing; nothing in 0065's own Decision section is reversed. Two
sentences of 0065's "Out of scope, deliberately" section are corrected below
(this ADR's own "Correction to ADR 0065" section) — 0065 itself is not
edited.

Additive: no public API signature changes, no healthy-step numerics change.
`HostRow` gains one **provided** method, so no existing implementor breaks.

**Chosen shape.** `HostRow::row_is_finite(&self, scratch: &mut Vec<f32>) ->
bool` as a provided method on the core trait; a `FiniteObsGuard` in
`algorithms/shared.rs` mirroring `FiniteRewardGuard`; the check runs at
`remember` on `obs` and `next_obs`, dropping and counting the transition. At
`act` the same predicate runs for **detection only** — it warns and counts and
does not substitute an action. The four integer-backed observation types
override `row_is_finite` with a compile-time witness plus `true`.

## Context

### The harm is invisible to every existing guard, on the backend CI runs

This is the finding that justifies the ADR, and issue #1043 does not contain
it. On the `flex` (CPU) backend, `relu` rescues `NaN` to `0.0`:

```
[flex] relu([NaN, inf, -inf, -1, 2]) = [0.0, inf, 0.0, 0.0, 2.0]
[wgpu] relu([NaN, inf, -inf, -1, 2]) = [NaN, inf, 0.0, 0.0, 2.0]
```

Driven through a real `DqnAgent` on CartPole with a 4-64-64-2 ReLU MLP:

```
[flex] obs=finite   q_row=[ 0.02970457, -0.0838804 ] argmax=[0] act=Left is_valid=true
[flex] obs=one NaN  q_row=[0.023038015, -0.09688881] argmax=[0] act=Left is_valid=true
[flex] obs=all NaN  q_row=[0.023038015, -0.09688881] argmax=[0] act=Left is_valid=true
```

A **fully non-finite observation** yields a finite, in-domain, valid action.
The one-NaN and all-NaN rows are *bit-identical* because the first ReLU zeroed
everything and the output is bias-only: the observation has been erased, not
merely corrupted.

The consequence is the whole argument for this ADR. ADR 0056's
`FiniteLossGuard` cannot see this — the loss is finite. A Q-value check cannot
see it — the Q row is finite. An action check cannot see it —
`Action::is_valid()` returns `true`. **Only a check on the observation itself,
before `to_tensor`, can ever observe this failure**, and it is the case CI
reaches, because CI has no GPU.

### The two backends disagree, in both directions

| case | flex | wgpu / Metal |
|---|---|---|
| all-NaN row, widths 1..=80 | `0` at every width | `-1` or `width`; in range at **no** width |
| `[1, NaN, 3, 2]` | `1` (the NaN) | `2` (correct) |
| `[5, NaN, 0.5, -1]` | `1` (the NaN) | `0` (correct) |
| `[NaN, 2, 3, 1]` | `0` (the NaN) | `2` (correct) |

`flex`'s `argmax` is a `>` scan seeded at element 0, so a NaN never beats the
incumbent unless it *is* the incumbent — which makes the realistic partial-NaN
case return the index of the first NaN. The same case is silently *correct* on
wgpu. The out-of-range wgpu sentinel and its unclamped path into `gather` is a
distinct defect, filed as **#1050**; it is not resolved here.

### The cost question, measured

Benchmark `benches/obs_guard_bench.rs` (commit `a5a927d`), Apple M2 Pro, 12
CPU / 19 GPU cores, Darwin 25.5.0 arm64, rustc 1.94.1. Staging and ingestion
arms host-only; `learn_step` denominators on both `flex` (CPU) and `wgpu`
(Metal). No other machine or backend is covered; CI has no GPU.

| seam | 8-float observation | 27 648-float f32 observation |
|---|---|---|
| `remember` (% of env-step cycle) | 0.36 % | **84 – 385 %** |
| staging (% of `learn_step`) | 0.11 – 0.52 % | 0.04 – 1.79 % |

The predicate's *spelling* dominates its cost.
`buf.iter().all(|v| v.is_finite())` runs at ~9 GB/s against a ~62 GB/s memcpy —
8× the write it fuses into — because `Iterator::all` must short-circuit and so
cannot lower to a horizontal reduction. A branchless `u32::max` reduction over
the IEEE-754 exponent field restores ~1× fusion (0.057 ns/elem against a
0.065 ns/elem write) and is data-independent.

### The workspace splits cleanly, and the split decides the design

Every `row_shape` in `crates/rlevo-environments/src/` was enumerated. Every
image-shaped observation is **integer-backed** and therefore structurally
incapable of carrying a non-finite value:

| type | shape | backing |
|---|---|---|
| `CarRacingObservation` | 96×96×3 = 27 648 | `Arc<[u8; 27648]>` |
| `PixelObservation` | `IMG_SIDE²×3` | `u8` → `f32::from(b) / 255.0` |
| `GridObservation` | 7×7×3 = 147 | `f32::from(channel)` |
| `GoToDoorObservation` | 7×7×4 = 196 | `f32::from(channel)` |

Every observation that *can* go non-finite is a small f32 feature vector: 24
(`bipedal_walker`), 10 (`reacher`), 9 (`inverted_double_pendulum`), 8
(`swimmer`, `lunar_lander`), 6 (`acrobot`), 4 (`inverted_pendulum`,
`cartpole`), 3, 2, 1.

Note precisely what this does and does not say. The expensive **cost** case is
real and shipped — the default body would materialize 27 648 f32 from `u8` on
every env step for `CarRacing`. What has no instance is the expensive
**benefit** case: a large row that can actually be poisoned. Guarding
unconditionally with the simple spelling is therefore the one option that is
definitely wrong — it pays full cost on exactly the types whose benefit is
provably zero.

The poisonable set is the entire rapier/box2d physics family, which is exactly
where integrator blow-up produces a NaN state, and exactly the failure mode
0065's own warning text tells operators to look for ("an exploding
physics/dynamics term"). **The guard is simultaneously at its cheapest and at
its most necessary on the same set.**

## Decision

### 1. `HostRow::row_is_finite(&self, scratch: &mut Vec<f32>) -> bool`

A **provided** method, so all 42 existing `HostRow` impls compile unchanged.
The default body clears `scratch`, calls `write_host_row`, and runs the
branchless exponent-field reduction. The IEEE-754 derivation is documented
inline: this is exactly the code a well-meaning reviewer will "simplify" back
to `.all(is_finite)`, reintroducing an 8× cost and a data-dependent timing.

The `scratch: &mut Vec<f32>` parameter is taken **now**, not later. The default
body otherwise allocates per call, and the first f32-backed image observation
would pay ~110 KB per env step. Changing this signature after the fact is a
breaking change across 42 impls.

### 2. Overrides carry a compile-time witness, never a bare `true`

```rust
fn row_is_finite(&self, _scratch: &mut Vec<f32>) -> bool {
    // Compile-time witness for the structural claim below. If the payload type
    // ever changes, THIS LINE fails to compile and the override must be
    // re-derived, not re-asserted.
    let _: &[u8] = &self.pixels;
    // `u8 -> f32` is total: no element of this row can be non-finite.
    true
}
```

The override is **both** an assertion of a structural fact and a performance
decision, and the performance half is load-bearing. Framing it as pure
assertion is how it stops being maintained.

The witness must be a **concrete type ascription**. It must not be expressed
through `f32::from` or an `Into<f32>` bound: `impl<T> From<T> for T` means
`f32: Into<f32>`, so those spellings keep compiling after an f32 refactor and
guarantee nothing.

This follows the precedent `rules.md`'s Error Handling section already
blesses for derived constants (`const _: () = assert!(..)`, "so lowering it
breaks the build rather than the output").

### 3. The guard sits at `remember`. Not at staging. Not both.

Mirrors 0065: one ingestion chokepoint, one guard shape, one counter, one
place an operator looks. `FiniteObsGuard` in `algorithms/shared.rs`, one
instance per agent, drop-and-count with the unlatched 1/10/100/… decade warn
schedule and a public `dropped_observations() -> u64` accessor.

Staging is rejected as the primary seam because there is no good action there:
dropping a row breaks the batch shape, and the only clean response —
skip-the-step — is what 0056 already does one step later for free. A
staging-seam guard's entire marginal value over 0056 would be attribution.

Not both: two guards means two counters for one event, and they would
legitimately disagree (staging counts *samples*, ingestion counts
*transitions*).

### 4. At `act`, detect and report; do not substitute

The same predicate runs at `act` / `act_greedy` / `act_greedy_with`, warns on
the decade schedule, and increments a counter. It does **not** substitute an
action.

The reasoning is the flex-`relu` finding itself. Substituting a plausible
in-domain action is the same class of failure the finding demonstrates:
a legal-looking action that makes a broken run appear healthy. The value of
the `act` guard is *attribution* — it is the only place in the system that can
observe this failure at all — and attribution is worth having independently of
response. `Action::neutral()` is therefore **not** introduced: it would be a
breaking core-trait change purchased for semantics no site can consume.

This clause is the most open to challenge and is flagged as such.

## Correction to ADR 0065

0065 is not edited. Two sentences of its "Out of scope, deliberately" section
are incomplete in light of the benchmark at `a5a927d`:

1. *"checking an observation is a per-element tensor scan, not one `f32`
   register compare — so it is a different cost/benefit call."* True as stated,
   but it treats observation cost as uniform. The workspace splits cleanly into
   a cheap-and-poisonable set and an expensive-and-unpoisonable one, and that
   split — not the average cost — is what the design turns on.
2. The implicit assumption, repeated in #1043, that the check could ride the
   existing `write_host_row` traversal at near-zero marginal cost. **Measured
   false at `remember`**: `remember` stores the typed `O` and never flattens,
   so there is no traversal there to ride. It is true at the staging seam,
   which is not where the guard goes.

## Consequences

Each of the six agent files carries its own drop test, per 0065's reasoning
that a copy-paste omission is exactly how two of six sites went missing the
first time.

Under `PrioritizedReplay` (ADR 0050's fidelity-contract decision, Decision 10)
a new transition enters at running-max
priority. A poisoned row yields a NaN TD error, `Priority::try_new` rejects it,
the writeback is dropped, and the priority stays **pinned at max** — the poison
is resampled more often than average and never decays. This is an independent
argument for the ingestion seam and against staging-only.

A transition with both a non-finite reward and a non-finite observation
increments only `dropped_transitions()`: 0065's guard runs first and returns
early. The two counters will disagree, and the rustdoc on both must say so.

The degenerate case — a persistently non-finite observation source means the
buffer never fills and training silently never starts — is inherited from 0065
and is still the correct outcome, since the alternative fabricates. But it is
weaker here than for rewards: a NaN observation early in a diverging locomotion
episode is more likely a transient integrator excursion than a permanent signal
defect, so the decade `warn!` is doing more work in this ADR than in 0065.

**Out of scope, deliberately:**

- The unclamped `argmax` → `gather` path (**#1050**), which this ADR's evidence
  surfaced but does not fix.
- PPO/PPG rollout observations. `record_step` ingests an unguarded `obs`, and
  on `EpisodeStatus::Truncated` calls `value_of(next_obs)` — an immediate
  forward pass producing `bootstrap_value`, which seeds the GAE backward
  recursion. 0065's #1042 reasoning transfers unchanged.
- Evaluation-only rollouts that call `act` without `remember`; covered by
  this ADR's own Decision 4 ("detect and report; do not substitute") but by
  no drop.
- `Observation<R>: HostRow<R>` (ADR 0052's Decision 8, "deferred to a
  follow-up"), still open, 12 affected types.
- `ContinuousAction::from_slice` being unchecked by contract, which bypasses
  the action types' own NaN validation (`PendulumAction::new` rejects NaN; no
  agent calls it).

## Notes

`row_is_finite` lands on ~12 action impls and `BootstrapMask`, where the
question is well-formed but pointless. The natural home is `Observation<R>`,
which ADR 0052's Decision 8 made expressible but deferred. That is not forced as a rider
here; the rustdoc is worded as a statement about *rows*, not observations.

A proptest oracle pinning each override against the default body is worth
adding as a tripwire, but **not** as the guarantee. `proptest`'s `any::<f32>()`
excludes NaN and ±Inf by default, so the obvious spelling passes vacuously —
coverage that looks real. The strategy must explicitly request
`f32::INFINITE | f32::QUIET_NAN`. The witness is the guarantee.
