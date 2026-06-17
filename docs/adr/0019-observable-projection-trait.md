---
project: rlevo
status: active
type: decision
date: 2026-06-16
tags: [core, state, observation, pomdp, rank, modality, issue-62]
---

# ADR 0019: The `Observable<OR>` modality-changing projection trait

## Status

Active. Adopted 2026-06-16. Implements issue #62 (the trait half tracked as
issue #64) per the spec
image-observation-over-compact-state-spec. **Additive**: introduces one new
standalone trait in `rlevo-core::state` and changes no existing trait, impl, or
manifest. `State`, `Environment`, `Snapshot`, and `SnapshotBase` are byte-unchanged.
No new dependency.

## Context

`State<SR>` welds its observation to its own tensor order
(`type Observation: Observation<SR>`), so `observe()` can never change rank.
`Environment<R, SR, AR>` leaves `R` (observation order) and `SR` (state order) as
independent const generics and never binds `ObservationType = StateType::Observation`
— it *permits* `R != SR` — but every environment builds its snapshot from
`state.observe()`, so `R == SR` holds universally today (every env is
`Environment<1,1,1>` or `Environment<3,3,1>`).

Issue #62 asked whether to decouple the two. Reviewing the canonical POMDP
benchmarks (Tiger, partially-observable grid worlds, RockSample, LQG, contact
manipulation) against the tension produced the load-bearing finding, recorded in
rank-vs-dimensionality-in-pomdp-observations: **"rank" is overloaded across
three meanings** — tensor order (ndim, the const generics), matrix rank
(linear-algebra, e.g. LQG's `C ∈ ℝᵐˣⁿ`), and axis cardinality (`shape()`). Almost
every canonical example reduces *cardinality* or applies *stochastic emission* at
**constant tensor order** — including LQG, whose "rank `m`" is the matrix rank of `C`
over two rank-1 tensors. All of those are already expressible through `observe()`
returning a smaller-`shape`, same-order observation; `Observation::shape()` is
already decoupled from `State::shape()`.

The **only** canonical case that changes tensor *order* is **modality change** — a
compact low-order latent state observed through a higher-order sensor (Atari RAM →
rank-2/3 pixels). That is the single motivating case, and `State::observe()`
structurally cannot express it.

The user confirmed pixel-over-compact-state environments are on the roadmap and
chose **Option 3** (a dedicated projection trait) over Option 1 (document the
escape hatch) and Option 2 (a second const generic / associated order on `State`,
which would touch every `State` impl).

## Decision

Add one trait to `crates/rlevo-core/src/state.rs` (the POMDP-seam module, beside
`MarkovState`/`BeliefState`/`HiddenState`/`LatentState`/`StateAggregation`):

```rust
pub trait Observable<const OR: usize> {
    type Observation: Observation<OR>;
    fn project(&self) -> Self::Observation;
}
```

A modality-changing environment's state implements `State<SR>` for its full
representation **and** `Observable<OR>` for the projected observation, then builds
its snapshots from `state.project()` instead of `state.observe()`. The environment
is `Environment<R, SR, AR>` with `R == OR != SR`; no environment-contract change is
needed.

### 1. Standalone trait; no `State` supertrait

`Observable` does not bound `Self: State<SR>`. Any type — a `State` impl or a bare
latent representation — may implement it. This keeps it purely additive and avoids
threading a second const generic through `State`'s already-parameterised surface.

### 2. `OR` as a const generic, not an associated const

The order must drive the `type Observation: Observation<OR>` bound, which an
associated const cannot do (a const cannot appear in a trait bound). Both sibling
seams `HiddenState<R>` and `LatentState<R, AR>` use a const generic for the same
reason. The parameter is named **`OR`** rather than the siblings' `R` to make the
state→observation order decoupling self-documenting at every impl site
(`impl Observable<2> for Ram`) and to distinguish it from the state order `SR`. The
naming deviation is deliberate and is called out in the trait doc so a future reader
does not "normalise" it to `R`.

### 3. `project()` is infallible

A pure projection from an owned, already-valid state into a fixed-shape observation
has no runtime failure mode: the output shape is the compile-time constant
`Self::Observation::shape()`, the state is assumed valid (the environment validates),
and there is no I/O. This mirrors the infallible `State::observe` and
`LatentState::decode`. A future emulator that can fail to read a framebuffer models
that failure at the `Environment::step` boundary via `EnvironmentError`, not inside
`project`. Wrapping `project` in `Result` pre-emptively would impose error handling
on every total projection for a failure mode this trait does not own.

### 4. No blanket impl

No `impl<const SR: usize, S: State<SR>> Observable<SR> for S`. Such an impl would
merely duplicate `observe()`, make `.project()` ambiguous on any type that also
hand-writes an `Observable<OR>`, and be an unremovable coherence lock-in in a
contract crate (foreclosing downstream same-order `Observable` impls). Opt-in costs
each modality-changing environment one trivial impl block.

### 5. No `numel()` analogue

`State` carries `numel()`; `Observable` deliberately does not. Observation element
count is reachable via `Self::Observation::shape().iter().product()`; a redundant
method is omitted.

### 6. Home and method name

Lives in `state.rs`, consumed as `rlevo_core::state::Observable` (matching the
no-crate-root-re-export convention of the other state seams). The method is
`project` — it must not collide with `State::observe`, and it reads as a
space/order-changing map, distinct from `observe`'s "read out current perception"
and `LatentState::encode`'s learned/invertible connotation.

## Consequences

**Positive**

- Modality-changing POMDPs (RAM-behind-pixels, camera-over-proprioception) now have a
  typed, tested home instead of inline `step`/`reset` projection code.
- The `Environment` contract is validated as already-sufficient: the cross-crate test
  `crates/rlevo/tests/observable_modality_change.rs` builds an `Environment<2, 1, 1>`
  with no change to `Environment`/`Snapshot`/`SnapshotBase` — `R != SR` was reachable
  all along; only the projection lacked a name.
- The rank-vs-dimensionality clarification (matrix rank vs tensor order vs cardinality)
  is recorded for future env work and for the user-book.

**Negative / accepted costs**

- A type that implements both `State<SR>` and `Observable<OR>` must disambiguate
  `.project()` vs `.observe()` at call sites — trivial and intentional (they are
  different modalities). A type implementing two `Observable<OR>` orders needs
  turbofish; rare, documented.
- `Observable` joins the "defined-but-unconsumed seam" set until a real
  modality-changing environment lands (tracked as issue #65); the `MockRam`
  integration test exercises the contract end-to-end in the meantime.

**Neutral**

- The trait doc names `OR` against the siblings' `R`; the inconsistency is
  intentional and documented in-source.

## Alternatives considered

**Option 1 — document the escape hatch only.** Rejected: with pixel-over-RAM envs on
the roadmap, leaving the projection as untyped inline `step`/`reset` code (declaring an
unrelated `ObservationType` and routing around `observe()`) is unidiomatic and untested.

**Option 2 — second const generic / associated observation order on `State`.** Rejected:
would touch every `State` impl in the workspace and parameterise the most-used trait
further, for no expressive gain over a standalone trait. The standalone trait reaches
exactly the same `R != SR` envs.

**Blanket `Observable<SR> for S: State<SR>`.** Rejected (decision 4): duplicates
`observe()`, worsens `.project()` inference, unremovable coherence lock-in.

**Fallible `project() -> Result<Self::Observation, _>`.** Rejected (decision 3): no
failure mode for a total projection over valid states; framebuffer-read failure belongs
to `Environment::step`/`EnvironmentError`.

**`numel()` on `Observable`.** Rejected (decision 5): redundant with
`Observation::shape().iter().product()`.

## References

- Cassandra, Kaelbling & Littman (1994); Smith & Simmons (2004); Kalman (1960) — the
  canonical POMDP benchmarks surveyed in
  canonical-modality-changing-pomdp-benchmarks.
- image-observation-over-compact-state-spec — governing spec.
- rank-vs-dimensionality-in-pomdp-observations — the rank-taxonomy finding.
- [0011-lift-construction-off-environment-trait](0011-lift-construction-off-environment-trait.md) — the "separate concern → standalone
  trait, not a supertrait" pattern this ADR follows.
- `crates/rlevo-core/src/state.rs` — the trait + in-source unit tests.
- `crates/rlevo/tests/observable_modality_change.rs` — the cross-crate `R != SR` proof.
- Issues #62 (design tracker), #64 (this implementation), #65 (real env follow-up).
