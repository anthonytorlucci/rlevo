---
project: rlevo
status: active
type: decision
date: 2026-07-26
tags: [adr, decision, rlevo-core, observation, serde, trait-bounds, issue-405]
---

# ADR 0064: `Observation<R>` carries no serde supertrait; persistence is declared at the consuming seam

## Status

**Accepted (2026-07-26).** Resolves issue #405 (`Observation`'s
`Serialize + for<'de> Deserialize<'de>` supertrait is justified by a doc comment
that describes a capability nothing in the workspace uses). `Observation<R>`'s
supertrait list becomes `Debug + Clone + Send + Sync`. **Follows the precedent set
by ADR [0052](0052-hostrow-supertrait-splits-layout-from-backend.md) §8** — a
supertrait that is *expressible* is not thereby *warranted* — and applies it in the
removal direction. Supersedes nothing; `Action` and `Reward` are untouched.

## Context

### The bound's stated justification was false in every direction

`crates/rlevo-core/src/base.rs` declared

```rust
pub trait Observation<const R: usize>:
    Debug + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>
{ /* RANK, shape(), ... */ }
```

with a doc comment attributing the serde half to *storage in a replay buffer*.
Verified against the tree:

- **No function, struct, or trait in the workspace generically requires
  `O: Serialize` or `O: Deserialize`.** Not one bound anywhere consumes what the
  supertrait supplies.
- **The replay path does not serialize.** `ExperienceTuple` and `History`
  (`rlevo-reinforcement-learning/src/experience.rs:29,79`) derive only
  `Clone, Debug`. ADR [0050](0050-replay-strategy-seam.md) records that nothing in
  the replay path derives serde, and that the strategy seam **erases the action
  payload precisely to avoid imposing bounds** — the opposite of the policy this
  supertrait encoded.
- **The one real persistence consumer of a domain type requires nothing of the
  observation.** `RecordingTap` declares `where E::ActionType: Serialize + Clone`
  explicitly (`rlevo-benchmarks/src/record/env_tap.rs:300,322`). `capture_frame`
  (`env_tap.rs:291-315`) persists action bytes, reward, ascii, styled, and the
  family payload; **no observation is present in any persisted payload**, which is
  also why removing the bound needs no record-schema `FORMAT_VERSION` bump.

So the bound constrained neither a wire format nor round-trip fidelity. It bought
no invariant. Contrast the `TensorConvertible` clauses of ADR
[0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md), where the
bound-adjacent contract text says something checkable about the data: nothing
about `Serialize` on `Observation` said anything about anything.

### What the bound cost, concretely

Two hand-written serde impls exist **only** to satisfy it, and no consumer
exercises either:

- `ContextualBanditObservation`'s ~30-line validating `Deserialize`
  (`rlevo-environments/src/classic/bandit/contextual.rs:118`), hand-written
  because `docs/rules.md` §4 requires deserialized data to yield `Err` rather than
  panic — a correct impl, upholding a real rule, for a code path no consumer
  reaches.
- `CarRacingObservation`'s `Visitor`
  (`rlevo-environments/src/box2d/car_racing/observation.rs:99,110`), hand-written
  because serde derives no array impl above length 32 and the observation is
  27,648 bytes.

### It made the real bound set invisible at the use site

`experience.rs:24-28,76-80` cites the Rust API Guidelines' **C-STRUCT-BOUNDS** to
justify declaring *no* `Send`/`Sync` bounds on `ExperienceTuple`/`History` — those
auto-traits propagate from the parameters, and stating them "would restrict the
type without adding guarantees."

That claim is narrow and it is **correct** — before this change and after it. It
speaks only to `Send`/`Sync`, and this ADR does not amend or contradict it. The
problem is what the prose does *not* mention: the same struct headers write
`O: Observation<D>`, which silently imported a `Serialize + Deserialize`
obligation the doc comment never speaks to. A reader of that thorough,
deliberately-reasoned comment about which bounds the struct declares could not see
the full set of bounds the struct actually *required*. The defect is
**incompleteness, not contradiction** — and it is a real argument for removing the
supertrait, because a supertrait-imported obligation is invisible exactly where
callers read for it.

**Cite C-STRUCT-BOUNDS honestly, though**: that guideline is literally about
bounds on `struct` *definitions*, not about supertraits, and it is invoked here
for its **rationale** (do not restrict without guaranteeing) and for internal
consistency with `experience.rs` — not as governing authority over supertrait
choice. Overstating it would be the same move this ADR is removing: a citation
doing rhetorical work its text does not support.

### The counterargument, stated rather than hidden

ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) warns
that *"no current call site decodes"* is a snapshot of today's code, not a
property of the design. **That warning applies to this ADR's evidence directly**,
and it is not dismissed.

The distinguishing point is what deferring buys in each case. 0061's design
produced **wrong output** — a fabricated `Direction` that looked like a
measurement — so deferring there meant shipping a lie and waiting for a future
consumer to be misled by it. This design produces only a **cost**: an unused
obligation and two unexercised impls. And 0061's future consumer (an offline-RL
loader, a tensor-backed replay) would have had to *work around* the defect; this
ADR's future consumer is served by adding one `where` clause of its own.

The anticipated consumers are real and worth naming: `docs/.private/roadmap.md`
lists **distributed replay architectures**, and offline-RL loaders and checkpoint
restore are plausible near-term work. Each of those declares
`T: Serialize + for<'de> Deserialize<'de>` **at its own seam**, which is
**additive and reversible**. Re-removing a supertrait later is not — it breaks
every downstream generic that came to rely on it in the interim. That asymmetry
is the core reason this is a two-way door and the status quo was a one-way one:
keeping the bound "just in case" makes the cheap direction expensive.

## Decision

### 1. The supertrait list loses serde

```rust
pub trait Observation<const R: usize>: Debug + Clone + Send + Sync {
    const RANK: usize = R;
    fn shape(&self) -> [usize; R];
    /* unchanged */
}
```

`Debug + Clone + Send + Sync` is the whole contract: an observation must be
inspectable, copyable into a buffer, and movable across threads. Those four are
load-bearing — `Clone` for `ExperienceTuple`'s by-value storage, `Send + Sync` for
moving a `History` into a worker thread. Serialization is not in that set. The
now-unused `serde` import at `base.rs:10` is deleted with it.

This is a **strict relaxation for implementors**: every existing
`impl Observation<R>` still compiles unchanged, because none of them lose a
capability they had.

### 2. All existing serde derives and both manual impls are retained

Nothing is deleted from any concrete observation type. Every `#[derive(Serialize,
Deserialize)]` stays, and both hand-written impls
(`ContextualBanditObservation`'s validating `Deserialize`,
`CarRacingObservation`'s `Visitor`) stay. `docs/rules.md` §8 still expects
concrete domain types to be serde-capable; what changes is that this is a
*property of the types*, not an *obligation of the trait*. `serde_json::to_string(&concrete_obs)`
is unaffected everywhere.

Retention is deliberate, not inertia: deleting the derives would make the change
larger, would break any downstream persisting a concrete observation, and would
throw away the `contextual.rs` impl's rules-§4 validation logic, which is the only
place that reasoning is written down.

### 3. A serde requirement is declared at the seam that has it

The reference shape is `RecordingTap`'s explicit
`where E::ActionType: Serialize + Clone` (`env_tap.rs:322`). A future distributed
replay, offline-RL loader, or checkpoint writer that needs to persist an
observation writes `where O: Serialize + for<'de> Deserialize<'de>` on **its own**
type or function. `docs/rules.md` §8's `serde` row and §3's `Observation<D>`
invariant row are amended to say so, so the rule is enforceable at review time
rather than living only here.

### 4. `Action` and `Reward` deliberately do **not** gain the bound

The symmetric alternative — make the three domain traits consistent by adding
serde to `Action` and `Reward` — was measured, not estimated, and rejected. See
Alternatives Considered.

## Consequences

### Positive

- **The contract now states only what it enforces.** A doc comment that promised
  replay-buffer persistence is replaced by four supertraits that every consumer
  actually uses.
- **Two hand-written serde impls stop being load-bearing on a trait bound.** They
  remain, but as a property of their types, not as tax collected by
  `rlevo-core`. The next type serde cannot derive for — a large fixed array, an
  opaque handle, a borrowed frame buffer — implements `Observation` without a
  `Visitor`.
- **`ExperienceTuple`/`History`'s full bound set is now visible at the header.**
  `experience.rs`'s C-STRUCT-BOUNDS reasoning was already correct and is unchanged;
  what improves is that `O: Observation<D>` no longer imports an obligation the
  doc comment does not mention. Every bound the struct requires is one it uses.
- **Applies 0052 §8's ruling symmetrically.** 0052 declined to *add* an
  expressible supertrait (`Observation<R>: HostRow<R>`) because it bought no
  invariant and had no consumer; this removes an existing one on the same two
  grounds. The project now has one rule for supertraits on domain traits, applied
  in both directions.

### Negative / accepted costs

- **Breaking for downstream generic code**, even though it is a relaxation for
  implementors. `fn save<O: Observation<D>>(o: &O) { serde_json::to_string(o) }`
  compiles today and will not after. **In-tree affected sites: 0.** The migration
  is one `where O: Serialize` clause; see `CHANGELOG.md`.
- **The "someone will re-add this for the replay buffer" failure mode is live.**
  The removed bound's own doc comment is exactly the argument a future contributor
  will reconstruct from first principles when distributed replay lands, and adding
  a supertrait is a two-line diff that compiles. Mitigated by three artifacts and
  nothing else: this ADR, `base.rs`'s replacement rustdoc (which names the
  reference `where`-clause shape and says *do not reinstate*), and the sharpened
  `rules.md` rows. If a future PR proposes the supertrait again, the required
  answer is the `where` clause at the new seam.
- **Slight asymmetry with `docs/rules.md` §8's expectation** that concrete domain
  types derive serde: the trait no longer guarantees what the convention still
  expects. Accepted, and the reason the §8 row is being sharpened rather than left
  alone — a convention about concrete types must not read as licensing a
  supertrait obligation.

### Neutral

- **No runtime behaviour change of any kind.** No byte is serialized differently,
  no allocation moves, no tensor path is touched.
- **No persisted data format changes, and no record-schema `FORMAT_VERSION`
  bump.** No observation was ever present in any persisted payload
  (`env_tap.rs:291-315`).
- **No dependency change.** `serde` remains a dependency of `rlevo-core` (other
  types use it); only one unused `use` in `base.rs` is removed.
- **No `Action`/`Reward` change** (Decision §4), so `RecordingTap`'s existing
  `E::ActionType: Serialize` where-clause is untouched and continues to work
  exactly as before.

## Alternatives Considered

- **Option 1 — extend the bound to `Action` and `Reward` for consistency.**
  Rejected on **measured** cost. A trial edit produced **260+ compile errors**,
  and that is a **floor**, not a total: cargo aborted before reaching `rlevo`,
  `rlevo-examples`, and `rlevo-test-support`. Discharging them requires roughly
  **31 new derives across at least 7 crates**, reaching **25 unique types**.
  Notably, **zero of those types are non-derivable** — so this is rejected as a
  *cost* decision, not a feasibility one, and that distinction is recorded
  deliberately: nobody should re-litigate this believing it was blocked. The
  substantive objection is not the error count but that it buys the same nothing,
  three times over: no consumer requires serde on an action or a reward
  generically, and ADR 0050 went out of its way to *erase* the action payload in
  the replay seam rather than bound it.

- **Option 2 — keep the bound, fix only the doc comment.** Zero migration, and it
  removes the false claim. Rejected: it keeps every cost (the tax on future
  implementors, the two unexercised hand-written impls, and an obligation that
  stays invisible at `ExperienceTuple`/`History`'s headers, where callers read for
  it) and merely stops *explaining* them. A supertrait
  with an honest doc comment saying "required by nothing" is worse than no
  supertrait, because it invites the next reader to invent a justification.

- **Add the bound back behind a cargo feature** (`#[cfg_attr(feature = "serde",
  ...)]` on the supertrait list). Rejected: a supertrait that appears and
  disappears with a feature makes `Observation` two different traits, so a
  downstream generic compiles or fails depending on unrelated feature unification
  elsewhere in the graph — precisely the
  `--all-features`-hides-feature-gated-breakage class of hazard. A `where` clause
  at the consuming seam achieves the same optionality with no trait-identity
  split.

- **Introduce a separate `SerializableObservation: Observation + Serialize +
  Deserialize` marker trait.** Rejected as premature: it has zero implementors and
  zero consumers today, and it is exactly the abstraction a future distributed
  replay can introduce *if* one seam is not enough. Adding it now would be
  speculative, and per 0052 §8 the bar for a new trait obligation is a consumer,
  not an anticipation.

## References

- Issue #405 — `Observation`'s serde supertrait is justified by an unused
  capability.
- ADR [0052](0052-hostrow-supertrait-splits-layout-from-backend.md) §8 — declined
  to add `Observation<R>: HostRow<R>` although newly expressible, because it
  bought no invariant and had no consumer; the precedent this ADR applies in the
  removal direction.
- ADR [0050](0050-replay-strategy-seam.md) — the replay seam derives no serde and
  erases the action payload to avoid imposing bounds; the direct refutation of the
  removed doc comment's "for storage in a replay buffer" claim.
- ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md) — the
  "no current call site decodes is a snapshot, not a property" warning; applied to
  this ADR's own evidence in §Context, with the wrong-output-versus-cost
  distinction that separates the two cases.
- ADR [0026](0026-shared-config-validation-convention.md) and `docs/rules.md` §4 —
  the deserialized-data-must-`Err`-never-panic rule that
  `ContextualBanditObservation`'s hand-written `Deserialize` upholds.
- Rust API Guidelines, **C-STRUCT-BOUNDS** — cited by `experience.rs:24-28,76-80`
  for its rationale; see §Context for the honest scope caveat (it governs struct
  definitions, not supertraits).
- `docs/.private/roadmap.md` — "Distributed replay architectures", the named
  anticipated consumer whose requirement is additive at its own seam.
- Code: `crates/rlevo-core/src/base.rs` (`Observation`, the removed supertraits
  and the deleted `serde` import);
  `crates/rlevo-reinforcement-learning/src/experience.rs:24-29,76-80`
  (`ExperienceTuple`/`History`, C-STRUCT-BOUNDS reasoning, `Clone, Debug` only);
  `crates/rlevo-benchmarks/src/record/env_tap.rs:291-315,300,322`
  (`RecordingTap`'s explicit `where E::ActionType: Serialize + Clone`;
  `capture_frame` persists no observation);
  `crates/rlevo-environments/src/classic/bandit/contextual.rs:118` (validating
  `Deserialize`, retained);
  `crates/rlevo-environments/src/box2d/car_racing/observation.rs:99,110`
  (`Visitor` over 27,648 bytes, retained); `docs/rules.md` §3 invariant table and
  §8 `serde` row.
- Verification: `cargo check --workspace --all-targets`, `--all-features`, and
  eleven feature-gated combinations each reported 0 errors;
  `cargo test --doc --workspace` passed. Total diagnostic output across the change
  was two unused-import warnings in the edited file.
