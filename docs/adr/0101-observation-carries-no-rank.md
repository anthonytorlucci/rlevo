---
project: rlevo
status: active
type: decision
date: 2026-11-14
tags: [adr, decision, observation, host-row, tensor-convertible, const-generics, rank, environment-contract, rlevo-core, rlevo-reinforcement-learning, public-surface, breaking-change]
---

# ADR 0101: An observation states its shape, and the model states the rank

> This ADR is written as the change that follows ADR
> [0100](0100-action-carries-no-rank.md). The counts in Context are searches
> over `crates/`, `xtask/` and `templates/` on `main` at `af3ef1cf`, before
> either change.

## Status

**Accepted (2026-11-14).** Lands in one change with ADR
[0100](0100-action-carries-no-rank.md).

- Amends ADR [0100](0100-action-carries-no-rank.md) Decision 2.
  `Environment<OR>` becomes `Environment`.
- Amends ADR [0090](0090-the-environment-contract-keeps-only-what-generic-code-reads.md).
  The contract it kept names `Observation<R>` and a ranked `Environment`.
  They become `Observation` and `Environment`, with no const parameter.
- Amends ADR [0089](0089-the-contract-crate-does-not-depend-on-burn.md)
  Decision 5 (`HostRow<R>: Observation<R>` becomes `HostRow: Observation`)
  and Decision 2 (`to_tensor` takes its rank on the method). The rule that
  a type implements `HostRow` at exactly one rank has nothing left to
  forbid.
- Restores what ADR [0028](0028-tensor-batch-conversion-seam.md)
  Decision 2 called the single batch-rank chokepoint. Its
  `stack_to_tensor` no longer exists, and the layout it owned is written by
  hand in nine files.
- Does not decide whether a shape belongs to a type or to an environment
  instance. See "Out of scope".

## Context

```rust
pub trait Observation<const R: usize>: Debug + Clone + Send + Sync {
    fn shape() -> [usize; R];
}
pub trait HostRow<const R: usize>: Observation<R> { /* write_host_row, row_is_finite */ }
pub trait TensorConvertible<const R: usize>: HostRow<R> {
    fn to_tensor(&self, device: &Device) -> Tensor<R> { /* provided */ }
}
```

### What reads `Observation::shape()`

- `TensorConvertible::to_tensor`, as the shape of the one tensor it builds.
- The learn step of every agent (DQN, C51, QR-DQN, PPO, PPG and its aux
  buffer, DDPG, TD3, SAC). Each takes the product for a buffer capacity and
  copies the array into a `Vec<usize>` to build `[batch, ..shape]`. That
  block is written out ten times in nine files.
- Outside code that wants a feature count: the `rlevo-algo` agent and
  environment seams and the `byoa-1a` probe read `shape()[0]`.

Every reader uses the shape as a slice. The array length matters to the
compiler in one place, the `Tensor<R>` that `to_tensor` returns.

### What reads `Tensor<R>`

Twenty-one call sites in `rlevo-reinforcement-learning/src` call
`to_tensor`. Each is on an act or value path and feeds a network that takes
`Tensor<BOR>`, so the rank-`OR` tensor exists to be unsqueezed:

```rust
let obs_t: Tensor<OR> = obs.to_tensor(&self.device);
let batched: Tensor<BOR> = obs_t.unsqueeze::<BOR>();
```

No network in the workspace takes an unbatched observation.

### What the rank checks, and what it does not

| Property | Today |
|----------|-------|
| `shape()` has as many entries as the declared rank | compiler |
| `to_tensor` returns a tensor of the declared rank | compiler |
| `BOR == OR + 1` | **nothing**; a wrong `BOR` panics in `unsqueeze` or `TensorData` |
| the model's input rank suits the observation | **nothing**; by its bounds, `DqnAgent<M, O, A, 1, 4>` with a rank-1 `O` and a `DqnModel<4>` type-checks |

`OR + 1` is not expressible on stable Rust, so every agent takes `OR` and
`BOR` as independent parameters, and `docs/rules.md` states their relation
as a convention. The first two rows are checks on a number the observation
declares about itself. The last two are the ones a user gets wrong, and
they are unchecked.

### What the rank costs

- 45 `Observation` impls: 36 at rank 1, 3 at rank 2, 6 at rank 3. 30
  `HostRow` impls, each repeating the rank.
- 37 `const OR: usize` declarations in 10 files; 58 bounds outside `impl`
  headers that spell `Observation<..>` with a rank; 53 lines in 14 files
  that spell `TensorConvertible<..>`.
- After ADR 0100, `OR` is the only const parameter left on
  `Environment`, at 241 lines in 55 files.
- Ten calls need `<T as Observation<3>>::shape()` to name the rank.
- A user names an agent as
  `DqnAgent<MlpQNetwork, CartPoleObservation, CartPoleAction, 1, 2>`, where
  `1` restates what `CartPoleObservation` already says and `2` must be
  `1 + 1`.
- `HostRow`'s documentation carries an invariant, one rank per type, whose
  breach is an ambiguity error (E0283) at every unqualified call.

## Decision

### 1. `Observation` states its shape as a constant slice

```rust
pub trait Observation: Debug + Clone + Send + Sync {
    /// The size of each axis. Every entry is greater than zero.
    const SHAPE: &'static [usize];
}

pub trait HostRow: Observation {
    fn write_host_row(&self, buf: &mut Vec<f32>);
    fn row_is_finite(&self, scratch: &mut Vec<f32>) -> bool { /* provided */ }
}
```

`write_host_row` pushes exactly `SHAPE.iter().product()` values, as before.
The rank is `SHAPE.len()`.

It is a constant and not a function because Decision 3 reads it in a
`const` block. A type whose size is a const generic of its own states it
directly: `const SHAPE: &'static [usize] = &[C];`.

### 2. `Environment` has no const parameter

```rust
pub trait Environment {
    type ObservationType: Observation;
    type ActionType: Action;
    // reset, step: unchanged
}
```

### 3. The batched rank belongs to the model, and the agent asserts the relation

An agent keeps `BOR`, the rank of the tensor its model takes, and loses
`OR`:

```rust
pub struct DqnAgent<M, O, A, const BOR: usize>
where
    M: DqnModel<BOR>,
    O: HostRow,
    A: DiscreteAction,
```

Each agent's constructor states the relation the convention stated:

```rust
const {
    assert!(
        O::SHAPE.len() + 1 == BOR,
        "the model's input rank must be the observation's rank plus one"
    )
};
```

A mismatch is an error when the crate is built, naming the instantiation.
`cargo check` does not report it: the assertion is evaluated when the
function is instantiated, which `check` does not do. `cargo build` and
`cargo test` do.

### 4. Agents stage rows through one function

`rlevo-reinforcement-learning` gets one crate-private function that stages
`n` rows into one host buffer and uploads `[n, ..O::SHAPE]` at rank `BOR`.
The act path calls it with one row, the learn path with a batch. It
replaces the 21 `to_tensor`-then-`unsqueeze` sites and the ten hand-written
staging blocks. It is not public: ADR 0081 asks a `pub` item for a consumer
outside its crate, and this has none.

### 5. `to_tensor` takes its rank on the method

```rust
#[cfg(feature = "burn")]
pub trait TensorConvertible: HostRow {
    fn to_tensor<const R: usize>(&self, device: &Device) -> Tensor<R> {
        const { assert!(Self::SHAPE.len() == R) };
        /* stage one row, upload once: unchanged */
    }
}
```

A caller that annotates the result (`let t: Tensor<1> = obs.to_tensor(&d)`)
writes what it writes today. After Decision 4 no agent calls it; see "Open
before landing".

### 6. `docs/rules.md` and the companion docs change in the same pull request

- The rank convention becomes: no rank parameter on a domain trait; `BOR`
  is the model's input rank; an agent asserts `O::SHAPE.len() + 1 == BOR`.
- "Dimension methods: `shape()`" becomes the constant `SHAPE`.
- The `Observation` invariant row loses "type-enforced by the `[usize; R]`
  return"; the `HostRow` row loses "at exactly one rank".
- `CLAUDE.md` loses the `OR` row of its generic-symbol table and the "Const
  Generics and Type Inference" section, which has no parameter left to
  describe.
- The probe templates and the `rlevo-algo` seams change with the API.

## Consequences

### What an implementor and a user stop writing

```rust
// before
impl Observation<2> for TankObs {
    fn shape() -> [usize; 2] { [TANKS, FEATURES] }
}
impl HostRow<2> for TankObs { /* ... */ }
impl Environment<2> for TankFarm { /* ... */ }
type Agent = DqnAgent<MlpQNetwork, CartPoleObservation, CartPoleAction, 1, 2>;

// after
impl Observation for TankObs {
    const SHAPE: &'static [usize] = &[TANKS, FEATURES];
}
impl HostRow for TankObs { /* ... */ }
impl Environment for TankFarm { /* ... */ }
type Agent = DqnAgent<MlpQNetwork, CartPoleObservation, CartPoleAction, 2>;
```

### The checks, before and after

| Property | Today | After |
|----------|-------|-------|
| `shape()` length equals the declared rank | compiler | no declared rank to disagree with |
| `to_tensor` rank equals the shape's rank | compiler | build-time assertion |
| `BOR == rank + 1` | unchecked | build-time assertion |
| the model's input rank suits the observation | unchecked | build-time assertion, same one |

### Positive

- `Environment` is a trait with two associated types and no parameters.
  Every evaluator, wrapper, tap and helper generic over an environment loses
  its rank parameters (`evaluator.rs`, `env_tap.rs`, `time_limit.rs`,
  `episode.rs`, the test-support baselines).
- Each agent loses a parameter, and the pairing of model and observation is
  checked where it was a convention.
- The batch layout is written once.
- `HostRow`'s one-rank invariant and the turbofish it forced are gone.

### Negative

- **Breaking for every implementor and every generic caller**: 45
  `Observation` impls, 30 `HostRow` impls, every `Environment` line, every
  named agent type. Mechanical, and a 0.7 change.
- **The new assertions are invisible to `cargo check`**, and so to an
  editor that runs it. They appear at `cargo build` and `cargo test`. CI
  builds, so CI reports them.
- **A bound can no longer require a rank.** `rlevo-hybrid`'s
  `rollout_fitness` bounds `E: Environment<1, 1>` and documents it as "rank
  fixing". Without the parameter it accepts any environment. If the code
  depends on rank 1, it asserts `SHAPE.len() == 1` the same way; if it only
  flattens a row, the restriction was never needed. Which it is has not been
  read.
- **`SHAPE` cannot be computed at runtime.** Neither could `shape()`, which
  took no `self`.

### Neutral

- `BOR` stays, at 183 lines in 15 files. It is the model's parameter and
  was never the observation's.
- `rlevo-core` still depends on Burn only under its `burn` feature.

### Surface delta (ADR 0081)

Removed: `Observation::shape`. Added: `Observation::SHAPE`, which replaces
it. Changed arity: `Observation`, `HostRow`, `TensorConvertible`,
`Environment`, and every agent struct (`DqnAgent`, `C51Agent`,
`QrDqnAgent`, `PpoAgent`, `PpgAgent`, `DdpgAgent`, `Td3Agent`, `SacAgent`).
Changed signature: `TensorConvertible::to_tensor`. The listing from
`tools/public-surface.sh` is the measure; this is the expectation.

## What was checked

A standalone crate on the pinned toolchain (1.99.0), not the workspace:

- `const SHAPE: &'static [usize]` implemented for a fixed type, for a type
  with a const generic (`&[C]`) and from named constants
  (`&[SIDE, SIDE, 3]`).
- An agent generic over `O: Observation` and `const BOR: usize` building
  `[n, ..O::SHAPE]` as `[usize; BOR]`. Ranks 1 and 3 ran and printed
  `[32, 4]`, `[32, 7]`, `[32, 20, 20, 3]`.
- The assertion of Decision 3 with a rank-3 observation and `BOR = 2`:
  `cargo check` exits 0, `cargo build` fails with E0080 and a note naming
  the instantiation.

- The assertion of Decision 5 in a provided method with a method-level
  const parameter, reached through a blanket impl over `HostRow`: the
  matching rank runs, the wrong one fails `cargo build` with E0080 naming
  `<V as TensorConvertible>::to_tensor::<2>`. The probe returned an array
  and a `Vec`, not a Burn tensor.

That probe was written before the workspace change and checked nothing in
the workspace or against Burn. The workspace change was checked as it
landed:

- Clippy with `-D warnings` is clean with `--all-features`, on each library
  alone, on `rlevo-core` with and without `burn`, and on the harness's
  `record` and `report` and the environments' `record` gates.
- `cargo xtask test-fast`, the workspace doctests, rustdoc with
  `-D warnings` and every probe pass.
- A `compile_fail` doctest on `DqnAgent::new` builds the agent at `BOR = 3`
  over a rank-1 observation and expects E0080. The same doctest with
  `BOR = 2` compiles, so the failure is the assertion of Decision 3.

## Alternatives considered

- **Keep the rank.** The status quo keeps two compiler checks on a number
  an observation states about itself and leaves the two relations a user
  can get wrong to a convention.
- **`fn shape() -> &'static [usize]`.** The smallest edit per impl, and the
  same spelling as `BoundedAction::low`. A function cannot be read in a
  `const` block, so the relation of Decision 3 would be a runtime check in
  each constructor, returned as a `ConfigError`. That is a real option: it
  is visible to `cargo check` users at first run and needs no
  instantiation-time evaluation. It was not chosen because the constant
  gives the earlier error and costs an implementor nothing more.
- **An associated `const RANK: usize` with `[usize; Self::RANK]`.** Needs
  `generic_const_exprs`.
- **Keep `OR` on agents only.** Removes the rank from the contract and
  leaves the agents as they are, with the unchecked `BOR`. It keeps the
  part of the cost users see.
- **Delete `TensorConvertible`.** See below.

## Out of scope

- A shape that belongs to an environment instance, so an environment sized
  at runtime can state it. That is a design for spaces. `SHAPE` neither
  helps nor hinders it.
- The model traits' own parameter. `MlpQNetwork` implements `DqnModel<BOR>`
  at every `BOR` by flattening, so for it `BOR` carries no information
  either. Whether a model should take a flat batch and reshape is a
  separate question about the network seam.

## Landing

- **Order.** ADR 0100 and this ADR land in one change, so an implementor
  meets one break and `Environment` changes arity once.
- **Decision 4, as landed.** The chokepoint is four crate-private functions
  in `algorithms/shared.rs`: `row_len::<O>()`, `assert_batch_rank::<O,
  BOR>()`, `upload_rows::<O, BOR>` for a staged batch and `upload_one::<O,
  BOR>` for the act and value paths. Every agent uploads through them. The
  learn loops still write each row into the host buffer themselves, because
  they interleave observations, next observations, actions and rewards in
  one pass; the layout and the upload are in one place.
- **`rlevo-hybrid`.** `RolloutFitness` depends on no observation rank, so it
  gets no assertion and no restriction; its documentation describes the
  shape instead.
- **Surface delta, measured.** `tools/public-surface.sh` on eight crates,
  at `af3ef1cf` and with this change, moves only `rlevo-core`:
  `Observation::shape` is removed and `Observation::SHAPE` added. It also
  lists `DiscreteAction::ACTION_COUNT` as added, an item that exists at the
  base without a recorded path. The listing records paths, not generic
  parameters, so the arity changes under "Surface delta" do not appear in
  it, and that list stands as the record.

## Open

- **Does `TensorConvertible` keep a consumer?** After Decision 4 its
  callers in the workspace are benches and `rlevo-core`'s own tests. It is
  the only item behind `rlevo-core`'s `burn` feature, so deleting it
  removes the feature and makes the core Burn-free unconditionally, which
  would amend ADR 0089 Decision 1. Against that: an outside author writing
  a Burn agent on `rlevo-core` alone uses it to get a tensor. This ADR keeps
  it. The count of outside-style callers (probes, template seams) decides.
