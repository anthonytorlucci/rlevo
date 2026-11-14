---
project: rlevo
status: draft
type: decision
date: 2026-11-14
tags: [adr, draft, action, const-generics, rank, environment-contract, rlevo-core, rlevo-reinforcement-learning, public-surface, breaking-change]
---

# ADR 0100: An action carries no rank

> Draft. The number is taken at landing; until then this is `P-action-rank`.
> Nothing here has been compiled. The counts are searches over `crates/`,
> `xtask/` and `templates/` on `main` at `af3ef1cf`.

## Status

**Proposed (2026-11-14).**

- Supersedes ADR [0053](0053-bounded-action-per-component-bounds.md)
  Decision 8, which kept the rank-per-component action form ("Convention B")
  permitted. This ADR retires it, together with the rank it needs.
- Amends ADR [0090](0090-the-environment-contract-keeps-only-what-generic-code-reads.md).
  The contract it kept names `Action<AR>` and `Environment<OR, AR>`. They
  become `Action` and `Environment<OR>`.
- Leaves ADR [0038](0038-continuous-action-components-const.md)'s
  decision as it is: `ContinuousAction::COMPONENTS` stays the one authority
  for a continuous action's scalar count. Its Context describes three
  quantities (rank, shape, components); after this ADR an action has one.
- Does not decide the observation rank. `Observation<R>` and `HostRow<R>`
  are untouched. See "Out of scope".

## Context

`Action<const R: usize>` has two members:

```rust
pub trait Action<const R: usize>: Debug + Clone + Sized {
    fn shape() -> [usize; R];
    fn is_valid(&self) -> bool;
}
```

`DiscreteAction<R>`, `ContinuousAction<R>` and `BoundedAction<R>` inherit the
parameter, `Environment<OR, AR>` carries it to bound `ActionType: Action<AR>`,
and the continuous-control agents carry it twice, as `AR` and as `BAR`, the
rank of a batched action tensor.

### What reads `Action::shape()`

Four production sites read it. Every other call is a test assertion of a
literal (`assert_eq!(ReacherAction::shape(), [2])`) or of a rank
(`CarRacingAction::shape().len() == 1`). The four are:

| Site | What it does with the shape |
|------|------------------------------|
| `algorithms/shared.rs`, `action_bound_tensors` | asserts `shape().product() == COMPONENTS`, then builds `[1, ..shape]` |
| `ddpg_agent.rs`, learn step | builds `[batch, ..shape]` |
| `td3_agent.rs`, learn step | builds `[batch, ..shape]` |
| `sac_agent.rs`, learn step | builds `[batch, ..shape]` |

No discrete agent reads it. DQN, C51, QR-DQN and the categorical PPO head use
`DiscreteAction::ACTION_COUNT` and `to_index`. A discrete action's `shape()`
is `[ACTION_COUNT]`, the same number stated a second time.

The continuous agents fill the batched tensor from the replay's flat
`COMPONENTS`-long slice, so the shape they read must multiply out to
`COMPONENTS`, and `action_bound_tensors` panics when it does not. The shape
can therefore say only one thing those agents accept: `[COMPONENTS]`.

### What the rank is, in every implementor

- 42 `Action` impls. 41 are rank 1.
- The one that is not is `ContinuousActionTest`, a test fixture in
  `rlevo-core/src/action.rs`: `Action<3>`, three `f32` fields,
  `shape() == [1, 1, 1]`. This is the "Convention B" of ADR 0038. Its shape
  multiplies out to 1, not 3, so DDPG, TD3 and SAC refuse it by panic.
- Every implementor of `DeterministicPolicy`, `ContinuousQ` and
  `SquashedGaussianPolicy` in the workspace, shipped or test, has `BAR = 2`:
  ten impls, none at another value.

### What the rank checks

Nothing at compile time beyond the length of the array `shape()` returns.
`BAR = AR + 1` is stated in the agents' docs and in `docs/rules.md` and is
not expressible on stable Rust, so the two are independent parameters. A
wrong `BAR` is a runtime `TensorData` shape error.

### What the rank costs

- 21 `const AR: usize` declarations in 9 files, and 106 lines naming `BAR`
  in 7.
- 51 trait bounds outside `impl` headers that spell an action trait with its
  rank (`A: BoundedAction<AR>`, `A: DiscreteAction<1>`).
- 241 lines in 55 files spell `Environment<..>` with two parameters.
- Every environment author writes `impl Action<1> for X` and a `fn shape`
  whose value nothing they call will read. The `rlevo-algo` environment seam
  and both `byoe-1` probes do.
- `docs/rules.md` spends three convention bullets and three invariant rows
  on keeping `AR`, `BAR` and `COMPONENTS` apart. ADR 0038 and ADR 0053 both
  exist because code confused them.

ADR 0053 Decision 8 kept Convention B because retiring it was "a breaking
change to an in-trait test type bought with no correctness gain". That weighed
the cost of the fixture against nothing. The gain is the parameter: with
Convention B gone, no action in the workspace needs a rank other than the
one its component count implies.

## Decision

### 1. `Action` has no rank and no shape

```rust
pub trait Action: Debug + Clone + Sized {
    fn is_valid(&self) -> bool;
}

pub trait DiscreteAction: Action {
    const ACTION_COUNT: usize;
    fn from_index(index: usize) -> Self;
    fn to_index(&self) -> usize;
}

pub trait ContinuousAction: Action {
    const COMPONENTS: usize;
    fn as_slice(&self) -> &[f32];
    fn from_slice(values: &[f32]) -> Self;
}

pub trait BoundedAction: ContinuousAction {
    fn low() -> &'static [f32];
    fn high() -> &'static [f32];
}
```

`Action::shape` is deleted, not replaced. The size of a discrete action space
is `ACTION_COUNT`; the size of a continuous one is `COMPONENTS`. Neither
changes.

### 2. `Environment` has one const parameter

```rust
pub trait Environment<const OR: usize> {
    type ObservationType: Observation<OR>;
    type ActionType: Action;
    // reset, step: unchanged
}
```

### 3. A batch of continuous actions is `[batch, COMPONENTS]`

The rank is 2 and is written as 2:

```rust
pub trait DeterministicPolicy<const BOR: usize>: Module {
    fn forward(&self, obs: Tensor<BOR>) -> Tensor<2>;
}

pub trait ContinuousQ<const BOR: usize>: Module {
    fn forward(&self, obs: Tensor<BOR>, act: Tensor<2>) -> Tensor<1>;
}
```

`SquashedGaussianPolicy` changes the same way. `DdpgAgent`, `Td3Agent` and
`SacAgent` lose `AR` and `BAR`. `action_bound_tensors` returns
`[1, COMPONENTS]` tensors and loses its panic, because the product it
checked no longer exists to disagree.

A network that wants its action in another layout reshapes inside `forward`.

### 4. Convention B is retired

`ContinuousActionTest` becomes an ordinary three-component action. Nothing
else in the workspace used the form.

### 5. A multi-discrete action brings its own cardinalities

ADR 0090 Decision 6 deleted `MultiDiscreteAction` and said it returns with
the chess move encoding. When it does, the per-axis cardinalities are a
member of that trait. They are not a reason to keep `shape()` on every
action: the one trait that has several axes states them, and the 41 actions
that have one do not.

### 6. `docs/rules.md` and the companion docs change in the same pull request

- The convention bullets become: one rank parameter, `OR`; one batch rank,
  `BOR = OR + 1`; domain bounds `O: Observation<OR>`, `A: Action`.
- The invariant rows for `ContinuousAction` and `BoundedAction` drop their
  "`AR` is the tensor rank, never the component count" and "(**not** `AR`)"
  clauses. There is no `AR` left to confuse with `COMPONENTS`.
- `CLAUDE.md`'s generic-symbol table and "Const Generics and Type Inference"
  section lose `AR`.
- The probe templates and the `rlevo-algo` seams change with the API, as the
  probe rule requires.

## Consequences

### What an implementor stops writing

```rust
// before
impl Action<1> for PumpCommand {
    fn shape() -> [usize; 1] { [2] }
    fn is_valid(&self) -> bool { /* ... */ }
}
impl ContinuousAction<1> for PumpCommand { /* ... */ }
impl Environment<2, 1> for TankFarm { /* ... */ }

// after
impl Action for PumpCommand {
    fn is_valid(&self) -> bool { /* ... */ }
}
impl ContinuousAction for PumpCommand { /* ... */ }
impl Environment<2> for TankFarm { /* ... */ }
```

### Positive

- One const parameter leaves `Environment`, and with it every evaluator,
  wrapper and helper that is generic over an environment.
- Two leave each continuous-control agent; one leaves each of three model
  traits.
- The rank-versus-components confusion behind ADR 0038 and ADR 0053 has no
  second quantity left to be confused with.
- A runtime panic (`action_bound_tensors`) is removed, not moved.

### Negative

- **Breaking for every implementor and every generic caller.** 42 `Action`
  impls, 21 `DiscreteAction`, 16 `ContinuousAction`, 14 `BoundedAction`, and
  the 241 `Environment<..>` lines. The edit is mechanical: delete a
  parameter, delete `fn shape`. It is a 0.7 change.
- **An action with real spatial structure flattens.** An action that is a
  field over a grid would be a `COMPONENTS`-long slice that its network
  reshapes. No such action exists in the workspace, and the continuous
  agents already require the flat layout.
- **The test assertions on a literal `Action::shape()` are deleted.** They
  pinned a value nothing read. A test that asserts only that goes with
  them; the count is taken when the change is made.

### Neutral

- No compile-time check is lost. The only one the rank gave was the length
  of the `shape()` array.
- Replay storage is unchanged: it already holds a continuous action as its
  flat slice.

### Surface delta (ADR 0081)

Removed: `Action::shape`. Changed arity: `Action`, `DiscreteAction`,
`ContinuousAction`, `BoundedAction`, `Environment`, `DeterministicPolicy`,
`ContinuousQ`, `SquashedGaussianPolicy`, `DdpgAgent`, `Td3Agent`, `SacAgent`,
and the `rlevo-test-support` baselines (`uniform_discrete`,
`uniform_bounded`, `random_return`). Added: nothing. The listing from
`tools/public-surface.sh` at base and on the branch is the measure; this
list is the expectation it is checked against.

## Alternatives considered

- **Keep the rank and document it** (ADR 0053 Decision 8). This is the
  status quo. It keeps the parameter at every site counted above for one
  test fixture.
- **Keep `shape()` without the rank**, as `fn shape() -> &'static [usize]`.
  Removes the parameter and keeps a member with no reader whose only
  accepted value is `[COMPONENTS]` or `[ACTION_COUNT]`.
- **Make the rank an associated const**, `const RANK: usize` with
  `[usize; Self::RANK]`. Needs `generic_const_exprs`, which is unstable.
  ADR 0090 Decision 7 also deleted the `RANK` constants as members nothing
  called.
- **Drop the observation rank in the same change.** The two are different
  trades. The observation rank has live readers: every agent lays out
  `[batch, ..Observation::shape()]` and nine observation impls are rank 2
  or 3. Dropping it changes `HostRow`, `TensorConvertible::to_tensor` and
  where a rank mismatch is reported. It deserves its own record.

## Out of scope

- `Observation<R>`, `HostRow<R>`, `TensorConvertible<R>`, and `BOR`.
- Whether shape is a property of a type or of an environment instance (an
  environment sized at runtime cannot state a static shape). That is a
  question about spaces, and it applies to observations as well.

## Open before landing

- Compile it. Every count above is a search result, and a search finds
  spellings, not uses. In particular, confirm no downstream code outside the
  four production sites reaches `Action::shape` through a generic bound.
- Lint the feature-gated combinations as well as `--all-features`, and
  check `rlevo-core --lib` alone.
- Decide whether this lands before or with the observation-rank decision.
  Landing first means `Environment<OR, AR>` becomes `Environment<OR>` and
  may change again; landing together means one break for implementors.
