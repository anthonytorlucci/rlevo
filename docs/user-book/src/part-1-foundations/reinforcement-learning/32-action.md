# Action Spaces

The [previous chapter](31-state.md) covered the agent's *input* — what it
perceives and how that perception becomes a tensor. This chapter covers the
*output*: the actions it can take. The same rank-and-shape discipline returns,
but the action side has its own wrinkle. "An action" can mean three quite
different things — pick one of \\(n\\) buttons, pick several categorical options
at once, or output a vector of real numbers — and the algorithm that consumes
your policy needs to know which.

That ambiguity is the spine of the chapter, and `rlevo` resolves it with a
**layered** trait hierarchy: one minimal base trait every action shares, then
exactly one of three flavour traits that pins down *which* kind of action it is —
the distinction a DQN or a continuous-control actor reads off the type to know how
to produce a value. We build the base first, then each flavour in turn, and close
with the tensor bridge that carries actions back out to the environment.

## The base `Action` trait

Every action implements one minimal, framework-agnostic trait from
`rlevo::core::base`:

```rust
pub trait Action<const R: usize>: Debug + Clone + Sized {
    const RANK: usize = R;
    fn shape() -> [usize; R];
    fn is_valid(&self) -> bool;
}
```

This is deliberately spare. `Debug` for logging, `Clone` because actions get
stored in replay buffers and reused, `Sized` for stack allocation, and the same
`R`/`shape()` pair you met for states — `R` is the **rank** (number of axes),
`shape()` gives the cardinality of each axis. `is_valid()` checks the action's
own structural invariants and *not* environment-specific legality: whether a
chess move is legal in the current position is the environment's job, not the
action type's.

> **One asymmetry worth noting.** Unlike `Observation`, the base `Action` trait
> does *not* require `Serialize`/`Deserialize`. The minimal contract stays
> framework-agnostic; concrete action types that need to land in a replay buffer
> simply derive `serde` themselves (as `CartPoleAction` and `PendulumAction`
> do). The bound isn't forced because not every consumer needs it.

On its own, `Action` doesn't tell an algorithm *how* to produce a value — a DQN
needs an integer index to `argmax` over, a continuous-control actor needs a
float vector to emit. That "how" lives in three extension traits.

## Three flavours of action space

`rlevo::core::action` layers three type-specific traits on top of the base
`Action`, plus one refinement for bounded continuous spaces:

| Trait | Use it when the action is… | Bridge to the policy |
| ----- | -------------------------- | -------------------- |
| `DiscreteAction<R>` | one of \\(n\\) mutually exclusive choices | an index — `argmax` of Q-values, or a categorical sample |
| `MultiDiscreteAction<R>` | several independent categorical choices at once | one index per axis |
| `ContinuousAction<R>` | a real-valued vector | the raw output of an actor network |
| `BoundedAction<R>` | a continuous action with known `[low, high]` per component | lets you scale/clip network outputs and sample warm-up actions |

You implement `Action` always, then exactly one of the first three (and
optionally `BoundedAction` on top of `ContinuousAction`). The algorithm picks
the right one through trait bounds, so a discrete-only method like DQN simply
won't compile against a continuous action type — the mismatch is caught at the
type level rather than producing nonsense at runtime.

## Discrete actions

The most common case: a finite set of mutually exclusive choices, indexed
\\(0 \ldots n-1\\).

```rust
pub trait DiscreteAction<const R: usize>: Action<R> {
    const ACTION_COUNT: usize;
    fn from_index(index: usize) -> Self;   // index → action
    fn to_index(&self) -> usize;           // action → index
    fn random() -> Self { /* uniform over 0..ACTION_COUNT */ }
    fn enumerate() -> Vec<Self> { /* all actions, in index order */ }
}
```

`ACTION_COUNT` is the cardinality. `from_index`/`to_index` are the load-bearing
pair, and the contract is a **round-trip law** in both directions:

```math
\forall i \in [0, n):\quad i = \texttt{from\_index}(i).\texttt{to\_index}()
\qquad
\forall a:\quad a = \texttt{from\_index}(a.\texttt{to\_index}())
```

That law is what makes the index a safe currency between your enum and a neural
network. A DQN's output layer has `ACTION_COUNT` units; you take the `argmax` to
get an index, then `from_index` to recover the typed action. Going the other
way, a stored action becomes `to_index` for a one-hot target. `random()` gives
you ε-greedy exploration for free, and `enumerate()` hands tabular Q-learning
the whole action set.

CartPole's two-button action is the minimal example:

```rust
pub enum CartPoleAction { Left, Right }

impl DiscreteAction<1> for CartPoleAction {
    const ACTION_COUNT: usize = 2;

    fn from_index(index: usize) -> Self {
        match index { 0 => Self::Left, 1 => Self::Right, _ => panic!("out of bounds") }
    }
    fn to_index(&self) -> usize {
        match self { Self::Left => 0, Self::Right => 1 }
    }
}
```

Note `from_index` panics on an out-of-bounds index. That is intentional: an
index outside `[0, ACTION_COUNT)` is a programming error (a network configured
with the wrong number of outputs, say), not a recoverable runtime condition.

## Multi-discrete actions

Sometimes the agent makes several *independent* categorical choices in one
step — pick a direction *and* an intensity, select a unit *and* a target. That
is `MultiDiscreteAction`, where each axis has its own cardinality and the rank
`R` is the number of axes:

```rust
// shape() = [4, 3]: four directions × three intensity levels
impl MultiDiscreteAction<2> for MoveAction {
    fn from_indices(indices: [usize; 2]) -> Self { /* ... */ }
    fn to_indices(&self) -> [usize; 2] { /* ... */ }
}
```

The total number of distinct actions is the product of the axis sizes:

```math
\text{total actions} = \prod_{i=0}^{R-1} \texttt{shape}()[i]
```

That product is exactly why `enumerate()` carries a loud warning. `[10, 10, 10]`
is a thousand actions — fine. `[100, 100, 100]` is a million — almost certainly
not. The whole point of *multi*-discrete is to avoid flattening into one giant
discrete head: you keep the choices factored so a policy can have one small
output head per axis instead of one enormous head over the cartesian product.

## Continuous actions

For motor control — torques, steering, joint angles — the action is a vector of
reals, produced directly by an actor network:

```rust
pub trait ContinuousAction<const R: usize>: Action<R> {
    const COMPONENTS: usize;               // scalar count == as_slice().len()
    fn as_slice(&self) -> &[f32];          // action → components
    fn from_slice(values: &[f32]) -> Self; // components → action
    fn clip(&self, min: f32, max: f32) -> Self;
    fn random() -> Self { /* uniform in [-1, 1) per component */ }
}
```

`as_slice`/`from_slice` are the continuous analogue of `to_index`/`from_index`:
the bridge between your typed action and the flat `f32` vector a network emits.
`clip` matters more than it looks — an actor with a `tanh` head can still drift
slightly out of range under exploration noise or numerical error, and `clip`
pulls it back into the valid box before the environment ever sees it.

`COMPONENTS` deserves a word, because it is easy to confuse with the rank `R`.
The two are genuinely different: `R` is the *tensor rank* — the number of axes —
while `COMPONENTS` is the number of scalars you flatten to, exactly the length
`as_slice` returns and `from_slice` consumes. A four-motor walker is a single
rank-1 vector (`R == 1`) carrying four components; `shape()` gives \\([4]\\), not
\\([1]\\). We make you declare `COMPONENTS` explicitly — there is deliberately no
default — so the built-in `random()` samples the right number of values. Keying
it off `R` instead would sample one value and hand it to a `from_slice` expecting
four, panicking on every call; requiring the constant makes that trap
unreachable. See [ADR 0038](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0038-continuous-action-components-const.md)
for the full rationale. The default `random()` draws each component uniformly from
\\([-1, 1)\\), so if your action space is asymmetric — CarRacing's throttle and
brake live in \\([0, 1]\\) — override `random()` to sample within the true bounds.

When the bounds are known, layer on `BoundedAction`:

```rust
pub trait BoundedAction<const R: usize>: ContinuousAction<R> {
    fn low()  -> &'static [f32];
    fn high() -> &'static [f32];
}
```

This exists because continuous-control algorithms (DDPG, TD3, SAC) genuinely
need the per-component bounds: to rescale a `[-1, 1]` network output into the
environment's real range, to clip a target action, and to sample uniform
warm-up actions before the policy has learned anything. The invariant is
`low().len() == high().len() == COMPONENTS` and `low()[i] < high()[i]` for
every component `i`, and `clip` must be a no-op on an action already inside
`[low, high]`.

Notice the return type is a slice, not `[f32; R]`. That is deliberate, and it
is the same rank-vs-component distinction `COMPONENTS` drew above, one level
up: `R` names the `ContinuousAction<R>` supertrait but never appears in either
signature, because keying the bounds on rank instead of `COMPONENTS` was
exactly the bug ([issue #253](https://github.com/anthonytorlucci/rlevo/issues/253),
the same conflation as [issue #100](https://github.com/anthonytorlucci/rlevo/issues/100))
— a rank-1 walker action with four components would only ever get one bound
pair back. `&'static [f32]` sidesteps `[f32; Self::COMPONENTS]`, which needs
the unstable `generic_const_exprs` feature and isn't available on the stable
toolchain this workspace pins; see
[ADR 0053](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0053-bounded-action-per-component-bounds.md)
for the full design rationale, including why the length agreement is a
contract test rather than something the compiler checks for you.

Pendulum is a clean single-component example — one torque value bounded to
\\([-2, 2]\\):

```rust
pub struct PendulumAction(f32);

impl ContinuousAction<1> for PendulumAction {
    const COMPONENTS: usize = 1;
    fn as_slice(&self) -> &[f32] { std::slice::from_ref(&self.0) }
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 1, "PendulumAction expects a 1-element slice");
        Self::unchecked(values[0])
    }
    fn clip(&self, min: f32, max: f32) -> Self { Self::unchecked(self.0.clamp(min, max)) }
}

impl BoundedAction<1> for PendulumAction {
    fn low()  -> &'static [f32] { &[-2.0] }
    fn high() -> &'static [f32] { &[2.0] }
}
```

CarRacing is the case that motivates the whole design: three components —
`[steer, gas, brake]` — sharing one rank-1 `Action<1>`, with bounds that are
*not* symmetric across components. Steering ranges over \\([-1, 1]\\), but gas
and brake only make sense as \\([0, 1]\\); a scalar `low()`/`high()` pair could
never express that gas has a different floor than steering.

```rust
impl ContinuousAction<1> for CarRacingAction {
    const COMPONENTS: usize = 3;
    fn as_slice(&self) -> &[f32] { &self.components }
    // ...
}

impl BoundedAction<1> for CarRacingAction {
    fn low()  -> &'static [f32] { &[-1.0, 0.0, 0.0] }
    fn high() -> &'static [f32] { &[1.0, 1.0, 1.0] }
}
```

## From policy output to action: `HostRow` and `TensorConvertible`

Just as observations cross into tensor land, actions cross back out of it via
the same `HostRow<R>` / `TensorConvertible<R, B>` pair from the
[state chapter](31-state.md#crossing-into-tensor-land-hostrow-and-tensorconvertible)
— and the encoding differs by flavour.

**Discrete actions** become a one-hot row on the way in, and decode by
`argmax` on the way out. As in the state chapter, you write the host-side
`f32` row on `HostRow` (`row_shape` + `write_host_row`), and the
`TensorConvertible`-provided `to_tensor` derives the tensor from it. CartPole's
action does exactly this:

```rust
impl HostRow<1> for CartPoleAction {
    fn row_shape() -> [usize; 1] {
        [2]
    }
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        let mut one_hot = [0.0_f32; 2];
        one_hot[self.to_index()] = 1.0;          // index drives the hot slot
        buf.extend_from_slice(&one_hot);
    }
}

impl<B: Backend> TensorConvertible<1, B> for CartPoleAction {
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        // shape-check elided; recover the index via argmax, then from_index
        // ...
        Ok(Self::from_index(argmax_index))
    }
}
```

**Continuous actions** convert straight through their component slice — no
one-hot, the floats *are* the representation. In both cases the round-trip
invariant from the state chapter still holds: `from_tensor(a.to_tensor(dev))`
must give back `a`.

## Validation and errors

`is_valid()` is the structural gate — for Pendulum, "finite and within
\\([-2, 2]\\)". As with states, a robust pattern is to make illegal actions
unrepresentable by validating at construction and returning a `Result`:

```rust
impl PendulumAction {
    pub fn new(torque: f32) -> Result<Self, InvalidActionError> {
        if torque.is_finite() && torque.abs() <= 2.0 {
            Ok(Self(torque))
        } else {
            Err(InvalidActionError {
                message: format!("torque {torque} outside [-2.0, 2.0] or non-finite"),
            })
        }
    }
}
```

`rlevo::core::action` provides `InvalidActionError` (a message-carrying struct
implementing `std::error::Error`) for exactly these failures — an out-of-bounds
index, a non-finite component, a slice of the wrong length.

## Putting it together

The action side mirrors the state side, with the data flowing the other way:

- Every action implements the minimal **`Action`** trait — rank, shape,
  validity.
- It then implements one of **`DiscreteAction`**, **`MultiDiscreteAction`**, or
  **`ContinuousAction`** (plus optional **`BoundedAction`**), which tells an
  algorithm how to *produce* it — an index, a tuple of indices, or a float
  vector.
- **`HostRow`/`TensorConvertible`** bridge policy outputs and typed actions —
  one-hot / `argmax` for discrete, the component slice for continuous — with
  the same round-trip guarantee.
- **`is_valid()`** and **`InvalidActionError`** keep actions structurally honest.

With states and actions in hand, we have both halves of the agent's interface.
The next step is to see how the **`Environment`** trait stitches them together
into the step-by-step loop — turning an action into a reward and the next
observation.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
