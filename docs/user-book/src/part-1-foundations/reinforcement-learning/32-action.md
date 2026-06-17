# Action Spaces

The [previous chapter](31-state.md) covered the agent's *input* — what it
perceives and how that perception becomes a tensor. This chapter covers the
*output*: the actions it can take. The same rank-and-shape discipline returns,
but the action side has its own wrinkle. "An action" can mean three quite
different things — pick one of \\(n\\) buttons, pick several categorical options
at once, or output a vector of real numbers — and the algorithm that consumes
your policy needs to know which. `rlevo` encodes that distinction in a small
layered trait hierarchy.

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

When the bounds are known, layer on `BoundedAction`:

```rust
pub trait BoundedAction<const R: usize>: ContinuousAction<R> {
    fn low()  -> [f32; R];
    fn high() -> [f32; R];
}
```

This exists because continuous-control algorithms (DDPG, TD3, SAC) genuinely
need the per-component bounds: to rescale a `[-1, 1]` network output into the
environment's real range, and to sample uniform warm-up actions before the
policy has learned anything. The invariant is `low()[i] < high()[i]`, and `clip`
must be a no-op on an action already inside `[low, high]`.

Pendulum is a clean single-axis example — one torque value bounded to
\\([-2, 2]\\):

```rust
pub struct PendulumAction(f32);

impl ContinuousAction<1> for PendulumAction {
    fn as_slice(&self) -> &[f32] { std::slice::from_ref(&self.0) }
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 1, "PendulumAction expects a 1-element slice");
        Self::unchecked(values[0])
    }
    fn clip(&self, min: f32, max: f32) -> Self { Self::unchecked(self.0.clamp(min, max)) }
}

impl BoundedAction<1> for PendulumAction {
    fn low()  -> [f32; 1] { [-2.0] }
    fn high() -> [f32; 1] { [ 2.0] }
}
```

## From policy output to action: `TensorConvertible`

Just as observations cross into tensor land, actions cross back out of it via
the same `TensorConvertible<R, B>` trait — and the encoding differs by flavour.

**Discrete actions** become a one-hot tensor on the way in, and decode by
`argmax` on the way out. CartPole's action does exactly this:

```rust
impl<B: Backend> TensorConvertible<1, B> for CartPoleAction {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        let mut one_hot = [0.0_f32; 2];
        one_hot[self.to_index()] = 1.0;          // index drives the hot slot
        Tensor::from_floats(one_hot, device)
    }
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
- **`TensorConvertible`** bridges policy outputs and typed actions — one-hot /
  `argmax` for discrete, the component slice for continuous — with the same
  round-trip guarantee.
- **`is_valid()`** and **`InvalidActionError`** keep actions structurally honest.

With states and actions in hand, we have both halves of the agent's interface.
The next step is to see how the **`Environment`** trait stitches them together
into the step-by-step loop — turning an action into a reward and the next
observation.

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
