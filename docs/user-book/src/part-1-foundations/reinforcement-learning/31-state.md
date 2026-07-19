# State and Observation Spaces

The previous chapter set up the agent–environment loop and promised that the
*state* \\(s_t\\) is what makes the whole thing Markov. This chapter is about how
`rlevo` represents that state in actual Rust types — and, just as importantly,
about the line it draws between the **state** (everything the environment knows)
and the **observation** (the slice of it the agent is allowed to see).

That distinction is not pedantry. Conflating the two is one of the most common
ways to accidentally cheat in RL: you hand the agent a feature it could not
possibly have at deployment time, your numbers look great, and then the policy
falls over in the real world. `rlevo` makes the boundary a type-level wall so
you have to cross it on purpose.

So the spine of this chapter is that wall, walked from one side to the other:
the two traits that name each side, the [`Sensor`](https://docs.rs/rlevo-core/latest/rlevo_core/environment/trait.Sensor.html)
seam that crosses it (and quietly decides MDP vs POMDP), the `HostRow`/`TensorConvertible`
bridge that only the *visible* side gets, the validation that keeps both sides
honest, and finally the higher-level traits for the cases where one observation
is not enough.

## Two traits, two homes

The relevant definitions live in `rlevo::core::base`:

```rust
pub trait State<const R: usize>: Debug + Clone + Send + Sync {
    const RANK: usize = R;

    fn shape() -> [usize; R];          // cardinality of each axis
    fn is_valid(&self) -> bool;
    fn numel(&self) -> usize {         // defaults to the product of shape()
        Self::shape().iter().product()
    }
}

pub trait Observation<const R: usize>:
    Debug + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
    const RANK: usize = R;
    fn shape() -> [usize; R];
}
```

Notice what is missing compared with earlier `rlevo` releases: `State` no longer
owns a `type Observation` and no longer has an `observe()` method. That is not an
oversight — it is the point of this chapter. In the POMDP tuple
\\(\langle \mathcal{S}, \mathcal{A}, T, R, \Omega, O \rangle\\), the emission
model \\(O\\) that turns a state into an observation is a property of the
*problem* (the environment), not of a single point \\(s \in \mathcal{S}\\).
`State<R>` now carries only what genuinely belongs to a point in state space:
its rank, `shape()`, `numel()`, and `is_valid()`. We come back to where `O` lives
instead in the next section. (ADR 0047, superseding
[ADR 0019](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0019-observable-projection-trait.md).)

Before anything else, though, we have to be precise about what `R` is — it
appears on both traits, but the two `R`s are independent const generics with no
structural link between them at this level (a state of rank `1` does not imply
its observation is also rank `1`; that link, when it exists, is made by the
trait we meet next).

### `R` is the rank, not the size

`R` is the **rank** of the space — the number of axes, the count of indices you
need to address a single element. It is NumPy's `ndim`, and it maps directly to
the dimensionality parameter Burn calls `D` in `Tensor<B, D>` (`rlevo` names the
same quantity `R` for "rank" in its own traits). It is *not* matrix rank, and it
is *not* how big the space is.

The *size* of each axis is what `shape()` returns: an array of length `R` whose
entries are the cardinalities of the axes.

A few concrete cases make this click:

| Space | `R` (rank) | `shape()` | `numel()` |
| ----- | ---------- | --------- | --------- |
| CartPole's 4 sensors | `1` | `[4]` | `4` |
| A flat 8-dim feature vector | `1` | `[8]` | `8` |
| An \\(84 \times 84\\) greyscale frame | `2` | `[84, 84]` | `7056` |
| A 4-frame stack of RGB images | `3` | `[4, 84, 84]` | `28224` |

`numel()` is the product of the shape — the total scalar count — and it has a
default implementation that computes exactly that. You only override it when
your state uses a non-product layout (the in-tree examples override it purely to
be explicit; the default would give the same answer).

> **Why a const generic?** Encoding the rank in the type means a network built
> for rank-1 observations cannot be silently fed a rank-2 image. The mismatch is
> a compile error, not a runtime shape panic three hours into training. This is
> the same const-generic-dimensions discipline used across `rlevo`'s `Environment`,
> `Action`, `HostRow`, and `TensorConvertible` traits — see the [overview](00-overview.md)
> for the rationale.

## The state/observation split: the `Sensor` seam

If `State` no longer produces an observation, something has to. That something
is [`Sensor`](https://docs.rs/rlevo-core/latest/rlevo_core/environment/trait.Sensor.html),
defined in `rlevo::core::environment` and implemented on the **environment**,
not on the state:

```rust
pub trait Sensor<const OR: usize, const AR: usize, const SR: usize> {
    type Action: Action<AR>;
    type State: State<SR>;
    type Observation: Observation<OR>;

    fn observe(&self, action: &Self::Action, next_state: &Self::State) -> Self::Observation;
    fn observe_reset(&self, state: &Self::State) -> Self::Observation;
}
```

`Sensor` is the seam. It takes an action and the state that resulted from it
and produces what the agent perceives — `observe(a, s')`, the canonical
emission model \\(O\\). `reset()` has no preceding action, so a companion
`observe_reset` produces the very first observation from the initial state
alone; an environment whose observation ignores the action typically forwards
both methods to the same body.

Two things about the signature are worth dwelling on:

- **`&self` is the environment.** A bare state value cannot see the simulator
  world it lives in — no physics geometry to raycast against, no rendered frame
  to sample. Because `Sensor` is implemented on the environment struct, its
  methods have direct access to that world context. This is what let `rlevo`
  retire the old workaround of caching a lidar reading or a rendered frame onto
  the state purely so a `&self`-only `observe()` could return it.
- **The observation rank `OR` is a free parameter.** It is not required to equal
  the state rank `SR`. Most environments' sensors are same-rank projections
  (`OR == SR`), but a sensor is free to observe a compact state through a
  higher-rank modality — the pixel-observation case the
  [Environments chapter](34-environment.md) and
  [Appendix D](../../appendix-d-suppl/tensor-rank-vs-matrix-rank.md) develop
  further.

Whether the sensor preserves all the information needed for the Markov property
or throws some of it away is entirely up to your implementation — and that
single choice decides whether you are modelling an MDP or a POMDP.

**Full observability (an MDP).** When a sensor's `observe`/`observe_reset`
preserve all the information a state carries, state and observation carry the
same content. CartPole is the canonical example — its state is four real numbers
and its observation is the same four numbers:

```rust
pub struct CartPoleState {
    pub x: f32,         // cart position
    pub x_dot: f32,     // cart velocity
    pub theta: f32,     // pole angle
    pub theta_dot: f32, // pole angular velocity
}

impl State<1> for CartPoleState {
    fn shape() -> [usize; 1] { [4] }
    fn numel(&self) -> usize { 4 }

    fn is_valid(&self) -> bool {
        self.x.is_finite() && self.x_dot.is_finite()
            && self.theta.is_finite() && self.theta_dot.is_finite()
    }
}

impl Sensor<1, 1, 1> for CartPole {
    type Action = CartPoleAction;
    type State = CartPoleState;
    type Observation = CartPoleObservation;

    // The observation ignores the action; both methods forward to one
    // shared free function so their bodies cannot drift apart.
    fn observe(&self, _action: &CartPoleAction, next_state: &CartPoleState) -> CartPoleObservation {
        cartpole_observation(next_state)
    }

    fn observe_reset(&self, state: &CartPoleState) -> CartPoleObservation {
        cartpole_observation(state)
    }
}

fn cartpole_observation(state: &CartPoleState) -> CartPoleObservation {
    CartPoleObservation {
        cart_pos:     state.x,
        cart_vel:     state.x_dot,
        pole_angle:   state.theta,
        pole_ang_vel: state.theta_dot,
    }
}
```

Here the sensor is essentially a rename — every state field flows into the
observation. The agent sees everything; the Markov property holds; classical
value-based methods are on solid ground. Note the shared free function: since
the observation ignores the action entirely, both `Sensor` methods forward to
the same body rather than duplicating it — the pattern the trait's own
documentation recommends for sensors that do not care which action produced the
state.

**Partial observability (a POMDP).** Now suppose the sensor dropped the two
velocity fields and returned only `cart_pos` and `pole_angle`. The agent can no
longer tell a pole falling left from one swinging back toward centre — two
genuinely different states produce an identical observation. A single
observation is no longer Markov. You would need to recover the missing dynamics
some other way: stacking consecutive frames, or carrying a recurrent hidden
state (more on that below).

> **The takeaway.** The MDP-vs-POMDP question is not a property of the
> environment in the abstract — it is a property of *what you choose to return
> from your `Sensor`*. `rlevo` makes you write that function, so the choice is
> explicit and reviewable rather than buried in a `gym.make` flag. This is also
> why the choice moved off `State`: full/partial observability is a fact about
> how the *problem* emits observations, not a fact about any one state value.

### Why only `Observation` is serialisable

Look again at the supertraits: `Observation` requires `Serialize` +
`Deserialize`, but `State` does not. That asymmetry is deliberate. Observations
get written into replay buffers, shipped into `EpisodeRecord`s, and replayed
later; they need to round-trip through `serde`. The full state never leaves the
environment, so it carries no serialisation tax. The type system encodes which
side of the wall each value lives on.

## Crossing into tensor land: `HostRow` and `TensorConvertible`

Traits like `State` and `Observation` are deliberately framework-agnostic —
they say nothing about Burn. The bridge to actual tensors (and therefore to
neural networks) is two traits, split along exactly the line you'd expect: the
[`HostRow`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.HostRow.html)
row layout, which never touches a backend, and the device-facing
[`TensorConvertible`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.TensorConvertible.html)
conversion, which does.

```rust
/// Backend-independent row layout. No `B` parameter — row_shape/write_host_row
/// never touch a backend element type.
pub trait HostRow<const R: usize> {
    /// Per-item shape, e.g. `[4]` for CartPole.
    fn row_shape() -> [usize; R];
    /// Append this value's row-major `f32` payload to a host buffer.
    fn write_host_row(&self, buf: &mut Vec<f32>);
}

pub trait TensorConvertible<const R: usize, B: Backend>: HostRow<R> + Sized {
    /// Provided: derived from `HostRow`. Do not override.
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, R> { /* ... */ }
    fn from_tensor(tensor: Tensor<B, R>) -> Result<Self, TensorConversionError>;
}
```

Three things worth noticing.

First, both are implemented on the **observation** (and the action), not on
the state. That is the whole point of the split paying off: the only values
that ever become network inputs are the ones the agent is allowed to see. You
literally cannot `to_tensor()` the hidden state, because nobody implemented
either trait for the state type.

Second, you implement the **host-side row** on `HostRow`, not the tensor
itself. `write_host_row` appends your value's flat `f32` payload to a plain
`Vec<f32>`, and `row_shape` says how to fold that row back into a rank-`R`
tensor. Neither answer depends on which Burn backend the row eventually
uploads to, which is exactly why `HostRow` carries no `B` parameter — a fact
that used to live only in the docs of a single combined trait, and now the
compiler enforces it. `TensorConvertible::to_tensor` is a *provided* method
derived from those two `HostRow` methods — one row, one upload — and
implementations should never override it. The payoff is batching: the free
function `stack_to_tensor` writes N rows into a single host buffer and uploads
the whole `[N, ...]` batch to the device in **one** transfer. It needs only the
`HostRow` bound, not the full `TensorConvertible` — staging is host-only and
never touches a device — and because the single-item and batch paths share the
same row-writer, their layouts cannot drift apart (this is ADR 0028).

Third, the contract is a **round-trip invariant**:
`from_tensor(x.to_tensor(device))` must equal `Ok(x)` for any valid `x`. Replay
buffers and strategies lean on this — a transition stored as a tensor and read
back must be the same transition. CartPole's implementation is about as simple
as it gets:

```rust
impl HostRow<1> for CartPoleObservation {
    fn row_shape() -> [usize; 1] {
        [4]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.to_array());   // [pos, vel, angle, ang_vel]
    }
}

impl<B: Backend> TensorConvertible<1, B> for CartPoleObservation {
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [4] {
            return Err(TensorConversionError {
                message: format!("expected shape [4], got {dims:?}"),
            });
        }
        let v = tensor.into_data().into_vec::<f32>()
            .map_err(|e| TensorConversionError { message: e.to_string() })?;
        Ok(Self { cart_pos: v[0], cart_vel: v[1], pole_angle: v[2], pole_ang_vel: v[3] })
    }
}
```

Note the shape check on the way in: `from_tensor` is the place a malformed
tensor gets caught and turned into a typed `TensorConversionError` rather than a
panic deep inside a matmul.

## Validation: `is_valid` and `StateError`

`is_valid()` checks a value's **structural** invariants — the constraints that
make it a well-formed instance of its own type. It explicitly does *not* check
environment-specific legality (whether a move is legal in the current game
position is the environment's job, not the state's). For CartPole, "well-formed"
just means "all four numbers are finite". For a bounded workspace it means
"inside the box":

```rust
// from crates/rlevo-core/examples/state_constraints.rs
fn is_valid(&self) -> bool {
    self.x_mm >= 0 && self.x_mm <= 1000
        && self.y_mm >= 0 && self.y_mm <= 1000
        && self.theta_mdeg >= -180_000 && self.theta_mdeg <= 180_000
}
```

A common pattern — shown in that example — is to make illegal states
unrepresentable by validating at construction: a `new(...) -> Option<Self>`
that returns `None` when `is_valid()` would be false. The constraint then lives
in one place and the rest of your code can assume any value it holds is valid.

When validation logic needs to *report* what went wrong (during tensor reshaping
or deserialisation, say), `rlevo::core::state` provides a `StateError` enum with
three variants — `InvalidShape { expected, got }`, `InvalidData(String)`, and
`InvalidSize { expected, got }` — covering the three ways a representation can be
malformed: wrong axes, bad contents, wrong element count.

## Beyond a single Markov state

Everything above assumes one observation is enough to act on. When it is not —
POMDPs, recurrent policies, world models — `rlevo::core::state` defines a set of
higher-level traits that extend the base contract. In honest terms: one of these
is load-bearing today and the rest are deliberately-placed seams for algorithms
that are still on the roadmap.

| Trait | Purpose | Status |
| ----- | ------- | ------ |
| `MarkovState` | Declares whether a representation is Markov (`is_markov() -> bool`, default `true`) | Used as a bound on the RL history representation in `rlevo::rl::experience` |
| `HiddenState` | Recurrent agent memory (an RNN/GRU/LSTM `h_t`): `update(obs)` / `reset()` | Defined; for recurrent policies |
| `BeliefState` | A probability distribution over true states, updated by Bayes' rule from `(action, observation)`; carries its own `Observation` associated type (independent of any `State`'s rank, mirroring `HiddenState`/`LatentState`) | Defined; for POMDP belief tracking |
| `LatentState` | A learned compact representation with `encode` / `predict_next` / `decode` | Defined; for world-model agents (DreamerV3-style) |
| `StateAggregation` | Maps concrete states to abstract representatives for function approximation / hierarchical RL | Defined |

The honest status matters more than the trait list: `MarkovState` is wired into
the experience-history machinery, while `BeliefState`, `HiddenState`,
`LatentState`, and `StateAggregation` are currently shapes without consumers —
they exist so the API has somewhere obvious to grow, not because there is a
DreamerV3 hiding in the crate. Treat them as a sketch of where state handling is
headed, and check the [status page](../part-4-open-problems/01-where-rlevo-stands.md)
before building on them.

## Putting it together

The mental model to carry into the rest of the book:

- A **`State`** is the environment's full, Markov account of the world. It stays
  inside the environment and, since ADR 0047, knows nothing about how it is
  observed — no `observe()`, no `type Observation`.
- The environment implements **[`Sensor`](https://docs.rs/rlevo-core/latest/rlevo_core/environment/trait.Sensor.html)**,
  whose `observe(a, s')` / `observe_reset(s)` project a state into an
  **`Observation`** — the only thing the agent sees, and the only thing that
  gets serialised into a replay buffer. `&self` being the environment is what
  lets a sensor read world context a bare state value never could.
- **`HostRow`** owns the backend-independent row layout (`row_shape`,
  `write_host_row`); **`TensorConvertible`** builds on it to turn that
  observation into a Burn tensor for the network, with a round-trip guarantee.
- **`is_valid()`** keeps every value structurally honest, and `StateError`
  reports the failures.

Next we look at the other half of the agent's interface — the **action space** —
where the same rank-and-shape discipline shows up again, this time on the way
*out* of the policy.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
