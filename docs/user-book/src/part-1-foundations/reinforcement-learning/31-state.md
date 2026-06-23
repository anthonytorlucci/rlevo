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

## Two traits, one rank

The relevant definitions live in `rlevo::core::base`:

```rust
pub trait State<const R: usize>: Debug + Clone + Send + Sync {
    const RANK: usize = R;
    type Observation: Observation<R>;

    fn shape() -> [usize; R];          // cardinality of each axis
    fn observe(&self) -> Self::Observation;
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

A `State` *owns* its `Observation` type (`type Observation: Observation<R>`),
and the two share the same const generic `R`. So before anything else, we have
to be precise about what `R` is.

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
> `Action`, and `TensorConvertible` traits — see the [overview](00-overview.md)
> for the rationale.

## The state/observation split

`State::observe()` is the seam. It takes the full state and produces what the
agent perceives. Whether that is a lossless view or a lossy projection is
entirely up to your implementation — and that single choice decides whether you
are modelling an MDP or a POMDP.

**Full observability (an MDP).** When `observe()` preserves all the information
needed for the Markov property, state and observation carry the same content.
CartPole is the canonical example — its state is four real numbers and its
observation is the same four numbers:

```rust
pub struct CartPoleState {
    pub x: f32,         // cart position
    pub x_dot: f32,     // cart velocity
    pub theta: f32,     // pole angle
    pub theta_dot: f32, // pole angular velocity
}

impl State<1> for CartPoleState {
    type Observation = CartPoleObservation;

    fn shape() -> [usize; 1] { [4] }
    fn numel(&self) -> usize { 4 }

    fn is_valid(&self) -> bool {
        self.x.is_finite() && self.x_dot.is_finite()
            && self.theta.is_finite() && self.theta_dot.is_finite()
    }

    fn observe(&self) -> CartPoleObservation {
        CartPoleObservation {
            cart_pos:     self.x,
            cart_vel:     self.x_dot,
            pole_angle:   self.theta,
            pole_ang_vel: self.theta_dot,
        }
    }
}
```

Here `observe()` is essentially a rename — every state field flows into the
observation. The agent sees everything; the Markov property holds; classical
value-based methods are on solid ground.

**Partial observability (a POMDP).** Now suppose `observe()` dropped the two
velocity fields and returned only `cart_pos` and `pole_angle`. The agent can no
longer tell a pole falling left from one swinging back toward centre — two
genuinely different states produce an identical observation. A single
observation is no longer Markov. You would need to recover the missing dynamics
some other way: stacking consecutive frames, or carrying a recurrent hidden
state (more on that below).

> **The takeaway.** The MDP-vs-POMDP question is not a property of the
> environment in the abstract — it is a property of *what you choose to return
> from `observe()`*. `rlevo` makes you write that function, so the choice is
> explicit and reviewable rather than buried in a `gym.make` flag.

### Why only `Observation` is serialisable

Look again at the supertraits: `Observation` requires `Serialize` +
`Deserialize`, but `State` does not. That asymmetry is deliberate. Observations
get written into replay buffers, shipped into `EpisodeRecord`s, and replayed
later; they need to round-trip through `serde`. The full state never leaves the
environment, so it carries no serialisation tax. The type system encodes which
side of the wall each value lives on.

## Crossing into tensor land: `TensorConvertible`

Traits like `State` and `Observation` are deliberately framework-agnostic —
they say nothing about Burn. The bridge to actual tensors (and therefore to
neural networks) is a separate trait, `TensorConvertible`:

```rust
pub trait TensorConvertible<const R: usize, B: Backend>: Sized {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, R>;
    fn from_tensor(tensor: Tensor<B, R>) -> Result<Self, TensorConversionError>;
}
```

Two things worth noticing.

First, it is implemented on the **observation** (and the action), not on the
state. That is the whole point of the split paying off: the only values that
ever become network inputs are the ones the agent is allowed to see. You
literally cannot `to_tensor()` the hidden state, because nobody implemented it
for the state type.

Second, the contract is a **round-trip invariant**:
`from_tensor(x.to_tensor(device))` must equal `Ok(x)` for any valid `x`. Replay
buffers and strategies lean on this — a transition stored as a tensor and read
back must be the same transition. CartPole's implementation is about as simple
as it gets:

```rust
impl<B: Backend> TensorConvertible<1, B> for CartPoleObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.to_array(), device)   // [pos, vel, angle, ang_vel]
    }

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
| `BeliefState` | A probability distribution over true states, updated by Bayes' rule from `(action, observation)` | Defined; for POMDP belief tracking |
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
  inside the environment.
- **`observe()`** projects it to an **`Observation`** — the only thing the agent
  sees, and the only thing that gets serialised into a replay buffer.
- **`TensorConvertible`** turns that observation into a Burn tensor for the
  network, with a round-trip guarantee.
- **`is_valid()`** keeps every value structurally honest, and `StateError`
  reports the failures.

Next we look at the other half of the agent's interface — the **action space** —
where the same rank-and-shape discipline shows up again, this time on the way
*out* of the policy.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
