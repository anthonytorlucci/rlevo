# Environments

The last three chapters introduced the agent's three nouns — [state](31-state.md),
[action](32-action.md), [reward](33-reward.md). This chapter gives them a verb.
The **`Environment`** trait is the `reset`/`step` protocol that turns that
vocabulary into the interactive loop from the
[reinforcement-learning chapter](30-reinforcement-learning.md): the agent sends
an action, the environment answers with the next observation, a reward, and
whether the episode is over. It is the Markov Decision Process made executable.

## The `Environment` trait

From `rlevo::core::environment`:

```rust
pub trait Environment<const R: usize, const SR: usize, const AR: usize> {
    type StateType:       State<SR>;
    type ObservationType: Observation<R>;
    type ActionType:      Action<AR>;
    type RewardType:      Reward;
    type SnapshotType:
        Snapshot<R, ObservationType = Self::ObservationType, RewardType = Self::RewardType>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError>;
    fn step(&mut self, action: Self::ActionType)
        -> Result<Self::SnapshotType, EnvironmentError>;
}
```

That is the whole behavioural contract: two methods and five associated types.
But the type signature is doing a lot of quiet work, so it is worth slowing down.

### Three rank parameters — but only two degrees of freedom

The trait carries *three* const generics — `R`, `SR`, `AR` — one for each tensor
space:

| Parameter | Rank of… | Pinned to |
| --------- | -------- | --------- |
| `R`  | the observation (and snapshot) | `Observation<R>`, `Snapshot<R>` |
| `SR` | the full state | `State<SR>` |
| `AR` | the action | `Action<AR>` |

It is tempting to read these as three independent knobs, but in practice `R` and
`SR` move together. The reason is worth unpacking, because it touches the
difference between a tensor's *rank* and the *information* it carries.

**Rank is not information.** Partial observability — the defining feature of a
POMDP — is about how much of the state the observation reveals, not about how
many axes it has. Drop the velocity fields from CartPole and the observation
becomes strictly less informative (`[4]` → `[2]`), a genuine POMDP, yet it is
still a flat rank-1 vector. The *shape* shrank; the *rank* did not. That is the
flavour of partial observability `rlevo` models directly: same rank, less
information.

**Why `R == SR` in every current environment.** Recall from the
[state chapter](31-state.md) that `State<SR>` welds its observation to its *own*
rank: `type Observation: Observation<SR>`. A rank-`SR` state can therefore only
produce a rank-`SR` observation through `observe()`. Since every environment
builds its snapshots from `state.observe()`, the observation rank equals the
state rank. The codebase bears this out: every environment is either
`Environment<1, 1, 1>` (classic-control and toy-text) or `Environment<3, 3, 1>`
(the MiniGrid-style grids and CarRacing, whose rank-3 *image is itself the
state*). `R` never diverges from `SR`.

**So why two parameters?** Because the `Environment` trait does *not* assert the
equality — it never binds `ObservationType = StateType::Observation`. Those are
independent associated types, which leaves a deliberate seam: a *modality-changing*
POMDP, where the observation is a different rank from the state. The textbook
case is Atari — a low-rank emulator-RAM state behind a rank-2 pixel observation.
`State::observe()` cannot express that, because the `State` trait pins its
observation to the state's *own* rank.

That is exactly what the [`Observable`](https://docs.rs/rlevo-core/latest/rlevo_core/state/trait.Observable.html)
trait is for. `Observable<OR>` is a standalone projection trait
(`fn project(&self) -> Self::Observation`, where `Self::Observation: Observation<OR>`)
that lets a state map into an observation of a *different* rank `OR`. A
modality-changing environment's state implements `State<SR>` for its full
representation **and** `Observable<OR>` for the projected modality, then builds its
snapshots from `state.project()` instead of `state.observe()`. Because `Environment`
already permits `R != SR`, no change to the environment contract is needed — this is
the typed home for the rank change, resolving
[issue #62](https://github.com/anthonytorlucci/rlevo/issues/62).

So the practical rule is precise: **`R == SR` for any environment that observes via
`State::observe()` (all of them today); an environment that observes via
`Observable::project()` may have `R != SR`.** The other parameter that genuinely
varies is **`AR`**: in `Environment<3, 3, 1>` the state and observation are rank-3
while the action is a single rank-1 discrete choice.

### The associated types form a consistent set

The five associated types are not independent — look at the bound on
`SnapshotType`:

```rust
type SnapshotType:
    Snapshot<R, ObservationType = Self::ObservationType, RewardType = Self::RewardType>;
```

This says: the snapshot an environment returns must carry *exactly this
environment's own* observation and reward types. You cannot declare
`ObservationType = CartPoleObservation` and then hand back a snapshot full of
some other observation — it won't compile. The whole noun-set is welded into one
self-consistent unit at the type level, which is why downstream code (agents,
replay buffers, recorders) can name `Env::ObservationType` and trust it lines up
with what `step` actually produces.

## `reset` and `step`

The two methods mirror the MDP lifecycle. `reset` starts a fresh episode and
returns the first snapshot — initial observation, a reward of `zero()`, and
`EpisodeStatus::Running`. `step` applies an action, advances the internal state,
and returns the resulting snapshot. Both are fallible (`Result<_,
EnvironmentError>`), because a real environment can fail to load a level or a
physics step can blow up — more on errors below.

CartPole's implementation is small enough to read whole, and it shows the shape
every environment follows:

```rust
impl Environment<1, 1, 1> for CartPole {
    type StateType       = CartPoleState;
    type ObservationType = CartPoleObservation;
    type ActionType      = CartPoleAction;
    type RewardType      = ScalarReward;
    type SnapshotType    = SnapshotBase<1, CartPoleObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.state = self.sample_init_state();
        self.steps = 0;
        Ok(SnapshotBase::running(self.state.observe(), ScalarReward(0.0)))
    }

    fn step(&mut self, action: CartPoleAction) -> Result<Self::SnapshotType, EnvironmentError> {
        let next = Self::step_physics(&self.state, action, &self.config);
        self.state = next;
        self.steps += 1;

        let terminated = Self::is_terminal(&self.state, &self.config);
        let reward = if terminated && self.config.sutton_barto_reward {
            ScalarReward(-1.0)
        } else if self.config.sutton_barto_reward {
            ScalarReward(0.0)
        } else {
            ScalarReward(1.0)
        };

        let snap = if terminated {
            SnapshotBase::terminated(self.state.observe(), reward)
        } else {
            SnapshotBase::running(self.state.observe(), reward)
        };
        Ok(snap)
    }
}
```

A few things connect back to earlier chapters. The full `CartPoleState` lives
*inside* the struct and only `state.observe()` — the observation — ever crosses
the boundary into a snapshot, exactly the [state/observation wall](31-state.md)
from before. The reward is a `ScalarReward`. And `SnapshotBase` (the default
`Snapshot` implementation) has three constructors — `running`, `terminated`,
`truncated` — matching the [`EpisodeStatus`](33-reward.md) variants.

Notice what CartPole's `step` *never* produces: `Truncated`. It emits `Running`
until the pole falls or the cart leaves the track, then `Terminated`. That is
correct — truncation is not part of CartPole's MDP. Where does the 500-step time
limit come from, then? Not from here. (Hold that thought for the wrappers
section.)

## Errors are part of the contract

`reset` and `step` return `Result` rather than a bare snapshot, and the error
type is a small, closed enum:

```rust
pub enum EnvironmentError {
    InvalidAction(String),   // action illegal in the current state
    RenderFailed(String),    // display/rendering failure
    IoError(std::io::Error), // e.g. loading level data
}
```

Making fallibility explicit means a misbehaving agent that submits an illegal
action gets a typed `Err(InvalidAction(..))` it can handle, not a panic that
takes down the training run. Many simple environments (CartPole included) are
in practice infallible and always return `Ok`, but the signature keeps the door
open for the ones that aren't.

## Construction is a *separate* trait

You may have noticed `Environment` has no constructor. Building an environment
lives in its own one-method trait:

```rust
pub trait ConstructableEnv {
    fn new(render: bool) -> Self;
}
```

This split is deliberate ([ADR-0011](../part-4-open-problems/01-where-rlevo-stands.md)).
Construction is a different concern from the `reset`/`step` *behaviour*, and
keeping it separate is what makes **transparent decorators** possible. A wrapper
like `TimeLimit` or a recording tap is built from an *existing* inner
environment, not from nothing — forcing it to implement a standalone `new`
would mean synthesising a meaningless constructor just to satisfy a bound. By
lifting construction out, a decorator implements only the behaviour it actually
forwards. Generic code that genuinely needs to build an environment from scratch
bounds on `E: ConstructableEnv`; everything else bounds on `Environment` alone.

The `render` flag controls whether the environment emits display output each
step — pass `false` for training, where rendering is pure overhead. (CartPole's
`new` ignores it and defers to `with_config` for a fully configured build.)

## Wrappers: where truncation comes from

Back to the dangling thread. CartPole never truncates itself, yet the classic
benchmark caps episodes at 500 steps. That cap is a **wrapper** — a decorator
that implements `Environment` by forwarding to an inner environment and adjusting
the result. `TimeLimit` is the canonical one:

```rust
pub struct TimeLimit<E> { inner: E, max_steps: usize, /* counter */ }
```

It counts steps, forwards each `step` to `inner`, and when the counter reaches
`max_steps` while the inner episode is still `Running`, it flips the snapshot's
status to `Truncated` — matching the Gymnasium `TimeLimit` convention. This is
the clean realisation of the [terminated-vs-truncated distinction](33-reward.md):
the *environment* owns intrinsic termination (`Terminated`), and a *wrapper* owns
the extrinsic step limit (`Truncated`). Neither has to know about the other, and
an algorithm downstream reads the two flags off the snapshot and bootstraps
accordingly.

This is the decorator pattern that the `ConstructableEnv` split exists to
enable: wrappers compose around a core environment without polluting it.

## Related abstractions: `TransitionDynamics` and `UpdateFunction`

Two more traits in `rlevo::core::base` sit conceptually next to `Environment`,
and it's worth knowing where they fit — and, just as honestly, that they are
currently **defined-but-unconsumed seams**, like the latent-state traits from
the [state chapter](31-state.md). They describe shapes the API may grow into;
no algorithm in the crate implements them today.

`TransitionDynamics` models the environment's *transition function* directly:

```rust
pub trait TransitionDynamics<const SR: usize, const AR: usize, S: State<SR>, A: Action<AR>> {
    fn transition(&self, state: &S, action: &A) -> S;   // s_{t+1} = f(s_t, a_t)
}
```

This is the MDP's \\(P(s' \mid s, a)\\) — but note the signature returns a single
`S`, so it captures only the **deterministic** case. That restriction is the
reason it is a *separate* trait rather than part of `Environment`. The real
workhorse is `step`, which is strictly more general: it handles stochastic
transitions (it can sample internally), computes the reward, and decides
termination — none of which `transition` does. Where `TransitionDynamics` would
earn its keep is model-based RL: an agent that learns or is given an explicit
\\(f\\) it can roll forward in imagination, separate from the live environment.

`UpdateFunction<Input, Output>` is the generic primitive that
`TransitionDynamics` specialises — "given a current value and an input, produce
the next value":

```rust
pub trait UpdateFunction<Input, Output> {
    fn update(&self, current: &Output, input: &Input) -> Output;
}
```

A deterministic transition is exactly an `UpdateFunction<Action, State>`. It is
the most abstract statement of "something that evolves over time," and it sits in
the crate as a vocabulary anchor more than a workhorse.

> **The honest status.** Use `Environment` (and `step`) for everything you
> actually run today. Treat `TransitionDynamics` and `UpdateFunction` as
> sign-posts for where model-based methods would plug in — and check the
> [status page](../part-4-open-problems/01-where-rlevo-stands.md) before building
> on them.

## Putting it together

- **`Environment<R, SR, AR>`** is the executable MDP: five associated types
  (state, observation, action, reward, snapshot) welded into a consistent unit,
  plus `reset` and `step`.
- `R == SR` for every environment that observes via `State::observe()` (all of
  them today); `AR` is the rank that genuinely varies. A modality-changing POMDP
  can break that equality by observing through `Observable::project()` instead —
  the typed home for a state→observation rank change ([issue #62](https://github.com/anthonytorlucci/rlevo/issues/62)) —
  and the `SnapshotType` bound forces the noun-set to agree.
- `reset`/`step` return `Result<Snapshot, EnvironmentError>` — fallibility is
  part of the contract.
- **`ConstructableEnv`** keeps construction off the behaviour trait so
  **wrappers** like `TimeLimit` compose cleanly — and that is where `Truncated`
  comes from.
- **`TransitionDynamics`** / **`UpdateFunction`** name the (deterministic)
  transition function for future model-based work; `step` is the general
  workhorse today.

That completes the foundational vocabulary — state, action, reward, and the
environment loop that drives them. From here, [Part II](../part-2-guided-tour/00-overview.md)
puts the whole interface to work on real problems, and the
[evolutionary-computation](20-evolutionary-computation.md) track shows how the
same environments serve as fitness landscapes for population-based search.

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
