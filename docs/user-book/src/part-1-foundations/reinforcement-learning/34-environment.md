# Environments

The last three chapters introduced the agent's three nouns — [state](31-state.md),
[action](32-action.md), [reward](33-reward.md). This chapter gives them a verb,
and in doing so it is the capstone of the reinforcement-learning track: the place
those nouns stop being things you *call* and become one running loop. The
**`Environment`** trait is the `reset`/`step` protocol that turns that vocabulary
into the interactive loop from the
[reinforcement-learning chapter](30-reinforcement-learning.md): the agent sends
an action, the environment answers with the next observation, a reward, and
whether the episode is over. It is the Markov Decision Process made executable.

We read the trait first — including the quiet work its type signature does to weld
the five nouns into one self-consistent set — then `reset`/`step` themselves, the
errors they may return, and the construction split that keeps the behaviour trait
clean. That split is not bookkeeping: it is what lets **wrappers** compose around a
core environment, and it answers a question CartPole leaves dangling — where the
classic 500-step truncation actually comes from.

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
`truncated` — matching the [`EpisodeStatus`](33-reward.md) variants. Each of
these leaves `SnapshotBase`'s fourth field, `metadata: Option<SnapshotMetadata>`,
as `None`; CartPole never needs it, but an environment that does chains
`.with_metadata(..)` onto any of the three constructors, as the
[reward chapter](33-reward.md#shaped-and-multi-component-rewards) shows.

Notice what CartPole's `step` *never* produces: `Truncated`. It emits `Running`
until the pole falls or the cart leaves the track, then `Terminated`. That is
correct — truncation is not part of CartPole's MDP. Where does the 500-step time
limit come from, then? Not from here. (Hold that thought for the wrappers
section.)

## The post-terminal rule

`reset`/`step` cover the lifecycle you *call* correctly. There is one more edge
the type signature alone cannot rule out: what happens if you call `step` again
after a snapshot has already reported `is_done() == true`? The `Result` return
type doesn't stop you from trying — a call sequence isn't something the type
checker can forbid — so `rlevo` states the rule in the trait's contract and
enforces it in code: **once a snapshot is done, the only valid next call is
`reset()`.** Stepping again returns
[`EnvironmentError::StepAfterEpisodeEnd { status }`](https://docs.rs/rlevo-core/latest/rlevo_core/environment/enum.EnvironmentError.html#variant.StepAfterEpisodeEnd),
where `status` is the `EpisodeStatus` that ended the episode — so you can tell
an intrinsic MDP termination (`Terminated`) from a wrapper-imposed truncation
(`Truncated`) without re-deriving it.

This is not a hypothetical foot-gun. `CliffWalking`'s goal tile sits *adjacent*
to the cliff, so before this rule existed, a move made after reaching the goal
could walk the agent back off the goal and onto the cliff — teleporting it to
the start and paying the −100 cliff penalty on a snapshot still reporting
`Running`. A finished episode was silently brought back to life, and the
trajectory a caller recorded from it was quietly wrong. Rejecting the call turns
that silent corruption into a loud, typed error at the exact point it happens.

**Why reject instead of absorbing?** Sutton & Barto's *absorbing state* — a
terminal state that transitions only to itself and pays zero reward — is a
notational device that lets the return \\(G_t = \sum_k \gamma^k R_{t+k+1}\\) be
written as one infinite sum across episodic and continuing tasks [Sutton and
Barto, 2018]. It is a statement about the mathematical process, not a
specification for what a `step()` function should do when a caller invokes it
out of sequence — and the reference implementations don't fill that gap either:
Gymnasium's `CartPole` prints a one-time warning and then keeps integrating the
physics past termination. An `Err` is strictly more informative than either, and
the choice is asymmetric: a caller who genuinely wants an absorbing, zero-reward
tail can still build a wrapper over a rejecting environment, but a caller who
silently absorbed a bug into a replay buffer has no way back. Reject is the
reversible choice.

**This is not yet universal — check before you rely on it.** Only the
`toy_text` family (`Blackjack`, `CliffWalking`, `FrozenLake`, `Taxi`) and the
`TimeLimit` wrapper enforce this rule today. Every other environment's
behaviour after a terminal snapshot remains **undefined**; do not depend on it,
even by accident, until it lands for that family — the rollout is tracked
family-by-family in
[issue #289](https://github.com/anthonytorlucci/rlevo/issues/289). If you're
implementing your own environment, [Bring Your Own
Environment](../../part-2-guided-tour/99-extending-the-environment.md#step-4--guard-the-post-terminal-step)
shows the `EpisodeGuard` helper that gets you this behaviour without hand-rolling
the state machine.

## Errors are part of the contract

`reset` and `step` return `Result` rather than a bare snapshot, and the error
type is a small enum — small, but no longer *closed*:

```rust
#[non_exhaustive]
pub enum EnvironmentError {
    InvalidAction(String),               // action illegal in the current state
    RenderFailed(String),                // display/rendering failure
    IoError(std::io::Error),             // e.g. loading level data
    Config(ConfigError),                 // a config invariant failed at reset
    StepAfterEpisodeEnd { status: EpisodeStatus }, // step() called past a done snapshot
}
```

Making fallibility explicit means a misbehaving agent that submits an illegal
action gets a typed `Err(InvalidAction(..))` it can handle, not a panic that
takes down the training run. Many simple environments (CartPole included) are
in practice infallible for `InvalidAction`, but the signature keeps the door
open for the ones that aren't — and `StepAfterEpisodeEnd`, from the previous
section, is exactly that door being used for a lifecycle fault rather than a bad
action. The `#[non_exhaustive]` attribute means a `match` on `EnvironmentError`
outside `rlevo-core` needs a wildcard `_` arm; that's what let this new variant
land without breaking any downstream match.

## Construction is a *separate* trait

You may have noticed `Environment` has no constructor. Building an environment
lives in its own one-method trait:

```rust
pub trait ConstructableEnv {
    fn new(render: bool) -> Self;
}
```

This split is deliberate ([ADR-0011](../../part-4-open-problems/01-where-rlevo-stands.md)).
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
`TimeLimit` binds to `SnapshotType = SnapshotBase<D, Obs, Rew>` and only ever
touches the `status` field, so any [`SnapshotMetadata`](33-reward.md#shaped-and-multi-component-rewards)
an inner environment attached with `.with_metadata(..)` rides through unchanged
— including on the very step `TimeLimit` truncates. Because every environment's
`SnapshotType` is a `SnapshotBase` instance, metadata-carrying environments
compose with `TimeLimit` (and any other `SnapshotBase`-bound wrapper) the same
way a plain `ScalarReward`-only environment like CartPole does.

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
> [status page](../../part-4-open-problems/01-where-rlevo-stands.md) before building
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
  part of the contract, and `EnvironmentError` is `#[non_exhaustive]`.
- **A `step()` taken after `is_done()` is `true` is an error, not a silent
  resume.** It returns `EnvironmentError::StepAfterEpisodeEnd { status }`; call
  `reset()` to start a new episode. Only `toy_text` and `TimeLimit` enforce this
  today ([issue #289](https://github.com/anthonytorlucci/rlevo/issues/289) tracks
  the rest).
- **`ConstructableEnv`** keeps construction off the behaviour trait so
  **wrappers** like `TimeLimit` compose cleanly — and that is where `Truncated`
  comes from.
- **`TransitionDynamics`** / **`UpdateFunction`** name the (deterministic)
  transition function for future model-based work; `step` is the general
  workhorse today.

That completes the foundational vocabulary — state, action, reward, and the
environment loop that drives them. From here, [Part II](../../part-2-guided-tour/00-overview.md)
puts the whole interface to work on real problems, and the
[evolutionary-computation](../20-evolutionary-computation.md) track shows how the
same environments serve as fitness landscapes for population-based search.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
