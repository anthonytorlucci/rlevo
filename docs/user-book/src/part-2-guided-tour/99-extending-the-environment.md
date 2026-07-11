# Bring Your Own Environment

Every algorithm in this Part ran against a built-in environment — a `Landscape`
in Chapters 1–2, `CartPole` in Chapter 3. This section closes the loop: it shows
how to implement the `Environment` trait for *your own* domain, so that any
`rlevo` algorithm — evolutionary, RL, or hybrid — can drive it.

We work through a real, CI-tested example: the **k-armed bandit** from Sutton &
Barto §2. It is the smallest environment that is still a genuine sequential
decision problem, which makes it the right teacher. The complete source —
including the ~30 tests this section only summarises — lives in
[`crates/rlevo-environments/src/classic/bandit/k_armed.rs`](https://github.com/anthonytorlucci/rlevo/blob/main/crates/rlevo-environments/src/classic/bandit/k_armed.rs).

> **The bandit task.** There are `K` levers. Pulling arm `a` pays a reward drawn
> from \\(\mathcal{N}(q^*(a),\,1)\\), where each true mean \\(q^*(a)\\) is itself
> fixed at construction by a draw from \\(\mathcal{N}(0,1)\\). The agent never
> sees the means; it must *learn* which arm pays best by pulling and observing.
> The episode ends after `max_steps` pulls.

## The trait surface

`Environment` is the contract the rest of `rlevo` programs against. It is generic
over three const-generic **tensor ranks** and ties together four associated
types:

```rust,no_run
pub trait Environment<const R: usize, const SR: usize, const AR: usize> {
    type StateType:       State<SR>;
    type ObservationType: Observation<R>;
    type ActionType:      Action<AR>;
    type RewardType:      Reward;
    type SnapshotType:    Snapshot<R, ObservationType = Self::ObservationType,
                                       RewardType      = Self::RewardType>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError>;
    fn step(&mut self, action: Self::ActionType)
        -> Result<Self::SnapshotType, EnvironmentError>;
}
```

The three const generics are *ranks*, not sizes:

| Param | Meaning | Bandit | A 64×64 RGB camera env |
| ----- | ------- | ------ | ---------------------- |
| `R`   | observation rank — how many tensor axes the agent perceives | `1` | `3` (H × W × C) |
| `SR`  | state rank — axes of the full internal state | `1` | `3` |
| `AR`  | action rank — axes of an action | `1` | `1` |

For the same-modality case (the agent sees the full state) `R == SR`, as in the
bandit's `Environment<1, 1, 1>`. They are allowed to differ for
modality-changing POMDPs — see `Observable<OR>` and ADR 0019 — but you can ignore
that until you need it.

So implementing an environment means supplying four types and two methods. We
take them in order.

## Step 1 — `State` and `Observation`

`State<SR>` is the full ground truth of your world; `Observation<R>` is what the
agent is allowed to perceive. The bandit is *stateless* — the optimal action
never depends on history — so both are zero-field marker structs that exist only
to satisfy the trait bounds:

```rust,no_run
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KArmedBanditState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct KArmedBanditObservation;

impl Observation<1> for KArmedBanditObservation {
    fn shape() -> [usize; 1] { [1] }      // a length-1 feature vector
}

impl State<1> for KArmedBanditState {
    type Observation = KArmedBanditObservation;

    fn shape()  -> [usize; 1]        { [1] }
    fn observe(&self) -> Self::Observation { KArmedBanditObservation }
    fn is_valid(&self) -> bool       { true }
    fn numel(&self)  -> usize        { 1 }
}
```

The key method is **`observe`**: it projects the full state down to what the
agent sees. For the bandit that projection is trivial. For a stateful
environment — say a gridworld — `State` would carry the agent's `(x, y)` and the
grid contents, and `observe` would return only the slice the agent can sense
(its local neighbourhood, or the whole grid for a fully-observable task). The
`State`/`Observation` split *is* the observability seam: keep ground truth in
`State`, expose perception through `observe`.

To plug into neural-network agents, your observation also implements
`TensorConvertible<R, B>` (`to_tensor` / `from_tensor`). The bandit encodes its
empty state as the constant tensor `[0.0]` and validates shape `[1]` on the way
back.

## Step 2 — `Action`

`Action<AR>` is the set of choices the agent can make. The bandit's action is
*which arm to pull* — a discrete choice in `0..K`, so it also implements
`DiscreteAction<1>` (giving `ACTION_COUNT`, `from_index`, `to_index`,
`enumerate`):

```rust,no_run
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KArmedBanditAction<const K: usize> {
    selected_arm: usize,
}

impl<const K: usize> KArmedBanditAction<K> {
    /// Fallible constructor — the validating door for untrusted input.
    pub fn new(arm: usize) -> Result<Self, EnvironmentError> {
        if arm < K {
            Ok(Self { selected_arm: arm })
        } else {
            Err(EnvironmentError::InvalidAction(format!(
                "arm index {arm} out of range [0, {K})"
            )))
        }
    }
}

impl<const K: usize> Action<1> for KArmedBanditAction<K> {
    fn shape() -> [usize; 1] { [K] }      // one-hot length
}
```

Two design choices here are worth copying:

- **The action is a typed value, not a bare `usize`.** `KArmedBanditAction::Left`
  / arm indices carry meaning; an agent can't accidentally pass `42` where an
  arm index belongs.
- **Validation lives in a fallible `new`.** `DiscreteAction::from_index` *panics*
  on an out-of-range index by trait contract — fine when a policy mask already
  guarantees the range. `new` returns `Result` for anything that came from
  outside (config, RPC, an unmasked policy head). Offer both doors and let the
  caller pick.

Like observations, actions implement `TensorConvertible<1, B>` — here a one-hot
of length `K` — so a neural policy can emit and consume them.

## Step 3 — implement `Environment`

Now the two methods. `reset` starts a fresh episode and returns the first
snapshot; `step` advances one timestep. Both return a `SnapshotBase`, the stock
`Snapshot` implementation that bundles `(observation, reward, status)`:

```rust,no_run
impl<const K: usize> Environment<1, 1, 1> for KArmedBandit<K> {
    type StateType       = KArmedBanditState;
    type ObservationType = KArmedBanditObservation;
    type ActionType      = KArmedBanditAction<K>;
    type RewardType      = ScalarReward;
    type SnapshotType    = SnapshotBase<1, KArmedBanditObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        // Re-seed RNG and re-draw arm means so (config, actions) fully
        // determines the trajectory.
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.arm_means = sample_arm_means::<K>(&mut self.rng);
        self.steps = 0;
        self.done = false;
        Ok(SnapshotBase::running(self.state.observe(), ScalarReward::zero()))
    }

    fn step(&mut self, action: Self::ActionType)
        -> Result<Self::SnapshotType, EnvironmentError>
    {
        if !action.is_valid() {
            return Err(EnvironmentError::InvalidAction(format!(
                "arm index {} out of range [0, {K})", action.arm(),
            )));
        }
        let reward = ScalarReward(self.sample_reward(action.arm()));
        self.steps += 1;
        let obs = self.state.observe();

        // Mark the snapshot terminal at the step budget; running otherwise.
        let snap = if self.steps >= self.config.max_steps {
            self.done = true;
            SnapshotBase::terminated(obs, reward)
        } else {
            SnapshotBase::running(obs, reward)
        };
        Ok(snap)
    }
}
```

Note the three `SnapshotBase` constructors that encode episode status:
`running`, `terminated` (the task reached a natural end state), and `truncated`
(a time/step limit cut it short). The caller reads them back through
`snapshot.is_done()`, `is_terminated()`, and `is_truncated()` — exactly the
status queries Chapter 3's training loop used.

> **If your environment needs to log reward components.** The bandit's reward
> is a single clean number, so it never needs this, but a richer environment —
> a locomotion task with a forward-progress term, a control-cost penalty, and a
> "still alive" bonus — usually wants to *log* those components without
> changing `RewardType` away from `ScalarReward`. You do not need a bespoke
> `Snapshot` type for that: `SnapshotBase` carries an optional
> `metadata: Option<SnapshotMetadata>` field, and a fluent `with_metadata`
> chains onto any of the three constructors above, e.g.
> `SnapshotBase::running(obs, reward).with_metadata(my_metadata)`. Because the
> result is still a `SnapshotBase`, it stays compatible with `TimeLimit` and
> any other wrapper bound to `SnapshotBase` — see
> [Rewards §Shaped and multi-component rewards](../part-1-foundations/reinforcement-learning/33-reward.md#shaped-and-multi-component-rewards)
> for the full contract.

## Step 4 — guard the post-terminal step

We've covered the happy path — `reset` opens an episode, `step` advances it,
`SnapshotBase`'s status tells the caller when it ends. One lifecycle edge
remains, and `rlevo` treats it normatively rather than leaving it to each
environment's discretion: what should `step` do if a caller calls it *again*
after a snapshot has already reported `is_done() == true`? Left unguarded, the
answer is "whatever your code happens to do" — and for `CliffWalking`, whose
goal tile sits directly next to the cliff, the unguarded answer used to be a
real bug: a post-terminal move could walk the agent back off the goal and onto
the cliff, teleporting it to the start and paying the −100 cliff penalty on a
snapshot still reporting `Running`. A finished episode came back to life with a
corrupted trajectory. See [Environments §The post-terminal
rule](../part-1-foundations/reinforcement-learning/34-environment.md#the-post-terminal-rule)
for the full contract and why `rlevo` rejects rather than silently absorbs.

The rule your `step` must satisfy: once you have emitted a snapshot whose
status is done, the *only* legal next call is `reset()`; a further `step()`
must return `Err(EnvironmentError::StepAfterEpisodeEnd { status })`. You don't
have to hand-write that state machine — `rlevo_environments::episode::EpisodeGuard`
is the one-field helper every `toy_text` environment holds for exactly this.
Add it to your struct:

```rust,no_run
use rlevo_environments::episode::EpisodeGuard;

pub struct MyEnv {
    // ... your state, config, rng ...
    guard: EpisodeGuard,
}
```

and wire it into the three lifecycle points, following `CliffWalking`'s own
implementation
([`crates/rlevo-environments/src/toy_text/cliff_walking.rs`](https://github.com/anthonytorlucci/rlevo/blob/main/crates/rlevo-environments/src/toy_text/cliff_walking.rs)):

```rust,no_run
fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
    self.guard.reset();                    // only once reset has actually succeeded
    self.state = /* fresh initial state */;
    Ok(SnapshotBase::running(self.state.observe(), ScalarReward(0.0)))
}

fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
    // Guard first: before any state mutation, and before any RNG draw. A
    // rejected call must leave both the state and the RNG stream untouched —
    // otherwise the stream depends on how many illegal steps a caller made,
    // silently breaking seed -> trajectory reproducibility (ADR 0029).
    self.guard.check()?;

    // ... apply the action, compute the reward ...

    // Single exit: build exactly one snapshot, then feed the guard that same
    // snapshot's status, so the two cannot drift apart.
    let snapshot = /* SnapshotBase::running/terminated/truncated(..) */;
    self.guard.record(snapshot.status);
    Ok(snapshot)
}
```

Three details here are load-bearing, not stylistic:

- **`check()` is the first statement.** Not just before mutating state — before
  *any* RNG draw too. `CliffWalking`'s slippery-move resolution samples from
  `self.rng`; if the guard ran after that sample, a rejected step would still
  advance the RNG stream, and the same seed would no longer reproduce the same
  trajectory.
- **One exit, one `record`.** Build the snapshot once and record *its own*
  status on the way out, rather than setting the guard from a separately
  computed `terminated`/`truncated` flag. Two independent judgments about
  "did the episode end" can drift apart; one snapshot feeding both the caller
  and the guard cannot.
- **`reset()` only after reset succeeds.** If your `reset` can itself fail (a
  `ConfigError` from re-validating a procedurally rebuilt world, say), reopen
  the guard *after* the fallible work, not before — a failed reset must not
  silently re-open a finished episode.

> **Scope: only four environments enforce this today.** `EpisodeGuard` ships in
> `rlevo-environments`, and the `toy_text` family (`Blackjack`, `CliffWalking`,
> `FrozenLake`, `Taxi`) plus the `TimeLimit` wrapper use it. The other ~44
> built-in environments do not yet — their post-terminal behaviour is
> undefined, tracked family-by-family in
> [issue #289](https://github.com/anthonytorlucci/rlevo/issues/289). Your own
> environment doesn't have to wait for that rollout: reach for `EpisodeGuard`
> the same way `CliffWalking` does, and it conforms from day one.

## Step 5 — construction

Construction lives on a *separate* factory trait, `ConstructableEnv`, never on
`Environment` itself (ADR 0011). That keeps the runtime contract — `reset`/`step`
— free of constructor signatures, so generic code can build any environment the
same way:

```rust,no_run
impl<const K: usize> ConstructableEnv for KArmedBandit<K> {
    fn new(render: bool) -> Self {
        let _ = render;                       // bandit has nothing to render
        Self::with_config(KArmedBanditConfig::default())
            .expect("default config must validate")
    }
}
```

Offer a richer builder alongside it for callers that need to configure the
environment — the bandit exposes `with_config(KArmedBanditConfig { max_steps,
seed })` and a `with_seed(seed)` shortcut. `with_config` validates the config and
returns `Result<Self, ConfigError>` (see the config-validation contract below), so
a caller supplying its own values can react to a bad one. `ConstructableEnv::new`
stays infallible: it feeds the *default* config, which your `Validate` impl
guarantees is valid, so the single `.expect` is the one blessed panic. Generic
harness code calls `new`; an experiment script that needs a specific seed calls
`with_config` and handles the `Result`.

## The contract your implementation must satisfy

These are the invariants the rest of `rlevo` assumes. Hold them and any
algorithm will drive your environment correctly:

- **`reset()` always yields a valid, non-terminal first snapshot** — or an
  `EnvironmentError`. It must fully re-initialise episode state.
- **`step(action)` advances exactly one timestep** and flags the snapshot
  terminal (or truncated) the moment the episode ends. A further `step()` after
  that is a caller error, not a fresh transition, and should be rejected with
  `EnvironmentError::StepAfterEpisodeEnd` rather than silently continuing — see
  [Step 4](#step-4--guard-the-post-terminal-step) above for the `EpisodeGuard`
  recipe. (Only the `toy_text` family and `TimeLimit` enforce this today; treat
  it as the target for any environment you write, not yet a workspace-wide
  guarantee.)
- **Neither method panics on valid input.** Return `EnvironmentError::InvalidAction`
  for out-of-range actions; reserve panics for genuine internal-logic bugs.
- **Your config validates its own invariants.** Implement
  [`Validate`](https://docs.rs/rlevo-core) for `KArmedBanditConfig` (checking, say,
  `max_steps > 0`) and call it at the `with_config` chokepoint so a bad
  hyperparameter is rejected as a field-named `ConfigError` at construction, not as
  a confusing panic mid-episode. A config that derives `Deserialize` is
  user-supplied runtime data, so this must be a `Result`, never a panic (ADR 0026).
  Back it with a `assert!(KArmedBanditConfig::default().validate().is_ok())` unit
  test — that is what lets `ConstructableEnv::new` stay infallible.
- **Determinism is seedable.** Thread all randomness through a seed you store, so
  `(config, action sequence)` reproduces the trajectory — the bandit re-draws its
  arm means from `config.seed` on every `reset`. This is what makes RL results
  re-runnable (Chapter 2's reproducibility argument applies to environments too).

## Testing your environment

Mirror the bandit's test module. The shape that matters:

1. **`reset` returns a running snapshot** with a valid observation and zero/initial reward.
2. **`step` with a known action** produces the expected observation and a reward in range.
3. **The terminal condition fires** and is flagged (`is_terminated` / `is_truncated`) correctly.
4. **`InvalidAction` is returned** for out-of-bounds actions — test the error path, not just the happy path.
5. **Same seed ⇒ same trajectory.** Construct twice with one seed, run the same actions, assert identical rewards.
6. **`TensorConvertible` round-trips** and rejects wrong-shaped tensors.
7. **If you added an `EpisodeGuard`, test the post-terminal rejection directly.**
   Drive the environment to a done snapshot, `step()` once more with a legal
   action, and assert you get back
   `Err(EnvironmentError::StepAfterEpisodeEnd { status })` carrying the status
   that ended the episode — not `Ok`, and not a different error variant. Then
   assert `reset()` re-opens the environment for a second episode.

Once these pass, your environment is a first-class citizen: the evolutionary
harness, the DQN/PPO agents from Chapter 3, and the hybrid loops in Part IV all
accept it through the same `Environment` interface — you wrote the world, and the
whole library can now learn in it.

## Where this leaves you

You have now seen the full vertical slice the Guided Tour set out to cover:

- a **problem** (`Landscape`, then `Environment`),
- a **searcher** (`Strategy` / GA, then a DQN agent),
- the **loops** that tie them together (ask/tell, and the sequential RL loop),
- and how to **bring your own** problem into that machinery.

Where to go next depends on what you want:

- **Go deeper on the algorithms** → the appendices derive each one:
  [Evolutionary Computation](../appendix-a-ec-algorithms/index.md),
  [Reinforcement Learning](../appendix-b-rl-algorithms/index.md),
  [Hybrid](../appendix-c-hybrid-algorithms/index.md).
- **Understand the design rationale** → the
  [Contributor Book](https://github.com/anthonytorlucci/rlevo/tree/main/docs/contributor-book)
  and the ADRs explain *why* the seams fall where they do.
- **See where the project is headed** →
  [Part IV — Open Problems](../part-4-open-problems/00-overview.md).

> **Foundations link.** The agent–environment loop and MDP formulation these
> traits encode are introduced in
> [Reinforcement Learning](../part-1-foundations/30-reinforcement-learning.md);
> the `State`/`Observation` distinction and the observability seam are developed
> in [State and Observation Spaces](../part-1-foundations/reinforcement-learning/31-state.md).

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
