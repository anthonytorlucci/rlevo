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

## Step 4 — construction

Construction lives on a *separate* factory trait, `ConstructableEnv`, never on
`Environment` itself (ADR 0011). That keeps the runtime contract — `reset`/`step`
— free of constructor signatures, so generic code can build any environment the
same way:

```rust,no_run
impl<const K: usize> ConstructableEnv for KArmedBandit<K> {
    fn new(render: bool) -> Self {
        let _ = render;                       // bandit has nothing to render
        Self::with_config(KArmedBanditConfig::default())
    }
}
```

Offer a richer builder alongside it for callers that need to configure the
environment — the bandit exposes `with_config(KArmedBanditConfig { max_steps,
seed })` and a `with_seed(seed)` shortcut. Generic harness code calls
`ConstructableEnv::new`; an experiment script that needs a specific seed calls
`with_config`.

## The contract your implementation must satisfy

These are the invariants the rest of `rlevo` assumes. Hold them and any
algorithm will drive your environment correctly:

- **`reset()` always yields a valid, non-terminal first snapshot** — or an
  `EnvironmentError`. It must fully re-initialise episode state.
- **`step(action)` advances exactly one timestep** and flags the snapshot
  terminal (or truncated) the moment the episode ends. After a terminal step the
  caller will `reset()` before the next `step()`.
- **Neither method panics on valid input.** Return `EnvironmentError::InvalidAction`
  for out-of-range actions; reserve panics for genuine internal-logic bugs.
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
> in [State and Observation Spaces](../part-1-foundations/31-state.md).

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
