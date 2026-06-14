# Classic Control: CartPole with DQN

The previous two sections optimised a static function — the same score for the
same input, every time. Real sequential decision-making is different: the agent's
actions change the state of the world, and the reward comes from a *sequence* of
decisions, not a single evaluation.

**CartPole** is the canonical entry point for this regime. The task: balance a
pole attached to a cart by nudging the cart left or right. The episode ends when
the pole falls past 15° or the cart leaves the track. The longer you balance, the
higher the reward.

> **Example name.** This section is built from
> `rlevo-examples/examples/book/ch03_cartpole_dqn.rs`. Run it with:
> ```bash
> cargo run -p rlevo-examples --example ch03_cartpole_dqn
> ```

## The environment

CartPole is implemented in `rlevo-environments::classic::CartPoleEnv`. Its state
is a 4-vector — cart position, cart velocity, pole angle, pole angular velocity —
and its action space is binary: push left (0) or push right (1).

```rust,no_run
use rlevo_environments::classic::CartPoleEnv;
use rlevo_core::environment::Environment;

let mut env = CartPoleEnv::new();
let snapshot = env.reset()?;

println!("initial obs: {:?}", snapshot.observation());
// e.g. [0.04, -0.03, 0.01, -0.02]
```

`reset()` returns a `SnapshotBase` — the unified type that carries the first
observation. Each call to `step(action)` returns the next snapshot containing:

- **observation** — the 4-vector describing the current state,
- **reward** — `1.0` for every timestep the pole remains balanced,
- **terminal** — `true` when the episode ends.

This is the `Environment` trait from `rlevo-core`. Notice the parallel with
the `Landscape` from Part II: `Landscape::evaluate` was a one-shot score;
`Environment::step` is score-per-timestep inside an episode.

> **Foundations link.** The agent–environment loop, MDP formulation, and the
> meaning of discounted cumulative reward are introduced in
> [Reinforcement Learning](../part-1-foundations/03-reinforcement-learning.md).

## The agent: DQN

We will train a **Deep Q-Network** (DQN) to solve CartPole. Recall from the
foundations section: DQN approximates the action-value function
\\(Q(s, a; \theta)\\) with a neural network and stabilises training with
experience replay and a target network.

`rlevo` provides `DqnAgent` with a configuration struct:

```rust,no_run
use rlevo_reinforcement_learning::algorithms::dqn::{DqnAgent, DqnConfig};

let config = DqnConfig {
    state_dim: 4,
    action_dim: 2,
    hidden_dim: 128,
    learning_rate: 1e-3,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_end: 0.01,
    epsilon_decay_steps: 5_000,
    replay_capacity: 10_000,
    batch_size: 64,
    target_update_freq: 100,
};
let mut agent = DqnAgent::<NdArray>::new(config);
```

Read it top to bottom:
- `state_dim: 4` / `action_dim: 2` match CartPole's observation and action sizes.
- `gamma: 0.99` — the agent plans ~100 steps into the future
  (\\(0.99^{100} \approx 0.37\\)).
- `epsilon_start: 1.0` → `epsilon_end: 0.01` over 5,000 steps — the
  ε-greedy exploration schedule starts fully random and anneals to nearly greedy.
- `replay_capacity: 10_000` — the experience buffer holds 10,000 transitions.
- `target_update_freq: 100` — the target network is copied every 100 steps.

## The training loop

```rust,no_run
// See crates/rlevo-examples/examples/book/ch03_cartpole_dqn.rs
let mut total_steps = 0u64;
for episode in 0..500 {
    let mut snapshot = env.reset()?;
    let mut episode_reward = 0.0f64;

    loop {
        let action = agent.select_action(&snapshot.observation(), total_steps);
        let next_snapshot = env.step(action)?;

        agent.store_transition(&snapshot, action, &next_snapshot);
        agent.train_step();

        episode_reward += next_snapshot.reward().value();
        total_steps += 1;
        snapshot = next_snapshot;

        if snapshot.is_terminal() { break; }
    }
    // log episode_reward ...
}
```

At each step, the agent selects an action (ε-greedy), the environment returns a
transition, and the agent stores that transition in its replay buffer and runs
one gradient update on a sampled mini-batch.

This is *not* an ask/tell loop in the evolutionary sense — the agent acts
sequentially and updates after every step, not after a full population evaluation.
The connection to ask/tell will reappear when we combine evolution and RL: the EA
will `ask` for a population of DQN policies, evaluate each by running a full
episode, and `tell` the scores back.

## What you should see

CartPole is considered solved when the agent averages ≥ 195 reward over 100
consecutive episodes. With the config above, expect:

```text
episode   50   avg_reward =  38.2   ε = 0.71
episode  100   avg_reward =  62.1   ε = 0.43
episode  200   avg_reward = 142.5   ε = 0.09
episode  250   avg_reward = 196.3   ε = 0.01   ← solved
```

Training time is ~30 seconds on CPU with the `ndarray` backend.

## What DQN is learning

Early on (high ε), the agent acts randomly and the replay buffer fills with
diverse but uninformative transitions. As ε decays, the policy becomes greedier
and the buffer fills with better trajectories. The target network copy every 100
steps prevents the Q-estimates from chasing a moving target — without it, updates
become unstable.

The 4-dimensional observation space is small enough that DQN solves CartPole
easily. The same agent configuration, with larger hidden layers and a convolutional
front-end, was used to solve Atari games from 84×84 pixel inputs — the original
DQN paper's contribution was showing the architecture and stabilisation tricks
scale to that regime.

> **Foundations link.** The Bellman equation that DQN minimises, the role of the
> target network, and the experience replay mechanism are derived in
> [Reinforcement Learning](../part-1-foundations/03-reinforcement-learning.md).
> Full pseudocode for DQN as implemented in `rlevo` is in
> [Appendix B](../appendix-b-rl-algorithms/index.md).

## Make it yours

- **Increase `epsilon_decay_steps`** to 20,000 and watch the agent explore longer
  before converging — useful on harder tasks where early greedy behaviour gets
  trapped.
- **Reduce `hidden_dim`** to 32 and observe that the agent struggles: CartPole is
  simple but not *trivial* for a network that cannot represent the value function
  accurately.
- **Swap CartPole for Acrobot** (`AcrobotEnv` in `rlevo-environments`) — a
  two-link pendulum with a sparser reward. The same DQN config will fail without
  reward shaping or a larger network; this is the motivation for the hybrid
  methods in later sections.

## Up next

The [next section](04-extending-the-environment.md) shows you how to implement
the `Environment` trait for your own domain — the prerequisite for applying any
`rlevo` algorithm to a problem you bring to the library.
