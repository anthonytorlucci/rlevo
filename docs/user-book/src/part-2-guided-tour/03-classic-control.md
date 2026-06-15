# Classic Control: CartPole with DQN

The previous two sections optimised a static function — the same score for the
same input, every time. Real sequential decision-making is different: the agent's
actions change the state of the world, and the reward comes from a *sequence* of
decisions, not a single evaluation.

**CartPole** is the canonical entry point for this regime. The task: balance a
pole attached to a cart by nudging the cart left or right. The episode ends when
the pole falls too far or the cart leaves the track. The longer you balance, the
more reward you accumulate — one point per surviving timestep.

> **Status: preview.** The DQN agent is still in progress (see
> [Where rlevo Stands Today](../part-3-open-problems/01-where-rlevo-stands.md)),
> and there is not yet a shipped, CI-tested CartPole example in
> `rlevo-examples`. This section sketches the real API so you can see the shape
> of an RL run, but unlike Chapters 1–2 it is **not** backed by a runnable
> example, and it deliberately quotes no training curve we have not reproduced.
> The full, tuned walkthrough — with numbers — lands when the agent stabilises.

## The environment

CartPole lives in `rlevo::envs::classic::cartpole::CartPole`. It implements
`Environment<1, 1, 1>` — observation, state, and action spaces are all rank-1
(flat). Construction goes through `with_config` (or `ConstructableEnv::new`):

```rust,no_run
use rlevo::envs::classic::cartpole::{CartPole, CartPoleConfig};
use rlevo::core::environment::Environment;

let mut env = CartPole::with_config(CartPoleConfig::default());
let snapshot = env.reset()?;

let obs = snapshot.observation();          // &CartPoleObservation
println!("pole angle: {}", obs.pole_angle);
```

The observation is a struct of four `f32`s — `cart_pos`, `cart_vel`,
`pole_angle`, `pole_ang_vel` — not a bare vector, so each field is named and
typed. Each `step(action)` returns a `SnapshotBase` carrying:

- **observation** — the next `CartPoleObservation`,
- **reward** — a `ScalarReward`; read it with `(*snapshot.reward()).into()` to get
  an `f32` (`1.0` per surviving timestep),
- **status** — query `snapshot.is_done()` (terminated *or* truncated) or
  `snapshot.is_terminated()`.

The action is a two-variant enum, not a raw integer:

```rust,no_run
use rlevo::envs::classic::cartpole::CartPoleAction;

let push_left = CartPoleAction::Left;       // index 0
let push_right = CartPoleAction::Right;     // index 1
```

Notice the parallel with the `Landscape` from Part II: `Landscape::evaluate` was
a one-shot score; `Environment::step` is score-per-timestep inside an episode.

> **Foundations link.** The agent–environment loop, MDP formulation, and the
> meaning of discounted cumulative reward are introduced in
> [Reinforcement Learning](../part-1-foundations/30-reinforcement-learning.md).

## The agent: DQN

A **Deep Q-Network** approximates the action-value function \\(Q(s, a; \theta)\\)
with a neural network and stabilises training with experience replay and a
target network (see the foundations chapter for the derivation).

Two things differ from the evolutionary side:

- **You bring the network.** `DqnAgent` is generic over a `DqnModel` — a Burn
  module you define (e.g. a small MLP: `4 → 64 → 64 → 2` for CartPole). The
  input/output sizes live in *the model*, not in a config struct.
- **The backend must be autodiff-capable.** DQN learns by backpropagation, so
  the backend is `Autodiff<Flex>`, not bare `Flex`.

```rust,no_run
use burn::backend::{Autodiff, Flex};
use rlevo::rl::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo::rl::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;

type B = Autodiff<Flex>;

let device = Default::default();
let config = DqnTrainingConfigBuilder::new()
    .batch_size(64)
    .gamma(0.99)
    .learning_rate(5e-4)
    .epsilon_start(1.0)
    .epsilon_end(0.05)
    .epsilon_decay(0.9995)        // multiplicative per-step decay, not a step count
    .replay_buffer_capacity(50_000)
    .learning_starts(1_000)       // fill the buffer before learning
    .train_frequency(4)           // one gradient step every 4 env steps
    .target_update_frequency(500)
    .build();

let model: DqnMlp<B> = DqnMlp::new(&device);   // your network — see below
let mut agent = DqnAgent::new(model, config, device);
```

`DqnTrainingConfig::default()` also exists with sensible CartPole-shaped values.

## The training loop

Unlike ask/tell, the RL loop acts *sequentially* and learns after individual
steps. The agent exposes the loop as a handful of methods:

```rust,no_run
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
let mut snapshot = env.reset()?;

for _step in 0..50_000 {
    let obs = snapshot.observation().clone();
    let action = agent.act(&obs, &mut rng);            // ε-greedy

    let next = env.step(action)?;
    let reward: f32 = (*next.reward()).into();
    let done = next.is_done();

    agent.remember(obs, &action, reward, next.observation().clone(), done);
    agent.on_env_step();

    if agent.should_train() {
        agent.learn_step(&mut rng);                    // one gradient update
    }
    agent.sync_target();                               // periodic target copy
    agent.decay_exploration();

    snapshot = if done { env.reset()? } else { next };
}
```

A provided `train(&mut agent, &mut env, &mut rng, total_steps, max_episode_steps)`
helper wraps exactly this loop with episode bookkeeping, if you would rather not
write it yourself.

This is *not* an ask/tell loop in the evolutionary sense — the agent acts
sequentially and updates after every few steps, not after a full population
evaluation. The connection to ask/tell reappears when we combine evolution and
RL: the EA will `ask` for a population of policies, evaluate each by running a
full episode, and `tell` the scores back.

## Up next

The [next section](04-extending-the-environment.md) shows how to implement the
`Environment` trait for your own domain — the prerequisite for applying any
`rlevo` algorithm to a problem you bring to the library.

> **Foundations link.** The Bellman equation DQN minimises, the role of the
> target network, and experience replay are derived in
> [Reinforcement Learning](../part-1-foundations/30-reinforcement-learning.md).
> Full pseudocode for DQN as implemented in `rlevo` will appear in
> [Appendix B](../appendix-b-rl-algorithms/index.md).

---

*Co-Authored-By: Anthropic Claude Opus 4.8*
*Reviewed-By: (Human) Anthony Torlucci*
