//! Random baseline vs. DQN on [`EmptyEnv`] — quality summary + throughput.
//!
//! A uniformly random policy makes a poor *example* (it never learns), but a
//! useful *bench baseline*: it is the floor every learned policy must clear.
//! This bench pairs that baseline with a Deep Q-Network so the two are
//! measured on the same footing. It does two things:
//!
//! 1. **Quality comparison** — trains a DQN on a small `EmptyEnv`, then prints
//!    mean terminal reward and success rate for the random policy vs. the
//!    (near-)greedy DQN policy. This is the "does learning beat chance?"
//!    sanity check; it is printed once before the timing harness runs.
//! 2. **Throughput** — a Criterion group timing per-step rollout cost of the
//!    random policy vs. DQN-greedy inference (a network forward per step).
//!
//! The DQN model is a flatten + MLP over the `7×7×3` egocentric grid
//! observation (rank-3 → rank-4 batched), mapping to the 7-way
//! [`GridAction`] space; the model and Polyak target update are shared with
//! the other `*_dqn` benches via [`support`].
//!
//! # Run with
//!
//! ```bash
//! cargo bench -p rlevo --bench grid_empty_rl
//! ```

#[path = "support/value_nets.rs"]
mod support;

use std::hint::black_box;

use burn::backend::{Autodiff, Flex};

use criterion::{BenchmarkId, Criterion, Throughput};

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::grids::core::{GridObservation, OBS_CHANNELS, VIEW_SIZE};
use rlevo_environments::grids::{EmptyConfig, EmptyEnv, GridAction};

use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::train::train;
use rlevo_reinforcement_learning::target::TargetUpdate;

use support::GridMlpDqn;

const SEED: u64 = 2026;
/// Flattened observation width: `7 * 7 * 3 = 147`.
const OBS_FEATURES: usize = VIEW_SIZE * VIEW_SIZE * OBS_CHANNELS;
/// Discrete action count for the grid family (turn ×2, forward, pickup,
/// drop, toggle, done).
const ACTIONS: usize = GridAction::ACTION_COUNT;
/// Environment steps used to train the DQN before the comparison.
const TRAIN_TIMESTEPS: usize = 30_000;
/// Episodes used to estimate each policy's reward statistics.
const EVAL_EPISODES: usize = 200;

type Backend_ = Autodiff<Flex>;
type GridAgent = DqnAgent<Backend_, GridMlpDqn<Backend_>, GridObservation, GridAction, 3, 4>;

/// Small, fast-to-solve `EmptyEnv` so DQN converges within `TRAIN_TIMESTEPS`.
fn grid_config() -> EmptyConfig {
    EmptyConfig::new(5, 100, SEED)
}

/// Trains a DQN agent on a fresh, seeded `EmptyEnv`.
///
/// `epsilon_end` is driven near zero so the agent is effectively greedy by
/// the time it is handed to the evaluation and throughput stages.
fn train_dqn() -> GridAgent {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = EmptyEnv::with_config(grid_config(), false).expect("valid config");

    let config = DqnTrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .target_update(TargetUpdate::polyak(0.005, 1))
        .learning_rate(5e-4)
        .epsilon_start(1.0)
        .epsilon_end(0.01)
        .epsilon_decay(0.9995)
        .learning_starts(500)
        .train_frequency(4)
        .replay_buffer_capacity(20_000)
        .double_q(true)
        .build()
        .expect("valid config");

    let model: GridMlpDqn<Backend_> = GridMlpDqn::new(OBS_FEATURES, 128, ACTIONS, &device);
    let mut agent: GridAgent = DqnAgent::new(model, config, device).expect("valid config");

    train(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("dqn training loop");
    agent
}

/// Rolls a single episode under a policy that picks each action via
/// `next_action`. Returns the terminal reward and whether the goal was
/// reached (reward > 0).
fn roll_out(
    env: &mut EmptyEnv,
    mut next_action: impl FnMut(&GridObservation) -> GridAction,
) -> (f32, bool) {
    let mut snapshot = env.reset().expect("reset");
    loop {
        let action = next_action(snapshot.observation());
        snapshot = env.step(action).expect("step");
        if snapshot.is_done() {
            let reward = f32::from(*snapshot.reward());
            return (reward, reward > 0.0);
        }
    }
}

/// Estimates `(mean_reward, success_rate)` over `EVAL_EPISODES` episodes for a
/// policy described by `next_action`.
#[allow(clippy::cast_precision_loss)]
fn evaluate(mut next_action: impl FnMut(&GridObservation) -> GridAction) -> (f32, f32) {
    let mut env = EmptyEnv::with_config(grid_config(), false).expect("valid config");
    let mut total_reward = 0.0_f32;
    let mut successes = 0_usize;
    for _ in 0..EVAL_EPISODES {
        let (reward, solved) = roll_out(&mut env, &mut next_action);
        total_reward += reward;
        successes += usize::from(solved);
    }
    let n = EVAL_EPISODES as f32;
    (total_reward / n, successes as f32 / n)
}

/// Prints the random-vs-DQN reward/success comparison.
fn print_quality_comparison(agent: &GridAgent) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let (rand_reward, rand_success) =
        evaluate(|_obs| GridAction::from_index(rng.random_range(0..ACTIONS)));

    let mut eval_rng = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let (dqn_reward, dqn_success) = evaluate(|obs| agent.act(obs, &mut eval_rng));

    println!();
    println!(
        "EmptyEnv policy quality | grid={}x{} episodes={EVAL_EPISODES} train_steps={TRAIN_TIMESTEPS}",
        grid_config().size,
        grid_config().size,
    );
    println!("  policy        mean_reward   success_rate");
    println!("  random        {rand_reward:>11.4}   {rand_success:>11.2}");
    println!("  dqn (greedy)  {dqn_reward:>11.4}   {dqn_success:>11.2}");
    println!();
}

/// Drives `steps` of a `next_action`-chosen rollout on a fresh seeded env,
/// resetting on episode end. The unit measured by the throughput group.
fn rollout_steps(steps: usize, mut next_action: impl FnMut(&GridObservation) -> GridAction) {
    let mut env = EmptyEnv::with_config(grid_config(), false).expect("valid config");
    let mut snapshot = env.reset().expect("reset");
    for _ in 0..steps {
        let action = next_action(snapshot.observation());
        snapshot = env.step(action).expect("step");
        if snapshot.is_done() {
            snapshot = env.reset().expect("reset");
        }
    }
}

fn bench_policies(c: &mut Criterion, agent: &GridAgent) {
    let mut group = c.benchmark_group("grid_empty_policy_rollout");
    for &steps in &[1_000_usize, 4_000, 16_000] {
        group.throughput(Throughput::Elements(steps as u64));

        group.bench_with_input(BenchmarkId::new("random", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |_obs| {
                    GridAction::from_index(rng.random_range(0..ACTIONS))
                });
            });
        });

        group.bench_with_input(
            BenchmarkId::new("dqn_greedy", steps),
            &steps,
            |b, &steps| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(SEED);
                    rollout_steps(black_box(steps), |obs| agent.act(obs, &mut rng));
                });
            },
        );
    }
    group.finish();
}

fn main() {
    // Train once, print the learning-quality comparison, then run the
    // Criterion timing groups against the shared trained agent.
    let agent = train_dqn();
    print_quality_comparison(&agent);

    let mut criterion = Criterion::default().configure_from_args();
    bench_policies(&mut criterion, &agent);
    criterion.final_summary();
}
