//! Random baseline vs. DQN on [`MemoryEnv`] — quality summary + throughput.
//!
//! `MemoryEnv` exists to demonstrate *why memory matters*: the agent sees a
//! cue at step 0 and must pick the matching fork option hundreds of steps
//! later. A uniformly random policy wins only by luck (~50% on the binary
//! fork, lower end-to-end).
//!
//! This bench keeps that random policy as the baseline and pairs it with a
//! **feedforward** DQN — and that pairing is the point. A memoryless network
//! cannot carry the cue across the corridor, so the DQN is expected to land
//! near chance, *not* to solve the task. The comparison therefore documents a
//! real limitation: clearing this env needs a recurrent or memory-augmented
//! policy, which the library does not yet provide. Treat the DQN column as a
//! "memoryless learners do not help here" control, not a success story.
//!
//! 1. **Quality comparison** — trains a DQN, then prints mean terminal reward
//!    and success rate for the random policy vs. the (near-)greedy DQN policy.
//! 2. **Throughput** — a Criterion group timing per-step rollout cost of the
//!    random policy vs. DQN-greedy inference.
//!
//! The flatten + MLP model over the `7×7×3` observation is shared with the
//! other `*_dqn` benches via [`support`].
//!
//! # Run with
//!
//! ```bash
//! cargo bench -p rlevo --bench grid_memory_rl
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
use rlevo_environments::grids::{GridAction, MemoryConfig, MemoryEnv};

use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::train::train;

use support::GridMlpDqn;

const SEED: u64 = 2026;
/// Flattened observation width: `7 * 7 * 3 = 147`.
const OBS_FEATURES: usize = VIEW_SIZE * VIEW_SIZE * OBS_CHANNELS;
const ACTIONS: usize = GridAction::ACTION_COUNT;
/// Episode step cap baked into the env config.
const MAX_STEPS: usize = 140;
const TRAIN_TIMESTEPS: usize = 40_000;
const EVAL_EPISODES: usize = 200;

type Backend_ = Autodiff<Flex>;
type MemoryAgent = DqnAgent<Backend_, GridMlpDqn<Backend_>, GridObservation, GridAction, 3, 4>;

fn memory_config() -> MemoryConfig {
    MemoryConfig::new(MAX_STEPS, SEED, false)
}

/// Trains a feedforward DQN on a fresh, seeded `MemoryEnv`. See the module
/// docs: this is not expected to solve the task, only to confirm a memoryless
/// learner does not beat chance.
fn train_dqn() -> MemoryAgent {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = MemoryEnv::with_config(memory_config(), false).expect("valid config");

    let config = DqnTrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .tau(0.005)
        .learning_rate(5e-4)
        .epsilon_start(1.0)
        .epsilon_end(0.01)
        .epsilon_decay(0.9997)
        .learning_starts(500)
        .train_frequency(4)
        .target_update_frequency(200)
        .replay_buffer_capacity(20_000)
        .double_q(true)
        .build()
        .expect("valid config");

    let model: GridMlpDqn<Backend_> = GridMlpDqn::new(OBS_FEATURES, 128, ACTIONS, &device);
    let mut agent: MemoryAgent = DqnAgent::new(model, config, device).expect("valid config");

    train(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("dqn training loop");
    agent
}

/// Runs one episode under `next_action`. Returns the terminal reward and
/// whether the correct fork was chosen (reward > 0).
fn roll_out(
    env: &mut MemoryEnv,
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

/// Estimates `(mean_reward, success_rate)` over `EVAL_EPISODES` episodes.
#[allow(clippy::cast_precision_loss)]
fn evaluate(mut next_action: impl FnMut(&GridObservation) -> GridAction) -> (f32, f32) {
    let mut env = MemoryEnv::with_config(memory_config(), false).expect("valid config");
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

fn print_quality_comparison(agent: &MemoryAgent) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let (rand_reward, rand_success) =
        evaluate(|_obs| GridAction::from_index(rng.random_range(0..ACTIONS)));

    let mut eval_rng = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let (dqn_reward, dqn_success) = evaluate(|obs| agent.act(obs, &mut eval_rng));

    println!();
    println!(
        "MemoryEnv policy quality | max_steps={MAX_STEPS} episodes={EVAL_EPISODES} train_steps={TRAIN_TIMESTEPS}"
    );
    println!("  (feedforward DQN is memoryless; near-chance is the expected result)");
    println!("  policy        mean_reward   success_rate");
    println!("  random        {rand_reward:>11.4}   {rand_success:>11.2}");
    println!("  dqn (greedy)  {dqn_reward:>11.4}   {dqn_success:>11.2}");
    println!();
}

/// Drives `steps` of a `next_action`-chosen rollout, resetting on episode end.
fn rollout_steps(steps: usize, mut next_action: impl FnMut(&GridObservation) -> GridAction) {
    let mut env = MemoryEnv::with_config(memory_config(), false).expect("valid config");
    let mut snapshot = env.reset().expect("reset");
    for _ in 0..steps {
        let action = next_action(snapshot.observation());
        snapshot = env.step(action).expect("step");
        if snapshot.is_done() {
            snapshot = env.reset().expect("reset");
        }
    }
}

fn bench_policies(c: &mut Criterion, agent: &MemoryAgent) {
    let mut group = c.benchmark_group("grid_memory_policy_rollout");
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
    let agent = train_dqn();
    print_quality_comparison(&agent);

    let mut criterion = Criterion::default().configure_from_args();
    bench_policies(&mut criterion, &agent);
    criterion.final_summary();
}
