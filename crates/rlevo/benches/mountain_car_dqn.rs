//! Random baseline vs. DQN on [`MountainCar`] — quality summary + throughput.
//!
//! A uniformly random policy essentially never escapes the valley; it is the
//! floor a learned policy must clear. This bench pairs that baseline with a
//! DQN over the 3-way discrete action space and:
//!
//! 1. **Quality comparison** — trains a DQN, then prints mean episode return
//!    and goal-reach (termination) rate for the random policy vs. the
//!    (near-)greedy DQN policy.
//! 2. **Throughput** — a Criterion group timing per-step rollout cost of the
//!    random policy vs. DQN-greedy inference.
//!
//! `MountainCar`'s reward is `-1` per step with no shaping, which makes it a
//! notoriously hard exploration problem for value-based methods: the DQN
//! needs a generous step budget to find the goal at all, and may still only
//! partially close the gap to a hand-tuned solver within this bench. Episodes
//! are capped at 200 steps (the Gymnasium default) via [`TimeLimit`]. The MLP
//! model over the rank-1 `[2]` observation is shared with the other `*_dqn`
//! benches via [`support`].
//!
//! # Run with
//!
//! ```bash
//! cargo bench -p rlevo --bench mountain_car_dqn
//! ```

#[path = "support/dqn.rs"]
mod support;

use std::hint::black_box;

use burn::backend::{Autodiff, Flex};

use criterion::{BenchmarkId, Criterion, Throughput};

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::classic::{
    MountainCar, MountainCarAction, MountainCarConfig, MountainCarObservation,
};
use rlevo_environments::wrappers::TimeLimit;

use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::train::train;

use support::VecMlpDqn;

const SEED: u64 = 2026;
/// Observation width: `[position, velocity]`.
const OBS_FEATURES: usize = 2;
const ACTIONS: usize = MountainCarAction::ACTION_COUNT;
/// Per-episode step cap (Gymnasium default).
const TIME_LIMIT: usize = 200;
const TRAIN_TIMESTEPS: usize = 120_000;
const EVAL_EPISODES: usize = 100;

type Backend_ = Autodiff<Flex>;
type Env = TimeLimit<MountainCar>;
type MountainCarAgent =
    DqnAgent<Backend_, VecMlpDqn<Backend_>, MountainCarObservation, MountainCarAction, 1, 2>;

fn make_env() -> Env {
    TimeLimit::new(
        MountainCar::with_config(MountainCarConfig::default()),
        TIME_LIMIT,
    )
}

/// Trains a DQN agent on a fresh, `TimeLimit`-capped `MountainCar`.
fn train_dqn() -> MountainCarAgent {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();

    let config = DqnTrainingConfigBuilder::new()
        .batch_size(128)
        .gamma(0.99)
        .tau(0.005)
        .learning_rate(1e-3)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay(0.9999)
        .learning_starts(1_000)
        .train_frequency(4)
        .target_update_frequency(500)
        .replay_buffer_capacity(100_000)
        .double_q(true)
        .build();

    let model: VecMlpDqn<Backend_> = VecMlpDqn::new(OBS_FEATURES, 128, ACTIONS, &device);
    let mut agent: MountainCarAgent = DqnAgent::new(model, config, device);

    train(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("dqn training loop");
    agent
}

/// Runs one capped episode under `next_action`. Returns the episode return
/// (sum of rewards) and whether it ended on the goal (termination, not the
/// time-limit truncation).
fn roll_out(
    env: &mut Env,
    mut next_action: impl FnMut(&MountainCarObservation) -> MountainCarAction,
) -> (f32, bool) {
    let mut snapshot = env.reset().expect("reset");
    let mut episode_return = 0.0_f32;
    loop {
        let action = next_action(snapshot.observation());
        snapshot = env.step(action).expect("step");
        episode_return += f32::from(*snapshot.reward());
        if snapshot.is_done() {
            return (episode_return, snapshot.is_terminated());
        }
    }
}

/// Estimates `(mean_return, goal_rate)` over `EVAL_EPISODES` episodes.
#[allow(clippy::cast_precision_loss)]
fn evaluate(
    mut next_action: impl FnMut(&MountainCarObservation) -> MountainCarAction,
) -> (f32, f32) {
    let mut env = make_env();
    let mut total_return = 0.0_f32;
    let mut goals = 0_usize;
    for _ in 0..EVAL_EPISODES {
        let (episode_return, reached) = roll_out(&mut env, &mut next_action);
        total_return += episode_return;
        goals += usize::from(reached);
    }
    let n = EVAL_EPISODES as f32;
    (total_return / n, goals as f32 / n)
}

fn print_quality_comparison(agent: &MountainCarAgent) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let (rand_return, rand_goal) =
        evaluate(|_obs| MountainCarAction::from_index(rng.random_range(0..ACTIONS)));

    let mut eval_rng = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let (dqn_return, dqn_goal) = evaluate(|obs| agent.act(obs, &mut eval_rng));

    println!();
    println!(
        "MountainCar policy quality | time_limit={TIME_LIMIT} episodes={EVAL_EPISODES} train_steps={TRAIN_TIMESTEPS}"
    );
    println!("  policy        mean_return   goal_rate");
    println!("  random        {rand_return:>11.2}   {rand_goal:>9.2}");
    println!("  dqn (greedy)  {dqn_return:>11.2}   {dqn_goal:>9.2}");
    println!();
}

/// Drives `steps` of a `next_action`-chosen rollout, resetting on episode end.
fn rollout_steps(
    steps: usize,
    mut next_action: impl FnMut(&MountainCarObservation) -> MountainCarAction,
) {
    let mut env = make_env();
    let mut snapshot = env.reset().expect("reset");
    for _ in 0..steps {
        let action = next_action(snapshot.observation());
        snapshot = env.step(action).expect("step");
        if snapshot.is_done() {
            snapshot = env.reset().expect("reset");
        }
    }
}

fn bench_policies(c: &mut Criterion, agent: &MountainCarAgent) {
    let mut group = c.benchmark_group("mountain_car_policy_rollout");
    for &steps in &[1_000_usize, 4_000, 16_000] {
        group.throughput(Throughput::Elements(steps as u64));

        group.bench_with_input(BenchmarkId::new("random", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |_obs| {
                    MountainCarAction::from_index(rng.random_range(0..ACTIONS))
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
