//! Random baseline vs. PPO on [`CartPole`] — quality summary + throughput.
//!
//! A uniformly random left/right policy keeps the pole up for only a couple
//! dozen steps; it is the floor a learned policy must clear. This bench pairs
//! that baseline with a PPO agent (categorical policy head + MLP critic) over
//! the 2-way discrete action space and:
//!
//! 1. **Quality comparison** — trains a PPO agent, then prints mean episode
//!    return and solve rate (fraction of episodes that survive to the time
//!    limit) for the random policy vs. the trained PPO policy.
//! 2. **Throughput** — a Criterion group timing per-step rollout cost of the
//!    random policy vs. PPO inference.
//!
//! `CartPole` rewards `+1` per step until the pole falls or the cart leaves
//! the track; episodes are capped at 500 steps via [`TimeLimit`], so a perfect
//! policy scores 500 and "solving" means reaching that cap (truncation rather
//! than termination). Unlike the `*_dqn` benches this one defines its own
//! actor/critic networks inline — PPO needs both a policy head and a separate
//! value network, neither of which the shared DQN scaffolding provides.
//!
//! # Run with
//!
//! ```bash
//! cargo bench -p rlevo --bench cartpole_ppo
//! ```

use std::hint::black_box;

use burn::backend::{Autodiff, Flex};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};

use criterion::{BenchmarkId, Criterion, Throughput};

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::classic::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};
use rlevo_environments::wrappers::TimeLimit;

use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;
use rlevo_reinforcement_learning::algorithms::ppo::train::train_discrete;

const SEED: u64 = 2026;
/// Observation width: `[cart_pos, cart_vel, pole_angle, pole_ang_vel]`.
const OBS_FEATURES: usize = 4;
const ACTIONS: usize = CartPoleAction::ACTION_COUNT;
const HIDDEN: usize = 64;
/// Per-episode step cap (Gymnasium `CartPole-v1` default).
const TIME_LIMIT: usize = 500;
const TRAIN_TIMESTEPS: usize = 100_000;
const EVAL_EPISODES: usize = 100;

type Backend_ = Autodiff<Flex>;
type Env = TimeLimit<CartPole>;
type CartPoleAgent =
    PpoAgent<Backend_, CategoricalPolicyHead<Backend_>, ValueMlp<Backend_>, CartPoleObservation, 1, 2>;

// ---------------------------------------------------------------------------
// Value network — two-hidden-layer tanh MLP (CleanRL-style critic).
// ---------------------------------------------------------------------------

/// Two-hidden-layer tanh MLP mapping batched `[batch, features]` observations
/// to a scalar state-value per row.
#[derive(Module, Debug)]
pub struct ValueMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> ValueMlp<B> {
    fn new(obs_dim: usize, hidden: usize, device: &<B as BackendTypes>::Device) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            head: LinearConfig::new(hidden, 1).init(device),
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = tanh(self.fc1.forward(obs));
        let h = tanh(self.fc2.forward(h));
        self.head.forward(h).squeeze_dim::<1>(1)
    }
}

impl<B: AutodiffBackend> PpoValue<B, 2> for ValueMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs)
    }
}

fn make_env() -> Env {
    TimeLimit::new(
        CartPole::with_config(CartPoleConfig {
            seed: SEED,
            ..CartPoleConfig::default()
        }),
        TIME_LIMIT,
    )
}

/// Maps a PPO env-action row (a single categorical index) to a `CartPoleAction`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn action_from_row(row: &[f32]) -> CartPoleAction {
    CartPoleAction::from_index(row[0] as usize)
}

/// Trains a PPO agent on a fresh, `TimeLimit`-capped `CartPole`.
fn train_ppo() -> CartPoleAgent {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();

    let policy: CategoricalPolicyHead<Backend_> = CategoricalPolicyHeadConfig {
        obs_dim: OBS_FEATURES,
        hidden: HIDDEN,
        num_actions: ACTIONS,
    }
    .init::<Backend_>(&device);
    let value: ValueMlp<Backend_> = ValueMlp::new(OBS_FEATURES, HIDDEN, &device);

    let config = PpoTrainingConfigBuilder::new()
        .num_envs(1)
        .num_steps(128)
        .num_minibatches(4)
        .update_epochs(4)
        .learning_rate(2.5e-4)
        .clip_coef(0.2)
        .entropy_coef(0.01)
        .value_coef(0.5)
        .gamma(0.99)
        .gae_lambda(0.95)
        .build();

    let total_iterations = TRAIN_TIMESTEPS / config.batch_size().max(1);
    let mut agent: CartPoleAgent =
        PpoAgent::new(policy, value, config, device, total_iterations);

    train_discrete::<Backend_, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        TRAIN_TIMESTEPS,
        0,
    )
    .expect("ppo training loop");
    agent
}

/// Runs one capped episode under `next_action`. Returns the episode return
/// (sum of rewards) and whether it survived to the time limit (truncation, as
/// opposed to terminating early on a fall / off-track failure).
fn roll_out(
    env: &mut Env,
    mut next_action: impl FnMut(&CartPoleObservation) -> CartPoleAction,
) -> (f32, bool) {
    let mut snapshot = env.reset().expect("reset");
    let mut episode_return = 0.0_f32;
    loop {
        let action = next_action(snapshot.observation());
        snapshot = env.step(action).expect("step");
        episode_return += f32::from(*snapshot.reward());
        if snapshot.is_done() {
            return (episode_return, snapshot.is_truncated());
        }
    }
}

/// Estimates `(mean_return, solve_rate)` over `EVAL_EPISODES` episodes, where
/// "solved" means surviving the full `TIME_LIMIT`.
#[allow(clippy::cast_precision_loss)]
fn evaluate(mut next_action: impl FnMut(&CartPoleObservation) -> CartPoleAction) -> (f32, f32) {
    let mut env = make_env();
    let mut total_return = 0.0_f32;
    let mut solved = 0_usize;
    for _ in 0..EVAL_EPISODES {
        let (episode_return, survived) = roll_out(&mut env, &mut next_action);
        total_return += episode_return;
        solved += usize::from(survived);
    }
    let n = EVAL_EPISODES as f32;
    (total_return / n, solved as f32 / n)
}

fn print_quality_comparison(agent: &CartPoleAgent) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let (rand_return, rand_solve) =
        evaluate(|_obs| CartPoleAction::from_index(rng.random_range(0..ACTIONS)));

    let mut eval_rng = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let (ppo_return, ppo_solve) =
        evaluate(|obs| action_from_row(&agent.act(obs, &mut eval_rng).env_row));

    println!();
    println!(
        "CartPole policy quality | time_limit={TIME_LIMIT} episodes={EVAL_EPISODES} train_steps={TRAIN_TIMESTEPS}"
    );
    println!("  policy        mean_return   solve_rate");
    println!("  random        {rand_return:>11.2}   {rand_solve:>10.2}");
    println!("  ppo           {ppo_return:>11.2}   {ppo_solve:>10.2}");
    println!();
}

/// Drives `steps` of a `next_action`-chosen rollout, resetting on episode end.
fn rollout_steps(
    steps: usize,
    mut next_action: impl FnMut(&CartPoleObservation) -> CartPoleAction,
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

fn bench_policies(c: &mut Criterion, agent: &CartPoleAgent) {
    let mut group = c.benchmark_group("cartpole_policy_rollout");
    for &steps in &[1_000_usize, 4_000, 16_000] {
        group.throughput(Throughput::Elements(steps as u64));

        group.bench_with_input(BenchmarkId::new("random", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |_obs| {
                    CartPoleAction::from_index(rng.random_range(0..ACTIONS))
                });
            });
        });

        group.bench_with_input(BenchmarkId::new("ppo", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |obs| {
                    action_from_row(&agent.act(obs, &mut rng).env_row)
                });
            });
        });
    }
    group.finish();
}

fn main() {
    let agent = train_ppo();
    print_quality_comparison(&agent);

    let mut criterion = Criterion::default().configure_from_args();
    bench_policies(&mut criterion, &agent);
    criterion.final_summary();
}
