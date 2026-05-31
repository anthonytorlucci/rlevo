//! Random baseline vs. PPO / DDPG / TD3 / SAC on [`Pendulum`] — quality
//! summary + throughput.
//!
//! Pendulum has a continuous torque action and no terminal condition: every
//! episode runs to the [`TimeLimit`] (200 steps). A uniformly random torque
//! policy scores around −1 200 per episode; it is the floor every learned
//! policy must clear. This bench trains four agents sequentially and:
//!
//! 1. **Quality comparison** — prints mean episode return for random vs. each
//!    trained policy (no solve rate — Pendulum is always truncated).
//! 2. **Throughput** — a Criterion group timing per-step rollout cost of the
//!    random policy vs. each trained policy's inference.
//!
//! Each algorithm trains for 100 000 steps; they run serially.
//!
//! # Run with
//!
//! ```bash
//! # Quality summary + Criterion timing
//! cargo bench -p rlevo --bench pendulum_rl
//!
//! # Quality summary only (skip Criterion timing)
//! cargo bench -p rlevo --bench pendulum_rl -- --test
//! ```

#[path = "support/pendulum.rs"]
mod pendulum_support;

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

use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::classic::{Pendulum, PendulumAction, PendulumConfig, PendulumObservation};
use rlevo_environments::wrappers::TimeLimit;

use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_agent::DdpgAgent;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_config::DdpgTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ddpg::train::train as train_ddpg;
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    TanhGaussianPolicyHead, TanhGaussianPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;
use rlevo_reinforcement_learning::algorithms::ppo::train::train_continuous as train_ppo;
use rlevo_reinforcement_learning::algorithms::sac::sac_agent::SacAgent;
use rlevo_reinforcement_learning::algorithms::sac::sac_config::SacTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::sac::train::train as train_sac;
use rlevo_reinforcement_learning::algorithms::td3::td3_agent::Td3Agent;
use rlevo_reinforcement_learning::algorithms::td3::td3_config::Td3TrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::td3::train::train as train_td3;

use pendulum_support::{ActorMlp, CriticMlp, StochasticActor};

const SEED: u64 = 2026;
const OBS_DIM: usize = 3;
const ACTION_DIM: usize = 1;
const HIDDEN: usize = 256;
const TIME_LIMIT: usize = 200;
const TRAIN_TIMESTEPS: usize = 100_000;
const EVAL_EPISODES: usize = 100;

type Backend_ = Autodiff<Flex>;
type Env = TimeLimit<Pendulum>;

type PpoAgent_ = PpoAgent<
    Backend_,
    TanhGaussianPolicyHead<Backend_>,
    ValueMlp<Backend_>,
    PendulumObservation,
    1,
    2,
>;
type DdpgAgent_ = DdpgAgent<
    Backend_,
    ActorMlp<Backend_>,
    CriticMlp<Backend_>,
    PendulumObservation,
    PendulumAction,
    1,
    2,
    1,
    2,
>;
type Td3Agent_ = Td3Agent<
    Backend_,
    ActorMlp<Backend_>,
    CriticMlp<Backend_>,
    PendulumObservation,
    PendulumAction,
    1,
    2,
    1,
    2,
>;
type SacAgent_ = SacAgent<
    Backend_,
    StochasticActor<Backend_>,
    CriticMlp<Backend_>,
    PendulumObservation,
    PendulumAction,
    1,
    2,
    1,
    2,
>;

// ---------------------------------------------------------------------------
// PPO value network — two-layer tanh MLP → scalar
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ValueMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> ValueMlp<B> {
    fn new(device: &<B as BackendTypes>::Device) -> Self {
        Self {
            fc1: LinearConfig::new(OBS_DIM, 64).init(device),
            fc2: LinearConfig::new(64, 64).init(device),
            head: LinearConfig::new(64, 1).init(device),
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

// ---------------------------------------------------------------------------
// Environment factory
// ---------------------------------------------------------------------------

fn make_env() -> Env {
    TimeLimit::new(
        Pendulum::with_config(PendulumConfig {
            seed: SEED,
            ..PendulumConfig::default()
        }),
        TIME_LIMIT,
    )
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

fn train_ppo_agent() -> PpoAgent_ {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let policy: TanhGaussianPolicyHead<Backend_> = TanhGaussianPolicyHeadConfig {
        obs_dim: OBS_DIM,
        hidden: 64,
        action_dim: ACTION_DIM,
        log_std_init: 0.0,
        action_scale: 2.0,
    }
    .init::<Backend_>(&device);
    let value: ValueMlp<Backend_> = ValueMlp::new(&device);
    let config = PpoTrainingConfigBuilder::new()
        .num_envs(1)
        .num_steps(2_048)
        .num_minibatches(32)
        .update_epochs(10)
        .learning_rate(3e-4)
        .clip_coef(0.2)
        .entropy_coef(0.0)
        .value_coef(0.5)
        .gamma(0.9)
        .gae_lambda(0.95)
        .action_log_std_init(0.0)
        .action_scale(2.0)
        .build();
    let total_iterations = TRAIN_TIMESTEPS / config.batch_size().max(1);
    let mut agent: PpoAgent_ = PpoAgent::new(policy, value, config, device, total_iterations);
    train_ppo::<Backend_, _, _, _, _, PendulumAction, _, 1, 1, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        TRAIN_TIMESTEPS,
        0,
    )
    .expect("ppo training");
    agent
}

fn train_ddpg_agent() -> DdpgAgent_ {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let actor: ActorMlp<Backend_> = ActorMlp::new(OBS_DIM, HIDDEN, ACTION_DIM, &device);
    let critic: CriticMlp<Backend_> = CriticMlp::new(OBS_DIM, ACTION_DIM, HIDDEN, &device);
    let config = DdpgTrainingConfigBuilder::new()
        .buffer_capacity(100_000)
        .batch_size(256)
        .learning_starts(5_000)
        .actor_lr(1e-4)
        .critic_lr(1e-3)
        .gamma(0.99)
        .tau(0.005)
        .exploration_noise(0.1)
        .policy_frequency(2)
        .build();
    let mut agent: DdpgAgent_ = DdpgAgent::new(actor, critic, config, device);
    train_ddpg::<Backend_, _, _, _, _, PendulumAction, _, 1, 1, 2, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        TRAIN_TIMESTEPS,
        0,
    )
    .expect("ddpg training");
    agent
}

fn train_td3_agent() -> Td3Agent_ {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let actor: ActorMlp<Backend_> = ActorMlp::new(OBS_DIM, HIDDEN, ACTION_DIM, &device);
    let critic_1: CriticMlp<Backend_> = CriticMlp::new(OBS_DIM, ACTION_DIM, HIDDEN, &device);
    let critic_2: CriticMlp<Backend_> = CriticMlp::new(OBS_DIM, ACTION_DIM, HIDDEN, &device);
    let config = Td3TrainingConfigBuilder::new()
        .buffer_capacity(100_000)
        .batch_size(256)
        .learning_starts(5_000)
        .actor_lr(1e-4)
        .critic_lr(1e-3)
        .gamma(0.99)
        .tau(0.005)
        .exploration_noise(0.1)
        .policy_noise(0.2)
        .noise_clip(0.5)
        .policy_frequency(2)
        .build();
    let mut agent: Td3Agent_ = Td3Agent::new(actor, critic_1, critic_2, config, device);
    train_td3::<Backend_, _, _, _, _, PendulumAction, _, 1, 1, 2, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        TRAIN_TIMESTEPS,
        0,
    )
    .expect("td3 training");
    agent
}

fn train_sac_agent() -> SacAgent_ {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let actor: StochasticActor<Backend_> = StochasticActor::new(OBS_DIM, HIDDEN, ACTION_DIM, &device);
    let critic_1: CriticMlp<Backend_> = CriticMlp::new(OBS_DIM, ACTION_DIM, HIDDEN, &device);
    let critic_2: CriticMlp<Backend_> = CriticMlp::new(OBS_DIM, ACTION_DIM, HIDDEN, &device);
    let config = SacTrainingConfigBuilder::new()
        .buffer_capacity(100_000)
        .batch_size(256)
        .learning_starts(5_000)
        .actor_lr(3e-4)
        .critic_lr(1e-3)
        .alpha_lr(1e-3)
        .gamma(0.99)
        .tau(0.005)
        .autotune(true)
        .initial_alpha(1.0)
        .policy_frequency(2)
        .build();
    let mut agent: SacAgent_ = SacAgent::new(actor, critic_1, critic_2, config, device);
    train_sac::<Backend_, _, _, _, _, PendulumAction, _, 1, 1, 2, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        TRAIN_TIMESTEPS,
        0,
    )
    .expect("sac training");
    agent
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

fn roll_out(
    env: &mut Env,
    mut next_action: impl FnMut(&PendulumObservation) -> PendulumAction,
) -> f32 {
    let mut snap = env.reset().expect("reset");
    let mut ret = 0.0_f32;
    loop {
        let action = next_action(snap.observation());
        snap = env.step(action).expect("step");
        ret += f32::from(*snap.reward());
        if snap.is_done() {
            return ret;
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn evaluate(mut next_action: impl FnMut(&PendulumObservation) -> PendulumAction) -> f32 {
    let mut env = make_env();
    let mut total = 0.0_f32;
    for _ in 0..EVAL_EPISODES {
        total += roll_out(&mut env, &mut next_action);
    }
    total / EVAL_EPISODES as f32
}

fn rollout_steps(
    steps: usize,
    mut next_action: impl FnMut(&PendulumObservation) -> PendulumAction,
) {
    let mut env = make_env();
    let mut snap = env.reset().expect("reset");
    for _ in 0..steps {
        let action = next_action(snap.observation());
        snap = env.step(action).expect("step");
        if snap.is_done() {
            snap = env.reset().expect("reset");
        }
    }
}

// ---------------------------------------------------------------------------
// Quality comparison
// ---------------------------------------------------------------------------

fn random_action(rng: &mut StdRng) -> PendulumAction {
    let torque: f32 = rng.random_range(-2.0_f32..=2.0_f32);
    PendulumAction::new(torque).expect("valid random torque")
}

fn print_quality_comparison(
    ppo: &PpoAgent_,
    ddpg: &DdpgAgent_,
    td3: &Td3Agent_,
    sac: &SacAgent_,
) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let rand_ret = evaluate(|_| random_action(&mut rng));

    let mut r = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let ppo_ret = evaluate(|obs| {
        let row = ppo.act(obs, &mut r).env_row;
        PendulumAction::new(row[0]).expect("ppo action in range")
    });

    let mut r = StdRng::seed_from_u64(SEED.wrapping_add(2));
    let ddpg_ret = evaluate(|obs| ddpg.act(obs, false, &mut r));

    let mut r = StdRng::seed_from_u64(SEED.wrapping_add(3));
    let td3_ret = evaluate(|obs| td3.act(obs, false, &mut r));

    let mut r = StdRng::seed_from_u64(SEED.wrapping_add(4));
    let sac_ret = evaluate(|obs| sac.act(obs, false, &mut r));

    println!();
    println!(
        "Pendulum policy quality | time_limit={TIME_LIMIT} episodes={EVAL_EPISODES} train_steps={TRAIN_TIMESTEPS}"
    );
    println!("  policy        mean_return");
    println!("  random        {rand_ret:>11.2}");
    println!("  ppo           {ppo_ret:>11.2}");
    println!("  ddpg          {ddpg_ret:>11.2}");
    println!("  td3           {td3_ret:>11.2}");
    println!("  sac           {sac_ret:>11.2}");
    println!();
}

// ---------------------------------------------------------------------------
// Criterion throughput
// ---------------------------------------------------------------------------

fn bench_policies(
    c: &mut Criterion,
    ppo: &PpoAgent_,
    ddpg: &DdpgAgent_,
    td3: &Td3Agent_,
    sac: &SacAgent_,
) {
    let mut group = c.benchmark_group("pendulum_policy_rollout");
    for &steps in &[1_000_usize, 4_000, 16_000] {
        group.throughput(Throughput::Elements(steps as u64));

        group.bench_with_input(BenchmarkId::new("random", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |_| random_action(&mut rng));
            });
        });

        group.bench_with_input(BenchmarkId::new("ppo", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |obs| {
                    PendulumAction::new(ppo.act(obs, &mut rng).env_row[0])
                        .expect("ppo action in range")
                });
            });
        });

        group.bench_with_input(BenchmarkId::new("ddpg", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |obs| ddpg.act(obs, false, &mut rng));
            });
        });

        group.bench_with_input(BenchmarkId::new("td3", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |obs| td3.act(obs, false, &mut rng));
            });
        });

        group.bench_with_input(BenchmarkId::new("sac", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |obs| sac.act(obs, false, &mut rng));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let ppo = train_ppo_agent();
    let ddpg = train_ddpg_agent();
    let td3 = train_td3_agent();
    let sac = train_sac_agent();

    print_quality_comparison(&ppo, &ddpg, &td3, &sac);

    let mut criterion = Criterion::default().configure_from_args();
    bench_policies(&mut criterion, &ppo, &ddpg, &td3, &sac);
    criterion.final_summary();
}
