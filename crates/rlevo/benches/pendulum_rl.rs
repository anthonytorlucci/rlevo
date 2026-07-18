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
const OBS_RANK: usize = 3; // the order of the observation space
const ACTION_RANK: usize = 1; // the order of the action space
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

/// Two-layer tanh MLP that maps a batch of observations to scalar state-values
/// for the PPO critic. Architecture: `obs_dim → 64 → 64 → 1` with `tanh`
/// activations. Implements [`PpoValue`] and is local to this bench because PPO
/// requires a separate value network that DDPG/TD3/SAC do not.
#[derive(Module, Debug)]
pub struct ValueMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> ValueMlp<B> {
    fn new(device: &<B as BackendTypes>::Device) -> Self {
        Self {
            fc1: LinearConfig::new(OBS_RANK, 64).init(device),
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
        })
        .expect("valid config"),
        TIME_LIMIT,
    )
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

fn train_ppo_agent() -> PpoAgent_ {
    let device = Default::default();
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let policy: TanhGaussianPolicyHead<Backend_> = TanhGaussianPolicyHeadConfig {
        obs_dim: OBS_RANK,
        hidden: 64,
        action_dim: ACTION_RANK,
        log_std_init: 0.0,
        log_std_min: -20.0,
        log_std_max: 2.0,
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
        .build()
        .expect("valid config");
    let total_iterations = TRAIN_TIMESTEPS / config.batch_size().max(1);
    let mut agent: PpoAgent_ =
        PpoAgent::new(policy, value, config, device, total_iterations).expect("valid config");
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
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let actor: ActorMlp<Backend_> = ActorMlp::new(OBS_RANK, HIDDEN, ACTION_RANK, &device);
    let critic: CriticMlp<Backend_> = CriticMlp::new(OBS_RANK, ACTION_RANK, HIDDEN, &device);
    let config = DdpgTrainingConfigBuilder::new()
        .replay_buffer_capacity(100_000)
        .batch_size(256)
        .learning_starts(5_000)
        .actor_lr(1e-4)
        .critic_lr(1e-3)
        .gamma(0.99)
        .tau(0.005)
        .exploration_noise(0.1)
        .policy_frequency(2)
        .build()
        .expect("valid config");
    let mut agent: DdpgAgent_ =
        DdpgAgent::new(actor, critic, config, device).expect("valid config");
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
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let actor: ActorMlp<Backend_> = ActorMlp::new(OBS_RANK, HIDDEN, ACTION_RANK, &device);
    let critic_1: CriticMlp<Backend_> = CriticMlp::new(OBS_RANK, ACTION_RANK, HIDDEN, &device);
    let critic_2: CriticMlp<Backend_> = CriticMlp::new(OBS_RANK, ACTION_RANK, HIDDEN, &device);
    let config = Td3TrainingConfigBuilder::new()
        .replay_buffer_capacity(100_000)
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
        .build()
        .expect("valid config");
    let mut agent: Td3Agent_ =
        Td3Agent::new(actor, critic_1, critic_2, config, device).expect("valid config");
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
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let actor: StochasticActor<Backend_> =
        StochasticActor::new(OBS_RANK, HIDDEN, ACTION_RANK, &device);
    let critic_1: CriticMlp<Backend_> = CriticMlp::new(OBS_RANK, ACTION_RANK, HIDDEN, &device);
    let critic_2: CriticMlp<Backend_> = CriticMlp::new(OBS_RANK, ACTION_RANK, HIDDEN, &device);
    let config = SacTrainingConfigBuilder::new()
        .replay_buffer_capacity(100_000)
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
        .build()
        .expect("valid config");
    let mut agent: SacAgent_ =
        SacAgent::new(actor, critic_1, critic_2, config, device).expect("valid config");
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

/// Runs a single episode to completion and returns the undiscounted return.
///
/// Drives the environment with `next_action` until `snap.is_done()` is true.
/// Because [`Env`] wraps [`Pendulum`] in a [`TimeLimit`] of 200 steps,
/// termination is always by truncation — there is no natural episode end.
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

/// Evaluates a policy over [`EVAL_EPISODES`] episodes and returns the mean
/// undiscounted return. Constructs its own fresh environment so the call is
/// self-contained and does not interfere with a caller's environment state.
///
/// `cast_precision_loss` is allowed because dividing by [`EVAL_EPISODES`]
/// (a `usize`) is exact for the episode counts used here.
#[allow(clippy::cast_precision_loss)]
fn evaluate(mut next_action: impl FnMut(&PendulumObservation) -> PendulumAction) -> f32 {
    let mut env = make_env();
    let mut total = 0.0_f32;
    for _ in 0..EVAL_EPISODES {
        total += roll_out(&mut env, &mut next_action);
    }
    total / EVAL_EPISODES as f32
}

/// Drives the environment for exactly `steps` environment steps, resetting
/// automatically at episode boundaries. Used by the Criterion throughput
/// benchmarks to measure per-step inference cost independent of episode length.
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

fn print_quality_comparison(ppo: &PpoAgent_, ddpg: &DdpgAgent_, td3: &Td3Agent_, sac: &SacAgent_) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let rand_ret = evaluate(|_| random_action(&mut rng));

    // Learned policies are evaluated with their deterministic (greedy) policy
    // against a once-snapshotted inner (non-autodiff) network. `PpoAgent::act`
    // samples from the stochastic policy — that injects exploration noise into
    // the score; the deterministic mean is the policy we actually want to
    // measure. The value-based actors run far cheaper on the inner backend,
    // skipping the per-step autodiff graph. SAC's `act(_, false, _)` already
    // runs deterministically on its own inner snapshot, so it stays as-is.
    let ppo_infer = ppo.inference_net();
    let ppo_ret = evaluate(|obs| {
        let row = ppo.act_greedy_env_row_with(&ppo_infer, obs);
        PendulumAction::new(row[0]).expect("ppo action in range")
    });

    let ddpg_infer = ddpg.inference_net();
    let ddpg_ret = evaluate(|obs| ddpg.act_with(&ddpg_infer, obs));

    let td3_infer = td3.inference_net();
    let td3_ret = evaluate(|obs| td3.act_with(&td3_infer, obs));

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
    // Snapshot each actor onto the inner (non-autodiff) backend once; per-step
    // inference then skips autodiff-graph construction entirely, the dominant
    // cost at batch size 1. SAC already snapshots internally in `act`.
    let ppo_infer = ppo.inference_net();
    let ddpg_infer = ddpg.inference_net();
    let td3_infer = td3.inference_net();

    let mut group = c.benchmark_group("pendulum_policy_rollout");
    // Per-step throughput is independent of rollout length (the rollout is
    // linear and `Throughput::Elements` normalises per step), so a single size
    // captures the same signal the 1k/4k/16k sweep did. Ten samples keep a
    // usable confidence interval without multi-second-per-iteration runs.
    group.sample_size(10);
    let steps = 4_000_usize;
    group.throughput(Throughput::Elements(steps as u64));

    group.bench_with_input(BenchmarkId::new("random", steps), &steps, |b, &steps| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(SEED);
            rollout_steps(black_box(steps), |_| random_action(&mut rng));
        });
    });

    // Learned policies time their greedy (evaluation) inference path on the
    // inner backend, matching how the quality comparison rolls them out.
    group.bench_with_input(BenchmarkId::new("ppo", steps), &steps, |b, &steps| {
        b.iter(|| {
            rollout_steps(black_box(steps), |obs| {
                PendulumAction::new(ppo.act_greedy_env_row_with(&ppo_infer, obs)[0])
                    .expect("ppo action in range")
            });
        });
    });

    group.bench_with_input(BenchmarkId::new("ddpg", steps), &steps, |b, &steps| {
        b.iter(|| {
            rollout_steps(black_box(steps), |obs| ddpg.act_with(&ddpg_infer, obs));
        });
    });

    group.bench_with_input(BenchmarkId::new("td3", steps), &steps, |b, &steps| {
        b.iter(|| {
            rollout_steps(black_box(steps), |obs| td3.act_with(&td3_infer, obs));
        });
    });

    group.bench_with_input(BenchmarkId::new("sac", steps), &steps, |b, &steps| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(SEED);
            rollout_steps(black_box(steps), |obs| sac.act(obs, false, &mut rng));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    // Reproducibility has two requirements, both handled here + in each trainer:
    //
    // 1. Seed the backend RNG (`<Backend_>::seed` in every `train_*_agent`). The
    //    `StdRng` we thread through training only drives action sampling/replay;
    //    network weight init (`LinearConfig::init`) draws from the *backend* RNG,
    //    which is seeded from entropy unless we set it. This is the dominant
    //    source of run-to-run policy divergence.
    // 2. Pin the global rayon pool to one thread (below). Flex parallelises
    //    matmul through `gemm`'s rayon feature, and parallel float accumulation
    //    is non-associative, so multi-threaded training still drifts bitwise even
    //    with a seeded init. Single-threaded gemm fixes the reduction order.
    //    `.ok()` because the pool may already be initialised (e.g. under `--test`).
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();

    let ppo = train_ppo_agent();
    let ddpg = train_ddpg_agent();
    let td3 = train_td3_agent();
    let sac = train_sac_agent();

    print_quality_comparison(&ppo, &ddpg, &td3, &sac);

    let mut criterion = Criterion::default().configure_from_args();
    bench_policies(&mut criterion, &ppo, &ddpg, &td3, &sac);
    criterion.final_summary();
}
