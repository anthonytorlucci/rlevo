//! Random baseline vs. DQN / C51 / QR-DQN / PPG on [`CartPole`] — quality
//! summary + throughput.
//!
//! A uniformly random left/right policy keeps the pole up for only a couple
//! dozen steps; it is the floor every learned policy must clear. This bench
//! trains four agents sequentially and:
//!
//! 1. **Quality comparison** — prints mean episode return and solve rate
//!    (fraction of episodes surviving the time limit) for random vs. each
//!    trained policy.
//! 2. **Throughput** — a Criterion group timing per-step rollout cost of the
//!    random policy vs. each trained policy's inference.
//!
//! `CartPole` rewards `+1` per step; episodes are capped at 500 steps via
//! [`TimeLimit`], so "solving" means reaching truncation rather than
//! termination. Each algorithm trains for 50 000 steps; they run serially.
//!
//! # Run with
//!
//! ```bash
//! # Quality summary + Criterion timing
//! cargo bench -p rlevo --bench cartpole_rl
//!
//! # Quality summary only (skip Criterion timing)
//! cargo bench -p rlevo --bench cartpole_rl -- --test
//! ```

#[path = "support/value_nets.rs"]
mod value_nets;

use std::hint::black_box;

use burn::backend::{Autodiff, Flex};
use burn::tensor::backend::Backend;

use criterion::{BenchmarkId, Criterion, Throughput};

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::classic::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};
use rlevo_environments::wrappers::TimeLimit;

use rlevo_reinforcement_learning::algorithms::c51::c51_agent::C51Agent;
use rlevo_reinforcement_learning::algorithms::c51::c51_config::C51TrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::c51::train::train as train_c51;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::train::train as train_dqn;
use rlevo_reinforcement_learning::algorithms::ppg::policies::{
    PpgCategoricalPolicyHead, PpgCategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppg::ppg_agent::PpgAgent;
use rlevo_reinforcement_learning::algorithms::ppg::ppg_config::PpgConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppg::train::train_discrete as train_ppg;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_agent::QrDqnAgent;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::qrdqn::train::train as train_qrdqn;

use value_nets::{C51Mlp, QrDqnMlp, ValueMlp, VecMlpDqn};

const SEED: u64 = 2026;
const OBS_FEATURES: usize = 4;
const ACTIONS: usize = CartPoleAction::ACTION_COUNT;
const HIDDEN: usize = 64;
const TIME_LIMIT: usize = 500;
const TRAIN_TIMESTEPS: usize = 50_000;
const EVAL_EPISODES: usize = 100;
const NUM_ATOMS: usize = 51;
const NUM_QUANTILES: usize = 200;

type Backend_ = Autodiff<Flex>;
type Env = TimeLimit<CartPole>;

type DqnCartPoleAgent =
    DqnAgent<Backend_, VecMlpDqn<Backend_>, CartPoleObservation, CartPoleAction, 1, 2>;
type C51CartPoleAgent =
    C51Agent<Backend_, C51Mlp<Backend_>, CartPoleObservation, CartPoleAction, 1, 2>;
type QrDqnCartPoleAgent =
    QrDqnAgent<Backend_, QrDqnMlp<Backend_>, CartPoleObservation, CartPoleAction, 1, 2>;
type PpgCartPoleAgent = PpgAgent<
    Backend_,
    PpgCategoricalPolicyHead<Backend_>,
    ValueMlp<Backend_>,
    CartPoleObservation,
    1,
    2,
>;

// ---------------------------------------------------------------------------
// Environment factory
// ---------------------------------------------------------------------------

fn make_env() -> Env {
    TimeLimit::new(
        CartPole::with_config(CartPoleConfig {
            seed: SEED,
            ..CartPoleConfig::default()
        })
        .expect("valid config"),
        TIME_LIMIT,
    )
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

fn train_dqn_agent() -> DqnCartPoleAgent {
    let device = Default::default();
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let config = DqnTrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .tau(0.005)
        .learning_rate(5e-4)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay(0.9995)
        .learning_starts(1_000)
        .train_frequency(4)
        .target_update_frequency(500)
        .replay_buffer_capacity(50_000)
        .double_q(false)
        .build()
        .expect("valid config");
    let model: VecMlpDqn<Backend_> = VecMlpDqn::new(OBS_FEATURES, HIDDEN, ACTIONS, &device);
    let mut agent: DqnCartPoleAgent = DqnAgent::new(model, config, device).expect("valid config");
    train_dqn(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("dqn training");
    agent
}

fn train_c51_agent() -> C51CartPoleAgent {
    let device = Default::default();
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let config = C51TrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .tau(0.005)
        .learning_rate(5e-4)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay(0.9995)
        .learning_starts(1_000)
        .train_frequency(4)
        .target_update_frequency(500)
        .replay_buffer_capacity(50_000)
        .num_atoms(NUM_ATOMS)
        .v_min(0.0)
        .v_max(500.0)
        .build()
        .expect("valid config");
    let model: C51Mlp<Backend_> = C51Mlp::new(OBS_FEATURES, HIDDEN, ACTIONS, NUM_ATOMS, &device);
    let mut agent: C51CartPoleAgent = C51Agent::new(model, config, device).expect("valid config");
    train_c51(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("c51 training");
    agent
}

fn train_qrdqn_agent() -> QrDqnCartPoleAgent {
    let device = Default::default();
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let config = QrDqnTrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .tau(0.005)
        .learning_rate(5e-4)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay(0.9995)
        .learning_starts(1_000)
        .train_frequency(4)
        .target_update_frequency(500)
        .replay_buffer_capacity(50_000)
        .num_quantiles(NUM_QUANTILES)
        .kappa(1.0)
        .build()
        .expect("valid config");
    let model: QrDqnMlp<Backend_> =
        QrDqnMlp::new(OBS_FEATURES, HIDDEN, ACTIONS, NUM_QUANTILES, &device);
    let mut agent: QrDqnCartPoleAgent =
        QrDqnAgent::new(model, config, device).expect("valid config");
    train_qrdqn(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("qrdqn training");
    agent
}

fn train_ppg_agent() -> PpgCartPoleAgent {
    let device = Default::default();
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let policy: PpgCategoricalPolicyHead<Backend_> = PpgCategoricalPolicyHeadConfig {
        obs_dim: OBS_FEATURES,
        hidden: HIDDEN,
        num_actions: ACTIONS,
    }
    .try_init::<Backend_>(&device)
    .expect("valid head config");
    let value: ValueMlp<Backend_> = ValueMlp::new(OBS_FEATURES, HIDDEN, &device);
    let config = PpgConfigBuilder::new()
        .with_ppo(|p| {
            PpoTrainingConfigBuilder::new()
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
                .anneal_lr(p.anneal_lr)
                .build()
                .expect("valid config")
        })
        .n_iteration(32)
        .e_aux(6)
        .beta_clone(1.0)
        .build()
        .expect("valid config");
    let total_iterations = TRAIN_TIMESTEPS / config.batch_size().max(1);
    let mut agent: PpgCartPoleAgent =
        PpgAgent::new(policy, value, config, device, total_iterations).expect("valid config");
    train_ppg::<Backend_, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        TRAIN_TIMESTEPS,
        0,
    )
    .expect("ppg training");
    agent
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

/// Runs one complete episode and returns `(cumulative_return, truncated)`.
///
/// `truncated` is `true` when the episode ends because the [`TimeLimit`] fired
/// (the pole survived) and `false` when it ended by termination (pole fell).
fn roll_out(
    env: &mut Env,
    mut next_action: impl FnMut(&CartPoleObservation) -> CartPoleAction,
) -> (f32, bool) {
    let mut snap = env.reset().expect("reset");
    let mut ret = 0.0_f32;
    loop {
        let action = next_action(snap.observation());
        snap = env.step(action).expect("step");
        ret += f32::from(*snap.reward());
        if snap.is_done() {
            return (ret, snap.is_truncated());
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn evaluate(mut next_action: impl FnMut(&CartPoleObservation) -> CartPoleAction) -> (f32, f32) {
    let mut env = make_env();
    let (mut total, mut solved) = (0.0_f32, 0_usize);
    for _ in 0..EVAL_EPISODES {
        let (r, survived) = roll_out(&mut env, &mut next_action);
        total += r;
        solved += usize::from(survived);
    }
    let n = EVAL_EPISODES as f32;
    (total / n, solved as f32 / n)
}

/// Runs exactly `steps` environment steps across episode boundaries, discarding
/// returns. Used by [`bench_policies`] to give Criterion a fixed-length loop
/// whose cost is independent of episode length.
fn rollout_steps(
    steps: usize,
    mut next_action: impl FnMut(&CartPoleObservation) -> CartPoleAction,
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

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn action_from_row(row: &[f32]) -> CartPoleAction {
    CartPoleAction::from_index(row[0] as usize)
}

fn print_quality_comparison(
    dqn: &DqnCartPoleAgent,
    c51: &C51CartPoleAgent,
    qrdqn: &QrDqnCartPoleAgent,
    ppg: &PpgCartPoleAgent,
) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let (rand_ret, rand_solve) =
        evaluate(|_| CartPoleAction::from_index(rng.random_range(0..ACTIONS)));

    // Evaluation uses each agent's greedy policy: `act` is ε-greedy and floors
    // at `epsilon_end`, so it injects exploration noise that destabilises the
    // pole and caps the solve rate even for a well-trained policy. The
    // value-based agents run inference on a once-snapshotted inner (non-autodiff)
    // network — far cheaper than rebuilding an autodiff graph every step.
    let dqn_infer = dqn.inference_net();
    let (dqn_ret, dqn_solve) = evaluate(|obs| dqn.act_greedy_with(&dqn_infer, obs));

    let c51_infer = c51.inference_net();
    let c51_support = c51.inference_support();
    let (c51_ret, c51_solve) = evaluate(|obs| c51.act_greedy_with(&c51_infer, &c51_support, obs));

    let qrdqn_infer = qrdqn.inference_net();
    let (qrdqn_ret, qrdqn_solve) = evaluate(|obs| qrdqn.act_greedy_with(&qrdqn_infer, obs));

    let ppg_infer = ppg.inference_net();
    let (ppg_ret, ppg_solve) =
        evaluate(|obs| action_from_row(&ppg.act_greedy_env_row_with(&ppg_infer, obs)));

    println!();
    println!(
        "CartPole policy quality | time_limit={TIME_LIMIT} episodes={EVAL_EPISODES} train_steps={TRAIN_TIMESTEPS}"
    );
    println!("  policy        mean_return   solve_rate");
    println!("  random        {rand_ret:>11.2}   {rand_solve:>10.2}");
    println!("  dqn           {dqn_ret:>11.2}   {dqn_solve:>10.2}");
    println!("  c51           {c51_ret:>11.2}   {c51_solve:>10.2}");
    println!("  qrdqn         {qrdqn_ret:>11.2}   {qrdqn_solve:>10.2}");
    println!("  ppg           {ppg_ret:>11.2}   {ppg_solve:>10.2}");
    println!();
}

// ---------------------------------------------------------------------------
// Criterion throughput
// ---------------------------------------------------------------------------

fn bench_policies(
    c: &mut Criterion,
    dqn: &DqnCartPoleAgent,
    c51: &C51CartPoleAgent,
    qrdqn: &QrDqnCartPoleAgent,
    ppg: &PpgCartPoleAgent,
) {
    // Snapshot each value-based net onto the inner (non-autodiff) backend once;
    // per-step inference then skips autodiff-graph construction entirely, which
    // is the dominant cost for the distributional agents at batch size 1.
    let dqn_infer = dqn.inference_net();
    let c51_infer = c51.inference_net();
    let c51_support = c51.inference_support();
    let qrdqn_infer = qrdqn.inference_net();
    let ppg_infer = ppg.inference_net();

    let mut group = c.benchmark_group("cartpole_policy_rollout");
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
            rollout_steps(black_box(steps), |_| {
                CartPoleAction::from_index(rng.random_range(0..ACTIONS))
            });
        });
    });

    // Learned policies time their greedy (evaluation) inference path on the
    // inner backend, matching how the quality comparison rolls them out.
    group.bench_with_input(BenchmarkId::new("dqn", steps), &steps, |b, &steps| {
        b.iter(|| {
            rollout_steps(black_box(steps), |obs| dqn.act_greedy_with(&dqn_infer, obs));
        });
    });

    group.bench_with_input(BenchmarkId::new("c51", steps), &steps, |b, &steps| {
        b.iter(|| {
            rollout_steps(black_box(steps), |obs| {
                c51.act_greedy_with(&c51_infer, &c51_support, obs)
            });
        });
    });

    group.bench_with_input(BenchmarkId::new("qrdqn", steps), &steps, |b, &steps| {
        b.iter(|| {
            rollout_steps(black_box(steps), |obs| {
                qrdqn.act_greedy_with(&qrdqn_infer, obs)
            });
        });
    });

    group.bench_with_input(BenchmarkId::new("ppg", steps), &steps, |b, &steps| {
        b.iter(|| {
            rollout_steps(black_box(steps), |obs| {
                action_from_row(&ppg.act_greedy_env_row_with(&ppg_infer, obs))
            });
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

    let dqn = train_dqn_agent();
    let c51 = train_c51_agent();
    let qrdqn = train_qrdqn_agent();
    let ppg = train_ppg_agent();

    print_quality_comparison(&dqn, &c51, &qrdqn, &ppg);

    let mut criterion = Criterion::default().configure_from_args();
    bench_policies(&mut criterion, &dqn, &c51, &qrdqn, &ppg);
    criterion.final_summary();
}
