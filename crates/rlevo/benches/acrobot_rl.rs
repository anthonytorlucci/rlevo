//! Random baseline vs. DQN / C51 / QR-DQN / PPG on [`Acrobot`] — quality
//! summary + throughput.
//!
//! A uniformly random torque policy almost never swings the Acrobot up to the
//! goal height; it is the floor every learned policy must clear. This bench
//! trains four agents sequentially and:
//!
//! 1. **Quality comparison** — prints mean episode return and goal-reach
//!    (termination) rate for the random policy vs. each trained policy over a
//!    batch of episodes.
//! 2. **Throughput** — a Criterion group timing per-step rollout cost of the
//!    random policy vs. each trained policy's greedy inference.
//!
//! `Acrobot` rewards `-1` per non-terminal step and `0` on reaching the goal;
//! episodes are capped at 500 steps via [`TimeLimit`] (Acrobot only terminates
//! on the goal, so an unbounded random rollout could run forever). "Solving"
//! means reaching the goal (termination) rather than the time-limit
//! truncation. Each algorithm trains for 60 000 steps; they run serially.
//!
//! # Run with
//!
//! ```bash
//! # Quality summary + Criterion timing
//! cargo bench -p rlevo --bench acrobot_rl
//!
//! # Quality summary only (skip Criterion timing)
//! cargo bench -p rlevo --bench acrobot_rl -- --test
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

use rlevo_environments::classic::{
    Acrobot, AcrobotAction, AcrobotConfig, AcrobotObservation, BookDynamics,
};
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
use rlevo_reinforcement_learning::target::TargetUpdate;

use value_nets::{C51Mlp, QrDqnMlp, ValueMlp, VecMlpDqn};

const SEED: u64 = 2026;
/// Observation width: `[cos θ1, sin θ1, cos θ2, sin θ2, θ̇1, θ̇2]`.
const OBS_FEATURES: usize = 6;
const ACTIONS: usize = AcrobotAction::ACTION_COUNT;
const HIDDEN: usize = 128;
/// Per-episode step cap (Gymnasium uses 500).
const TIME_LIMIT: usize = 500;
const TRAIN_TIMESTEPS: usize = 60_000;
const EVAL_EPISODES: usize = 100;
const NUM_ATOMS: usize = 51;
const NUM_QUANTILES: usize = 200;
/// Return support for the distributional (C51) critic. Acrobot pays `-1` per
/// step over a 500-step cap, so returns live in roughly `[-500, 0]`.
const V_MIN: f32 = -500.0;
const V_MAX: f32 = 0.0;

type Backend_ = Autodiff<Flex>;
type Env = TimeLimit<Acrobot<BookDynamics>>;

type DqnAcrobotAgent =
    DqnAgent<Backend_, VecMlpDqn<Backend_>, AcrobotObservation, AcrobotAction, 1, 2>;
type C51AcrobotAgent =
    C51Agent<Backend_, C51Mlp<Backend_>, AcrobotObservation, AcrobotAction, 1, 2>;
type QrDqnAcrobotAgent =
    QrDqnAgent<Backend_, QrDqnMlp<Backend_>, AcrobotObservation, AcrobotAction, 1, 2>;
type PpgAcrobotAgent = PpgAgent<
    Backend_,
    PpgCategoricalPolicyHead<Backend_>,
    ValueMlp<Backend_>,
    AcrobotObservation,
    1,
    2,
>;

// ---------------------------------------------------------------------------
// Environment factory
// ---------------------------------------------------------------------------

fn make_env() -> Env {
    TimeLimit::new(
        Acrobot::<BookDynamics>::with_config(AcrobotConfig {
            seed: SEED,
            ..AcrobotConfig::default()
        })
        .expect("valid config"),
        TIME_LIMIT,
    )
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

fn train_dqn_agent() -> DqnAcrobotAgent {
    let device = Default::default();
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let config = DqnTrainingConfigBuilder::new()
        .batch_size(128)
        .gamma(0.99)
        .target_update(TargetUpdate::polyak(0.005, 1))
        .learning_rate(1e-3)
        .epsilon_start(1.0)
        .epsilon_end(0.02)
        .epsilon_decay(0.9997)
        .learning_starts(1_000)
        .train_frequency(4)
        .replay_buffer_capacity(50_000)
        .double_q(true)
        .build()
        .expect("valid config");
    let model: VecMlpDqn<Backend_> = VecMlpDqn::new(OBS_FEATURES, HIDDEN, ACTIONS, &device);
    let mut agent: DqnAcrobotAgent = DqnAgent::new(model, config, device).expect("valid config");
    train_dqn(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("dqn training");
    agent
}

fn train_c51_agent() -> C51AcrobotAgent {
    let device = Default::default();
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let config = C51TrainingConfigBuilder::new()
        .batch_size(128)
        .gamma(0.99)
        .target_update(TargetUpdate::polyak(0.005, 1))
        .learning_rate(1e-3)
        .epsilon_start(1.0)
        .epsilon_end(0.02)
        .epsilon_decay(0.9997)
        .learning_starts(1_000)
        .train_frequency(4)
        .replay_buffer_capacity(50_000)
        .num_atoms(NUM_ATOMS)
        .v_min(V_MIN)
        .v_max(V_MAX)
        .build()
        .expect("valid config");
    let model: C51Mlp<Backend_> = C51Mlp::new(OBS_FEATURES, HIDDEN, ACTIONS, NUM_ATOMS, &device);
    let mut agent: C51AcrobotAgent = C51Agent::new(model, config, device).expect("valid config");
    train_c51(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("c51 training");
    agent
}

fn train_qrdqn_agent() -> QrDqnAcrobotAgent {
    let device = Default::default();
    <Backend_ as Backend>::seed(&device, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let config = QrDqnTrainingConfigBuilder::new()
        .batch_size(128)
        .gamma(0.99)
        .target_update(TargetUpdate::polyak(0.005, 1))
        .learning_rate(1e-3)
        .epsilon_start(1.0)
        .epsilon_end(0.02)
        .epsilon_decay(0.9997)
        .learning_starts(1_000)
        .train_frequency(4)
        .replay_buffer_capacity(50_000)
        .num_quantiles(NUM_QUANTILES)
        .kappa(1.0)
        .build()
        .expect("valid config");
    let model: QrDqnMlp<Backend_> =
        QrDqnMlp::new(OBS_FEATURES, HIDDEN, ACTIONS, NUM_QUANTILES, &device);
    let mut agent: QrDqnAcrobotAgent =
        QrDqnAgent::new(model, config, device).expect("valid config");
    train_qrdqn(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("qrdqn training");
    agent
}

fn train_ppg_agent() -> PpgAcrobotAgent {
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
    let mut agent: PpgAcrobotAgent =
        PpgAgent::new(policy, value, config, device, total_iterations).expect("valid config");
    train_ppg::<Backend_, _, _, _, _, AcrobotAction, _, 1, 1, 2>(
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

/// Runs one capped episode under `next_action`. Returns the episode return
/// (sum of rewards) and whether it ended on the goal (termination, not the
/// time-limit truncation).
fn roll_out(
    env: &mut Env,
    mut next_action: impl FnMut(&AcrobotObservation) -> AcrobotAction,
) -> (f32, bool) {
    let mut snap = env.reset().expect("reset");
    let mut ret = 0.0_f32;
    loop {
        let action = next_action(snap.observation());
        snap = env.step(action).expect("step");
        ret += f32::from(*snap.reward());
        if snap.is_done() {
            return (ret, snap.is_terminated());
        }
    }
}

/// Estimates `(mean_return, goal_rate)` over `EVAL_EPISODES` episodes.
#[allow(clippy::cast_precision_loss)]
fn evaluate(mut next_action: impl FnMut(&AcrobotObservation) -> AcrobotAction) -> (f32, f32) {
    let mut env = make_env();
    let (mut total, mut goals) = (0.0_f32, 0_usize);
    for _ in 0..EVAL_EPISODES {
        let (r, reached) = roll_out(&mut env, &mut next_action);
        total += r;
        goals += usize::from(reached);
    }
    let n = EVAL_EPISODES as f32;
    (total / n, goals as f32 / n)
}

/// Drives `steps` of a `next_action`-chosen rollout, resetting on episode end.
fn rollout_steps(steps: usize, mut next_action: impl FnMut(&AcrobotObservation) -> AcrobotAction) {
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

/// Converts a PPG greedy-policy output row to an [`AcrobotAction`].
///
/// `act_greedy_env_row_with` returns a `Vec<f32>` whose first element is the
/// chosen action index (already argmaxed by the policy head). This helper casts
/// it to `usize` and delegates to [`AcrobotAction::from_index`].
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn action_from_row(row: &[f32]) -> AcrobotAction {
    AcrobotAction::from_index(row[0] as usize)
}

/// Prints a formatted quality table comparing the random baseline against each
/// trained policy over [`EVAL_EPISODES`] episodes.
///
/// Each policy is evaluated on its greedy (non-exploratory) inference path via
/// the respective `act_greedy_with` method. The table reports mean episode
/// return and goal-reach rate (fraction of episodes that terminated on the goal
/// rather than hitting the [`TIME_LIMIT`] cap).
fn print_quality_comparison(
    dqn: &DqnAcrobotAgent,
    c51: &C51AcrobotAgent,
    qrdqn: &QrDqnAcrobotAgent,
    ppg: &PpgAcrobotAgent,
) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let (rand_ret, rand_goal) =
        evaluate(|_| AcrobotAction::from_index(rng.random_range(0..ACTIONS)));

    // Evaluation uses each agent's greedy policy: `act` is ε-greedy and floors
    // at `epsilon_end`, so it injects exploration noise that hurts the goal
    // rate even for a well-trained policy. The value-based agents run inference
    // on a once-snapshotted inner (non-autodiff) network — far cheaper than
    // rebuilding an autodiff graph every step.
    let dqn_infer = dqn.inference_net();
    let (dqn_ret, dqn_goal) = evaluate(|obs| dqn.act_greedy_with(&dqn_infer, obs));

    let c51_infer = c51.inference_net();
    let c51_support = c51.inference_support();
    let (c51_ret, c51_goal) = evaluate(|obs| c51.act_greedy_with(&c51_infer, &c51_support, obs));

    let qrdqn_infer = qrdqn.inference_net();
    let (qrdqn_ret, qrdqn_goal) = evaluate(|obs| qrdqn.act_greedy_with(&qrdqn_infer, obs));

    let ppg_infer = ppg.inference_net();
    let (ppg_ret, ppg_goal) =
        evaluate(|obs| action_from_row(&ppg.act_greedy_env_row_with(&ppg_infer, obs)));

    println!();
    println!(
        "Acrobot policy quality | time_limit={TIME_LIMIT} episodes={EVAL_EPISODES} train_steps={TRAIN_TIMESTEPS}"
    );
    println!("  policy        mean_return   goal_rate");
    println!("  random        {rand_ret:>11.2}   {rand_goal:>9.2}");
    println!("  dqn           {dqn_ret:>11.2}   {dqn_goal:>9.2}");
    println!("  c51           {c51_ret:>11.2}   {c51_goal:>9.2}");
    println!("  qrdqn         {qrdqn_ret:>11.2}   {qrdqn_goal:>9.2}");
    println!("  ppg           {ppg_ret:>11.2}   {ppg_goal:>9.2}");
    println!();
}

// ---------------------------------------------------------------------------
// Criterion throughput
// ---------------------------------------------------------------------------

fn bench_policies(
    c: &mut Criterion,
    dqn: &DqnAcrobotAgent,
    c51: &C51AcrobotAgent,
    qrdqn: &QrDqnAcrobotAgent,
    ppg: &PpgAcrobotAgent,
) {
    // Snapshot each value-based net onto the inner (non-autodiff) backend once;
    // per-step inference then skips autodiff-graph construction entirely, which
    // is the dominant cost for the distributional agents at batch size 1.
    let dqn_infer = dqn.inference_net();
    let c51_infer = c51.inference_net();
    let c51_support = c51.inference_support();
    let qrdqn_infer = qrdqn.inference_net();
    let ppg_infer = ppg.inference_net();

    let mut group = c.benchmark_group("acrobot_policy_rollout");
    // Per-step throughput is independent of rollout length (the rollout is
    // linear and `Throughput::Elements` normalises per step), so a single size
    // captures the same signal a 1k/4k/16k sweep did. Ten samples keep a
    // usable confidence interval without multi-second-per-iteration runs.
    group.sample_size(10);
    let steps = 4_000_usize;
    group.throughput(Throughput::Elements(steps as u64));

    group.bench_with_input(BenchmarkId::new("random", steps), &steps, |b, &steps| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(SEED);
            rollout_steps(black_box(steps), |_| {
                AcrobotAction::from_index(rng.random_range(0..ACTIONS))
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
