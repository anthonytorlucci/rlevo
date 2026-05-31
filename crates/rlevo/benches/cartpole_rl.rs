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

#[path = "support/dqn.rs"]
mod dqn_support;

use std::collections::HashMap;
use std::hint::black_box;

use burn::backend::{Autodiff, Flex};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use burn::tensor::{TensorData, activation};

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
use rlevo_reinforcement_learning::algorithms::c51::c51_model::C51Model;
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
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_agent::QrDqnAgent;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_model::QrDqnModel;
use rlevo_reinforcement_learning::algorithms::qrdqn::train::train as train_qrdqn;

use dqn_support::VecMlpDqn;

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
// C51 model — (batch, 4) → (batch, actions, atoms)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct C51Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    num_atoms: usize,
}

impl<B: Backend> C51Mlp<B> {
    fn new(num_atoms: usize, device: &<B as BackendTypes>::Device) -> Self {
        Self {
            l1: LinearConfig::new(OBS_FEATURES, HIDDEN).init(device),
            l2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            l3: LinearConfig::new(HIDDEN, ACTIONS * num_atoms).init(device),
            num_atoms,
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _] = obs.dims();
        let x = activation::relu(self.l1.forward(obs));
        let x = activation::relu(self.l2.forward(x));
        self.l3.forward(x).reshape([batch, ACTIONS, self.num_atoms])
    }
}

impl<B: AutodiffBackend> C51Model<B, 2> for C51Mlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 3> {
        self.forward_impl(obs)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 3> {
        inner.forward_impl(obs)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        c51_polyak::<B::InnerBackend, C51Mlp<B::InnerBackend>>(&active.valid(), target, tau as f32)
    }
}

// ---------------------------------------------------------------------------
// QR-DQN model — (batch, 4) → (batch, actions, quantiles)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct QrDqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    num_quantiles: usize,
}

impl<B: Backend> QrDqnMlp<B> {
    fn new(num_quantiles: usize, device: &<B as BackendTypes>::Device) -> Self {
        Self {
            l1: LinearConfig::new(OBS_FEATURES, HIDDEN).init(device),
            l2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            l3: LinearConfig::new(HIDDEN, ACTIONS * num_quantiles).init(device),
            num_quantiles,
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _] = obs.dims();
        let x = activation::relu(self.l1.forward(obs));
        let x = activation::relu(self.l2.forward(x));
        self.l3
            .forward(x)
            .reshape([batch, ACTIONS, self.num_quantiles])
    }
}

impl<B: AutodiffBackend> QrDqnModel<B, 2> for QrDqnMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 3> {
        self.forward_impl(obs)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 3> {
        inner.forward_impl(obs)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        qrdqn_polyak::<B::InnerBackend, QrDqnMlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// PPG value network — two-layer tanh MLP → scalar
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
            fc1: LinearConfig::new(OBS_FEATURES, HIDDEN).init(device),
            fc2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            head: LinearConfig::new(HIDDEN, 1).init(device),
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
// Polyak averaging — one copy per model type (each needs its own M bound)
// ---------------------------------------------------------------------------

fn polyak_impl<B: Backend, M: Module<B>>(active: &M, target: M, tau: f32) -> M {
    struct Collector<B: Backend> {
        tensors: HashMap<ParamId, TensorData>,
        _m: std::marker::PhantomData<B>,
    }
    impl<B: Backend> ModuleVisitor<B> for Collector<B> {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
            self.tensors.insert(p.id, p.val().to_data());
        }
    }
    struct Mapper<B: Backend> {
        active: HashMap<ParamId, TensorData>,
        tau: f32,
        _m: std::marker::PhantomData<B>,
    }
    impl<B: Backend> ModuleMapper<B> for Mapper<B> {
        fn map_float<const D: usize>(
            &mut self,
            param: Param<Tensor<B, D>>,
        ) -> Param<Tensor<B, D>> {
            let id = param.id;
            let active = self.active.remove(&id).expect("param missing");
            let tau = self.tau;
            param.map(move |t| {
                let dev = t.device();
                t.mul_scalar(1.0 - tau) + Tensor::<B, D>::from_data(active, &dev).mul_scalar(tau)
            })
        }
    }

    let mut c = Collector::<B> {
        tensors: HashMap::new(),
        _m: std::marker::PhantomData,
    };
    active.visit(&mut c);
    let mut m = Mapper::<B> {
        active: c.tensors,
        tau,
        _m: std::marker::PhantomData,
    };
    target.map(&mut m)
}

fn c51_polyak<B: Backend, M: Module<B>>(active: &M, target: M, tau: f32) -> M {
    polyak_impl(active, target, tau)
}

fn qrdqn_polyak<B: Backend, M: Module<B>>(active: &M, target: M, tau: f32) -> M {
    polyak_impl(active, target, tau)
}

// ---------------------------------------------------------------------------
// Environment factory
// ---------------------------------------------------------------------------

fn make_env() -> Env {
    TimeLimit::new(
        CartPole::with_config(CartPoleConfig {
            seed: SEED,
            ..CartPoleConfig::default()
        }),
        TIME_LIMIT,
    )
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

fn train_dqn_agent() -> DqnCartPoleAgent {
    let device = Default::default();
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
        .build();
    let model: VecMlpDqn<Backend_> = VecMlpDqn::new(OBS_FEATURES, HIDDEN, ACTIONS, &device);
    let mut agent: DqnCartPoleAgent = DqnAgent::new(model, config, device);
    train_dqn(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("dqn training");
    agent
}

fn train_c51_agent() -> C51CartPoleAgent {
    let device = Default::default();
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
        .build();
    let model: C51Mlp<Backend_> = C51Mlp::new(NUM_ATOMS, &device);
    let mut agent: C51CartPoleAgent = C51Agent::new(model, config, device);
    train_c51(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("c51 training");
    agent
}

fn train_qrdqn_agent() -> QrDqnCartPoleAgent {
    let device = Default::default();
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
        .build();
    let model: QrDqnMlp<Backend_> = QrDqnMlp::new(NUM_QUANTILES, &device);
    let mut agent: QrDqnCartPoleAgent = QrDqnAgent::new(model, config, device);
    train_qrdqn(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("qrdqn training");
    agent
}

fn train_ppg_agent() -> PpgCartPoleAgent {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = make_env();
    let policy: PpgCategoricalPolicyHead<Backend_> = PpgCategoricalPolicyHeadConfig {
        obs_dim: OBS_FEATURES,
        hidden: HIDDEN,
        num_actions: ACTIONS,
    }
    .init::<Backend_>(&device);
    let value: ValueMlp<Backend_> = ValueMlp::new(&device);
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
        })
        .n_iteration(32)
        .e_aux(6)
        .beta_clone(1.0)
        .build();
    let total_iterations = TRAIN_TIMESTEPS / config.batch_size().max(1);
    let mut agent: PpgCartPoleAgent =
        PpgAgent::new(policy, value, config, device, total_iterations);
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

fn rollout_steps(steps: usize, mut next_action: impl FnMut(&CartPoleObservation) -> CartPoleAction) {
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

    let mut r = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let (dqn_ret, dqn_solve) = evaluate(|obs| dqn.act(obs, &mut r));

    let mut r = StdRng::seed_from_u64(SEED.wrapping_add(2));
    let (c51_ret, c51_solve) = evaluate(|obs| c51.act(obs, &mut r));

    let mut r = StdRng::seed_from_u64(SEED.wrapping_add(3));
    let (qrdqn_ret, qrdqn_solve) = evaluate(|obs| qrdqn.act(obs, &mut r));

    let mut r = StdRng::seed_from_u64(SEED.wrapping_add(4));
    let (ppg_ret, ppg_solve) = evaluate(|obs| action_from_row(&ppg.act(obs, &mut r).env_row));

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
    let mut group = c.benchmark_group("cartpole_policy_rollout");
    for &steps in &[1_000_usize, 4_000, 16_000] {
        group.throughput(Throughput::Elements(steps as u64));

        group.bench_with_input(BenchmarkId::new("random", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |_| {
                    CartPoleAction::from_index(rng.random_range(0..ACTIONS))
                });
            });
        });

        group.bench_with_input(BenchmarkId::new("dqn", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |obs| dqn.act(obs, &mut rng));
            });
        });

        group.bench_with_input(BenchmarkId::new("c51", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |obs| c51.act(obs, &mut rng));
            });
        });

        group.bench_with_input(BenchmarkId::new("qrdqn", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |obs| qrdqn.act(obs, &mut rng));
            });
        });

        group.bench_with_input(BenchmarkId::new("ppg", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |obs| {
                    action_from_row(&ppg.act(obs, &mut rng).env_row)
                });
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let dqn = train_dqn_agent();
    let c51 = train_c51_agent();
    let qrdqn = train_qrdqn_agent();
    let ppg = train_ppg_agent();

    print_quality_comparison(&dqn, &c51, &qrdqn, &ppg);

    let mut criterion = Criterion::default().configure_from_args();
    bench_policies(&mut criterion, &dqn, &c51, &qrdqn, &ppg);
    criterion.final_summary();
}
