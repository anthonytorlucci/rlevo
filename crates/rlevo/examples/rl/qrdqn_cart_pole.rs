//! QR-DQN (Quantile Regression DQN) on CartPole with Burn's ndarray backend.
//!
//! Usage:
//!
//! ```text
//! cargo run -p rlevo-reinforcement-learning --release --example qrdqn_cart_pole -- \
//!     --seed 42 --total-timesteps 50000 --log-every 1000 --num-quantiles 200
//! ```

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData, activation};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_agent::QrDqnAgent;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_model::QrDqnModel;
use rlevo_reinforcement_learning::algorithms::qrdqn::train::train;

// ---------------------------------------------------------------------------
// Model — two hidden layers; output reshaped to (batch, actions, quantiles).
// Raw quantile values: no activation at the head.
// ---------------------------------------------------------------------------

const N_ACTIONS: usize = 2;

#[derive(Module, Debug)]
pub struct QrDqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    num_quantiles: usize,
}

impl<B: Backend> QrDqnMlp<B> {
    fn new(num_quantiles: usize, device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, N_ACTIONS * num_quantiles).init(device),
            num_quantiles,
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _] = observations.dims();
        let x = activation::relu(self.l1.forward(observations));
        let x = activation::relu(self.l2.forward(x));
        let flat = self.l3.forward(x); // (B, A*N)
        flat.reshape([batch_size, N_ACTIONS, self.num_quantiles])
    }
}

impl<B: AutodiffBackend> QrDqnModel<B, 2> for QrDqnMlp<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        self.forward_impl(observations)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 3> {
        inner.forward_impl(observations)
    }

    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, QrDqnMlp<B::InnerBackend>>(
            active.valid(),
            target,
            tau as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// Polyak averaging via Burn's ModuleVisitor / ModuleMapper
// ---------------------------------------------------------------------------

struct ParamCollector<B: Backend> {
    tensors: HashMap<ParamId, TensorData>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.tensors.insert(param.id, param.val().to_data());
    }
}

struct PolyakMapper<B: Backend> {
    active: HashMap<ParamId, TensorData>,
    tau: f32,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> ModuleMapper<B> for PolyakMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let id = param.id;
        let active = self
            .active
            .remove(&id)
            .expect("param not collected from active network");
        let tau = self.tau;
        param.map(move |target_tensor| {
            let device = target_tensor.device();
            let active_tensor = Tensor::<B, D>::from_data(active, &device);
            target_tensor.mul_scalar(1.0 - tau) + active_tensor.mul_scalar(tau)
        })
    }
}

fn polyak_update<B: Backend, M: Module<B>>(active: M, target: M, tau: f32) -> M {
    let mut collector = ParamCollector::<B> {
        tensors: HashMap::new(),
        _marker: std::marker::PhantomData,
    };
    active.visit(&mut collector);
    let mut mapper = PolyakMapper::<B> {
        active: collector.tensors,
        tau,
        _marker: std::marker::PhantomData,
    };
    target.map(&mut mapper)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

type Backend_ = Autodiff<NdArray>;

struct CliArgs {
    seed: u64,
    total_timesteps: usize,
    log_every: usize,
    num_quantiles: usize,
}

fn parse_args() -> CliArgs {
    let mut seed = 42_u64;
    let mut total_timesteps = 50_000_usize;
    let mut log_every = 1_000_usize;
    let mut num_quantiles = 200_usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--seed" => {
                seed = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--seed expects a u64");
            }
            "--total-timesteps" => {
                total_timesteps = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--total-timesteps expects a usize");
            }
            "--log-every" => {
                log_every = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--log-every expects a usize");
            }
            "--num-quantiles" => {
                num_quantiles = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--num-quantiles expects a usize");
            }
            other => panic!("unknown flag: {other}"),
        }
    }
    CliArgs {
        seed,
        total_timesteps,
        log_every,
        num_quantiles,
    }
}

fn main() {
    tracing_subscriber::fmt().with_target(false).init();

    let args = parse_args();
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(args.seed);

    let mut env = CartPole::with_config(CartPoleConfig {
        seed: args.seed,
        ..CartPoleConfig::default()
    });

    // QR-DQN needs no v_min/v_max — the quantile support adapts to the
    // reward distribution observed in the replay buffer.
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
        .num_quantiles(args.num_quantiles)
        .kappa(1.0)
        .build();

    let model: QrDqnMlp<Backend_> = QrDqnMlp::new(args.num_quantiles, &device);
    let mut agent: QrDqnAgent<
        Backend_,
        QrDqnMlp<Backend_>,
        CartPoleObservation,
        CartPoleAction,
        1,
        2,
    > = QrDqnAgent::new(model, config, device);

    train(
        &mut agent,
        &mut env,
        &mut rng,
        args.total_timesteps,
        args.log_every,
    )
    .expect("training loop");

    let avg = agent.stats().avg_score().unwrap_or(0.0);
    println!(
        "final avg reward over last {} episodes: {avg:.2}",
        agent.stats().recent_history.len()
    );
}
