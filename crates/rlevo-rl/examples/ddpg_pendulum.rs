//! DDPG on Pendulum-v1 with Burn's ndarray backend.
//!
//! Usage:
//!
//! ```text
//! cargo run -p evorl-rl --release --example ddpg_pendulum -- \
//!     --seed 42 --total-timesteps 100000 --log-every 5000
//! ```

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_envs::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_envs::wrappers::TimeLimit;
use rlevo_rl::algorithms::ddpg::ddpg_agent::DdpgAgent;
use rlevo_rl::algorithms::ddpg::ddpg_config::DdpgTrainingConfigBuilder;
use rlevo_rl::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};
use rlevo_rl::algorithms::ddpg::train::train;

// ---------------------------------------------------------------------------
// Actor: (batch, 3) -> (batch, 1) in [-2, 2]
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ActorMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
    action_scale: f32,
    action_bias: f32,
}

impl<B: Backend> ActorMlp<B> {
    fn new(obs_dim: usize, hidden: usize, action_dim: usize, device: &B::Device) -> Self {
        // Pendulum action range is [-2, 2], so scale=2, bias=0.
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            head: LinearConfig::new(hidden, action_dim).init(device),
            action_scale: 2.0,
            action_bias: 0.0,
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.fc1.forward(obs));
        let h = relu(self.fc2.forward(h));
        let raw = tanh(self.head.forward(h));
        raw.mul_scalar(self.action_scale)
            .add_scalar(self.action_bias)
    }
}

impl<B: AutodiffBackend> DeterministicPolicy<B, 2, 2> for ActorMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_impl(obs)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(obs)
    }

    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, ActorMlp<B::InnerBackend>>(
            active.valid(),
            target,
            tau as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// Critic: concat(obs, action) -> (batch,) Q-value
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct CriticMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> CriticMlp<B> {
    fn new(obs_dim: usize, action_dim: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim + action_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            head: LinearConfig::new(hidden, 1).init(device),
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = Tensor::cat(vec![obs, act], 1);
        let h = relu(self.fc1.forward(x));
        let h = relu(self.fc2.forward(h));
        self.head.forward(h).squeeze_dim::<1>(1)
    }
}

impl<B: AutodiffBackend> ContinuousQ<B, 2, 2> for CriticMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs, act)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        act: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 1> {
        inner.forward_impl(obs, act)
    }

    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, CriticMlp<B::InnerBackend>>(
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
// CLI + main
// ---------------------------------------------------------------------------

type Be = Autodiff<NdArray>;

struct CliArgs {
    seed: u64,
    total_timesteps: usize,
    log_every: usize,
}

fn parse_args() -> CliArgs {
    let mut seed = 42_u64;
    let mut total_timesteps = 100_000_usize;
    let mut log_every = 5_000_usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).expect("u64"),
            "--total-timesteps" => {
                total_timesteps = args.next().and_then(|v| v.parse().ok()).expect("usize");
            }
            "--log-every" => {
                log_every = args.next().and_then(|v| v.parse().ok()).expect("usize");
            }
            other => panic!("unknown flag: {other}"),
        }
    }
    CliArgs {
        seed,
        total_timesteps,
        log_every,
    }
}

fn main() {
    tracing_subscriber::fmt().with_target(false).init();
    let args = parse_args();
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(args.seed);

    let base_env = Pendulum::with_config(PendulumConfig {
        seed: args.seed,
        ..PendulumConfig::default()
    });
    let mut env = TimeLimit::new(base_env, 200);

    let actor: ActorMlp<Be> = ActorMlp::new(3, 256, 1, &device);
    let critic: CriticMlp<Be> = CriticMlp::new(3, 1, 256, &device);

    // Keep learning_starts modest for the example so users see progress
    // well before the 25k-step CleanRL default.
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

    let mut agent: DdpgAgent<
        Be,
        ActorMlp<Be>,
        CriticMlp<Be>,
        PendulumObservation,
        PendulumAction,
        1,
        2,
        1,
        2,
    > = DdpgAgent::new(actor, critic, config, device);

    train::<Be, _, _, _, _, PendulumAction, _, 1, 1, 2, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        args.total_timesteps,
        args.log_every,
    )
    .expect("training");

    let avg = agent.stats().avg_score().unwrap_or(0.0);
    println!(
        "ddpg_pendulum: final avg reward over last {} episodes: {avg:.2}",
        agent.stats().recent_history.len()
    );
}
