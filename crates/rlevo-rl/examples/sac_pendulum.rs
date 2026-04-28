//! SAC on Pendulum-v1 with Burn's ndarray backend.
//!
//! CLI mirrors `td3_pendulum` so the two agents can be A/B'd by swapping the
//! example name:
//!
//! ```text
//! cargo run -p evorl-rl --release --example sac_pendulum -- \
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

use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_rl::algorithms::sac::sac_agent::SacAgent;
use rlevo_rl::algorithms::sac::sac_config::SacTrainingConfigBuilder;
use rlevo_rl::algorithms::sac::sac_model::{ContinuousQ, SampleOutput, SquashedGaussianPolicy};
use rlevo_rl::algorithms::sac::train::train;

// ---------------------------------------------------------------------------
// Actor: (batch, 3) -> (batch, 1) ∈ [-2, 2], squashed-Gaussian reparameterized
// sample. Shares the μ-head MLP with td3_pendulum but adds a state-conditional
// log-σ head so SAC's entropy can shrink/grow per-state.
// ---------------------------------------------------------------------------

const LOG_STD_MIN: f32 = -5.0;
const LOG_STD_MAX: f32 = 2.0;

#[derive(Module, Debug)]
pub struct StochasticActor<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    mean: Linear<B>,
    log_std: Linear<B>,
    action_dim: usize,
    action_scale: f32,
    action_bias: f32,
}

impl<B: Backend> StochasticActor<B> {
    fn new(obs_dim: usize, hidden: usize, action_dim: usize, device: &B::Device) -> Self {
        // Pendulum action range is [-2, 2], so scale=2, bias=0.
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            mean: LinearConfig::new(hidden, action_dim).init(device),
            log_std: LinearConfig::new(hidden, action_dim).init(device),
            action_dim,
            action_scale: 2.0,
            action_bias: 0.0,
        }
    }

    fn features(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.fc1.forward(obs));
        relu(self.fc2.forward(h))
    }

    fn mean_and_log_std(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.features(obs);
        let mean = self.mean.forward(h.clone());
        let log_std = self.log_std.forward(h).clamp(LOG_STD_MIN, LOG_STD_MAX);
        (mean, log_std)
    }

    fn squashed_sample(
        &self,
        obs: Tensor<B, 2>,
        eps: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let (mean, log_std) = self.mean_and_log_std(obs);
        let action_dim = mean.dims()[1];
        let std = log_std.clone().exp();
        let z = mean.clone() + std * eps;

        let diff = z.clone() - mean;
        let scaled = diff / log_std.clone().exp();
        let scaled_sq = scaled.clone() * scaled;
        let log_2pi = (2.0_f32 * std::f32::consts::PI).ln();
        let per_dim_gauss: Tensor<B, 2> = scaled_sq.mul_scalar(-0.5) - log_std - log_2pi * 0.5;

        let ln_2 = std::f32::consts::LN_2;
        let neg_two_z = z.clone().mul_scalar(-2.0);
        let sp = burn::tensor::activation::softplus(neg_two_z, 1.0);
        let per_dim_jac: Tensor<B, 2> = (z.clone().neg() - sp + ln_2).mul_scalar(2.0);

        let per_dim = per_dim_gauss - per_dim_jac;
        let log_prob_z = per_dim.sum_dim(1).squeeze_dim::<1>(1);
        let log_scale_abs = self.action_scale.abs().ln();
        let log_prob = log_prob_z.sub_scalar(log_scale_abs * action_dim as f32);

        let action = tanh(z)
            .mul_scalar(self.action_scale)
            .add_scalar(self.action_bias);
        (action, log_prob)
    }
}

impl<B: AutodiffBackend> SquashedGaussianPolicy<B, 2, 2> for StochasticActor<B> {
    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn forward_sample(&self, obs: Tensor<B, 2>, eps: Tensor<B, 2>) -> SampleOutput<B, 2> {
        let (action, log_prob) = self.squashed_sample(obs, eps);
        SampleOutput { action, log_prob }
    }

    fn forward_sample_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        eps: Tensor<B::InnerBackend, 2>,
    ) -> SampleOutput<B::InnerBackend, 2> {
        let (action, log_prob) = inner.squashed_sample(obs, eps);
        SampleOutput { action, log_prob }
    }

    fn deterministic_action(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mean, _) = self.mean_and_log_std(obs);
        tanh(mean)
            .mul_scalar(self.action_scale)
            .add_scalar(self.action_bias)
    }
}

// ---------------------------------------------------------------------------
// Critic: concat(obs, action) -> (batch,) Q-value. Identical to td3_pendulum.
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

    let actor: StochasticActor<Be> = StochasticActor::new(3, 256, 1, &device);
    let critic_1: CriticMlp<Be> = CriticMlp::new(3, 1, 256, &device);
    let critic_2: CriticMlp<Be> = CriticMlp::new(3, 1, 256, &device);

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

    let mut agent: SacAgent<
        Be,
        StochasticActor<Be>,
        CriticMlp<Be>,
        PendulumObservation,
        PendulumAction,
        1,
        2,
        1,
        2,
    > = SacAgent::new(actor, critic_1, critic_2, config, device);

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
        "sac_pendulum: final avg reward over last {} episodes: {avg:.2} (alpha={:.4})",
        agent.stats().recent_history.len(),
        agent.last_alpha()
    );
}
