//! Micro benchmarks for the SAC agent.
//!
//! Measures actor-sampling latency and full `learn_step` wall-clock (twin
//! critic update + actor + α step + Polyak) on a primed replay buffer.
//! Mirrors `td3_bench`'s pattern so the two algorithms can be compared
//! side-by-side.

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softplus, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

use criterion::{Criterion, criterion_group, criterion_main};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_rl::algorithms::sac::sac_agent::SacAgent;
use rlevo_rl::algorithms::sac::sac_config::SacTrainingConfigBuilder;
use rlevo_rl::algorithms::sac::sac_model::{ContinuousQ, SampleOutput, SquashedGaussianPolicy};

#[derive(Module, Debug)]
struct StochasticActor<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    mean: Linear<B>,
    log_std: Linear<B>,
    action_dim: usize,
    action_scale: f32,
}

impl<B: Backend> StochasticActor<B> {
    fn new(
        obs_dim: usize,
        hidden: usize,
        action_dim: usize,
        scale: f32,
        device: &B::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            mean: LinearConfig::new(hidden, action_dim).init(device),
            log_std: LinearConfig::new(hidden, action_dim).init(device),
            action_dim,
            action_scale: scale,
        }
    }

    fn mean_and_log_std(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = relu(self.fc1.forward(obs));
        let h = relu(self.fc2.forward(h));
        let mean = self.mean.forward(h.clone());
        let log_std = self.log_std.forward(h).clamp(-5.0_f32, 2.0_f32);
        (mean, log_std)
    }

    fn sample_impl(&self, obs: Tensor<B, 2>, eps: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
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
        let sp = softplus(neg_two_z, 1.0);
        let per_dim_jac: Tensor<B, 2> = (z.clone().neg() - sp + ln_2).mul_scalar(2.0);
        let per_dim = per_dim_gauss - per_dim_jac;
        let log_prob_z = per_dim.sum_dim(1).squeeze_dim::<1>(1);
        let log_scale_abs = self.action_scale.abs().ln();
        let log_prob = log_prob_z.sub_scalar(log_scale_abs * action_dim as f32);
        let action = tanh(z).mul_scalar(self.action_scale);
        (action, log_prob)
    }
}

impl<B: AutodiffBackend> SquashedGaussianPolicy<B, 2, 2> for StochasticActor<B> {
    fn action_dim(&self) -> usize {
        self.action_dim
    }
    fn forward_sample(&self, obs: Tensor<B, 2>, eps: Tensor<B, 2>) -> SampleOutput<B, 2> {
        let (action, log_prob) = self.sample_impl(obs, eps);
        SampleOutput { action, log_prob }
    }
    fn forward_sample_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        eps: Tensor<B::InnerBackend, 2>,
    ) -> SampleOutput<B::InnerBackend, 2> {
        let (action, log_prob) = inner.sample_impl(obs, eps);
        SampleOutput { action, log_prob }
    }
    fn deterministic_action(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mean, _) = self.mean_and_log_std(obs);
        tanh(mean).mul_scalar(self.action_scale)
    }
}

#[derive(Module, Debug)]
struct CriticMlp<B: Backend> {
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
        let active = self.active.remove(&id).expect("paired active param");
        let tau = self.tau;
        param.map(move |t| {
            let device = t.device();
            let a = Tensor::<B, D>::from_data(active, &device);
            t.mul_scalar(1.0 - tau) + a.mul_scalar(tau)
        })
    }
}
fn polyak_update<B: Backend, M: Module<B>>(active: M, target: M, tau: f32) -> M {
    let mut c = ParamCollector::<B> {
        tensors: HashMap::new(),
        _marker: std::marker::PhantomData,
    };
    active.visit(&mut c);
    let mut m = PolyakMapper::<B> {
        active: c.tensors,
        tau,
        _marker: std::marker::PhantomData,
    };
    target.map(&mut m)
}

type Be = Autodiff<NdArray>;
type Agent = SacAgent<
    Be,
    StochasticActor<Be>,
    CriticMlp<Be>,
    PendulumObservation,
    PendulumAction,
    1,
    2,
    1,
    2,
>;

fn build_agent() -> Agent {
    let device = Default::default();
    let actor: StochasticActor<Be> = StochasticActor::new(3, 256, 1, 2.0, &device);
    let critic_1: CriticMlp<Be> = CriticMlp::new(3, 1, 256, &device);
    let critic_2: CriticMlp<Be> = CriticMlp::new(3, 1, 256, &device);
    let config = SacTrainingConfigBuilder::new()
        .buffer_capacity(10_000)
        .batch_size(256)
        .learning_starts(0)
        .policy_frequency(2)
        .build();
    SacAgent::new(actor, critic_1, critic_2, config, device)
}

fn bench_act(c: &mut Criterion) {
    let agent = build_agent();
    let obs = PendulumObservation {
        cos_theta: 1.0,
        sin_theta: 0.0,
        theta_dot: 0.0,
    };
    let mut rng = StdRng::seed_from_u64(0);
    c.bench_function("sac_act_single_obs_eval", |b| {
        b.iter(|| {
            let _: PendulumAction = agent.act(&obs, false, &mut rng);
        });
    });
}

fn bench_learn(c: &mut Criterion) {
    let mut agent = build_agent();
    let mut env = Pendulum::with_config(PendulumConfig::default());
    let mut rng = StdRng::seed_from_u64(0);

    let mut snap = env.reset().expect("reset");
    for _ in 0..2_000 {
        let obs = *snap.observation();
        let action = agent.act(&obs, true, &mut rng);
        let next = env.step(action).expect("step");
        let r: f32 = (*next.reward()).into();
        let done = next.is_done();
        agent.remember(
            obs,
            &PendulumAction::new(0.0).unwrap(),
            r,
            *next.observation(),
            done,
        );
        agent.on_env_step();
        snap = if done {
            env.reset().expect("reset")
        } else {
            next
        };
    }
    c.bench_function("sac_learn_step_batch256", |b| {
        b.iter(|| {
            let _ = agent.learn_step(&mut rng);
        });
    });
}

criterion_group!(micro, bench_act, bench_learn);
criterion_main!(micro);
