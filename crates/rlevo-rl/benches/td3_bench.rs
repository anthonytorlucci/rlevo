//! Micro benchmarks for the TD3 agent.
//!
//! Measures actor-forward latency and full `learn_step` wall-clock (twin
//! critic update + delayed actor/Polyak step) on a primed replay buffer,
//! following the pattern established by `ddpg_bench`.

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

use criterion::{Criterion, criterion_group, criterion_main};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_rl::algorithms::td3::td3_agent::Td3Agent;
use rlevo_rl::algorithms::td3::td3_config::Td3TrainingConfigBuilder;
use rlevo_rl::algorithms::td3::td3_model::{ContinuousQ, DeterministicPolicy};

#[derive(Module, Debug)]
struct ActorMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
    action_scale: f32,
}

impl<B: Backend> ActorMlp<B> {
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
            head: LinearConfig::new(hidden, action_dim).init(device),
            action_scale: scale,
        }
    }
    fn forward_impl(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.fc1.forward(x));
        let h = relu(self.fc2.forward(h));
        tanh(self.head.forward(h)).mul_scalar(self.action_scale)
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
type Agent =
    Td3Agent<Be, ActorMlp<Be>, CriticMlp<Be>, PendulumObservation, PendulumAction, 1, 2, 1, 2>;

fn build_agent() -> Agent {
    let device = Default::default();
    let actor: ActorMlp<Be> = ActorMlp::new(3, 256, 1, 2.0, &device);
    let critic_1: CriticMlp<Be> = CriticMlp::new(3, 1, 256, &device);
    let critic_2: CriticMlp<Be> = CriticMlp::new(3, 1, 256, &device);
    let config = Td3TrainingConfigBuilder::new()
        .buffer_capacity(10_000)
        .batch_size(256)
        .learning_starts(0)
        .policy_frequency(2)
        .build();
    Td3Agent::new(actor, critic_1, critic_2, config, device)
}

fn bench_act(c: &mut Criterion) {
    let agent = build_agent();
    let obs = PendulumObservation {
        cos_theta: 1.0,
        sin_theta: 0.0,
        theta_dot: 0.0,
    };
    let mut rng = StdRng::seed_from_u64(0);
    c.bench_function("td3_act_single_obs_eval", |b| {
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
    c.bench_function("td3_learn_step_batch256", |b| {
        b.iter(|| {
            let _ = agent.learn_step(&mut rng);
        });
    });
}

criterion_group!(micro, bench_act, bench_learn);
criterion_main!(micro);
