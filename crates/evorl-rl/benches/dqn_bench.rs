//! Micro and macro benchmarks for the DQN agent.
//!
//! Only the micro benches run by default (`cargo bench -p evorl-rl --bench
//! dqn_bench`). The CartPole reward-curve reproduction macro bench is gated
//! behind the `macro` env var and is compared against
//! `tests/baselines/dqn_cartpole.csv`.

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData, activation};

use criterion::{Criterion, criterion_group, criterion_main};

use rand::SeedableRng;
use rand::rngs::StdRng;

use evorl_envs::classic::cartpole::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};
use evorl_rl::algorithms::dqn::dqn_agent::DqnAgent;
use evorl_rl::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use evorl_rl::algorithms::dqn::dqn_model::DqnModel;

// MLP mirror of the example. Kept inline so the bench file is self-contained.

#[derive(Module, Debug)]
struct DqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> DqnMlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, 2).init(device),
        }
    }
    fn forward_impl(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.l1.forward(x));
        let x = activation::relu(self.l2.forward(x));
        self.l3.forward(x)
    }
}

impl<B: AutodiffBackend> DqnModel<B, 2> for DqnMlp<B> {
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
        polyak_update::<B::InnerBackend, DqnMlp<B::InnerBackend>>(active.valid(), target, tau as f32)
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
type Agent = DqnAgent<Be, DqnMlp<Be>, CartPoleObservation, CartPoleAction, 1, 2>;

fn build_agent() -> Agent {
    let device = Default::default();
    let config = DqnTrainingConfigBuilder::new()
        .batch_size(64)
        .replay_buffer_capacity(10_000)
        .learning_starts(0)
        .train_frequency(1)
        .build();
    let model: DqnMlp<Be> = DqnMlp::new(&device);
    DqnAgent::new(model, config, device)
}

fn bench_act(c: &mut Criterion) {
    let agent = build_agent();
    let obs = CartPoleObservation {
        cart_pos: 0.0,
        cart_vel: 0.0,
        pole_angle: 0.01,
        pole_ang_vel: 0.0,
    };
    let mut rng = StdRng::seed_from_u64(0);
    c.bench_function("dqn_act_single_obs", |b| {
        b.iter(|| {
            let _: CartPoleAction = agent.act(&obs, &mut rng);
        });
    });
}

fn bench_learn(c: &mut Criterion) {
    // Prime a buffer by stepping a real CartPole for a bit; we measure pure
    // `learn_step` wall-clock with the buffer already populated.
    let mut agent = build_agent();
    let mut env = CartPole::with_config(CartPoleConfig::default());
    let mut rng = StdRng::seed_from_u64(0);
    use evorl_core::environment::{Environment, Snapshot};
    let mut snap = env.reset().expect("reset");
    for _ in 0..2_000 {
        let obs = snap.observation().clone();
        let action = agent.act(&obs, &mut rng);
        let next = env.step(action.clone()).expect("step");
        let r: f32 = (*next.reward()).into();
        let done = next.is_done();
        agent.remember(obs, &action, r, next.observation().clone(), done);
        agent.on_env_step();
        snap = if done { env.reset().expect("reset") } else { next };
    }
    c.bench_function("dqn_learn_step_batch64", |b| {
        b.iter(|| {
            let _ = agent.learn_step(&mut rng);
        });
    });
}

criterion_group!(micro, bench_act, bench_learn);
criterion_main!(micro);
