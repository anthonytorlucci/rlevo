//! Micro benchmarks for the TD3 agent.
//!
//! Measures actor-forward latency and full `learn_step` wall-clock (twin
//! critic update + delayed actor/Polyak step) on a primed replay buffer,
//! following the pattern established by `ddpg_bench`.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, tanh};
use burn::tensor::{Device, Tensor};

use criterion::{Criterion, criterion_group, criterion_main};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::environment::Environment;
use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_reinforcement_learning::algorithms::td3::td3_agent::Td3Agent;
use rlevo_reinforcement_learning::algorithms::td3::td3_config::Td3TrainingConfig;
use rlevo_reinforcement_learning::algorithms::td3::td3_model::{ContinuousQ, DeterministicPolicy};

#[derive(Module, Debug)]
struct ActorMlp {
    fc1: Linear,
    fc2: Linear,
    head: Linear,
    action_scale: f32,
}

impl ActorMlp {
    fn new(obs_dim: usize, hidden: usize, action_dim: usize, scale: f32, device: &Device) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            head: LinearConfig::new(hidden, action_dim).init(device),
            action_scale: scale,
        }
    }
    fn forward_impl(&self, x: Tensor<2>) -> Tensor<2> {
        let h = relu(self.fc1.forward(x));
        let h = relu(self.fc2.forward(h));
        tanh(self.head.forward(h)).mul_scalar(self.action_scale)
    }
}

impl DeterministicPolicy<2, 2> for ActorMlp {
    fn forward(&self, obs: Tensor<2>) -> Tensor<2> {
        self.forward_impl(obs)
    }
}

#[derive(Module, Debug)]
struct CriticMlp {
    fc1: Linear,
    fc2: Linear,
    head: Linear,
}

impl CriticMlp {
    fn new(obs_dim: usize, action_dim: usize, hidden: usize, device: &Device) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim + action_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            head: LinearConfig::new(hidden, 1).init(device),
        }
    }
    fn forward_impl(&self, obs: Tensor<2>, act: Tensor<2>) -> Tensor<1> {
        let x = Tensor::cat(vec![obs, act], 1);
        let h = relu(self.fc1.forward(x));
        let h = relu(self.fc2.forward(h));
        self.head.forward(h).squeeze_dim::<1>(1)
    }
}

impl ContinuousQ<2, 2> for CriticMlp {
    fn forward(&self, obs: Tensor<2>, act: Tensor<2>) -> Tensor<1> {
        self.forward_impl(obs, act)
    }
}

type Agent = Td3Agent<ActorMlp, CriticMlp, PendulumObservation, PendulumAction, 1, 2, 1, 2>;

fn build_agent() -> Agent {
    let device = Device::flex();
    let actor: ActorMlp = ActorMlp::new(3, 256, 1, 2.0, &device);
    let critic_1: CriticMlp = CriticMlp::new(3, 1, 256, &device);
    let critic_2: CriticMlp = CriticMlp::new(3, 1, 256, &device);
    let config = Td3TrainingConfig {
        replay_buffer_capacity: 10_000,
        batch_size: 256,
        learning_starts: 0,
        policy_frequency: 2,
        ..Default::default()
    };
    Td3Agent::new(actor, critic_1, critic_2, config, device).expect("valid config")
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
    let mut env = Pendulum::with_config(PendulumConfig::default()).expect("valid config");
    let mut rng = StdRng::seed_from_u64(0);

    let mut snap = env.reset().expect("reset");
    for _ in 0..2_000 {
        let obs = snap.observation;
        let action = agent.act(&obs, true, &mut rng);
        let next = env.step(action).expect("step");
        let r: f32 = (next.reward).into();
        // `done` drives the episode reset; `terminated` is the Bellman
        // bootstrap mask. `Pendulum` only ever truncates, so these differ.
        let done = next.status.is_done();
        let terminated = next.status.is_terminated();
        agent.remember(
            obs,
            &PendulumAction::new(0.0).unwrap(),
            r,
            next.observation,
            terminated,
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
