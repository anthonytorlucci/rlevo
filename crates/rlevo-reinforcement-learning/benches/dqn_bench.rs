//! Micro benchmarks for the DQN agent.
//!
//! Measures `act` on a single `CartPole` observation and full `learn_step`
//! wall-clock at batch size 64 on a replay buffer primed with 2 000 `CartPole`
//! steps (`cargo bench -p rlevo-reinforcement-learning --bench dqn_bench`).
//! The agent keeps the default `epsilon_start = 1.0` and never decays it, so
//! every timed `act` call takes the exploring branch and does not run the
//! Q-network.

use burn::tensor::Device;

use criterion::{Criterion, criterion_group, criterion_main};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::environment::Environment;
use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfig;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::{MlpQNetwork, MlpQNetworkConfig};

type Agent = DqnAgent<MlpQNetwork, CartPoleObservation, CartPoleAction, 1, 2>;

fn build_agent() -> Agent {
    let device = Device::flex();
    let config = DqnTrainingConfig {
        batch_size: 64,
        replay_buffer_capacity: 10_000,
        learning_starts: 0,
        train_frequency: 1,
        ..Default::default()
    };
    let model = MlpQNetworkConfig {
        obs_dim: 4,
        hidden: vec![64, 64],
        num_actions: 2,
    }
    .try_init(&device)
    .expect("valid Q-network config");
    DqnAgent::new(model, config, device).expect("valid config")
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
    let mut env = CartPole::with_config(CartPoleConfig::default()).expect("valid config");
    let mut rng = StdRng::seed_from_u64(0);

    let mut snap = env.reset().expect("reset");
    for _ in 0..2_000 {
        let obs = snap.observation;
        let action = agent.act(&obs, &mut rng);
        let next = env.step(action).expect("step");
        let r: f32 = (next.reward).into();
        // `done` drives the episode reset; `remember` masks the Bellman
        // bootstrap on the status, and only on termination. CartPole truncates at its step cap, so these differ.
        let done = next.status.is_done();
        agent.remember(obs, &action, r, next.observation, next.status);
        agent.on_env_step();
        snap = if done {
            env.reset().expect("reset")
        } else {
            next
        };
    }
    c.bench_function("dqn_learn_step_batch64", |b| {
        b.iter(|| {
            let _ = agent.learn_step(&mut rng);
        });
    });
}

criterion_group!(micro, bench_act, bench_learn);
criterion_main!(micro);
