//! Micro benchmarks for the QR-DQN agent's learning step, across quantile
//! counts.
//!
//! Each case primes a replay buffer with 2 000 `CartPole` steps, then times
//! one `learn_step` at a spread of `(quantiles, batch_size)` settings. The
//! quantile Huber loss inside the step is a `(batch, N, N)` broadcast, so a
//! regression that widened it to `(batch, batch, N, N)` shows as step time
//! growing with the square of the batch
//! (`cargo bench -p rlevo-reinforcement-learning --bench qrdqn_bench`).

use burn::tensor::Device;

use criterion::{Criterion, criterion_group, criterion_main};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::agent::Agent as _;
use rlevo_core::environment::Environment;
use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::{
    MlpDistributionalQ, MlpDistributionalQConfig,
};
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_agent::QrDqnAgent;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfig;

type Agent = QrDqnAgent<MlpDistributionalQ, CartPoleObservation, CartPoleAction, 1, 2>;

fn primed_agent(num_quantiles: usize, batch_size: usize) -> Agent {
    let device = Device::flex();
    let config = QrDqnTrainingConfig {
        batch_size,
        replay_buffer_capacity: 10_000,
        learning_starts: 0,
        train_frequency: 1,
        num_quantiles,
        ..Default::default()
    };
    let model = MlpDistributionalQConfig {
        obs_dim: 4,
        hidden: vec![64, 64],
        num_actions: 2,
        num_atoms: num_quantiles,
    }
    .try_init(&device)
    .expect("valid network config");
    let mut agent = QrDqnAgent::new(model, config, device).expect("valid config");

    let mut env = CartPole::with_config(CartPoleConfig::default()).expect("valid config");
    let mut rng = StdRng::seed_from_u64(0);
    let mut snap = env.reset().expect("reset");
    for _ in 0..2_000 {
        let obs = snap.observation;
        let action = agent.act(&obs, &mut rng);
        let next = env.step(action).expect("step");
        let reward: f32 = (next.reward).into();
        let done = next.status.is_done();
        agent.remember(obs, &action, reward, next.observation, next.status);
        snap = if done {
            env.reset().expect("reset")
        } else {
            next
        };
    }
    agent
}

fn bench_learn_step(c: &mut Criterion) {
    for &num_quantiles in &[51_usize, 101, 200] {
        for &batch in &[32_usize, 128] {
            let mut agent = primed_agent(num_quantiles, batch);
            let mut rng = StdRng::seed_from_u64(1);
            let label = format!("qrdqn_learn_step/quantiles={num_quantiles}/batch={batch}");
            c.bench_function(&label, |b| {
                b.iter(|| {
                    let _ = agent.learn_step(&mut rng);
                });
            });
        }
    }
}

criterion_group!(micro, bench_learn_step);
criterion_main!(micro);
