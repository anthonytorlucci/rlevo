//! Micro benchmarks for the C51 agent's learning step, across support sizes.
//!
//! Each case primes a replay buffer with 2 000 `CartPole` steps, then times
//! one `learn_step` at a spread of `(atoms, batch_size)` settings. The
//! categorical projection inside the step is linear in the atoms, so a
//! regression that made it quadratic shows as step time growing faster than
//! the atoms do (`cargo bench -p rlevo-reinforcement-learning --bench c51_bench`).

use burn::tensor::Device;

use criterion::{Criterion, criterion_group, criterion_main};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::agent::Agent as _;
use rlevo_core::environment::Environment;
use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_reinforcement_learning::algorithms::c51::c51_agent::C51Agent;
use rlevo_reinforcement_learning::algorithms::c51::c51_config::C51TrainingConfig;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::{
    MlpDistributionalQ, MlpDistributionalQConfig,
};

type Agent = C51Agent<MlpDistributionalQ, CartPoleObservation, CartPoleAction, 1, 2>;

fn primed_agent(num_atoms: usize, batch_size: usize) -> Agent {
    let device = Device::flex();
    let config = C51TrainingConfig {
        batch_size,
        replay_buffer_capacity: 10_000,
        learning_starts: 0,
        train_frequency: 1,
        num_atoms,
        v_min: -10.0,
        v_max: 10.0,
        ..Default::default()
    };
    let model = MlpDistributionalQConfig {
        obs_dim: 4,
        hidden: vec![64, 64],
        num_actions: 2,
        num_atoms,
    }
    .try_init(&device)
    .expect("valid network config");
    let mut agent = C51Agent::new(model, config, device).expect("valid config");

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
    for &num_atoms in &[21_usize, 51, 101] {
        for &batch in &[32_usize, 128] {
            let mut agent = primed_agent(num_atoms, batch);
            let mut rng = StdRng::seed_from_u64(1);
            let label = format!("c51_learn_step/atoms={num_atoms}/batch={batch}");
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
