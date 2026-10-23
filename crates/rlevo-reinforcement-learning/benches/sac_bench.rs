//! Micro benchmarks for the SAC agent.
//!
//! Measures actor-sampling latency and full `learn_step` wall-clock (twin critic update + actor +
//! `$\alpha$` step + Polyak) on a primed replay buffer. Mirrors `td3_bench`'s pattern so the two
//! algorithms can be compared side-by-side.

use burn::tensor::Device;

use criterion::{Criterion, criterion_group, criterion_main};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::bounds::Bounds;
use rlevo_core::environment::Environment;
use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_model::{MlpCritic, MlpCriticConfig};
use rlevo_reinforcement_learning::algorithms::sac::sac_agent::SacAgent;
use rlevo_reinforcement_learning::algorithms::sac::sac_config::SacTrainingConfig;
use rlevo_reinforcement_learning::algorithms::sac::sac_policy::{
    SquashedGaussianPolicyHead, SquashedGaussianPolicyHeadConfig,
};

type Agent = SacAgent<
    SquashedGaussianPolicyHead,
    MlpCritic,
    PendulumObservation,
    PendulumAction,
    1,
    2,
    1,
    2,
>;

fn build_agent() -> Agent {
    let device = Device::flex();
    let actor = SquashedGaussianPolicyHeadConfig {
        obs_dim: 3,
        hidden: 256,
        action_dim: 1,
        log_std: Bounds::new(-5.0, 2.0),
        action_scale: 2.0,
        action_bias: 0.0,
    }
    .try_init(&device)
    .expect("valid SAC actor config");
    let critic_1 = MlpCriticConfig {
        obs_dim: 3,
        action_dim: 1,
        hidden: vec![256, 256],
    }
    .try_init(&device)
    .expect("valid critic config");
    let critic_2 = MlpCriticConfig {
        obs_dim: 3,
        action_dim: 1,
        hidden: vec![256, 256],
    }
    .try_init(&device)
    .expect("valid critic config");
    let config = SacTrainingConfig {
        replay_buffer_capacity: 10_000,
        batch_size: 256,
        learning_starts: 0,
        policy_frequency: 2,
        ..Default::default()
    };
    SacAgent::new(actor, critic_1, critic_2, config, device).expect("valid config")
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
    c.bench_function("sac_learn_step_batch256", |b| {
        b.iter(|| {
            let _ = agent.learn_step(&mut rng);
        });
    });
}

criterion_group!(micro, bench_act, bench_learn);
criterion_main!(micro);
