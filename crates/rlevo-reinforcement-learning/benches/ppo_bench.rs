//! Micro benchmarks for one PPO rollout and its update, driven through
//! `Agent`.
//!
//! Each iteration shows the agent `num_steps` transitions on a synthetic
//! `CartPole` observation; the last one fills the rollout, so that call runs
//! GAE, advantage normalisation and the clipped-surrogate epochs. That is the per-rollout cost a training run pays
//! (`cargo bench -p rlevo-reinforcement-learning --bench ppo_bench`).

use burn::tensor::Device;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::agent::{Agent, Transition};
use rlevo_core::environment::EpisodeStatus;
use rlevo_environments::classic::cartpole::{CartPoleAction, CartPoleObservation};
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfig;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::{MlpValue, MlpValueConfig};

type CartPoleAgent = PpoAgent<CategoricalPolicyHead, MlpValue, CartPoleObservation, 1, 2>;

fn build_agent(num_steps: usize) -> CartPoleAgent {
    let device = Device::flex();
    let policy = CategoricalPolicyHeadConfig {
        obs_dim: 4,
        hidden: 64,
        num_actions: 2,
    }
    .try_init(&device)
    .expect("valid head config");
    let value = MlpValueConfig {
        obs_dim: 4,
        hidden: vec![64, 64],
    }
    .try_init(&device)
    .expect("valid value network config");
    // No annealing, so every timed rollout updates at the same rate.
    let ppo = PpoTrainingConfig {
        num_envs: 1,
        num_steps,
        num_minibatches: 4,
        update_epochs: 4,
        anneal_lr: false,
        ..Default::default()
    };
    let config = ppo;
    PpoAgent::new(policy, value, config, device, 1).expect("valid config")
}

/// One rollout of a slowly moving synthetic observation, with an episode
/// ending every 50 steps.
// The step index is bounded by `num_steps`, at most 2 048 here, far below
// f32's 2^24 exact-integer limit.
#[allow(clippy::cast_precision_loss)]
fn rollout(agent: &mut CartPoleAgent, num_steps: usize, rng: &mut StdRng) {
    for i in 0..num_steps {
        let x = (i as f32 * 0.01).sin();
        let obs = CartPoleObservation {
            cart_pos: x,
            cart_vel: 0.0,
            pole_angle: 0.01 * x,
            pole_ang_vel: 0.0,
        };
        let next_obs = CartPoleObservation {
            cart_pos: x + 0.01,
            ..obs
        };
        let action: CartPoleAction = agent.act(&obs, rng);
        let status = if (i + 1) % 50 == 0 {
            EpisodeStatus::Terminated
        } else {
            EpisodeStatus::Running
        };
        agent.on_transition(
            Transition {
                obs: &obs,
                action: &action,
                reward: 1.0,
                next_obs: &next_obs,
                status,
            },
            rng,
        );
    }
}

fn bench_rollout_and_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("ppo_rollout_and_update");
    group.sample_size(10);
    for &num_steps in &[128_usize, 512, 2048] {
        let mut agent = build_agent(num_steps);
        let mut rng = StdRng::seed_from_u64(0);
        group.bench_with_input(
            BenchmarkId::from_parameter(num_steps),
            &num_steps,
            |b, &n| {
                b.iter(|| rollout(&mut agent, n, &mut rng));
            },
        );
    }
    group.finish();
}

criterion_group!(micro, bench_rollout_and_update);
criterion_main!(micro);
