//! End-to-end integration tests for PPO.
//!
//! Two learning tests at modest budgets for CI throughput:
//! - `ppo_cartpole_converges` (discrete, absolute floor)
//! - `ppo_pendulum_improves_over_random` (continuous, measured random baseline)
//!
//! Heavier parity checks behind `#[ignore]` follow the DQN/C51 convention.
//!
//! The `Flex` determinism preamble ([`flex_guard`] / [`seeded_device`]), the
//! shared seeded `CartPole` fixture ([`cartpole_seeded`]), and the acceptance
//! assertions live in the `rlevo-test-support` dev-crate; only the value
//! network and the algorithm-specific tests remain here.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{CartPoleAction, CartPoleObservation};
use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig, TanhGaussianPolicyHead,
    TanhGaussianPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::{
    PpoTrainingConfig, PpoTrainingConfigBuilder,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;
use rlevo_reinforcement_learning::algorithms::ppo::train::{train_continuous, train_discrete};

use rlevo_test_support::assert::assert_all_finite;
use rlevo_test_support::baseline::{random_return, uniform_bounded};
use rlevo_test_support::env::cartpole_seeded;
use rlevo_test_support::flex::{FlexAutodiff as Be, flex_guard, seeded_device};
use rlevo_test_support::{TrainOutcome, rl_learning_test, rl_reproducibility_test};

// ---------------------------------------------------------------------------
// Shared value MLP. Implements PPO's value trait and so stays test-local.
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct ValueMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> ValueMlp<B> {
    fn new(
        obs_dim: usize,
        hidden: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            head: LinearConfig::new(hidden, 1).init(device),
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = tanh(self.fc1.forward(obs));
        let h = tanh(self.fc2.forward(h));
        self.head.forward(h).squeeze_dim::<1>(1)
    }
}

impl<B: AutodiffBackend> PpoValue<B, 2> for ValueMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs)
    }
}

// ---------------------------------------------------------------------------
// Discrete: CartPole
// ---------------------------------------------------------------------------

/// Constructs a deterministic discrete [`PpoAgent`] for `CartPole`.
///
/// Seeds the backend via [`seeded_device`] so two calls with the same `seed`
/// produce bit-for-bit identical initial weights. Callers must hold the
/// [`flex_guard`] lock for the duration of the test.
///
/// The hyperparameters (clip coefficient, entropy coefficient, GAE λ, learning
/// rate, minibatch count) are tuned for the 4-observation / 2-action discrete
/// `CartPole` task with a 128-step rollout buffer.
fn make_cart_pole_agent(
    seed: u64,
    num_steps: usize,
    total_timesteps: usize,
) -> PpoAgent<Be, CategoricalPolicyHead<Be>, ValueMlp<Be>, CartPoleObservation, 1, 2> {
    let device = seeded_device::<Be>(seed);

    let policy = CategoricalPolicyHeadConfig {
        obs_dim: 4,
        hidden: 64,
        num_actions: 2,
    }
    .init::<Be>(&device);
    let value = ValueMlp::new(4, 64, &device);

    let config = PpoTrainingConfigBuilder::new()
        .num_envs(1)
        .num_steps(num_steps)
        .num_minibatches(4)
        .update_epochs(4)
        .learning_rate(2.5e-4)
        .clip_coef(0.2)
        .entropy_coef(0.01)
        .value_coef(0.5)
        .gamma(0.99)
        .gae_lambda(0.95)
        .build()
        .expect("valid config");
    let total_iterations = total_timesteps / config.batch_size().max(1);
    PpoAgent::new(policy, value, config, device, total_iterations).expect("valid config")
}

/// Builds and trains a discrete PPO agent on the shared seeded `CartPole` for
/// `total` steps (128-step rollouts), returning the standardised outcome
/// consumed by the suite macros.
fn run_cartpole(seed: u64, total: usize) -> TrainOutcome {
    let mut env = TimeLimit::new(cartpole_seeded(seed), 500);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, 128, total);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    let stats = agent.stats();
    TrainOutcome {
        avg_score: stats.avg_score().unwrap_or(0.0),
        rewards: stats.recent_history.iter().map(|m| m.reward).collect(),
    }
}

// `CartPole` target: after 50k steps the moving average should clear 80.
// (Generated by `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "50 000-step discrete PPO CartPole run (several minutes on CPU); confirms avg reward ≥ 80 — run with `cargo test -- --ignored`"]
    ppo_cartpole_converges,
    reaches(80.0),
    seed = 42,
    total = 50_000,
    run = run_cartpole,
}

// Seeded reproducibility: two same-seed CartPole runs must produce identical
// reward sequences. (Generated by `rl_reproducibility_test!`.)
rl_reproducibility_test! {
    #[ignore = "4 096-step reproducibility check (two sequential CartPole runs); run with `cargo test -- --ignored`"]
    ppo_cartpole_flex_reproducibility,
    seq,
    seed = 123,
    total = 2_048,
    run = run_cartpole,
}

/// `PpoAgent::new` rejects `num_envs != 1` (v1 supports sequential rollout
/// only). The config is assembled directly so the agent-level validation seam
/// is what rejects it, not the builder.
#[test]
fn ppo_agent_new_rejects_multiple_envs() {
    let _guard = flex_guard();
    let device = seeded_device::<Be>(1);
    let policy = CategoricalPolicyHeadConfig {
        obs_dim: 4,
        hidden: 64,
        num_actions: 2,
    }
    .init::<Be>(&device);
    let value = ValueMlp::new(4, 64, &device);
    let config = PpoTrainingConfig {
        num_envs: 2,
        ..PpoTrainingConfig::default()
    };
    let result =
        PpoAgent::<Be, _, _, CartPoleObservation, 1, 2>::new(policy, value, config, device, 1);
    let err = result.unwrap_err();
    assert_eq!(err.field, "num_envs");
}

#[test]
#[ignore = "2 048-timestep PPO training run; checks finite rewards and losses — run with `cargo test -- --ignored`"]
fn ppo_cartpole_produces_finite_rewards() {
    let _guard = flex_guard();
    let seed: u64 = 7;
    let total = 2_048_usize;
    let num_steps = 128_usize;
    let mut env = TimeLimit::new(cartpole_seeded(seed), 500);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, num_steps, total);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    let history = &agent.stats().recent_history;
    assert_all_finite(
        "reward",
        &history.iter().map(|m| m.reward).collect::<Vec<_>>(),
    );
    assert_all_finite(
        "policy_loss",
        &history.iter().map(|m| m.policy_loss).collect::<Vec<_>>(),
    );
    assert_all_finite(
        "value_loss",
        &history.iter().map(|m| m.value_loss).collect::<Vec<_>>(),
    );
}

// ---------------------------------------------------------------------------
// Continuous: Pendulum
// ---------------------------------------------------------------------------

/// Constructs a deterministic continuous [`PpoAgent`] for Pendulum.
///
/// Seeds the backend via [`seeded_device`] so two calls with the same `seed`
/// produce identical initial weights. Callers must hold the [`flex_guard`] lock
/// for the duration of the test.
///
/// The hyperparameters (higher `update_epochs`, zero entropy coefficient,
/// `action_scale` matching the ±2 N·m torque limit, GAE λ, γ=0.9) are tuned
/// for the 3-observation / 1-action continuous Pendulum task with a 2 048-step
/// rollout buffer.
fn make_pendulum_agent(
    seed: u64,
    num_steps: usize,
    total_timesteps: usize,
) -> PpoAgent<Be, TanhGaussianPolicyHead<Be>, ValueMlp<Be>, PendulumObservation, 1, 2> {
    let device = seeded_device::<Be>(seed);

    let policy = TanhGaussianPolicyHeadConfig {
        obs_dim: 3,
        hidden: 64,
        action_dim: 1,
        log_std_init: 0.0,
        action_scale: 2.0,
    }
    .init::<Be>(&device);
    let value = ValueMlp::new(3, 64, &device);

    let config = PpoTrainingConfigBuilder::new()
        .num_envs(1)
        .num_steps(num_steps)
        .num_minibatches(4)
        .update_epochs(4)
        .learning_rate(3e-4)
        .clip_coef(0.2)
        .entropy_coef(0.0)
        .value_coef(0.5)
        .gamma(0.9)
        .gae_lambda(0.95)
        .action_scale(2.0)
        .build()
        .expect("valid config");
    let total_iterations = total_timesteps / config.batch_size().max(1);
    PpoAgent::new(policy, value, config, device, total_iterations).expect("valid config")
}

/// Builds and trains a continuous PPO agent on Pendulum for `total` steps
/// (512-step rollouts), returning the standardised outcome consumed by the
/// suite macros. Emits a one-line calibration print of the final average —
/// PPO Pendulum drifts run-to-run, so the printed number aids threshold tuning.
fn run_pendulum(seed: u64, total: usize) -> TrainOutcome {
    let mut env = TimeLimit::new(
        Pendulum::with_config(PendulumConfig {
            seed,
            ..PendulumConfig::default()
        })
        .expect("valid config"),
        200,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_pendulum_agent(seed, 512, total);
    train_continuous::<Be, _, _, _, _, PendulumAction, _, 1, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    let stats = agent.stats();
    let avg = stats.avg_score().unwrap_or(f32::NEG_INFINITY);
    eprintln!("CALIBRATION: PPO Pendulum avg = {avg:.2}");
    TrainOutcome {
        avg_score: avg,
        rewards: stats.recent_history.iter().map(|m| m.reward).collect(),
    }
}

/// Mean episode return of a uniform-random torque policy on the `TimeLimit`ed
/// Pendulum, measured over 100 episodes from `seed`. Baseline the trained PPO
/// agent must beat.
fn random_pendulum(seed: u64) -> f32 {
    let mut env = TimeLimit::new(
        Pendulum::with_config(PendulumConfig {
            seed,
            ..PendulumConfig::default()
        })
        .expect("valid config"),
        200,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    random_return(
        &mut env,
        100,
        200,
        &mut rng,
        uniform_bounded::<1, PendulumAction>,
    )
}

// Well-trained PPO clears -200; a uniform-random torque policy scores
// `random_pendulum`. The margin here is deliberately lax for CI. (Generated by
// `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "30 000-step continuous PPO Pendulum run (~30 s on CPU); confirms avg reward beats a measured uniform-random baseline — run with `cargo test -- --ignored`"]
    ppo_pendulum_improves_over_random,
    improves_over_random(margin = 50.0),
    seed = 42,
    total = 24_000,
    run = run_pendulum,
    random = random_pendulum,
}
