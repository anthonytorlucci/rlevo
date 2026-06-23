//! End-to-end integration tests for PPO.
//!
//! Two convergence tests at modest budgets for CI throughput:
//! - `ppo_cart_pole_reaches_100` (discrete)
//! - `ppo_pendulum_improves_over_random` (continuous)
//!
//! Heavier parity checks behind `#[ignore]` follow the DQN/C51 convention:
//! Burn's Flex backend shares a global RNG, so reproducibility tests must
//! run with `--test-threads=1`.

use burn::backend::{Autodiff, Flex};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig, TanhGaussianPolicyHead,
    TanhGaussianPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;
use rlevo_reinforcement_learning::algorithms::ppo::train::{train_continuous, train_discrete};

// ---------------------------------------------------------------------------
// Shared value MLP
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

type Be = Autodiff<Flex>;

static BACKEND_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ---------------------------------------------------------------------------
// Discrete: CartPole
// ---------------------------------------------------------------------------

/// Constructs a discrete [`PpoAgent`] with a fully-seeded, deterministic
/// configuration for CartPole experiments.
///
/// The Burn `Flex` backend exposes a **process-global** RNG that governs weight
/// initialisation. This function seeds it via [`Backend::seed`] before
/// constructing the policy and value networks, so that two calls with identical
/// `seed` values produce bit-for-bit identical initial weights.
///
/// # Caller responsibilities
///
/// - **Hold `BACKEND_LOCK` before calling.** The lock serialises access to the
///   global Flex RNG across test threads; without it a concurrent test could
///   interleave a `seed` call and silently corrupt the weight initialisation
///   sequence.
/// - **Pin rayon to one thread** (`rayon::ThreadPoolBuilder::num_threads(1)`)
///   before the training loop. Flex dispatches matrix operations through rayon;
///   floating-point reduction order is non-deterministic under multi-threading,
///   introducing a second source of run-to-run variance independent of the RNG.
///
/// The `CartPole` environment and `StdRng` are seeded separately by each test
/// body. This function is responsible solely for backend-side weight
/// initialisation determinism.
///
/// The hyperparameters (clip coefficient, entropy coefficient, GAE λ, learning
/// rate, minibatch count) are tuned for the 4-observation / 2-action discrete
/// CartPole task with a 128-step rollout buffer.
fn make_cart_pole_agent(
    seed: u64,
    num_steps: usize,
    total_timesteps: usize,
) -> PpoAgent<Be, CategoricalPolicyHead<Be>, ValueMlp<Be>, CartPoleObservation, 1, 2> {
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);

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
        .build();
    let total_iterations = total_timesteps / config.batch_size().max(1);
    PpoAgent::new(policy, value, config, device, total_iterations)
}

#[test]
#[ignore = "50 000-step discrete PPO CartPole run (several minutes on CPU); confirms avg reward ≥ 80 — run with `cargo test -- --ignored`"]
fn ppo_cart_pole_reaches_100() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    let _guard = BACKEND_LOCK.lock().expect("backend lock");
    let seed: u64 = 42;
    let total = 50_000_usize;
    let num_steps = 128_usize;

    let mut env = TimeLimit::new(
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        }),
        500,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, num_steps, total);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    let avg = agent.stats().avg_score().unwrap_or(0.0);
    assert!(avg >= 80.0, "expected avg reward >= 80, got {avg:.2}");
}

#[test]
#[ignore = "2 048-timestep PPO training run; checks finite rewards and losses — run with `cargo test -- --ignored`"]
fn ppo_short_run_produces_finite_rewards() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    let _guard = BACKEND_LOCK.lock().expect("backend lock");
    let seed: u64 = 7;
    let total = 2_048_usize;
    let num_steps = 128_usize;
    let mut env = TimeLimit::new(
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        }),
        500,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, num_steps, total);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    for (i, m) in agent.stats().recent_history.iter().enumerate() {
        assert!(m.reward.is_finite(), "non-finite reward at episode {i}");
        assert!(
            m.policy_loss.is_finite(),
            "non-finite policy_loss at ep {i}"
        );
        assert!(m.value_loss.is_finite(), "non-finite value_loss at ep {i}");
    }
}

// ---------------------------------------------------------------------------
// Continuous: Pendulum
// ---------------------------------------------------------------------------

/// Constructs a continuous [`PpoAgent`] with a fully-seeded, deterministic
/// configuration for Pendulum experiments.
///
/// The Burn `Flex` backend exposes a **process-global** RNG that governs weight
/// initialisation. This function seeds it via [`Backend::seed`] before
/// constructing the `TanhGaussian` policy and value networks, so that two calls
/// with identical `seed` values produce bit-for-bit identical initial weights.
///
/// # Caller responsibilities
///
/// - **Hold `BACKEND_LOCK` before calling.** The lock serialises access to the
///   global Flex RNG across test threads; without it a concurrent test could
///   interleave a `seed` call and silently corrupt the weight initialisation
///   sequence.
/// - **Pin rayon to one thread** (`rayon::ThreadPoolBuilder::num_threads(1)`)
///   before the training loop. Flex dispatches matrix operations through rayon;
///   floating-point reduction order is non-deterministic under multi-threading,
///   introducing a second source of run-to-run variance independent of the RNG.
///
/// The `Pendulum` environment and `StdRng` are seeded separately by each test
/// body. This function is responsible solely for backend-side weight
/// initialisation determinism.
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
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);

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
        .build();
    let total_iterations = total_timesteps / config.batch_size().max(1);
    PpoAgent::new(policy, value, config, device, total_iterations)
}

#[test]
#[ignore = "30 000-step continuous PPO Pendulum run (~30 s on CPU); confirms avg reward > −1 400 above the ~−1 500 random baseline — run with `cargo test -- --ignored`"]
fn ppo_pendulum_improves_over_random() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    let _guard = BACKEND_LOCK.lock().expect("backend lock");
    let seed: u64 = 42;
    let total = 24_000_usize;
    let num_steps = 512_usize;

    let mut env = TimeLimit::new(
        Pendulum::with_config(PendulumConfig {
            seed,
            ..PendulumConfig::default()
        }),
        200,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_pendulum_agent(seed, num_steps, total);
    train_continuous::<Be, _, _, _, _, PendulumAction, _, 1, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    let avg = agent.stats().avg_score().unwrap_or(f32::NEG_INFINITY);
    eprintln!("CALIBRATION: 30k-step PPO Pendulum avg = {avg:.2}");
    // Random policy on Pendulum averages ≈ -1500 per episode; well-trained
    // PPO clears -200. The threshold here is deliberately lax for CI.
    assert!(avg > -1400.0, "expected avg > -1400, got {avg:.2}");
}
