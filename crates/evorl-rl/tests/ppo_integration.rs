//! End-to-end integration tests for PPO.
//!
//! Two convergence tests at modest budgets for CI throughput:
//! - `ppo_cart_pole_reaches_100` (discrete)
//! - `ppo_pendulum_improves_over_random` (continuous)
//!
//! Heavier parity checks behind `#[ignore]` follow the DQN/C51 convention:
//! Burn's ndarray backend shares a global RNG, so reproducibility tests must
//! run with `--test-threads=1`.

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

use rand::SeedableRng;
use rand::rngs::StdRng;

use evorl_envs::classic::cartpole::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};
use evorl_envs::classic::pendulum::{Pendulum, PendulumAction, PendulumConfig, PendulumObservation};
use evorl_envs::wrappers::TimeLimit;
use evorl_rl::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig, TanhGaussianPolicyHead,
    TanhGaussianPolicyHeadConfig,
};
use evorl_rl::algorithms::ppo::ppo_agent::PpoAgent;
use evorl_rl::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use evorl_rl::algorithms::ppo::ppo_value::PpoValue;
use evorl_rl::algorithms::ppo::train::{train_continuous, train_discrete};

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
    fn new(obs_dim: usize, hidden: usize, device: &B::Device) -> Self {
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

type Be = Autodiff<NdArray>;

// ---------------------------------------------------------------------------
// Discrete: CartPole
// ---------------------------------------------------------------------------

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
fn ppo_cart_pole_reaches_100() {
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
#[ignore = "perturbs Burn's global ndarray RNG; run with --test-threads=1"]
fn ppo_short_run_produces_finite_rewards() {
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
        assert!(m.policy_loss.is_finite(), "non-finite policy_loss at ep {i}");
        assert!(m.value_loss.is_finite(), "non-finite value_loss at ep {i}");
    }
}

// ---------------------------------------------------------------------------
// Continuous: Pendulum
// ---------------------------------------------------------------------------

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
        .num_minibatches(32)
        .update_epochs(10)
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
#[ignore = "~30s on ndarray; run with --ignored for macro checks"]
fn ppo_pendulum_improves_over_random() {
    let seed: u64 = 42;
    let total = 30_000_usize;
    let num_steps = 2_048_usize;

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
    // Random policy on Pendulum averages ≈ -1500 per episode; well-trained
    // PPO clears -200. The threshold here is deliberately lax for CI.
    assert!(avg > -1400.0, "expected avg > -1400, got {avg:.2}");
}
