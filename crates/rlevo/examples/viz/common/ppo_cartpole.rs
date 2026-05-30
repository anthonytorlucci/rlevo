//! Shared PPO-on-[`CartPole`] scaffolding for the viz examples.
//!
//! The three cartpole viz examples ([`tui_ppo_cartpole`],
//! [`record_ppo_cartpole`], [`report_ppo_cartpole_with_client`]) differ
//! only in which visualisation tier they wire up — the *agent*, the
//! *value network*, the *hyperparameters*, and the *training call* are
//! byte-for-byte identical. That shared core lives here so each example
//! is a thin wrapper around its viz tier rather than a copy of the PPO
//! boilerplate.
//!
//! Included from each example via:
//!
//! ```ignore
//! #[path = "common/ppo_cartpole.rs"]
//! mod ppo_cartpole;
//! ```
//!
//! Each example builds its own env composition (the taps differ per
//! tier), then calls [`build_agent`] + [`train`] to drive learning.

// Each example pulls a subset of this surface; the unused remainder is
// expected, not a smell.
#![allow(dead_code)]

use burn::backend::{Autodiff, Flex};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::Rng;

use rlevo_core::environment::Environment;
use rlevo_core::reward::ScalarReward;

use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::{PpoAgent, PpoAgentError};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;
use rlevo_reinforcement_learning::algorithms::ppo::train::train_discrete;

/// Deterministic seed shared by the env, the RNG, and the recording config.
pub const SEED: u64 = 42;
/// Rollout horizon per PPO update.
pub const NUM_STEPS: usize = 128;
/// Emit a structured training-progress event every this many steps.
pub const LOG_EVERY: usize = 1_024;
/// Episodes are truncated at this length via [`TimeLimit`].
pub const EPISODE_TIME_LIMIT: usize = 500;

const HIDDEN: usize = 64;
const OBS_DIM: usize = 4;
const NUM_ACTIONS: usize = 2;

/// Autodiff backend the cartpole viz examples train on.
pub type Be = Autodiff<Flex>;

/// Concrete PPO agent type the examples drive — a categorical policy head
/// over a two-hidden-layer value MLP.
pub type CartPoleAgent =
    PpoAgent<Be, CategoricalPolicyHead<Be>, ValueMlp<Be>, CartPoleObservation, 1, 2>;

/// Two-hidden-layer `tanh` MLP critic, matching the `ppo_cart_pole`
/// reference example so comparisons stay apples-to-apples.
#[derive(Module, Debug)]
pub struct ValueMlp<B: Backend> {
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

/// Builds the base [`CartPole`] wrapped in a [`TimeLimit`]. Each example
/// then adds its own viz tap(s) on top of this.
pub fn base_env() -> TimeLimit<CartPole> {
    let base = CartPole::with_config(CartPoleConfig {
        seed: SEED,
        ..CartPoleConfig::default()
    });
    TimeLimit::new(base, EPISODE_TIME_LIMIT)
}

/// Builds a fresh PPO agent with the shared hyperparameters. `total_timesteps`
/// sizes the learning-rate / clip annealing schedule.
pub fn build_agent(total_timesteps: usize) -> CartPoleAgent {
    let device = Default::default();

    let policy: CategoricalPolicyHead<Be> = CategoricalPolicyHeadConfig {
        obs_dim: OBS_DIM,
        hidden: HIDDEN,
        num_actions: NUM_ACTIONS,
    }
    .init::<Be>(&device);
    let value: ValueMlp<Be> = ValueMlp::new(OBS_DIM, HIDDEN, &device);

    let config = PpoTrainingConfigBuilder::new()
        .num_envs(1)
        .num_steps(NUM_STEPS)
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

/// Trains `agent` against `env` for `total_timesteps`, hiding the long
/// `train_discrete` turbofish every call site would otherwise repeat.
///
/// `env` is generic over the viz-tier composition: a bare [`TimeLimit`],
/// a `TuiEnvTap`-wrapped env, a `RecordingTap`-wrapped env, or any nesting
/// of those — they all forward `CartPole`'s observation / action / reward
/// associated types.
pub fn train<E>(
    agent: &mut CartPoleAgent,
    env: &mut E,
    rng: &mut impl Rng,
    total_timesteps: usize,
) -> Result<(), PpoAgentError>
where
    E: Environment<
            1,
            1,
            1,
            ObservationType = CartPoleObservation,
            ActionType = CartPoleAction,
            RewardType = ScalarReward,
        >,
{
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        agent,
        env,
        rng,
        total_timesteps,
        LOG_EVERY,
    )
}
