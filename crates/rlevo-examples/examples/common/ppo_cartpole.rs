//! Shared PPO-on-[`CartPole`] scaffolding for the viz examples.
//!
//! This file packages a reproducible PPO-on-CartPole training setup;
//!
//! The two cartpole viz examples ([`tui_ppo_cartpole`] and
//! [`report_ppo_cartpole_with_client`]) differ only in which
//! visualisation tier they wire up — the *agent*, the *value network*,
//! the *hyperparameters*, and the *training call* are byte-for-byte
//! identical. That shared core lives here so each example is a thin
//! wrapper around its viz tier rather than a copy of the PPO boilerplate.
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
const OBS_RANK: usize = 4; // todo! this should be dim. rank = 1
const NUM_ACTIONS: usize = 2;

/// Autodiff backend the cartpole viz examples train on.
///
/// `Flex` (a portable CPU backend), not `Wgpu`, is deliberate. A
/// single-environment PPO rollout is tiny and sequential — every step is a
/// handful of small ops with a host sync to pick an action — so the GPU's
/// per-dispatch overhead dominates and CPU is ~70× faster here. The GPU wins
/// on the opposite shape (large batched work): see the
/// `backend_sweep_neuroevolution` example and the user-book chapter
/// "Choosing a Backend: CPU vs GPU" for the measured comparison.
pub type Be = Autodiff<Flex>;

/// Concrete PPO agent type the examples drive — a categorical policy head
/// over a two-hidden-layer value MLP.
///
/// pub struct PpoAgent<B, P, V, O, const DO: usize, const DB: usize>
/// where
///     B: AutodiffBackend,  --> Be = Autodiff<Flex>
///     P: PpoPolicy<B, DB>,  --> CategoricalPolicyHead<Be>
///     V: PpoValue<B, DB>,  --> ValueMlp<Be>
///     O: Observation<DO> + TensorConvertible<DO, B>, --> CartPoleObservation
/// { /* private fields */ }
///
/// todo! further discuss CategoricalPolicyHead<Be> and CarPoleObservation
/// links:
/// - https://docs.rs/rlevo-reinforcement-learning/0.3.0/rlevo_reinforcement_learning/algorithms/ppo/ppo_agent/struct.PpoAgent.html
/// - https://docs.rs/rlevo-environments/0.3.0/rlevo_environments/classic/cartpole/index.html
pub type CartPoleAgent =
    PpoAgent<Be, CategoricalPolicyHead<Be>, ValueMlp<Be>, CartPoleObservation, 1, 2>;

/// The neural network "brain"
/// This is the critic. It's a multi-layer perceptron (MLP) or
/// feed-forward neural network.
/// It takes `obs_dim` (4) inputs -> `hidden` (64) -> `hidden` (64) -> 1 output.
/// The single output is the estimated value (expected discounted return) of a
/// state. `tanh` activations in `forward_impl` squash intermediate layers into
/// range [-1, 1].
/// todo! discuss why the activation function is important here ...
///
/// Note the policy (actor) is *not* defined here - it's imported as
/// `CategoricalPolicyHead`. That's a categorical distribution over the 2
/// actions, which is what you want for discrete actions like CartPole's.
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

    /// Applies a foward pass ...
    /// The 2 is the input tensor rank (batched observations are 2D:
    /// `[batch_size, obs_dim]`), and `squeeze_dims` the trailing singleton
    /// so the output is a flat vector of values, one per observation.
    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = tanh(self.fc1.forward(obs));
        let h = tanh(self.fc2.forward(h));
        self.head.forward(h).squeeze_dim::<1>(1)
    }
}

/// This is the adapter that lets the PPO framework call this network.
impl<B: AutodiffBackend> PpoValue<B, 2> for ValueMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs)
    }
}

/// Builds the base [`CartPole`] wrapped in a [`TimeLimit`]. Each example
/// then adds its own viz tap(s) on top of this.
///
/// CartPole can theoretically run forever if the agent is perfect, i.e.
/// it learns to oscillate back and forth keeping the pole upright, which
/// would stall training. So it's wrapped in `TimeLimit`, which truncates an
/// episode after `max_steps` or in this case `EPISODE_TIME_LIMIT = 500` steps.
/// This is standard RL practice: **truncation != failure**, it just means "stop
/// after maximum number of steps allowed."
///
/// The `seed = 42` makes the run reproducible.
///
/// todo! document the default values here for reference
pub fn base_env() -> TimeLimit<CartPole> {
    let base = CartPole::with_config(CartPoleConfig {
        seed: SEED,
        ..CartPoleConfig::default()
    })
    .expect("valid config");
    TimeLimit::new(base, EPISODE_TIME_LIMIT)
}

/// Builds a fresh PPO agent with the shared hyperparameters. `total_timesteps`
/// sizes the learning-rate / clip annealing schedule.
///
/// This wires teh policy + value + hyperparameters into a `PpoAgent`. The
/// hyperparameters are:
///
/// | Hyperparameter | Value | Meaning |
/// |---|---|---|
/// | `num_envs` | 1 | One parallel environment (no vectorization here) |
/// | `num_steps` | 128 | Collect 128 steps of experience before each update |
/// | `num_minibatches` | 4 | Split those 128 steps into 4 mini-batches of 32 |
/// | `update_epochs` | 4 | Reuse the collected data 4× per rollout (PPO's signature reuse) |
/// | `learning_rate` | 2.5e-4 | Optimizer step size |
/// | `clip_coef` | 0.2 | The PPO clip radius — the heart of the algorithm |
/// | `entropy_coef` | 0.01 | Bonus to keep the policy *exploring* (avoid collapse to one action) |
/// | `value_coef` | 0.5 | Weight of the value-loss term in the total loss |
/// | `gamma` | 0.99 | Discount factor — future rewards matter 99% as much as now |
/// | `gae_lambda` | 0.95 | Bias/variance knob for advantage estimation (GAE) |
///
/// `total_iterations = total_timesteps / batch_size` tells PPO how many updates
///  it'll do total, so it can **anneal** (gradually shrink) the learning rate
/// and clip over training. This is why `total_timesteps` is a parameter rather
/// than a constant.
///
/// todo! discuss the relationship between `total_timesteps` and
/// `TimeLimit` wrapper
pub fn build_agent(total_timesteps: usize) -> CartPoleAgent {
    let device = Default::default();

    let policy: CategoricalPolicyHead<Be> = CategoricalPolicyHeadConfig {
        obs_dim: OBS_RANK,
        hidden: HIDDEN,
        num_actions: NUM_ACTIONS,
    }
    .init::<Be>(&device);
    let value: ValueMlp<Be> = ValueMlp::new(OBS_RANK, HIDDEN, &device);

    let config = PpoTrainingConfigBuilder::new()
        .num_envs(1) // One parallel environment (no vectorization here)
        .num_steps(NUM_STEPS)
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

/// Trains `agent` against `env` for `total_timesteps`, hiding the long
/// `train_discrete` turbofish every call site would otherwise repeat.
///
/// `env` is generic over the viz-tier composition: a bare [`TimeLimit`],
/// a `TuiEnvTap`-wrapped env, a `RecordingTap`-wrapped env, or any nesting
/// of those — they all forward `CartPole`'s observation / action / reward
/// associated types. In other words, all those wrappers forward CartPole's
/// observation/action/reward types, so a single generic function handles them
/// all.
///
/// The const generics `<1, 1, 1>` and the turbofish in
/// `train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>` encode
/// type-level facts:
/// - Reward rank `1`, state rank `1`, action rank `1` (scalar rewards, 1D state tensors)
/// - `2` actions at the end (CartPole's push-left / push-right)
///
/// The whole `train` function exists mainly to **hide that gnarly turbofish** from each example call site — a small but real ergonomics win.
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
