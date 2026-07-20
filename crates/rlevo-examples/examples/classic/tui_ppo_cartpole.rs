//! Live (metrics-only) TUI dashboard wrapping a PPO training run on
//! [`CartPole`].
//!
//! Demonstrates the live product end-to-end on a non-harness training loop.
//! The live TUI is metrics-only (ADR-0013): it answers *"is it learning?"*
//! from learning curves and renders no environment — env playback lives in
//! the post-run report (see the `record_*`/`report_*` examples).
//!
//! 1. [`TuiRunner::start`] enters raw mode + alt screen and spawns the
//!    render thread.
//! 2. [`TuiCaptureLayer`] is installed as the global tracing subscriber;
//!    every `tracing::info!` PPO emits during training feeds the live
//!    dashboard's metric sparklines and scrolling log panel through the
//!    same channel.
//! 3. The env (`CartPole` → `TimeLimit` → [`TuiEnvTap`]) is wrapped in an
//!    episode-return emitter so the reward sparkline lights up from a raw
//!    `Environment` driver — no benchmarks-harness `Suite`/`Evaluator` flow
//!    required.
//! 4. PPO calls
//!    [`train_discrete`](rlevo_reinforcement_learning::algorithms::ppo::train::train_discrete)
//!    directly against the wrapped env (via the shared `ppo_cartpole::train`).
//! 5. [`TuiRunner::shutdown`] joins the render thread and restores the
//!    terminal.
//!
//! # Which panels light up
//!
//! - **Reward sparkline** — `EpisodeReturn` events pushed by
//!   [`TuiEnvTap`] on each episode termination (`CartPole` both
//!   `Terminated` from pole failure and `Truncated` from the
//!   `TimeLimit`). Because PPO *learns*, this sparkline climbs toward the
//!   500-step ceiling rather than hovering at the ~20-step random floor.
//! - **`policy_loss` / `entropy` / `approx_kl` panels** — PPO emits these
//!   as structured fields at every `log_every` interval; the
//!   canonical-metric registry in `rlevo-benchmarks::tui::log_layer`
//!   routes them into the dashboard. Here each gets its own bordered
//!   panel ([`MetricsLayout::Separate`]).
//! - **Log panel** — every PPO "training progress" line scrolls in,
//!   styled by level.
//!
//! The EA-only `best_fitness` signal is omitted from this run's metric
//! set, so no empty panel takes up space.
//!
//! # Run with
//!
//! ```bash
//! cargo run -p rlevo-examples --example tui_ppo_cartpole --features viz-tui --release
//! ```
//!
//! The `--release` matters: PPO's training loop is unusable at debug
//! speed. After training completes the dashboard stays open; press any
//! key to exit.
//!
//! [`TuiEnvTap`]: rlevo_benchmarks::env_wrappers::TuiEnvTap

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use burn::backend::{Autodiff, Flex};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rlevo_benchmarks::env_wrappers::TuiEnvTap;
use rlevo_benchmarks::tui::{MetricsLayout, TuiCaptureLayer, TuiConfig, TuiRunner};

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

// Sized so the dashboard runs for ~30 s of training on a release build,
// giving the user time to read every panel.
const TOTAL_TIMESTEPS: usize = 20_000;

// The three metrics PPO actually emits. Naming them explicitly drops the
// EA-only `best_fitness` panel that would otherwise sit permanently empty
// during an RL run.
const PPO_METRICS: &[&str] = &["policy_loss", "entropy", "approx_kl"];

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
/// ```text
/// pub struct PpoAgent<B, P, V, O, const DO: usize, const DB: usize>
/// where
///     B: AutodiffBackend,                             --> Be = Autodiff<Flex>
///     P: PpoPolicy<B, DB>,                            --> CategoricalPolicyHead<Be>
///     V: PpoValue<B, DB>,                             --> ValueMlp<Be>
///     O: Observation<DO> + TensorConvertible<DO, B>,  --> CartPoleObservation
/// { /* private fields */ }
/// ```
///
/// todo! further discuss `CategoricalPolicyHead<Be>` and `CartPoleObservation`
///
/// links:
/// - <https://docs.rs/rlevo-reinforcement-learning/0.3.0/rlevo_reinforcement_learning/algorithms/ppo/ppo_agent/struct.PpoAgent.html>
/// - <https://docs.rs/rlevo-environments/0.3.0/rlevo_environments/classic/cartpole/index.html>
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
/// actions, which is what you want for discrete actions like `CartPole`'s.
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
/// `CartPole` can theoretically run forever if the agent is perfect, i.e.
/// it learns to oscillate back and forth keeping the pole upright, which
/// would stall training. So it's wrapped in `TimeLimit`, which truncates an
/// episode after `max_steps` or in this case `EPISODE_TIME_LIMIT = 500` steps.
/// This is standard RL practice: **truncation != failure**, it just means "stop
/// after maximum number of steps allowed."
///
/// The `seed = 42` makes the run reproducible.
///
/// todo! document the default values here for reference
///
/// # Panics
///
/// Panics if the hard-coded [`CartPoleConfig`] is rejected as invalid.
#[must_use]
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
///
/// # Panics
///
/// Panics if the hard-coded PPO hyperparameters are rejected as invalid.
#[must_use]
pub fn build_agent(total_timesteps: usize) -> CartPoleAgent {
    let device = Default::default();

    let policy: CategoricalPolicyHead<Be> = CategoricalPolicyHeadConfig {
        obs_dim: OBS_RANK,
        hidden: HIDDEN,
        num_actions: NUM_ACTIONS,
    }
    .try_init::<Be>(&device)
    .expect("valid head config");
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
/// associated types. In other words, all those wrappers forward `CartPole`'s
/// observation/action/reward types, so a single generic function handles them
/// all.
///
/// The const generics `<1, 1, 1>` and the turbofish in
/// `train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>` encode
/// type-level facts:
/// - Reward rank `1`, state rank `1`, action rank `1` (scalar rewards, 1D state tensors)
/// - `2` actions at the end (`CartPole`'s push-left / push-right)
///
/// The whole `train` function exists mainly to **hide that gnarly turbofish** from each example call site — a small but real ergonomics win.
///
/// # Errors
///
/// Returns [`PpoAgentError`] if a rollout or policy update fails.
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. TUI owns the terminal from here until shutdown. Each metric gets
    //    its own bordered panel so the loss/entropy/KL trajectories are
    //    readable side-by-side rather than crammed into single rows.
    let cfg = TuiConfig {
        metrics: PPO_METRICS,
        metrics_layout: MetricsLayout::Separate,
        ..TuiConfig::default()
    };
    let runner = TuiRunner::start(cfg)?;
    let handle = runner.handle();

    // 2. Install the capture layer as the global tracing subscriber. No
    //    stdout fmt layer — alt-screen + stdout would corrupt the
    //    dashboard. All log content goes to the TUI log panel.
    tracing_subscriber::registry()
        .with(TuiCaptureLayer::new(handle.clone()))
        .try_init()?;

    // 3. Env: CartPole → TimeLimit → TuiEnvTap. The tap forwards episode
    //    returns to the dashboard through the same channel as the
    //    tracing-driven metric sparklines. Without it the reward sparkline
    //    stays empty.
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(base_env(), handle);
    let mut agent = build_agent(TOTAL_TIMESTEPS);

    // 4. Train. Every `LOG_EVERY` steps PPO emits a structured tracing
    //    event; the capture layer fans the loss/entropy/kl fields into
    //    the metric panels and the formatted message into the log panel.
    train(&mut agent, &mut env, &mut rng, TOTAL_TIMESTEPS)?;

    // 5. Hold the dashboard open until the user dismisses it — the final
    //    metric trajectories and log tail are the whole point.
    runner.wait_for_keypress()?;
    runner.shutdown()?;
    Ok(())
}
