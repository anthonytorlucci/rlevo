//! Live (metrics-only) TUI dashboard wrapping a PPO training run on
//! [`CartPole`].
//!
//! Demonstrates the live product end-to-end on a non-harness training loop.
//! The live TUI is metrics-only (ADR-0013): it answers *"is it learning?"*
//! from learning curves and renders no environment — env playback lives in
//! the post-run report (see the `report_*` examples).
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
//! 4. The harness's `train` drives the agent against the wrapped env, and
//!    PPO logs each update.
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
//!   as structured fields after every rollout's update; the
//!   canonical-metric registry in `rlevo-harness::tui::log_layer`
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
//! [`TuiEnvTap`]: rlevo_harness::env_wrappers::TuiEnvTap

use rand::SeedableRng;
use rand::rngs::StdRng;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use burn::tensor::Device;

use rlevo_harness::env_wrappers::TuiEnvTap;
use rlevo_harness::evaluator::train;
use rlevo_harness::tui::{MetricsLayout, TuiCaptureLayer, TuiConfig, TuiRunner};

use rlevo_environments::classic::cartpole::{CartPole, CartPoleConfig, CartPoleObservation};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfig;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::{MlpValue, MlpValueConfig};

// Sized so the dashboard runs for ~30 s of training on a release build,
// giving the user time to read every panel.
const TOTAL_TIMESTEPS: usize = 20_000;

// The three metrics PPO actually emits. Naming them explicitly drops the
// EA-only `best_fitness` panel that would otherwise sit permanently empty
// during an RL run.
const PPO_METRICS: &[&str] = &["policy_loss", "entropy", "approx_kl"];

/// Deterministic seed shared by the env and the RNG.
pub const SEED: u64 = 42;
/// Rollout horizon per PPO update.
pub const NUM_STEPS: usize = 128;
/// Episodes are truncated at this length via [`TimeLimit`].
pub const EPISODE_TIME_LIMIT: usize = 500;

const HIDDEN: usize = 64;
const OBS_RANK: usize = 4; // todo! this should be dim. rank = 1
const NUM_ACTIONS: usize = 2;

/// Concrete PPO agent type the examples drive — a categorical policy head
/// over the crate's default value network, `MlpValue`, both with two
/// `hidden`-wide `tanh` layers. Neither is defined here: PPO ships both.
///
/// ```text
/// pub struct PpoAgent<P, V, O, const OR: usize, const BOR: usize>
/// where
///     P: PpoPolicy<BOR>,                           --> CategoricalPolicyHead
///     V: PpoValue<BOR>,                            --> MlpValue
///     O: Observation<OR> + TensorConvertible<OR>,  --> CartPoleObservation
/// { /* private fields */ }
/// ```
///
/// todo! further discuss `CategoricalPolicyHead` and `CartPoleObservation`
///
/// links:
/// - <https://docs.rs/rlevo-reinforcement-learning/latest/rlevo_reinforcement_learning/algorithms/ppo/ppo_agent/struct.PpoAgent.html>
/// - <https://docs.rs/rlevo-environments/latest/rlevo_environments/classic/cartpole/index.html>
pub type CartPoleAgent = PpoAgent<CategoricalPolicyHead, MlpValue, CartPoleObservation, 1, 2>;

/// Builds the base [`CartPole`] wrapped in a [`TimeLimit`]; `main` adds the
/// `TuiEnvTap` on top of this.
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

/// Builds a fresh PPO agent with this example's hyperparameters.
/// `total_timesteps` sizes the learning-rate / clip annealing schedule.
///
/// This wires teh policy + value + hyperparameters into a `PpoAgent`. The
/// hyperparameters are:
///
/// | Hyperparameter | Value | Meaning |
/// |---|---|---|
/// | `num_envs` | 1 | One parallel environment (no vectorization here) |
/// | `num_steps` | 128 | Collect 128 steps of experience before each update |
/// | `num_minibatches` | 4 | Split those 128 steps into 4 mini-batches of 32 |
/// | `update_epochs` | 4 | Reuse the collected data `$4\times$` per rollout (PPO's signature reuse) |
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
    let device = Device::flex();

    let policy: CategoricalPolicyHead = CategoricalPolicyHeadConfig {
        obs_dim: OBS_RANK,
        hidden: HIDDEN,
        num_actions: NUM_ACTIONS,
    }
    .try_init(&device)
    .expect("valid head config");
    let value = MlpValueConfig {
        obs_dim: OBS_RANK,
        hidden: vec![HIDDEN, HIDDEN],
    }
    .try_init(&device)
    .expect("valid value network config");

    let config = PpoTrainingConfig {
        num_envs: 1,
        // One parallel environment (no vectorization here)
        num_steps: NUM_STEPS,
        num_minibatches: 4,
        update_epochs: 4,
        learning_rate: 2.5e-4,
        clip_coef: 0.2,
        entropy_coef: 0.01,
        value_coef: 0.5,
        gamma: 0.99,
        gae_lambda: 0.95,
        ..Default::default()
    };

    let total_iterations = total_timesteps / config.batch_size().max(1);

    PpoAgent::new(policy, value, config, device, total_iterations).expect("valid config")
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
    let mut env: TuiEnvTap<_, 1, 1> = TuiEnvTap::new(base_env(), handle);
    let mut agent = build_agent(TOTAL_TIMESTEPS);

    // 4. Train. After every rollout PPO emits a structured tracing
    //    event; the capture layer fans the loss/entropy/kl fields into
    //    the metric panels and the formatted message into the log panel.
    train(&mut env, &mut agent, TOTAL_TIMESTEPS, usize::MAX, &mut rng)?;

    // 5. Hold the dashboard open until the user dismisses it — the final
    //    metric trajectories and log tail are the whole point.
    runner.wait_for_keypress()?;
    runner.shutdown()?;
    Ok(())
}
