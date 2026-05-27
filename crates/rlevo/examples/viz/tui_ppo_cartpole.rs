//! Live TUI dashboard wrapping a PPO training run on [`CartPole`].
//!
//! Demonstrates Milestone-3's metric and log panels end-to-end:
//!
//! 1. [`TuiRunner::start`] enters raw mode + alt screen and spawns the
//!    render thread.
//! 2. [`TuiCaptureLayer`] is installed as the global tracing subscriber;
//!    every `tracing::info!` PPO emits during training feeds the live
//!    dashboard's metric sparklines and scrolling log panel through the
//!    same channel.
//! 3. PPO calls
//!    [`train_discrete`](rlevo_reinforcement_learning::algorithms::ppo::train::train_discrete)
//!    directly — the full `Environment` trait path, not the
//!    benchmarks-harness `Suite`/`Evaluator` flow.
//! 4. [`TuiRunner::shutdown`] joins the render thread and restores the
//!    terminal.
//!
//! # Which panels light up
//!
//! - **`policy_loss` / `entropy` / `approx_kl` sparklines** — PPO emits
//!   these as structured fields at every `log_every` interval; the
//!   canonical-metric registry in `rlevo-benchmarks::tui::log_layer`
//!   routes them into the dashboard.
//! - **Log panel** — every PPO "training progress" line scrolls in,
//!   styled by level.
//! - **Reward sparkline** — *empty.* M2's reward panel reads from
//!   `EpisodeEnd` events on the [`Reporter`] surface; this example
//!   bypasses the benchmarks harness, so no `EpisodeEnd` events flow.
//!   PPO's `avg_reward` field is a string (Display-formatted), so the
//!   canonical-metric registry intentionally ignores it.
//! - **Env panel** — *empty (shows "waiting for first frame…").*
//!   `RenderTap` wraps a [`BenchEnv`]; PPO drives the full `Environment`
//!   trait directly, with no benchmarks layer to tap into. Wrapping the
//!   env in a frame-emitter is its own work, deferred.
//! - **`best_fitness` sparkline** — *empty.* This is an EA-only signal.
//!
//! # Run with
//!
//! ```bash
//! cargo run -p rlevo --example tui_ppo_cartpole --features viz-tui --release
//! ```
//!
//! The `--release` matters: PPO's training loop is unusable at debug
//! speed.
//!
//! [`Reporter`]: rlevo_benchmarks::reporter::Reporter
//! [`BenchEnv`]: rlevo_core::evaluation::BenchEnv

use burn::backend::{Autodiff, Flex};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use rlevo_benchmarks::tui::{TuiCaptureLayer, TuiConfig, TuiRunner};

use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;
use rlevo_reinforcement_learning::algorithms::ppo::train::train_discrete;

// Knobs intentionally hardcoded for the smoke run — no CLI noise.
// `TOTAL_TIMESTEPS` is sized so the dashboard runs for ~30 s of training
// on a release build, giving the user time to read every panel.
const SEED: u64 = 42;
const TOTAL_TIMESTEPS: usize = 20_000;
const NUM_STEPS: usize = 128;
const LOG_EVERY: usize = 1_024;
const EPISODE_TIME_LIMIT: usize = 500;
const HIDDEN: usize = 64;
const OBS_DIM: usize = 4;
const NUM_ACTIONS: usize = 2;

// Value network — two-hidden-layer MLP, matching the existing
// `ppo_cart_pole` example so the comparison stays apples-to-apples.

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. TUI owns the terminal from here until shutdown.
    let runner = TuiRunner::start(TuiConfig::default())?;
    let handle = runner.handle();

    // 2. Install the capture layer as the global tracing subscriber. No
    //    stdout fmt layer — alt-screen + stdout would corrupt the
    //    dashboard. All log content goes to the TUI log panel.
    tracing_subscriber::registry()
        .with(TuiCaptureLayer::new(handle))
        .try_init()?;

    // 3. PPO setup — same shape as the non-TUI `ppo_cart_pole` example.
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);

    let base_env = CartPole::with_config(CartPoleConfig {
        seed: SEED,
        ..CartPoleConfig::default()
    });
    let mut env = TimeLimit::new(base_env, EPISODE_TIME_LIMIT);

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

    let total_iterations = TOTAL_TIMESTEPS / config.batch_size().max(1);

    let mut agent: PpoAgent<
        Be,
        CategoricalPolicyHead<Be>,
        ValueMlp<Be>,
        CartPoleObservation,
        1,
        2,
    > = PpoAgent::new(policy, value, config, device, total_iterations);

    // 4. Train. Every `LOG_EVERY` steps PPO emits a structured tracing
    //    event; the capture layer fans the loss/entropy/kl fields into
    //    the metric panels and the formatted message into the log panel.
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        TOTAL_TIMESTEPS,
        LOG_EVERY,
    )?;

    // 5. Hold the dashboard open until the user dismisses it — the
    //    final metric trajectories and log tail are the whole point.
    //    The status line shows the dismissal hint.
    runner.wait_for_keypress()?;

    runner.shutdown()?;
    Ok(())
}
