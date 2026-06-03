//! Live TUI dashboard wrapping a PPO training run on [`CartPole`].
//!
//! Demonstrates the full live tier end-to-end on a non-harness training
//! loop:
//!
//! 1. [`TuiRunner::start`] enters raw mode + alt screen and spawns the
//!    render thread.
//! 2. [`TuiCaptureLayer`] is installed as the global tracing subscriber;
//!    every `tracing::info!` PPO emits during training feeds the live
//!    dashboard's metric sparklines and scrolling log panel through the
//!    same channel.
//! 3. The env (`CartPole` → `TimeLimit` → [`TuiEnvTap`]) is wrapped in a
//!    frame + episode-return emitter so the env panel and reward
//!    sparkline light up from a raw `Environment` driver — no
//!    benchmarks-harness `Suite`/`Evaluator` flow required.
//! 4. PPO calls
//!    [`train_discrete`](rlevo_reinforcement_learning::algorithms::ppo::train::train_discrete)
//!    directly against the wrapped env (via the shared `ppo_cartpole::train`).
//! 5. [`TuiRunner::shutdown`] joins the render thread and restores the
//!    terminal.
//!
//! # Which panels light up
//!
//! - **Env panel** — per-step `StyledFrame`s pushed by [`TuiEnvTap`],
//!   drawing both the cart on its track and the balancing pole. The
//!   pole's tilt (and its colour, green→red) tracks how close the angle
//!   is to the failure threshold, so the panel reads as live motion.
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
//! speed.
//!
//! [`TuiEnvTap`]: rlevo_benchmarks::env_wrappers::TuiEnvTap

#[path = "../common/ppo_cartpole.rs"]
mod ppo_cartpole;

use rand::SeedableRng;
use rand::rngs::StdRng;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use rlevo_benchmarks::env_wrappers::TuiEnvTap;
use rlevo_benchmarks::tui::{MetricsLayout, TuiCaptureLayer, TuiConfig, TuiRunner};

use ppo_cartpole::{SEED, base_env, build_agent, train};

// Sized so the dashboard runs for ~30 s of training on a release build,
// giving the user time to read every panel.
const TOTAL_TIMESTEPS: usize = 20_000;

// The three metrics PPO actually emits. Naming them explicitly drops the
// EA-only `best_fitness` panel that would otherwise sit permanently empty
// during an RL run.
const PPO_METRICS: &[&str] = &["policy_loss", "entropy", "approx_kl"];

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

    // 3. Env: CartPole → TimeLimit → TuiEnvTap. The tap forwards per-step
    //    frames and episode returns to the dashboard through the same
    //    channel as the tracing-driven metric sparklines. Without it the
    //    env panel + reward sparkline stay empty.
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
