//! Live `ratatui` TUI dashboard wrapping a [`CartPole`] benchmark run.
//!
//! Demonstrates the full Milestone-2 wiring:
//!
//! 1. [`TuiRunner::start`] enters raw mode + alt screen and spawns the
//!    render thread.
//! 2. The env factory wraps each [`CartPole`] in
//!    [`RenderTap`](rlevo_benchmarks::env_wrappers::RenderTap) so per-step
//!    [`StyledFrame`](rlevo_core::render::StyledFrame)s land in the TUI's
//!    env panel.
//! 3. [`TuiHandle::as_reporter`] feeds the same channel as the rollout's
//!    [`Reporter`] callbacks — episode returns drive the reward sparkline,
//!    suite metadata fills the status line.
//! 4. [`TuiRunner::shutdown`] joins the render thread and restores the
//!    terminal.
//!
//! The agent is uniformly random; the dashboard's purpose is to verify
//! the live tier works end-to-end. The per-step throttle below is a
//! visualisation aid: without it [`CartPole`] episodes complete in
//! milliseconds and the user sees nothing on the env panel.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo --example tui_cartpole --features viz-tui
//! ```
//!
//! Ctrl-C exits cleanly — the panic hook installed by `ratatui::try_init`
//! restores the terminal regardless of how the process ends.

use std::thread;
use std::time::Duration;

use rand::Rng;
use rand_distr::{Distribution, Uniform};

use rlevo_benchmarks::agent::BenchableAgent;
use rlevo_benchmarks::env_wrappers::RenderTap;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_benchmarks::suite::Suite;
use rlevo_benchmarks::tui::{TuiConfig, TuiRunner};

use rlevo_core::action::DiscreteAction;

use rlevo_environments::bench::BenchAdapter;
use rlevo_environments::classic::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};

/// Wall-clock delay between agent decisions. Tuned so an episode takes
/// long enough to read on screen — at zero throttle the rollout outruns
/// the 60 ms TUI tick and the env panel reads as a blur.
const STEP_THROTTLE: Duration = Duration::from_millis(20);

/// Uniformly random left/right policy. The TUI smoke run doesn't need a
/// learning agent — the dashboard reads the same regardless of policy.
struct RandomCartPoleAgent {
    dist: Uniform<usize>,
}

impl RandomCartPoleAgent {
    fn new() -> Self {
        Self {
            dist: Uniform::new(0, CartPoleAction::ACTION_COUNT).expect("non-empty action set"),
        }
    }
}

impl BenchableAgent<CartPoleObservation, CartPoleAction> for RandomCartPoleAgent {
    fn act(&mut self, _obs: &CartPoleObservation, rng: &mut dyn Rng) -> CartPoleAction {
        thread::sleep(STEP_THROTTLE);
        let idx = self.dist.sample(rng);
        CartPoleAction::from_index(idx)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Live dashboard owns the terminal from here until shutdown.
    let runner = TuiRunner::start(TuiConfig::default())?;
    let handle = runner.handle();

    // 50 episodes × ~20 steps/episode × 20 ms throttle ≈ 20 s of dashboard.
    let cfg = EvaluatorConfig {
        num_episodes: 50,
        num_trials_per_env: 1,
        max_steps: 500,
        base_seed: 2026,
        num_threads: Some(1),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: Some(195.0),
    };

    let suite = {
        let handle = handle.clone();
        Suite::new("cartpole-tui", cfg.clone()).with_env("cartpole", move |seed| {
            let env = CartPole::with_config(CartPoleConfig {
                seed,
                ..CartPoleConfig::default()
            });
            RenderTap::new(BenchAdapter::new(env), handle.clone())
        })
    };

    let mut reporter = handle.as_reporter();
    let evaluator = Evaluator::new(cfg);
    let _report = evaluator.run_suite(&suite, |_| RandomCartPoleAgent::new(), &mut reporter);

    // Hold the dashboard open until the user dismisses it — the final
    // reward sparkline, episode count, and "finished" status are
    // worth studying before the terminal restores. The status line
    // displays the dismissal hint.
    runner.wait_for_keypress()?;
    runner.shutdown()?;
    Ok(())
}
