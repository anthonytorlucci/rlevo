//! Harness-driven [`CartPole`] run that records per-episode files to
//! disk alongside the live TUI dashboard.
//!
//! Demonstrates the on-disk recording composition for benchmark flows:
//!
//! 1. A shared [`RecordWriter`] sits behind an
//!    `Arc<Mutex<dyn RecordSink>>` that is handed to both producers.
//! 2. The env factory wraps each [`CartPole`] in [`RecordingTap`]
//!    *inside* [`BenchAdapter`], then in [`RenderTap`] *outside*. The
//!    composition is `RenderTap(BenchAdapter(RecordingTap(env)))` so
//!    every step lights up the live env panel **and** writes a
//!    [`FrameRecord`] to the open `episode_<N>.rec` file.
//! 3. The harness reporter chain is
//!    [`MultiReporter`]`([TuiReporter, RecordingReporter::without_lifecycle])`.
//!    `TuiReporter` drives the live sparkline; `RecordingReporter::without_lifecycle`
//!    only finalises the `run.toml` manifest at suite end — the
//!    [`RecordingTap`] inside the env owns the per-episode start /
//!    frame / end signals.
//!
//! # Layout the run produces
//!
//! ```text
//! runs/<run_id>/
//!   ├── episode_000000.rec   ← N×FrameRecord + Metrics chunk
//!   ├── episode_000001.rec
//!   ├── ...
//!   └── run.toml             ← RunManifest (seed, env_family, …)
//! ```
//!
//! # Run with
//!
//! ```bash
//! cargo run -p rlevo --example record_cartpole \
//!   --features viz-tui,viz-record
//! ```
//!
//! [`MultiReporter`]: rlevo_benchmarks::reporter::MultiReporter

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use rand::Rng;
use rand_distr::{Distribution, Uniform};

use rlevo_benchmarks::agent::BenchableAgent;
use rlevo_benchmarks::env_wrappers::RenderTap;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_benchmarks::record::{
    EnvFamily, RecordSink, RecordWriter, RecordingConfig, RecordingReporter, RecordingTap,
};
use rlevo_benchmarks::reporter::MultiReporter;
use rlevo_benchmarks::suite::Suite;
use rlevo_benchmarks::tui::{TuiConfig, TuiRunner};

use rlevo_core::action::DiscreteAction;

use rlevo_environments::bench::BenchAdapter;
use rlevo_environments::classic::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};

const STEP_THROTTLE: Duration = Duration::from_millis(20);
const SEED: u64 = 2026;
const NUM_EPISODES: usize = 12;

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
    let runner = TuiRunner::start(TuiConfig::default().with_env_family(EnvFamily::Classic))?;
    let handle = runner.handle();

    let record_cfg = RecordingConfig::new(EnvFamily::Classic, SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    let cfg = EvaluatorConfig {
        num_episodes: NUM_EPISODES,
        num_trials_per_env: 1,
        max_steps: 500,
        base_seed: SEED,
        num_threads: Some(1),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: Some(195.0),
    };

    let suite = {
        let handle = handle.clone();
        let sink = sink.clone();
        Suite::new("cartpole-record", cfg.clone()).with_env("cartpole", move |seed| {
            let env = CartPole::with_config(CartPoleConfig {
                seed,
                ..CartPoleConfig::default()
            });
            // RecordingTap → BenchAdapter → RenderTap. RecordingTap owns the
            // FrameRecord stream; BenchAdapter exposes the BenchEnv surface
            // the harness drives; RenderTap forwards styled frames to the
            // live TUI panel.
            let recorded: RecordingTap<CartPole, 1, 1, 1> =
                RecordingTap::new(env, sink.clone());
            RenderTap::new(BenchAdapter::new(recorded), handle.clone())
        })
    };

    let mut reporter = MultiReporter::new(vec![
        Box::new(handle.as_reporter()),
        Box::new(RecordingReporter::without_lifecycle(
            sink.clone(),
            manifest,
        )),
    ]);
    let evaluator = Evaluator::new(cfg);
    let _report = evaluator.run_suite(&suite, |_| RandomCartPoleAgent::new(), &mut reporter);

    // Reporter is dropped here (end of scope before wait_for_keypress). The
    // RecordingReporter's drop fires no extra writes — on_run_end already
    // flushed the manifest atomically.
    drop(reporter);

    runner.wait_for_keypress()?;
    runner.shutdown()?;
    Ok(())
}
