//! Harness-driven [`EmptyEnv`] (grids family) run that writes M4
//! per-episode `.rec` files alongside the live TUI dashboard.
//!
//! Mirrors `record_cartpole.rs` for the grids family — the on-disk
//! manifest carries `EnvFamily::Grids`, and the resulting run replays
//! through the M6 grids playback adapter.
//!
//! # Run with
//!
//! ```bash
//! cargo run -p rlevo --example record_grids \
//!   --features viz-tui,viz-record
//! ```
//!
//! Then ship the run through the M6 emitter via `report_grids_with_client`.

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
use rlevo_environments::grids::{EmptyConfig, EmptyEnv, GridAction};
use rlevo_environments::grids::core::observation::GridObservation;

const STEP_THROTTLE: Duration = Duration::from_millis(20);
const SEED: u64 = 2026;
const NUM_EPISODES: usize = 5;
const GRID_SIZE: usize = 6;
const MAX_STEPS_PER_EPISODE: usize = 200;

struct RandomGridAgent {
    dist: Uniform<usize>,
}

impl RandomGridAgent {
    fn new() -> Self {
        Self {
            dist: Uniform::new(0, GridAction::ACTION_COUNT).expect("non-empty action set"),
        }
    }
}

impl BenchableAgent<GridObservation, GridAction> for RandomGridAgent {
    fn act(&mut self, _obs: &GridObservation, rng: &mut dyn Rng) -> GridAction {
        thread::sleep(STEP_THROTTLE);
        GridAction::from_index(self.dist.sample(rng))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runner = TuiRunner::start(TuiConfig::default().with_env_family(EnvFamily::Grids))?;
    let handle = runner.handle();

    let record_cfg = RecordingConfig::new(EnvFamily::Grids, SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    let cfg = EvaluatorConfig {
        num_episodes: NUM_EPISODES,
        num_trials_per_env: 1,
        max_steps: MAX_STEPS_PER_EPISODE,
        base_seed: SEED,
        num_threads: Some(1),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: None,
    };

    let suite = {
        let handle = handle.clone();
        let sink = sink.clone();
        Suite::new("empty-grid-record", cfg.clone()).with_env("empty", move |seed| {
            let env = EmptyEnv::with_config(
                EmptyConfig::new(GRID_SIZE, MAX_STEPS_PER_EPISODE, seed),
                false,
            );
            let recorded: RecordingTap<EmptyEnv, 3, 3, 1> = RecordingTap::new(env, sink.clone());
            RenderTap::new(BenchAdapter::new(recorded), handle.clone())
        })
    };

    let mut reporter = MultiReporter::new(vec![
        Box::new(handle.as_reporter()),
        Box::new(RecordingReporter::without_lifecycle(sink.clone(), manifest)),
    ]);
    let evaluator = Evaluator::new(cfg);
    let _report = evaluator.run_suite(&suite, |_| RandomGridAgent::new(), &mut reporter);

    drop(reporter);

    runner.wait_for_keypress()?;
    runner.shutdown()?;
    Ok(())
}
