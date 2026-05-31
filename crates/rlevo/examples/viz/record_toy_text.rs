//! Harness-driven [`FrozenLake`] (`toy_text` family) run that writes
//! per-episode `.rec` files alongside the live TUI dashboard.
//!
//! Mirrors `record_ppo_cartpole.rs` for the `toy_text` family — the on-disk
//! manifest carries `EnvFamily::ToyText`, and the resulting run replays
//! through the interactive `toy_text` playback adapter in the report client.
//!
//! # Run with
//!
//! ```bash
//! cargo run -p rlevo --example record_toy_text \
//!   --features viz-tui,viz-record
//! ```

use std::sync::Arc;
use std::thread;

use parking_lot::Mutex;
use std::time::Duration;

use rand::Rng;
use rand_distr::{Distribution, Uniform};

use rlevo_benchmarks::agent::BenchableAgent;
use rlevo_benchmarks::env_wrappers::RenderTap;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_benchmarks::record::{
    RecordSink, RecordWriter, RecordedEnvFamily, RecordingConfig, RecordingReporter, RecordingTap,
};
use rlevo_benchmarks::reporter::MultiReporter;
use rlevo_benchmarks::suite::Suite;
use rlevo_benchmarks::tui::{TuiConfig, TuiRunner};

use rlevo_core::action::DiscreteAction;

use rlevo_environments::bench::BenchAdapter;
use rlevo_environments::toy_text::frozen_lake::{
    FrozenLake, FrozenLakeAction, FrozenLakeConfig, FrozenLakeObservation, FrozenMapSpec,
    FrozenPreset,
};

const STEP_THROTTLE: Duration = Duration::from_millis(40);
const SEED: u64 = 2026;
const NUM_EPISODES: usize = 8;
const MAX_STEPS_PER_EPISODE: usize = 100;

struct RandomFrozenLakeAgent {
    dist: Uniform<usize>,
}

impl RandomFrozenLakeAgent {
    fn new() -> Self {
        Self {
            dist: Uniform::new(0, FrozenLakeAction::ACTION_COUNT).expect("non-empty action set"),
        }
    }
}

impl BenchableAgent<FrozenLakeObservation, FrozenLakeAction> for RandomFrozenLakeAgent {
    fn act(&mut self, _obs: &FrozenLakeObservation, rng: &mut dyn Rng) -> FrozenLakeAction {
        thread::sleep(STEP_THROTTLE);
        FrozenLakeAction::from_index(self.dist.sample(rng))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Family declared once by the env type (`FrozenLake: RecordedEnvFamily`).
    let runner = TuiRunner::start(TuiConfig::default().with_env_family(FrozenLake::FAMILY))?;
    let handle = runner.handle();

    let record_cfg = RecordingConfig::for_env::<FrozenLake>(SEED);
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
        Suite::new("frozen-lake-record", cfg.clone()).with_env("frozen_lake", move |seed| {
            let env = FrozenLake::with_config(FrozenLakeConfig {
                map: FrozenMapSpec::Preset(FrozenPreset::Four4x4),
                is_slippery: false,
                seed,
                ..FrozenLakeConfig::default()
            })
            .expect("FrozenLake construction with preset map cannot fail");
            let recorded: RecordingTap<FrozenLake, 1, 1, 1> =
                RecordingTap::new(env, sink.clone());
            RenderTap::new(BenchAdapter::new(recorded), handle.clone())
        })
    };

    let mut reporter = MultiReporter::new(vec![
        Box::new(handle.as_reporter()),
        Box::new(RecordingReporter::without_lifecycle(sink.clone(), manifest)),
    ]);
    let evaluator = Evaluator::new(cfg);
    let _report = evaluator.run_suite(&suite, |_| RandomFrozenLakeAgent::new(), &mut reporter);

    drop(reporter);

    // Fail loud if recording hit a write error. `TuiRunner::drop` restores
    // the terminal on the error path.
    if let Some(e) = sink.lock().take_error() {
        return Err(e.into());
    }

    runner.wait_for_keypress()?;
    runner.shutdown()?;
    Ok(())
}
