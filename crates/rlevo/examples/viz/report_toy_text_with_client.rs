//! Headless [`FrozenLake`] (toy_text family) harness recording →
//! static-HTML report that mounts the Leptos/WASM client. Mirrors
//! `report_cartpole_with_client.rs` for the toy_text family — the
//! manifest carries `EnvFamily::ToyText`, dispatching to the interactive
//! toy_text playback adapter.
//!
//! # Run with
//!
//! ```bash
//! cd crates/rlevo-benchmarks-report-client && trunk build --release
//! cd ../../
//! cargo run -p rlevo --example report_toy_text_with_client \
//!     --features viz-record,viz-report
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use rand::Rng;
use rand_distr::{Distribution, Uniform};

use rlevo_benchmarks::agent::BenchableAgent;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_benchmarks::record::{
    EnvFamily, RecordSink, RecordWriter, RecordingConfig, RecordingReporter, RecordingTap,
};
use rlevo_benchmarks::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};
use rlevo_benchmarks::suite::Suite;

use rlevo_core::action::DiscreteAction;

use rlevo_environments::bench::BenchAdapter;
use rlevo_environments::toy_text::frozen_lake::{
    FrozenLake, FrozenLakeAction, FrozenLakeConfig, FrozenLakeObservation, FrozenMapSpec,
    FrozenPreset,
};

const SEED: u64 = 2026;
const NUM_EPISODES: usize = 8;
const MAX_STEPS_PER_EPISODE: usize = 100;

const CLIENT_DIST: &str = "crates/rlevo-benchmarks-report-client/dist";

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
        FrozenLakeAction::from_index(self.dist.sample(rng))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::new(EnvFamily::ToyText, SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
    let run_dir: PathBuf = writer.run_dir().to_path_buf();
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
        let sink = sink.clone();
        Suite::new("frozen-lake-report-client", cfg.clone()).with_env(
            "frozen_lake",
            move |seed| {
                let env = FrozenLake::with_config(FrozenLakeConfig {
                    map: FrozenMapSpec::Preset(FrozenPreset::Four4x4),
                    is_slippery: false,
                    seed,
                    ..FrozenLakeConfig::default()
                })
                .expect("FrozenLake construction with preset map cannot fail");
                let recorded: RecordingTap<FrozenLake, 1, 1, 1> =
                    RecordingTap::new(env, sink.clone());
                BenchAdapter::new(recorded)
            },
        )
    };

    let mut reporter = RecordingReporter::without_lifecycle(sink.clone(), manifest);
    let evaluator = Evaluator::new(cfg);
    let _report = evaluator.run_suite(&suite, |_| RandomFrozenLakeAgent::new(), &mut reporter);
    drop(reporter);

    // Fail loud on a recording write error before building the report.
    if let Some(e) = sink.lock().take_error() {
        return Err(e.into());
    }

    drop(sink);

    let run = RecordedRun::open(&run_dir)?;
    for w in run.warnings() {
        eprintln!("warning: {w:?}");
    }

    let dist = PathBuf::from(CLIENT_DIST);
    let assets = ClientAssets::from_trunk_dist(&dist).map_err(|e| {
        format!(
            "could not load client assets from {}: {e}\n\
             Did you run `trunk build --release` in {} first?",
            dist.display(),
            "crates/rlevo-benchmarks-report-client"
        )
    })?;

    let out = run_dir.join("index.html");
    let outcome = emit_static_html(
        &run,
        &out,
        &EmitConfig {
            client_assets: Some(assets),
            ..EmitConfig::default()
        },
    )?;
    println!(
        "wrote {} ({} episodes, {} bytes{})",
        out.display(),
        outcome.episode_count,
        outcome.bytes_written,
        if outcome.size_warning {
            " — over size budget"
        } else {
            ""
        }
    );
    Ok(())
}
