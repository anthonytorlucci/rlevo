//! Headless [`EmptyEnv`] (grids family) harness recording → static-HTML
//! report that mounts the Leptos/WASM client. Mirrors
//! `report_ppo_cartpole_with_client.rs` for the grids family — the emitted
//! HTML's manifest carries `EnvFamily::Grids`, which the client uses to
//! dispatch the interactive grids playback adapter (scrubber +
//! play/pause + styled-frame rendering).
//!
//! A random agent plays [`NUM_EPISODES`] episodes on a [`GRID_SIZE`]×[`GRID_SIZE`]
//! [`EmptyEnv`]. Each step is recorded as a frame via [`RecordingTap`];
//! the completed run is serialised to a per-run directory under the workspace
//! `target/` tree. `emit_static_html` then writes a single `index.html` into
//! that same directory — open it in a browser to replay the episodes.
//!
//! # Run with
//!
//! ```bash
//! # 1) Build the client (one-time per code change).
//! cd crates/rlevo-benchmarks-report-client && trunk build --release
//! cd ../../
//!
//! # 2) Run this example.
//! cargo run -p rlevo-examples --example report_grids_with_client \
//!     --features viz-report
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use rand::Rng;
use rand_distr::{Distribution, Uniform};

use rlevo_benchmarks::agent::BenchableAgent;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_benchmarks::record::{
    RecordSink, RecordWriter, RecordingConfig, RecordingReporter, RecordingTap,
};
use rlevo_benchmarks::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};
use rlevo_benchmarks::suite::Suite;

use rlevo_core::action::DiscreteAction;

use rlevo_environments::bench::BenchAdapter;
use rlevo_environments::grids::core::observation::GridObservation;
use rlevo_environments::grids::{EmptyConfig, EmptyEnv, GridAction};

const SEED: u64 = 2026;
const NUM_EPISODES: usize = 5;
const GRID_SIZE: usize = 6;
const MAX_STEPS_PER_EPISODE: usize = 200;

const CLIENT_DIST: &str = "crates/rlevo-benchmarks-report-client/dist";

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
        GridAction::from_index(self.dist.sample(rng))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::for_env::<EmptyEnv>(SEED);
    let writer = RecordWriter::open_default(record_cfg)?;
    let run_dir: PathBuf = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    let cfg = EvaluatorConfig {
        num_episodes: NUM_EPISODES,
        num_trials_per_env: 1,
        max_steps: MAX_STEPS_PER_EPISODE,
        base_seed: SEED,
        // Recording is single-stream: a `RecordWriter` holds one open
        // episode file, so the harness must run on a single thread. A value
        // >1 trips `RecordError::ConcurrentUse` (surfaced post-run via
        // `take_error`) rather than corrupting the in-flight episode.
        num_threads: Some(1),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: None,
    };

    let suite = {
        let sink = sink.clone();
        Suite::new("empty-grid-report-client", cfg.clone()).with_env("empty", move |seed| {
            let env = EmptyEnv::with_config(
                EmptyConfig::new(GRID_SIZE, MAX_STEPS_PER_EPISODE, seed),
                false,
            ).expect("valid config");
            let recorded: RecordingTap<EmptyEnv, 3, 3, 1> = RecordingTap::new(env, sink.clone());
            BenchAdapter::new(recorded)
        })
    };

    // v6 manifest provenance: a random grid agent, plus build/platform
    // reproducibility metadata stamped through the reporter's builders.
    let mut reporter = RecordingReporter::without_lifecycle(sink.clone(), manifest)
        .with_algorithm("random")
        .with_num_seeds(1)
        .with_build_provenance();
    let evaluator = Evaluator::new(cfg);
    let _report = evaluator.run_suite(&suite, |_| RandomGridAgent::new(), &mut reporter);
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
