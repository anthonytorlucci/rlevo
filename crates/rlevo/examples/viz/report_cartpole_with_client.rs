//! Headless [`CartPole`] harness recording → static-HTML report that
//! mounts the M5.1 Leptos/WASM client (Milestone 5.1).
//!
//! Two-step build flow:
//!
//! ```bash
//! # 1) Build the WASM client (one-time per code change).
//! cd crates/rlevo-benchmarks-report-client
//! trunk build --release
//!
//! # 2) Run this example. It records a fresh CartPole run, opens it
//! #    via RecordedRun::open, loads the trunk artefacts, and writes
//! #    a single-file index.html that mounts the client.
//! cd ../../  # back to repo root
//! cargo run -p rlevo --example report_cartpole_with_client \
//!     --features viz-record,viz-report
//! ```
//!
//! The emitted HTML is fully offline: no network requests, no external
//! assets. Open it in any modern browser; the client decodes the
//! inlined payloads and renders the manifest + episode table.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
use rlevo_environments::classic::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};

const SEED: u64 = 2026;
const NUM_EPISODES: usize = 8;

const CLIENT_DIST: &str = "crates/rlevo-benchmarks-report-client/dist";

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
        let idx = self.dist.sample(rng);
        CartPoleAction::from_index(idx)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::new(EnvFamily::Classic, SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
    let run_dir: PathBuf = writer.run_dir().to_path_buf();
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
        let sink = sink.clone();
        Suite::new("cartpole-report-client", cfg.clone()).with_env("cartpole", move |seed| {
            let env = CartPole::with_config(CartPoleConfig {
                seed,
                ..CartPoleConfig::default()
            });
            let recorded: RecordingTap<CartPole, 1, 1, 1> =
                RecordingTap::new(env, sink.clone());
            BenchAdapter::new(recorded)
        })
    };

    let mut reporter = RecordingReporter::without_lifecycle(sink.clone(), manifest);
    let evaluator = Evaluator::new(cfg);
    let _report = evaluator.run_suite(&suite, |_| RandomCartPoleAgent::new(), &mut reporter);
    drop(reporter);
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
