//! Headless [`CartPole`] harness run that records to disk and then
//! emits a single self-contained `index.html` report.
//!
//! Demonstrates the static-HTML emitter pipeline end-to-end:
//!
//! 1. The harness wraps each `CartPole` in [`RecordingTap`] inside
//!    [`BenchAdapter`], pushing per-step frames into a shared
//!    [`RecordWriter`].
//! 2. The [`RecordingReporter`] (lifecycle-only mode) finalises the
//!    `run.toml` manifest at suite end.
//! 3. After the suite returns, the example opens the recording with
//!    [`RecordedRun::open`] and calls [`emit_static_html`] to write
//!    `runs/<run_id>/index.html`.
//!
//! # Run with
//!
//! ```bash
//! cargo run -p rlevo --example report_cartpole \
//!   --features viz-record,viz-report
//! ```
//!
//! The example prints the path to the generated `index.html` on
//! success. Open it in any browser to confirm the manifest header
//! renders and the episode table lists every captured episode.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use rand::Rng;
use rand_distr::{Distribution, Uniform};

use rlevo_benchmarks::agent::BenchableAgent;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_benchmarks::record::{
    EnvFamily, RecordSink, RecordWriter, RecordingConfig, RecordingReporter, RecordingTap,
};
use rlevo_benchmarks::report::{EmitConfig, RecordedRun, emit_static_html};
use rlevo_benchmarks::suite::Suite;

use rlevo_core::action::DiscreteAction;

use rlevo_environments::bench::BenchAdapter;
use rlevo_environments::classic::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};

const SEED: u64 = 2026;
const NUM_EPISODES: usize = 8;

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
        Suite::new("cartpole-report", cfg.clone()).with_env("cartpole", move |seed| {
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

    // Drop the writer so episode files + run.toml are fully flushed
    // before the loader opens them.
    drop(sink);

    let run = RecordedRun::open(&run_dir)?;
    for w in run.warnings() {
        eprintln!("warning: {w:?}");
    }
    let out = run_dir.join("index.html");
    let outcome = emit_static_html(&run, &out, &EmitConfig::default())?;
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
