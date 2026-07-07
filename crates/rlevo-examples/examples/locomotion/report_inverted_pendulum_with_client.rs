//! [`InvertedPendulumRapier`] recording → static-HTML report that mounts the
//! Leptos/WASM client with the **locomotion SVG adapter** (sagittal-plane
//! stick figure).
//!
//! Locomotion is the canonical use case for the rich SVG payload:
//! locomotion envs have no `AsciiRenderable` impl, so the sagittal-plane
//! SVG is the only rendering of the env anywhere in the stack.
//!
//! The policy is purely random (uniform force in `[-2, 2]`); no learner is
//! attached.  The run records [`NUM_EPISODES`] episodes capped at
//! [`MAX_STEPS_PER_EPISODE`] steps each, writes the `EpisodeRecord` files to
//! a timestamped run directory, then emits a single `index.html` alongside
//! them.  Opening that file shows the run manifest, an episode table, and the
//! sagittal-plane SVG replaying each episode.
//!
//! Two-step build flow:
//!
//! ```bash
//! # 1) Build the WASM client (one-time per code change).
//! cd crates/rlevo-benchmarks-report-client
//! trunk build --release
//!
//! # 2) Run this example from the repo root.
//! cd ../../
//! cargo run -p rlevo-examples --example report_inverted_pendulum_with_client \
//!     --features locomotion,viz-report
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use rand::{RngExt, SeedableRng, rngs::StdRng};

use rlevo_benchmarks::record::{RecordSink, RecordWriter, RecordingConfig, RecordingTap};
use rlevo_benchmarks::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::locomotion::inverted_pendulum::{
    InvertedPendulumAction, InvertedPendulumConfig, InvertedPendulumRapier,
};

const SEED: u64 = 2026;
const NUM_EPISODES: usize = 4;
const MAX_STEPS_PER_EPISODE: usize = 200;
const CLIENT_DIST: &str = "crates/rlevo-benchmarks-report-client/dist";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::for_env::<InvertedPendulumRapier>(SEED);
    let writer = RecordWriter::open_default(record_cfg)?;
    let run_dir: PathBuf = writer.run_dir().to_path_buf();
    // v6 manifest provenance: a random forcing policy (no learner), plus
    // build/platform reproducibility metadata.
    let manifest = writer
        .manifest_template()
        .with_algorithm("random")
        .with_num_seeds(1)
        .with_build_provenance();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    let env = InvertedPendulumRapier::with_config(InvertedPendulumConfig {
        seed: SEED,
        ..InvertedPendulumConfig::default()
    })
    .expect("valid config");
    let mut tap: RecordingTap<_, 1, 1, 1> =
        RecordingTap::with_locomotion_payload(env, sink.clone());

    for ep in 0..NUM_EPISODES {
        let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(ep as u64));
        tap.reset()?;
        for _ in 0..MAX_STEPS_PER_EPISODE {
            let force: f32 = rng.random_range(-2.0..2.0);
            let snap = tap.step(InvertedPendulumAction::new(force))?;
            if snap.is_done() {
                break;
            }
        }
    }

    // Finalise the run manifest before we open the directory for emit.
    sink.lock().on_run_end(manifest);

    // Fail loud on a recording write error before building the report.
    if let Some(e) = sink.lock().take_error() {
        return Err(e.into());
    }

    drop(tap);
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
