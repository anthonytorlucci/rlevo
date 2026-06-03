//! Headless PPO training on [`CartPole`] → static-HTML report that
//! mounts the Leptos/WASM report client with **RL convergence plots**.
//!
//! This is the canonical "does the viz mean anything?" example: because
//! PPO actually learns, the on-disk record carries a real RL metric
//! stream — `policy_loss` / `value_loss` / `entropy` / `approx_kl` /
//! `clip_frac` — and the convergence panel surfaces them as line charts
//! alongside per-episode reward / length curves that *climb* toward the
//! optimal 500-step policy rather than flatlining at the random floor.
//!
//! Composition map:
//!
//! ```text
//!   CartPole
//!     └─ TimeLimit
//!         └─ RecordingTap   (frame + metric stream to disk)
//!             ↓
//!          train_discrete
//!             ↑ tracing::info!(policy_loss = …, value_loss = …, …)
//!             └─ RecordingLayer (captures canonical RL metrics)
//! ```
//!
//! Two-step build flow:
//!
//! ```bash
//! # 1) Build the WASM client (one-time per code change).
//! cd crates/rlevo-benchmarks-report-client
//! trunk build --release
//!
//! # 2) Run this example. Trains PPO, records frames + metrics, opens
//! #    the run, and emits a single-file index.html.
//! cd ../../  # back to repo root
//! cargo run -p rlevo-examples --example report_ppo_cartpole_with_client \
//!     --features viz-record,viz-report --release
//! ```
//!
//! `--release` matters: PPO is unusable at debug speed.

#[path = "../common/ppo_cartpole.rs"]
mod ppo_cartpole;

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use rand::SeedableRng;
use rand::rngs::StdRng;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use rlevo_benchmarks::record::{
    RecordSink, RecordWriter, RecordingConfig, RecordingLayer, RecordingTap,
};
use rlevo_benchmarks::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};

use rlevo_environments::classic::cartpole::CartPole;

use ppo_cartpole::{SEED, base_env, build_agent, train};

// Fewer timesteps than the live demos — enough to surface a clear
// learning curve in the convergence panel without a long headless wait.
const TOTAL_TIMESTEPS: usize = 12_000;

const CLIENT_DIST: &str = "crates/rlevo-benchmarks-report-client/dist";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::for_env::<CartPole>(SEED);
    let writer = RecordWriter::open_default(record_cfg)?;
    let run_dir: PathBuf = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    tracing_subscriber::registry()
        .with(RecordingLayer::new(sink.clone()))
        .try_init()?;

    let mut rng = StdRng::seed_from_u64(SEED);

    // CartPole → TimeLimit → RecordingTap. No TUI tap here — the metric
    // stream is captured headlessly and replayed by the report client.
    // Structured-only (ADR-0013): record `FamilyPayload::Classic2D` line-art.
    let mut env: RecordingTap<_, 1, 1, 1> =
        RecordingTap::with_classic2d_payload(base_env(), sink.clone());
    let mut agent = build_agent(TOTAL_TIMESTEPS);

    train(&mut agent, &mut env, &mut rng, TOTAL_TIMESTEPS)?;

    // Finalise the run manifest before we open the directory for emit.
    sink.lock().on_run_end(manifest);

    // Fail loud on a recording write error before building the report.
    if let Some(e) = sink.lock().take_error() {
        return Err(e.into());
    }

    drop(env);
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
