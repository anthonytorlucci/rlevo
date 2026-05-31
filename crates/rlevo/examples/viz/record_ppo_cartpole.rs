//! PPO + live TUI + on-disk recording on [`CartPole`]. The full
//! env-wrapper + recording composition demonstrated end-to-end.
//!
//! Composition map:
//!
//! ```text
//!   CartPole
//!     └─ TimeLimit
//!         └─ TuiEnvTap   (frames + episode returns to live TUI)
//!             └─ RecordingTap (frame + metric stream to disk)
//!                 ↓
//!              train_discrete
//! ```
//!
//! Tracing layers: `TuiCaptureLayer` (live metric sparklines + log) and
//! `RecordingLayer` (on-disk `MetricSample` stream) installed side-by-side
//! through the same `tracing_subscriber::Registry`.
//!
//! # Run with
//!
//! ```bash
//! cargo run -p rlevo --example record_ppo_cartpole \
//!   --features viz-tui,viz-record --release
//! ```
//!
//! `--release` matters: PPO is unusable at debug speed.

#[path = "common/ppo_cartpole.rs"]
mod ppo_cartpole;

use std::sync::Arc;

use parking_lot::Mutex;

use rand::SeedableRng;
use rand::rngs::StdRng;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use rlevo_benchmarks::env_wrappers::TuiEnvTap;
use rlevo_benchmarks::record::{
    RecordSink, RecordWriter, RecordedEnvFamily, RecordingConfig, RecordingLayer, RecordingTap,
};
use rlevo_benchmarks::tui::{TuiCaptureLayer, TuiConfig, TuiRunner};

use rlevo_environments::classic::cartpole::CartPole;

use ppo_cartpole::{SEED, base_env, build_agent, train};

const TOTAL_TIMESTEPS: usize = 20_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Family declared once by the underlying env type (`CartPole`), shared by
    // the TUI and recording config even though `base_env()` wraps it.
    let runner = TuiRunner::start(TuiConfig::default().with_env_family(CartPole::FAMILY))?;
    let handle = runner.handle();

    let record_cfg = RecordingConfig::for_env::<CartPole>(SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    tracing_subscriber::registry()
        .with(TuiCaptureLayer::new(handle.clone()))
        .with(RecordingLayer::new(sink.clone()))
        .try_init()?;

    let mut rng = StdRng::seed_from_u64(SEED);

    // CartPole → TimeLimit → TuiEnvTap → RecordingTap. RecordingTap wraps
    // the TuiEnvTap so the same trajectory feeds the live TUI AND the
    // on-disk record from one wrap site.
    let tui_tapped: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(base_env(), handle);
    let mut env: RecordingTap<_, 1, 1, 1> = RecordingTap::new(tui_tapped, sink.clone());
    let mut agent = build_agent(TOTAL_TIMESTEPS);

    train(&mut agent, &mut env, &mut rng, TOTAL_TIMESTEPS)?;

    // Finalise the run manifest on the shared sink before we tear the TUI
    // down. PPO has no Reporter chain to drive this, so the example calls
    // on_run_end directly.
    sink.lock().on_run_end(manifest);

    // Fail loud if recording hit a write error. `TuiRunner::drop` restores
    // the terminal on the error path.
    if let Some(e) = sink.lock().take_error() {
        return Err(e.into());
    }

    runner.wait_for_keypress()?;
    runner.shutdown()?;
    Ok(())
}
