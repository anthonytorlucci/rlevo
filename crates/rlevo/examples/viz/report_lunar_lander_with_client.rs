//! [`LunarLanderDiscrete`] recording → static-HTML report mounting the
//! Leptos/WASM client with the **box2d SVG adapter** (rigid-body
//! polygons: lander, legs, helipad).
//!
//! ```bash
//! cd crates/rlevo-benchmarks-report-client && trunk build --release
//! cd ../../
//! cargo run -p rlevo --example report_lunar_lander_with_client \
//!     --features box2d,viz-record,viz-report
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use rand::{RngExt, SeedableRng, rngs::StdRng};

use rlevo_benchmarks::record::{
    EnvFamily, RecordSink, RecordWriter, RecordingConfig, RecordingTap,
};
use rlevo_benchmarks::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::box2d::lunar_lander::{
    LunarLanderConfig, LunarLanderDiscrete, LunarLanderDiscreteAction,
};

const SEED: u64 = 2026;
const NUM_EPISODES: usize = 4;
const MAX_STEPS_PER_EPISODE: usize = 250;
const CLIENT_DIST: &str = "crates/rlevo-benchmarks-report-client/dist";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::new(EnvFamily::Box2d, SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
    let run_dir: PathBuf = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    let env = LunarLanderDiscrete::with_config(LunarLanderConfig {
        seed: SEED,
        ..LunarLanderConfig::default()
    });
    let mut tap: RecordingTap<_, 1, 1, 1> = RecordingTap::with_box2d_payload(env, sink.clone());

    for ep in 0..NUM_EPISODES {
        let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(ep as u64));
        tap.reset()?;
        for _ in 0..MAX_STEPS_PER_EPISODE {
            let idx: u32 = rng.random_range(0..4);
            let action = match idx {
                0 => LunarLanderDiscreteAction::DoNothing,
                1 => LunarLanderDiscreteAction::LeftEngine,
                2 => LunarLanderDiscreteAction::MainEngine,
                _ => LunarLanderDiscreteAction::RightEngine,
            };
            let snap = tap.step(action)?;
            if snap.is_done() {
                break;
            }
        }
    }
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
