//! [`LunarLanderDiscrete`] recording with the rich **box2d** payload
//! (rigid-body polygons for the report-tier SVG adapter).
//!
//! Uses [`RecordingTap::with_box2d_payload`] — the lander, both legs,
//! and the ground slab project as [`RigidBody2D`] polygons with typed
//! [`BodyKind`] discriminants. Leg-contact dots round-trip into the
//! payload's `contacts` vector.
//!
//! ```bash
//! cargo run -p rlevo --example record_lunar_lander \
//!     --features box2d,viz-record
//! ```
//!
//! [`RecordingTap::with_box2d_payload`]: rlevo_benchmarks::record::RecordingTap::with_box2d_payload
//! [`RigidBody2D`]: rlevo_core::render::RigidBody2D
//! [`BodyKind`]: rlevo_core::render::BodyKind

use std::sync::Arc;

use parking_lot::Mutex;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use rlevo_benchmarks::record::{
    EnvFamily, RecordSink, RecordWriter, RecordingConfig, RecordingTap,
};
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::box2d::lunar_lander::{
    LunarLanderConfig, LunarLanderDiscrete, LunarLanderDiscreteAction,
};

const SEED: u64 = 2026;
const NUM_EPISODES: usize = 4;
const MAX_STEPS_PER_EPISODE: usize = 250;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::new(EnvFamily::Box2d, SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
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
            // 4 discrete actions: DoNothing/LeftEngine/MainEngine/RightEngine.
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

    // Fail loud on a recording write error rather than reporting success.
    if let Some(e) = sink.lock().take_error() {
        return Err(e.into());
    }

    drop(tap);
    drop(sink);
    println!("wrote {NUM_EPISODES} lunar-lander episodes under runs/");
    Ok(())
}
