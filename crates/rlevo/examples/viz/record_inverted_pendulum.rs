//! [`InvertedPendulum`] recording with the M7 **locomotion** payload —
//! the family's canonical view, since locomotion envs do not implement
//! [`AsciiRenderable`] per ADR-0008.
//!
//! Uses [`RecordingTap::with_locomotion_payload`] which goes through
//! [`RecordingTap::new_headless`] (no ASCII / styled capture; the
//! sagittal-plane joint topology is the only rendering surface).
//!
//! ```bash
//! cargo run -p rlevo --example record_inverted_pendulum \
//!     --features locomotion,viz-record
//! ```
//!
//! [`AsciiRenderable`]: rlevo_core::render::AsciiRenderable
//! [`RecordingTap::with_locomotion_payload`]: rlevo_benchmarks::record::RecordingTap::with_locomotion_payload
//! [`RecordingTap::new_headless`]: rlevo_benchmarks::record::RecordingTap::new_headless

use std::sync::{Arc, Mutex};

use rand::{RngExt, SeedableRng, rngs::StdRng};

use rlevo_benchmarks::record::{
    EnvFamily, RecordSink, RecordWriter, RecordingConfig, RecordingTap,
};
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::locomotion::inverted_pendulum::{
    InvertedPendulumAction, InvertedPendulumConfig, InvertedPendulumRapier,
};

const SEED: u64 = 2026;
const NUM_EPISODES: usize = 4;
const MAX_STEPS_PER_EPISODE: usize = 200;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::new(EnvFamily::Locomotion, SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    let env = InvertedPendulumRapier::with_config(InvertedPendulumConfig {
        seed: SEED,
        ..InvertedPendulumConfig::default()
    });
    let mut tap: RecordingTap<_, 1, 1, 1> =
        RecordingTap::with_locomotion_payload(env, sink.clone());

    for ep in 0..NUM_EPISODES {
        let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(ep as u64));
        tap.reset()?;
        for _step in 0..MAX_STEPS_PER_EPISODE {
            // Mildly biased random control — explores both directions
            // so the joint trajectory has visible motion.
            let force: f32 = rng.random_range(-2.0..2.0);
            let snap = tap.step(InvertedPendulumAction::new(force))?;
            if snap.is_done() {
                break;
            }
        }
    }

    sink.lock().unwrap().on_run_end(manifest);
    drop(tap);
    drop(sink);
    println!("wrote {NUM_EPISODES} inverted-pendulum episodes under runs/");
    Ok(())
}
