//! Synthesised Sphere-landscape hill-climber recording → static-HTML
//! report that mounts the Leptos/WASM report client with the **landscape
//! SVG adapter** (search-space projection: trail polyline + current
//! candidate marker + best-so-far marker).
//!
//! Build flow mirrors `report_ppo_cartpole_with_client`:
//!
//! ```bash
//! cd crates/rlevo-benchmarks-report-client && trunk build --release
//! cd ../../
//! cargo run -p rlevo --example report_sphere_landscape_with_client \
//!     --features viz-record,viz-report
//! ```
//!
//! Opening the emitted `index.html` shows the manifest, episode table,
//! and the landscape SVG: bounded search domain + current candidate
//! marker + best-so-far + trail polyline.

// Example driver: f64 world coordinates are deliberately narrowed to the f32
// render precision of `Point2` (and the episode index to `u32`); the linear
// recording script also runs past the default function-length lint.
#![allow(clippy::cast_possible_truncation, clippy::too_many_lines)]

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use rand::{RngExt, SeedableRng, rngs::StdRng};

use rlevo_benchmarks::record::{
    EnvFamily, FamilyPayload, FrameRecord, Landscape2DPayload, RecordSink, RecordWriter,
    RecordingConfig,
};
use rlevo_benchmarks::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};
use rlevo_core::render::{Landscape2DSnapshot, Point2};

use rlevo_environments::landscapes::sphere::Sphere;

const SEED: u64 = 2026;
const NUM_EPISODES: usize = 6;
const STEPS_PER_EPISODE: u32 = 80;
const TRAIL_CAP: usize = 32;
const CLIENT_DIST: &str = "crates/rlevo-benchmarks-report-client/dist";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::new(EnvFamily::Landscapes, SEED);
    let writer = RecordWriter::open_default(record_cfg)?;
    let run_dir: PathBuf = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    let sphere = Sphere::new(2);
    let (lo, hi) = sphere.bounds();
    let bounds_f32 = (lo as f32, hi as f32);

    for ep in 0..NUM_EPISODES {
        let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(ep as u64));
        let mut current = [
            rng.random_range(lo..hi),
            rng.random_range(lo..hi),
        ];
        let mut current_f = sphere.evaluate(&current);
        let mut best = current;
        let mut best_f = current_f;
        let mut trail: Vec<Point2> = Vec::new();

        sink.lock().on_episode_start(ep as u32);
        let mut episode_return = 0.0f64;

        for step in 0..STEPS_PER_EPISODE {
            let dx = rng.random_range(-0.4..0.4);
            let dy = rng.random_range(-0.4..0.4);
            let candidate = [
                (current[0] + dx).clamp(lo, hi),
                (current[1] + dy).clamp(lo, hi),
            ];
            let cand_f = sphere.evaluate(&candidate);
            let improved = cand_f < current_f;
            if improved {
                current = candidate;
                current_f = cand_f;
                if cand_f < best_f {
                    best = candidate;
                    best_f = cand_f;
                }
            }
            trail.push(Point2::new(current[0] as f32, current[1] as f32));
            if trail.len() > TRAIL_CAP {
                trail.remove(0);
            }
            let reward = if improved { (current_f - cand_f) as f32 } else { 0.0 };
            episode_return += f64::from(reward);

            let snapshot = Landscape2DSnapshot {
                bounds_x: bounds_f32,
                bounds_y: bounds_f32,
                current: Point2::new(current[0] as f32, current[1] as f32),
                best: Some(Point2::new(best[0] as f32, best[1] as f32)),
                trail: trail.clone(),
                label: "sphere".into(),
            };
            let frame = FrameRecord {
                step,
                action: Vec::new(),
                reward,
                ascii: Some(format!(
                    "sphere  ep={ep} step={step}  current=({:.3}, {:.3})  f={:.3}  best={:.3}",
                    current[0], current[1], current_f, best_f,
                )),
                styled: None,
                family_payload: FamilyPayload::Landscape2D(Landscape2DPayload::from(snapshot)),
            };
            sink.lock().on_frame(frame);
        }
        sink.lock()
            .on_episode_end(episode_return, STEPS_PER_EPISODE);
    }
    sink.lock().on_run_end(manifest);

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
