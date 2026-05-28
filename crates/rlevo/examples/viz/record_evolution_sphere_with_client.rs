//! Genetic-algorithm Sphere-D2 run → static-HTML report that mounts
//! the Leptos/WASM client with **EA population panels** (per-generation
//! box plot, diversity trace, selection-pressure indicator) alongside
//! the RL-style convergence section (fitness aggregates).
//!
//! Two producers feed the same shared sink:
//!
//! ```text
//!  EvolutionaryHarness (GA, Sphere-D2)
//!    ├─ with_observer(PopulationReporter) → RecordChunk::Population
//!    │                                       per generation
//!    └─ tracing::info!(best_fitness=…, mean_fitness=…, worst_fitness=…,
//!                       best_fitness_ever=…)
//!         └─ RecordingLayer (canonical metric registry)
//!              → RecordChunk::Metrics
//!
//!  best-so-far genome  → FrameRecord(Landscape2D payload)
//!                        once per generation, for the playback pane
//! ```
//!
//! Two-step build flow:
//!
//! ```bash
//! cd crates/rlevo-benchmarks-report-client
//! trunk build --release
//!
//! cd ../../
//! cargo run -p rlevo --example record_evolution_sphere_with_client \
//!     --features viz-record,viz-report --release
//! ```
//!
//! Opening the emitted `index.html` shows the interactive playback pane
//! (landscape SVG over generations) plus the Convergence section
//! (fitness aggregates) plus the **Population section** (per-generation
//! box plot, selection-pressure indicator).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use burn::backend::Flex;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use rlevo_benchmarks::record::{
    EnvFamily, FamilyPayload, FrameRecord, Landscape2DPayload, PopulationReporter, RecordSink,
    RecordWriter, RecordingConfig, RecordingLayer,
};
use rlevo_benchmarks::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};
use rlevo_core::render::{Landscape2DSnapshot, Point2};

use rlevo_environments::landscapes::sphere::Sphere;
use rlevo_evolution::SharedPopulationObserver;
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::fitness::FromLandscape;
use rlevo_evolution::strategy::EvolutionaryHarness;

type B = Flex;

const DIM: usize = 2;
const POP_SIZE: usize = 64;
const GENERATIONS: usize = 50;
const SEED: u64 = 42;
const TRAIL_CAP: usize = 32;
const CLIENT_DIST: &str = "crates/rlevo-benchmarks-report-client/dist";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::new(EnvFamily::Landscapes, SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
    let run_dir: PathBuf = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    tracing_subscriber::registry()
        .with(RecordingLayer::new(sink.clone()))
        .try_init()?;

    let reporter: Arc<Mutex<PopulationReporter>> =
        Arc::new(Mutex::new(PopulationReporter::new(sink.clone())));
    let observer: SharedPopulationObserver = reporter.clone();

    let sphere = Sphere::new(DIM);
    let (lo, hi) = sphere.bounds();
    #[allow(clippy::cast_possible_truncation)]
    let bounds_f32 = (lo as f32, hi as f32);

    let device = Default::default();
    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        GeneticAlgorithm::<B>::new(),
        GaConfig {
            pop_size: POP_SIZE,
            genome_dim: DIM,
            bounds: bounds_f32,
            mutation_sigma: 0.3,
            selection: GaSelection::Tournament { size: 2 },
            crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
            replacement: GaReplacement::Elitist { elitism_k: 2 },
        },
        FromLandscape::new(sphere),
        SEED,
        device,
        GENERATIONS,
    )
    .with_observer(observer);

    harness.reset();
    sink.lock().unwrap().on_episode_start(0);

    let mut trail: Vec<Point2> = Vec::new();
    let mut episode_return = 0.0_f64;
    let mut frames_emitted: u32 = 0;

    loop {
        let outcome = harness.step(());
        episode_return += outcome.reward;
        frames_emitted += 1;

        if let Some((best_genome, best_fitness)) = harness.best() {
            let data = best_genome
                .into_data()
                .into_vec::<f32>()
                .unwrap_or_default();
            let (x, y) = (data.first().copied().unwrap_or(0.0), data.get(1).copied().unwrap_or(0.0));
            trail.push(Point2::new(x, y));
            if trail.len() > TRAIL_CAP {
                trail.remove(0);
            }
            let snap = Landscape2DSnapshot {
                bounds_x: bounds_f32,
                bounds_y: bounds_f32,
                current: Point2::new(x, y),
                best: Some(Point2::new(x, y)),
                trail: trail.clone(),
                label: "sphere".into(),
            };
            #[allow(clippy::cast_possible_truncation)]
            let frame = FrameRecord {
                step: frames_emitted,
                action: Vec::new(),
                reward: outcome.reward as f32,
                ascii: Some(format!(
                    "gen={frames_emitted} best=({x:.3},{y:.3}) f={best_fitness:.4e}",
                )),
                styled: None,
                family_payload: FamilyPayload::Landscape2D(Landscape2DPayload::from(snap)),
            };
            sink.lock().unwrap().on_frame(frame);
        }

        if outcome.done {
            break;
        }
    }

    sink.lock()
        .unwrap()
        .on_episode_end(episode_return, frames_emitted);
    sink.lock().unwrap().on_run_end(manifest);
    drop(harness);
    drop(reporter);
    drop(sink);

    let run = RecordedRun::open(&run_dir)?;
    for w in run.warnings() {
        eprintln!("warning: {w:?}");
    }

    // Smoke check: confirm the .rec stream picked up Population chunks
    // by re-reading episode 0 directly. The report emitter base64-inlines
    // raw bytes (population samples ride along), but `RecordedRun` only
    // surfaces frames + metrics, so this read is the easiest place to
    // assert the producer wrote what the client will later decode.
    let episode_path = run_dir.join("episode_000000.rec");
    let probe = rlevo_benchmarks::record::read_episode_record(&episode_path)?;
    println!(
        "  → episode_000000.rec: frames={} metrics={} population_samples={}",
        probe.frames.len(),
        probe.metrics.len(),
        probe.population_samples.len()
    );

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
