//! End-to-end smoke test for the record → report pipeline.
//!
//! Records a uniformly-random [`CartPole`] rollout to a temp run
//! directory, reopens it with [`RecordedRun::open`], and emits a
//! client-less static HTML report. Asserts the pipeline round-trips: the
//! manifest's episode count matches the recorded episodes and the emitter
//! writes a non-empty file.
//!
//! A random policy is fine here — we're exercising the *plumbing*, not
//! the learning. The learning-curve story lives in the
//! `report_ppo_cartpole_with_client` example.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::Mutex;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_benchmarks::record::{
    EnvFamily, RecordSink, RecordWriter, RecordingConfig, RecordingTap,
};
use rlevo_benchmarks::report::{EmitConfig, RecordedRun, emit_static_html};

use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::classic::{CartPole, CartPoleAction, CartPoleConfig};

const SEED: u64 = 2026;
/// Record until this many episodes have terminated, so the report has a
/// populated episode table regardless of per-episode length.
const TARGET_EPISODES: u32 = 4;

/// Removes its directory on drop so the test leaves no artefacts behind.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Unique temp directory under the system temp root — no `tempfile` dep,
/// no collisions across concurrent test binaries.
///
/// The path is `<tmp>/rlevo-report-smoke-{pid}-{n}` where `pid` is the
/// current process ID (isolates parallel test *binaries*) and `n` is a
/// process-local [`AtomicU32`] counter (isolates parallel calls within
/// the same binary).
fn temp_dir() -> TempDir {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let path = std::env::temp_dir().join(format!("rlevo-report-smoke-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create temp dir");
    TempDir(path)
}

/// Records a random [`CartPole`] rollout into `root` and returns the run
/// directory path.
///
/// The loop stops on the `TARGET_EPISODES`-th terminal step without calling
/// [`RecordingTap::reset`] afterward. A trailing reset would open a new
/// episode header that `on_run_end` never closes, causing
/// `RecordedRun::open` to emit an `EpisodeCountMismatch` warning on
/// round-trip.
fn record_run(root: &Path) -> PathBuf {
    let record_cfg = RecordingConfig::new(EnvFamily::Classic, SEED);
    let writer = RecordWriter::open(root, record_cfg).expect("open writer");
    let run_dir = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    let env = CartPole::with_config(CartPoleConfig {
        seed: SEED,
        ..CartPoleConfig::default()
    });
    let mut tap: RecordingTap<CartPole, 1, 1, 1> = RecordingTap::new(env, sink.clone());
    let mut rng = StdRng::seed_from_u64(SEED);

    tap.reset().expect("reset");
    let mut episodes = 0;
    loop {
        let idx = rng.random_range(0..CartPoleAction::ACTION_COUNT);
        let snapshot = tap.step(CartPoleAction::from_index(idx)).expect("step");
        if snapshot.is_done() {
            episodes += 1;
            // Stop on the final done *without* resetting — a trailing
            // reset would open a phantom episode that the manifest count
            // wouldn't include, tripping an EpisodeCountMismatch warning.
            if episodes >= TARGET_EPISODES {
                break;
            }
            tap.reset().expect("reset");
        }
    }

    sink.lock().on_run_end(manifest);
    assert!(
        sink.lock().take_error().is_none(),
        "recording hit a write error"
    );
    drop(tap);
    drop(sink);

    run_dir
}

#[test]
fn record_then_report_round_trips() {
    let root = temp_dir();
    let run_dir = record_run(&root.0);

    let run = RecordedRun::open(&run_dir).expect("open recorded run");
    assert!(
        run.warnings().is_empty(),
        "unexpected open warnings: {:?}",
        run.warnings()
    );
    assert_eq!(
        run.episodes().len(),
        TARGET_EPISODES as usize,
        "every terminated episode should be recorded"
    );

    let out = run_dir.join("index.html");
    let outcome = emit_static_html(&run, &out, &EmitConfig::default()).expect("emit html");

    assert_eq!(outcome.episode_count, TARGET_EPISODES);
    assert!(outcome.bytes_written > 0, "report should be non-empty");
    assert!(out.exists(), "index.html should be written to disk");
}
