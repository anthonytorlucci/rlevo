//! BYOE-1 — the bring-your-own-environment acceptance program.
//!
//! This file is **not** part of the rlevo workspace. `scripts/byoe_acceptance.sh`
//! copies it into a throwaway crate outside the repository, points that crate at
//! `cargo package` tarballs, and runs it. It may only use the public API of
//! published crates: reaching into workspace internals would defeat the test.
//!
//! It plays a researcher who has never seen the repository and wants to run
//! *their own* environment — here a building thermostat, deliberately unlike
//! anything in `rlevo-environments` — through the harness to a rendered report.
//!
//! Every step prints one machine-readable line:
//!
//! ```text
//! BYOE-STEP <n> <PASS|FAIL> <label> :: <detail>
//! ```
//!
//! The process exits with the number of the first failing step, or 0 if all
//! pass. Step 9 is expected to fail today (see issue #24); the script treats
//! that as the known baseline rather than a surprise.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use parking_lot::Mutex;
use rand::Rng;

use rlevo::prelude::{Action, Environment, EnvironmentError, Observation, Snapshot, State};
use rlevo::core::environment::{EpisodeStatus, SnapshotBase};
use rlevo::core::render::AsciiRenderable;
use rlevo::core::reward::ScalarReward;

use rlevo::benchmarks::agent::BenchableAgent;
use rlevo::benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo::benchmarks::record::{
    EnvFamily, FamilyPayload, RecordSink, RecordWriter, RecordingConfig, RecordingTap,
};
use rlevo::benchmarks::report::{EmitConfig, RecordedRun, emit_static_html};
use rlevo::benchmarks::reporter::logging::LoggingReporter;
use rlevo::benchmarks::suite::Suite;

const SEED: u64 = 2026;
const TARGET_C: f32 = 21.0;
const HORIZON: usize = 24;
const TARGET_EPISODES: u32 = 3;

// ---------------------------------------------------------------------------
// Step 3 — the researcher's own environment, written from scratch.
// ---------------------------------------------------------------------------

/// What the controller sees: the room and the weather.
#[derive(Debug, Clone)]
struct RoomObs {
    indoor_c: f32,
    outdoor_c: f32,
}

impl Observation<1> for RoomObs {
    fn shape() -> [usize; 1] {
        [2]
    }
}

/// Full room state. Rank 1, two elements — same shape as the observation here,
/// because this domain happens to be fully observed.
#[derive(Debug, Clone)]
struct RoomState {
    indoor_c: f32,
    hour: usize,
}

impl State<1> for RoomState {
    fn shape() -> [usize; 1] {
        [2]
    }

    fn is_valid(&self) -> bool {
        self.indoor_c.is_finite() && self.hour <= HORIZON
    }
}

/// Boiler command for one hour.
///
/// FINDING (spec blocker B9): `Serialize` is not required by the `Action`
/// trait, which is only `Debug + Clone + Sized`. It is required by
/// `impl Environment for RecordingTap`, whose bounds read
/// `E::ActionType: Serialize + Clone` — a consuming-seam declaration, correct
/// per `docs/rules.md`, but invisible to anyone reading `Action`,
/// `Environment`, or the recording docs. Discovering it costs a fourth
/// hand-added dependency (`serde`) and a compiler error that does not say so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
enum Boiler {
    Off,
    On,
}

impl Action<1> for Boiler {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        true
    }
}

/// A day of thermostat control. The boiler adds heat, the outdoors removes it,
/// and the reward is the negative absolute deviation from the target plus a
/// small penalty for running the boiler.
#[derive(Debug, Clone)]
struct Thermostat {
    indoor_c: f32,
    outdoor_c: f32,
    hour: usize,
    /// Hand-rolled post-terminal guard.
    ///
    /// FINDING (spec blocker B5): `rlevo-core` and ADR 0044 place this
    /// obligation on every `Environment` implementor, but the only
    /// implementation of it — `EpisodeGuard` — ships inside
    /// `rlevo-environments`, which a bring-your-own-environment researcher
    /// has no other reason to depend on. Note it cannot be a `bool`:
    /// `EnvironmentError::StepAfterEpisodeEnd` carries the `EpisodeStatus`
    /// that ended the episode, so every implementor re-derives a
    /// status-carrying state machine.
    ended: Option<EpisodeStatus>,
}

impl Thermostat {
    fn with_seed(seed: u64) -> Self {
        // A fixed problem instance sampled once at construction: the outdoor
        // temperature for this building. Per the host-RNG convention, this is
        // preserved across resets rather than re-drawn.
        let outdoor_c = 2.0 + (seed % 11) as f32;
        Self {
            indoor_c: outdoor_c,
            outdoor_c,
            hour: 0,
            ended: None,
        }
    }

    fn obs(&self) -> RoomObs {
        RoomObs {
            indoor_c: self.indoor_c,
            outdoor_c: self.outdoor_c,
        }
    }

}

impl Environment<1, 1, 1> for Thermostat {
    type StateType = RoomState;
    type ObservationType = RoomObs;
    type ActionType = Boiler;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, RoomObs, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.indoor_c = self.outdoor_c;
        self.hour = 0;
        self.ended = None;
        Ok(SnapshotBase::running(self.obs(), ScalarReward(0.0)))
    }

    fn step(&mut self, action: Boiler) -> Result<Self::SnapshotType, EnvironmentError> {
        // Step 4 lives here: the check is at the very top, before any state
        // mutation, so a rejected call leaves the environment untouched.
        if let Some(status) = self.ended {
            return Err(EnvironmentError::StepAfterEpisodeEnd { status });
        }

        let heat = match action {
            Boiler::On => 3.0,
            Boiler::Off => 0.0,
        };
        // Newton cooling toward outdoor, plus boiler input.
        let loss = 0.25 * (self.indoor_c - self.outdoor_c);
        self.indoor_c += heat - loss;
        self.hour += 1;

        let comfort = -(self.indoor_c - TARGET_C).abs();
        let fuel = if action == Boiler::On { 0.4 } else { 0.0 };
        let reward = ScalarReward(comfort - fuel);

        if self.hour >= HORIZON {
            self.ended = Some(EpisodeStatus::Terminated);
            Ok(SnapshotBase::terminated(self.obs(), reward))
        } else {
            Ok(SnapshotBase::running(self.obs(), reward))
        }
    }
}

/// FINDING (spec blocker B3a): implementing this is not required by the
/// architecture — ADR 0007 and `docs/rules.md` demote `AsciiRenderable` to
/// "an optional debug helper" that env families are *not* required to
/// implement. But `RecordingTap::new` and `RecordingTap::with_payload_extractor`
/// both sit in a `where E: AsciiRenderable` impl block, so without it the only
/// recording constructor left is `new_headless`. The ergonomic path requires a
/// trait the architecture calls optional.
impl AsciiRenderable for Thermostat {
    fn render_ascii(&self) -> String {
        format!(
            "hour {:>2}/{}  indoor {:>5.1}C  outdoor {:>5.1}C  target {:.1}C",
            self.hour, HORIZON, self.indoor_c, self.outdoor_c, TARGET_C
        )
    }
}

// ---------------------------------------------------------------------------
// Step 5 — the researcher's own policy.
// ---------------------------------------------------------------------------

/// Bang-bang controller: heat whenever the room is below target.
#[derive(Debug)]
struct BangBang;

impl BenchableAgent<RoomObs, Boiler> for BangBang {
    fn act(&mut self, obs: &RoomObs, _rng: &mut dyn Rng) -> Boiler {
        // Widen the deadband when it is cold out: the room bleeds heat faster,
        // so cutting off exactly at target undershoots for the rest of the hour.
        let deadband = if obs.outdoor_c < 5.0 { 0.5 } else { 0.0 };
        if obs.indoor_c < TARGET_C + deadband {
            Boiler::On
        } else {
            Boiler::Off
        }
    }
}

// ---------------------------------------------------------------------------
// Step reporting
// ---------------------------------------------------------------------------

fn pass(n: u32, label: &str, detail: &str) {
    println!("BYOE-STEP {n} PASS {label} :: {detail}");
}

fn fail(n: u32, label: &str, detail: &str) -> ExitCode {
    println!("BYOE-STEP {n} FAIL {label} :: {detail}");
    ExitCode::from(n as u8)
}

/// Removes its directory on drop so a local run leaves nothing behind.
struct TempRun(PathBuf);

impl Drop for TempRun {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn main() -> ExitCode {
    // Steps 1-3 and 5 are discharged by this program existing and compiling:
    // the crate was scaffolded outside the workspace, its dependencies
    // resolved, and `Thermostat`/`BangBang` satisfy `Environment` and
    // `BenchableAgent` using public API only. The shell driver reports them.

    // -- Step 4: post-terminal contract (ADR 0044) ---------------------------
    {
        let mut env = Thermostat::with_seed(SEED);
        if let Err(e) = env.reset() {
            return fail(4, "post-terminal-contract", &format!("reset failed: {e}"));
        }
        let mut last_done = false;
        for _ in 0..HORIZON {
            match env.step(Boiler::On) {
                Ok(s) => last_done = s.is_done(),
                Err(e) => return fail(4, "post-terminal-contract", &format!("step failed: {e}")),
            }
        }
        if !last_done {
            return fail(
                4,
                "post-terminal-contract",
                "episode did not terminate within its own horizon",
            );
        }
        match env.step(Boiler::On) {
            Err(EnvironmentError::StepAfterEpisodeEnd { status }) => pass(
                4,
                "post-terminal-contract",
                &format!("rejected with status {status:?}; guard hand-rolled (blocker B5)"),
            ),
            Err(e) => {
                return fail(
                    4,
                    "post-terminal-contract",
                    &format!("wrong error variant: {e}"),
                );
            }
            Ok(_) => {
                return fail(
                    4,
                    "post-terminal-contract",
                    "step after done returned Ok - contract violated",
                );
            }
        }
    }

    // -- Step 6: drive it through the harness --------------------------------
    let eval_cfg = EvaluatorConfig {
        num_episodes: 3,
        num_trials_per_env: 2,
        max_steps: HORIZON + 2,
        base_seed: SEED,
        num_threads: Some(2),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: None,
    };
    let suite: Suite<Thermostat> =
        Suite::new("byoe-thermostat", eval_cfg.clone()).with_env("thermostat", Thermostat::with_seed);
    let mut reporter = LoggingReporter::new();
    let report = Evaluator::new(eval_cfg).run_suite(&suite, |_seed| BangBang, &mut reporter);

    if report.trials.len() != 2 {
        return fail(
            6,
            "run-suite",
            &format!("expected 2 trials, got {}", report.trials.len()),
        );
    }
    pass(
        6,
        "run-suite",
        &format!("{} trials completed via public API", report.trials.len()),
    );

    // -- Step 7: record the run ----------------------------------------------
    let root = std::env::temp_dir().join(format!("byoe-run-{}", std::process::id()));
    if std::fs::create_dir_all(&root).is_err() {
        return fail(7, "record", "could not create a temp run root");
    }
    let _cleanup = TempRun(root.clone());

    // FINDING (spec blocker B3): a thermostat is none of the six EnvFamily
    // variants. It must masquerade as somebody else's family to record at all.
    let record_cfg = RecordingConfig::new(EnvFamily::Classic, SEED);
    let writer = match RecordWriter::open(&root, record_cfg) {
        Ok(w) => w,
        Err(e) => return fail(7, "record", &format!("RecordWriter::open failed: {e}")),
    };
    let run_dir = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    let mut tap: RecordingTap<Thermostat, 1, 1, 1> =
        RecordingTap::new(Thermostat::with_seed(SEED), sink.clone());
    let mut policy = BangBang;
    let mut rng = rand::rng();

    if tap.reset().is_err() {
        return fail(7, "record", "tap reset failed");
    }
    let mut episodes = 0u32;
    let mut guard = 0u32;
    // Bounded: the loop must not depend on the implementation terminating.
    let max_iters = (HORIZON as u32 + 2) * (TARGET_EPISODES + 1);
    loop {
        guard += 1;
        if guard > max_iters {
            return fail(7, "record", "episode loop exceeded its own bound");
        }
        let obs = RoomObs {
            indoor_c: TARGET_C - 1.0,
            outdoor_c: 0.0,
        };
        let action = policy.act(&obs, &mut rng);
        let snapshot = match tap.step(action) {
            Ok(s) => s,
            Err(e) => return fail(7, "record", &format!("tap step failed: {e}")),
        };
        if snapshot.is_done() {
            episodes += 1;
            if episodes >= TARGET_EPISODES {
                break;
            }
            if tap.reset().is_err() {
                return fail(7, "record", "tap reset failed mid-run");
            }
        }
    }
    sink.lock().on_run_end(manifest);
    if let Some(e) = sink.lock().take_error() {
        return fail(7, "record", &format!("recording hit a write error: {e}"));
    }
    drop(tap);
    pass(
        7,
        "record",
        &format!("{episodes} episodes written to {}", run_dir.display()),
    );

    // -- Step 8: emit a static HTML report, and check it is worth having -----
    let run = match RecordedRun::open(&run_dir) {
        Ok(r) => r,
        Err(e) => return fail(8, "static-report", &format!("RecordedRun::open failed: {e}")),
    };
    if !run.warnings().is_empty() {
        return fail(
            8,
            "static-report",
            &format!("open warnings: {:?}", run.warnings()),
        );
    }
    if run.episodes().len() != TARGET_EPISODES as usize {
        return fail(
            8,
            "static-report",
            &format!(
                "expected {TARGET_EPISODES} episodes, manifest has {}",
                run.episodes().len()
            ),
        );
    }
    let out = run_dir.join("index.html");
    let outcome = match emit_static_html(&run, &out, &EmitConfig::default()) {
        Ok(o) => o,
        Err(e) => return fail(8, "static-report", &format!("emit failed: {e}")),
    };
    if outcome.bytes_written == 0 {
        return fail(8, "static-report", "report is empty");
    }

    // The richness assertion. Without it this step passes on a report whose
    // every frame is the ASCII fallback — the exact degradation the test
    // exists to catch (spec blocker B8, issues #306/#307/#309).
    let frame = run.frame_at_or_before(0, u32::MAX);
    match frame {
        None => {
            return fail(
                8,
                "static-report",
                "no frame recorded for episode 0 - cannot assess payload richness",
            );
        }
        Some(f) if matches!(f.family_payload, FamilyPayload::Ascii) => {
            return fail(
                8,
                "static-report",
                "frames carry FamilyPayload::Ascii - report renders the generic \
                 fallback, not a rich view. A third-party env has no rich payload \
                 shape of its own (blocker B3/B8)",
            );
        }
        Some(_) => pass(
            8,
            "static-report",
            &format!("{} bytes, rich payload present", outcome.bytes_written),
        ),
    }

    // -- Step 9: emit with the interactive client ----------------------------
    // `ClientAssets::from_trunk_dist` is the only constructor. It reads a
    // `trunk build --release` output directory produced from
    // `rlevo-benchmarks-report-client`, which is `publish = false` — so a
    // crates.io consumer cannot obtain it at all (issue #24, blocker B1).
    match std::env::var("BYOE_TRUNK_DIST") {
        Ok(dir) if Path::new(&dir).is_dir() => pass(
            9,
            "interactive-report",
            &format!("client assets supplied out of band from {dir}"),
        ),
        _ => {
            return fail(
                9,
                "interactive-report",
                "no bundled client assets reachable from crates.io: \
                 ClientAssets::from_trunk_dist is the only constructor and it \
                 needs a trunk build of the publish=false report client (#24)",
            );
        }
    }

    ExitCode::SUCCESS
}
