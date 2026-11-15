//! The update layer: per-update statistics read back from the trial files.
//!
//! The harness runs trials on worker threads, so the layer must be in the
//! process's global subscriber, and a process has one, set once. So this
//! file is one test: it installs the layer once and runs its cases one
//! after another, each with a sink of its own. One layer serves them all,
//! which is what it must get right when several runs share a process.
//!
//! The cases run in turn, not as tests side by side, because the last one
//! logs outside any trial, and every sink then recording counts that.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rand::Rng;
use rlevo_core::agent::{Agent, Transition};
use rlevo_core::fitness::UPDATE_TARGET;
use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_harness::report::{BenchmarkReport, TrialReport};
use rlevo_harness::reporter::{Boundary, Reporter};
use rlevo_harness::suite::{Suite, SuiteInfo, TrialInfo, TrialKey};
use rlevo_rerun::{RerunConfig, RerunSink, UpdateLayer};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use common::{path, read, trial_file};

/// What an agent logged, by the tag it stamps its values with.
type Logged = Arc<Mutex<BTreeMap<u64, Vec<(i64, f64)>>>>;

/// Pushes left and right in turn, and "updates" every fifth training step:
/// one update-rate event, as an outside learner would log it. Its loss is
/// `tag + update index`, so a value says which agent logged it.
struct Learner {
    tag: u64,
    training: bool,
    steps: u64,
    updates: u64,
    /// Whether the event carries `step`.
    names_step: bool,
    logged: Logged,
}

impl Agent<CartPoleObservation, CartPoleAction> for Learner {
    fn act(&mut self, _obs: &CartPoleObservation, _rng: &mut dyn Rng) -> CartPoleAction {
        self.steps += 1;
        if self.steps.is_multiple_of(2) {
            CartPoleAction::Left
        } else {
            CartPoleAction::Right
        }
    }

    fn on_transition(
        &mut self,
        _transition: Transition<&CartPoleObservation, &CartPoleAction>,
        _rng: &mut dyn Rng,
    ) {
        if !self.training || !self.steps.is_multiple_of(5) {
            return;
        }
        #[allow(clippy::cast_precision_loss)]
        let loss = (self.tag * 1000 + self.updates) as f64;
        // Present on even updates only: an `Option` that is `None` must
        // leave no row.
        let sometimes = self.updates.is_multiple_of(2).then_some(loss + 0.5);
        if self.names_step {
            tracing::info!(
                target: UPDATE_TARGET,
                step = self.steps,
                "test/loss" = loss,
                "test/sometimes" = sometimes,
                done = false,
                "learner update"
            );
        } else {
            tracing::info!(target: UPDATE_TARGET, { "test/loss" = loss }, "learner update");
        }
        self.logged
            .lock()
            .unwrap()
            .entry(self.tag)
            .or_default()
            .push((i64::try_from(self.steps).unwrap(), loss));
        self.updates += 1;
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

fn suite() -> Suite<CartPole> {
    Suite::new("layer-suite").with_env("cartpole", |seed| {
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        })
        .expect("valid config")
    })
}

fn config(trials: usize) -> EvaluatorConfig {
    EvaluatorConfig {
        num_episodes: 1,
        train_steps: 30,
        num_trials_per_env: trials,
        max_steps: 12,
        num_threads: Some(2),
        ..EvaluatorConfig::default()
    }
}

/// The file of trial `idx` of `report`'s run, written into `dir` as `label`.
fn file(dir: &Path, report: &BenchmarkReport, label: &str, idx: usize) -> PathBuf {
    let run_id = report.run_id.as_deref().expect("a run has an ID");
    trial_file(dir, run_id, label, 0, "cartpole", idx)
}

fn sink(dir: &tempfile::TempDir, label: &str) -> RerunSink {
    RerunSink::new(RerunConfig::new(dir.path(), label)).expect("the directory exists")
}

/// Runs the suite into `sink`, whose layer is installed, and returns the
/// report and what each agent logged.
fn run(sink: &mut RerunSink, trials: usize, names_step: bool) -> (BenchmarkReport, Logged) {
    let logged = Logged::default();
    let report = Evaluator::new(config(trials)).run_suite(
        &suite(),
        |agent_seed| Learner {
            tag: agent_seed % 1_000_000,
            training: true,
            steps: 0,
            updates: 0,
            names_step,
            logged: Arc::clone(&logged),
        },
        sink,
    );
    (report, logged)
}

/// The cases, in one process and one subscriber. The stray event comes
/// last: a sink recording when it is logged counts it.
#[test]
fn update_events_are_filed_under_their_trials() {
    let dirs: Vec<tempfile::TempDir> = (0..7).map(|_| tempfile::tempdir().unwrap()).collect();
    let mut attrib = sink(&dirs[0], "attrib");
    let mut nostep = sink(&dirs[1], "nostep");
    let mut idle = sink(&dirs[2], "idle");
    let mut busy = sink(&dirs[3], "busy");
    let mut early = sink(&dirs[4], "early");
    let mut left = sink(&dirs[5], "left");
    let mut right = sink(&dirs[6], "right");
    tracing_subscriber::registry()
        .with(UpdateLayer::new())
        .init();

    each_trials_updates_are_filed_under_that_trial(&mut attrib, dirs[0].path());
    an_update_with_no_step_is_stamped_with_the_last_boundary(&mut nostep, dirs[1].path());
    another_runs_updates_are_not_filed(&mut idle, dirs[2].path(), &mut busy, dirs[3].path());
    an_update_before_the_first_boundary_waits_for_the_trials_kind(&mut early, dirs[4].path());
    two_sinks_on_one_run_both_record_it(&mut left, dirs[5].path(), &mut right, dirs[6].path());
    an_update_outside_any_trial_is_counted(&mut attrib);
}

/// Row 9. Four trials on two threads, every agent logging updates: each
/// trial's file holds exactly one agent's values, every agent's values are
/// in some file, each on `update` and on the `step` the event named, and no
/// event went unfiled. A field that is `None` leaves no row, and a field
/// that is not a number is not a metric.
fn each_trials_updates_are_filed_under_that_trial(sink: &mut RerunSink, dir: &Path) {
    let (report, logged) = run(sink, 4, true);
    let logged = logged.lock().unwrap().clone();
    assert_eq!(logged.len(), 4, "every agent logged updates");
    assert_eq!(sink.unattributed_updates(), 0);

    let mut found = Vec::new();
    for trial in &report.trials {
        let idx = trial.key.trial_idx;
        let rec = read(&file(dir, &report, "attrib", idx));
        let loss = path(&format!("/metrics/test/loss/attrib/cartpole/{idx}"));

        let on_step = rec.series(&loss, "step");
        let (tag, expected) = logged
            .iter()
            .find(|(_, values)| **values == on_step)
            .unwrap_or_else(|| {
                panic!("trial {idx}'s file holds no one agent's updates: {on_step:?}")
            });
        found.push(*tag);
        assert_eq!(
            expected.len(),
            6,
            "30 training steps, an update every fifth"
        );

        let on_update: Vec<(i64, f64)> = expected
            .iter()
            .enumerate()
            .map(|(n, (_, loss))| (i64::try_from(n).unwrap(), *loss))
            .collect();
        assert_eq!(rec.series(&loss, "update"), on_update);
        assert_eq!(
            rec.timelines(&loss),
            ["step", "update"].into_iter().map(String::from).collect()
        );

        // Present on updates 0, 2 and 4.
        let sometimes = path(&format!("/metrics/test/sometimes/attrib/cartpole/{idx}"));
        let updates: Vec<i64> = rec
            .series(&sometimes, "update")
            .into_iter()
            .map(|(update, _)| update)
            .collect();
        assert_eq!(updates, [0, 2, 4]);

        assert!(
            rec.entities()
                .iter()
                .all(|e| !e.contains("/done/") && !e.contains("/step/") && !e.contains("message")),
            "only numbers other than `step` are metrics: {:?}",
            rec.entities()
        );
    }
    found.sort_unstable();
    found.dedup();
    assert_eq!(
        found.len(),
        4,
        "each file holds a different agent's updates"
    );
}

/// An event that names no `step` is stamped with the trial's step count at
/// its last episode boundary, so it still draws on `step`. Episodes here end
/// at steps 12 and 24, and updates come at steps 5, 10, 15, 20, 25 and 30.
fn an_update_with_no_step_is_stamped_with_the_last_boundary(sink: &mut RerunSink, dir: &Path) {
    let (report, _) = run(sink, 1, false);
    assert_eq!(sink.unattributed_updates(), 0);

    let rec = read(&file(dir, &report, "nostep", 0));
    let loss = path("/metrics/test/loss/nostep/cartpole/0");
    let steps: Vec<i64> = rec
        .series(&loss, "step")
        .into_iter()
        .map(|(step, _)| step)
        .collect();
    assert_eq!(steps, [0, 0, 12, 12, 24, 24]);
    let updates: Vec<i64> = rec
        .series(&loss, "update")
        .into_iter()
        .map(|(update, _)| update)
        .collect();
    assert_eq!(updates, [0, 1, 2, 3, 4, 5]);
}

/// Two runs in one process, each with a layer in the one subscriber, and a
/// trial of each under the same key: a run's updates are filed in its own
/// files and never in the other's.
fn another_runs_updates_are_not_filed(
    idle: &mut RerunSink,
    idle_dir: &Path,
    busy: &mut RerunSink,
    busy_dir: &Path,
) {
    // `idle` has trial (0, 0) open, of a run nobody is running.
    let key = TrialKey {
        env_idx: 0,
        trial_idx: 0,
    };
    let info = TrialInfo::new(key, "cartpole".to_string(), 7);
    idle.on_suite_start(&SuiteInfo::new(
        "idle-suite".to_string(),
        vec!["cartpole".to_string()],
        1,
        None,
        "00000000000001d1".to_string(),
    ));
    idle.on_trial_start(&info);

    // A real run, whose only trial is also (0, 0), logs its updates.
    let (report, logged) = run(busy, 1, true);
    assert_eq!(
        logged.lock().unwrap().len(),
        1,
        "the busy run logged updates"
    );
    let busy = read(&file(busy_dir, &report, "busy", 0));
    assert_eq!(
        busy.series(&path("/metrics/test/loss/busy/cartpole/0"), "update")
            .len(),
        6
    );

    idle.on_trial_end(&info, &TrialReport::new(key, "cartpole".to_string(), 7));
    idle.on_suite_end(&BenchmarkReport::new("idle-suite".to_string(), 0));
    let rec = read(&trial_file(
        idle_dir,
        "00000000000001d1",
        "idle",
        0,
        "cartpole",
        0,
    ));
    assert!(
        rec.entities().iter().all(|e| !e.starts_with("/metrics")),
        "the idle run's file holds another run's updates: {:?}",
        rec.entities()
    );
    assert_eq!(
        idle.unattributed_updates(),
        0,
        "nor are they counted against it"
    );
}

/// The layer does not know a trial's kind, so it never opens a file. An
/// update logged before the trial's first boundary is kept and written when
/// the sink opens the file, under the kind the boundary gives: here a
/// generation trial with an inner gradient step, which must not be filed as
/// an episodic one.
fn an_update_before_the_first_boundary_waits_for_the_trials_kind(sink: &mut RerunSink, dir: &Path) {
    const RUN: &str = "00000000000000e1";
    let key = TrialKey {
        env_idx: 0,
        trial_idx: 0,
    };
    let info = TrialInfo::new(key, "sphere".to_string(), 3);
    sink.on_suite_start(&SuiteInfo::new(
        "early-suite".to_string(),
        vec!["sphere".to_string()],
        1,
        None,
        RUN.to_string(),
    ));
    sink.on_trial_start(&info);
    let file = trial_file(dir, RUN, "early", 0, "sphere", 0);

    // The harness's span, as `run_trials` opens it.
    let span = tracing::info_span!(
        target: "rlevo_harness",
        parent: None,
        "trial",
        run = RUN,
        env = "sphere",
        env_idx = 0_usize,
        trial = 0_usize,
        seed = 3_u64,
    );
    {
        let _in_trial = span.enter();
        tracing::info!(target: UPDATE_TARGET, step = 7_u64, "test/loss" = 0.25_f64, "inner step");
        tracing::info!(target: UPDATE_TARGET, step = 9_u64, "test/loss" = 0.5_f64, "inner step");
        assert!(!file.exists(), "the layer does not open the file");
        sink.on_metrics(&info, Boundary::Generation { generation: 1 }, &[]);
        tracing::info!(target: UPDATE_TARGET, step = 11_u64, "test/loss" = 0.75_f64, "inner step");
    }
    sink.on_trial_end(&info, &TrialReport::new(key, "sphere".to_string(), 3));
    sink.on_suite_end(&BenchmarkReport::new("early-suite".to_string(), 0));
    assert_eq!(sink.unattributed_updates(), 0);

    let rec = read(&file);
    assert_eq!(rec.application_id, "rlevo-generation");
    let loss = path("/metrics/test/loss/early/sphere/0");
    assert_eq!(
        rec.series(&loss, "update"),
        [(0, 0.25), (1, 0.5), (2, 0.75)]
    );
    assert_eq!(rec.series(&loss, "step"), [(7, 0.25), (9, 0.5), (11, 0.75)]);
}

/// Two sinks recording one run, as two in a `MultiReporter` do: each files
/// the run's updates in its own directory. An update that comes after its
/// trial has ended is of this run and of no open trial: each sink counts it.
fn two_sinks_on_one_run_both_record_it(
    left: &mut RerunSink,
    left_dir: &Path,
    right: &mut RerunSink,
    right_dir: &Path,
) {
    const RUN: &str = "00000000000000b2";
    let key = TrialKey {
        env_idx: 0,
        trial_idx: 0,
    };
    let info = TrialInfo::new(key, "cartpole".to_string(), 5);
    let suite = SuiteInfo::new(
        "both-suite".to_string(),
        vec!["cartpole".to_string()],
        1,
        None,
        RUN.to_string(),
    );
    let span = tracing::info_span!(
        target: "rlevo_harness",
        parent: None,
        "trial",
        run = RUN,
        env = "cartpole",
        env_idx = 0_usize,
        trial = 0_usize,
        seed = 5_u64,
    );
    let _in_trial = span.enter();
    for sink in [&mut *left, &mut *right] {
        sink.on_suite_start(&suite);
        sink.on_trial_start(&info);
        sink.on_metrics(&info, Boundary::TrainingEpisode { index: 0, step: 4 }, &[]);
    }
    tracing::info!(target: UPDATE_TARGET, step = 6_u64, "test/loss" = 2.5_f64, "update");
    for sink in [&mut *left, &mut *right] {
        sink.on_trial_end(&info, &TrialReport::new(key, "cartpole".to_string(), 5));
        assert_eq!(sink.unattributed_updates(), 0);
    }
    tracing::info!(target: UPDATE_TARGET, step = 8_u64, "test/loss" = 9.0_f64, "too late");
    for (sink, dir, label) in [(left, left_dir, "left"), (right, right_dir, "right")] {
        assert_eq!(sink.unattributed_updates(), 1, "{label}");
        sink.on_suite_end(&BenchmarkReport::new("both-suite".to_string(), 0));
        let rec = read(&trial_file(dir, RUN, label, 0, "cartpole", 0));
        let loss = path(&format!("/metrics/test/loss/{label}/cartpole/0"));
        assert_eq!(rec.series(&loss, "step"), [(6, 2.5)], "{label}");
    }
}

/// An update event outside any trial names no run, so it cannot be filed:
/// a sink recording at the time counts it, and it is not dropped unseen. An
/// event at another target is no update, whatever it carries.
fn an_update_outside_any_trial_is_counted(sink: &mut RerunSink) {
    sink.on_suite_start(&SuiteInfo::new(
        "stray-suite".to_string(),
        vec!["cartpole".to_string()],
        1,
        None,
        "00000000000005a1".to_string(),
    ));
    assert_eq!(sink.unattributed_updates(), 0);

    tracing::info!(target: UPDATE_TARGET, { "test/loss" = 1.0_f64 }, "no trial");
    tracing::info!(target: "somewhere::else", { "test/loss" = 1.0_f64 }, "not an update");

    assert_eq!(sink.unattributed_updates(), 1);

    // The count is of the run being recorded: the next run starts from none.
    sink.on_suite_start(&SuiteInfo::new(
        "next-suite".to_string(),
        vec!["cartpole".to_string()],
        1,
        None,
        "00000000000005a2".to_string(),
    ));
    assert_eq!(sink.unattributed_updates(), 0);
}
