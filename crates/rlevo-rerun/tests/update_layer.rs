//! The update layer: per-update statistics read back from the trial files.
//!
//! The harness runs trials on worker threads, so the layer must be in the
//! process's global subscriber, and a process has one, set once. So this
//! file is one test: it builds every sink it needs, installs one subscriber
//! holding all their layers, and runs its cases one after another. Each
//! layer therefore sees every case's events, which is what a layer must get
//! right when two runs share a process.
//!
//! The layers are fixed before any event is logged. A reloadable layer
//! would not do: rerun's own code has `tracing` callsites, so a thread
//! inside the subscriber registers one and re-enters the reload lock, which
//! deadlocks against a test waiting to add a layer.

mod common;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use rand::Rng;
use rlevo_core::agent::{Agent, Transition};
use rlevo_core::fitness::UPDATE_TARGET;
use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_harness::report::{BenchmarkReport, TrialReport};
use rlevo_harness::reporter::Reporter;
use rlevo_harness::suite::{Suite, SuiteInfo, TrialInfo, TrialKey};
use rlevo_rerun::{RerunConfig, RerunSink};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use common::{path, read};

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
/// last: every layer counts it, so before it every count must be zero.
#[test]
fn update_events_are_filed_under_their_trials() {
    let dirs: Vec<tempfile::TempDir> = (0..4).map(|_| tempfile::tempdir().unwrap()).collect();
    let mut attrib = sink(&dirs[0], "attrib");
    let mut nostep = sink(&dirs[1], "nostep");
    let mut idle = sink(&dirs[2], "idle");
    let mut busy = sink(&dirs[3], "busy");
    tracing_subscriber::registry()
        .with(vec![
            attrib.layer(),
            nostep.layer(),
            idle.layer(),
            busy.layer(),
        ])
        .init();

    each_trials_updates_are_filed_under_that_trial(&mut attrib);
    an_update_with_no_step_is_stamped_with_the_last_boundary(&mut nostep);
    another_runs_updates_are_not_filed(&mut idle, &mut busy);
    an_update_outside_any_trial_is_counted(&[&attrib, &nostep, &idle, &busy]);
}

/// Row 9. Four trials on two threads, every agent logging updates: each
/// trial's file holds exactly one agent's values, every agent's values are
/// in some file, each on `update` and on the `step` the event named, and no
/// event went unfiled. A field that is `None` leaves no row, and a field
/// that is not a number is not a metric.
fn each_trials_updates_are_filed_under_that_trial(sink: &mut RerunSink) {
    let (report, logged) = run(sink, 4, true);
    let logged = logged.lock().unwrap().clone();
    assert_eq!(logged.len(), 4, "every agent logged updates");
    assert_eq!(sink.unattributed_updates(), 0);

    let mut found = Vec::new();
    for trial in &report.trials {
        let idx = trial.key.trial_idx;
        let rec = read(
            trial
                .recording_path
                .as_ref()
                .expect("the trial is recorded"),
        );
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
fn an_update_with_no_step_is_stamped_with_the_last_boundary(sink: &mut RerunSink) {
    let (report, _) = run(sink, 1, false);
    assert_eq!(sink.unattributed_updates(), 0);

    let rec = read(report.trials[0].recording_path.as_ref().unwrap());
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
fn another_runs_updates_are_not_filed(idle: &mut RerunSink, busy: &mut RerunSink) {
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
    let busy = read(report.trials[0].recording_path.as_ref().unwrap());
    assert_eq!(
        busy.series(&path("/metrics/test/loss/busy/cartpole/0"), "update")
            .len(),
        6
    );

    let recording = idle.recording(&info).expect("the idle trial has a file");
    idle.on_trial_end(&info, &TrialReport::new(key, "cartpole".to_string(), 7));
    let rec = read(&recording.path);
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

/// An update event outside any trial cannot be filed: every layer counts
/// it, so it is not dropped unseen. An event at another target is no
/// update, whatever it carries.
fn an_update_outside_any_trial_is_counted(sinks: &[&RerunSink]) {
    for sink in sinks {
        assert_eq!(sink.unattributed_updates(), 0);
    }

    tracing::info!(target: UPDATE_TARGET, { "test/loss" = 1.0_f64 }, "no trial");
    tracing::info!(target: "somewhere::else", { "test/loss" = 1.0_f64 }, "not an update");

    for sink in sinks {
        assert_eq!(sink.unattributed_updates(), 1);
    }
}
