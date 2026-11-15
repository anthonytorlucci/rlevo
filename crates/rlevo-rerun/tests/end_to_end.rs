//! The sink under a real `Evaluator` run: parallel trials, the report's
//! recording fields, and a resume.

mod common;

use rand::Rng;
use rlevo_core::agent::Agent;
use rlevo_core::fitness::Metric;
use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_harness::report::{BenchmarkReport, EpisodeSummary, Phase, TrialReport};
use rlevo_harness::reporter::{Boundary, Recording, Reporter};
use rlevo_harness::suite::{Suite, SuiteInfo, TrialInfo, TrialKey};
use rlevo_rerun::{RerunConfig, RerunSink};

use common::{path, read};

/// Pushes left and right in turn, and counts its actions: one per
/// environment step, so its counter equals the trial's step count.
struct Alternating {
    acts: u64,
}

impl Agent<CartPoleObservation, CartPoleAction> for Alternating {
    fn act(&mut self, _obs: &CartPoleObservation, _rng: &mut dyn Rng) -> CartPoleAction {
        self.acts += 1;
        if self.acts.is_multiple_of(2) {
            CartPoleAction::Left
        } else {
            CartPoleAction::Right
        }
    }

    fn emit_metrics(&self) -> Vec<Metric> {
        vec![Metric::Counter {
            name: "rl/acts".to_string(),
            count: self.acts,
        }]
    }
}

fn suite() -> Suite<CartPole> {
    Suite::new("cartpole-suite").with_env("cartpole", |seed| {
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        })
        .expect("valid config")
    })
}

fn config(trials: usize) -> EvaluatorConfig {
    EvaluatorConfig {
        num_episodes: 2,
        train_steps: 30,
        num_trials_per_env: trials,
        max_steps: 12,
        num_threads: Some(2),
        ..EvaluatorConfig::default()
    }
}

/// Hands every call on to the sink and keeps each episode the harness
/// delivered, in order: the report keeps evaluation episodes only, so this
/// is what the files are checked against.
struct Tee<'a> {
    sink: &'a mut RerunSink,
    delivered: Vec<(TrialKey, EpisodeSummary)>,
}

impl Reporter for Tee<'_> {
    fn on_suite_start(&mut self, suite: &SuiteInfo) {
        self.sink.on_suite_start(suite);
    }
    fn on_trial_start(&mut self, trial: &TrialInfo) {
        self.sink.on_trial_start(trial);
    }
    fn on_episode(&mut self, trial: &TrialInfo, ep: &EpisodeSummary) {
        self.delivered.push((trial.key, ep.clone()));
        self.sink.on_episode(trial, ep);
    }
    fn on_metrics(&mut self, trial: &TrialInfo, at: Boundary, metrics: &[Metric]) {
        self.sink.on_metrics(trial, at, metrics);
    }
    fn wants_metrics(&self) -> bool {
        self.sink.wants_metrics()
    }
    fn recording(&self, trial: &TrialInfo) -> Option<Recording> {
        self.sink.recording(trial)
    }
    fn on_trial_end(&mut self, trial: &TrialInfo, report: &TrialReport) {
        self.sink.on_trial_end(trial, report);
    }
    fn on_suite_end(&mut self, report: &BenchmarkReport) {
        self.sink.on_suite_end(report);
    }
}

/// A run's report, its sink, and the episodes the harness delivered.
type Run = (BenchmarkReport, RerunSink, Vec<(TrialKey, EpisodeSummary)>);

fn run(cfg: EvaluatorConfig, dir: &std::path::Path) -> Run {
    let mut sink = RerunSink::new(RerunConfig::new(dir, "alt")).expect("the directory exists");
    let mut tee = Tee {
        sink: &mut sink,
        delivered: Vec::new(),
    };
    let report = Evaluator::new(cfg).run_suite(&suite(), |_| Alternating { acts: 0 }, &mut tee);
    let delivered = tee.delivered;
    (report, sink, delivered)
}

/// The episodes of one phase, as `(index, return)`.
fn episodes(all: &[&EpisodeSummary], phase: Phase) -> Vec<(i64, f64)> {
    all.iter()
        .filter(|e| e.phase == phase)
        .map(|e| (i64::try_from(e.episode_idx).unwrap(), e.return_value))
        .collect()
}

/// Three trials on two threads. Each trial's file is named in the report,
/// holds exactly the episodes the harness delivered for that trial, and stamps
/// them with the step counts the harness counted; all share the run's ID.
#[test]
fn a_parallel_run_records_each_trial_in_its_own_file() {
    let dir = tempfile::tempdir().unwrap();
    let (report, sink, delivered) = run(config(3), dir.path());

    assert_eq!(report.trials.len(), 3);
    let run_id = report.run_id.clone().expect("a run has an ID");
    let mut files: Vec<_> = report
        .trials
        .iter()
        .map(|t| t.recording_path.clone().expect("each trial is recorded"))
        .collect();
    files.sort();
    files.dedup();
    assert_eq!(files.len(), 3, "one file per trial");
    assert_eq!(sink.files().len(), 3);

    for trial in &report.trials {
        let idx = trial.key.trial_idx;
        let rec = read(trial.recording_path.as_ref().unwrap());
        assert_eq!(rec.recording_id, run_id);
        assert_eq!(rec.application_id, "rlevo-episodic");
        assert_eq!(
            trial.recording_sdk_version, report.recording_sdk_version,
            "one build wrote every file"
        );

        let of = |name: &str| path(&format!("/metrics/{name}/alt/cartpole/{idx}"));
        let own: Vec<&EpisodeSummary> = delivered
            .iter()
            .filter(|(key, _)| *key == trial.key)
            .map(|(_, ep)| ep)
            .collect();
        let training = episodes(&own, Phase::Training);
        let evaluation = episodes(&own, Phase::Evaluation);
        assert!(!training.is_empty(), "the trial trained");
        assert_eq!(evaluation.len(), 2);
        assert_eq!(rec.series(&of("episode_return/train"), "episode"), training);
        assert_eq!(
            rec.series(&of("episode_return/eval"), "eval_episode"),
            evaluation
        );

        // The agent acts once per environment step, so its counter is the
        // trial's step count, kept by the agent and not by the harness or
        // the sink. Every value must sit at the step that counter names.
        // Summed episode lengths would not do: the training budget cuts a
        // last episode short, whose steps count and which is never
        // delivered as an episode.
        let ends: Vec<i64> = rec
            .series(&of("rl/acts"), "step")
            .into_iter()
            .map(|(step, _)| step)
            .collect();
        assert_eq!(ends.len(), own.len(), "one read per episode");
        let summed: i64 = own.iter().map(|e| i64::try_from(e.length).unwrap()).sum();
        assert!(
            *ends.last().unwrap() > summed,
            "this run has undelivered training steps, or the check above proves less"
        );
        let mut stamped: Vec<i64> = rec
            .series(&of("episode_return/train"), "step")
            .into_iter()
            .chain(rec.series(&of("episode_return/eval"), "step"))
            .map(|(step, _)| step)
            .collect();
        stamped.sort_unstable();
        assert_eq!(stamped, ends);
        #[allow(clippy::cast_precision_loss)]
        let acts: Vec<(i64, f64)> = ends.iter().map(|&s| (s, s as f64)).collect();
        assert_eq!(rec.series(&of("rl/acts"), "step"), acts);
    }
}

/// A run resumed from its checkpoint, into the same directory: the finished
/// trial's file is left as it was, the pending trial gets one, and both
/// carry the first run's recording ID.
#[test]
fn a_resumed_run_adds_to_the_recording_it_began() {
    let dir = tempfile::tempdir().unwrap();
    let checkpoints = tempfile::tempdir().unwrap();
    let cfg = |trials| EvaluatorConfig {
        checkpoint_dir: Some(checkpoints.path().to_path_buf()),
        ..config(trials)
    };

    let (first, _, _) = run(cfg(1), dir.path());
    let run_id = first.run_id.clone().expect("a run has an ID");
    let file_0 = first.trials[0].recording_path.clone().unwrap();
    let before = std::fs::read(&file_0).unwrap();

    let (second, sink, _) = run(cfg(2), dir.path());
    assert_eq!(second.run_id.as_deref(), Some(run_id.as_str()));
    assert_eq!(sink.files().len(), 1, "only the pending trial was written");
    assert_eq!(second.trials[0].recording_path.as_ref(), Some(&file_0));
    assert_eq!(
        std::fs::read(&file_0).unwrap(),
        before,
        "the finished trial's file is untouched"
    );
    let file_1 = second.trials[1].recording_path.clone().unwrap();
    assert_ne!(file_1, file_0);
    assert_eq!(read(&file_1).recording_id, run_id);
    assert_eq!(read(&file_0).recording_id, run_id);
}
