//! A checkpoint written by 0.6.1 still loads, and resumes.
//!
//! `BenchmarkReport` is also the checkpoint, and `Evaluator::run_trials` loads
//! one with `.ok().flatten()`: a checkpoint that no longer parses restarts the
//! run from nothing, silently. Every field added since 0.6.1 therefore takes
//! `#[serde(default)]`, and this test holds them to it.
//!
//! `fixtures/checkpoint-0.6.1.json` was written by `checkpoint::save` with the
//! 0.6.1 types (two trials, one with two episodes and a metric of each kind,
//! one errored).

#![cfg(feature = "json")]

use std::path::Path;

use rlevo_core::environment::EpisodeStatus;
use rlevo_harness::checkpoint;
use rlevo_harness::report::Phase;
use rlevo_harness::suite::TrialKey;

fn fixture() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/checkpoint-0.6.1.json"
    ))
}

#[test]
fn a_0_6_1_checkpoint_loads_with_defaults_for_new_fields() {
    let report = checkpoint::load(fixture())
        .expect("the fixture is readable")
        .expect("the fixture parses with today's types");

    assert_eq!(report.suite_name, "fixture-suite");
    assert_eq!(report.recording_sdk_version, None);
    assert_eq!(report.run_id, None, "0.6.1 gave a run no ID");
    assert_eq!(report.provenance, None, "0.6.1 recorded no provenance");
    assert!(report.tasks.is_empty(), "0.6.1 recorded no tasks");
    assert_eq!(report.trials.len(), 2);

    let trial = &report.trials[0];
    assert_eq!(trial.env_id, "cartpole", "read from 0.6.1's `env_name`");
    assert_eq!(trial.task_hash, None);
    assert_eq!(trial.provenance, None);
    assert_eq!(trial.recording_path, None);
    assert_eq!(trial.recording_sdk_version, None);
    let episodes: Vec<(usize, Phase, Option<EpisodeStatus>)> = trial
        .episodes
        .iter()
        .map(|e| (e.episode_idx, e.phase, e.status))
        .collect();
    assert_eq!(
        episodes,
        [(0, Phase::Evaluation, None), (1, Phase::Evaluation, None)],
        "0.6.1 kept evaluation episodes only and recorded no status"
    );
    assert!(report.trials[1].errored);
}

#[test]
fn a_0_6_1_checkpoint_still_marks_its_trials_completed() {
    let report = checkpoint::load(fixture()).unwrap().unwrap();
    let done = checkpoint::completed_keys(&report);
    assert!(done.contains(&TrialKey {
        env_idx: 0,
        trial_idx: 0
    }));
}

#[test]
fn a_new_report_round_trips_its_new_fields() {
    let mut report = checkpoint::load(fixture()).unwrap().unwrap();
    report.run_id = Some("00000000000000a7".to_string());
    report.recording_sdk_version = Some("0.43.2".to_string());
    report.trials[0].recording_sdk_version = Some("0.43.1".to_string());
    report.trials[0].recording_path = Some("out/trial-0.rrd".into());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("checkpoint.json");
    checkpoint::save(&path, &report).unwrap();
    let back = checkpoint::load(&path).unwrap().unwrap();

    assert_eq!(back.run_id.as_deref(), Some("00000000000000a7"));
    assert_eq!(back.recording_sdk_version.as_deref(), Some("0.43.2"));
    assert_eq!(
        back.trials[0].recording_sdk_version.as_deref(),
        Some("0.43.1")
    );
    assert_eq!(
        back.trials[0].recording_path.as_deref(),
        Some(Path::new("out/trial-0.rrd"))
    );
}
