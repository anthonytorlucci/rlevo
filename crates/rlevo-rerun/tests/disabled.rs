//! rerun switched off by the environment: the sink records nothing and says
//! nothing was recorded.
//!
//! `RERUN=off` makes rerun's file sink succeed and write nothing. The
//! variable is read from the process's environment, so the check runs in a
//! child process: this test runs its own binary again with the variable set.

use std::process::Command;

use rlevo_core::environment::EpisodeStatus;
use rlevo_harness::report::{BenchmarkReport, EpisodeSummary, Phase, TrialReport};
use rlevo_harness::reporter::{Boundary, Reporter};
use rlevo_harness::suite::{SuiteInfo, TrialInfo, TrialKey};
use rlevo_rerun::{RerunConfig, RerunSink};

/// Set in the child, which does the checking.
const CHILD: &str = "RLEVO_RERUN_DISABLED_CHILD";

#[test]
fn a_disabled_rerun_records_nothing_and_reports_nothing() {
    if std::env::var_os(CHILD).is_none() {
        let child = Command::new(std::env::current_exe().expect("this test's binary"))
            .args([
                "--exact",
                "a_disabled_rerun_records_nothing_and_reports_nothing",
            ])
            .env("RERUN", "off")
            .env(CHILD, "1")
            .output()
            .expect("the test binary runs");
        assert!(
            child.status.success(),
            "with RERUN=off:\n{}\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let mut sink = RerunSink::new(RerunConfig::new(dir.path(), "ppo")).expect("a sink");
    let key = TrialKey {
        env_idx: 0,
        trial_idx: 0,
    };
    let trial = TrialInfo::new(key, "cartpole".to_string(), 1);
    sink.on_suite_start(&SuiteInfo::new(
        "suite".to_string(),
        vec!["cartpole".to_string()],
        1,
        None,
        "00000000000000d0".to_string(),
    ));
    sink.on_trial_start(&trial);
    sink.on_episode(
        &trial,
        &EpisodeSummary::new(Phase::Evaluation, 0, 1.0, 3, EpisodeStatus::Terminated),
    );
    sink.on_metrics(
        &trial,
        Boundary::EvaluationEpisode { index: 0, step: 3 },
        &[],
    );
    assert_eq!(sink.recording(&trial), None, "no file is being written");
    sink.on_trial_end(&trial, &TrialReport::new(key, "cartpole".to_string(), 1));

    // A trial that ends before its first episode has not tried to open a
    // file; it must not be promised one either.
    let key = TrialKey {
        env_idx: 0,
        trial_idx: 1,
    };
    let unstarted = TrialInfo::new(key, "cartpole".to_string(), 2);
    sink.on_trial_start(&unstarted);
    assert_eq!(sink.recording(&unstarted), None);
    sink.on_trial_end(
        &unstarted,
        &TrialReport::new(key, "cartpole".to_string(), 2),
    );

    sink.on_suite_end(&BenchmarkReport::new("suite".to_string(), 0));

    assert_eq!(sink.files(), Vec::<std::path::PathBuf>::new());
    assert_eq!(sink.viewer_instructions(), None);
    let written: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
    assert!(written.is_empty(), "{written:?}");
}
