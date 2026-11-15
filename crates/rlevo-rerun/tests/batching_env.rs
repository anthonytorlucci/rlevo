//! rerun's batching variables still reach a trial's stream.
//!
//! The sink gives each stream a batching configuration of its own (ten
//! seconds between writes), and rerun applies its environment variables only
//! to a stream given none, so the sink applies them itself. The variable is
//! read from the process's environment, so the check runs in a child
//! process: this test runs its own binary again with the variable set.

mod common;

use std::process::Command;
use std::time::Duration;

use rlevo_harness::metrics::Metric;
use rlevo_harness::report::TrialReport;
use rlevo_harness::reporter::{Boundary, Reporter};
use rlevo_harness::suite::{SuiteInfo, TrialInfo, TrialKey};
use rlevo_rerun::{RerunConfig, RerunSink};

use common::{path, read_as_written, trial_file};

/// Set in the child, which does the checking.
const CHILD: &str = "RLEVO_RERUN_BATCHING_CHILD";
const RUN_ID: &str = "00000000000000b0";

#[test]
fn a_flush_interval_from_the_environment_is_used() {
    if std::env::var_os(CHILD).is_none() {
        let child = Command::new(std::env::current_exe().expect("this test's binary"))
            .args(["--exact", "a_flush_interval_from_the_environment_is_used"])
            .env("RERUN_FLUSH_TICK_SECS", "0.01")
            .env(CHILD, "1")
            .output()
            .expect("the test binary runs");
        assert!(
            child.status.success(),
            "with RERUN_FLUSH_TICK_SECS=0.01:\n{}\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        return;
    }
    if !rerun::decide_logging_enabled(true) {
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
        RUN_ID.to_string(),
    ));
    sink.on_trial_start(&trial);
    for (index, step) in [(0_usize, 3_u64), (1, 6), (2, 9)] {
        sink.on_metrics(
            &trial,
            Boundary::TrainingEpisode { index, step },
            &[Metric::Scalar {
                name: "rl/loss".to_string(),
                value: 1.0,
            }],
        );
        // Many times the interval asked for, and far below the sink's own.
        std::thread::sleep(Duration::from_millis(300));
    }
    sink.on_trial_end(&trial, &TrialReport::new(key, "cartpole".to_string(), 1));

    let rec = read_as_written(&trial_file(dir.path(), RUN_ID, "ppo", 0, "cartpole", 0));
    let loss = path("/metrics/rl/loss/ppo/cartpole/0");
    assert_eq!(rec.series(&loss, "step").len(), 3);
    assert_eq!(
        rec.chunks(&loss),
        3,
        "each row was written before the next was logged"
    );
}
