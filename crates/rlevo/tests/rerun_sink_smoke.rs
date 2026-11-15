//! The `rerun` feature's path, as a consumer of the umbrella takes it: the
//! sink is reached as `rlevo::rerun`, a run through it leaves one recording
//! per trial, and the report names each.
//!
//! What is inside the files is `rlevo-rerun`'s own tests' business: reading
//! a recording back needs rerun's reader, and this crate may not depend on
//! rerun, in its tests either.

use rand::Rng;
use rlevo::core::agent::Agent;
use rlevo::envs::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo::harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo::harness::suite::Suite;
use rlevo::rerun::{RerunConfig, RerunSink};

struct PushLeft;

impl Agent<CartPoleObservation, CartPoleAction> for PushLeft {
    fn act(&mut self, _obs: &CartPoleObservation, _rng: &mut dyn Rng) -> CartPoleAction {
        CartPoleAction::Left
    }
}

#[test]
fn a_run_through_the_umbrellas_sink_leaves_a_recording_per_trial() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = RerunSink::new(RerunConfig::new(dir.path(), "left")).expect("a sink");
    let suite = Suite::new("smoke").with_env("cartpole", |seed| {
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        })
        .expect("valid config")
    });
    let cfg = EvaluatorConfig {
        num_episodes: 2,
        num_trials_per_env: 2,
        max_steps: 20,
        ..EvaluatorConfig::default()
    };

    let report = Evaluator::new(cfg).run_suite(&suite, |_| PushLeft, &mut sink);

    assert_eq!(report.trials.len(), 2);
    assert!(report.run_id.is_some());
    assert!(report.recording_sdk_version.is_some());
    for trial in &report.trials {
        let path = trial
            .recording_path
            .as_ref()
            .expect("the trial is recorded");
        assert!(path.starts_with(dir.path()), "{}", path.display());
        let bytes = std::fs::metadata(path).expect("the file exists").len();
        assert!(bytes > 0, "{} is empty", path.display());
    }
    assert_eq!(sink.files().len(), 2);
    assert!(
        sink.viewer_instructions()
            .is_some_and(|how| how.contains("rerun --bind 127.0.0.1"))
    );
}
