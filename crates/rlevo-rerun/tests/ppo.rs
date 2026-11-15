//! Real PPO through the sink and its layer: the emit site in
//! `rlevo-reinforcement-learning` and the reader here, end to end.
//!
//! One test, because it installs the process's global subscriber.

mod common;

use std::collections::BTreeSet;

use burn::tensor::Device;
use rlevo_environments::classic::cartpole::{CartPole, CartPoleConfig, CartPoleObservation};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_harness::suite::Suite;
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfig;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::{MlpValue, MlpValueConfig};
use rlevo_rerun::{RerunConfig, RerunSink, UpdateLayer};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use common::{path, read, trial_file};

const ROLLOUT: usize = 64;
/// Four whole rollouts and a tail that never fills a fifth.
const TRAIN_STEPS: usize = 300;

fn agent() -> PpoAgent<CategoricalPolicyHead, MlpValue, CartPoleObservation, 2> {
    let device = Device::flex();
    let policy = CategoricalPolicyHeadConfig {
        obs_dim: 4,
        hidden: 16,
        num_actions: 2,
    }
    .try_init(&device)
    .expect("valid head config");
    let value = MlpValueConfig {
        obs_dim: 4,
        hidden: vec![16, 16],
    }
    .try_init(&device)
    .expect("valid value network config");
    let config = PpoTrainingConfig {
        num_envs: 1,
        num_steps: ROLLOUT,
        num_minibatches: 4,
        update_epochs: 2,
        ..Default::default()
    };
    PpoAgent::new(policy, value, config, device, TRAIN_STEPS / ROLLOUT).expect("valid config")
}

/// Row 9, with the workspace's own learner. Two PPO trials on two threads:
/// each trial's file has one row per update for every statistic PPO logs,
/// at the steps the updates happened, beside the counters `emit_metrics`
/// reports; a categorical head's log-std bounds are nowhere; nothing went
/// unfiled.
#[test]
fn ppo_updates_are_recorded_per_trial() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = RerunSink::new(RerunConfig::new(dir.path(), "ppo")).expect("a sink");
    tracing_subscriber::registry()
        .with(UpdateLayer::new())
        .init();

    let suite = Suite::new("ppo-suite").with_env("cartpole", |seed| {
        TimeLimit::new(
            CartPole::with_config(CartPoleConfig {
                seed,
                ..CartPoleConfig::default()
            })
            .expect("valid config"),
            500,
        )
    });
    let cfg = EvaluatorConfig {
        num_episodes: 1,
        train_steps: TRAIN_STEPS,
        num_trials_per_env: 2,
        max_steps: 500,
        num_threads: Some(2),
        ..EvaluatorConfig::default()
    };
    let report = Evaluator::new(cfg).run_suite(&suite, |_| agent(), &mut sink);

    assert_eq!(sink.unattributed_updates(), 0);
    assert_eq!(report.trials.len(), 2);

    let per_update = [
        "approx_kl",
        "clip_frac",
        "entropy",
        "epochs_run",
        "explained_variance",
        "learning_rate",
        "old_approx_kl",
        "policy_loss",
        "value_loss",
    ];
    let mut losses = Vec::new();
    for trial in &report.trials {
        assert!(!trial.errored, "{:?}", trial.error_message);
        let idx = trial.key.trial_idx;
        let run_id = report.run_id.as_deref().expect("a run has an ID");
        let rec = read(&trial_file(dir.path(), run_id, "ppo", 0, "cartpole", idx));
        let of = |name: &str| path(&format!("/metrics/rl/{name}/ppo/cartpole/{idx}"));

        for name in per_update {
            let rows = rec.series(&of(name), "update");
            let updates: Vec<i64> = rows.iter().map(|(update, _)| *update).collect();
            assert_eq!(updates, [0, 1, 2, 3], "rl/{name}");
            let steps: Vec<i64> = rec
                .series(&of(name), "step")
                .into_iter()
                .map(|(step, _)| step)
                .collect();
            assert_eq!(steps, [64, 128, 192, 256], "rl/{name}");
            assert!(rows.iter().all(|(_, value)| value.is_finite()), "rl/{name}");
        }

        // The agent's own count of its updates, read at the last boundary,
        // is the number of rows the layer filed.
        let iterations = rec.series(&of("iterations"), "step");
        assert_eq!(iterations.last().map(|(_, count)| *count), Some(4.0));

        // Everything under `rl/` in this file, by metric name.
        let prefix = "/metrics/rl/";
        let recorded: BTreeSet<String> = rec
            .entities()
            .iter()
            .filter_map(|entity| entity.strip_prefix(prefix))
            .filter_map(|rest| rest.split('/').next())
            .map(str::to_string)
            .collect();
        let mut expected: BTreeSet<String> = per_update.iter().map(ToString::to_string).collect();
        expected.extend(["env_steps", "iterations", "skipped_updates"].map(String::from));
        assert_eq!(
            recorded, expected,
            "a categorical head has no log-std bounds"
        );

        losses.push(rec.series(&of("policy_loss"), "update"));
    }
    assert_ne!(
        losses[0], losses[1],
        "two trials of two seeds do not share a loss curve"
    );
}
