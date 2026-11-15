//! A task fingerprint says whether two runs faced the same task, for a
//! shipped environment run through the harness: `FrozenLake`, registered with
//! `Suite::with_task`. The checks are the ones BYOA-1a's step 9 makes from
//! outside the workspace:
//!
//! - **same task, different seeds**: two runs that differ only in
//!   `base_seed` carry equal fingerprints, and so does every trial of one run;
//! - **different task**: a run that differs in one other config field
//!   carries a different one, for each field set here;
//! - **recompute**: the fingerprint the report carries equals the one
//!   `task_fingerprint` computes from the config the test built;
//! - **the id is not the task**: the same config registered under another id
//!   is the same task.

use rand::Rng;
use rlevo_core::agent::Agent;
use rlevo_environments::toy_text::frozen_lake::{
    FrozenLake, FrozenLakeAction, FrozenLakeConfig, FrozenLakeObservation, FrozenMapSpec,
    FrozenPreset, RewardSchedule,
};
use rlevo_harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_harness::report::BenchmarkReport;
use rlevo_harness::reporter::logging::LoggingReporter;
use rlevo_harness::suite::Suite;
use rlevo_harness::task::{TaskFingerprint, task_fingerprint};

const ENV_ID: &str = "frozen-lake";

/// Always moves right; the task, not the policy, is under test.
struct Right;

impl Agent<FrozenLakeObservation, FrozenLakeAction> for Right {
    fn act(&mut self, _obs: &FrozenLakeObservation, _rng: &mut dyn Rng) -> FrozenLakeAction {
        FrozenLakeAction::Right
    }
}

fn config() -> FrozenLakeConfig {
    FrozenLakeConfig {
        map: FrozenMapSpec::Preset(FrozenPreset::Four4x4),
        is_slippery: true,
        ..FrozenLakeConfig::default()
    }
}

fn run(config: FrozenLakeConfig, base_seed: u64) -> BenchmarkReport {
    run_as(ENV_ID, config, base_seed)
}

fn run_as(id: &str, config: FrozenLakeConfig, base_seed: u64) -> BenchmarkReport {
    let suite = Suite::new("fingerprint")
        .with_task(id, config, |c: &FrozenLakeConfig, seed| {
            FrozenLake::with_config(FrozenLakeConfig { seed, ..c.clone() })
                .expect("a preset map is valid")
        })
        .expect("FrozenLakeConfig serializes");
    let cfg = EvaluatorConfig {
        num_episodes: 2,
        num_trials_per_env: 2,
        max_steps: 20,
        base_seed,
        num_threads: Some(1),
        ..EvaluatorConfig::default()
    };
    Evaluator::new(cfg).run_suite(&suite, |_| Right, &mut LoggingReporter::new())
}

/// The report's one fingerprint; asserts every trial carries its hash.
fn fingerprint(report: &BenchmarkReport) -> TaskFingerprint {
    let task = report.tasks[0].clone().expect("the unit has a task");
    for t in &report.trials {
        assert_eq!(t.task_hash.as_deref(), Some(task.hash.as_str()), "{t:?}");
    }
    task
}

#[test]
fn same_task_different_seeds() {
    let a = fingerprint(&run(config(), 1));
    let b = fingerprint(&run(config(), 2));
    assert_eq!(a.hash, b.hash, "{a:?} against {b:?}");
}

#[test]
fn a_different_task_has_a_different_fingerprint() {
    let base = fingerprint(&run(config(), 1));
    let changed = [
        (
            "is_slippery",
            FrozenLakeConfig {
                is_slippery: false,
                ..config()
            },
        ),
        (
            "map",
            FrozenLakeConfig {
                map: FrozenMapSpec::Preset(FrozenPreset::Eight8x8),
                ..config()
            },
        ),
        (
            "success_rate",
            FrozenLakeConfig {
                success_rate: 0.5,
                ..config()
            },
        ),
        (
            "reward_schedule",
            FrozenLakeConfig {
                reward_schedule: RewardSchedule {
                    goal: 2.0,
                    ..RewardSchedule::default()
                },
                ..config()
            },
        ),
    ];
    for (field, changed) in changed {
        let other = fingerprint(&run(changed, 1));
        assert_ne!(base.hash, other.hash, "changing `{field}` kept {base:?}");
    }
}

#[test]
fn the_report_fingerprint_recomputes_from_the_config() {
    let report = run(config(), 1);
    assert_eq!(fingerprint(&report), task_fingerprint(&config()).unwrap());
}

#[test]
fn the_id_is_not_part_of_the_task() {
    assert_eq!(
        fingerprint(&run(config(), 1)),
        fingerprint(&run_as("FrozenLake-4x4", config(), 1))
    );
}
