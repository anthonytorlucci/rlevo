//! Smoke tests for the evaluator run loop (suite → trial → report path).

use rand::Rng;
use rlevo_benchmarks::agent::BenchableAgent;
use rlevo_benchmarks::env::{BenchEnv, BenchError, BenchStep};
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_benchmarks::reporter::logging::LoggingReporter;
use rlevo_benchmarks::suite::Suite;

/// Deterministic toy env: observation is a step counter, reward = action as f64,
/// terminates after 3 steps. Seed influences a constant reward offset so we can
/// confirm env seeding reaches the evaluator.
#[derive(Debug, Clone)]
struct ToyEnv {
    offset: f64,
    t: usize,
}

impl ToyEnv {
    #[allow(clippy::cast_precision_loss)]
    fn with_seed(seed: u64) -> Self {
        Self {
            offset: (seed % 7) as f64,
            t: 0,
        }
    }
}

impl BenchEnv for ToyEnv {
    type Observation = usize;
    type Action = f64;

    fn reset(&mut self) -> Result<Self::Observation, BenchError> {
        self.t = 0;
        Ok(0)
    }

    fn step(&mut self, action: Self::Action) -> Result<BenchStep<Self::Observation>, BenchError> {
        self.t += 1;
        Ok(BenchStep {
            observation: self.t,
            reward: action + self.offset,
            done: self.t >= 3,
        })
    }
}

#[derive(Debug)]
struct ConstAgent {
    value: f64,
}

impl BenchableAgent<usize, f64> for ConstAgent {
    fn act(&mut self, _obs: &usize, _rng: &mut dyn Rng) -> f64 {
        self.value
    }
}

fn basic_cfg() -> EvaluatorConfig {
    EvaluatorConfig {
        num_episodes: 4,
        num_trials_per_env: 2,
        max_steps: 10,
        base_seed: 123,
        num_threads: Some(2),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: Some(5.0),
    }
}

#[test]
fn runs_suite_and_collects_trials() {
    let cfg = basic_cfg();
    let suite: Suite<ToyEnv> = Suite::new("toy", cfg.clone())
        .with_env("toy-a", ToyEnv::with_seed)
        .with_env("toy-b", ToyEnv::with_seed);
    let evaluator = Evaluator::new(cfg);
    let mut reporter = LoggingReporter::new();
    let report = evaluator.run_suite(&suite, |_seed| ConstAgent { value: 1.0 }, &mut reporter);

    assert_eq!(report.trials.len(), 4); // 2 envs * 2 trials
    assert!(report.trials.iter().all(|t| !t.errored));
    // Each trial ran 4 episodes * 3 steps = 12 steps.
    for t in &report.trials {
        assert_eq!(t.episodes.len(), 4);
        assert!(t.scalars.contains_key("return/mean"));
        assert!(t.scalars.contains_key("success_rate"));
    }
    assert!(!report.in_progress);
}

#[test]
fn determinism_same_seed_same_metrics() {
    let cfg = basic_cfg();
    let suite: Suite<ToyEnv> = Suite::new("toy", cfg.clone()).with_env("toy", ToyEnv::with_seed);
    let evaluator = Evaluator::new(cfg);

    let mut r1 = LoggingReporter::new();
    let mut r2 = LoggingReporter::new();
    let a = evaluator.run_suite(&suite, |_s| ConstAgent { value: 2.0 }, &mut r1);
    let b = evaluator.run_suite(&suite, |_s| ConstAgent { value: 2.0 }, &mut r2);

    assert_eq!(a.trials.len(), b.trials.len());
    for (ta, tb) in a.trials.iter().zip(&b.trials) {
        assert_eq!(ta.trial_seed, tb.trial_seed);
        assert_eq!(ta.key, tb.key);
        // Exclude wall-clock-dependent metrics from determinism checks.
        let skip = ["wall_clock_seconds", "throughput/steps_per_sec"];
        for (k, v) in &ta.scalars {
            if skip.contains(&k.as_str()) {
                continue;
            }
            assert_eq!(tb.scalars.get(k), Some(v), "mismatch at {k}");
        }
    }
}

#[derive(Debug)]
struct PanicAgent;

impl BenchableAgent<usize, f64> for PanicAgent {
    fn act(&mut self, _obs: &usize, _rng: &mut dyn Rng) -> f64 {
        panic!("intentional");
    }
}

#[test]
fn panicking_trial_does_not_abort_suite() {
    let cfg = EvaluatorConfig {
        num_trials_per_env: 1,
        num_threads: Some(1),
        ..basic_cfg()
    };
    let suite: Suite<ToyEnv> = Suite::new("panic", cfg.clone()).with_env("toy", ToyEnv::with_seed);
    let evaluator = Evaluator::new(cfg);
    let mut reporter = LoggingReporter::new();
    let report = evaluator.run_suite(&suite, |_s| PanicAgent, &mut reporter);

    assert_eq!(report.trials.len(), 1);
    assert!(report.trials[0].errored);
    assert!(
        report.trials[0]
            .error_message
            .as_deref()
            .unwrap()
            .contains("intentional")
    );
}
