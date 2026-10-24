//! Benchmark: sample-average `$\epsilon$`-greedy agent on the ten-armed bandit.
//!
//! Demonstrates the harness path (`Suite` → `Evaluator::run_suite` →
//! `BenchmarkReport`) running against [`TenArmedBandit`]. The environment is
//! registered as itself: `Suite::with_env` takes a constructor from a seed,
//! and `TenArmedBandit::with_seed` is one.
//!
//! Each trial trains its agent for `train_steps` pulls, then evaluates it.
//! The agent learns through the two hooks of the `Agent` trait it needs:
//! `on_transition` updates the pulled arm's mean reward, and `set_training`
//! turns exploration off for the evaluation episodes the report holds.
//!
//! **DQN integration is deferred.** A `FrozenDqnPolicy` adapter needs a
//! `HostRow` impl on `TenArmedBandit`'s observation, the row a DQN reads,
//! and the observation has none yet. The harness path is exercised here
//! with a non-DL stand-in; a follow-up release will add the DQN adapter once
//! that impl lands.
//!
//! # Running
//!
//! ```text
//! cargo run -p rlevo --example tabular_bandit
//! ```
//!
//! No feature flags are required.
//!
//! [`TenArmedBandit`]: rlevo::envs::classic::TenArmedBandit

use rand::Rng;
use rand_distr::{Distribution, Uniform};
use rlevo::envs::classic::{TenArmedBandit, TenArmedBanditAction, TenArmedBanditObservation};
use rlevo::harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo::harness::reporter::logging::LoggingReporter;
use rlevo::harness::suite::Suite;
use rlevo::prelude::{Agent, DiscreteAction, Transition};

/// Sample-average `$\epsilon$`-greedy agent: the mean reward of each arm,
/// learned from the pulls the harness shows it, with exploration only while
/// training.
struct SampleAverageAgent {
    epsilon: f64,
    q: [f64; 10],
    pulls: [u32; 10],
    training: bool,
}

impl SampleAverageAgent {
    fn new(epsilon: f64) -> Self {
        Self {
            epsilon,
            q: [0.0; 10],
            pulls: [0; 10],
            training: true,
        }
    }

    fn argmax(&self) -> usize {
        let mut best = 0;
        let mut best_v = f64::NEG_INFINITY;
        for (i, &v) in self.q.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best
    }
}

impl Agent<TenArmedBanditObservation, TenArmedBanditAction> for SampleAverageAgent {
    fn act(&mut self, _obs: &TenArmedBanditObservation, rng: &mut dyn Rng) -> TenArmedBanditAction {
        let unit = Uniform::new(0.0_f64, 1.0).unwrap();
        let arms = Uniform::new(0u32, 10).unwrap();
        let arm = if self.training && unit.sample(rng) < self.epsilon {
            arms.sample(rng) as usize
        } else {
            self.argmax()
        };
        TenArmedBanditAction::new(arm).expect("arm in 0..10")
    }

    fn on_transition(
        &mut self,
        t: Transition<&TenArmedBanditObservation, &TenArmedBanditAction>,
        _rng: &mut dyn Rng,
    ) {
        if !self.training {
            return;
        }
        let arm = t.action.to_index();
        self.pulls[arm] += 1;
        self.q[arm] += (f64::from(t.reward) - self.q[arm]) / f64::from(self.pulls[arm]);
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

fn main() {
    tracing_subscriber::fmt().with_target(false).init();

    let cfg = EvaluatorConfig {
        num_episodes: 20,
        train_steps: 5_000,
        num_trials_per_env: 5,
        max_steps: 500,
        base_seed: 2026,
        num_threads: None,
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: Some(0.0),
    };

    // `with_seed` routes the per-trial seed into the arm-mean RNG.
    let suite = Suite::new("ten-armed-bandit")
        .with_env("ten-armed-bandit-default", TenArmedBandit::with_seed);

    let evaluator = Evaluator::new(cfg);
    let mut reporter = LoggingReporter::new();
    let report = evaluator.run_suite(&suite, |_s| SampleAverageAgent::new(0.1), &mut reporter);

    println!("=== {} ===", report.suite_name);
    for trial in &report.trials {
        let ret = trial.scalars.get("return/mean").copied().unwrap_or(0.0);
        println!(
            "{:>16} trial={} seed={:>20} return/mean={:.3}",
            trial.env_name, trial.key.trial_idx, trial.trial_seed, ret
        );
    }
}
