//! Benchmark: sample-average ε-greedy agent on the ten-armed bandit.
//!
//! Demonstrates the full harness path (`Suite` → `Evaluator::run_suite` →
//! `BenchmarkReport`) against a real environment from `rlevo-envs`.
//!
//! **DQN integration is deferred.** A `FrozenDqnPolicy` adapter needs
//! concrete backend selection and full `DiscreteAction`/`TensorConvertible`
//! wiring for `TenArmedBandit`, neither of which exists yet in
//! `rlevo-envs`. The acceptance path is demonstrated here with a non-DL
//! stand-in; a follow-up release will add the DQN adapter once the
//! underlying trait impls land.

use rand::Rng;
use rand_distr::{Distribution, Uniform};
use rlevo_benchmarks::agent::BenchableAgent;
use rlevo_benchmarks::env::{BenchEnv, BenchStep};
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_benchmarks::reporter::logging::LoggingReporter;
use rlevo_benchmarks::suite::Suite;
use rlevo_envs::classic::ten_armed_bandit::TenArmedBandit;

struct BanditEnv {
    inner: TenArmedBandit,
    horizon: usize,
    t: usize,
}

impl BanditEnv {
    fn new(seed: u64, horizon: usize) -> Self {
        Self {
            inner: TenArmedBandit::with_seed(seed),
            horizon,
            t: 0,
        }
    }
}

impl BenchEnv for BanditEnv {
    type Observation = ();
    type Action = usize;

    fn reset(&mut self) -> Self::Observation {
        self.inner.reset();
        self.t = 0;
    }

    fn step(&mut self, action: Self::Action) -> BenchStep<Self::Observation> {
        let reward = f64::from(self.inner.pull(action));
        self.t += 1;
        BenchStep {
            observation: (),
            reward,
            done: self.t >= self.horizon,
        }
    }
}

/// ε-greedy agent over a fixed Q prior.
///
/// Online learning from rewards would require a feedback channel that the
/// `BenchableAgent` trait intentionally omits (the harness benchmarks frozen
/// policies). A follow-up spec can introduce a learning-agent variant; for
/// this example the agent exercises exploration vs. exploitation over a
/// uniform zero prior.
struct SampleAverageAgent {
    epsilon: f64,
    q: [f64; 10],
}

impl SampleAverageAgent {
    fn new(epsilon: f64) -> Self {
        Self {
            epsilon,
            q: [0.0; 10],
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

impl BenchableAgent<(), usize> for SampleAverageAgent {
    fn act(&mut self, _obs: &(), rng: &mut dyn Rng) -> usize {
        let unit = Uniform::new(0.0_f64, 1.0).unwrap();
        let arms = Uniform::new(0u32, 10).unwrap();
        if unit.sample(rng) < self.epsilon {
            arms.sample(rng) as usize
        } else {
            self.argmax()
        }
    }
}

fn main() {
    tracing_subscriber::fmt().with_target(false).init();

    let cfg = EvaluatorConfig {
        num_episodes: 20,
        num_trials_per_env: 5,
        max_steps: 500,
        base_seed: 2026,
        num_threads: None,
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: Some(0.0),
    };

    let horizon = cfg.max_steps;
    let suite: Suite<BanditEnv> = Suite::new("ten-armed-bandit", cfg.clone())
        .with_env("bandit-seedA", move |seed| BanditEnv::new(seed, horizon))
        .with_env("bandit-seedB", move |seed| {
            BanditEnv::new(seed.wrapping_add(999), horizon)
        });

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
