//! Suite evaluator with rayon trial-level parallelism and checkpoint resume.
//!
//! The core entry point is [`Evaluator::run_suite`], which drives every
//! `(env, seed)` trial in a [`Suite`] through a [`BenchableAgent`], collects
//! [`TrialReport`]s, and delivers them to a [`Reporter`].
//!
//! Parallelism is controlled via [`EvaluatorConfig::num_threads`] (`None` reuses
//! the global rayon pool). Panics inside a trial are caught with
//! `catch_unwind` and recorded as errored trials rather than aborting the run.
//! [`EvaluatorConfig::checkpoint_dir`] enables crash-recovery: the partial
//! [`BenchmarkReport`] is atomically written after every trial so a resumed run
//! can skip already-finished keys.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use rlevo_core::evaluation::BenchEnv;
use rlevo_core::fitness::BenchableAgent;
use rlevo_core::util::seed::SeedStream;

use crate::checkpoint;
use crate::metrics::core::core_metrics;
use crate::report::{BenchmarkReport, EpisodeSummary, TrialReport};
use crate::reporter::Reporter;
use crate::suite::{Suite, SuiteInfo, TrialInfo, TrialKey};

/// Parameters governing how the evaluator runs a suite.
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Number of episodes to run per trial.
    pub num_episodes: usize,
    /// Number of independent seeds to evaluate per environment.
    pub num_trials_per_env: usize,
    /// Hard step ceiling per episode; an episode that has not terminated by
    /// `max_steps` is truncated with whatever return accumulated so far.
    pub max_steps: usize,
    /// Root seed from which all trial and episode seeds are derived
    /// via [`SeedStream`].
    pub base_seed: u64,
    /// Fixed rayon thread count; `None` uses the global pool.
    ///
    /// Recording runs require `Some(1)`: a single `RecordWriter` is
    /// single-stream (it holds one open episode file), so a recording sink
    /// driven from multiple worker threads is rejected at runtime with
    /// `RecordError::ConcurrentUse` rather than being allowed to truncate
    /// the in-flight episode. See the `record` module's `RecordSink` docs
    /// for the full single-stream constraint.
    pub num_threads: Option<usize>,
    /// If set, the partial [`BenchmarkReport`] is checkpointed here after every
    /// trial so a crashed run can be resumed.
    pub checkpoint_dir: Option<PathBuf>,
    /// Abort the suite after the first errored trial.
    pub fail_fast: bool,
    /// When `Some(threshold)`, a `success_rate` metric is emitted per trial
    /// counting the fraction of episodes whose return ≥ `threshold`.
    pub success_threshold: Option<f64>,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            num_episodes: 100,
            num_trials_per_env: 1,
            max_steps: 1_000,
            base_seed: 0,
            num_threads: None,
            checkpoint_dir: None,
            fail_fast: false,
            success_threshold: None,
        }
    }
}

/// Runs a benchmark suite according to an [`EvaluatorConfig`].
#[derive(Debug, Clone)]
pub struct Evaluator {
    /// Configuration applied to every suite run.
    pub cfg: EvaluatorConfig,
}

impl Evaluator {
    /// Creates an evaluator from the supplied configuration.
    #[must_use]
    pub const fn new(cfg: EvaluatorConfig) -> Self {
        Self { cfg }
    }

    /// Run a suite of `BenchEnv`s against a `BenchableAgent`.
    ///
    /// `agent_factory` is called once per trial with the derived agent seed;
    /// it is the implementor's contract to initialize the agent deterministically.
    ///
    /// Returns a [`BenchmarkReport`] containing every trial result. The function
    /// never returns an `Err`: environment panics and step/reset errors are caught
    /// and folded into the corresponding [`TrialReport`] via `errored`/`error_message`
    /// rather than aborting the run.
    ///
    /// # Panics
    /// Panics if the rayon thread pool cannot be built.
    #[allow(clippy::too_many_lines)]
    pub fn run_suite<E, A, R, FA>(
        &self,
        suite: &Suite<E>,
        agent_factory: FA,
        reporter: &mut R,
    ) -> BenchmarkReport
    where
        E: BenchEnv + Send,
        A: BenchableAgent<E::Observation, E::Action> + Send,
        R: Reporter,
        FA: Fn(u64) -> A + Sync + Send,
    {
        let cfg = &self.cfg;
        let seeds = SeedStream::new(cfg.base_seed);

        let suite_info = SuiteInfo {
            name: suite.name.clone(),
            env_names: suite.envs.iter().map(|(n, _)| n.clone()).collect(),
            num_trials_per_env: cfg.num_trials_per_env,
            success_threshold: cfg.success_threshold,
        };

        // Load checkpoint (if any) and start report.
        let mut report = cfg
            .checkpoint_dir
            .as_ref()
            .and_then(|dir| {
                let path = checkpoint::checkpoint_path(dir, &suite.name);
                checkpoint::load(&path).ok().flatten()
            })
            .unwrap_or_else(|| BenchmarkReport::new(suite.name.clone(), cfg.base_seed));
        let skip = checkpoint::completed_keys(&report);

        reporter.on_suite_start(&suite_info);

        // Enumerate trial keys, filtering already-completed.
        let mut pending: Vec<TrialKey> = Vec::new();
        for (env_idx, _) in suite.envs.iter().enumerate() {
            for trial_idx in 0..cfg.num_trials_per_env {
                let key = TrialKey { env_idx, trial_idx };
                if !skip.contains(&key) {
                    pending.push(key);
                }
            }
        }

        let reporter_lock: Mutex<&mut R> = Mutex::new(reporter);
        let report_lock: Mutex<&mut BenchmarkReport> = Mutex::new(&mut report);
        let aborted = AtomicBool::new(false);

        let pool = cfg.num_threads.map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("build rayon pool")
        });

        let run_all = || {
            pending.par_iter().for_each(|&key| {
                if aborted.load(Ordering::Relaxed) {
                    return;
                }

                let (env_name, factory) = &suite.envs[key.env_idx];
                let trial_seed = seeds.trial_seed(key.env_idx, key.trial_idx);
                let env_seed = seeds.env_seed(trial_seed);
                let agent_seed = seeds.agent_seed(trial_seed);

                let info = TrialInfo {
                    key,
                    env_name: env_name.clone(),
                    trial_seed,
                };
                {
                    let mut r = reporter_lock.lock().unwrap();
                    r.on_trial_start(&info);
                }

                let trial_report = catch_unwind(AssertUnwindSafe(|| {
                    run_one_trial::<E, A, R>(
                        cfg,
                        &info,
                        (factory.as_ref())(env_seed),
                        agent_factory(agent_seed),
                        agent_seed,
                        &reporter_lock,
                    )
                }));

                let tr = match trial_report {
                    Ok(tr) => tr,
                    Err(payload) => {
                        let msg = panic_message(&*payload);
                        let mut err = TrialReport::new(key, env_name.clone(), trial_seed);
                        err.errored = true;
                        err.error_message = Some(msg);
                        err
                    }
                };

                if tr.errored && cfg.fail_fast {
                    aborted.store(true, Ordering::Relaxed);
                }

                {
                    let mut r = reporter_lock.lock().unwrap();
                    r.on_trial_end(&info, &tr);
                }

                let mut rep = report_lock.lock().unwrap();
                rep.trials.push(tr);
                if let Some(dir) = &cfg.checkpoint_dir {
                    let path = checkpoint::checkpoint_path(dir, &suite.name);
                    if let Err(e) = checkpoint::save(&path, &rep) {
                        tracing::warn!(
                            target: "rlevo_benchmarks",
                            path = %path.display(),
                            error = %e,
                            "checkpoint save failed"
                        );
                    }
                }
            });
        };

        if let Some(p) = pool {
            p.install(run_all);
        } else {
            run_all();
        }

        report
            .trials
            .sort_by_key(|t| (t.key.env_idx, t.key.trial_idx));
        report.finalize();
        reporter.on_suite_end(&report);

        if let Some(dir) = &cfg.checkpoint_dir {
            let path = checkpoint::checkpoint_path(dir, &suite.name);
            if let Err(e) = checkpoint::save(&path, &report) {
                tracing::warn!(
                    target: "rlevo_benchmarks",
                    path = %path.display(),
                    error = %e,
                    "checkpoint save failed"
                );
            }
        }

        report
    }
}

fn run_one_trial<E, A, R>(
    cfg: &EvaluatorConfig,
    info: &TrialInfo,
    mut env: E,
    mut agent: A,
    agent_seed: u64,
    reporter_lock: &Mutex<&mut R>,
) -> TrialReport
where
    E: BenchEnv,
    A: BenchableAgent<E::Observation, E::Action>,
    R: Reporter,
{
    let mut rng = StdRng::seed_from_u64(agent_seed);
    let mut report = TrialReport::new(info.key, info.env_name.clone(), info.trial_seed);
    let mut returns: Vec<f64> = Vec::with_capacity(cfg.num_episodes);
    let mut lengths: Vec<usize> = Vec::with_capacity(cfg.num_episodes);

    let start = Instant::now();

    'episodes: for episode_idx in 0..cfg.num_episodes {
        let mut obs = match env.reset() {
            Ok(o) => o,
            Err(e) => {
                report.errored = true;
                report.error_message = Some(e.to_string());
                break 'episodes;
            }
        };
        let mut total_reward = 0.0;
        let mut length = 0;

        for _ in 0..cfg.max_steps {
            let action = agent.act(&obs, &mut rng);
            let step = match env.step(action) {
                Ok(s) => s,
                Err(e) => {
                    report.errored = true;
                    report.error_message = Some(e.to_string());
                    break 'episodes;
                }
            };
            total_reward += step.reward;
            length += 1;
            obs = step.observation;
            if step.done {
                break;
            }
        }

        let rec = EpisodeSummary {
            episode_idx,
            return_value: total_reward,
            length,
        };
        returns.push(total_reward);
        lengths.push(length);
        report.episodes.push(rec.clone());
        {
            let mut r = reporter_lock.lock().unwrap();
            r.on_episode_end(info, &rec);
        }
    }

    let wall = start.elapsed().as_secs_f64();
    report.absorb_metrics(core_metrics(
        &returns,
        &lengths,
        wall,
        cfg.success_threshold,
    ));
    report.absorb_metrics(agent.emit_metrics());
    report
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(test)]
mod tests {
    //! Stub-based unit tests for `Evaluator::run_suite`.
    //!
    //! A minimal `StubEnv` and `StubAgent` replace real environments so the
    //! evaluator logic is exercised without pulling in `rlevo-environments`
    //! as a dev-dependency (which would drag in the full physics crate tree).

    use std::sync::{Arc, Mutex};

    use rand::Rng;

    use rlevo_core::environment::EnvironmentError;
    use rlevo_core::evaluation::{BenchEnv, BenchError, BenchStep};
    use rlevo_core::fitness::BenchableAgent;

    use super::{Evaluator, EvaluatorConfig};
    use crate::report::{BenchmarkReport, EpisodeSummary, TrialReport};
    use crate::reporter::Reporter;
    use crate::suite::{Suite, SuiteInfo, TrialInfo};

    // ── Stub env ─────────────────────────────────────────────────────────────

    struct StubEnv {
        /// Return `done = true` once this many steps have been taken.
        steps_until_done: usize,
        reset_fails: bool,
        step_fails: bool,
        steps_taken: usize,
    }

    impl StubEnv {
        fn new(steps_until_done: usize) -> Self {
            Self {
                steps_until_done,
                reset_fails: false,
                step_fails: false,
                steps_taken: 0,
            }
        }
    }

    impl BenchEnv for StubEnv {
        type Observation = u32;
        type Action = ();

        fn reset(&mut self) -> Result<Self::Observation, BenchError> {
            if self.reset_fails {
                return Err(BenchError::Reset(EnvironmentError::InvalidAction(
                    "stub reset fail".into(),
                )));
            }
            self.steps_taken = 0;
            Ok(0)
        }

        fn step(&mut self, _action: ()) -> Result<BenchStep<Self::Observation>, BenchError> {
            if self.step_fails {
                return Err(BenchError::Step(EnvironmentError::InvalidAction(
                    "stub step fail".into(),
                )));
            }
            self.steps_taken += 1;
            let done = self.steps_taken >= self.steps_until_done;
            Ok(BenchStep {
                observation: self.steps_taken as u32,
                reward: 1.0,
                done,
            })
        }
    }

    // ── Stub agent ───────────────────────────────────────────────────────────

    struct StubAgent;

    impl BenchableAgent<u32, ()> for StubAgent {
        fn act(&mut self, _obs: &u32, _rng: &mut dyn Rng) {}
    }

    // ── Spy reporter ─────────────────────────────────────────────────────────

    #[derive(Debug, Clone, PartialEq)]
    enum Event {
        SuiteStart,
        TrialStart,
        EpisodeEnd,
        TrialEnd,
        SuiteEnd,
    }

    struct SpyReporter {
        log: Arc<Mutex<Vec<Event>>>,
    }

    impl SpyReporter {
        fn new() -> (Self, Arc<Mutex<Vec<Event>>>) {
            let log = Arc::new(Mutex::new(Vec::new()));
            (Self { log: Arc::clone(&log) }, log)
        }
    }

    impl Reporter for SpyReporter {
        fn on_suite_start(&mut self, _: &SuiteInfo) {
            self.log.lock().unwrap().push(Event::SuiteStart);
        }
        fn on_trial_start(&mut self, _: &TrialInfo) {
            self.log.lock().unwrap().push(Event::TrialStart);
        }
        fn on_episode_end(&mut self, _: &TrialInfo, _: &EpisodeSummary) {
            self.log.lock().unwrap().push(Event::EpisodeEnd);
        }
        fn on_trial_end(&mut self, _: &TrialInfo, _: &TrialReport) {
            self.log.lock().unwrap().push(Event::TrialEnd);
        }
        fn on_suite_end(&mut self, _: &BenchmarkReport) {
            self.log.lock().unwrap().push(Event::SuiteEnd);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn single_thread_cfg(num_episodes: usize, max_steps: usize) -> EvaluatorConfig {
        EvaluatorConfig {
            num_episodes,
            num_trials_per_env: 1,
            max_steps,
            base_seed: 42,
            num_threads: Some(1),
            checkpoint_dir: None,
            fail_fast: false,
            success_threshold: None,
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn one_trial_per_env_in_report() {
        let cfg = single_thread_cfg(1, 10);
        let suite = Suite::new("s", cfg.clone())
            .with_env("e1", |_| StubEnv::new(3))
            .with_env("e2", |_| StubEnv::new(3));

        let (mut reporter, _log) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        assert_eq!(report.trials.len(), 2);
    }

    #[test]
    fn report_is_finalized_after_run() {
        let cfg = single_thread_cfg(1, 5);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| StubEnv::new(2));

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        assert!(!report.in_progress);
    }

    #[test]
    fn reporter_callback_order_suite_wraps_trials() {
        let cfg = single_thread_cfg(2, 10);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| StubEnv::new(3));

        let (mut reporter, log) = SpyReporter::new();
        Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let events = log.lock().unwrap().clone();
        assert_eq!(events.first().unwrap(), &Event::SuiteStart);
        assert_eq!(events.last().unwrap(), &Event::SuiteEnd);

        let trial_start = events.iter().position(|e| e == &Event::TrialStart).unwrap();
        let trial_end = events.iter().rposition(|e| e == &Event::TrialEnd).unwrap();
        assert!(trial_start < trial_end);
    }

    #[test]
    fn on_episode_end_fires_once_per_episode() {
        let cfg = single_thread_cfg(3, 20);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| StubEnv::new(5));

        let (mut reporter, log) = SpyReporter::new();
        Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let episode_count = log
            .lock()
            .unwrap()
            .iter()
            .filter(|e| *e == &Event::EpisodeEnd)
            .count();
        assert_eq!(episode_count, 3);
    }

    #[test]
    fn episode_return_equals_steps_taken() {
        // StubEnv rewards 1.0 per step and signals done after `steps_until_done`.
        let cfg = single_thread_cfg(4, 100);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| StubEnv::new(7));

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let trial = &report.trials[0];
        assert_eq!(trial.episodes.len(), 4);
        for ep in &trial.episodes {
            assert!((ep.return_value - 7.0).abs() < f64::EPSILON);
            assert_eq!(ep.length, 7);
        }
    }

    #[test]
    fn max_steps_truncates_non_terminating_episode() {
        let cfg = EvaluatorConfig {
            num_episodes: 1,
            max_steps: 4,
            num_trials_per_env: 1,
            base_seed: 0,
            num_threads: Some(1),
            checkpoint_dir: None,
            fail_fast: false,
            success_threshold: None,
        };
        // done triggers at step 1000 — the evaluator must stop at max_steps=4.
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| StubEnv::new(1000));

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let ep = &report.trials[0].episodes[0];
        assert_eq!(ep.length, 4);
        assert!((ep.return_value - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn reset_error_marks_trial_errored() {
        let cfg = single_thread_cfg(5, 10);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| {
            let mut env = StubEnv::new(3);
            env.reset_fails = true;
            env
        });

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        assert!(report.trials[0].errored);
        assert!(report.trials[0].error_message.is_some());
    }

    #[test]
    fn step_error_marks_trial_errored() {
        let cfg = single_thread_cfg(5, 10);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| {
            let mut env = StubEnv::new(3);
            env.step_fails = true;
            env
        });

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        assert!(report.trials[0].errored);
        assert!(report.trials[0].error_message.is_some());
    }

    #[test]
    fn panic_in_env_is_caught_not_propagated() {
        struct PanickingEnv;
        impl BenchEnv for PanickingEnv {
            type Observation = u32;
            type Action = ();
            fn reset(&mut self) -> Result<u32, BenchError> {
                Ok(0)
            }
            fn step(&mut self, _: ()) -> Result<BenchStep<u32>, BenchError> {
                panic!("deliberate evaluator test panic");
            }
        }

        let cfg = single_thread_cfg(1, 10);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| PanickingEnv);

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        assert!(report.trials[0].errored);
        assert!(report.trials[0]
            .error_message
            .as_deref()
            .unwrap_or("")
            .contains("deliberate evaluator test panic"));
    }

    #[test]
    fn trials_are_sorted_by_env_then_trial_index() {
        let cfg = EvaluatorConfig {
            num_episodes: 1,
            num_trials_per_env: 2,
            max_steps: 5,
            base_seed: 0,
            num_threads: Some(1),
            checkpoint_dir: None,
            fail_fast: false,
            success_threshold: None,
        };
        let suite = Suite::new("s", cfg.clone())
            .with_env("e0", |_| StubEnv::new(3))
            .with_env("e1", |_| StubEnv::new(3));

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let keys: Vec<(usize, usize)> = report
            .trials
            .iter()
            .map(|t| (t.key.env_idx, t.key.trial_idx))
            .collect();
        let mut expected = keys.clone();
        expected.sort();
        assert_eq!(keys, expected);
    }

    #[test]
    fn success_threshold_lands_in_trial_metrics() {
        let cfg = EvaluatorConfig {
            num_episodes: 5,
            num_trials_per_env: 1,
            max_steps: 10,
            base_seed: 0,
            num_threads: Some(1),
            checkpoint_dir: None,
            fail_fast: false,
            success_threshold: Some(3.0), // StubEnv returns 5.0 per episode
        };
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| StubEnv::new(5));

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        // core_metrics emits "success_rate" when a threshold is set.
        assert!(report.trials[0].scalars.contains_key("success_rate"));
    }
}
