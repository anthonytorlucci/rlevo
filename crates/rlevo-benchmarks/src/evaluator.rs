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

use rlevo_core::environment::{Environment, Snapshot};
use rlevo_core::evaluation::GenerationProbe;
use rlevo_core::fitness::{BenchableAgent, Metric, MetricsProvider};
use rlevo_core::reward::ScalarReward;
use rlevo_core::util::seed::SeedStream;

use crate::checkpoint;
use crate::metrics::core::{RETURN_NON_FINITE_STEPS, core_metrics};
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

    /// Run a suite of [`Environment`]s against a [`BenchableAgent`].
    ///
    /// `agent_factory` is called once per trial with the derived agent seed;
    /// it is the implementor's contract to initialize the agent deterministically.
    ///
    /// Returns a [`BenchmarkReport`] containing every trial result. The function
    /// never returns an `Err`: environment panics and step/reset errors are caught
    /// and folded into the corresponding [`TrialReport`] via `errored`/`error_message`
    /// rather than aborting the run.
    ///
    /// This is a thin wrapper: it builds one [`EpisodicTrial`] per key and hands
    /// the work to [`Evaluator::run_trials`], which owns the parallelism,
    /// panic capture, and checkpointing for every trial shape.
    ///
    /// # Panics
    /// Panics if the rayon thread pool cannot be built.
    pub fn run_suite<E, A, R, FA, const D: usize, const SD: usize, const AD: usize>(
        &self,
        suite: &Suite<E>,
        agent_factory: FA,
        reporter: &mut R,
    ) -> BenchmarkReport
    where
        E: Environment<D, SD, AD, RewardType = ScalarReward> + Send,
        A: BenchableAgent<E::ObservationType, E::ActionType> + Send,
        R: Reporter,
        FA: Fn(u64) -> A + Sync + Send,
    {
        let env_names: Vec<String> = suite.envs.iter().map(|(n, _)| n.clone()).collect();
        self.run_trials(
            &suite.name,
            &env_names,
            |key, env_seed, agent_seed| EpisodicTrial {
                env: (suite.envs[key.env_idx].1.as_ref())(env_seed),
                agent: agent_factory(agent_seed),
                agent_seed,
            },
            reporter,
        )
    }

    /// Drive `num_trials_per_env` trials of every named unit to completion.
    ///
    /// This is the shape-agnostic half of the evaluator: rayon fan-out,
    /// per-trial `catch_unwind`, checkpoint load/skip/save, fail-fast, and the
    /// reporter lifecycle. Nothing here knows whether a trial is an episodic
    /// environment rollout or something else — that lives entirely behind
    /// [`Trial::run`].
    ///
    /// `make_trial` is called once per pending key with `(key, env_seed,
    /// agent_seed)`, all three derived from the configured base seed.
    ///
    /// # Panics
    /// Panics if the rayon thread pool cannot be built.
    #[allow(clippy::too_many_lines)]
    pub fn run_trials<T, MK, R>(
        &self,
        suite_name: &str,
        env_names: &[String],
        make_trial: MK,
        reporter: &mut R,
    ) -> BenchmarkReport
    where
        T: Trial + Send,
        MK: Fn(TrialKey, u64, u64) -> T + Sync + Send,
        R: Reporter,
    {
        let cfg = &self.cfg;
        let seeds = SeedStream::new(cfg.base_seed);

        let suite_info = SuiteInfo {
            name: suite_name.to_string(),
            env_names: env_names.to_vec(),
            num_trials_per_env: cfg.num_trials_per_env,
            success_threshold: cfg.success_threshold,
        };

        // Load checkpoint (if any) and start report.
        let mut report = cfg
            .checkpoint_dir
            .as_ref()
            .and_then(|dir| {
                let path = checkpoint::checkpoint_path(dir, suite_name);
                checkpoint::load(&path).ok().flatten()
            })
            .unwrap_or_else(|| BenchmarkReport::new(suite_name.to_string(), cfg.base_seed));
        let skip = checkpoint::completed_keys(&report);

        reporter.on_suite_start(&suite_info);

        // Enumerate trial keys, filtering already-completed.
        let mut pending: Vec<TrialKey> = Vec::new();
        for env_idx in 0..env_names.len() {
            for trial_idx in 0..cfg.num_trials_per_env {
                let key = TrialKey { env_idx, trial_idx };
                if !skip.contains(&key) {
                    pending.push(key);
                }
            }
        }

        let dyn_reporter: &mut dyn Reporter = reporter;
        let reporter_lock: Mutex<&mut dyn Reporter> = Mutex::new(dyn_reporter);
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

                let env_name = &env_names[key.env_idx];
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
                    make_trial(key, env_seed, agent_seed).run(cfg, &info, &reporter_lock)
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
                    let path = checkpoint::checkpoint_path(dir, suite_name);
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
            let path = checkpoint::checkpoint_path(dir, suite_name);
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

/// One unit of benchmarked work: something the evaluator runs once, under a
/// derived seed, to produce a [`TrialReport`].
///
/// This is the seam that lets [`Evaluator::run_trials`] own everything
/// expensive and easy to get wrong — rayon fan-out, `catch_unwind`,
/// checkpointing, fail-fast, reporter lifecycle — while the shape of the work
/// varies. An episodic environment rollout ([`EpisodicTrial`]) is one shape; an
/// evolutionary generation loop, which has no episode axis at all, is another.
///
/// # Contract
///
/// - `run` consumes the trial: one trial value drives one trial.
/// - `run` MUST NOT catch its own panics. `run_trials` wraps every call in
///   `catch_unwind` and folds the payload into an errored [`TrialReport`];
///   swallowing a panic here hides a genuine bug behind a passing report.
/// - Recoverable failures belong in [`TrialReport::errored`] /
///   `error_message`, not in a panic.
pub trait Trial {
    /// Run this trial to completion and summarize it.
    ///
    /// `reporter` is shared across rayon workers; hold the lock only for the
    /// duration of a single callback.
    fn run(
        self,
        cfg: &EvaluatorConfig,
        info: &TrialInfo,
        reporter: &Mutex<&mut dyn Reporter>,
    ) -> TrialReport;
}

/// A [`Trial`] that rolls an [`Environment`] out against a [`BenchableAgent`]
/// for `num_episodes` episodes of at most `max_steps` steps.
///
/// This is the classic RL benchmarking shape and the only one `run_suite`
/// builds.
///
/// # Why the const parameters are on the struct
///
/// `Trial` carries no rank parameters, so `D`/`SD`/`AD` would appear in
/// neither the implemented trait nor the self type and the impl would be
/// rejected by E0207. Declaring them here constrains them. Const generics need
/// no `PhantomData` to be left unused by fields.
///
/// This is the same constraint that forced `BenchAdapter` to carry three
/// phantom const parameters, relocated rather than removed: erasing rank at
/// the `Trial` boundary has the same cost as erasing it at the `BenchEnv`
/// boundary did. Inference still fills them in from the `Environment` bound,
/// so callers never write them.
pub struct EpisodicTrial<E, A, const D: usize, const SD: usize, const AD: usize> {
    /// The environment under test, already seeded by the suite factory.
    pub env: E,
    /// The agent under test, already seeded by the agent factory.
    pub agent: A,
    /// Seed for the action-sampling RNG handed to the agent each step.
    pub agent_seed: u64,
}

impl<E, A, const D: usize, const SD: usize, const AD: usize> std::fmt::Debug
    for EpisodicTrial<E, A, D, SD, AD>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EpisodicTrial")
            .field("agent_seed", &self.agent_seed)
            .finish_non_exhaustive()
    }
}

impl<E, A, const D: usize, const SD: usize, const AD: usize> Trial
    for EpisodicTrial<E, A, D, SD, AD>
where
    E: Environment<D, SD, AD, RewardType = ScalarReward>,
    A: BenchableAgent<E::ObservationType, E::ActionType>,
{
    fn run(
        mut self,
        cfg: &EvaluatorConfig,
        info: &TrialInfo,
        reporter: &Mutex<&mut dyn Reporter>,
    ) -> TrialReport {
        let mut rng = StdRng::seed_from_u64(self.agent_seed);
        let mut report = TrialReport::new(info.key, info.env_name.clone(), info.trial_seed);
        let mut returns: Vec<f64> = Vec::with_capacity(cfg.num_episodes);
        let mut lengths: Vec<usize> = Vec::with_capacity(cfg.num_episodes);
        // Steps (not episodes) whose reward was not finite, over the whole
        // trial. Summarised once after the episode loop.
        let mut non_finite_steps: u64 = 0;

        let start = Instant::now();

        'episodes: for episode_idx in 0..cfg.num_episodes {
            let mut obs = match self.env.reset() {
                Ok(snap) => snap.into_observation(),
                Err(e) => {
                    report.errored = true;
                    report.error_message = Some(e.to_string());
                    break 'episodes;
                }
            };
            let mut total_reward = 0.0;
            let mut length = 0;

            for _ in 0..cfg.max_steps {
                let action = self.agent.act(&obs, &mut rng);
                let snap = match self.env.step(action) {
                    Ok(s) => s,
                    Err(e) => {
                        report.errored = true;
                        report.error_message = Some(e.to_string());
                        break 'episodes;
                    }
                };
                let reward = f64::from(snap.reward().value());
                // A non-finite reward is counted and reported, never asserted
                // on and never filtered out. Accumulation stays raw and
                // unconditional: a non-finite return is a *true* statement
                // about a run whose environment emitted one, matching the
                // training-loop policy in `c51/train.rs`. Dropping the term
                // would fabricate a plausible return the run never earned.
                //
                // A `debug_assert!` tripwire used to live here. It made the
                // harness's *output* a function of `cfg(debug_assertions)`: in
                // release the trial completed with a poisoned return, while in
                // debug it panicked into `run_trials`' `catch_unwind`, which
                // replaces the trial with a fresh errored `TrialReport` —
                // discarding every episode already completed — and under
                // `fail_fast` killed the whole suite. A measurement harness
                // whose artefact shape depends on the build profile is broken
                // on its own terms.
                if !reward.is_finite() {
                    non_finite_steps += 1;
                }
                total_reward += reward;
                length += 1;

                // Read `is_done()` before consuming the snapshot: `into_observation`
                // takes `self` by value and must be its last use.
                let done = snap.is_done();
                obs = snap.into_observation();
                if done {
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
                let mut r = reporter.lock().unwrap();
                r.on_episode_end(info, &rec);
            }
        }

        // One deterministic summary line per affected trial — not the
        // escalating decade schedule used by the RL crate's
        // `FiniteRewardGuard`, which exists because `remember` is a hot path
        // with no aggregation point. This loop has one, and each trial owns its
        // own counter, so a per-trial decade schedule would restart at 1 and be
        // strictly noisier than a single total.
        if non_finite_steps > 0 {
            tracing::warn!(
                target: "rlevo_benchmarks",
                env = %info.env_name,
                non_finite_steps,
                "environment produced non-finite rewards; episode returns are poisoned"
            );
        }

        let wall = start.elapsed().as_secs_f64();
        report.absorb_metrics(core_metrics(
            &returns,
            &lengths,
            wall,
            cfg.success_threshold,
        ));
        report.absorb_metrics(vec![Metric::Counter {
            name: RETURN_NON_FINITE_STEPS.to_string(),
            count: non_finite_steps,
        }]);
        report.absorb_metrics(self.agent.emit_metrics());
        report
    }
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

/// A [`Trial`] that drives a [`GenerationProbe`] to its budget.
///
/// The counterpart to [`EpisodicTrial`] for work that has no episode axis. It
/// calls `begin` once, then `advance` until the probe reports exhaustion,
/// and reports the final unit's metrics plus a generation count.
///
/// # Why no episodes
///
/// [`TrialReport::episodes`] is left **empty**, deliberately. A probe re-seeds
/// on `begin`, so a second pass would be a byte-identical replay — duplicated
/// compute carrying zero information. Rather than fabricate one synthetic
/// `EpisodeSummary` per run — which would make `num_episodes > 1` silently
/// multiply cost for no signal, the trap the retired `BenchEnv` path had —
/// this trial reports no episodes at all and `on_episode_end` never fires.
/// Consumers read the `ea/*` or `coea/*` scalars instead.
pub struct GenerationTrial<P> {
    /// The probe under test, already seeded by the suite factory.
    pub probe: P,
}

impl<P> std::fmt::Debug for GenerationTrial<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationTrial").finish_non_exhaustive()
    }
}

impl<P> Trial for GenerationTrial<P>
where
    P: GenerationProbe,
    P::Metrics: MetricsProvider,
{
    fn run(
        mut self,
        cfg: &EvaluatorConfig,
        info: &TrialInfo,
        _reporter: &Mutex<&mut dyn Reporter>,
    ) -> TrialReport {
        let mut report = TrialReport::new(info.key, info.env_name.clone(), info.trial_seed);
        let start = Instant::now();

        self.probe.begin();

        let mut units = 0usize;
        let mut last: Option<P::Metrics> = None;
        // `max_steps` is a backstop only: the probe owns its own budget and
        // ends the loop by returning `None`. The cap exists so a probe with a
        // broken exhaustion check cannot hang the suite.
        while units < cfg.max_steps {
            match self.probe.advance() {
                Some(m) => {
                    units += 1;
                    last = Some(m);
                }
                None => break,
            }
        }

        let wall = start.elapsed().as_secs_f64();
        report.absorb_metrics(vec![
            Metric::Scalar {
                name: "generations".to_string(),
                #[allow(clippy::cast_precision_loss)]
                value: units as f64,
            },
            Metric::Scalar {
                name: "wall_clock_seconds".to_string(),
                value: wall,
            },
        ]);
        if let Some(m) = last {
            report.absorb_metrics(m.emit());
        }
        report
    }
}

#[cfg(test)]
mod tests {
    //! Stub-based unit tests for `Evaluator::run_suite`.
    //!
    //! A minimal `StubEnv` and `StubAgent` replace real environments so the
    //! evaluator logic is exercised without pulling in `rlevo-environments`
    //! as a dev-dependency (which would drag in the full physics crate tree).

    use std::cell::RefCell;
    use std::sync::{Arc, Mutex, Once};

    use rand::Rng;
    use tracing::field::{Field, Visit};
    use tracing::{Event as TracingEvent, Level, Subscriber};
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

    use rlevo_core::base::{Action, Observation, State};
    use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
    use rlevo_core::fitness::{BenchableAgent, Metric};
    use rlevo_core::reward::ScalarReward;

    use super::{EpisodicTrial, Evaluator, EvaluatorConfig, Trial};
    use crate::metrics::core::{RETURN_MEAN, RETURN_NON_FINITE_STEPS};
    use crate::report::{BenchmarkReport, EpisodeSummary, TrialReport};
    use crate::reporter::Reporter;
    use crate::suite::{Suite, SuiteInfo, TrialInfo, TrialKey};

    // ── Stub env ─────────────────────────────────────────────────────────────

    struct StubEnv {
        /// Return `done = true` once this many steps have been taken.
        steps_until_done: usize,
        reset_fails: bool,
        step_fails: bool,
        steps_taken: usize,
        /// When set, `step` emits this reward instead of `1.0`. Used to drive
        /// the non-finite-reward tripwire.
        reward_override: Option<f32>,
    }

    impl StubEnv {
        fn new(steps_until_done: usize) -> Self {
            Self {
                steps_until_done,
                reset_fails: false,
                step_fails: false,
                steps_taken: 0,
                reward_override: None,
            }
        }
    }

    // Minimal rank-1 contract types so the stub is a real `Environment`.
    // The observation carries no payload: no test asserts on it, and the
    // evaluator only ever clones it through to the agent.

    #[derive(Debug, Clone)]
    struct StubObs;
    impl Observation<1> for StubObs {
        fn shape() -> [usize; 1] {
            [1]
        }
    }

    #[derive(Debug, Clone)]
    struct StubState;
    impl State<1> for StubState {
        fn shape() -> [usize; 1] {
            [1]
        }
        fn is_valid(&self) -> bool {
            true
        }
    }

    #[derive(Debug, Clone)]
    struct StubAction;
    impl Action<1> for StubAction {
        fn shape() -> [usize; 1] {
            [1]
        }
        fn is_valid(&self) -> bool {
            true
        }
    }

    impl Environment<1, 1, 1> for StubEnv {
        type StateType = StubState;
        type ObservationType = StubObs;
        type ActionType = StubAction;
        type RewardType = ScalarReward;
        type SnapshotType = SnapshotBase<1, StubObs, ScalarReward>;

        fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
            if self.reset_fails {
                return Err(EnvironmentError::InvalidAction("stub reset fail".into()));
            }
            self.steps_taken = 0;
            Ok(SnapshotBase::running(StubObs, ScalarReward(0.0)))
        }

        fn step(&mut self, _action: StubAction) -> Result<Self::SnapshotType, EnvironmentError> {
            if self.step_fails {
                return Err(EnvironmentError::InvalidAction("stub step fail".into()));
            }
            self.steps_taken += 1;
            let obs = StubObs;
            let reward = ScalarReward(self.reward_override.unwrap_or(1.0));
            if self.steps_taken >= self.steps_until_done {
                Ok(SnapshotBase::terminated(obs, reward))
            } else {
                Ok(SnapshotBase::running(obs, reward))
            }
        }
    }

    // ── Stub agent ───────────────────────────────────────────────────────────

    struct StubAgent;

    impl BenchableAgent<StubObs, StubAction> for StubAgent {
        fn act(&mut self, _obs: &StubObs, _rng: &mut dyn Rng) -> StubAction {
            StubAction
        }
    }

    /// A stub agent that emits a fixed metric list at trial end.
    ///
    /// `StubAgent` takes the `BenchableAgent::emit_metrics` default (an empty
    /// `Vec`), so no test driving it can observe the order in which the trial
    /// absorbs harness metrics and agent metrics. This one can.
    struct EmittingAgent {
        metrics: Vec<Metric>,
    }

    impl BenchableAgent<StubObs, StubAction> for EmittingAgent {
        fn act(&mut self, _obs: &StubObs, _rng: &mut dyn Rng) -> StubAction {
            StubAction
        }

        fn emit_metrics(&self) -> Vec<Metric> {
            self.metrics.clone()
        }
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
            (
                Self {
                    log: Arc::clone(&log),
                },
                log,
            )
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

    // ── Tracing capture ──────────────────────────────────────────────────────

    /// One `tracing` event, reduced to the parts these tests assert on.
    ///
    /// `env` and `non_finite_steps` are the two fields the trial-summary
    /// `warn!` carries; both are `Option` because most events carry neither.
    #[derive(Debug, Clone)]
    struct CapturedEvent {
        level: Level,
        env: Option<String>,
        non_finite_steps: Option<u64>,
    }

    thread_local! {
        /// Per-thread sink [`CaptureLayer`] appends to, `Some` only while
        /// [`capturing`] is on the stack for this thread.
        static SINK: RefCell<Option<Vec<CapturedEvent>>> = const { RefCell::new(None) };
    }

    /// Records every event whose thread has an active [`capturing`] scope.
    ///
    /// Mirrors `rlevo_test_support::capture::FieldCapture`, which is not used
    /// here for two reasons: `rlevo-test-support` pulls in `burn` and
    /// `rlevo-environments`, which this module's stubs exist to avoid, and it
    /// collects a single integer field, so it could not observe the `env` name.
    struct CaptureLayer;

    impl<S: Subscriber> Layer<S> for CaptureLayer {
        fn on_event(&self, event: &TracingEvent<'_>, _ctx: Context<'_, S>) {
            SINK.with(|sink| {
                let mut sink = sink.borrow_mut();
                let Some(events) = sink.as_mut() else { return };
                let mut visitor = EventVisitor {
                    env: None,
                    non_finite_steps: None,
                };
                event.record(&mut visitor);
                events.push(CapturedEvent {
                    level: *event.metadata().level(),
                    env: visitor.env,
                    non_finite_steps: visitor.non_finite_steps,
                });
            });
        }
    }

    /// Installs [`CaptureLayer`] as the **process-global** subscriber, once.
    ///
    /// Global rather than `with_default`-scoped, and called by every test that
    /// can emit the trial-summary `warn!`, because `tracing` caches each
    /// callsite's `Interest` globally and computes it from whatever dispatcher
    /// the thread that *first* reached the callsite happened to have. Three
    /// tests here reach that `warn!` from a rayon worker with no subscriber
    /// installed; if one of them registers the callsite first, `Interest::never`
    /// is cached for the rest of the process and a scoped capture silently sees
    /// nothing. That is not hypothetical — it was observed once in ~26 runs of
    /// an earlier `with_default` version of this helper, i.e. as a rare false
    /// pass rather than a failure.
    ///
    /// A *global* default closes that hole from both sides: `Dispatch::new`
    /// rebuilds the interest of every already-registered callsite against it,
    /// and every later registration — on any thread, including a rayon worker
    /// with no scoped default — resolves through it, because `get_default`
    /// falls back to the global. `Once::call_once` blocks every other caller
    /// until it is stored, so calling this first makes the ordering irrelevant.
    /// **A new test that emits this `warn!` must call this too**, or it reopens
    /// the race for the others.
    fn install_capture_subscriber() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            tracing::subscriber::set_global_default(
                tracing_subscriber::registry().with(CaptureLayer),
            )
            .expect("no other global tracing subscriber may be installed in this test binary");
        });
    }

    /// Runs `f` with this thread's event sink active, returning what it emitted.
    fn capturing<T>(f: impl FnOnce() -> T) -> (T, Vec<CapturedEvent>) {
        install_capture_subscriber();
        SINK.with(|sink| *sink.borrow_mut() = Some(Vec::new()));
        let out = f();
        let events = SINK
            .with(|sink| sink.borrow_mut().take())
            .expect("sink installed above");
        (out, events)
    }

    /// Extracts the two named fields from one event.
    ///
    /// `non_finite_steps` is a `u64` and arrives through `record_u64`; `env` is
    /// recorded with `%` (`tracing::field::display`), which routes through
    /// `record_debug` with a `format_args!` payload — whose `Debug` is its
    /// `Display`, so the formatted string is the environment name verbatim.
    struct EventVisitor {
        env: Option<String>,
        non_finite_steps: Option<u64>,
    }

    impl Visit for EventVisitor {
        fn record_u64(&mut self, field: &Field, value: u64) {
            if field.name() == "non_finite_steps" {
                self.non_finite_steps = Some(value);
            }
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "env" {
                self.env = Some(value.to_string());
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "env" {
                self.env = Some(format!("{value:?}"));
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Runs one `EpisodicTrial` directly, returning its report and every
    /// `tracing` event it emitted.
    ///
    /// The trial is driven directly rather than through `run_suite` because the
    /// capture sink is **per-thread**, and `run_suite` fans trials out onto
    /// rayon workers — a captured-nothing result there would be an artefact of
    /// the thread, not of the code.
    fn run_trial_capturing_events(
        env: StubEnv,
        cfg: &EvaluatorConfig,
        env_name: &str,
    ) -> (TrialReport, Vec<CapturedEvent>) {
        let info = TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_name: env_name.to_string(),
            trial_seed: 0,
        };
        let (mut reporter, _) = SpyReporter::new();
        let dyn_reporter: &mut dyn Reporter = &mut reporter;
        let lock = Mutex::new(dyn_reporter);
        let trial = EpisodicTrial {
            env,
            agent: StubAgent,
            agent_seed: 0,
        };

        capturing(|| trial.run(cfg, &info, &lock))
    }

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

    /// The rollout loop must move the observation out of the snapshot rather
    /// than clone it (issue #554). `CountingObs` makes a clone observable:
    /// if the loop ever clones, this counter is non-zero.
    #[test]
    fn rollout_moves_the_observation_instead_of_cloning() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        static CLONES: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        struct CountingObs;
        impl Clone for CountingObs {
            fn clone(&self) -> Self {
                CLONES.fetch_add(1, AtomicOrdering::Relaxed);
                Self
            }
        }
        impl Observation<1> for CountingObs {
            fn shape() -> [usize; 1] {
                [1]
            }
        }

        struct CountingEnv {
            steps: usize,
        }
        impl Environment<1, 1, 1> for CountingEnv {
            type StateType = StubState;
            type ObservationType = CountingObs;
            type ActionType = StubAction;
            type RewardType = ScalarReward;
            type SnapshotType = SnapshotBase<1, CountingObs, ScalarReward>;

            fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
                self.steps = 0;
                Ok(SnapshotBase::running(CountingObs, ScalarReward(0.0)))
            }

            fn step(&mut self, _a: StubAction) -> Result<Self::SnapshotType, EnvironmentError> {
                self.steps += 1;
                let obs = CountingObs;
                let r = ScalarReward(1.0);
                if self.steps >= 3 {
                    Ok(SnapshotBase::terminated(obs, r))
                } else {
                    Ok(SnapshotBase::running(obs, r))
                }
            }
        }

        struct CountingAgent;
        impl BenchableAgent<CountingObs, StubAction> for CountingAgent {
            fn act(&mut self, _obs: &CountingObs, _rng: &mut dyn Rng) -> StubAction {
                StubAction
            }
        }

        CLONES.store(0, AtomicOrdering::Relaxed);
        let cfg = single_thread_cfg(2, 50);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| CountingEnv { steps: 0 });
        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| CountingAgent, &mut reporter);

        assert!(!report.trials[0].errored);
        assert_eq!(report.trials[0].episodes.len(), 2, "both episodes ran");
        assert_eq!(
            CLONES.load(AtomicOrdering::Relaxed),
            0,
            "rollout must move the observation out of the snapshot, not clone it"
        );
    }

    /// A non-finite reward is counted and reported, never asserted on (issues
    /// #567, #1115). This test is deliberately **not** `debug_assertions`-gated:
    /// the whole point of the fix is that the harness's output no longer
    /// depends on the build profile, so it must pin identical behaviour in both.
    ///
    /// It pins the *values*, not merely "did not panic": a version that
    /// silently dropped the non-finite term from `total_reward` would leave the
    /// returns finite, and a version that counted episodes rather than steps
    /// would report 2 instead of 6.
    #[test]
    fn non_finite_reward_is_counted_not_asserted() {
        // 2 episodes x 3 steps, each step rewarding +inf → 6 non-finite steps.
        // `max_steps = 10` bounds the loop, so a done-detection defect fails
        // the length assertion instead of hanging.
        const EPISODES: usize = 2;
        const STEPS_PER_EPISODE: usize = 3;

        // This trial emits the non-finite `warn!` from a rayon worker; see
        // `install_capture_subscriber` for why every such test must call it.
        install_capture_subscriber();
        let cfg = single_thread_cfg(EPISODES, 10);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| {
            let mut env = StubEnv::new(STEPS_PER_EPISODE);
            env.reward_override = Some(f32::INFINITY);
            env
        });

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);
        let trial = &report.trials[0];

        assert!(
            !trial.errored,
            "a non-finite reward is an observation about the run, not a trial failure"
        );
        assert_eq!(
            trial.episodes.len(),
            EPISODES,
            "every episode must survive into the report"
        );
        for ep in &trial.episodes {
            assert_eq!(
                ep.length, STEPS_PER_EPISODE,
                "the episode still ran to termination"
            );
            assert!(
                ep.return_value.is_infinite() && ep.return_value.is_sign_positive(),
                "the poisoned return reaches the report; the term is never dropped"
            );
        }
        let mean = trial.scalars[RETURN_MEAN];
        assert!(
            mean.is_infinite() && mean.is_sign_positive(),
            "downstream aggregates carry the poison rather than hiding it"
        );
        assert_eq!(
            trial.counters[RETURN_NON_FINITE_STEPS],
            (EPISODES * STEPS_PER_EPISODE) as u64,
            "the counter counts steps over the whole trial, not episodes"
        );
    }

    /// The counter is present and zero on a clean trial, so a consumer can tell
    /// "no non-finite rewards" from "this harness version does not report it".
    #[test]
    fn non_finite_counter_is_zero_on_a_clean_trial() {
        let cfg = single_thread_cfg(2, 10);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| StubEnv::new(3));

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        assert_eq!(
            report.trials[0].counters[RETURN_NON_FINITE_STEPS], 0,
            "a finite trial reports the counter explicitly as zero"
        );
    }

    /// Number of non-finite steps the harness itself counts in the collision
    /// tests below: 2 episodes x 4 steps. Distinct from both `0` and the
    /// agent's `999`, so no assertion can pass by coincidence.
    const HARNESS_COUNT: u64 = 8;

    /// Runs one poisoned trial (`+inf` every step) against an agent that emits
    /// `metrics` from `emit_metrics`, and returns its report.
    fn poisoned_trial_with_agent_metrics(metrics: &[Metric]) -> TrialReport {
        // This trial emits the non-finite `warn!` from a rayon worker; see
        // `install_capture_subscriber` for why every such test must call it.
        install_capture_subscriber();
        let cfg = single_thread_cfg(2, 10);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| {
            let mut env = StubEnv::new(4);
            env.reward_override = Some(f32::INFINITY);
            env
        });

        let (mut reporter, _) = SpyReporter::new();
        let mut report = Evaluator::new(cfg).run_suite(
            &suite,
            |_| EmittingAgent {
                metrics: metrics.to_vec(),
            },
            &mut reporter,
        );
        report.trials.remove(0)
    }

    /// The `return/` namespace is what Decision 4 of ADR 0078 buys: an agent
    /// emitting the unprefixed name a harness author would plausibly choose
    /// leaves the harness's own counter alone, and both keys coexist.
    ///
    /// Kills the mutant that drops the prefix from `RETURN_NON_FINITE_STEPS`.
    #[test]
    fn agent_metric_cannot_collide_with_the_namespaced_counter() {
        let trial = poisoned_trial_with_agent_metrics(&[Metric::Counter {
            name: "non_finite_steps".to_string(),
            count: 999,
        }]);

        assert_eq!(
            trial.counters[RETURN_NON_FINITE_STEPS], HARNESS_COUNT,
            "the namespaced key must carry the harness's own count, not the agent's"
        );
        assert_eq!(
            trial.counters["non_finite_steps"], 999,
            "the agent's free-form key is recorded verbatim beside it"
        );
    }

    /// Emission order, pinned: `absorb_metrics` overwrites same-name entries
    /// and the agent's metrics are absorbed **last**, so an agent that names
    /// the harness's exact key takes it. The prefix makes that collision
    /// deliberate rather than accidental; it does not make it impossible.
    ///
    /// Kills the mutant that swaps the counter and `emit_metrics` absorbs —
    /// under the swapped order the harness's `8` would win instead.
    ///
    /// If this assertion is ever inverted so the harness wins, that is a
    /// behaviour change to `absorb_metrics`' last-writer-wins contract and
    /// needs an ADR, not a test edit.
    #[test]
    fn agent_metrics_are_absorbed_after_the_harness_counter() {
        let trial = poisoned_trial_with_agent_metrics(&[Metric::Counter {
            name: RETURN_NON_FINITE_STEPS.to_string(),
            count: 999,
        }]);

        assert_eq!(
            trial.counters[RETURN_NON_FINITE_STEPS], 999,
            "the last absorb wins, and the agent's is last"
        );
    }

    /// One `warn!` per affected trial — not one per affected step, and not
    /// zero (ADR 0078, Decision 3). The line must carry the environment name
    /// and the trial total, since it is the TUI's only channel for this
    /// failure.
    #[test]
    fn non_finite_reward_warns_once_per_trial_with_env_and_total() {
        let cfg = single_thread_cfg(2, 10);
        let mut env = StubEnv::new(3);
        env.reward_override = Some(f32::INFINITY);

        let (report, events) = run_trial_capturing_events(env, &cfg, "poisoned-env");

        let warnings: Vec<&CapturedEvent> =
            events.iter().filter(|e| e.level == Level::WARN).collect();
        assert_eq!(
            warnings.len(),
            1,
            "exactly one warning per affected trial; this trial had 6 affected steps"
        );
        assert_eq!(
            warnings[0].env.as_deref(),
            Some("poisoned-env"),
            "the warning must name the environment that produced the rewards"
        );
        assert_eq!(
            warnings[0].non_finite_steps,
            Some(6),
            "the warning must carry the trial total, matching the counter"
        );
        assert_eq!(
            report.counters[RETURN_NON_FINITE_STEPS], 6,
            "log line and counter must report the same total"
        );
    }

    /// Zero lines on a clean trial — the other half of Decision 3's bound.
    #[test]
    fn a_clean_trial_emits_no_non_finite_warning() {
        let cfg = single_thread_cfg(2, 10);

        let (report, events) = run_trial_capturing_events(StubEnv::new(3), &cfg, "clean-env");

        assert!(
            events.iter().all(|e| e.level != Level::WARN),
            "a trial with no non-finite reward must emit no warning"
        );
        assert_eq!(
            report.counters[RETURN_NON_FINITE_STEPS], 0,
            "the counter is still emitted, as zero"
        );
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
        impl Environment<1, 1, 1> for PanickingEnv {
            type StateType = StubState;
            type ObservationType = StubObs;
            type ActionType = StubAction;
            type RewardType = ScalarReward;
            type SnapshotType = SnapshotBase<1, StubObs, ScalarReward>;

            fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
                Ok(SnapshotBase::running(StubObs, ScalarReward(0.0)))
            }

            fn step(
                &mut self,
                _action: StubAction,
            ) -> Result<Self::SnapshotType, EnvironmentError> {
                panic!("deliberate evaluator test panic");
            }
        }

        let cfg = single_thread_cfg(1, 10);
        let suite = Suite::new("s", cfg.clone()).with_env("e", |_| PanickingEnv);

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        assert!(report.trials[0].errored);
        assert!(
            report.trials[0]
                .error_message
                .as_deref()
                .unwrap_or("")
                .contains("deliberate evaluator test panic")
        );
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
        expected.sort_unstable();
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
