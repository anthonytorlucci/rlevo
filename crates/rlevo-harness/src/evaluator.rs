//! Suite evaluator with rayon trial-level parallelism and checkpoint resume.
//!
//! The core entry point is [`Evaluator::run_suite`], which drives every
//! `(env, seed)` trial in a [`Suite`] through an [`Agent`], training it first
//! when [`EvaluatorConfig::train_steps`] asks, collects [`TrialReport`]s, and
//! delivers them to a [`Reporter`]. [`train`] is the training loop on its own,
//! for a caller that keeps the agent afterwards.
//!
//! Parallelism is controlled via [`EvaluatorConfig::num_threads`] (`None` reuses
//! the global rayon pool). Panics inside a trial are caught with
//! `catch_unwind` and recorded as errored trials rather than aborting the run.
//! [`EvaluatorConfig::checkpoint_dir`] enables crash-recovery: the partial
//! [`BenchmarkReport`] is atomically written after every trial so a resumed run
//! can skip already-finished keys.
//!
//! Each trial runs inside a `trial` tracing span (target `rlevo_harness`) with
//! fields `run`, `env`, `env_idx`, `trial` and `seed`, so a tracing layer can
//! file an event under the trial whose work emitted it. `run` is the run's ID
//! ([`BenchmarkReport::run_id`]): with two runs in one process, each with a
//! layer of its own, it says whose trial `env_idx` and `trial` name. Work a
//! trial hands to another thread runs outside the span.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use rlevo_core::agent::{Agent, Transition};
use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus};
use rlevo_core::evaluation::GenerationProbe;
use rlevo_core::fitness::Metric;
use rlevo_core::util::seed::SeedStream;

use crate::checkpoint;
use crate::metrics::{GENERATIONS, RETURN_NON_FINITE_STEPS, WALL_CLOCK_SECONDS, core_metrics};
use crate::report::{BenchmarkReport, EpisodeSummary, Phase, Provenance, TrialReport};
use crate::reporter::{Boundary, Reporter};
use crate::suite::{Suite, SuiteInfo, TrialInfo, TrialKey, TrialUnit};

/// Parameters governing how the evaluator runs a suite.
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Number of episodes to run per trial.
    ///
    /// For an [`EpisodicTrial`] these are the evaluation episodes, the ones
    /// its report holds. Training, when [`train_steps`](Self::train_steps)
    /// asks for it, comes first and is budgeted in steps.
    pub num_episodes: usize,
    /// Environment steps an [`EpisodicTrial`] trains its agent for before it
    /// evaluates it; `0`, the default, evaluates the agent as built.
    ///
    /// Training runs through [`train`]: the agent is told
    /// [`set_training(true)`](Agent::set_training), shown every step, and its
    /// episodes are capped at [`max_steps`](Self::max_steps) as evaluation's
    /// are. They are not reported: the report's episodes, and every metric
    /// computed from them, are the evaluation episodes. A
    /// [`GenerationTrial`] ignores this field.
    pub train_steps: usize,
    /// Number of independent seeds to evaluate per environment.
    pub num_trials_per_env: usize,
    /// Steps per episode, a hard ceiling for an [`EpisodicTrial`].
    ///
    /// An episode that has not ended by `max_steps` is cut there with whatever
    /// return accumulated so far, and its agent is told it ended
    /// [`Truncated`](EpisodeStatus::Truncated). A [`GenerationTrial`] ignores
    /// this field; its ceiling is [`max_generations`](Self::max_generations).
    pub max_steps: usize,
    /// Generations per trial, a backstop for a [`GenerationTrial`].
    ///
    /// The probe owns the real budget, and this should never be the one to end
    /// the trial. When it is, the trial says so with a warning. Set it at
    /// least as large as the probe's budget. An [`EpisodicTrial`] ignores this
    /// field.
    pub max_generations: usize,
    /// The suite's seed: the one number every seed in the suite is derived
    /// from, through [`SeedStream`].
    ///
    /// It is not the seed of any one run. [`Evaluator::run_trials`] derives a
    /// trial seed for each (environment, trial) pair, and from that an
    /// environment seed and an agent seed, which it hands to the closure that
    /// builds the trial. A generation loop built there takes one of those as
    /// its own seed. Two suites with the same `base_seed` and the same shape
    /// run the same trials; trial 0 and trial 1 of one suite do not.
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
    /// When `Some(threshold)`, a `success_rate` metric is emitted per trial counting the fraction
    /// of episodes whose return `$\geq$` `threshold`.
    pub success_threshold: Option<f64>,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            num_episodes: 100,
            train_steps: 0,
            num_trials_per_env: 1,
            max_steps: 1_000,
            max_generations: 1_000,
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

    /// Run a suite of [`Environment`]s against an [`Agent`].
    ///
    /// `agent_factory` is called once per trial with the derived agent seed;
    /// it is the implementor's contract to initialize the agent deterministically.
    ///
    /// Each trial trains its agent for [`EvaluatorConfig::train_steps`]
    /// environment steps, then evaluates it for
    /// [`EvaluatorConfig::num_episodes`] episodes; with `train_steps` at `0`
    /// it only evaluates. A learner therefore needs nothing beyond its
    /// [`Agent`] impl to be trained and measured here, and a random baseline
    /// run under the same configuration spends the same training steps
    /// learning nothing. See [`EpisodicTrial`] for what the agent is shown.
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
    pub fn run_suite<E, A, R, FA>(
        &self,
        suite: &Suite<E>,
        agent_factory: FA,
        reporter: &mut R,
    ) -> BenchmarkReport
    where
        E: Environment + Send,
        A: Agent<E::ObservationType, E::ActionType> + Send,
        R: Reporter,
        FA: Fn(u64) -> A + Sync + Send,
    {
        let units: Vec<TrialUnit> = suite.envs.iter().map(|e| e.unit.clone()).collect();
        self.run_trials(
            &suite.name,
            &units,
            |key, env_seed, agent_seed| EpisodicTrial {
                env: (suite.envs[key.env_idx].factory.as_ref())(env_seed),
                agent: agent_factory(agent_seed),
                agent_seed,
            },
            reporter,
        )
    }

    /// Drive `num_trials_per_env` trials of every unit to completion.
    ///
    /// This is the shape-agnostic half of the evaluator: rayon fan-out,
    /// per-trial `catch_unwind`, checkpoint load/skip/save, fail-fast, and the
    /// reporter lifecycle. Nothing here knows whether a trial is an episodic
    /// environment rollout or something else — that lives entirely behind
    /// [`Trial::run`].
    ///
    /// `units[i]` names what the trials with `key.env_idx == i` run: every
    /// one of their reports carries the unit's id as `env_id` and its task's
    /// hash, if it has one, as `task_hash`; the report keeps each unit's full
    /// fingerprint once, in `tasks`. A checkpointed trial is skipped only when
    /// its id and task still match its unit's; one that does not is run
    /// again. A unit is built from an id
    /// (`"sphere-d8".into()`) or, to fingerprint the task, with
    /// `TrialUnit::with_task` (feature `json`).
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
        units: &[TrialUnit],
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

        // Load checkpoint (if any) and start report.
        let mut report = cfg
            .checkpoint_dir
            .as_ref()
            .and_then(|dir| {
                let path = checkpoint::checkpoint_path(dir, suite_name);
                checkpoint::load(&path).ok().flatten()
            })
            .unwrap_or_else(|| BenchmarkReport::new(suite_name.to_string(), cfg.base_seed));
        // A resumed run is the run it resumes: it keeps the checkpoint's ID,
        // so a sink that names its output by the ID adds to what is there. A
        // checkpoint from before 0.7.0 has none and is given one.
        let run_id = report
            .run_id
            .get_or_insert_with(crate::report::new_run_id)
            .clone();
        let suite_info = SuiteInfo::new(
            suite_name.to_string(),
            units.iter().map(|u| u.id.clone()).collect(),
            cfg.num_trials_per_env,
            cfg.success_threshold,
            run_id.clone(),
        );
        // A resumed run reports the build that finishes it, and says so when
        // the checkpoint's trials came from another; each finished trial keeps
        // the provenance of the build that ran it.
        let this_build = Provenance::this_build();
        if let Some(earlier) = &report.provenance
            && *earlier != this_build
        {
            tracing::warn!(
                target: "rlevo_harness",
                suite = %suite_name,
                checkpoint_commit = ?earlier.git_commit,
                commit = ?this_build.git_commit,
                "resuming a checkpoint written by another build; each trial keeps the build that ran it"
            );
        }
        report.provenance = Some(this_build.clone());
        // A checkpointed trial counts as done only for the same unit and task:
        // one whose environment id, task or task hash no longer matches is
        // dropped and run again, so a report never mixes two tasks under one
        // unit.
        let checkpoint_tasks = std::mem::take(&mut report.tasks);
        let before = report.trials.len();
        report.trials.retain(|t| {
            units.get(t.key.env_idx).is_none_or(|unit| {
                t.env_id == unit.id
                    && checkpoint_tasks.get(t.key.env_idx).cloned().flatten() == unit.task
                    && t.task_hash.as_deref() == unit.task.as_ref().map(|f| f.hash.as_str())
            })
        });
        let dropped = before - report.trials.len();
        if dropped > 0 {
            tracing::warn!(
                target: "rlevo_harness",
                suite = %suite_name,
                dropped,
                "checkpointed trials were of another unit or task and are run again"
            );
        }
        report.tasks = units.iter().map(|u| u.task.clone()).collect();
        let skip = checkpoint::completed_keys(&report);

        reporter.on_suite_start(&suite_info);

        // Enumerate trial keys, filtering already-completed.
        let mut pending: Vec<TrialKey> = Vec::new();
        for env_idx in 0..units.len() {
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

                let unit = &units[key.env_idx];
                let env_id = &unit.id;
                let trial_seed = seeds.trial_seed(key.env_idx, key.trial_idx);
                let env_seed = seeds.env_seed(trial_seed);
                let agent_seed = seeds.agent_seed(trial_seed);

                let info = TrialInfo::new(key, env_id.clone(), trial_seed);
                // Everything the trial does on this thread, its reporter
                // callbacks included, is inside the span. A root span, with no
                // parent: when the trial's own parallel work lets rayon run
                // another trial's job on this thread, that trial's span must
                // not nest under this one.
                let span = tracing::info_span!(
                    target: "rlevo_harness",
                    parent: None,
                    "trial",
                    run = run_id.as_str(),
                    env = %env_id,
                    env_idx = key.env_idx,
                    trial = key.trial_idx,
                    seed = trial_seed,
                );
                let _in_trial = span.enter();
                lock_reporter(&reporter_lock).on_trial_start(&info);

                let trial_report = catch_unwind(AssertUnwindSafe(|| {
                    make_trial(key, env_seed, agent_seed).run(cfg, &info, &reporter_lock)
                }));

                let tr = match trial_report {
                    Ok(tr) => tr,
                    Err(payload) => {
                        let msg = panic_message(&*payload);
                        let mut err = TrialReport::new(key, env_id.clone(), trial_seed);
                        err.errored = true;
                        err.error_message = Some(msg);
                        err
                    }
                };

                let mut tr = tr;
                tr.task_hash = unit.task.as_ref().map(|f| f.hash.clone());
                tr.provenance = Some(this_build.clone());

                if tr.errored && cfg.fail_fast {
                    aborted.store(true, Ordering::Relaxed);
                }

                lock_reporter(&reporter_lock).on_trial_end(&info, &tr);

                let mut rep = report_lock.lock().unwrap_or_else(PoisonError::into_inner);
                rep.trials.push(tr);
                if let Some(dir) = &cfg.checkpoint_dir {
                    let path = checkpoint::checkpoint_path(dir, suite_name);
                    if let Err(e) = checkpoint::save(&path, &rep) {
                        tracing::warn!(
                            target: "rlevo_harness",
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
                    target: "rlevo_harness",
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
    /// duration of a single callback. `run_trials` calls
    /// [`on_trial_start`](Reporter::on_trial_start) and
    /// [`on_trial_end`](Reporter::on_trial_end) around it; what happens
    /// between, [`on_episode`](Reporter::on_episode) and
    /// [`on_metrics`](Reporter::on_metrics), is the trial's to call.
    fn run(
        self,
        cfg: &EvaluatorConfig,
        info: &TrialInfo,
        reporter: &Mutex<&mut dyn Reporter>,
    ) -> TrialReport;
}

/// A [`Trial`] that trains an [`Agent`] in an [`Environment`] for
/// [`EvaluatorConfig::train_steps`] steps, then evaluates it for
/// `num_episodes` episodes of at most `max_steps` steps.
///
/// This is the classic RL benchmarking shape and the only one `run_suite`
/// builds.
///
/// # What the agent is shown
///
/// Training, when there is any, is [`train`]: the trial calls
/// [`set_training(true)`](Agent::set_training) and runs the agent for that
/// many steps. Then it calls `set_training(false)` and runs the evaluation
/// episodes. In both phases, after every step it calls
/// [`on_transition`](Agent::on_transition) with the snapshot's status
/// unchanged, and when the status is done,
/// [`on_episode_end`](Agent::on_episode_end) with it. An episode the trial cuts
/// at `max_steps` ends with `on_episode_end(EpisodeStatus::Truncated)` after a
/// last transition whose status is still `Running`, since the environment did
/// not end it. One RNG, seeded from `agent_seed`, serves both phases.
///
/// The report holds the evaluation episodes only, so its returns, and every
/// metric computed from them, measure the trained agent. Wall-clock covers
/// both phases, less the time spent reading metrics for the reporter and in
/// its callbacks, so a sink does not slow the throughput the report states.
/// An environment error in either phase marks the trial errored.
///
/// # What the reporter is shown
///
/// After every episode that ends, in both phases, the trial calls
/// [`on_episode`](Reporter::on_episode) with its summary, then reads the
/// agent's [`emit_metrics`](Agent::emit_metrics) and calls
/// [`on_metrics`](Reporter::on_metrics) with them at a
/// [`Boundary::TrainingEpisode`] or [`Boundary::EvaluationEpisode`], whose
/// `step` counts the trial's environment steps across both phases. When no
/// sink [wants metrics](Reporter::wants_metrics), the read and the
/// `on_metrics` call are skipped. A training episode the step budget leaves
/// open is not reported. The report's own metrics are read once more, at the
/// end.
pub struct EpisodicTrial<E, A> {
    /// The environment under test, already seeded by the suite factory.
    pub env: E,
    /// The agent under test, already seeded by the agent factory.
    pub agent: A,
    /// Seed for the RNG handed to the agent at every step.
    pub agent_seed: u64,
}

impl<E, A> std::fmt::Debug for EpisodicTrial<E, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EpisodicTrial")
            .field("agent_seed", &self.agent_seed)
            .finish_non_exhaustive()
    }
}

impl<E, A> Trial for EpisodicTrial<E, A>
where
    E: Environment,
    A: Agent<E::ObservationType, E::ActionType>,
{
    fn run(
        mut self,
        cfg: &EvaluatorConfig,
        info: &TrialInfo,
        reporter: &Mutex<&mut dyn Reporter>,
    ) -> TrialReport {
        let mut rng = StdRng::seed_from_u64(self.agent_seed);
        let mut report = TrialReport::new(info.key, info.env_id.clone(), info.trial_seed);
        let mut returns: Vec<f64> = Vec::with_capacity(cfg.num_episodes);
        let mut lengths: Vec<usize> = Vec::with_capacity(cfg.num_episodes);
        // Evaluation steps (not episodes) whose reward was not finite, over
        // the whole trial. Summarised once after the episode loop.
        let mut non_finite_steps: u64 = 0;

        let start = Instant::now();
        let mut port = TrialPort::new(reporter, info);

        // Environment steps the trial has taken, training and evaluation
        // together: the `step` every boundary is plotted against.
        let mut step: u64 = 0;
        let trained = if cfg.train_steps > 0 {
            run_training(
                &mut self.env,
                &mut self.agent,
                cfg.train_steps,
                cfg.max_steps,
                &mut rng,
                |agent, ep, taken| {
                    let at = Boundary::TrainingEpisode {
                        index: ep.episode_idx,
                        step: taken,
                    };
                    port.episode(&ep, at, || agent.emit_metrics());
                },
            )
            .map(|taken| step = taken)
        } else {
            Ok(())
        };
        if let Err(e) = trained {
            report.errored = true;
            report.error_message = Some(e.to_string());
        } else {
            self.agent.set_training(false);
            // Evaluation episodes end by the environment or by `max_steps`,
            // never by a step budget.
            let mut unbudgeted = usize::MAX;
            for episode_idx in 0..cfg.num_episodes {
                // A non-finite reward is counted and reported, never asserted
                // on and never filtered out. Accumulation stays raw and
                // unconditional: a non-finite return is a *true* statement
                // about a run whose environment emitted one. Dropping the
                // term would fabricate a plausible return the run never
                // earned.
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
                let played = play_episode(
                    &mut self.env,
                    &mut self.agent,
                    cfg.max_steps,
                    &mut unbudgeted,
                    &mut rng,
                    &mut non_finite_steps,
                );
                let (total_reward, length, status) = match played {
                    Ok(Some(episode)) => episode,
                    Ok(None) => unreachable!("an evaluation episode has no step budget"),
                    Err(e) => {
                        report.errored = true;
                        report.error_message = Some(e.to_string());
                        break;
                    }
                };

                step += length as u64;
                let rec = EpisodeSummary::new(
                    Phase::Evaluation,
                    episode_idx,
                    total_reward,
                    length,
                    status,
                );
                returns.push(total_reward);
                lengths.push(length);
                let at = Boundary::EvaluationEpisode {
                    index: episode_idx,
                    step,
                };
                port.episode(&rec, at, || self.agent.emit_metrics());
                report.episodes.push(rec);
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
                target: "rlevo_harness",
                env = %info.env_id,
                non_finite_steps,
                "environment produced non-finite rewards; episode returns are poisoned"
            );
        }

        let wall = port.wall_clock(start);
        // Harness measurements take the privileged path; the agent's take the
        // checked one, so an agent that names a harness key is re-homed under
        // `agent/` instead of replacing the value this loop just measured.
        report.absorb_harness_metrics(core_metrics(
            &returns,
            &lengths,
            wall,
            cfg.success_threshold,
        ));
        report.absorb_harness_metrics(vec![Metric::Counter {
            name: RETURN_NON_FINITE_STEPS.to_string(),
            count: non_finite_steps,
        }]);
        report.absorb_metrics(self.agent.emit_metrics());
        report
    }
}

/// Trains `agent` in `env` for `steps` environment steps, and returns the
/// episodes it completed.
///
/// This is the training phase of an [`EpisodicTrial`], for a caller that
/// wants the trained agent back: the trial consumes its agent, this borrows
/// it. It calls [`set_training(true)`](Agent::set_training) first and leaves
/// it on. Every step the agent acts with `rng`, the environment steps, and the
/// agent is shown the step through [`on_transition`](Agent::on_transition)
/// with the snapshot's status unchanged; an episode that ends gets
/// [`on_episode_end`](Agent::on_episode_end) with that status. An episode
/// that reaches `max_steps` without ending is cut there, and the agent is
/// told it ended [`Truncated`](EpisodeStatus::Truncated) after a last
/// transition whose status is still `Running`. Pass `usize::MAX` to leave the
/// episodes' length to the environment. With `max_steps` at 0 no episode can
/// take a step, so it trains on nothing and returns no episodes.
///
/// The environment is reset at the start of each episode, and only while
/// steps remain, so no episode is opened that the budget cannot step. An
/// episode still open when the budget runs out is left open: the agent is not
/// told it ended, and it is not among the returned summaries.
///
/// # Errors
///
/// Returns the environment's error if a `reset` or `step` fails; the agent
/// keeps whatever it learned before it.
pub fn train<E, A>(
    env: &mut E,
    agent: &mut A,
    steps: usize,
    max_steps: usize,
    rng: &mut dyn Rng,
) -> Result<Vec<EpisodeSummary>, EnvironmentError>
where
    E: Environment,
    A: Agent<E::ObservationType, E::ActionType>,
{
    let mut episodes = Vec::new();
    run_training(env, agent, steps, max_steps, rng, |_, ep, _| {
        episodes.push(ep);
    })?;
    Ok(episodes)
}

/// [`train`]'s loop, calling `on_episode` with the agent, the summary and the
/// steps taken so far as each episode ends. Returns the steps taken in all,
/// those of an episode the budget left open included.
fn run_training<E, A>(
    env: &mut E,
    agent: &mut A,
    steps: usize,
    max_steps: usize,
    rng: &mut dyn Rng,
    mut on_episode: impl FnMut(&A, EpisodeSummary, u64),
) -> Result<u64, EnvironmentError>
where
    E: Environment,
    A: Agent<E::ObservationType, E::ActionType>,
{
    agent.set_training(true);
    // An episode of at most zero steps takes nothing from the budget, so the
    // loop below would never end.
    if max_steps == 0 {
        return Ok(0);
    }
    let mut budget = steps;
    let mut index = 0;
    // Training rewards are the agent's to judge; the trial counts only the
    // evaluation phase's.
    let mut non_finite_steps = 0;
    while budget > 0 {
        let Some((return_value, length, status)) = play_episode(
            env,
            agent,
            max_steps,
            &mut budget,
            rng,
            &mut non_finite_steps,
        )?
        else {
            break;
        };
        let taken = (steps - budget) as u64;
        let ep = EpisodeSummary::new(Phase::Training, index, return_value, length, status);
        on_episode(agent, ep, taken);
        index += 1;
    }
    Ok((steps - budget) as u64)
}

/// Locks the shared reporter, taking it even when poisoned.
///
/// A reporter that panics inside a trial poisons the lock; `run_trials`
/// records that trial as errored, and the other trials, and the callbacks
/// `run_trials` makes outside `catch_unwind`, must still reach the reporter
/// rather than abort the suite.
fn lock_reporter<'a, 'r>(
    reporter: &'a Mutex<&'r mut dyn Reporter>,
) -> MutexGuard<'a, &'r mut dyn Reporter> {
    reporter.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A trial's side of the shared reporter.
///
/// Reads the trial's metrics only when a sink wants them (asked once, when
/// the trial starts), and adds up the time it spends reading and in reporter
/// callbacks, which the trial leaves out of its wall clock.
struct TrialPort<'a, 'r> {
    reporter: &'a Mutex<&'r mut dyn Reporter>,
    info: &'a TrialInfo,
    wants_metrics: bool,
    spent: Duration,
}

impl<'a, 'r> TrialPort<'a, 'r> {
    fn new(reporter: &'a Mutex<&'r mut dyn Reporter>, info: &'a TrialInfo) -> Self {
        let wants_metrics = lock_reporter(reporter).wants_metrics();
        Self {
            reporter,
            info,
            wants_metrics,
            spent: Duration::ZERO,
        }
    }

    /// One finished episode, then the metrics `read` returns at `at`.
    fn episode(&mut self, ep: &EpisodeSummary, at: Boundary, read: impl FnOnce() -> Vec<Metric>) {
        let started = Instant::now();
        let metrics = if self.wants_metrics {
            read()
        } else {
            Vec::new()
        };
        {
            let mut r = lock_reporter(self.reporter);
            r.on_episode(self.info, ep);
            if self.wants_metrics {
                r.on_metrics(self.info, at, &metrics);
            }
        }
        self.spent += started.elapsed();
    }

    /// The metrics `read` returns at `at`, with no episode.
    fn metrics(&mut self, at: Boundary, read: impl FnOnce() -> Vec<Metric>) {
        if !self.wants_metrics {
            return;
        }
        let started = Instant::now();
        let metrics = read();
        lock_reporter(self.reporter).on_metrics(self.info, at, &metrics);
        self.spent += started.elapsed();
    }

    /// Seconds since `start`, less the time spent reporting.
    fn wall_clock(&self, start: Instant) -> f64 {
        start.elapsed().saturating_sub(self.spent).as_secs_f64()
    }
}

/// Plays one episode from a fresh `reset`: at most `max_steps` steps, each
/// taken from `budget`. Returns the episode's raw return, length and how it
/// ended (`Truncated` when cut at `max_steps`), or `None` when the budget ran
/// out before the episode ended.
///
/// The one place the agent is shown a step. Both phases of an
/// [`EpisodicTrial`] go through it, so the status rule holds once: the
/// transition carries the snapshot's status unchanged, and an episode cut at
/// `max_steps` ends `Truncated` for the agent only.
fn play_episode<E, A>(
    env: &mut E,
    agent: &mut A,
    max_steps: usize,
    budget: &mut usize,
    rng: &mut dyn Rng,
    non_finite_steps: &mut u64,
) -> Result<Option<(f64, usize, EpisodeStatus)>, EnvironmentError>
where
    E: Environment,
    A: Agent<E::ObservationType, E::ActionType>,
{
    let mut obs = env.reset()?.observation;
    let mut total_reward = 0.0;
    let mut length = 0;
    loop {
        if length == max_steps {
            agent.on_episode_end(EpisodeStatus::Truncated);
            return Ok(Some((total_reward, length, EpisodeStatus::Truncated)));
        }
        if *budget == 0 {
            return Ok(None);
        }
        let action = agent.act(&obs, rng);
        let snap = env.step(action.clone())?;
        *budget -= 1;
        let reward = snap.reward.value();
        if !reward.is_finite() {
            *non_finite_steps += 1;
        }
        total_reward += f64::from(reward);
        length += 1;
        // Read the status before moving the observation out of the snapshot.
        let status = snap.status;
        agent.on_transition(
            Transition {
                obs: &obs,
                action: &action,
                reward,
                next_obs: &snap.observation,
                status,
            },
            rng,
        );
        obs = snap.observation;
        if status.is_done() {
            agent.on_episode_end(status);
            return Ok(Some((total_reward, length, status)));
        }
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
/// and reports the probe's [`emit_metrics`](GenerationProbe::emit_metrics),
/// read once after the last generation, plus a generation count. After every
/// generation it also reads them and hands them to the reporter's
/// [`on_metrics`](Reporter::on_metrics) at a [`Boundary::Generation`], unless
/// no sink [wants metrics](Reporter::wants_metrics); that time is left out of
/// the trial's wall clock.
///
/// # The cap
///
/// The trial stops when the probe returns `None` or after
/// [`EvaluatorConfig::max_generations`] generations, whichever comes first. The cap
/// is a backstop against a probe that never returns `None`. `max_generations`
/// defaults to 1000, so a probe with a larger budget is cut short unless
/// `max_generations` is raised to at least that budget. The `generations` scalar
/// in the report is the number of generations the report covers: the
/// probe's budget, or the cap when the cap ended the trial.
///
/// A trial the cap ends is not an error, but it is not the run that was asked
/// for either, so the trial emits one `tracing::warn!` naming the
/// environment, the trial and the cap. A budget equal to the cap is not cut
/// short and does not warn.
///
/// To tell the two apart the trial has to ask the probe, and `advance` is the
/// only question [`GenerationProbe`] has. So when the count reaches the cap,
/// `advance` is called once more. An exhausted probe returns `None` and does
/// no work, which the trait's contract requires. A probe with budget left
/// runs one more generation, and that generation is dropped: it is in neither
/// the `generations` scalar, the reported metrics nor the wall clock, because
/// the metrics are read and the clock stopped before that call. A fitness
/// function that counts its own calls sees it.
///
/// # What a probe must be
///
/// `P` is `Send`, because trials run on a thread pool.
/// [`Evaluator::run_trials`] builds one trial per seed from a closure, so
/// each trial owns its own probe.
///
/// # Why no episodes
///
/// [`TrialReport::episodes`] is left **empty**, deliberately. A probe re-seeds
/// on `begin`, so a second pass would be a byte-identical replay — duplicated
/// compute carrying zero information. Rather than fabricate one synthetic
/// `EpisodeSummary` per run — which would make `num_episodes > 1` silently
/// multiply cost for no signal, the trap the retired `BenchEnv` path had —
/// this trial reports no episodes at all and `on_episode` never fires.
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
{
    fn run(
        mut self,
        cfg: &EvaluatorConfig,
        info: &TrialInfo,
        reporter: &Mutex<&mut dyn Reporter>,
    ) -> TrialReport {
        let mut report = TrialReport::new(info.key, info.env_id.clone(), info.trial_seed);
        let start = Instant::now();
        let mut port = TrialPort::new(reporter, info);

        self.probe.begin();

        let mut units = 0usize;
        // `max_generations` is a backstop only: the probe owns its own budget and
        // ends the loop by returning `None`. The cap exists so a probe with a
        // broken exhaustion check cannot hang the suite.
        while units < cfg.max_generations && self.probe.advance().is_some() {
            units += 1;
            port.metrics(Boundary::Generation { generation: units }, || {
                self.probe.emit_metrics()
            });
        }

        let wall = port.wall_clock(start);
        // Read before the cap check below, which may run one more generation
        // that this report must not describe.
        let probe_metrics = self.probe.emit_metrics();

        // The loop above ends at the cap exactly when `units == cfg.max_generations`:
        // a probe that returns `None` breaks out before the count reaches it.
        // Whether the probe had more to give is something only the probe
        // knows, and `advance` is the one way to ask, so it is called once
        // more and what it returns is dropped. On an exhausted probe that call
        // does no work (the `GenerationProbe` contract), which is what keeps a
        // budget equal to the cap silent. It comes after the wall clock and the
        // probe's metrics are read, so the dropped generation is in no number
        // this trial reports.
        if units == cfg.max_generations && self.probe.advance().is_some() {
            tracing::warn!(
                target: "rlevo_harness",
                env = %info.env_id,
                trial = info.key.trial_idx,
                max_generations = cfg.max_generations,
                "generation trial stopped at max_generations before its probe was exhausted; \
                 set EvaluatorConfig::max_generations to at least the probe's budget \
                 (a generation trial ignores max_steps)"
            );
        }

        // Same provenance split as `EpisodicTrial::run`: the two scalars below
        // are this trial's own measurements, while the probe's metrics are
        // checked against the harness-owned namespace.
        report.absorb_harness_metrics(vec![
            Metric::Scalar {
                name: GENERATIONS.to_string(),
                #[allow(clippy::cast_precision_loss)]
                value: units as f64,
            },
            Metric::Scalar {
                name: WALL_CLOCK_SECONDS.to_string(),
                value: wall,
            },
        ]);
        report.absorb_metrics(probe_metrics);
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

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id};
    use tracing::{Event as TracingEvent, Level, Subscriber};
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;

    use rlevo_core::agent::{Agent, Transition};
    use rlevo_core::base::{Action, Observation};
    use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, Snapshot};
    use rlevo_core::fitness::Metric;
    use rlevo_core::reward::ScalarReward;

    use super::{EpisodicTrial, Evaluator, EvaluatorConfig, Trial};
    use crate::metrics::{RETURN_MEAN, RETURN_NON_FINITE_STEPS};
    use crate::report::{BenchmarkReport, EpisodeSummary, Phase, TrialReport};
    use crate::reporter::{Boundary, Reporter};
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
    impl Observation for StubObs {
        const SHAPE: &'static [usize] = &[1];
    }

    #[derive(Debug, Clone)]
    struct StubAction;
    impl Action for StubAction {
        fn is_valid(&self) -> bool {
            true
        }
    }

    impl Environment for StubEnv {
        type ObservationType = StubObs;
        type ActionType = StubAction;

        fn reset(&mut self) -> Result<Snapshot<StubObs>, EnvironmentError> {
            if self.reset_fails {
                return Err(EnvironmentError::InvalidAction("stub reset fail".into()));
            }
            self.steps_taken = 0;
            Ok(Snapshot::running(StubObs, ScalarReward(0.0)))
        }

        fn step(&mut self, _action: StubAction) -> Result<Snapshot<StubObs>, EnvironmentError> {
            if self.step_fails {
                return Err(EnvironmentError::InvalidAction("stub step fail".into()));
            }
            self.steps_taken += 1;
            let obs = StubObs;
            let reward = ScalarReward(self.reward_override.unwrap_or(1.0));
            if self.steps_taken >= self.steps_until_done {
                Ok(Snapshot::terminated(obs, reward))
            } else {
                Ok(Snapshot::running(obs, reward))
            }
        }
    }

    // ── Stub agent ───────────────────────────────────────────────────────────

    struct StubAgent;

    impl Agent<StubObs, StubAction> for StubAgent {
        fn act(&mut self, _obs: &StubObs, _rng: &mut dyn Rng) -> StubAction {
            StubAction
        }
    }

    /// A stub agent that emits a fixed metric list at trial end.
    ///
    /// `StubAgent` takes the `Agent::emit_metrics` default (an empty
    /// `Vec`), so no test driving it can observe the order in which the trial
    /// absorbs harness metrics and agent metrics. This one can.
    struct EmittingAgent {
        metrics: Vec<Metric>,
    }

    impl Agent<StubObs, StubAction> for EmittingAgent {
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
        /// An episode's phase, index and status.
        Episode(Phase, usize, Option<EpisodeStatus>),
        /// A metrics read: where, and how many metrics it carried.
        Metrics(Boundary, usize),
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
        fn on_episode(&mut self, _: &TrialInfo, ep: &EpisodeSummary) {
            self.log
                .lock()
                .unwrap()
                .push(Event::Episode(ep.phase, ep.episode_idx, ep.status));
        }
        fn on_metrics(&mut self, _: &TrialInfo, at: Boundary, metrics: &[Metric]) {
            self.log
                .lock()
                .unwrap()
                .push(Event::Metrics(at, metrics.len()));
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
    /// `warn!` carries; `key` is the colliding metric name carried by the
    /// displacement `warn!` in `TrialReport::absorb_metrics`; `max_generations` is
    /// the cap carried by the `warn!` a cut-short `GenerationTrial` emits,
    /// beside `env`. All four are `Option` because no event carries all of
    /// them.
    #[derive(Debug, Clone)]
    struct CapturedEvent {
        level: Level,
        env: Option<String>,
        non_finite_steps: Option<u64>,
        key: Option<String>,
        max_generations: Option<u64>,
    }

    thread_local! {
        /// Per-thread sink [`CaptureLayer`] appends to, `Some` only while
        /// [`capturing`] is on the stack for this thread.
        static SINK: RefCell<Option<Vec<CapturedEvent>>> = const { RefCell::new(None) };
    }

    /// The fields of a `trial` span, as `run_trials` opens it, and whether it
    /// had a parent span.
    #[derive(Debug, Clone, Default, PartialEq)]
    struct TrialSpan {
        run: String,
        env: String,
        env_idx: u64,
        trial: u64,
        seed: u64,
        nested: bool,
    }

    impl Visit for TrialSpan {
        fn record_u64(&mut self, field: &Field, value: u64) {
            match field.name() {
                "env_idx" => self.env_idx = value,
                "trial" => self.trial = value,
                "seed" => self.seed = value,
                _ => {}
            }
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "run" {
                self.run = value.to_string();
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "env" {
                self.env = format!("{value:?}");
            }
        }
    }

    /// Environment names starting with this are the span tests'; only their
    /// events are kept in [`SPANNED`], so it holds what those tests emit and
    /// nothing else.
    const SPAN_TEST_ENV: &str = "span-test";

    /// Every event, on any thread, whose `env` starts with [`SPAN_TEST_ENV`],
    /// with the innermost `trial` span it was emitted in, if any.
    /// Process-wide; each test takes out its own environment's entries.
    static SPANNED: Mutex<Vec<(String, Option<TrialSpan>)>> = Mutex::new(Vec::new());

    /// Removes and returns the `trial` spans of `env`'s events, in trial order.
    fn take_spanned(env: &str) -> Vec<TrialSpan> {
        let mut all = SPANNED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mine, rest): (Vec<_>, Vec<_>) = all.drain(..).partition(|(e, _)| e == env);
        *all = rest;
        drop(all);
        let mut spans: Vec<TrialSpan> = mine
            .into_iter()
            .map(|(_, span)| span.expect("the warn! is emitted inside a trial span"))
            .collect();
        spans.sort_by_key(|s| s.trial);
        spans
    }

    /// Records every event whose thread has an active [`capturing`] scope.
    ///
    /// Mirrors `rlevo_test_support::capture::FieldCapture`, which is not used
    /// here for two reasons: `rlevo-test-support` pulls in `burn` and
    /// `rlevo-environments`, which this module's stubs exist to avoid, and it
    /// collects a single integer field, so it could not observe the `env` name.
    struct CaptureLayer;

    impl<S> Layer<S> for CaptureLayer
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            if attrs.metadata().name() == "trial"
                && let Some(span) = ctx.span(id)
            {
                let mut fields = TrialSpan {
                    nested: span.parent().is_some(),
                    ..TrialSpan::default()
                };
                attrs.record(&mut fields);
                span.extensions_mut().insert(fields);
            }
        }

        fn on_event(&self, event: &TracingEvent<'_>, ctx: Context<'_, S>) {
            let mut fields = EventVisitor {
                env: None,
                non_finite_steps: None,
                key: None,
                max_generations: None,
            };
            event.record(&mut fields);
            if let Some(env) = fields.env.filter(|e| e.starts_with(SPAN_TEST_ENV)) {
                // Innermost first: the trial whose work emitted the event.
                let trial = ctx.event_scope(event).and_then(|mut scope| {
                    scope
                        .find(|span| span.name() == "trial")
                        .and_then(|span| span.extensions().get::<TrialSpan>().cloned())
                });
                // A failed assertion elsewhere must not poison every later
                // test's events.
                SPANNED
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push((env, trial));
            }
            SINK.with(|sink| {
                let mut sink = sink.borrow_mut();
                let Some(events) = sink.as_mut() else { return };
                let mut visitor = EventVisitor {
                    env: None,
                    non_finite_steps: None,
                    key: None,
                    max_generations: None,
                };
                event.record(&mut visitor);
                events.push(CapturedEvent {
                    level: *event.metadata().level(),
                    env: visitor.env,
                    non_finite_steps: visitor.non_finite_steps,
                    key: visitor.key,
                    max_generations: visitor.max_generations,
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

    /// Extracts the named fields these tests assert on from one event.
    ///
    /// `non_finite_steps` is a `u64` and `max_generations` a `usize`, which
    /// `tracing` widens to one; both arrive through `record_u64`. `env`
    /// and `key` are recorded with `%` (`tracing::field::display`), which routes
    /// through `record_debug` with a `format_args!` payload — whose `Debug` is
    /// its `Display`, so the formatted string is the value verbatim.
    struct EventVisitor {
        env: Option<String>,
        non_finite_steps: Option<u64>,
        key: Option<String>,
        max_generations: Option<u64>,
    }

    impl Visit for EventVisitor {
        fn record_u64(&mut self, field: &Field, value: u64) {
            match field.name() {
                "non_finite_steps" => self.non_finite_steps = Some(value),
                "max_generations" => self.max_generations = Some(value),
                _ => {}
            }
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            match field.name() {
                "env" => self.env = Some(value.to_string()),
                "key" => self.key = Some(value.to_string()),
                _ => {}
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            match field.name() {
                "env" => self.env = Some(format!("{value:?}")),
                "key" => self.key = Some(format!("{value:?}")),
                _ => {}
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
    ///
    /// Generic over the agent so the same helper serves both warning families:
    /// pass `StubAgent` for the non-finite-reward line, or an `EmittingAgent`
    /// for the metric-displacement line, which only fires when the agent emits
    /// a harness-owned name.
    fn run_trial_capturing_events<E, A>(
        env: E,
        agent: A,
        cfg: &EvaluatorConfig,
        env_id: &str,
    ) -> (TrialReport, Vec<CapturedEvent>)
    where
        E: Environment<ObservationType = StubObs, ActionType = StubAction>,
        A: Agent<StubObs, StubAction>,
    {
        let info = TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_id: env_id.to_string(),
            trial_seed: 0,
        };
        let (mut reporter, _) = SpyReporter::new();
        let dyn_reporter: &mut dyn Reporter = &mut reporter;
        let lock = Mutex::new(dyn_reporter);
        let trial = EpisodicTrial {
            env,
            agent,
            agent_seed: 0,
        };

        capturing(|| trial.run(cfg, &info, &lock))
    }

    fn single_thread_cfg(num_episodes: usize, max_steps: usize) -> EvaluatorConfig {
        EvaluatorConfig {
            num_episodes,
            train_steps: 0,
            num_trials_per_env: 1,
            max_steps,
            max_generations: 1_000,
            base_seed: 42,
            num_threads: Some(1),
            checkpoint_dir: None,
            fail_fast: false,
            success_threshold: None,
        }
    }

    /// A single-threaded configuration for one generation trial capped at
    /// `max_generations`.
    fn generation_cfg(max_generations: usize) -> EvaluatorConfig {
        EvaluatorConfig {
            max_generations,
            ..single_thread_cfg(1, 1_000)
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn one_trial_per_env_in_report() {
        let cfg = single_thread_cfg(1, 10);
        let suite = Suite::new("s")
            .with_env("e1", |_| StubEnv::new(3))
            .with_env("e2", |_| StubEnv::new(3));

        let (mut reporter, _log) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        assert_eq!(report.trials.len(), 2);
    }

    #[test]
    fn report_is_finalized_after_run() {
        let cfg = single_thread_cfg(1, 5);
        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(2));

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        assert!(!report.in_progress);
    }

    #[test]
    fn reporter_callback_order_suite_wraps_trials() {
        let cfg = single_thread_cfg(2, 10);
        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(3));

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
    fn on_episode_fires_once_per_episode() {
        let cfg = single_thread_cfg(3, 20);
        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(5));

        let (mut reporter, log) = SpyReporter::new();
        Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let episode_count = log
            .lock()
            .unwrap()
            .iter()
            .filter(|e| matches!(e, Event::Episode(..)))
            .count();
        assert_eq!(episode_count, 3);
    }

    /// Every episode of both phases reaches the reporter, each followed by the
    /// agent's metrics at its boundary, with `step` counting across the two
    /// phases. Budget 10 over 3-step episodes: three training episodes end at
    /// steps 3, 6 and 9, a fourth is left open at step 10 and not reported,
    /// and the two evaluation episodes end at 13 and 16.
    #[test]
    fn both_phases_reach_the_reporter_with_their_metrics() {
        let cfg = EvaluatorConfig {
            train_steps: 10,
            ..single_thread_cfg(2, 20)
        };
        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(3));
        let metric = Metric::Scalar {
            name: "rl/probe".to_string(),
            value: 1.0,
        };

        let (mut reporter, log) = SpyReporter::new();
        Evaluator::new(cfg).run_suite(
            &suite,
            |_| EmittingAgent {
                metrics: vec![metric.clone()],
            },
            &mut reporter,
        );

        let events = log.lock().unwrap().clone();
        let start = events.iter().position(|e| e == &Event::TrialStart).unwrap();
        let end = events.iter().position(|e| e == &Event::TrialEnd).unwrap();
        let done = Some(EpisodeStatus::Terminated);
        let train = |index, step| Event::Metrics(Boundary::TrainingEpisode { index, step }, 1);
        let eval = |index, step| Event::Metrics(Boundary::EvaluationEpisode { index, step }, 1);
        assert_eq!(
            events[start + 1..end],
            [
                Event::Episode(Phase::Training, 0, done),
                train(0, 3),
                Event::Episode(Phase::Training, 1, done),
                train(1, 6),
                Event::Episode(Phase::Training, 2, done),
                train(2, 9),
                Event::Episode(Phase::Evaluation, 0, done),
                eval(0, 13),
                Event::Episode(Phase::Evaluation, 1, done),
                eval(1, 16),
            ]
        );
    }

    /// Every trial of an environment registered with `with_task` carries its
    /// task's hash, including an errored trial, and the report the full
    /// fingerprint once per unit; one registered with `with_env` carries
    /// none. The report and every trial name this build.
    #[cfg(feature = "json")]
    #[test]
    fn trials_carry_their_task_and_build() {
        use crate::report::Provenance;

        #[derive(serde::Serialize)]
        struct Config {
            steps: usize,
            seed: u64,
        }
        let cfg = EvaluatorConfig {
            num_trials_per_env: 2,
            ..single_thread_cfg(1, 10)
        };
        let suite = Suite::new("s")
            .with_task(
                "tasked",
                Config { steps: 3, seed: 0 },
                |c: &Config, _seed| StubEnv::new(c.steps),
            )
            .unwrap()
            .with_env("broken", |_| {
                let mut e = StubEnv::new(3);
                e.reset_fails = true;
                e
            })
            .with_task(
                "broken-tasked",
                Config { steps: 4, seed: 0 },
                |c: &Config, _seed| {
                    let mut e = StubEnv::new(c.steps);
                    e.step_fails = true;
                    e
                },
            )
            .unwrap();

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let task = |steps| crate::task::task_fingerprint(&Config { steps, seed: 1 }).unwrap();
        assert_eq!(report.tasks, [Some(task(3)), None, Some(task(4))]);
        let hash = |steps| Some(task(steps).hash);
        let trials: Vec<(usize, bool, Option<String>)> = report
            .trials
            .iter()
            .map(|t| (t.key.env_idx, t.errored, t.task_hash.clone()))
            .collect();
        assert_eq!(
            trials,
            [
                (0, false, hash(3)),
                (0, false, hash(3)),
                (1, true, None),
                (1, true, None),
                (2, true, hash(4)),
                (2, true, hash(4)),
            ]
        );
        assert_eq!(report.provenance, Some(Provenance::this_build()));
        assert!(
            report
                .trials
                .iter()
                .all(|t| t.provenance == Some(Provenance::this_build()))
        );
    }

    /// A run resumed from a 0.6.1 checkpoint, which named no build, names this
    /// one as the report's, keeps `None` on the trial it skips (it did not run
    /// it) and names this build on the one it runs again.
    #[cfg(feature = "json")]
    #[test]
    fn a_resumed_run_keeps_each_trials_build() {
        use crate::report::Provenance;

        let dir = tempfile::tempdir().unwrap();
        std::fs::copy(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/checkpoint-0.6.1.json"
            ),
            crate::checkpoint::checkpoint_path(dir.path(), "fixture-suite"),
        )
        .unwrap();
        let cfg = EvaluatorConfig {
            num_trials_per_env: 2,
            checkpoint_dir: Some(dir.path().to_path_buf()),
            ..single_thread_cfg(1, 10)
        };
        let suite = Suite::new("fixture-suite").with_env("cartpole", |_| StubEnv::new(3));
        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        assert_eq!(report.provenance, Some(Provenance::this_build()));
        // The fixture's trial 0 finished; its trial 1 errored, so it runs again
        // (the errored record stays beside the new one, as before 0.7.0).
        let builds: Vec<(usize, bool, Option<Provenance>)> = report
            .trials
            .iter()
            .map(|t| (t.key.trial_idx, t.errored, t.provenance.clone()))
            .collect();
        assert_eq!(
            builds,
            [
                (0, false, None),
                (1, true, None),
                (1, false, Some(Provenance::this_build())),
            ]
        );
    }

    /// A checkpointed trial counts as done only for the same task: resumed
    /// with the same config it is skipped; with a changed config every trial
    /// runs again, and the report holds the new task only.
    #[cfg(feature = "json")]
    #[test]
    fn a_resumed_run_reruns_trials_of_a_changed_task() {
        #[derive(serde::Serialize)]
        struct Config {
            steps: usize,
        }
        let dir = tempfile::tempdir().unwrap();
        let cfg = EvaluatorConfig {
            num_trials_per_env: 2,
            checkpoint_dir: Some(dir.path().to_path_buf()),
            ..single_thread_cfg(1, 10)
        };
        let run = |steps: usize| {
            let suite = Suite::new("resume")
                .with_task("e", Config { steps }, |c: &Config, _| StubEnv::new(c.steps))
                .unwrap();
            let (mut reporter, log) = SpyReporter::new();
            let report =
                Evaluator::new(cfg.clone()).run_suite(&suite, |_| StubAgent, &mut reporter);
            let started = log
                .lock()
                .unwrap()
                .iter()
                .filter(|e| **e == Event::TrialStart)
                .count();
            (report, started)
        };

        let (_, started) = run(3);
        assert_eq!(started, 2, "a fresh run runs both trials");
        let (same, started) = run(3);
        assert_eq!(started, 0, "the same task resumes with nothing to run");
        assert_eq!(same.trials.len(), 2);

        let (changed, started) = run(4);
        assert_eq!(started, 2, "a changed task runs every trial again");
        let new_hash = crate::task::task_fingerprint(&Config { steps: 4 })
            .unwrap()
            .hash;
        assert_eq!(changed.trials.len(), 2, "the old task's trials are gone");
        assert!(
            changed
                .trials
                .iter()
                .all(|t| t.task_hash.as_deref() == Some(new_hash.as_str()))
        );
        assert_eq!(changed.tasks[0].as_ref().map(|t| &t.hash), Some(&new_hash));
    }

    /// `run_trials` gives each trial its unit's id and task: a unit built
    /// from an id alone records no task; one built with `with_task`, as a
    /// landscape's caller would, records its fingerprint once and its hash on
    /// every trial.
    #[cfg(feature = "json")]
    #[test]
    fn run_trials_records_each_units_task() {
        use crate::suite::TrialUnit;

        #[derive(serde::Serialize)]
        struct Landscape {
            name: &'static str,
            dim: usize,
        }
        let tasked = TrialUnit::with_task(
            "sphere-d4",
            &Landscape {
                name: "sphere",
                dim: 4,
            },
        )
        .unwrap();
        let cfg = EvaluatorConfig {
            num_trials_per_env: 2,
            ..single_thread_cfg(1, 10)
        };
        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_trials(
            "s",
            &["plain".into(), tasked.clone()],
            |_, _, agent_seed| EpisodicTrial {
                env: StubEnv::new(3),
                agent: StubAgent,
                agent_seed,
            },
            &mut reporter,
        );
        assert_eq!(report.tasks, [None, tasked.task.clone()]);
        let hash = tasked.task.as_ref().map(|t| t.hash.as_str());
        let got: Vec<(&str, Option<&str>)> = report
            .trials
            .iter()
            .map(|t| (t.env_id.as_str(), t.task_hash.as_deref()))
            .collect();
        assert_eq!(
            got,
            [
                ("plain", None),
                ("plain", None),
                ("sphere-d4", hash),
                ("sphere-d4", hash)
            ]
        );
    }

    /// An episode the trial cuts at `max_steps` is reported `Truncated`, in
    /// the reporter's summary and the report's; one the environment ends is
    /// `Terminated`.
    #[test]
    fn an_episode_cut_at_max_steps_is_reported_truncated() {
        let suite = Suite::new("s")
            .with_env("ends", |_| StubEnv::new(3))
            .with_env("cut", |_| StubEnv::new(100));

        let (mut reporter, log) = SpyReporter::new();
        let report =
            Evaluator::new(single_thread_cfg(1, 4)).run_suite(&suite, |_| StubAgent, &mut reporter);

        let statuses: Vec<Option<EpisodeStatus>> =
            report.trials.iter().map(|t| t.episodes[0].status).collect();
        assert_eq!(
            statuses,
            [
                Some(EpisodeStatus::Terminated),
                Some(EpisodeStatus::Truncated)
            ]
        );
        let episodes: Vec<Event> = log
            .lock()
            .unwrap()
            .iter()
            .filter(|e| matches!(e, Event::Episode(..)))
            .cloned()
            .collect();
        assert_eq!(
            episodes,
            [
                Event::Episode(Phase::Evaluation, 0, Some(EpisodeStatus::Terminated)),
                Event::Episode(Phase::Evaluation, 0, Some(EpisodeStatus::Truncated)),
            ]
        );
    }

    /// `train` on its own returns training summaries with their status.
    #[test]
    fn train_returns_training_summaries() {
        let mut rng = StdRng::seed_from_u64(0);
        let episodes = super::train(&mut StubEnv::new(3), &mut StubAgent, 7, 2, &mut rng)
            .expect("the stub never fails");
        let got: Vec<(Phase, usize, usize, Option<EpisodeStatus>)> = episodes
            .iter()
            .map(|e| (e.phase, e.episode_idx, e.length, e.status))
            .collect();
        let cut = Some(EpisodeStatus::Truncated);
        assert_eq!(
            got,
            [
                (Phase::Training, 0, 2, cut),
                (Phase::Training, 1, 2, cut),
                (Phase::Training, 2, 2, cut),
            ],
            "budget 7 at max_steps 2: three cut episodes, the seventh step left open"
        );
    }

    #[test]
    fn episode_return_equals_steps_taken() {
        // StubEnv rewards 1.0 per step and signals done after `steps_until_done`.
        let cfg = single_thread_cfg(4, 100);
        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(7));

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
        impl Observation for CountingObs {
            const SHAPE: &'static [usize] = &[1];
        }

        struct CountingEnv {
            steps: usize,
        }
        impl Environment for CountingEnv {
            type ObservationType = CountingObs;
            type ActionType = StubAction;

            fn reset(&mut self) -> Result<Snapshot<CountingObs>, EnvironmentError> {
                self.steps = 0;
                Ok(Snapshot::running(CountingObs, ScalarReward(0.0)))
            }

            fn step(&mut self, _a: StubAction) -> Result<Snapshot<CountingObs>, EnvironmentError> {
                self.steps += 1;
                let obs = CountingObs;
                let r = ScalarReward(1.0);
                if self.steps >= 3 {
                    Ok(Snapshot::terminated(obs, r))
                } else {
                    Ok(Snapshot::running(obs, r))
                }
            }
        }

        struct CountingAgent;
        impl Agent<CountingObs, StubAction> for CountingAgent {
            fn act(&mut self, _obs: &CountingObs, _rng: &mut dyn Rng) -> StubAction {
                StubAction
            }
        }

        CLONES.store(0, AtomicOrdering::Relaxed);
        let cfg = single_thread_cfg(2, 50);
        let suite = Suite::new("s").with_env("e", |_| CountingEnv { steps: 0 });
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
        let suite = Suite::new("s").with_env("e", |_| {
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
        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(3));

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
        let suite = Suite::new("s").with_env("e", |_| {
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

    /// An agent that names the harness's exact key no longer takes it
    /// (ADR 0079). The harness's measurement survives, the
    /// agent's value is re-homed under `agent/<name>`, and the displacement is
    /// recorded rather than being silent.
    ///
    /// This inverts the former `agent_metrics_are_absorbed_after_the_harness_counter`,
    /// whose doc comment required an ADR to do so. ADR 0079 is that ADR: the
    /// last-writer-wins contract of `absorb_metrics` is replaced by a
    /// provenance split (`absorb_harness_metrics` privileged,
    /// `absorb_metrics` checked).
    ///
    /// The three values `0`, `HARNESS_COUNT` (8) and `999` stay pairwise
    /// distinct (ADR 0078's discipline) so no assertion here can pass by
    /// coincidence. Mutants killed: an `is_harness_reserved` that ignores
    /// prefixes (first assertion — the agent's 999 would win), a rename that
    /// forgets the `displaced_metrics` push (third), and one that drops the
    /// value instead of re-homing it (second).
    #[test]
    fn an_agent_cannot_overwrite_the_harness_non_finite_count() {
        let trial = poisoned_trial_with_agent_metrics(&[Metric::Counter {
            name: RETURN_NON_FINITE_STEPS.to_string(),
            count: 999,
        }]);

        assert_eq!(
            trial.counters[RETURN_NON_FINITE_STEPS], HARNESS_COUNT,
            "the harness-owned key must keep the harness's own count"
        );
        assert_eq!(
            trial.counters["agent/return/non_finite_steps"], 999,
            "the agent's value is preserved under the `agent/` namespace, not dropped"
        );
        assert!(
            trial
                .displaced_metrics
                .contains(&RETURN_NON_FINITE_STEPS.to_string()),
            "the displacement is recorded, not silent; got {:?}",
            trial.displaced_metrics
        );
    }

    /// A clean absorb leaves `displaced_metrics` empty — the complement that
    /// stops the assertion above passing against a fix that records every key.
    #[test]
    fn no_collision_leaves_displaced_metrics_empty() {
        let trial = poisoned_trial_with_agent_metrics(&[Metric::Counter {
            name: "non_finite_steps".to_string(),
            count: 999,
        }]);

        assert!(
            trial.displaced_metrics.is_empty(),
            "an unreserved agent name is not a displacement; got {:?}",
            trial.displaced_metrics
        );
    }

    /// A probe whose metrics name the two scalars `GenerationTrial` measures
    /// itself must not displace them (ADR 0079).
    ///
    /// This site had **zero** coverage for the hazard before: it absorbed its
    /// harness scalars and then the probe's metrics, last-writer-wins, exactly
    /// like the episodic path. The probe below advances a known number of
    /// times and emits deliberately wrong values for both keys, so a
    /// regression shows up as a wrong number rather than a missing one.
    #[test]
    fn a_probe_cannot_overwrite_the_harness_generation_scalars() {
        use rlevo_core::evaluation::GenerationProbe;

        use super::GenerationTrial;
        use crate::metrics::{GENERATIONS, WALL_CLOCK_SECONDS};

        /// Units the probe advances before reporting exhaustion. Distinct from
        /// the bogus `777.0` / `888.0` below so no assertion passes by
        /// coincidence.
        const UNITS: usize = 5;
        /// `UNITS` as the harness reports it — a scalar.
        const UNITS_F: f64 = 5.0;

        struct ColludingProbe {
            remaining: usize,
        }

        impl GenerationProbe for ColludingProbe {
            type Metrics = ();

            fn begin(&mut self) {
                self.remaining = UNITS;
            }

            fn advance(&mut self) -> Option<Self::Metrics> {
                if self.remaining == 0 {
                    return None;
                }
                self.remaining -= 1;
                Some(())
            }

            fn emit_metrics(&self) -> Vec<Metric> {
                vec![
                    Metric::Scalar {
                        name: GENERATIONS.to_string(),
                        value: 777.0,
                    },
                    Metric::Scalar {
                        name: WALL_CLOCK_SECONDS.to_string(),
                        value: 888.0,
                    },
                    // An unreserved name from the probe's own namespace, to
                    // pin that legitimate probe metrics are untouched.
                    Metric::Scalar {
                        name: "ea/best_fitness".to_string(),
                        value: 1.5,
                    },
                ]
            }
        }

        let info = TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_id: "probe".to_string(),
            trial_seed: 0,
        };
        let (mut reporter, _) = SpyReporter::new();
        let dyn_reporter: &mut dyn Reporter = &mut reporter;
        let lock = Mutex::new(dyn_reporter);
        // `max_generations` bounds the loop, so a broken exhaustion check fails the
        // count assertion instead of hanging.
        let cfg = generation_cfg(50);

        let report = GenerationTrial {
            probe: ColludingProbe { remaining: 0 },
        }
        .run(&cfg, &info, &lock);

        assert!(
            (report.scalars[GENERATIONS] - UNITS_F).abs() < f64::EPSILON,
            "the harness's own generation count must survive the probe's metric"
        );
        assert!(
            (report.scalars["agent/generations"] - 777.0).abs() < f64::EPSILON,
            "the probe's value is re-homed under `agent/`, not dropped"
        );
        assert!(
            (report.scalars["agent/wall_clock_seconds"] - 888.0).abs() < f64::EPSILON,
            "the wall-clock key is harness-owned too"
        );
        assert!(
            report.scalars[WALL_CLOCK_SECONDS] < 888.0,
            "the measured wall clock, not the probe's 888 seconds"
        );
        assert!(
            (report.scalars["ea/best_fitness"] - 1.5).abs() < f64::EPSILON,
            "`ea/*` is probe-owned and must be recorded verbatim"
        );
        assert_eq!(
            report.displaced_metrics,
            vec![GENERATIONS.to_string(), WALL_CLOCK_SECONDS.to_string()],
            "both displacements recorded, in emission order, and nothing else"
        );
    }

    /// Every key `GenerationTrial` emits on the **privileged** path must be
    /// harness-reserved — the counterpart of `core_metrics`'
    /// `every_core_metric_key_is_harness_reserved`, for the other harness call
    /// site.
    ///
    /// Enumerated, not hand-listed. The probe below emits **no** metrics, so
    /// every key the finished `TrialReport` carries came from
    /// `absorb_harness_metrics` inside `GenerationTrial::run`. A third harness
    /// scalar added there without a matching reservation therefore fails here
    /// the moment it is written; a hand-written list would keep passing while
    /// leaving the new key open to being clobbered by any probe that guesses
    /// the name.
    ///
    /// The two `contains_key` assertions guard against the vacuous pass where a
    /// refactor stops the trial emitting anything at all and the loop below
    /// iterates zero times.
    #[test]
    fn every_generation_trial_harness_key_is_harness_reserved() {
        use rlevo_core::evaluation::GenerationProbe;

        use super::GenerationTrial;
        use crate::metrics::{GENERATIONS, WALL_CLOCK_SECONDS, is_harness_reserved};

        /// Takes `emit_metrics`' default and emits nothing, so the report
        /// contains harness keys and only those.
        struct SilentProbe {
            remaining: usize,
        }

        impl GenerationProbe for SilentProbe {
            type Metrics = ();

            fn begin(&mut self) {
                self.remaining = 3;
            }

            fn advance(&mut self) -> Option<Self::Metrics> {
                if self.remaining == 0 {
                    return None;
                }
                self.remaining -= 1;
                Some(())
            }
        }

        let info = TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_id: "probe".to_string(),
            trial_seed: 0,
        };
        let (mut reporter, _) = SpyReporter::new();
        let dyn_reporter: &mut dyn Reporter = &mut reporter;
        let lock = Mutex::new(dyn_reporter);
        // `max_generations` bounds the loop, so a broken exhaustion check fails the
        // assertions instead of hanging.
        let cfg = generation_cfg(50);

        let report = GenerationTrial {
            probe: SilentProbe { remaining: 0 },
        }
        .run(&cfg, &info, &lock);

        let emitted: Vec<&String> = report
            .scalars
            .keys()
            .chain(report.histograms.keys())
            .chain(report.counters.keys())
            .collect();

        for name in &emitted {
            assert!(
                is_harness_reserved(name),
                "GenerationTrial emits `{name}` as its own measurement, but it is not \
                 reserved — a probe could clobber it"
            );
        }
        assert!(
            report.scalars.contains_key(GENERATIONS),
            "the trial must actually have emitted its generation count; got {emitted:?}"
        );
        assert!(
            report.scalars.contains_key(WALL_CLOCK_SECONDS),
            "the trial must actually have emitted its wall clock; got {emitted:?}"
        );
        assert!(
            report.displaced_metrics.is_empty(),
            "a silent probe contests nothing, so nothing is re-homed"
        );
    }

    /// Each trial's work on its thread runs inside a `trial` span carrying
    /// its run, environment, indices and seed, with trials on two threads. The
    /// non-finite-reward `warn!` each trial emits is the event observed.
    #[test]
    fn every_trial_runs_inside_its_trial_span() {
        install_capture_subscriber();
        let env = "span-test-flat";
        let cfg = EvaluatorConfig {
            num_trials_per_env: 2,
            num_threads: Some(2),
            ..single_thread_cfg(1, 10)
        };
        let suite = Suite::new("s").with_env(env, |_| {
            let mut e = StubEnv::new(2);
            e.reward_override = Some(f32::NAN);
            e
        });

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let spans = take_spanned(env);
        let expected: Vec<TrialSpan> = report
            .trials
            .iter()
            .map(|t| TrialSpan {
                run: report.run_id.clone().expect("a run has an ID"),
                env: env.to_string(),
                env_idx: t.key.env_idx as u64,
                trial: t.key.trial_idx as u64,
                seed: t.trial_seed,
                nested: false,
            })
            .collect();
        assert_eq!(spans, expected);
    }

    /// A trial's span is a root even when the thread that runs it is already
    /// inside a span, as it is when rayon runs a trial on a worker that is
    /// waiting on another trial's parallel work. Here the suite runs from
    /// inside a one-thread pool with an outer span entered, so each trial
    /// runs on that thread under it.
    #[test]
    fn a_trial_span_never_nests_under_another_span() {
        install_capture_subscriber();
        let env = "span-test-nested";
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let cfg = EvaluatorConfig {
            num_trials_per_env: 2,
            num_threads: None,
            ..single_thread_cfg(1, 10)
        };
        let suite = Suite::new("s").with_env(env, |_| {
            let mut e = StubEnv::new(2);
            e.reward_override = Some(f32::NAN);
            e
        });

        pool.install(|| {
            let outer = tracing::info_span!("outer");
            let _in_outer = outer.enter();
            let (mut reporter, _) = SpyReporter::new();
            Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);
        });

        let spans = take_spanned(env);
        assert_eq!(spans.len(), 2, "one warn! per trial; got {spans:?}");
        assert!(
            spans.iter().all(|s| !s.nested),
            "a trial span must have no parent; got {spans:?}"
        );
    }

    /// A reporter that panics in `on_metrics` errors its trial, poisoning the
    /// shared lock, and the suite still runs every other trial and reaches
    /// `on_suite_end`.
    #[test]
    fn a_panicking_reporter_errors_its_trial_not_the_suite() {
        struct PanicsOnce {
            panicked: bool,
            suite_ended: Arc<Mutex<bool>>,
        }
        impl Reporter for PanicsOnce {
            fn on_suite_start(&mut self, _: &SuiteInfo) {}
            fn on_trial_start(&mut self, _: &TrialInfo) {}
            fn on_episode(&mut self, _: &TrialInfo, _: &EpisodeSummary) {}
            fn on_metrics(&mut self, _: &TrialInfo, _: Boundary, _: &[Metric]) {
                if !self.panicked {
                    self.panicked = true;
                    panic!("a sink defect");
                }
            }
            fn on_trial_end(&mut self, _: &TrialInfo, _: &TrialReport) {}
            fn on_suite_end(&mut self, _: &BenchmarkReport) {
                *self.suite_ended.lock().unwrap() = true;
            }
        }

        let cfg = EvaluatorConfig {
            num_trials_per_env: 3,
            ..single_thread_cfg(2, 10)
        };
        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(3));
        let suite_ended = Arc::new(Mutex::new(false));
        let mut reporter = PanicsOnce {
            panicked: false,
            suite_ended: Arc::clone(&suite_ended),
        };
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let errored: Vec<bool> = report.trials.iter().map(|t| t.errored).collect();
        assert_eq!(errored, [true, false, false]);
        assert!(*suite_ended.lock().unwrap());
    }

    /// A sink that does not want metrics gets no `on_metrics` call, and the
    /// agent's metrics are read once, for the report, instead of after every
    /// episode.
    #[test]
    fn metrics_are_not_read_for_a_sink_that_does_not_want_them() {
        struct CountsReads {
            reads: Arc<std::sync::atomic::AtomicUsize>,
        }
        impl Agent<StubObs, StubAction> for CountsReads {
            fn act(&mut self, _obs: &StubObs, _rng: &mut dyn Rng) -> StubAction {
                StubAction
            }
            fn emit_metrics(&self) -> Vec<Metric> {
                self.reads
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Vec::new()
            }
        }
        struct NoMetrics(SpyReporter);
        impl Reporter for NoMetrics {
            fn on_suite_start(&mut self, s: &SuiteInfo) {
                self.0.on_suite_start(s);
            }
            fn on_trial_start(&mut self, t: &TrialInfo) {
                self.0.on_trial_start(t);
            }
            fn on_episode(&mut self, t: &TrialInfo, ep: &EpisodeSummary) {
                self.0.on_episode(t, ep);
            }
            fn on_metrics(&mut self, t: &TrialInfo, at: Boundary, m: &[Metric]) {
                self.0.on_metrics(t, at, m);
            }
            fn wants_metrics(&self) -> bool {
                false
            }
            fn on_trial_end(&mut self, t: &TrialInfo, r: &TrialReport) {
                self.0.on_trial_end(t, r);
            }
            fn on_suite_end(&mut self, r: &BenchmarkReport) {
                self.0.on_suite_end(r);
            }
        }

        let cfg = EvaluatorConfig {
            train_steps: 9,
            ..single_thread_cfg(2, 10)
        };
        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(3));
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (spy, log) = SpyReporter::new();
        let mut reporter = NoMetrics(spy);
        Evaluator::new(cfg).run_suite(
            &suite,
            |_| CountsReads {
                reads: Arc::clone(&reads),
            },
            &mut reporter,
        );

        let events = log.lock().unwrap().clone();
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Event::Episode(..)))
                .count(),
            5,
            "three training and two evaluation episodes still reach the sink"
        );
        assert!(!events.iter().any(|e| matches!(e, Event::Metrics(..))));
        assert_eq!(
            reads.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "read once, for the report"
        );
    }

    /// Time a sink spends in its callbacks is not the trial's: a sink that
    /// sleeps 40 ms per episode leaves the trial's wall clock near zero.
    #[test]
    fn reporter_time_is_left_out_of_the_wall_clock() {
        use crate::metrics::WALL_CLOCK_SECONDS;

        struct Slow;
        impl Reporter for Slow {
            fn on_suite_start(&mut self, _: &SuiteInfo) {}
            fn on_trial_start(&mut self, _: &TrialInfo) {}
            fn on_episode(&mut self, _: &TrialInfo, _: &EpisodeSummary) {
                std::thread::sleep(std::time::Duration::from_millis(40));
            }
            fn on_trial_end(&mut self, _: &TrialInfo, _: &TrialReport) {}
            fn on_suite_end(&mut self, _: &BenchmarkReport) {}
        }

        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(3));
        let report =
            Evaluator::new(single_thread_cfg(3, 10)).run_suite(&suite, |_| StubAgent, &mut Slow);

        let wall = report.trials[0].scalars[WALL_CLOCK_SECONDS];
        assert!(
            wall < 0.06,
            "three 40 ms callbacks must not count; wall clock {wall} s"
        );
    }

    /// A generation trial hands the probe's metrics to `on_metrics` after
    /// every generation, counted from 1 as the probe's `ea/generation` is, and
    /// never for the generation the cap check may run past the cap.
    #[test]
    fn on_metrics_fires_after_every_generation() {
        use rlevo_core::evaluation::GenerationProbe;

        use super::GenerationTrial;

        struct Probe {
            budget: usize,
            done: usize,
        }

        impl GenerationProbe for Probe {
            type Metrics = ();

            fn begin(&mut self) {
                self.done = 0;
            }

            fn advance(&mut self) -> Option<Self::Metrics> {
                if self.done == self.budget {
                    return None;
                }
                self.done += 1;
                Some(())
            }

            fn emit_metrics(&self) -> Vec<Metric> {
                vec![Metric::Scalar {
                    name: "ea/generation".to_string(),
                    #[allow(clippy::cast_precision_loss)]
                    value: self.done as f64,
                }]
            }
        }

        let info = TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_id: "probe".to_string(),
            trial_seed: 0,
        };
        // (budget, cap, reads expected): exhausted before the cap, and cut at it.
        for (budget, cap, reads) in [(4, 10, 4), (10, 3, 3)] {
            let (mut reporter, log) = SpyReporter::new();
            let dyn_reporter: &mut dyn Reporter = &mut reporter;
            let lock = Mutex::new(dyn_reporter);
            GenerationTrial {
                probe: Probe { budget, done: 0 },
            }
            .run(&generation_cfg(cap), &info, &lock);

            let expected: Vec<Event> = (1..=reads)
                .map(|generation| Event::Metrics(Boundary::Generation { generation }, 1))
                .collect();
            assert_eq!(*log.lock().unwrap(), expected, "budget {budget}, cap {cap}");
        }
    }

    /// Runs a [`GenerationTrial`](super::GenerationTrial) over a probe with
    /// `budget` generations under a cap of `max_generations`.
    ///
    /// Returns the report, every `tracing` event the trial emitted, and the
    /// number of times the trial called `advance`. The probe's metric is the
    /// number of generations it has run when it is read, so a report shows
    /// which generation it was read after.
    fn run_generation_trial_capturing_events(
        budget: usize,
        max_generations: usize,
        env_id: &str,
    ) -> (TrialReport, Vec<CapturedEvent>, usize) {
        use std::cell::Cell;
        use std::rc::Rc;

        use rlevo_core::evaluation::GenerationProbe;

        use super::GenerationTrial;

        struct CountingProbe {
            budget: usize,
            done: usize,
            advances: Rc<Cell<usize>>,
        }

        impl GenerationProbe for CountingProbe {
            type Metrics = ();

            fn begin(&mut self) {
                self.done = 0;
            }

            fn advance(&mut self) -> Option<Self::Metrics> {
                self.advances.set(self.advances.get() + 1);
                if self.done == self.budget {
                    return None;
                }
                self.done += 1;
                Some(())
            }

            fn emit_metrics(&self) -> Vec<Metric> {
                vec![Metric::Scalar {
                    name: "ea/generation".to_string(),
                    #[allow(clippy::cast_precision_loss)]
                    value: self.done as f64,
                }]
            }
        }

        let info = TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_id: env_id.to_string(),
            trial_seed: 0,
        };
        let (mut reporter, _) = SpyReporter::new();
        let dyn_reporter: &mut dyn Reporter = &mut reporter;
        let lock = Mutex::new(dyn_reporter);
        let cfg = generation_cfg(max_generations);
        let advances = Rc::new(Cell::new(0));
        let trial = GenerationTrial {
            probe: CountingProbe {
                budget,
                done: 0,
                advances: Rc::clone(&advances),
            },
        };

        let (report, events) = capturing(|| trial.run(&cfg, &info, &lock));
        (report, events, advances.get())
    }

    /// A probe with budget left when the cap is reached is cut short, and
    /// the trial says so: one `warn!`, naming the environment and the cap.
    ///
    /// The report is the capped run's, not the run that includes the one
    /// generation the trial spends finding out. That generation shows in the
    /// call count and nowhere else.
    #[test]
    fn generation_trial_warns_once_when_the_cap_cuts_its_probe_short() {
        use crate::metrics::GENERATIONS;

        let (report, events, advances) = run_generation_trial_capturing_events(7, 4, "capped");

        let warnings: Vec<&CapturedEvent> =
            events.iter().filter(|e| e.level == Level::WARN).collect();
        assert_eq!(
            warnings.len(),
            1,
            "a trial the cap ends early warns exactly once; got {warnings:?}"
        );
        assert_eq!(
            warnings[0].env.as_deref(),
            Some("capped"),
            "the warning must name the unit whose trial was cut short"
        );
        assert_eq!(
            warnings[0].max_generations,
            Some(4),
            "the warning must carry the cap that ended the trial"
        );
        assert!(
            (report.scalars[GENERATIONS] - 4.0).abs() < f64::EPSILON,
            "the trial reports the generations inside the cap, not the one it \
             ran to find out; got {}",
            report.scalars[GENERATIONS]
        );
        assert!(
            (report.scalars["ea/generation"] - 4.0).abs() < f64::EPSILON,
            "the reported metrics are the fourth generation's; the fifth is \
             dropped; got {}",
            report.scalars["ea/generation"]
        );
        assert_eq!(
            advances, 5,
            "finding out costs one call past the cap, and only one"
        );
    }

    /// A budget the cap does not cut short is silent, including the budget
    /// that equals the cap, which is the case a check on the count alone
    /// gets wrong.
    #[test]
    fn generation_trial_is_silent_when_its_probe_ends_within_the_cap() {
        use crate::metrics::GENERATIONS;

        for budget in [0usize, 3, 4] {
            let (report, events, advances) = run_generation_trial_capturing_events(budget, 4, "ok");

            assert!(
                events.iter().all(|e| e.level != Level::WARN),
                "a budget of {budget} under a cap of 4 is not cut short and \
                 must not warn; got {events:?}"
            );
            #[allow(clippy::cast_precision_loss)]
            let expected = budget as f64;
            assert!(
                (report.scalars[GENERATIONS] - expected).abs() < f64::EPSILON,
                "every generation of a budget of {budget} ran; got {}",
                report.scalars[GENERATIONS]
            );
            assert_eq!(
                advances,
                budget + 1,
                "one call per generation and the one that returned `None`"
            );
        }
    }

    /// One `warn!` per affected trial — not one per affected step, and not
    /// zero (ADR 0078, Decision 3). The line must carry the environment name
    /// and the trial total, since a run read through its log has no other
    /// channel for this failure.
    #[test]
    fn non_finite_reward_warns_once_per_trial_with_env_and_total() {
        let cfg = single_thread_cfg(2, 10);
        let mut env = StubEnv::new(3);
        env.reward_override = Some(f32::INFINITY);

        let (report, events) = run_trial_capturing_events(env, StubAgent, &cfg, "poisoned-env");

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

        let (report, events) =
            run_trial_capturing_events(StubEnv::new(3), StubAgent, &cfg, "clean-env");

        assert!(
            events.iter().all(|e| e.level != Level::WARN),
            "a trial with no non-finite reward must emit no warning"
        );
        assert_eq!(
            report.counters[RETURN_NON_FINITE_STEPS], 0,
            "the counter is still emitted, as zero"
        );
    }

    /// Collects the `key` field of every `WARN` a trial emitted, in order.
    ///
    /// Every environment these callers pass is finite, so the non-finite
    /// summary line never fires and the only warnings left are displacements.
    fn displacement_warning_keys(events: &[CapturedEvent]) -> Vec<String> {
        events
            .iter()
            .filter(|e| e.level == Level::WARN)
            .map(|e| {
                e.key
                    .clone()
                    .expect("every warning on a finite trial is a displacement and carries `key`")
            })
            .collect()
    }

    /// Re-homing an agent metric emits exactly one `warn!` per collision,
    /// naming the colliding key (ADR 0079).
    ///
    /// The rename is otherwise silent at runtime: `displaced_metrics` is only
    /// visible to whoever reads the finished report, so this line is the sole
    /// live signal that an agent is fighting the harness for a name. Deleting
    /// the `warn!` block in `TrialReport::absorb` leaves every other test in
    /// this crate green, which is why the assertion exists here.
    ///
    /// Bounds the count from both sides. Two of the three emitted metrics are
    /// harness-owned and one is not, so:
    /// - a missing `warn!` gives 0,
    /// - one line per *absorbed metric* rather than per collision gives 3
    ///   (or 11, counting the `core_metrics` set the harness absorbs),
    /// - one line per *trial* rather than per collision gives 1.
    ///
    /// Only the correct implementation gives 2, with these keys in this order.
    #[test]
    fn displaced_agent_metrics_warn_once_each_carrying_the_key() {
        use crate::metrics::WALL_CLOCK_SECONDS;

        let cfg = single_thread_cfg(1, 10);
        let agent = EmittingAgent {
            metrics: vec![
                Metric::Scalar {
                    name: RETURN_MEAN.to_string(),
                    value: 999.0,
                },
                // Agent-owned: absorbed verbatim, and must not warn.
                Metric::Scalar {
                    name: "rl/epsilon".to_string(),
                    value: 0.25,
                },
                Metric::Scalar {
                    name: WALL_CLOCK_SECONDS.to_string(),
                    value: 888.0,
                },
            ],
        };

        let (report, events) =
            run_trial_capturing_events(StubEnv::new(3), agent, &cfg, "colliding-env");

        assert_eq!(
            displacement_warning_keys(&events),
            vec![RETURN_MEAN.to_string(), WALL_CLOCK_SECONDS.to_string()],
            "one warning per collision, in emission order, and none for `rl/epsilon`"
        );
        assert_eq!(
            report.displaced_metrics,
            vec![RETURN_MEAN.to_string(), WALL_CLOCK_SECONDS.to_string()],
            "the log line and the recorded field must agree on what was displaced"
        );
    }

    /// The degenerate path warns too: when `agent/<name>` is already taken the
    /// incoming value is **dropped**, so the `warn!` is the only trace it ever
    /// existed — `displaced_metrics` records the collision but no map holds the
    /// number.
    ///
    /// Reached by emitting the same reserved name twice: the first lands in the
    /// free `agent/return/mean` slot, the second finds it occupied.
    #[test]
    fn a_dropped_value_still_warns() {
        let cfg = single_thread_cfg(1, 10);
        let agent = EmittingAgent {
            metrics: vec![
                Metric::Scalar {
                    name: RETURN_MEAN.to_string(),
                    value: 111.0,
                },
                Metric::Scalar {
                    name: RETURN_MEAN.to_string(),
                    value: 222.0,
                },
            ],
        };

        let (report, events) =
            run_trial_capturing_events(StubEnv::new(3), agent, &cfg, "dropping-env");

        assert_eq!(
            displacement_warning_keys(&events),
            vec![RETURN_MEAN.to_string(), RETURN_MEAN.to_string()],
            "the dropped second value is warned about, not swallowed"
        );
        assert!(
            (report.scalars["agent/return/mean"] - 111.0).abs() < f64::EPSILON,
            "first writer into the landing zone keeps the slot"
        );
        assert!(
            (report.scalars[RETURN_MEAN] - 3.0).abs() < f64::EPSILON,
            "the harness's own mean (1 episode x 3 steps x reward 1.0) is untouched"
        );
    }

    #[test]
    fn max_steps_truncates_non_terminating_episode() {
        let cfg = EvaluatorConfig {
            num_episodes: 1,
            train_steps: 0,
            max_steps: 4,
            max_generations: 1_000,
            num_trials_per_env: 1,
            base_seed: 0,
            num_threads: Some(1),
            checkpoint_dir: None,
            fail_fast: false,
            success_threshold: None,
        };
        // done triggers at step 1000 — the evaluator must stop at max_steps=4.
        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(1000));

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let ep = &report.trials[0].episodes[0];
        assert_eq!(ep.length, 4);
        assert!((ep.return_value - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn reset_error_marks_trial_errored() {
        let cfg = single_thread_cfg(5, 10);
        let suite = Suite::new("s").with_env("e", |_| {
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
        let suite = Suite::new("s").with_env("e", |_| {
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
        impl Environment for PanickingEnv {
            type ObservationType = StubObs;
            type ActionType = StubAction;

            fn reset(&mut self) -> Result<Snapshot<StubObs>, EnvironmentError> {
                Ok(Snapshot::running(StubObs, ScalarReward(0.0)))
            }

            fn step(&mut self, _action: StubAction) -> Result<Snapshot<StubObs>, EnvironmentError> {
                panic!("deliberate evaluator test panic");
            }
        }

        let cfg = single_thread_cfg(1, 10);
        let suite = Suite::new("s").with_env("e", |_| PanickingEnv);

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
            train_steps: 0,
            num_trials_per_env: 2,
            max_steps: 5,
            max_generations: 1_000,
            base_seed: 0,
            num_threads: Some(1),
            checkpoint_dir: None,
            fail_fast: false,
            success_threshold: None,
        };
        let suite = Suite::new("s")
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
            train_steps: 0,
            num_trials_per_env: 1,
            max_steps: 10,
            max_generations: 1_000,
            base_seed: 0,
            num_threads: Some(1),
            checkpoint_dir: None,
            fail_fast: false,
            success_threshold: Some(3.0), // StubEnv returns 5.0 per episode
        };
        let suite = Suite::new("s").with_env("e", |_| StubEnv::new(5));

        let (mut reporter, _) = SpyReporter::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        // core_metrics emits "success_rate" when a threshold is set.
        assert!(report.trials[0].scalars.contains_key("success_rate"));
    }

    // ── The agent seam ───────────────────────────────────────────────────────

    /// Ends its episodes on a schedule, cycling: episode `i` lasts
    /// `schedule[i % len].0` steps and ends with `schedule[i % len].1`.
    struct ScheduleEnv {
        schedule: Vec<(usize, EpisodeStatus)>,
        resets: usize,
        steps: usize,
    }

    impl ScheduleEnv {
        fn new(schedule: &[(usize, EpisodeStatus)]) -> Self {
            Self {
                schedule: schedule.to_vec(),
                resets: 0,
                steps: 0,
            }
        }
    }

    impl Environment for ScheduleEnv {
        type ObservationType = StubObs;
        type ActionType = StubAction;

        fn reset(&mut self) -> Result<Snapshot<StubObs>, EnvironmentError> {
            self.resets += 1;
            // Bounds a driver that resets without stepping: a failure, not a
            // hang.
            assert!(self.resets <= 1_000, "reset 1000 times");
            self.steps = 0;
            Ok(Snapshot::running(StubObs, ScalarReward(0.0)))
        }

        fn step(&mut self, _action: StubAction) -> Result<Snapshot<StubObs>, EnvironmentError> {
            self.steps += 1;
            let (length, end) = self.schedule[(self.resets - 1) % self.schedule.len()];
            let status = if self.steps >= length {
                end
            } else {
                EpisodeStatus::Running
            };
            Ok(Snapshot {
                observation: StubObs,
                reward: ScalarReward(1.0),
                status,
                metadata: None,
            })
        }
    }

    /// What an agent was told, in order.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Told {
        Training(bool),
        Step(EpisodeStatus),
        End(EpisodeStatus),
    }

    /// Writes down every hook call, into a log the test keeps.
    struct Recorder(Arc<Mutex<Vec<Told>>>);

    impl Agent<StubObs, StubAction> for Recorder {
        fn act(&mut self, _obs: &StubObs, _rng: &mut dyn Rng) -> StubAction {
            StubAction
        }

        fn on_transition(&mut self, t: Transition<&StubObs, &StubAction>, _rng: &mut dyn Rng) {
            self.0.lock().unwrap().push(Told::Step(t.status));
        }

        fn on_episode_end(&mut self, status: EpisodeStatus) {
            self.0.lock().unwrap().push(Told::End(status));
        }

        fn set_training(&mut self, training: bool) {
            self.0.lock().unwrap().push(Told::Training(training));
        }
    }

    fn run_recorded(env: ScheduleEnv, cfg: &EvaluatorConfig) -> (TrialReport, Vec<Told>, usize) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (report, _) =
            run_trial_capturing_events(env, Recorder(Arc::clone(&log)), cfg, "schedule");
        let told = log.lock().unwrap().clone();
        let episodes = report.episodes.len();
        (report, told, episodes)
    }

    #[test]
    fn trial_passes_each_status_through_and_reports_its_own_cut_as_truncated() {
        use EpisodeStatus::{Running, Terminated, Truncated};
        // The environment ends episode 1 and truncates episode 2 itself;
        // episode 3 would run 10 steps, and the trial cuts it at 5.
        let env = ScheduleEnv::new(&[(2, Terminated), (3, Truncated), (10, Terminated)]);
        let (report, told, _) = run_recorded(env, &single_thread_cfg(3, 5));

        let mut expected = vec![Told::Training(false)];
        expected.extend([
            Told::Step(Running),
            Told::Step(Terminated),
            Told::End(Terminated),
        ]);
        expected.extend([
            Told::Step(Running),
            Told::Step(Running),
            Told::Step(Truncated),
        ]);
        expected.push(Told::End(Truncated));
        // The cut episode's last transition keeps the environment's `Running`:
        // only its end says `Truncated`.
        expected.extend([Told::Step(Running); 5]);
        expected.push(Told::End(Truncated));
        assert_eq!(told, expected);
        let lengths: Vec<usize> = report.episodes.iter().map(|e| e.length).collect();
        assert_eq!(lengths, [2, 3, 5]);
    }

    #[test]
    fn trial_trains_before_it_evaluates_and_reports_evaluation_only() {
        use EpisodeStatus::{Running, Terminated};
        let env = ScheduleEnv::new(&[(2, Terminated)]);
        let cfg = EvaluatorConfig {
            train_steps: 4,
            ..single_thread_cfg(1, 10)
        };
        let (report, told, episodes) = run_recorded(env, &cfg);

        let episode = [
            Told::Step(Running),
            Told::Step(Terminated),
            Told::End(Terminated),
        ];
        let mut expected = vec![Told::Training(true)];
        expected.extend(episode);
        expected.extend(episode);
        expected.push(Told::Training(false));
        expected.extend(episode);
        assert_eq!(told, expected);
        assert_eq!(episodes, 1, "the training episodes are not reported");
        assert_eq!(report.episodes[0].episode_idx, 0);
    }

    #[test]
    fn train_opens_no_episode_its_budget_cannot_step() {
        use EpisodeStatus::Terminated;
        let mut rng = StdRng::seed_from_u64(0);
        for (budget, resets, complete) in [(4, 2, 2), (5, 3, 2)] {
            let mut env = ScheduleEnv::new(&[(2, Terminated)]);
            let log = Arc::new(Mutex::new(Vec::new()));
            let mut agent = Recorder(Arc::clone(&log));
            let episodes =
                super::train(&mut env, &mut agent, budget, usize::MAX, &mut rng).expect("no error");
            assert_eq!(env.resets, resets, "budget {budget}");
            assert_eq!(episodes.len(), complete, "budget {budget}");
            let steps = log
                .lock()
                .unwrap()
                .iter()
                .filter(|told| matches!(told, Told::Step(_)))
                .count();
            assert_eq!(steps, budget);
            assert_eq!(
                log.lock().unwrap().last(),
                Some(&if budget == 4 {
                    Told::End(Terminated)
                } else {
                    Told::Step(EpisodeStatus::Running)
                }),
                "an episode the budget leaves open is not ended",
            );
        }
    }

    /// With `max_steps` at 0 an episode cannot take a step, so it takes
    /// nothing from the budget: `train` must return rather than reset forever.
    #[test]
    fn train_with_max_steps_zero_opens_no_episode() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut env = ScheduleEnv::new(&[(2, EpisodeStatus::Terminated)]);
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut agent = Recorder(Arc::clone(&log));
        let episodes = super::train(&mut env, &mut agent, 10, 0, &mut rng).expect("no error");
        assert!(episodes.is_empty());
        assert_eq!(
            env.resets, 0,
            "no episode can be stepped, so none is opened"
        );
        assert_eq!(*log.lock().unwrap(), vec![Told::Training(true)]);
    }

    // ── Run identity ─────────────────────────────────────────────────────────

    /// What a sink is told: the run ID from `on_suite_start`, and the trials
    /// that ended.
    #[derive(Default)]
    struct SinkLog {
        run_id: Option<String>,
        ended: Vec<TrialKey>,
    }

    /// A sink that keeps what it is told.
    struct KeepsRunId {
        told: Arc<Mutex<SinkLog>>,
    }

    impl KeepsRunId {
        fn new() -> (Self, Arc<Mutex<SinkLog>>) {
            let told = Arc::new(Mutex::new(SinkLog::default()));
            (
                Self {
                    told: Arc::clone(&told),
                },
                told,
            )
        }
    }

    impl Reporter for KeepsRunId {
        fn on_suite_start(&mut self, suite: &SuiteInfo) {
            self.told.lock().unwrap().run_id = Some(suite.run_id.clone());
        }
        fn on_trial_start(&mut self, _: &TrialInfo) {}
        fn on_episode(&mut self, _: &TrialInfo, _: &EpisodeSummary) {}
        fn wants_metrics(&self) -> bool {
            false
        }
        fn on_trial_end(&mut self, trial: &TrialInfo, _: &TrialReport) {
            self.told.lock().unwrap().ended.push(trial.key);
        }
        fn on_suite_end(&mut self, _: &BenchmarkReport) {}
    }

    fn is_run_id(id: &str) -> bool {
        id.len() == 16 && id.bytes().all(|b| b.is_ascii_hexdigit())
    }

    /// A run has an ID of 16 hexadecimal digits, its sinks are given the same
    /// one, and a second run of the same suite and base seed has another.
    #[test]
    fn a_run_has_an_id_its_sinks_are_given() {
        let run = || {
            let suite = Suite::new("s").with_env("e", |_| StubEnv::new(3));
            let (mut reporter, told) = KeepsRunId::new();
            let report = Evaluator::new(single_thread_cfg(1, 10)).run_suite(
                &suite,
                |_| StubAgent,
                &mut reporter,
            );
            let given = told.lock().unwrap().run_id.clone();
            (report.run_id, given)
        };

        let (first, given) = run();
        let first = first.expect("a run has an ID");
        assert!(is_run_id(&first), "{first}");
        assert_eq!(given.as_deref(), Some(first.as_str()));

        let (second, _) = run();
        assert_ne!(second, Some(first), "the ID is not derived from the seed");
    }

    /// A run resumed from its checkpoint is the same run: the report and the
    /// sinks have the ID the first run drew, so a sink that names its output
    /// by the ID adds to what it began.
    #[cfg(feature = "json")]
    #[test]
    fn a_resumed_run_keeps_its_run_id() {
        let dir = tempfile::tempdir().unwrap();
        let run = |trials: usize| {
            let cfg = EvaluatorConfig {
                num_trials_per_env: trials,
                checkpoint_dir: Some(dir.path().to_path_buf()),
                ..single_thread_cfg(1, 10)
            };
            let suite = Suite::new("resume-id").with_env("e", |_| StubEnv::new(3));
            let (mut reporter, told) = KeepsRunId::new();
            let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);
            let told = std::mem::take(&mut *told.lock().unwrap());
            (report, told)
        };

        let (first, _) = run(1);
        let id = first.run_id.expect("a run has an ID");

        // One trial is done; the second run adds one more.
        let (second, told) = run(2);
        assert_eq!(told.ended.len(), 1, "only the pending trial ran");
        assert_eq!(second.run_id.as_deref(), Some(id.as_str()));
        assert_eq!(told.run_id.as_deref(), Some(id.as_str()));
    }

    /// A checkpoint from before 0.7.0 names no run. Resuming it draws an ID,
    /// which the report keeps and the sinks are given.
    #[cfg(feature = "json")]
    #[test]
    fn a_run_resumed_from_a_0_6_1_checkpoint_is_given_an_id() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::copy(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/checkpoint-0.6.1.json"
            ),
            crate::checkpoint::checkpoint_path(dir.path(), "fixture-suite"),
        )
        .unwrap();
        let cfg = EvaluatorConfig {
            num_trials_per_env: 2,
            checkpoint_dir: Some(dir.path().to_path_buf()),
            ..single_thread_cfg(1, 10)
        };
        let suite = Suite::new("fixture-suite").with_env("cartpole", |_| StubEnv::new(3));
        let (mut reporter, told) = KeepsRunId::new();
        let report = Evaluator::new(cfg).run_suite(&suite, |_| StubAgent, &mut reporter);

        let id = report.run_id.expect("the resumed run is given an ID");
        assert!(is_run_id(&id), "{id}");
        assert_eq!(told.lock().unwrap().run_id.as_deref(), Some(id.as_str()));
        let saved = crate::checkpoint::load(&crate::checkpoint::checkpoint_path(
            dir.path(),
            "fixture-suite",
        ))
        .unwrap()
        .unwrap();
        assert_eq!(saved.run_id, Some(id), "and the checkpoint keeps it");
    }
}
