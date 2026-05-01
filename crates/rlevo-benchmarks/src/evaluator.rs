//! Evaluator — orchestrates suite execution with rayon trial-level parallelism.

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
use crate::report::{BenchmarkReport, EpisodeRecord, TrialReport};
use crate::reporter::Reporter;
use crate::suite::{Suite, SuiteInfo, TrialInfo, TrialKey};

#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    pub num_episodes: usize,
    pub num_trials_per_env: usize,
    pub max_steps: usize,
    pub base_seed: u64,
    pub num_threads: Option<usize>,
    pub checkpoint_dir: Option<PathBuf>,
    pub fail_fast: bool,
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

#[derive(Debug, Clone)]
pub struct Evaluator {
    pub cfg: EvaluatorConfig,
}

impl Evaluator {
    #[must_use]
    pub const fn new(cfg: EvaluatorConfig) -> Self {
        Self { cfg }
    }

    /// Run a suite of `BenchEnv`s against a `BenchableAgent`.
    ///
    /// `agent_factory` is called once per trial with the derived agent seed;
    /// it is the implementor's contract to initialize the agent deterministically.
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

        let rec = EpisodeRecord {
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
