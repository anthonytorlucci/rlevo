//! `Suite` names a group of environments to evaluate together.
//!
//! Environments are built lazily per-trial via an [`EnvFactory`] so that
//! rayon workers can own their own instance and receive a deterministic seed.

use std::sync::Arc;

use crate::task::TaskFingerprint;

/// Factory that constructs a fresh environment instance from a seed.
///
/// The closure MUST route `seed` to every RNG the environment owns so that
/// two invocations with the same seed yield identical trajectories.
pub type EnvFactory<E> = Arc<dyn Fn(u64) -> E + Send + Sync>;

/// One unit an [`Evaluator`](crate::evaluator::Evaluator) runs trials of: an
/// environment, a landscape, or anything else a trial is built from, named by
/// its id and, when known, the task it is.
///
/// Every [`TrialReport`](crate::report::TrialReport) of the unit carries the id
/// as `env_id` and the task's hash as `task_hash`, and the
/// [`BenchmarkReport`](crate::report::BenchmarkReport) the fingerprint, once,
/// in `tasks`. The id is not part of the task. Pass units to
/// [`Evaluator::run_trials`](crate::evaluator::Evaluator::run_trials); a
/// [`Suite`] holds one per environment it registers.
///
/// `#[non_exhaustive]`: built from an id (`From<&str>`, `From<String>`) or
/// with `with_task` (feature `json`), so a later field is not a breaking
/// change.
///
/// # Examples
///
/// ```
/// use rlevo_harness::suite::TrialUnit;
///
/// let plain: TrialUnit = "sphere-d8".into();
/// assert_eq!(plain.id, "sphere-d8");
/// assert!(plain.task.is_none());
///
/// #[derive(serde::Serialize)]
/// struct Landscape { name: &'static str, dim: usize }
/// let tasked = TrialUnit::with_task("sphere-d8", &Landscape { name: "sphere", dim: 8 })?;
/// assert!(tasked.task.is_some());
/// # Ok::<(), rlevo_harness::task::TaskFingerprintError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TrialUnit {
    /// The unit's id: the name its trial reports carry as `env_id`.
    pub id: String,
    /// The task's fingerprint, when the unit was built with `with_task`.
    pub task: Option<TaskFingerprint>,
}

impl TrialUnit {
    /// A unit named `id` whose task is `config`, fingerprinted as
    /// [`task_fingerprint`](crate::task::task_fingerprint) does: the config's
    /// top-level `seed` field is left out, every other field counts.
    ///
    /// # Errors
    ///
    /// [`TaskFingerprintError`](crate::task::TaskFingerprintError) when
    /// `config` does not serialize to JSON.
    #[cfg(feature = "json")]
    pub fn with_task<C>(
        id: impl Into<String>,
        config: &C,
    ) -> Result<Self, crate::task::TaskFingerprintError>
    where
        C: serde::Serialize + ?Sized,
    {
        let id = id.into();
        let task = crate::task::task_fingerprint(config)?;
        Ok(Self {
            id,
            task: Some(task),
        })
    }
}

/// A unit with no task fingerprint.
impl From<&str> for TrialUnit {
    fn from(id: &str) -> Self {
        Self::from(id.to_owned())
    }
}

/// A unit with no task fingerprint.
impl From<String> for TrialUnit {
    fn from(id: String) -> Self {
        Self { id, task: None }
    }
}

/// One environment a [`Suite`] runs: the unit it is, and how to build it.
///
/// `#[non_exhaustive]`: built only by [`Suite`], so a later field is not a
/// breaking change.
#[non_exhaustive]
pub struct SuiteEnv<E> {
    /// The environment's id and, when registered with `Suite::with_task`,
    /// its task.
    pub unit: TrialUnit,
    /// Builds a fresh, seeded environment for each trial.
    pub factory: EnvFactory<E>,
}

impl<E> std::fmt::Debug for SuiteEnv<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SuiteEnv")
            .field("unit", &self.unit)
            .finish_non_exhaustive()
    }
}

/// A named collection of environments to benchmark against.
///
/// Constructed via [`Suite::new`] and extended with [`Suite::with_env`] or
/// `Suite::with_task` (feature `json`) in a builder style. Each environment is stored with an
/// [`EnvFactory`] closure so that parallel workers can instantiate
/// independent, deterministically seeded copies at trial time. How the suite
/// runs is the [`Evaluator`](crate::evaluator::Evaluator)'s configuration,
/// not the suite's.
pub struct Suite<E> {
    /// Display name of the suite, used as a label in reports and checkpoints.
    pub name: String,
    /// Registered environments, in insertion order.
    pub envs: Vec<SuiteEnv<E>>,
}

impl<E> std::fmt::Debug for Suite<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Suite")
            .field("name", &self.name)
            .field(
                "envs",
                &self.envs.iter().map(|e| &e.unit.id).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl<E> Suite<E> {
    /// Creates an empty suite with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            envs: Vec::new(),
        }
    }

    /// Registers an environment under `id`, built on demand by `factory`.
    ///
    /// The factory is called once per trial with the trial-derived seed. The
    /// id is the environment's own name, which every trial report of it
    /// carries as `env_id`; give it once, from one constant. Its trials carry
    /// no task fingerprint; `with_task` (feature `json`) adds one.
    #[must_use]
    pub fn with_env(
        mut self,
        id: impl Into<String>,
        factory: impl Fn(u64) -> E + Send + Sync + 'static,
    ) -> Self {
        self.envs.push(SuiteEnv {
            unit: TrialUnit::from(id.into()),
            factory: Arc::new(factory),
        });
        self
    }

    /// Registers an environment under `id`, built from `config` by `factory`,
    /// and fingerprints the task.
    ///
    /// `factory` is called once per trial with `config` and the
    /// trial-derived seed, so the config the fingerprint covers is the one
    /// every trial's environment is built from. The report keeps the
    /// fingerprint in `tasks`, and every trial report of the environment its
    /// hash as `task_hash`; two runs faced the same task when the hashes are
    /// equal, whatever ids they registered it under. The config's top-level
    /// `seed` field, if it has one, is left out of the fingerprint, and the
    /// factory should take the seed from its second argument, not from the
    /// config. [`task_fingerprint`](crate::task::task_fingerprint) says what
    /// the fingerprint covers, and recomputes it from outside.
    ///
    /// # Errors
    ///
    /// [`TaskFingerprintError`](crate::task::TaskFingerprintError) when
    /// `config` does not serialize to JSON.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_harness::suite::Suite;
    ///
    /// #[derive(serde::Serialize)]
    /// struct Config { size: u32, seed: u64 }
    /// struct Env { size: u32, seed: u64 }
    ///
    /// let suite = Suite::new("s").with_task(
    ///     "grid",
    ///     Config { size: 4, seed: 0 },
    ///     |c: &Config, seed| Env { size: c.size, seed },
    /// )?;
    /// assert!(suite.envs[0].unit.task.is_some());
    /// # Ok::<(), rlevo_harness::task::TaskFingerprintError>(())
    /// ```
    #[cfg(feature = "json")]
    pub fn with_task<C>(
        mut self,
        id: impl Into<String>,
        config: C,
        factory: impl Fn(&C, u64) -> E + Send + Sync + 'static,
    ) -> Result<Self, crate::task::TaskFingerprintError>
    where
        C: serde::Serialize + Send + Sync + 'static,
    {
        let unit = TrialUnit::with_task(id, &config)?;
        self.envs.push(SuiteEnv {
            unit,
            factory: Arc::new(move |seed| factory(&config, seed)),
        });
        Ok(self)
    }
}

/// Static metadata about a suite, passed to reporters before trials begin.
///
/// Derived from a [`Suite`] at run start and does not hold any closures, so it
/// is cheaply cloneable and serializable.
///
/// `#[non_exhaustive]`: built only by the harness, so a later field is not a
/// breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SuiteInfo {
    /// Display name of the suite.
    pub name: String,
    /// The ids of all registered environments, in insertion order.
    pub env_ids: Vec<String>,
    /// Number of independent seeds evaluated per environment.
    pub num_trials_per_env: usize,
    /// Success threshold from the evaluator config, surfaced so recording
    /// reporters can stamp it onto the run manifest without reaching into
    /// [`EvaluatorConfig`](crate::evaluator::EvaluatorConfig). `None` when the
    /// suite defines no success criterion.
    pub success_threshold: Option<f64>,
    /// What tells this run from every other
    /// ([`BenchmarkReport::run_id`](crate::report::BenchmarkReport::run_id)).
    /// A run resumed from a checkpoint has the ID it began with, so a sink
    /// that names its output by this ID adds to what the earlier run wrote.
    pub run_id: String,
}

impl SuiteInfo {
    /// Suite metadata as `run_trials` builds it; public so a sink's own tests
    /// can build one.
    #[must_use]
    pub fn new(
        name: String,
        env_ids: Vec<String>,
        num_trials_per_env: usize,
        success_threshold: Option<f64>,
        run_id: String,
    ) -> Self {
        Self {
            name,
            env_ids,
            num_trials_per_env,
            success_threshold,
            run_id,
        }
    }
}

/// Identifies a single trial within a suite run.
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrialKey {
    /// Zero-based index into `Suite::envs`.
    pub env_idx: usize,
    /// Zero-based seed repetition index for this environment.
    pub trial_idx: usize,
}

/// Runtime metadata about an in-flight trial passed to reporters.
///
/// `#[non_exhaustive]`: built only by the harness, so a later field is not a
/// breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TrialInfo {
    /// Unique identifier for this trial within the suite.
    pub key: TrialKey,
    /// The id of the environment under evaluation: the name the suite
    /// registered it under. Named `env_name` before 0.7.0.
    pub env_id: String,
    /// Combined seed used for both the environment and the agent.
    pub trial_seed: u64,
}

impl TrialInfo {
    /// Trial metadata as `run_trials` builds it; public so a sink's own tests
    /// can build one.
    #[must_use]
    pub fn new(key: TrialKey, env_id: String, trial_seed: u64) -> Self {
        Self {
            key,
            env_id,
            trial_seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Suite, SuiteInfo, TrialKey};

    #[test]
    fn new_suite_is_empty() {
        let suite: Suite<()> = Suite::new("my-suite");
        assert_eq!(suite.name, "my-suite");
        assert!(suite.envs.is_empty());
    }

    #[test]
    fn with_env_appends_names_in_order() {
        let suite: Suite<u32> = Suite::new("s")
            .with_env("alpha", |_seed| 1_u32)
            .with_env("beta", |_seed| 2_u32);

        let names: Vec<&str> = suite.envs.iter().map(|e| e.unit.id.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn factory_receives_the_seed() {
        let suite: Suite<u64> = Suite::new("s").with_env("seeded", |seed| seed);

        let factory = &suite.envs[0].factory;
        assert_eq!(factory(42), 42);
        assert_eq!(factory(99), 99);
    }

    /// `with_task` builds every trial's environment from the config it
    /// fingerprinted, and records that fingerprint; `with_env` records none.
    #[cfg(feature = "json")]
    #[test]
    fn with_task_fingerprints_the_config_it_builds_from() {
        #[derive(serde::Serialize)]
        struct Config {
            size: u32,
            seed: u64,
        }
        let suite: Suite<(u32, u64)> = Suite::new("s")
            .with_task("grid", Config { size: 4, seed: 0 }, |c: &Config, seed| {
                (c.size, seed)
            })
            .unwrap()
            .with_env("plain", |seed| (0, seed));

        assert_eq!((suite.envs[0].factory)(9), (4, 9));
        assert_eq!(
            suite.envs[0].unit.task,
            Some(crate::task::task_fingerprint(&Config { size: 4, seed: 5 }).unwrap())
        );
        assert_eq!(suite.envs[1].unit.task, None);
    }

    #[test]
    fn debug_impl_lists_env_names() {
        let suite: Suite<u32> = Suite::new("s").with_env("env-a", |_| 0_u32);
        let debug = format!("{suite:?}");
        assert!(debug.contains("Suite"));
        assert!(debug.contains("env-a"));
    }

    #[test]
    fn suite_info_round_trip() {
        let info = SuiteInfo {
            name: "s".into(),
            env_ids: vec!["e1".into(), "e2".into()],
            num_trials_per_env: 3,
            success_threshold: None,
            run_id: "run".into(),
        };
        assert_eq!(info.env_ids.len(), 2);
        assert_eq!(info.num_trials_per_env, 3);
    }

    #[test]
    fn trial_key_equality_and_hash() {
        use std::collections::HashSet;

        let k1 = TrialKey {
            env_idx: 0,
            trial_idx: 1,
        };
        let k2 = TrialKey {
            env_idx: 0,
            trial_idx: 1,
        };
        let k3 = TrialKey {
            env_idx: 1,
            trial_idx: 0,
        };

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);

        let mut set = HashSet::new();
        set.insert(k1);
        assert!(set.contains(&k2));
        assert!(!set.contains(&k3));
    }
}
