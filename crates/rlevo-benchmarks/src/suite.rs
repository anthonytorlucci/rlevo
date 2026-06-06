//! `Suite` groups environments under a shared evaluator configuration.
//!
//! Environments are built lazily per-trial via an [`EnvFactory`] so that
//! rayon workers can own their own instance and receive a deterministic seed.

use std::sync::Arc;

use crate::evaluator::EvaluatorConfig;

/// Factory that constructs a fresh environment instance from a seed.
///
/// The closure MUST route `seed` to every RNG the environment owns so that
/// two invocations with the same seed yield identical trajectories.
pub type EnvFactory<E> = Arc<dyn Fn(u64) -> E + Send + Sync>;

/// A named collection of environments to benchmark against.
pub struct Suite<E> {
    /// Display name of the suite, used as a label in reports and checkpoints.
    pub name: String,
    /// Registered environments: `(display_name, factory)` pairs in insertion order.
    pub envs: Vec<(String, EnvFactory<E>)>,
    /// Default evaluator configuration applied when the suite is run.
    pub default_config: EvaluatorConfig,
}

impl<E> std::fmt::Debug for Suite<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Suite")
            .field("name", &self.name)
            .field(
                "envs",
                &self.envs.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            )
            .field("default_config", &self.default_config)
            .finish()
    }
}

impl<E> Suite<E> {
    /// Creates an empty suite with the given name and default evaluator config.
    pub fn new(name: impl Into<String>, default_config: EvaluatorConfig) -> Self {
        Self {
            name: name.into(),
            envs: Vec::new(),
            default_config,
        }
    }

    /// Registers an environment under `name`, built on demand by `factory`.
    ///
    /// The factory is called once per trial with the trial-derived seed.
    #[must_use]
    pub fn with_env(
        mut self,
        name: impl Into<String>,
        factory: impl Fn(u64) -> E + Send + Sync + 'static,
    ) -> Self {
        self.envs.push((name.into(), Arc::new(factory)));
        self
    }
}

/// Static metadata about a suite passed to reporters.
#[derive(Debug, Clone)]
pub struct SuiteInfo {
    /// Display name of the suite.
    pub name: String,
    /// Display names of all registered environments, in insertion order.
    pub env_names: Vec<String>,
    /// Number of independent seeds evaluated per environment.
    pub num_trials_per_env: usize,
    /// Success threshold from the evaluator config, surfaced so recording
    /// reporters can stamp it onto the run manifest without reaching into
    /// [`EvaluatorConfig`](crate::evaluator::EvaluatorConfig). `None` when the
    /// suite defines no success criterion.
    pub success_threshold: Option<f64>,
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
#[derive(Debug, Clone)]
pub struct TrialInfo {
    /// Unique identifier for this trial within the suite.
    pub key: TrialKey,
    /// Display name of the environment under evaluation.
    pub env_name: String,
    /// Combined seed used for both the environment and the agent.
    pub trial_seed: u64,
}

#[cfg(test)]
mod tests {
    use super::{Suite, SuiteInfo, TrialKey};
    use crate::evaluator::EvaluatorConfig;

    fn minimal_config() -> EvaluatorConfig {
        EvaluatorConfig {
            num_episodes: 1,
            num_trials_per_env: 1,
            max_steps: 10,
            base_seed: 0,
            num_threads: None,
            checkpoint_dir: None,
            fail_fast: false,
            success_threshold: None,
        }
    }

    #[test]
    fn new_suite_is_empty() {
        let suite: Suite<()> = Suite::new("my-suite", minimal_config());
        assert_eq!(suite.name, "my-suite");
        assert!(suite.envs.is_empty());
    }

    #[test]
    fn with_env_appends_names_in_order() {
        let suite: Suite<u32> = Suite::new("s", minimal_config())
            .with_env("alpha", |_seed| 1_u32)
            .with_env("beta", |_seed| 2_u32);

        let names: Vec<&str> = suite.envs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn factory_receives_the_seed() {
        let suite: Suite<u64> = Suite::new("s", minimal_config())
            .with_env("seeded", |seed| seed);

        let (_, factory) = &suite.envs[0];
        assert_eq!(factory(42), 42);
        assert_eq!(factory(99), 99);
    }

    #[test]
    fn debug_impl_lists_env_names() {
        let suite: Suite<u32> = Suite::new("s", minimal_config())
            .with_env("env-a", |_| 0_u32);
        let debug = format!("{suite:?}");
        assert!(debug.contains("Suite"));
        assert!(debug.contains("env-a"));
    }

    #[test]
    fn suite_info_round_trip() {
        let info = SuiteInfo {
            name: "s".into(),
            env_names: vec!["e1".into(), "e2".into()],
            num_trials_per_env: 3,
            success_threshold: None,
        };
        assert_eq!(info.env_names.len(), 2);
        assert_eq!(info.num_trials_per_env, 3);
    }

    #[test]
    fn trial_key_equality_and_hash() {
        use std::collections::HashSet;

        let k1 = TrialKey { env_idx: 0, trial_idx: 1 };
        let k2 = TrialKey { env_idx: 0, trial_idx: 1 };
        let k3 = TrialKey { env_idx: 1, trial_idx: 0 };

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);

        let mut set = HashSet::new();
        set.insert(k1);
        assert!(set.contains(&k2));
        assert!(!set.contains(&k3));
    }
}
