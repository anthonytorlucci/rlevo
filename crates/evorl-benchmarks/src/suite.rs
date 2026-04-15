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
    pub name: String,
    pub envs: Vec<(String, EnvFactory<E>)>,
    pub default_config: EvaluatorConfig,
}

impl<E> std::fmt::Debug for Suite<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Suite")
            .field("name", &self.name)
            .field("envs", &self.envs.iter().map(|(n, _)| n).collect::<Vec<_>>())
            .field("default_config", &self.default_config)
            .finish()
    }
}

impl<E> Suite<E> {
    pub fn new(name: impl Into<String>, default_config: EvaluatorConfig) -> Self {
        Self {
            name: name.into(),
            envs: Vec::new(),
            default_config,
        }
    }

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
    pub name: String,
    pub env_names: Vec<String>,
    pub num_trials_per_env: usize,
}

/// Identifies a single trial within a suite run.
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrialKey {
    pub env_idx: usize,
    pub trial_idx: usize,
}

/// Runtime metadata about an in-flight trial passed to reporters.
#[derive(Debug, Clone)]
pub struct TrialInfo {
    pub key: TrialKey,
    pub env_name: String,
    pub trial_seed: u64,
}
