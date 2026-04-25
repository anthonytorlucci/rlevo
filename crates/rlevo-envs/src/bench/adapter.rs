//! `BenchAdapter` — translates a typed [`Environment`] into the
//! object-friendly [`BenchEnv`] interface.
//!
//! Bounded to [`ScalarReward`] for v0.1: every env in this crate uses it,
//! and lifting the bound (e.g. via `BenchReward: Into<f64>`) is deferred
//! until a non-scalar-reward env actually lands.
//!
//! [`EnvironmentError`] is converted to [`BenchError`] via `Display`. The
//! string preserves the variant name + payload, and keeps `rlevo-benchmarks`
//! free of an `rlevo-core` runtime dep — the layering documented in
//! ADR-0001.
//!
//! # Const generics
//!
//! [`BenchAdapter`] carries the wrapped env's `(D, SD, AD)` const generics
//! so the [`BenchEnv`] impl can name them in its `where` clause without
//! tripping E0207 (unconstrained generic params). Inference picks them up
//! from the wrapped env at construction — callers write
//! `BenchAdapter::new(env)`, never the turbofish.
//!
//! [`Environment`]: rlevo_core::environment::Environment
//! [`ScalarReward`]: rlevo_core::reward::ScalarReward
//! [`BenchEnv`]: rlevo_benchmarks::env::BenchEnv
//! [`EnvironmentError`]: rlevo_core::environment::EnvironmentError

use std::marker::PhantomData;

use rlevo_benchmarks::env::{BenchEnv, BenchError, BenchStep};
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_core::reward::ScalarReward;

/// Object-safe wrapper around a typed [`Environment`].
///
/// # Example
///
/// ```
/// # #[cfg(feature = "bench")] {
/// use rlevo_envs::bench::BenchAdapter;
/// use rlevo_envs::classic::{CartPole, CartPoleConfig};
/// use rlevo_benchmarks::env::BenchEnv;
///
/// let env = CartPole::with_config(CartPoleConfig::default());
/// let mut adapter = BenchAdapter::new(env);
/// let _obs = adapter.reset().expect("reset");
/// # }
/// ```
///
/// [`Environment`]: rlevo_core::environment::Environment
#[derive(Debug)]
pub struct BenchAdapter<E, const D: usize, const SD: usize, const AD: usize> {
    env: E,
    _phantom: PhantomData<()>,
}

impl<E, const D: usize, const SD: usize, const AD: usize> BenchAdapter<E, D, SD, AD> {
    /// Wrap an [`Environment`] for use with the harness.
    ///
    /// [`Environment`]: rlevo_core::environment::Environment
    pub const fn new(env: E) -> Self {
        Self {
            env,
            _phantom: PhantomData,
        }
    }

    /// Borrow the wrapped environment.
    pub const fn inner(&self) -> &E {
        &self.env
    }

    /// Recover the wrapped environment.
    pub fn into_inner(self) -> E {
        self.env
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> BenchEnv for BenchAdapter<E, D, SD, AD>
where
    E: Environment<D, SD, AD, RewardType = ScalarReward>,
    E::ObservationType: Clone,
{
    type Observation = E::ObservationType;
    type Action = E::ActionType;

    fn reset(&mut self) -> Result<Self::Observation, BenchError> {
        let snap = self
            .env
            .reset()
            .map_err(|e| BenchError::Reset(e.to_string()))?;
        Ok(snap.observation().clone())
    }

    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<BenchStep<Self::Observation>, BenchError> {
        let snap = self
            .env
            .step(action)
            .map_err(|e| BenchError::Step(e.to_string()))?;
        Ok(BenchStep {
            observation: snap.observation().clone(),
            reward: f64::from(snap.reward().value()),
            done: snap.is_done(),
        })
    }
}
