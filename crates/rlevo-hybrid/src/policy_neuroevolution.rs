//! [`PolicyNeuroevolution`] — end-to-end driver for weight-only policy
//! neuroevolution against an RL environment.
//!
//! Pairs a [`WeightOnly`] strategy (any phase-1 `Strategy` evolving the
//! weights of a Burn `Module`) with a [`RolloutFitness`] (environment-rollout
//! scoring) inside an [`EvolutionaryHarness`], and exposes a small drive
//! surface (`reset` / `step` / `run` / `best`).

use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::environment::Environment;
use rlevo_evolution::WeightOnly;
use rlevo_evolution::strategy::{EvolutionaryHarness, Strategy, StrategyMetrics};

use crate::policy::StatefulPolicy;
use crate::rollout_fitness::RolloutFitness;

/// Drives weight-only neuroevolution of a policy network against an
/// environment.
///
/// # Type Parameters
///
/// - `B`: Burn backend (non-autodiff — gradient isolation).
/// - `S`: inner strategy with `Genome = Tensor<B, 2>` (GA, ES, DE, …).
/// - `M`: the policy network module.
/// - `E`: a rank-1 [`Environment`].
pub struct PolicyNeuroevolution<B, S, M, E>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    M: Module<B> + Sync + StatefulPolicy<B, E>,
    E: Environment<1, 1, 1> + Send,
{
    harness: EvolutionaryHarness<B, WeightOnly<B, S, M>, RolloutFitness<B, M, E>>,
}

impl<B, S, M, E> std::fmt::Debug for PolicyNeuroevolution<B, S, M, E>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    M: Module<B> + Sync + StatefulPolicy<B, E>,
    E: Environment<1, 1, 1> + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyNeuroevolution")
            .field("harness", &self.harness)
            .finish()
    }
}

impl<B, S, M, E> PolicyNeuroevolution<B, S, M, E>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    M: Module<B> + Sync + StatefulPolicy<B, E>,
    E: Environment<1, 1, 1> + Send,
{
    /// Assemble the driver.
    ///
    /// `template` is the policy architecture (its weights are evolved); `params`
    /// is the inner strategy's configuration — its genome width must equal the
    /// template's parameter count. `fitness` is a [`RolloutFitness`] built over
    /// a reshaper for the same template.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] when `params` fails [`Validate::validate`] at
    /// the underlying [`EvolutionaryHarness`] chokepoint.
    pub fn new(
        inner: S,
        params: S::Params,
        template: M,
        fitness: RolloutFitness<B, M, E>,
        seed: u64,
        device: B::Device,
        max_generations: usize,
    ) -> Result<Self, ConfigError>
    where
        S::Params: Validate,
    {
        let strategy = WeightOnly::new(inner, template);
        let harness =
            EvolutionaryHarness::new(strategy, params, fitness, seed, device, max_generations)?;
        Ok(Self { harness })
    }

    /// Reset to a fresh initial population.
    pub fn reset(&mut self) {
        self.harness.reset();
    }

    /// Run one generation (ask → rollout-evaluate → tell). Returns `true` when
    /// the generation budget is exhausted.
    ///
    /// # Panics
    ///
    /// Panics if [`reset`](Self::reset) has not been called first.
    pub fn step(&mut self) -> bool {
        self.harness.step(()).done
    }

    /// Reset and run to the configured generation budget.
    pub fn run(&mut self) {
        self.harness.reset();
        while !self.harness.step(()).done {}
    }

    /// Best `(genome, fitness)` found so far, if any.
    #[must_use]
    pub fn best(&self) -> Option<(Tensor<B, 2>, f32)> {
        self.harness.best()
    }

    /// Metrics for the most recent generation.
    #[must_use]
    pub fn latest_metrics(&self) -> Option<&StrategyMetrics> {
        self.harness.latest_metrics()
    }

    /// Completed-generation count.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.harness.generation()
    }
}
