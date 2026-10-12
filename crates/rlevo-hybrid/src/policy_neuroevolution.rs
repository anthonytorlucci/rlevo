//! [`PolicyNeuroevolution`] — end-to-end driver for weight-only policy
//! neuroevolution against an RL environment.
//!
//! Pairs a [`WeightOnly`] strategy (any phase-1 `Strategy` evolving the
//! weights of a Burn `Module`) with a [`RolloutFitness`] (environment-rollout
//! scoring) inside an [`EvolutionaryLoop`], and exposes the loop's drive
//! surface (`reset` / `step` / `is_done` / `best`) plus `run`, which resets
//! and steps to the budget.

use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::environment::Environment;
use rlevo_evolution::WeightOnly;
use rlevo_evolution::strategy::{EvolutionaryLoop, Strategy, StrategyMetrics};

use crate::policy::StatefulPolicy;
use crate::rollout_fitness::RolloutFitness;

/// Drives weight-only neuroevolution of a policy network against an
/// environment.
///
/// It wraps an [`EvolutionaryLoop`] and is driven the same way: `reset`,
/// then `step` until `is_done`, or [`run`](Self::run) for both. It does not
/// implement `GenerationProbe`, so the evaluation harness cannot run it;
/// to run a policy search through the harness, build the inner
/// `EvolutionaryLoop` yourself and give the harness that.
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
    evo_loop: EvolutionaryLoop<B, WeightOnly<B, S, M>, RolloutFitness<B, M, E>>,
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
            .field("evo_loop", &self.evo_loop)
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
    /// the underlying [`EvolutionaryLoop`] chokepoint.
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
        // single-source: `fitness` already owns a reshaper (built by the caller
        // from the same `template`), so the wrapper's reshaper and the
        // evaluator's cannot be collapsed here today.
        // TODO(#229): collapse via the paired `with_eval_fn` constructor once
        // added. Both derive from one `template`, so their widths still agree; a
        // mismatch would surface as the documented width-mismatch panic in
        // `evaluate_batch`.
        let strategy = WeightOnly::new(inner, template);
        let evo_loop =
            EvolutionaryLoop::new(strategy, params, fitness, seed, device, max_generations)?;
        Ok(Self { evo_loop })
    }

    /// Reset to a fresh initial population.
    pub fn reset(&mut self) {
        self.evo_loop.reset();
    }

    /// Run one generation (ask → rollout-evaluate → tell).
    ///
    /// Returns nothing, as [`EvolutionaryLoop::step`] does. Ask
    /// [`is_done`](Self::is_done) whether the generation budget is spent.
    /// Like the loop's, this `step` does not consult the budget: called past
    /// it, it runs one more generation.
    ///
    /// # Panics
    ///
    /// Panics if [`reset`](Self::reset) has not been called first.
    pub fn step(&mut self) {
        self.evo_loop.step();
    }

    /// Whether the generation budget is spent. See
    /// [`EvolutionaryLoop::is_done`].
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.evo_loop.is_done()
    }

    /// Reset and run to the configured generation budget.
    pub fn run(&mut self) {
        self.evo_loop.reset();
        while !self.evo_loop.is_done() {
            self.evo_loop.step();
        }
    }

    /// Best `(genome, fitness)` found so far, if any.
    #[must_use]
    pub fn best(&self) -> Option<(Tensor<B, 2>, f32)> {
        self.evo_loop.best()
    }

    /// Metrics for the most recent generation.
    #[must_use]
    pub fn latest_metrics(&self) -> Option<&StrategyMetrics> {
        self.evo_loop.latest_metrics()
    }

    /// Completed-generation count.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.evo_loop.generation()
    }
}
