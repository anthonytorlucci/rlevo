//! Ant Colony Optimization over permutation genomes — **stub**.
//!
//! This module path is reserved so the public API surface is stable,
//! but the implementation is deferred to a future release. A useful
//! ACO-TSP implementation is a substantial design exercise of its own
//! (pheromone-matrix update, tour construction, candidate lists) and
//! does not belong inside the continuous-domain strategies that the
//! other nine swarm algorithms in this module target.
//!
//! All trait methods below panic with `todo!()`. The struct exists so
//! the [`Permutation`](crate::genome::Permutation) genome kind has at
//! least one declared consumer and downstream crates can
//! unambiguously reference the future API surface.

use std::marker::PhantomData;

use burn::tensor::{backend::Backend, Int, Tensor};
use rand::Rng;

use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration placeholder for the permutation ACO.
#[derive(Debug, Clone)]
pub struct AcoPermConfig {
    /// Number of ants.
    pub pop_size: usize,
    /// Number of nodes in the graph (permutation length).
    pub n_nodes: usize,
    /// Pheromone evaporation rate, `ρ ∈ (0, 1]`.
    pub rho: f32,
    /// Heuristic exponent `α` (pheromone weight).
    pub alpha: f32,
    /// Heuristic exponent `β` (desirability weight).
    pub beta: f32,
}

impl AcoPermConfig {
    /// Default configuration for a given population size and graph node count.
    #[must_use]
    pub fn default_for(pop_size: usize, n_nodes: usize) -> Self {
        Self {
            pop_size,
            n_nodes,
            rho: 0.5,
            alpha: 1.0,
            beta: 2.0,
        }
    }
}

/// Generation state placeholder.
#[derive(Debug, Clone)]
pub struct AcoPermState<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

/// Ant Colony Optimization for permutation problems (TSP, QAP, …).
///
/// **Not yet implemented** — planned for a future release.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use evorl_evolution::algorithms::swarm::aco_perm::{AntColonyPermutation, AcoPermConfig};
///
/// let strategy = AntColonyPermutation::<NdArray>::new();
/// let params = AcoPermConfig::default_for(32, 20);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AntColonyPermutation<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> AntColonyPermutation<B> {
    /// Builds a new (stateless) strategy object (stub; all methods `todo!()`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Strategy<B> for AntColonyPermutation<B> {
    type Params = AcoPermConfig;
    type State = AcoPermState<B>;
    type Genome = Tensor<B, 2, Int>;

    fn init(
        &self,
        _params: &AcoPermConfig,
        _rng: &mut dyn Rng,
        _device: &B::Device,
    ) -> AcoPermState<B> {
        todo!(
            "permutation ACO is not yet implemented; \
             use AntColonyReal for continuous problems in the meantime"
        )
    }

    fn ask(
        &self,
        _params: &AcoPermConfig,
        _state: &AcoPermState<B>,
        _rng: &mut dyn Rng,
        _device: &B::Device,
    ) -> (Self::Genome, AcoPermState<B>) {
        todo!("permutation ACO is not yet implemented")
    }

    fn tell(
        &self,
        _params: &AcoPermConfig,
        _population: Self::Genome,
        _fitness: Tensor<B, 1>,
        _state: AcoPermState<B>,
        _rng: &mut dyn Rng,
    ) -> (AcoPermState<B>, StrategyMetrics) {
        todo!("permutation ACO is not yet implemented")
    }

    fn best(&self, _state: &AcoPermState<B>) -> Option<(Self::Genome, f32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn stub_is_constructible() {
        let _strategy = AntColonyPermutation::<TestBackend>::new();
        let _params = AcoPermConfig::default_for(32, 20);
    }

    #[test]
    #[should_panic(expected = "permutation ACO is not yet implemented")]
    fn init_panics_with_clear_message() {
        use rand::SeedableRng;
        let strategy = AntColonyPermutation::<TestBackend>::new();
        let params = AcoPermConfig::default_for(4, 5);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let _ = strategy.init(&params, &mut rng, &Default::default());
    }
}
