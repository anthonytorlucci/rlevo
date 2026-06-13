//! [`WeightOnly`] — a [`Strategy`] wrapper that ties a flat-genome strategy to
//! a concrete Burn [`Module`] for weight-only neuroevolution.
//!
//! `WeightOnly<B, S, M>` composes an inner `S: Strategy<B, Genome =
//! Tensor<B, 2>>` with a fixed-topology module `M`. It owns a
//! [`ModuleReshaper`] (the single source of truth for the genome width,
//! [`num_params`](WeightOnly::num_params)) but performs **no reshaping** in
//! `ask`/`tell` — the genome stays a flat `(pop_size, num_params)` tensor and
//! every `Strategy` method delegates verbatim to the inner strategy. Reshaping
//! to a module happens only at the fitness boundary, in
//! [`ModuleEvalFn`](crate::module_eval_fn::ModuleEvalFn) (supervised fitness)
//! or `rlevo_hybrid`'s `RolloutFitness` (RL fitness).
//!
//! This follows the [`MemeticWrapper`](crate::algorithms::memetic::MemeticWrapper)
//! pattern (ADR 0016): the inner strategy owns all population state; the
//! wrapper adds one orthogonal concern (here, the module binding).
//!
//! # Gradient isolation
//!
//! `B: Backend`, **not** `AutodiffBackend`. A caller holding an autodiff
//! module calls `.valid()` before constructing the wrapper, so gradient
//! tracking cannot leak into evolution — enforced at the type level.

use std::fmt::Debug;
use std::marker::PhantomData;

use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};
use rand::Rng;

use crate::param_reshaper::ModuleReshaper;
use crate::strategy::{Strategy, StrategyMetrics};

/// Composes a flat-genome [`Strategy`] with a fixed-topology Burn [`Module`].
///
/// # Type Parameters
///
/// - `B`: Burn backend (non-autodiff — see module docs).
/// - `S`: inner strategy with `Genome = Tensor<B, 2>` (GA, ES, DE, …).
/// - `M`: the network whose weights are evolved.
///
/// # Example
///
/// ```ignore
/// let template = MyMlp::<B>::new(&device);
/// let strategy = WeightOnly::new(GeneticAlgorithm::<B>::new(), template.clone());
/// let params = GaConfig::default_for(64, strategy.num_params());
/// // pair with a ModuleEvalFn over the same template in the harness
/// ```
pub struct WeightOnly<B, S, M>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    M: Module<B>,
{
    inner: S,
    reshaper: ModuleReshaper<B, M>,
    _backend: PhantomData<fn() -> B>,
}

impl<B, S, M> WeightOnly<B, S, M>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    M: Module<B>,
{
    /// Build a wrapper from an inner strategy and a template module.
    ///
    /// The template is cloned into a [`ModuleReshaper`]; its float-leaf count
    /// becomes [`num_params`](Self::num_params), which must equal the inner
    /// strategy's configured genome width.
    pub fn new(inner: S, template: M) -> Self {
        Self {
            inner,
            reshaper: ModuleReshaper::new(template),
            _backend: PhantomData,
        }
    }

    /// Genome width — the number of float parameters in the template module.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.reshaper.num_params()
    }

    /// Borrow the owned reshaper (e.g. to build a matching fitness adapter).
    #[must_use]
    pub fn reshaper(&self) -> &ModuleReshaper<B, M> {
        &self.reshaper
    }

    /// Borrow the inner strategy.
    #[must_use]
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

impl<B, S, M> Debug for WeightOnly<B, S, M>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    M: Module<B>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightOnly")
            .field("num_params", &self.reshaper.num_params())
            .finish_non_exhaustive()
    }
}

impl<B, S, M> Strategy<B> for WeightOnly<B, S, M>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    // `Sync` is required so the wrapper satisfies the `Strategy: Send + Sync`
    // supertrait via its `ModuleReshaper<B, M>` field; Burn modules built from
    // `Param<Tensor>` leaves are `Sync`.
    M: Module<B> + Sync,
{
    type Params = S::Params;
    type State = S::State;
    type Genome = Tensor<B, 2>;

    /// Pure delegation to the inner strategy's `init`.
    fn init(
        &self,
        params: &Self::Params,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State {
        self.inner.init(params, rng, device)
    }

    /// Pure delegation to the inner strategy's `ask`. No reshaping here — the
    /// genome stays a flat `(pop_size, num_params)` tensor.
    fn ask(
        &self,
        params: &Self::Params,
        state: &Self::State,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Self::Genome, Self::State) {
        self.inner.ask(params, state, rng, device)
    }

    /// Pure delegation to the inner strategy's `tell`. Reshaping to a module
    /// already happened in the fitness adapter that produced `fitness`.
    fn tell(
        &self,
        params: &Self::Params,
        population: Self::Genome,
        fitness: Tensor<B, 1>,
        state: Self::State,
        rng: &mut dyn Rng,
    ) -> (Self::State, StrategyMetrics) {
        self.inner.tell(params, population, fitness, state, rng)
    }

    /// Pure delegation to the inner strategy's `best`.
    fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)> {
        self.inner.best(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::de::DifferentialEvolution;
    use crate::algorithms::es_classical::EvolutionStrategy;
    use crate::algorithms::ga::GeneticAlgorithm;
    use burn::backend::Flex;
    use burn::module::Module;
    use burn::nn::{Linear, LinearConfig};

    type TestBackend = Flex;

    #[derive(Module, Debug)]
    struct Mlp<B: Backend> {
        l1: Linear<B>,
        l2: Linear<B>,
    }

    impl<B: Backend> Mlp<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                l1: LinearConfig::new(2, 3).init(device),
                l2: LinearConfig::new(3, 1).init(device),
            }
        }
    }

    /// AC #3: `WeightOnly` composes with GA, ES, and DE inner strategies and
    /// reports the template's parameter count. `2*3 + 3` + `3*1 + 1` = `13`.
    /// Confirms the wrapper satisfies the `Strategy<B>` bound for an inner `S`.
    fn assert_strategy<T: Strategy<TestBackend>>(_: &T) {}

    #[test]
    fn composes_with_all_phase1_strategies() {
        let device = Default::default();
        let template = Mlp::<TestBackend>::new(&device);

        let ga = WeightOnly::new(GeneticAlgorithm::<TestBackend>::new(), template.clone());
        let es = WeightOnly::new(EvolutionStrategy::<TestBackend>::new(), template.clone());
        let de = WeightOnly::new(DifferentialEvolution::<TestBackend>::new(), template);

        assert_eq!(ga.num_params(), 13);
        assert_eq!(es.num_params(), 13);
        assert_eq!(de.num_params(), 13);

        assert_strategy(&ga);
        assert_strategy(&es);
        assert_strategy(&de);
    }
}
