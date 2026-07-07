//! Smoke integration tests for the estimation-of-distribution strategies.
//!
//! Each test drives an [`EdaStrategy`] over one [`ProbabilityModel`] through
//! the [`EvolutionaryHarness`] for a handful of generations and asserts the
//! plumbing holds: no panic, metrics and best-so-far become available, and the
//! best fitness stays finite. These are not convergence tests — they only
//! exercise the full ask → evaluate → tell loop end to end.

use burn::backend::Flex;
use rlevo_core::bounds::Bounds;
use rlevo_core::fitness::FitnessEvaluable;

use rlevo_evolution::algorithms::eda::{
    CompactGenetic, CompactGeneticParams, DependencyChain, DependencyChainParams, EdaParams,
    EdaStrategy, UnivariateBernoulli, UnivariateBernoulliParams, UnivariateGaussian,
    UnivariateGaussianParams,
};
use rlevo_evolution::fitness::{BatchFitnessFn, FromFitnessEvaluable};
use rlevo_evolution::probability_model::ProbabilityModel;
use rlevo_evolution::strategy::{EvolutionaryHarness, Strategy};

type B = Flex;

struct Sphere;
struct SphereFit;
impl FitnessEvaluable for SphereFit {
    type Individual = Vec<f64>;
    type Landscape = Sphere;
    fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
}

/// Drive an EDA strategy for ten generations and assert the plumbing holds.
fn smoke<M>(model: M, params: EdaParams<M::Params>)
where
    M: ProbabilityModel<B> + 'static,
    EdaStrategy<B, M>: Strategy<
            B,
            Genome = burn::tensor::Tensor<B, 2>,
            Params = EdaParams<<M as ProbabilityModel<B>>::Params>,
        >,
    FromFitnessEvaluable<SphereFit, Sphere>:
        BatchFitnessFn<B, <EdaStrategy<B, M> as Strategy<B>>::Genome>,
{
    let device = Default::default();
    let strategy = EdaStrategy::<B, M>::new(model);
    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        strategy,
        params,
        FromFitnessEvaluable::new(SphereFit, Sphere),
        7,
        device,
        10,
    )
    .expect("valid params");
    harness.reset();
    loop {
        let step = harness.step(());
        if step.done {
            break;
        }
    }
    let metrics = harness.latest_metrics();
    assert!(metrics.is_some(), "latest_metrics must be Some after a run");
    assert!(
        metrics.unwrap().best_fitness_ever().is_finite(),
        "best fitness must be finite"
    );
    let best = harness.best();
    assert!(best.is_some(), "best() must be Some after the first tell");
    assert!(
        best.unwrap().1.is_finite(),
        "best genome fitness must be finite"
    );
}

#[test]
fn smoke_univariate_gaussian() {
    let params = EdaParams {
        pop_size: 20,
        selection_ratio: 0.5,
        bounds: Some(Bounds::new(-5.12, 5.12)),
        model: UnivariateGaussianParams::default_for(8),
    };
    smoke(UnivariateGaussian, params);
}

#[test]
fn smoke_dependency_chain() {
    let params = EdaParams {
        pop_size: 20,
        selection_ratio: 0.5,
        bounds: Some(Bounds::new(-5.12, 5.12)),
        model: DependencyChainParams::default_for(8),
    };
    smoke(DependencyChain, params);
}

#[test]
fn smoke_univariate_bernoulli() {
    let params = EdaParams {
        pop_size: 20,
        selection_ratio: 0.5,
        bounds: None,
        model: UnivariateBernoulliParams::default_for(8),
    };
    smoke(UnivariateBernoulli, params);
}

#[test]
fn smoke_compact_genetic() {
    let params = EdaParams {
        pop_size: 20,
        selection_ratio: 0.5,
        bounds: None,
        model: CompactGeneticParams::default_for(8),
    };
    smoke(CompactGenetic, params);
}
