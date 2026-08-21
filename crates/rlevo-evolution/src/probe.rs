//! The evolutionary drive seam: [`GenerationProbe`].
//!
//! An evolutionary run is not an episodic environment interaction. It has no
//! observation, no action, and no episode axis — driving one to its generation
//! budget twice from the same seed produces a byte-identical replay, so a
//! second "episode" is duplicated compute carrying zero information.
//!
//! `GenerationProbe` is the shape that says so. Where the benchmarking
//! `BenchEnv` surface forces a generation loop to present itself as an
//! environment (`Observation = ()`, `Action = ()`, a scalar `reward`, a `done`
//! flag), this trait names what the loop actually is: something that begins,
//! then advances one generation at a time until its budget is spent, reporting
//! typed metrics as it goes.
//!
//! # Relationship to `BenchEnv`
//!
//! This is now the **only** way the benchmark evaluator drives these harnesses.
//! They previously also implemented `rlevo-core::evaluation::BenchEnv`, which
//! required them to claim an observation, an action, and an episode axis that
//! an evolutionary run does not have; those impls were removed once every
//! caller migrated to `Evaluator::run_trials`. See ADR 0075 for why
//! `BenchEnv`'s shape was under review and ADR 0076 for the split.
//!
//! [`EvolutionaryHarness`]: crate::strategy::EvolutionaryHarness
//! [`CoEvolutionaryHarness`]: crate::coevolution::CoEvolutionaryHarness

use burn::tensor::backend::Backend;

use rlevo_core::evaluation::GenerationProbe;
use rlevo_core::fitness::{Metric, MetricsProvider};

use crate::coevolution::{CoEAMetrics, CoEvolutionaryAlgorithm, CoEvolutionaryHarness};
use crate::fitness::BatchFitnessFn;
use crate::strategy::{EvolutionaryHarness, Strategy, StrategyMetrics};

fn scalar(name: &str, value: f64) -> Metric {
    Metric::Scalar {
        name: name.to_string(),
        value,
    }
}

/// Reports the single-population generation summary as harness metrics.
///
/// Names are prefixed `ea/` so a generation trial's scalars never collide with
/// the episodic `core_metrics` names (`return/mean`, `episode/length_mean`, …)
/// in the same `TrialReport` metric maps.
impl MetricsProvider for StrategyMetrics {
    #[allow(clippy::cast_precision_loss)]
    fn emit(&self) -> Vec<Metric> {
        vec![
            scalar("ea/generation", self.generation() as f64),
            scalar("ea/population_size", self.population_size() as f64),
            scalar("ea/best_fitness", f64::from(self.best_fitness())),
            scalar("ea/best_fitness_ever", f64::from(self.best_fitness_ever())),
            scalar("ea/mean_fitness", f64::from(self.mean_fitness())),
            scalar("ea/worst_fitness", f64::from(self.worst_fitness())),
            scalar("ea/broken_count", self.broken_count() as f64),
        ]
    }
}

/// Reports the two-population co-evolutionary summary as harness metrics.
///
/// `coea/binding_fitness` is the canonical (engine-space) value — the weaker
/// population's best — while the per-population `best`/`mean` fields are in the
/// objective's natural declared sense (ADR 0023). Emitting both keeps that
/// distinction visible in the report rather than collapsing it.
impl MetricsProvider for CoEAMetrics {
    #[allow(clippy::cast_precision_loss)]
    fn emit(&self) -> Vec<Metric> {
        vec![
            scalar("coea/generation", self.generation as f64),
            scalar("coea/binding_fitness", f64::from(self.binding_fitness)),
            scalar("coea/best_fitness_a", f64::from(self.best_fitness_a)),
            scalar("coea/best_fitness_b", f64::from(self.best_fitness_b)),
            scalar("coea/mean_fitness_a", f64::from(self.mean_fitness_a)),
            scalar("coea/mean_fitness_b", f64::from(self.mean_fitness_b)),
            scalar("coea/hof_size_a", self.hof_size_a as f64),
            scalar("coea/hof_size_b", self.hof_size_b as f64),
        ]
    }
}

impl<B, S, F> GenerationProbe for EvolutionaryHarness<B, S, F>
where
    B: Backend,
    S: Strategy<B>,
    F: BatchFitnessFn<B, S::Genome>,
{
    type Metrics = StrategyMetrics;

    fn begin(&mut self) {
        EvolutionaryHarness::<B, S, F>::reset(self);
    }

    fn advance(&mut self) -> Option<Self::Metrics> {
        if self.generation() >= self.max_generations() {
            return None;
        }
        let _ = EvolutionaryHarness::<B, S, F>::step(self, ());
        self.latest_metrics().cloned()
    }
}

impl<B, C> GenerationProbe for CoEvolutionaryHarness<B, C>
where
    B: Backend,
    C: CoEvolutionaryAlgorithm<B>,
{
    type Metrics = CoEAMetrics;

    fn begin(&mut self) {
        CoEvolutionaryHarness::<B, C>::reset(self);
    }

    fn advance(&mut self) -> Option<Self::Metrics> {
        if self.generation() >= self.max_generations() {
            return None;
        }
        let _ = CoEvolutionaryHarness::<B, C>::step(self, ());
        self.latest_metrics().cloned()
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Flex;
    use burn::tensor::Tensor;
    use rlevo_core::bounds::Bounds;
    use rlevo_core::objective::ObjectiveSense;
    use rlevo_core::rate::NonNegativeRate;

    use super::GenerationProbe;
    use crate::algorithms::ga::{
        GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
    };
    use crate::fitness::BatchFitnessFn;
    use crate::strategy::EvolutionaryHarness;

    type B = Flex;

    /// Sphere cost over the population tensor: sums squares along the genome
    /// axis. Minimise-sense, so the harness negates into canonical space.
    struct SphereCost;

    impl BatchFitnessFn<B, Tensor<B, 2>> for SphereCost {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2>,
            _device: &<B as burn::tensor::backend::BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            population
                .clone()
                .powf_scalar(2.0)
                .sum_dim(1)
                .squeeze_dim::<1>(1)
        }

        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Minimize
        }
    }

    fn harness(max_generations: usize) -> EvolutionaryHarness<B, GeneticAlgorithm<B>, SphereCost> {
        let params = GaConfig {
            pop_size: 8,
            genome_dim: 3,
            bounds: Bounds::new(-5.0, 5.0),
            mutation_sigma: NonNegativeRate::new(0.3),
            selection: GaSelection::Tournament { size: 2 },
            crossover: GaCrossover::BlxAlpha {
                alpha: NonNegativeRate::new(0.5),
            },
            replacement: GaReplacement::Elitist { elitism_k: 1 },
        };
        EvolutionaryHarness::new(
            GeneticAlgorithm::<B>::new(),
            params,
            SphereCost,
            42,
            Default::default(),
            max_generations,
        )
        .expect("valid params")
    }

    /// Drains a probe to exhaustion under a hard cap, returning the metrics
    /// sequence.
    ///
    /// The cap is the point: an unbounded `while p.advance().is_some() {}`
    /// turns a missing budget guard into an infinite loop, so the test hangs
    /// on CI instead of failing with a diagnostic. Draining past the budget
    /// and asserting the observed length turns the same defect into a fast,
    /// legible failure.
    fn drain<P: GenerationProbe>(p: &mut P, cap: usize) -> Vec<P::Metrics> {
        let mut out = Vec::new();
        for _ in 0..cap {
            match p.advance() {
                Some(m) => out.push(m),
                None => return out,
            }
        }
        out
    }

    /// `advance` yields exactly `max_generations` metrics, then `None`.
    #[test]
    fn advance_yields_exactly_the_budget_then_none() {
        let mut p = harness(3);
        p.begin();

        let seen = drain(&mut p, 3 + 4);

        assert_eq!(
            seen.len(),
            3,
            "one metrics value per generation in the budget"
        );
        assert_eq!(p.generation(), 3, "budget consumed exactly, not over-run");
    }

    /// Past exhaustion `advance` is a cheap no-op: no panic, no extra
    /// generation. This is the contract that lets a driver poll to completion
    /// without tracking a separate `done` flag.
    #[test]
    fn advance_past_exhaustion_does_not_step() {
        let mut p = harness(2);
        p.begin();
        assert_eq!(drain(&mut p, 2 + 4).len(), 2, "budget is 2 generations");

        for _ in 0..5 {
            assert!(p.advance().is_none(), "must stay exhausted");
        }
        assert_eq!(p.generation(), 2, "no generation ran past the budget");
    }

    /// `begin` re-seeds, so a second run is a byte-identical replay.
    ///
    /// This pins the property measured while scoping the `BenchEnv` rework:
    /// the evaluator's episode axis is degenerate on the evolution path, so a
    /// second "episode" is duplicated compute carrying zero information. If
    /// this ever fails, that reasoning needs revisiting.
    #[test]
    fn begin_replays_identically() {
        let mut p = harness(4);

        p.begin();
        let first: Vec<f32> = drain(&mut p, 4 + 4)
            .iter()
            .map(super::StrategyMetrics::best_fitness)
            .collect();

        p.begin();
        let second: Vec<f32> = drain(&mut p, 4 + 4)
            .iter()
            .map(super::StrategyMetrics::best_fitness)
            .collect();

        assert_eq!(first.len(), 4, "budget is 4 generations");
        assert_eq!(
            first, second,
            "begin must re-seed: a replay is bit-identical, not merely similar"
        );
    }
}
