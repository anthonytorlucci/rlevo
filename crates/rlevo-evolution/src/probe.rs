//! The evolutionary drive seam: [`GenerationProbe`].
//!
//! An evolutionary run is not an episodic environment interaction. It has no
//! observation, no action, and no episode axis — driving one to its generation
//! budget twice from the same seed produces a byte-identical replay, so a
//! second "episode" is duplicated compute carrying zero information.
//!
//! `GenerationProbe` is the shape that says so: something that begins, then
//! advances one generation at a time until its budget is spent, reporting
//! typed metrics as it goes. It is the only way the evaluation harness drives
//! [`EvolutionaryLoop`] and [`CoEvolutionaryLoop`] (ADR 0076).
//!
//! # Two ways to drive a loop
//!
//! | Who drives | Start | One generation | Finished when |
//! |---|---|---|---|
//! | your own code | `reset()` | `step()` | `is_done()` |
//! | the evaluation harness | `begin()` | `advance()` | `advance()` returns `None` |
//!
//! A run is driven by one row or the other.
//!
//! - **Driving it yourself** is one optimisation run, with your own logging
//!   and your own stopping rule. This is the first row, and it is the
//!   inherent API of the two loops.
//! - **Giving it to the harness** is many seeded trials with a report at the
//!   end. `Evaluator::run_trials` takes a closure and calls it once per
//!   trial with that trial's seed; the closure builds a new loop and wraps it
//!   in a `GenerationTrial`. The harness then makes every call, `begin`
//!   included, so the loop needs no `reset` first. You write neither `begin`
//!   nor `advance`; they are the harness's side of the seam.
//!
//! The second row is built from the first, in this file. `begin` is `reset`.
//! `advance` is "if `is_done`, return `None`; otherwise `step`, and return a
//! copy of `latest_metrics`". So the harness always checks the budget before
//! it runs a generation, exactly as the hand-written
//! `while !evo_loop.is_done() { evo_loop.step(); }` does, and neither runs
//! one past it.
//!
//! The trait keeps names of its own because its caller is different. The
//! harness is generic: it knows nothing about strategies or populations, it
//! needs each generation's metrics by value, and it must drive any kind of
//! generation loop through one interface, the co-evolutionary one included.
//! `Option` gives it the stop signal and the metrics in one value. A person
//! writing a loop by hand is better served by a call that advances and a
//! question that reads as a question.
//!
//! # What the harness adds, and one thing to set
//!
//! A `GenerationTrial` stops when `advance` returns `None` or after
//! `EvaluatorConfig::max_steps` generations, whichever comes first, and it
//! does not warn when its own cap is the one that ends the trial. `max_steps`
//! defaults to 1000. Set it at least as large as the loop's budget. The
//! `generations` scalar in each trial's report says how many ran.
//!
//! # Writing a driver of your own
//!
//! A type that runs generations and should be usable both ways offers both
//! rows, and builds the second from the first, as the two impls below do:
//!
//! - for a person: `new`, `reset`, `step` (returning nothing), `is_done`,
//!   and a way to read the last generation's metrics;
//! - for the harness: a `GenerationProbe` impl whose `begin` is `reset` and
//!   whose `advance` is `is_done`, then `step`, then a clone of those
//!   metrics. `begin` must re-seed, so that a second run replays the first.
//!   The metrics type implements `MetricsProvider`, and the driver is `Send`.
//!
//! A driver that calls [`Strategy::ask`] and [`Strategy::tell`] itself must
//! negate a `Minimize` objective and sanitise the fitness itself, in that
//! order: `sanitize_fitness_tensor(cost.neg())`, with
//! [`sanitize_fitness_tensor`](crate::fitness::sanitize_fitness_tensor). One
//! that wraps an [`EvolutionaryLoop`] inherits both.
//!
//! [`EvolutionaryLoop`]: crate::strategy::EvolutionaryLoop
//! [`CoEvolutionaryLoop`]: crate::coevolution::CoEvolutionaryLoop

use burn::tensor::backend::Backend;

use rlevo_core::evaluation::GenerationProbe;
use rlevo_core::fitness::{Metric, MetricsProvider};

use crate::coevolution::{CoEAMetrics, CoEvolutionaryAlgorithm, CoEvolutionaryLoop};
use crate::fitness::BatchFitnessFn;
use crate::strategy::{EvolutionaryLoop, Strategy, StrategyMetrics};

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

impl<B, S, F> GenerationProbe for EvolutionaryLoop<B, S, F>
where
    B: Backend,
    S: Strategy<B>,
    F: BatchFitnessFn<B, S::Genome>,
{
    type Metrics = StrategyMetrics;

    fn begin(&mut self) {
        EvolutionaryLoop::<B, S, F>::reset(self);
    }

    fn advance(&mut self) -> Option<Self::Metrics> {
        if self.is_done() {
            return None;
        }
        EvolutionaryLoop::<B, S, F>::step(self);
        self.latest_metrics().cloned()
    }
}

impl<B, C> GenerationProbe for CoEvolutionaryLoop<B, C>
where
    B: Backend,
    C: CoEvolutionaryAlgorithm<B>,
{
    type Metrics = CoEAMetrics;

    fn begin(&mut self) {
        CoEvolutionaryLoop::<B, C>::reset(self);
    }

    fn advance(&mut self) -> Option<Self::Metrics> {
        if self.is_done() {
            return None;
        }
        CoEvolutionaryLoop::<B, C>::step(self);
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
    use crate::strategy::EvolutionaryLoop;

    type B = Flex;

    /// Sphere cost over the population tensor: sums squares along the genome
    /// axis. Minimise-sense, so the loop negates into canonical space.
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

    fn evo_loop(max_generations: usize) -> EvolutionaryLoop<B, GeneticAlgorithm<B>, SphereCost> {
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
        EvolutionaryLoop::new(
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
        let mut p = evo_loop(3);
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
        let mut p = evo_loop(2);
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
        let mut p = evo_loop(4);

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
