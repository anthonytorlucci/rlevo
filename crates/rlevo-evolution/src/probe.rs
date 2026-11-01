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
//! one past it. `emit_metrics` is `latest_metrics` as
//! [`Metric`]s, under `ea/` or `coea/`, and none before the first generation.
//!
//! The trait keeps names of its own because its caller is different. The
//! harness is generic: it knows nothing about strategies or populations, and
//! it must drive any kind of generation loop through one interface, the
//! co-evolutionary one included. `Option` gives it the stop signal.
//! `emit_metrics` gives it the report's metrics, with the name and the rule
//! an agent's have, so a report reads both kinds of trial the same way. A
//! person writing a loop by hand is better served by a call that advances
//! and a question that reads as a question.
//!
//! # What the harness adds, and one thing to set
//!
//! A `GenerationTrial` stops when `advance` returns `None` or after
//! `EvaluatorConfig::max_generations` generations, whichever comes first.
//! `max_generations` defaults to 1000. Set it at least as large as the loop's
//! budget. A trial that the cap ends with budget left logs one warning, and
//! the `generations` scalar in its report is the cap, not the budget.
//!
//! To know whether budget was left, the trial calls `advance` once more when
//! it reaches the cap. For the loops here that call returns `None` without
//! running anything when the budget is spent, because `advance` asks
//! `is_done` first. A driver of your own must do the same.
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
//!   `emit_metrics` returns those metrics as [`Metric`]s, none before the
//!   first generation, and the driver is `Send`.
//!
//! A driver that calls [`Strategy::ask`] and [`Strategy::tell`] itself must
//! negate a `Minimize` objective and sanitise the fitness itself, in that
//! order: `sanitize_fitness_tensor(cost.neg())`, with
//! [`sanitize_fitness_tensor`](crate::fitness::sanitize_fitness_tensor). One
//! that wraps an [`EvolutionaryLoop`] inherits both.
//!
//! [`EvolutionaryLoop`]: crate::strategy::EvolutionaryLoop
//! [`CoEvolutionaryLoop`]: crate::coevolution::CoEvolutionaryLoop
//! [`Metric`]: rlevo_core::fitness::Metric

use rlevo_core::evaluation::GenerationProbe;
use rlevo_core::fitness::Metric;

use crate::coevolution::{CoEAMetrics, CoEvolutionaryAlgorithm, CoEvolutionaryLoop};
use crate::fitness::BatchFitnessFn;
use crate::strategy::{EvolutionaryLoop, Strategy, StrategyMetrics};

fn scalar(name: &str, value: f64) -> Metric {
    Metric::Scalar {
        name: name.to_string(),
        value,
    }
}

/// The single-population generation summary as harness metrics.
///
/// Each name is the [`StrategyMetrics`] constant beside the value's accessor.
/// The `ea/` prefix keeps a generation trial's scalars apart from the episodic
/// names (`return/mean`, `episode_length/mean`, …) in the same `TrialReport`
/// metric maps.
#[allow(clippy::cast_precision_loss)]
fn strategy_metrics(m: &StrategyMetrics) -> Vec<Metric> {
    vec![
        scalar(StrategyMetrics::GENERATION, m.generation() as f64),
        scalar(StrategyMetrics::POPULATION_SIZE, m.population_size() as f64),
        scalar(StrategyMetrics::BEST_FITNESS, f64::from(m.best_fitness())),
        scalar(
            StrategyMetrics::BEST_FITNESS_EVER,
            f64::from(m.best_fitness_ever()),
        ),
        scalar(StrategyMetrics::MEAN_FITNESS, f64::from(m.mean_fitness())),
        scalar(StrategyMetrics::WORST_FITNESS, f64::from(m.worst_fitness())),
        scalar(StrategyMetrics::BROKEN_COUNT, m.broken_count() as f64),
    ]
}

/// The two-population co-evolutionary summary as harness metrics.
///
/// `coea/binding_fitness` is the canonical (engine-space) value — the weaker
/// population's best — while the per-population `best`/`mean` fields are in the
/// objective's natural declared sense (ADR 0023). Emitting both keeps that
/// distinction visible in the report rather than collapsing it.
#[allow(clippy::cast_precision_loss)]
fn coea_metrics(m: &CoEAMetrics) -> Vec<Metric> {
    vec![
        scalar(CoEAMetrics::GENERATION, m.generation as f64),
        scalar(CoEAMetrics::BINDING_FITNESS, f64::from(m.binding_fitness)),
        scalar(CoEAMetrics::BEST_FITNESS_A, f64::from(m.best_fitness_a)),
        scalar(CoEAMetrics::BEST_FITNESS_B, f64::from(m.best_fitness_b)),
        scalar(CoEAMetrics::MEAN_FITNESS_A, f64::from(m.mean_fitness_a)),
        scalar(CoEAMetrics::MEAN_FITNESS_B, f64::from(m.mean_fitness_b)),
        scalar(CoEAMetrics::HOF_SIZE_A, m.hof_size_a as f64),
        scalar(CoEAMetrics::HOF_SIZE_B, m.hof_size_b as f64),
    ]
}

impl<S, F> GenerationProbe for EvolutionaryLoop<S, F>
where
    S: Strategy,
    F: BatchFitnessFn<S::Genome>,
{
    type Metrics = StrategyMetrics;

    fn begin(&mut self) {
        EvolutionaryLoop::<S, F>::reset(self);
    }

    fn advance(&mut self) -> Option<Self::Metrics> {
        if self.is_done() {
            return None;
        }
        EvolutionaryLoop::<S, F>::step(self);
        self.latest_metrics().cloned()
    }

    /// The latest generation's metrics, under `ea/`.
    fn emit_metrics(&self) -> Vec<Metric> {
        self.latest_metrics()
            .map_or_else(Vec::new, strategy_metrics)
    }
}

impl<C> GenerationProbe for CoEvolutionaryLoop<C>
where
    C: CoEvolutionaryAlgorithm,
{
    type Metrics = CoEAMetrics;

    fn begin(&mut self) {
        CoEvolutionaryLoop::<C>::reset(self);
    }

    fn advance(&mut self) -> Option<Self::Metrics> {
        if self.is_done() {
            return None;
        }
        CoEvolutionaryLoop::<C>::step(self);
        self.latest_metrics().cloned()
    }

    /// The latest generation's metrics, under `coea/`.
    fn emit_metrics(&self) -> Vec<Metric> {
        self.latest_metrics().map_or_else(Vec::new, coea_metrics)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, Tensor, TensorData};
    use rlevo_core::bounds::Bounds;
    use rlevo_core::fitness::Metric;
    use rlevo_core::objective::ObjectiveSense;
    use rlevo_core::rate::NonNegativeRate;

    use super::GenerationProbe;
    use crate::algorithms::ga::{
        GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
    };
    use crate::coevolution::{
        CoEvolutionaryLoop, CompetitiveCoEA, CompetitiveCoEAParams, CoupledFitness,
    };
    use crate::fitness::BatchFitnessFn;
    use crate::strategy::EvolutionaryLoop;

    /// Sphere cost over the population tensor: sums squares along the genome
    /// axis. Minimise-sense, so the loop negates into canonical space.
    struct SphereCost;

    impl BatchFitnessFn<Tensor<2>> for SphereCost {
        fn evaluate_batch(&mut self, population: &Tensor<2>, _device: &Device) -> Tensor<1> {
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

    fn evo_loop(max_generations: usize) -> EvolutionaryLoop<GeneticAlgorithm, SphereCost> {
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
            GeneticAlgorithm::new(),
            params,
            SphereCost,
            42,
            Device::flex(),
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

    /// The value of the scalar called `name`, if `metrics` has one.
    fn scalar_named(metrics: &[Metric], name: &str) -> Option<f64> {
        metrics.iter().find_map(|m| match m {
            Metric::Scalar { name: n, value } if n == name => Some(*value),
            _ => None,
        })
    }

    /// `emit_metrics` describes the run so far: none before the first
    /// generation, then the latest generation's summary, which a second read
    /// repeats unchanged.
    #[test]
    fn emit_metrics_reports_the_latest_generation() {
        let mut p = evo_loop(3);
        p.begin();
        assert!(
            p.emit_metrics().is_empty(),
            "no generation has run, so there is nothing to report"
        );

        assert_eq!(drain(&mut p, 2).len(), 2, "two generations run");
        let read = p.emit_metrics();
        assert_eq!(
            scalar_named(&read, super::StrategyMetrics::GENERATION),
            Some(2.0),
            "the second generation's metrics; got {read:?}"
        );
        let latest = p.latest_metrics().expect("metrics after a generation");
        assert_eq!(
            scalar_named(&read, super::StrategyMetrics::BEST_FITNESS_EVER),
            Some(f64::from(latest.best_fitness_ever())),
            "the running best is the loop's own"
        );
        assert_eq!(
            format!("{read:?}"),
            format!("{:?}", p.emit_metrics()),
            "a read changes nothing, so a second read repeats the first"
        );
    }

    /// `begin` starts a new run, so a read after it reports none of the
    /// previous run's generations.
    #[test]
    fn emit_metrics_after_begin_reports_nothing_of_the_last_run() {
        let mut p = evo_loop(3);
        p.begin();
        assert_eq!(drain(&mut p, 3 + 4).len(), 3, "budget is 3 generations");
        assert!(!p.emit_metrics().is_empty(), "the run reports its metrics");

        p.begin();
        assert!(
            p.emit_metrics().is_empty(),
            "a new run has no generation yet; got {:?}",
            p.emit_metrics()
        );
    }

    /// Fitness `i` for row `i` of each population, maximised.
    struct RowRamp;

    impl CoupledFitness for RowRamp {
        fn evaluate_coupled(&self, populations: &[Tensor<2>]) -> Vec<Tensor<1>> {
            populations
                .iter()
                .map(|p| {
                    let n = p.dims()[0];
                    #[allow(clippy::cast_precision_loss)]
                    let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
                    Tensor::<1>::from_data(TensorData::new(v, [n]), &p.device())
                })
                .collect()
        }

        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Maximize
        }
    }

    /// The co-evolutionary loop reads the same way, under `coea/`.
    #[test]
    fn coevolution_emit_metrics_reports_the_latest_generation() {
        let ga = || GaConfig {
            pop_size: 4,
            genome_dim: 2,
            bounds: Bounds::new(0.0, 1.0),
            mutation_sigma: NonNegativeRate::new(0.1),
            selection: GaSelection::Tournament { size: 2 },
            crossover: GaCrossover::BlxAlpha {
                alpha: NonNegativeRate::new(0.5),
            },
            replacement: GaReplacement::Elitist { elitism_k: 1 },
        };
        let algo = CompetitiveCoEA::new(GeneticAlgorithm::new(), GeneticAlgorithm::new(), RowRamp);
        let params: CompetitiveCoEAParams<GaConfig, GaConfig> = CompetitiveCoEAParams {
            params_a: ga(),
            params_b: ga(),
        };
        let mut p =
            CoEvolutionaryLoop::new(algo, params, 7, Device::flex(), 2).expect("valid params");
        p.begin();
        assert!(
            p.emit_metrics().is_empty(),
            "no generation has run, so there is nothing to report"
        );

        assert_eq!(drain(&mut p, 2 + 4).len(), 2, "budget is 2 generations");
        let read = p.emit_metrics();
        assert_eq!(
            scalar_named(&read, super::CoEAMetrics::GENERATION),
            Some(2.0),
            "the last generation's metrics; got {read:?}"
        );
        assert_eq!(
            scalar_named(&read, super::CoEAMetrics::BINDING_FITNESS),
            Some(3.0),
            "the ramp's best row binds both populations; got {read:?}"
        );
    }

    /// The metric names are wire format, pinned here as strings: a report or
    /// recording an earlier binary wrote carries them, so changing one of the
    /// constants is a breaking change, not a rename. Each summary emits every
    /// name of its type, in this order, and nothing else.
    #[test]
    fn metric_names_are_pinned_wire_format() {
        let names = |metrics: Vec<Metric>| -> Vec<String> {
            metrics
                .into_iter()
                .map(|m| match m {
                    Metric::Scalar { name, .. } => name,
                    other => panic!("a summary emits scalars only; got {other:?}"),
                })
                .collect()
        };

        let strategy = super::StrategyMetrics::from_host_fitness(1, &[1.0, 2.0], 2.0);
        assert_eq!(
            names(super::strategy_metrics(&strategy)),
            [
                "ea/generation",
                "ea/population_size",
                "ea/best_fitness",
                "ea/best_fitness_ever",
                "ea/mean_fitness",
                "ea/worst_fitness",
                "ea/broken_count",
            ]
        );

        let coea = super::CoEAMetrics {
            generation: 1,
            best_fitness_a: 0.0,
            best_fitness_b: 0.0,
            mean_fitness_a: 0.0,
            mean_fitness_b: 0.0,
            binding_fitness: 0.0,
            hof_size_a: 0,
            hof_size_b: 0,
        };
        assert_eq!(
            names(super::coea_metrics(&coea)),
            [
                "coea/generation",
                "coea/binding_fitness",
                "coea/best_fitness_a",
                "coea/best_fitness_b",
                "coea/mean_fitness_a",
                "coea/mean_fitness_b",
                "coea/hof_size_a",
                "coea/hof_size_b",
            ]
        );
    }
}
