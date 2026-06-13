//! AC §7.3 — hall-of-fame mitigation prevents *forgetting* in a competitive
//! host–parasite coverage game over a non-stationary regime (Hillis 1990).
//!
//! Pure rock-paper-scissors is payoff-symmetric — every strategy's winrate
//! against any balanced opponent set is exactly 1/3 — so no winrate-based
//! metric can detect what a hall of fame does. We therefore use an asymmetric
//! construction with a genuine competence dimension and a non-stationary
//! adversary:
//!
//! - **Solver** (population A): `K = 3` coverage genes in `[0,1]`. Covering a
//!   target pays off when it is probed, but every unit of coverage costs
//!   `LAMBDA` — so the solver cannot afford to cover all targets at once.
//! - **Tester** (population B): probes one target (`argmax` of `K` genes); it
//!   is rewarded for matching the **current regime target**, which cycles
//!   every `PERIOD` generations — a controlled non-stationary opponent.
//!
//! Against the *current* (concentrated) regime, the solver covers only the
//! probed target and drops the rest (the budget cost outweighs an un-probed
//! target) — it **forgets** as the regime moves on. With a
//! [`HallOfFameFitness`], the solver is also scored against a *diverse* archive
//! of past regime targets, so retaining coverage of all of them pays off. The
//! shipped average-blend hall of fame (unchanged, blend `0.3`) keeps coverage
//! broad. The acceptance assertion is on **retained coverage of the
//! most-neglected target**: high with the hall of fame, low (forgetting)
//! without it.

#![allow(clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicUsize, Ordering};

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::coevolution::{
    CoEvolutionaryAlgorithm, CompetitiveCoEA, CompetitiveCoEAParams, CoupledFitness,
    HallOfFameFitness,
};

type B = Flex;

const POP: usize = 60;
const GENS: usize = 200;
const K: usize = 3; // number of targets / genome width
const PERIOD: usize = 4; // generations the tester dwells on each regime target
const LAMBDA: f32 = 0.08; // per-unit coverage budget cost

/// Per-individual coverage vector (population A), genes clamped to `[0,1]`.
fn coverage(pop: &Tensor<B, 2>) -> Vec<[f32; K]> {
    let dims = pop.dims();
    let (n, d) = (dims[0], dims[1]);
    debug_assert_eq!(d, K);
    let flat = pop.clone().into_data().into_vec::<f32>().unwrap();
    (0..n)
        .map(|i| {
            let mut c = [0.0f32; K];
            for (k, slot) in c.iter_mut().enumerate() {
                *slot = flat[i * d + k].clamp(0.0, 1.0);
            }
            c
        })
        .collect()
}

/// Per-individual probed target (population B) = argmax of the genome.
fn targets(pop: &Tensor<B, 2>) -> Vec<usize> {
    let dims = pop.dims();
    let (n, d) = (dims[0], dims[1]);
    let flat = pop.clone().into_data().into_vec::<f32>().unwrap();
    (0..n)
        .map(|i| {
            let row = &flat[i * d..i * d + d];
            let mut best = 0usize;
            let mut bv = row[0];
            for (k, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = k;
                }
            }
            best
        })
        .collect()
}

/// Asymmetric host–parasite fitness over a **non-stationary** regime
/// (minimization).
///
/// Population 0 is the solver; population 1 is the tester. The tester is
/// rewarded for probing the current regime target, which cycles every
/// [`PERIOD`] generations — a controlled non-stationary adversary. The solver
/// is rewarded for covering the probed targets but pays a budget cost
/// [`LAMBDA`] per unit of coverage, so beating the current regime means
/// dropping coverage of others (the seed of forgetting).
///
/// A generation counter is held internally (incremented once per generation,
/// detected by the opponent population having the full [`POP`] rows — archive
/// sub-evaluations, which pass fewer rows, do not advance it).
struct CoverageForgettingFitness {
    step: AtomicUsize,
}

impl CoverageForgettingFitness {
    fn new() -> Self {
        Self {
            step: AtomicUsize::new(0),
        }
    }
}

impl CoupledFitness<B> for CoverageForgettingFitness {
    fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>> {
        debug_assert_eq!(populations.len(), 2);
        let device = populations[0].device();
        let cov = coverage(&populations[0]);
        let tgt = targets(&populations[1]);

        // Advance the regime clock only on full-population (current-opponent)
        // evaluations, not on smaller hall-of-fame archive sub-evaluations.
        let generation = if populations[1].dims()[0] == POP {
            self.step.fetch_add(1, Ordering::Relaxed)
        } else {
            self.step.load(Ordering::Relaxed)
        };
        let regime_target = (generation / PERIOD) % K;

        // Solver fitness: 1 - mean coverage of probed targets + budget cost.
        let solver: Vec<f32> = cov
            .iter()
            .map(|c| {
                let benefit = if tgt.is_empty() {
                    0.0
                } else {
                    tgt.iter().map(|&t| c[t]).sum::<f32>() / tgt.len() as f32
                };
                let cost = LAMBDA * c.iter().sum::<f32>();
                (1.0 - benefit) + cost
            })
            .collect();

        // Tester fitness: match the current regime target (drives the tester
        // population to probe the cycling regime).
        let tester: Vec<f32> = tgt
            .iter()
            .map(|&t| if t == regime_target { 0.0 } else { 1.0 })
            .collect();

        vec![
            Tensor::<B, 1>::from_data(TensorData::new(solver.clone(), [solver.len()]), &device),
            Tensor::<B, 1>::from_data(TensorData::new(tester.clone(), [tester.len()]), &device),
        ]
    }
}

fn ga_config() -> GaConfig {
    GaConfig {
        pop_size: POP,
        genome_dim: K,
        bounds: (0.0, 1.0),
        mutation_sigma: 0.1,
        selection: GaSelection::Tournament { size: 3 },
        crossover: GaCrossover::Uniform { p: 0.5 },
        replacement: GaReplacement::Elitist { elitism_k: 1 },
    }
}

/// Per generation, the solver's *minimum* mean coverage across the `K`
/// targets — the strength of the most-neglected ("forgotten") target. High =>
/// the solver retains all targets; near zero => it has abandoned one.
fn run_coverage<F: CoupledFitness<B>>(fitness: F, seed: u64) -> Vec<f32> {
    let device = Default::default();
    let algo = CompetitiveCoEA::new(
        GeneticAlgorithm::<B>::new(),
        GeneticAlgorithm::<B>::new(),
        fitness,
    );
    let params = CompetitiveCoEAParams {
        params_a: ga_config(),
        params_b: ga_config(),
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = algo.init(&params, &mut rng, &device);
    let mut traj = Vec::with_capacity(GENS);
    for _ in 0..GENS {
        let (next, _m) = algo.step(&params, state, &mut rng, &device);
        state = next;
        let cov = coverage(&state.state_a.population);
        let n = cov.len().max(1) as f32;
        let min_cov: f32 = (0..K)
            .map(|k| cov.iter().map(|c| c[k]).sum::<f32>() / n)
            .fold(f32::INFINITY, f32::min);
        traj.push(min_cov);
    }
    traj
}

/// Mean total coverage over the last `window` generations.
fn tail_coverage(traj: &[f32], window: usize) -> f32 {
    let start = traj.len().saturating_sub(window);
    let tail = &traj[start..];
    tail.iter().sum::<f32>() / tail.len() as f32
}

/// Observation harness — run with:
/// `cargo test -p rlevo-evolution --test coevolution_cycling -- --ignored --nocapture`
#[test]
#[ignore = "observation only; prints trajectory statistics"]
fn observe_dynamics() {
    let device = Default::default();
    let seeds = [1_u64, 7, 42, 100, 5, 99, 13, 77];
    for &blend in &[0.0_f32, 0.3, 0.5, 0.7, 0.9] {
        let mut tails: Vec<f32> = Vec::new();
        for &seed in &seeds {
            let traj = if blend == 0.0 {
                run_coverage(CoverageForgettingFitness::new(), seed)
            } else {
                run_coverage(
                    HallOfFameFitness::new(CoverageForgettingFitness::new(), 2, POP, K, &device)
                        .with_blend_weight(blend),
                    seed,
                )
            };
            tails.push(tail_coverage(&traj, 50));
        }
        let avg = tails.iter().sum::<f32>() / tails.len() as f32;
        let min = tails.iter().copied().fold(f32::INFINITY, f32::min);
        let max = tails.iter().copied().fold(0.0, f32::max);
        eprintln!(
            "blend={:.2}: tail50 min-coverage (retention) avg={:.2} min={:.2} max={:.2}  per-seed={:?}",
            blend,
            avg,
            min,
            max,
            tails.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>()
        );
    }
}

/// Mean tail retention (most-forgotten target's coverage) over a set of seeds,
/// returned as `(per_seed_no_hof, per_seed_with_hof)`.
fn forgetting_sweep(seeds: &[u64], blend: f32) -> (Vec<f32>, Vec<f32>) {
    let device = Default::default();
    let mut no_hof = Vec::new();
    let mut with_hof = Vec::new();
    for &seed in seeds {
        no_hof.push(tail_coverage(&run_coverage(CoverageForgettingFitness::new(), seed), 50));
        with_hof.push(tail_coverage(
            &run_coverage(
                HallOfFameFitness::new(CoverageForgettingFitness::new(), 2, POP, K, &device)
                    .with_blend_weight(blend),
                seed,
            ),
            50,
        ));
    }
    (no_hof, with_hof)
}

/// AC §7.3: the shipped average-blend [`HallOfFameFitness`] (blend `0.3`)
/// prevents *forgetting* — the solver retains coverage of every regime target
/// — while without it the solver forgets targets as the regime cycles.
#[test]
fn hall_of_fame_prevents_forgetting() {
    let seeds = [1_u64, 7, 42, 100, 5, 99, 13, 77];
    let (no_hof, with_hof) = forgetting_sweep(&seeds, 0.3);

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let hof_mean = mean(&with_hof);
    let nohof_mean = mean(&no_hof);
    let hof_min = with_hof.iter().copied().fold(f32::INFINITY, f32::min);
    let forgot = no_hof.iter().filter(|&&c| c < 0.5).count();

    // With HoF, retention is robustly high on every seed.
    assert!(
        hof_min >= 0.75,
        "HoF should retain coverage of all targets on every seed; min={hof_min:.2}, per-seed={with_hof:?}"
    );
    // Without HoF, average retention is substantially lower.
    assert!(
        nohof_mean <= 0.70,
        "without HoF, mean retention should be markedly lower; got {nohof_mean:.2}, per-seed={no_hof:?}"
    );
    assert!(
        hof_mean - nohof_mean >= 0.15,
        "HoF should retain markedly more coverage; hof={hof_mean:.2} nohof={nohof_mean:.2}"
    );
    // Forgetting is observable without HoF on multiple seeds.
    assert!(
        forgot >= 2,
        "forgetting (retention < 0.5) should be observable without HoF; count={forgot}, per-seed={no_hof:?}"
    );
}
