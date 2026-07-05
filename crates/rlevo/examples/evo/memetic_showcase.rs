//! Memetic-algorithm showcase: bare DE vs `MemeticWrapper<DE, HillClimbing>`
//! on Rastrigin-D10, compared by **evaluations-to-target**.
//!
//! The other `*_showcase` examples give every strategy the same *generation*
//! budget. That comparison would flatter a memetic strategy: local refinement
//! spends extra fitness evaluations inside every generation (here `TopK{1}` ×
//! `max_iters=20` adds up to 20 evals on top of DE's 30), so its
//! per-generation `best` is bought with a bigger budget. The honest question
//! is the one this example asks: *how many total fitness evaluations does each
//! configuration need to first reach a shared target cost?*
//!
//! Evaluations are counted by a fitness wrapper that increments a shared
//! counter per row of every `evaluate_batch`. Each `MemeticWrapper`
//! local-search probe routes through a `[1, D]` batch, so refinement
//! evaluations are captured automatically; the memetic runs share one counter
//! across both fitness instances they own (harness + wrapper).
//!
//! # Reading the output
//!
//! Each configuration prints one row: the eval-counter reading at the first
//! generation where `best_fitness_ever` crossed below each target, then the
//! final best after the full generation budget. Lower eval counts are better;
//! `miss` means the target was never reached. Rastrigin-D10 has `f* = 0` over
//! `[-5.12, 5.12]^10`; targets below ~10 are deliberately absent because bare
//! DE often stalls there, making "fewer evals to a shared target" ill-posed.
//!
//! What to look for:
//!
//! - The **headline** row (`TopK{1}`, best-improvement hill climbing,
//!   `max_iters=20`, Lamarckian) reaches each target in substantially fewer
//!   evaluations than bare DE — the same calibrated configuration pinned by
//!   the `memetic_rastrigin` acceptance test (~74% fewer at target 30 on the
//!   pinned seed), which verified the win across five seeds.
//! - The **writeback** rows vary only `WritebackPolicy`. Lamarckian writes
//!   refined genomes back into the population; Baldwinian keeps the genomes
//!   untouched and only credits the refined fitness; `Partial(0.5)` (the
//!   default) flips a seeded coin per refined row. The policy visibly moves
//!   the eval counts — refinement overhead is paid either way, so a policy
//!   that exploits the refined genomes poorly can even fall behind bare DE at
//!   tight targets.
//! - The **untuned-defaults** row (`TopK{3}` × `default_for` hill climbing:
//!   `max_iters=100`, step ≈ 1.0) dominates everything on *this* landscape —
//!   and that is the trap. Rastrigin is fully **separable** with local minima
//!   on a unit grid, so axis-aligned hill climbing with basin-width steps is
//!   almost a direct solver for it; the same configuration's advantage
//!   collapses on non-separable or rotated landscapes (the reason benchmark
//!   suites include rotated variants). Treat that row as a demonstration that
//!   memetic performance is landscape-dependent, not as a config to copy.
//!
//! Rayon is pinned to one thread and every run shares the same seed, so the
//! printed numbers are bit-reproducible across invocations.
//!
//! Run with `cargo run --release -p rlevo --example memetic_showcase`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend, backend::BackendTypes};

use rlevo::envs::landscapes::rastrigin::Rastrigin;
use rlevo::evo::algorithms::de::{DeConfig, DifferentialEvolution};
use rlevo::evo::algorithms::memetic::{
    CoveragePolicy, MemeticParams, MemeticWrapper, WritebackPolicy,
};
use rlevo::evo::fitness::BatchFitnessFn;
use rlevo::evo::local_search::{HillClimbVariant, HillClimbing, HillClimbingParams};
use rlevo::evo::strategy::{EvolutionaryHarness, Strategy};

type B = Flex;

const DIM: usize = 10;
const BOUNDS: rlevo_core::bounds::Bounds = rlevo_core::bounds::Bounds::new(-5.12, 5.12);
/// The seed pinned by the `memetic_rastrigin` acceptance test, so this
/// example's headline row reproduces the test's provenance numbers.
const SEED: u64 = 7;
/// Generation budget; generous enough that every configuration converges past
/// the easiest target long before exhausting it.
const MAX_GENS: usize = 600;
/// Shared target costs, in crossing order (best-so-far only decreases).
const TARGETS: [f32; 3] = [40.0, 30.0, 20.0];

/// Row-counting Rastrigin fitness: increments `evals` by the number of rows in
/// each `evaluate_batch`, so `MemeticWrapper`'s `[1, D]` local-search probes
/// are counted alongside full population scorings.
struct CountingRastrigin {
    landscape: Rastrigin,
    evals: Arc<AtomicUsize>,
}

impl<Bk: Backend> BatchFitnessFn<Bk, Tensor<Bk, 2>> for CountingRastrigin {
    fn evaluate_batch(
        &mut self,
        population: &Tensor<Bk, 2>,
        device: &<Bk as BackendTypes>::Device,
    ) -> Tensor<Bk, 1> {
        let dims: [usize; 2] = population.dims();
        let (pop, dim): (usize, usize) = (dims[0], dims[1]);
        self.evals.fetch_add(pop, Ordering::Relaxed);
        let flat: Vec<f32> = population.clone().into_data().into_vec::<f32>().unwrap();
        let mut out: Vec<f32> = Vec::with_capacity(pop);
        for r in 0..pop {
            let start: usize = r * dim;
            let row: Vec<f64> = flat[start..start + dim].iter().map(|&v| f64::from(v)).collect();
            #[allow(clippy::cast_possible_truncation)]
            out.push(self.landscape.evaluate(&row) as f32);
        }
        Tensor::<Bk, 1>::from_data(TensorData::new(out, [pop]), device)
    }

    fn sense(&self) -> rlevo_core::objective::ObjectiveSense {
        // Rastrigin is a cost surface — lower is better.
        rlevo_core::objective::ObjectiveSense::Minimize
    }
}

/// Drives one strategy for [`MAX_GENS`] generations and prints its row: the
/// eval count at the first crossing of each entry in [`TARGETS`], then the
/// final `best_fitness_ever`. `evals` must be the same counter the strategy's
/// own fitness instances feed (for the memetic rows) so refinement
/// evaluations are included.
fn run_row<S>(label: &str, strategy: S, params: S::Params, evals: &Arc<AtomicUsize>)
where
    S: Strategy<B, Genome = Tensor<B, 2>>,
    S::Params: std::fmt::Debug + rlevo_core::config::Validate,
{
    let device: <B as BackendTypes>::Device = Default::default();
    let fitness: CountingRastrigin = CountingRastrigin {
        landscape: Rastrigin::new(DIM),
        evals: evals.clone(),
    };
    let mut harness = EvolutionaryHarness::<B, S, CountingRastrigin>::new(
        strategy, params, fitness, SEED, device, MAX_GENS,
    ).expect("valid params");
    harness.reset();
    let mut crossings: [Option<usize>; TARGETS.len()] = [None; TARGETS.len()];
    loop {
        let step = harness.step(());
        let best: f32 = harness.latest_metrics().unwrap().best_fitness_ever;
        for (slot, &target) in crossings.iter_mut().zip(TARGETS.iter()) {
            if slot.is_none() && best < target {
                *slot = Some(evals.load(Ordering::Relaxed));
            }
        }
        if step.done {
            break;
        }
    }
    let final_best: f32 = harness.latest_metrics().unwrap().best_fitness_ever;
    let cells: String = TARGETS
        .iter()
        .zip(crossings.iter())
        .map(|(&target, crossing)| match crossing {
            Some(e) => format!("→{target:>2.0}: {e:>7} evals"),
            None => format!("→{target:>2.0}: {:>7}      ", "miss"),
        })
        .collect::<Vec<String>>()
        .join(" | ");
    println!("{label:>45} | {cells} | final best={final_best:.4e}");
}

/// One memetic row: wraps a fresh DE in `MemeticWrapper` with the given
/// hill-climbing params and policies. The wrapper's fitness instance shares
/// the row's eval counter, so harness scorings and local-search probes land
/// in one budget.
fn run_memetic_row(
    label: &str,
    de: &DeConfig,
    hc: HillClimbingParams,
    writeback: WritebackPolicy,
    coverage: CoveragePolicy,
) {
    let evals: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let wrapper_fitness: CountingRastrigin = CountingRastrigin {
        landscape: Rastrigin::new(DIM),
        evals: evals.clone(),
    };
    let strategy: MemeticWrapper<B, _, _, _> = MemeticWrapper::<B, _, _, _>::new(
        DifferentialEvolution::<B>::new(),
        HillClimbing,
        wrapper_fitness,
    );
    let params: MemeticParams<DeConfig, HillClimbingParams> = MemeticParams {
        inner: de.clone(),
        local: hc,
        writeback,
        coverage,
    };
    run_row(label, strategy, params, &evals);
}

/// The calibrated headline hill-climbing config from the acceptance test:
/// cheap enough per generation that refinement overhead never erases DE's
/// progress, greedy enough that each polished elite pulls the population
/// toward a basin.
fn headline_hc() -> HillClimbingParams {
    let mut hc: HillClimbingParams = HillClimbingParams::default_for(BOUNDS);
    hc.max_iters = 20;
    hc.step_size = 0.4;
    hc.step_decay = 0.5;
    hc.variant = HillClimbVariant::BestImprovement;
    hc
}

fn main() {
    // One thread keeps the Flex backend bit-reproducible run to run.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();

    println!(
        "
Memetic showcase — Rastrigin-D{DIM}, evals-to-target, Flex backend, seed={SEED}

Each row reports the TOTAL fitness evaluations consumed when best-so-far
first crossed below each target cost (lower is better; `miss` = never
reached within {MAX_GENS} generations), then the final best. Unlike the
fixed-generation showcases, this counts every evaluation — including the
ones MemeticWrapper's local search spends — so the comparison is
budget-fair. All rows share one DE config (pop=30, Rand1Bin, F=0.5,
CR=0.9) and one seed.\n\n{:-<120}",
        "",
    );

    let de: DeConfig = DeConfig::default_for(30, DIM);

    let bare_evals: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    run_row(
        "bare DE/Rand1Bin",
        DifferentialEvolution::<B>::new(),
        de.clone(),
        &bare_evals,
    );

    run_memetic_row(
        "memetic k=1/it=20/best/Lamarckian (headline)",
        &de,
        headline_hc(),
        WritebackPolicy::Lamarckian,
        CoveragePolicy::TopK { k: 1 },
    );
    run_memetic_row(
        "memetic k=1/it=20/best/Baldwinian",
        &de,
        headline_hc(),
        WritebackPolicy::Baldwinian,
        CoveragePolicy::TopK { k: 1 },
    );
    run_memetic_row(
        "memetic k=1/it=20/best/Partial(0.5) (default)",
        &de,
        headline_hc(),
        WritebackPolicy::default(),
        CoveragePolicy::TopK { k: 1 },
    );

    // Untuned `default_for` hill climbing (max_iters=100, step ≈ 1.0,
    // first-improvement) on three rows per generation — up to ~300 refinement
    // evals/gen on top of DE's 30. Dominates here because Rastrigin is
    // separable; see the module docs before copying it.
    let heavy: HillClimbingParams = HillClimbingParams::default_for(BOUNDS);
    run_memetic_row(
        "memetic k=3/untuned HC defaults (it=100)",
        &de,
        heavy,
        WritebackPolicy::Lamarckian,
        CoveragePolicy::TopK { k: 3 },
    );

    println!(
        "{:-<120}
Takeaway: the headline row is the robust, multi-seed-calibrated win the
acceptance test pins (~74% fewer evals than bare DE to reach 30 on the pinned
seed). The writeback rows show the policy choice alone can move — or lose — the
advantage at tight targets. The untuned-defaults row dominates only because
Rastrigin is separable and axis-aligned hill climbing with basin-width steps
exploits that perfectly; do not expect it to transfer to non-separable
problems. The general recipe: pick a target cost, count evaluations to reach
it (as this example does), and tune writeback/coverage/budget against that
number rather than against per-generation best.",
        "",
    );
}
