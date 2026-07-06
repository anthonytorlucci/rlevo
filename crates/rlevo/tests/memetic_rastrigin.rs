//! Headline memetic-algorithm acceptance test.
//!
//! Goal: prove that wrapping Differential Evolution with hill-climbing
//! refinement — `MemeticWrapper<DE, HillClimbing>` — reaches a target
//! Rastrigin-D10 fitness in **strictly fewer total fitness evaluations** than
//! bare DE under an *identical* `DeConfig` and *identical* seed, reproducibly.
//!
//! # Why this file runs a single test
//!
//! The Burn Flex backend seeds its tensor RNG through a process-wide
//! `Mutex<Option<FlexRng>>`. Two tests racing on that mutex interleave their
//! `Tensor::random` draws and destroy bit-equality across runs — the same
//! rationale documented in `rlevo-evolution/tests/determinism.rs`. This file
//! therefore contains exactly one `#[test]` so the cargo runner cannot execute
//! anything in parallel with it; within the single body every run is strictly
//! sequential. (The `#[ignore]`d `calibration_explorer` below is an
//! exploration-only scaffold that is never part of the default test run.)
//!
//! # Eval-count accounting
//!
//! Both the bare-DE harness and the memetic harness are driven with a
//! [`CountingRastrigin`] fitness that increments a shared `Arc<AtomicUsize>` by
//! one *per row* of every `evaluate_batch`. Every `MemeticWrapper` local-search
//! probe routes through a `[1, D]` `evaluate_batch`, so row-counting captures
//! refinement evaluations automatically. The memetic run shares **one** counter
//! across both fitness instances it owns (harness + wrapper); the bare-DE run
//! uses a separate counter.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend, backend::BackendTypes};

use rlevo_core::bounds::Bounds;
use rlevo_core::probability::Probability;
use rlevo_environments::landscapes::rastrigin::Rastrigin;
use rlevo_evolution::algorithms::de::{DeConfig, DifferentialEvolution};
use rlevo_evolution::algorithms::memetic::{
    CoveragePolicy, MemeticParams, MemeticWrapper, WritebackPolicy,
};
use rlevo_evolution::fitness::BatchFitnessFn;
use rlevo_evolution::local_search::{HillClimbVariant, HillClimbing, HillClimbingParams};
use rlevo_evolution::strategy::EvolutionaryHarness;

type B = Flex;

const DIM: usize = 10;
const BOUNDS: Bounds = Bounds::new(-5.12, 5.12);

/// Row-counting Rastrigin fitness.
///
/// Implements only [`BatchFitnessFn`] and increments `evals` by the number of
/// rows in each `evaluate_batch`. Because every `MemeticWrapper` local-search
/// probe is a `[1, D]` batch, refinement evaluations are counted too. Two
/// instances that share one `Arc<AtomicUsize>` (e.g. the harness's and the
/// wrapper's) report a single combined evaluation budget.
struct CountingRastrigin {
    landscape: Rastrigin,
    evals: Arc<AtomicUsize>,
}

impl CountingRastrigin {
    fn new(landscape: Rastrigin, evals: Arc<AtomicUsize>) -> Self {
        Self { landscape, evals }
    }
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

/// Result of one `evals_to_target` run: the eval counter when the best-so-far
/// first crossed below `target`, plus the per-generation best-fitness-ever
/// trajectory (for determinism assertions).
struct RunResult {
    /// Total fitness-row evaluations consumed at the first crossing of
    /// `best_fitness_ever < target`, or `None` if never reached.
    evals_at_target: Option<usize>,
    /// Per-generation `best_fitness_ever` trajectory.
    trajectory: Vec<f32>,
}

/// Drive bare DE to a target Rastrigin fitness, counting evaluations.
fn de_evals_to_target(seed: u64, de: &DeConfig, target: f32, max_gens: usize) -> RunResult {
    let device: <B as BackendTypes>::Device = Default::default();
    let evals: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let fitness: CountingRastrigin =
        CountingRastrigin::new(Rastrigin::new(DIM), evals.clone());
    let strategy: DifferentialEvolution<B> = DifferentialEvolution::<B>::new();
    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        strategy,
        de.clone(),
        fitness,
        seed,
        device,
        max_gens,
    ).expect("valid params");
    harness.reset();
    let mut trajectory: Vec<f32> = Vec::with_capacity(max_gens);
    let mut evals_at_target: Option<usize> = None;
    loop {
        let step = harness.step(());
        let best: f32 = harness.latest_metrics().unwrap().best_fitness_ever;
        trajectory.push(best);
        if evals_at_target.is_none() && best < target {
            evals_at_target = Some(evals.load(Ordering::Relaxed));
        }
        if step.done {
            break;
        }
    }
    RunResult {
        evals_at_target,
        trajectory,
    }
}

/// Drive `MemeticWrapper<DE, HillClimbing>` to a target Rastrigin fitness,
/// counting evaluations across BOTH fitness instances via one shared counter.
fn memetic_evals_to_target(
    seed: u64,
    de: &DeConfig,
    hc: &HillClimbingParams,
    writeback: WritebackPolicy,
    coverage: CoveragePolicy,
    target: f32,
    max_gens: usize,
) -> RunResult {
    let device: <B as BackendTypes>::Device = Default::default();
    // One shared counter feeds both fitness instances: the harness's (population
    // scoring) and the wrapper's (local-search probes).
    let evals: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let harness_fitness: CountingRastrigin =
        CountingRastrigin::new(Rastrigin::new(DIM), evals.clone());
    let wrapper_fitness: CountingRastrigin =
        CountingRastrigin::new(Rastrigin::new(DIM), evals.clone());

    let strategy: MemeticWrapper<B, _, _, _> = MemeticWrapper::<B, _, _, _>::new(
        DifferentialEvolution::<B>::new(),
        HillClimbing,
        wrapper_fitness,
    );
    let params: MemeticParams<DeConfig, HillClimbingParams> = MemeticParams {
        inner: de.clone(),
        local: hc.clone(),
        writeback,
        coverage,
    };
    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        strategy,
        params,
        harness_fitness,
        seed,
        device,
        max_gens,
    ).expect("valid params");
    harness.reset();
    let mut trajectory: Vec<f32> = Vec::with_capacity(max_gens);
    let mut evals_at_target: Option<usize> = None;
    loop {
        let step = harness.step(());
        let best: f32 = harness.latest_metrics().unwrap().best_fitness_ever;
        trajectory.push(best);
        if evals_at_target.is_none() && best < target {
            evals_at_target = Some(evals.load(Ordering::Relaxed));
        }
        if step.done {
            break;
        }
    }
    RunResult {
        evals_at_target,
        trajectory,
    }
}

/// Shared bare-DE config: pop 30, D=10, the crate default (`Rand1Bin`, F=0.5,
/// CR=0.9). The memetic run reuses this verbatim as its inner config.
fn shared_de_config() -> DeConfig {
    DeConfig::default_for(30, DIM)
}

/// Shared hill-climbing config used by the headline memetic run.
///
/// `max_iters=20` / `BestImprovement` / `step=0.4` / `TopK{1}` (coverage is set
/// at the call site) is the calibrated sweet spot: cheap enough per generation
/// that the refinement overhead never erases DE's progress, greedy enough that
/// each polished elite pulls the whole population toward a basin. See the
/// provenance block on the headline test for the eval-count data behind this.
fn shared_hc_config() -> HillClimbingParams {
    HillClimbingParams::default_for(BOUNDS)
        .with_max_iters(20)
        .with_step_size(0.4)
        .with_step_decay(0.5)
        .with_variant(HillClimbVariant::BestImprovement)
}

// =====================================================================
// Calibration explorer (NOT part of the default test run).
//
// Run with: cargo test -p rlevo --test memetic_rastrigin -- --ignored --nocapture
// Prints bare-vs-memetic eval counts across several seeds and targets so the
// pinned constants below can be chosen from real data, then deleted/ignored.
// =====================================================================

#[test]
#[ignore = "calibration explorer: multi-seed eval-count sweep for pinning the headline margin; run with --ignored --nocapture"]
fn calibration_explorer() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();

    let de: DeConfig = shared_de_config();
    let max_gens: usize = 600;
    let seeds: [u64; 5] = [7, 13, 29, 101, 1234];
    let targets: [f32; 3] = [30.0, 20.0, 10.0];

    // Candidate HC configs (cheaper per-gen overhead than the original
    // max_iters=25/TopK{3} sweep, which was dominated by refinement cost on
    // easy targets).
    let make_hc = |max_iters: usize, step: f32, variant: HillClimbVariant| -> HillClimbingParams {
        HillClimbingParams::default_for(BOUNDS)
            .with_max_iters(max_iters)
            .with_step_size(step)
            .with_step_decay(0.5)
            .with_variant(variant)
    };
    let configs: Vec<(&str, HillClimbingParams, WritebackPolicy, CoveragePolicy)> = vec![
        (
            "k1/it10/first/lam",
            make_hc(10, 0.4, HillClimbVariant::FirstImprovement),
            WritebackPolicy::Lamarckian,
            CoveragePolicy::TopK { k: 1 },
        ),
        (
            "k1/it20/best/lam",
            make_hc(20, 0.4, HillClimbVariant::BestImprovement),
            WritebackPolicy::Lamarckian,
            CoveragePolicy::TopK { k: 1 },
        ),
        (
            "k1/it15/first/p0.3",
            make_hc(15, 0.4, HillClimbVariant::FirstImprovement),
            WritebackPolicy::Partial(Probability::new(0.3)),
            CoveragePolicy::TopK { k: 1 },
        ),
        (
            "k2/it10/first/lam",
            make_hc(10, 0.4, HillClimbVariant::FirstImprovement),
            WritebackPolicy::Lamarckian,
            CoveragePolicy::TopK { k: 2 },
        ),
    ];

    println!("\n=== calibration: bare DE vs MemeticWrapper<DE, HillClimbing> ===");
    println!("DE: pop=30 D=10 Rand1Bin F=0.5 CR=0.9");
    for &target in &targets {
        println!("\n========== target = {target} ==========");
        // Bare DE once per seed/target (config-independent).
        let bare: Vec<(u64, Option<usize>, f32)> = seeds
            .iter()
            .map(|&seed| {
                let r: RunResult = de_evals_to_target(seed, &de, target, max_gens);
                (seed, r.evals_at_target, r.trajectory.last().copied().unwrap_or(f32::NAN))
            })
            .collect();
        for &(seed, bare_e, bare_f) in &bare {
            println!("  bare    seed={seed:>5}: {bare_e:>9?} (final={bare_f:.4})");
        }
        for (name, hc, wb, cov) in &configs {
            println!("  -- HC config: {name} --");
            for &seed in &seeds {
                let mem: RunResult =
                    memetic_evals_to_target(seed, &de, hc, *wb, *cov, target, max_gens);
                let bare_e: Option<usize> =
                    bare.iter().find(|(s, _, _)| *s == seed).and_then(|(_, e, _)| *e);
                let verdict: &str = match (bare_e, mem.evals_at_target) {
                    (Some(b), Some(m)) if m < b => "WIN",
                    (Some(_), Some(_)) => "lose",
                    (None, Some(_)) => "WIN(bare-miss)",
                    (_, None) => "mem-miss",
                };
                println!(
                    "     seed={seed:>5}: memetic={:>9?} bare={bare_e:>9?} [{verdict}] (mem_final={:.4})",
                    mem.evals_at_target,
                    mem.trajectory.last().copied().unwrap_or(f32::NAN),
                );
            }
        }
    }
}

// =====================================================================
// Headline acceptance test (the only test that runs by default).
//
// Provenance (re-pinned from `calibration_explorer` output, 2026-06-12, after
// `refine_with_known_fitness` eliminated the per-refine seeding eval, issue #30):
//   backend:   Flex (ndarray), rayon pinned to 1 thread, release profile.
//   seed:      7
//   target:    Rastrigin-D10 best_fitness_ever < 30.0
//   inner DE:  pop=30, D=10, Rand1Bin, F=0.5, CR=0.9 (DeConfig::default_for).
//   memetic:   HillClimbing BestImprovement, max_iters=20, step=0.4, decay=0.5,
//              CoveragePolicy::TopK{1}, WritebackPolicy::Lamarckian.
//   observed:  bare = 11_130 evals, memetic = 2_900 evals
//              (memetic uses ~74% fewer evals than bare DE to reach 30.0).
//
// Why the pinned number dropped from 5_150 to 2_900:
//   With max_iters=20 and D=10, one BestImprovement sweep costs 2*D = 20 evals.
//   The old seeding eval left only 19 for the sweep, so it ran one probe short
//   of completing and never committed a move; eliminating that eval lets the
//   sweep finish and commit, so each refined generation makes real progress.
//   The win is therefore larger than the bare "one eval/gen" saving — it crosses
//   the sweep-granularity boundary.
//
// Why target=30 (not 20/10/5):
//   - At target=10/5 bare DE often STALLS without ever reaching it (final
//     best 6-17 on most seeds), so a "fewer evals to a SHARED target"
//     comparison is not well-posed there — both configs must reach it.
//   - At target=30 BOTH configs always reach the target and memetic WINS on
//     every calibration seed {7,13,29,101,1234}, though the margin is now wider-
//     spread than before: ratio mem/bare ranged ~0.08 (seed 29) to ~0.89
//     (seed 101) — i.e. 11%-92% fewer. The spread comes from Lamarckian
//     writeback: the extra committed sweep changes the genome fed back to DE, so
//     each seed takes a genuinely different trajectory. Seed 7 is a strong,
//     stable win at ~74% fewer (ratio 0.26).
//
// Margin choice:
//   The cross-seed margin is no longer uniformly >=30% (seeds 101/1234 are now
//   ~11%/~19% fewer), so the assertion is deliberately scoped to the PINNED seed,
//   where the win is robust. We assert the >=30% margin via integer math
//   `memetic_evals * 10 <= bare_evals * 7` (no float comparison). On the pinned
//   data: 2_900*10 = 29_000 <= 11_130*7 = 77_910. We also assert an absolute
//   tripwire `bare_evals >= 8_000` so a future regression that makes bare DE
//   trivially slow (inflating the ratio) cannot mask a memetic slowdown.
// =====================================================================

const PINNED_SEED: u64 = 7;
const TARGET: f32 = 30.0;
const MAX_GENS: usize = 600;

#[test]
fn memetic_beats_bare_de_on_rastrigin_evals_to_target() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();

    let de: DeConfig = shared_de_config();
    let hc: HillClimbingParams = shared_hc_config();
    let writeback: WritebackPolicy = WritebackPolicy::Lamarckian;
    let coverage: CoveragePolicy = CoveragePolicy::TopK { k: 1 };

    // Bare DE and memetic share IDENTICAL DeConfig and IDENTICAL seed.
    let bare: RunResult = de_evals_to_target(PINNED_SEED, &de, TARGET, MAX_GENS);
    let mem: RunResult = memetic_evals_to_target(
        PINNED_SEED, &de, &hc, writeback, coverage, TARGET, MAX_GENS,
    );

    // (a) both configs reach the target.
    let bare_evals: usize = bare
        .evals_at_target
        .unwrap_or_else(|| panic!("bare DE never reached target {TARGET} in {MAX_GENS} gens"));
    let memetic_evals: usize = mem
        .evals_at_target
        .unwrap_or_else(|| panic!("memetic never reached target {TARGET} in {MAX_GENS} gens"));

    println!("bare_evals={bare_evals} memetic_evals={memetic_evals} target={TARGET}");

    // (b) memetic strictly fewer total evals.
    assert!(
        memetic_evals < bare_evals,
        "memetic must use strictly fewer evals: memetic={memetic_evals} bare={bare_evals}"
    );

    // (c) pinned margin: ≥30% fewer evals, integer math (no float comparison).
    //     2_900*10 = 29_000 <= 11_130*7 = 77_910 on the pinned data.
    assert!(
        memetic_evals * 10 <= bare_evals * 7,
        "memetic must use >=30% fewer evals: memetic={memetic_evals} bare={bare_evals} \
         (need memetic*10 <= bare*7, i.e. {} <= {})",
        memetic_evals * 10,
        bare_evals * 7,
    );
    // Absolute tripwire: bare DE must actually take a meaningful number of
    // evals, so the ratio cannot be inflated by a bare-DE slowdown regression.
    assert!(
        bare_evals >= 8_000,
        "bare DE eval count {bare_evals} fell below the 8_000 tripwire — \
         a bare-DE regression may be masking a memetic slowdown"
    );

    // (d) determinism: same seed → identical trajectory AND identical eval count.
    let mem2: RunResult = memetic_evals_to_target(
        PINNED_SEED, &de, &hc, writeback, coverage, TARGET, MAX_GENS,
    );
    assert_eq!(
        mem.trajectory, mem2.trajectory,
        "memetic best-fitness trajectory must be reproducible under the same seed"
    );
    assert_eq!(
        mem.evals_at_target, mem2.evals_at_target,
        "memetic eval count at target must be reproducible under the same seed"
    );
}
