//! EDA showcase: estimation-of-distribution algorithms *learn a model* of the
//! search space instead of recombining individuals.
//!
//! A classical GA produces the next generation by crossing over and mutating
//! parents. An EDA throws that away: each generation it (1) truncation-selects
//! the best individuals, (2) **fits a probability model** to them, and (3)
//! **samples** a fresh population from that model. The model *is* the algorithm,
//! so this example prints the model's internals as they evolve rather than
//! hiding them behind the [`EvolutionaryHarness`]. It drives the
//! [`Strategy`] `ask` → evaluate → `tell` loop by hand for exactly that reason.
//!
//! Three demos, each paired with the problem that actually exercises its models:
//!
//! ## Demo 1 — continuous, Rosenbrock-D10: why dependencies matter
//!
//! Compares [`UnivariateGaussian`] (UMDA, models each dimension independently)
//! against [`DependencyChain`] (MIMIC, models a first-order chain of pairwise
//! dependencies). Rosenbrock's `x_{i+1} ≈ x_i²` ridge *couples* adjacent
//! variables — structure a univariate model literally cannot represent. Watch
//! MIMIC's `link_corr` vector light up as it discovers the coupling, then beat
//! UMDA's final fitness. (The calibrated prior — `init_mean = 0.5`,
//! `min_variance = 1e-3`, `selection_ratio = 0.3`, de Jong bounds ±2.048 — puts
//! the population on the valley limb where the coupling is locally linear;
//! centred at the origin the ridge's tangent is flat and there is nothing to
//! model. See `crates/rlevo/tests/eda_convergence.rs` for the full rationale.)
//!
//! ## Demo 2 — binary, `OneMax`-D20: the probability-vector mechanic
//!
//! Runs [`UnivariateBernoulli`] (PBIL) and [`CompactGenetic`] (cGA) on `OneMax`
//! (maximise the number of 1-bits). Both emit raw `{0, 1}` genes and carry a
//! per-gene probability vector; watch it march to all-ones. `OneMax` is fully
//! *separable*, so a univariate model is already optimal here — the binary
//! analogue of "dependencies matter" needs a deceptive problem **and** a
//! multivariate binary model (BOA) — see Demo 3.
//!
//! ## Demo 3 — binary deception, `ConcatenatedTrap` trap-5 × 4: linkage matters
//!
//! Runs [`UnivariateGaussian`] (UMDA), [`DependencyChain`] (MIMIC), and
//! [`BayesianNetwork`] (BOA) on a deceptive order-5 trap of four blocks
//! (dim 20). Each block costs `cost(u) = 0` if `u == k` else `u + 1`, where `u`
//! is the block's unitation (1-bit count) and `k = 5` — so the all-ones optimum
//! costs `0` but the all-zeros deceptive basin costs `num_blocks = 4`. In a
//! random population, blocks with *fewer* ones are cheaper on average, so every
//! per-gene average marches the model toward all-zeros: UMDA's means collapse,
//! and MIMIC's first-order chain — which can tie at most one neighbour per gene
//! — still cannot capture the order-5 intra-block linkage. Only BOA, learning a
//! bounded-in-degree DAG, discovers the within-block edges and samples whole
//! solved blocks; watch its intra-block edge count climb while UMDA/MIMIC are
//! deceived. Verdict: BOA median cost `0.0`, UMDA/MIMIC medians ≈ `3`.
//!
//! Demo 3 deliberately omits the *incremental* binary models. At this large
//! population the damped PBIL/cGA updates resist the deceptive average gradient
//! and partially escape the trap (cGA solved 9/10 calibration seeds), so they
//! muddy the structural story. The three full-refit models compared here make
//! the linkage punchline clean.
//!
//! # Running
//!
//! ```text
//! cargo run --release -p rlevo-examples --example eda_showcase
//! ```
//!
//! No feature flags are required. Release is strongly recommended — the run
//! drives a few thousand generations of Burn tensor ops, and Demo 3 alone adds
//! ~11 runs at population 2000 × 60 generations.

use burn::backend::Flex;
use burn::tensor::backend::BackendTypes;
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::bounds::Bounds;
use rlevo_environments::landscapes::concatenated_trap::ConcatenatedTrap;
use rlevo_environments::landscapes::rosenbrock::Rosenbrock;
use rlevo_evolution::algorithms::eda::{
    BayesianNetworkState, CompactGeneticState, DependencyChainState, UnivariateBernoulliState,
    UnivariateGaussianState,
};
use rlevo_evolution::{
    BayesianNetwork, BayesianNetworkParams, CompactGenetic, CompactGeneticParams, DependencyChain,
    DependencyChainParams, EdaParams, EdaStrategy, ProbabilityModel, Strategy, UnivariateBernoulli,
    UnivariateBernoulliParams, UnivariateGaussian, UnivariateGaussianParams,
};

type B = Flex;

/// Rosenbrock under the EDA minimization convention. The host genome row is
/// `f32`; widen to `f64` for the landscape, then narrow the scalar back.
#[allow(clippy::cast_possible_truncation)]
fn rosenbrock_fitness(landscape: Rosenbrock, row: &[f32]) -> f32 {
    let point: Vec<f64> = row.iter().map(|&g| f64::from(g)).collect();
    landscape.evaluate(&point) as f32
}

/// Evaluate every row of a `(pop_size, D)` population tensor with a host-side
/// scalar fitness closure and return the `(pop_size,)` fitness tensor.
///
/// This mirrors what [`rlevo_evolution::fitness::FromLandscape`] does inside
/// the harness; we inline it so the binary `OneMax` demo needs no `Landscape`
/// impl and both demos read identically.
#[allow(clippy::trivially_copy_pass_by_ref)] // device is a ZST; &-passing reads clearer
fn eval_population<F>(
    pop: &Tensor<B, 2>,
    device: &<B as BackendTypes>::Device,
    mut fitness: F,
) -> Tensor<B, 1>
where
    F: FnMut(&[f32]) -> f32,
{
    let [n, d] = pop.dims();
    let flat = pop.clone().into_data().into_vec::<f32>().unwrap();
    let values: Vec<f32> = flat.chunks(d).map(&mut fitness).collect();
    Tensor::<B, 1>::from_data(TensorData::new(values, [n]), device)
}

/// Drive an EDA model for `gens` generations via the bare `Strategy` loop,
/// calling `report(generation, &model_state)` after every `tell` so the caller
/// can print the freshly-refitted model. Returns the best fitness ever seen.
fn run_eda<M, F, R>(
    model: M,
    params: &EdaParams<M::Params>,
    seed: u64,
    gens: usize,
    mut fitness: F,
    mut report: R,
) -> f32
where
    M: ProbabilityModel<B> + 'static,
    F: FnMut(&[f32]) -> f32,
    R: FnMut(usize, &M::State),
{
    let device: <B as BackendTypes>::Device = Default::default();
    let strategy = EdaStrategy::<B, M>::new(model);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = strategy.init(params, &mut rng, &device);

    for generation in 0..gens {
        let (population, carried) = strategy.ask(params, &state, &mut rng, &device);
        let fit = eval_population(&population, &device, &mut fitness);
        let (next, _metrics) = strategy.tell(params, population, fit, carried, &mut rng);
        state = next;
        report(generation, &state.model_state);
    }

    strategy.best(&state).map_or(f32::INFINITY, |(_, f)| f)
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = xs.len() as f32;
    xs.iter().sum::<f32>() / n
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

/// Log schedule for the verbose single-seed traces: print at generations
/// 1, 2, 5, 10, 25, 50, 100, 250, 500. A converging EDA does most of its
/// model-shaping early — and MIMIC only captures significant correlations
/// while the selected set is still spread out — so the early generations
/// are exactly the ones worth seeing. Uniform spacing buries them.
fn logged(generation: usize) -> bool {
    matches!(generation + 1, 1 | 2 | 5 | 10 | 25 | 50 | 100 | 250 | 500)
}

// ── Demo 1 — continuous: UMDA vs MIMIC on Rosenbrock ──────────────────────────

const ROSEN_DIM: usize = 10;
const ROSEN_POP: usize = 50;
const ROSEN_GENS: usize = 500;
const ROSEN_RATIO: f32 = 0.3;
const ROSEN_INIT_MEAN: f32 = 0.5;
const ROSEN_INIT_STD: f32 = 0.5;
const ROSEN_MIN_VAR: f32 = 1e-3;
const ROSEN_BOUNDS: Option<Bounds> = Some(Bounds::new(-2.048, 2.048));
const ROSEN_SEEDS: [u64; 9] = [101, 202, 303, 404, 505, 606, 707, 808, 909];

fn umda_params() -> UnivariateGaussianParams {
    let mut p = UnivariateGaussianParams::default_for(ROSEN_DIM);
    p.init_mean = ROSEN_INIT_MEAN;
    p.init_std = ROSEN_INIT_STD;
    p.min_variance = ROSEN_MIN_VAR;
    p
}

fn mimic_params() -> DependencyChainParams {
    let mut p = DependencyChainParams::default_for(ROSEN_DIM);
    p.init_mean = ROSEN_INIT_MEAN;
    p.init_std = ROSEN_INIT_STD;
    p.min_variance = ROSEN_MIN_VAR;
    p
}

fn eda_params<MP>(model: MP) -> EdaParams<MP> {
    EdaParams {
        pop_size: ROSEN_POP,
        selection_ratio: ROSEN_RATIO,
        bounds: ROSEN_BOUNDS,
        model,
    }
}

fn rosenbrock_demo() {
    println!("══ Demo 1: UMDA vs MIMIC on Rosenbrock-D{ROSEN_DIM} ══");
    println!(
        "Rosenbrock couples adjacent variables (x_{{i+1}} ≈ x_i²). UMDA models each\n\
         dimension independently; MIMIC fits a dependency chain. Watch MIMIC's\n\
         strongest captured link correlation grow while UMDA has no such notion.\n"
    );

    let landscape = Rosenbrock::new(ROSEN_DIM);
    let trace_seed = ROSEN_SEEDS[0];

    // Verbose single-seed run: UMDA. Print mean position magnitude + mean σ.
    println!("UMDA (seed {trace_seed}) — independent per-dimension Gaussian:");
    println!("  {:>4} {:>12} {:>10}", "gen", "mean|μ|", "mean σ");
    let umda_best = run_eda(
        UnivariateGaussian,
        &eda_params(umda_params()),
        trace_seed,
        ROSEN_GENS,
        |row| rosenbrock_fitness(landscape, row),
        |g, st: &UnivariateGaussianState| {
            if logged(g) {
                let mean_abs_mu = mean(&st.mean.iter().map(|m| m.abs()).collect::<Vec<_>>());
                let mean_sigma = mean(&st.variance.iter().map(|v| v.sqrt()).collect::<Vec<_>>());
                println!("  {:>4} {mean_abs_mu:>12.5} {mean_sigma:>10.5}", g + 1);
            }
        },
    );
    println!("  → best fitness: {umda_best:.5}\n");

    // Verbose single-seed run: MIMIC. Print the strongest surviving link per
    // snapshot, and track the peak captured dependency across the whole run —
    // the chain regularizes its links to zero once the elite set tightens, so
    // the snapshot value alone understates how much structure it exploited.
    println!("MIMIC (seed {trace_seed}) — first-order dependency chain:");
    println!(
        "  {:>4} {:>12} {:>10} {:>10}  strongest surviving link",
        "gen", "mean|μ|", "mean σ", "max|r|"
    );
    let mut peak_corr = 0.0_f32;
    let mut peak_link: (usize, usize) = (0, 0);
    let mut link_gens = 0_usize;
    let mimic_best = run_eda(
        DependencyChain,
        &eda_params(mimic_params()),
        trace_seed,
        ROSEN_GENS,
        |row| rosenbrock_fitness(landscape, row),
        |g, st: &DependencyChainState| {
            // Strongest captured dependency this generation: the chain position
            // whose link correlation has the largest magnitude, and the (parent
            // → child) dims it ties. Position 0 is the root (no parent).
            let (pos, corr) = st
                .link_corr
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .map_or((0, 0.0), |(p, &c)| (p, c.abs()));
            let link = if pos >= 1 && st.chain.len() >= 2 {
                Some((st.chain[pos - 1], st.chain[pos]))
            } else {
                None
            };
            // Peak tracking runs every generation, not just on logged ones.
            if corr > 0.0 {
                link_gens += 1;
            }
            if corr > peak_corr {
                peak_corr = corr;
                if let Some(l) = link {
                    peak_link = l;
                }
            }
            if logged(g) {
                let mean_abs_mu = mean(&st.mean.iter().map(|m| m.abs()).collect::<Vec<_>>());
                let mean_sigma = mean(&st.std);
                let label = link.map_or_else(
                    || "(all filtered out)".to_string(),
                    |(parent, child)| format!("dim {parent} → dim {child}"),
                );
                println!("  {:>4} {mean_abs_mu:>12.5} {mean_sigma:>10.5} {corr:>10.4}  {label}", g + 1);
            }
        },
    );
    println!(
        "  peak |r| = {peak_corr:.3} (dim {} → dim {}); {link_gens}/{ROSEN_GENS} \
         generations captured ≥1 significant link",
        peak_link.0, peak_link.1
    );
    println!("  → best fitness: {mimic_best:.5}\n");

    rosenbrock_medians(landscape);
}

/// Robustness check: median final fitness over all nine seeds (the
/// acceptance-gate configuration), so the MIMIC win reads as structural rather
/// than a single lucky seed.
fn rosenbrock_medians(landscape: Rosenbrock) {
    let mut umda: Vec<f32> = ROSEN_SEEDS
        .iter()
        .map(|&seed| {
            run_eda(
                UnivariateGaussian,
                &eda_params(umda_params()),
                seed,
                ROSEN_GENS,
                |row| rosenbrock_fitness(landscape, row),
                |_gen, _st: &UnivariateGaussianState| {},
            )
        })
        .collect();
    let mut mimic: Vec<f32> = ROSEN_SEEDS
        .iter()
        .map(|&seed| {
            run_eda(
                DependencyChain,
                &eda_params(mimic_params()),
                seed,
                ROSEN_GENS,
                |row| rosenbrock_fitness(landscape, row),
                |_gen, _st: &DependencyChainState| {},
            )
        })
        .collect();
    let umda_med = median(&mut umda);
    let mimic_med = median(&mut mimic);
    println!("Median final fitness over {} seeds:", ROSEN_SEEDS.len());
    println!("  UMDA  : {umda_med:.4}");
    println!("  MIMIC : {mimic_med:.4}");
    let verdict = if mimic_med < umda_med {
        "MIMIC wins — modelling the coupling pays off"
    } else {
        "UMDA matched MIMIC this run"
    };
    println!("  → {verdict}\n");
}

// ── Demo 2 — binary: PBIL vs cGA on OneMax ────────────────────────────────────

const ONEMAX_DIM: usize = 20;
const ONEMAX_POP: usize = 50;
const ONEMAX_GENS: usize = 120;
const ONEMAX_RATIO: f32 = 0.5;
const ONEMAX_SEED: u64 = 7;

/// `OneMax` under the minimization convention: fitness is the number of `0`
/// genes, so the all-ones optimum scores `0.0`. Genes arrive as raw `{0, 1}`
/// `f32`; treat `>= 0.5` as a 1-bit.
fn onemax_zeros(row: &[f32]) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let zeros = row.iter().filter(|&&g| g < 0.5).count() as f32;
    zeros
}

/// Render a probability vector as one glyph per gene (`0`–`9` ≈ ⌊10·p⌋, `#` for
/// p ≈ 1) so convergence reads without relying on colour.
fn prob_bar(prob: &[f32]) -> String {
    prob.iter()
        .map(|&p| {
            if p >= 0.99 {
                '#'
            } else {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let d = (p * 10.0) as u8;
                char::from(b'0' + d.min(9))
            }
        })
        .collect()
}

fn onemax_demo() {
    println!("══ Demo 2: PBIL vs cGA on OneMax-D{ONEMAX_DIM} ══");
    println!(
        "Both keep a per-gene probability vector and sample raw {{0,1}} genes — no\n\
         crossover, no mutation. Watch the vector march to all-ones ('#'). OneMax is\n\
         separable, so a univariate model is already optimal; the deceptive binary\n\
         case that needs a multivariate model (BOA) is deferred to issue #37.\n"
    );

    println!("PBIL (seed {ONEMAX_SEED}):");
    println!("  {:>4} {:>9}  probability vector (0–9, '#'≈1)", "gen", "mean p");
    let pbil_best = run_eda(
        UnivariateBernoulli,
        &EdaParams {
            pop_size: ONEMAX_POP,
            selection_ratio: ONEMAX_RATIO,
            bounds: None,
            model: UnivariateBernoulliParams::default_for(ONEMAX_DIM),
        },
        ONEMAX_SEED,
        ONEMAX_GENS,
        onemax_zeros,
        |g, st: &UnivariateBernoulliState| {
            println!("  {:>4} {:>9.4}  {}", g + 1, mean(&st.prob), prob_bar(&st.prob));
        },
    );
    println!("  → best fitness (0 = all-ones found): {pbil_best:.1}\n");

    println!("cGA (seed {ONEMAX_SEED}):");
    println!("  {:>4} {:>9}  probability vector (0–9, '#'≈1)", "gen", "mean p");
    let cga_best = run_eda(
        CompactGenetic,
        &EdaParams {
            pop_size: ONEMAX_POP,
            selection_ratio: ONEMAX_RATIO,
            bounds: None,
            model: CompactGeneticParams::default_for(ONEMAX_DIM),
        },
        ONEMAX_SEED,
        ONEMAX_GENS,
        onemax_zeros,
        |g, st: &CompactGeneticState| {
            println!("  {:>4} {:>9.4}  {}", g + 1, mean(&st.prob), prob_bar(&st.prob));
        },
    );
    println!("  → best fitness (0 = all-ones found): {cga_best:.1}\n");
}

// ── Demo 3 — binary deception: UMDA vs MIMIC vs BOA on ConcatenatedTrap ────────

/// Number of order-`TRAP_K` trap blocks; total dim is `TRAP_BLOCKS · TRAP_K`.
const TRAP_BLOCKS: usize = 4;
/// Trap order `k`: bits per block. Reaching the per-block optimum requires
/// flipping all `k` bits together — the order-`k` linkage UMDA/MIMIC miss.
const TRAP_K: usize = 5;
/// Genome dimension: `TRAP_BLOCKS · TRAP_K = 20`.
const TRAP_DIM: usize = TRAP_BLOCKS * TRAP_K;
/// Selected-row count is load-bearing: the BIC edge gain grows with `N` while
/// the complexity penalty grows like `ln N`, so the intra-block edges clear the
/// penalty only when thousands of rows are selected (ADR 0018 calibration).
const TRAP_POP: usize = 2000;
/// Truncation ratio; `0.3` enriches solved-block carriers fast enough that BOA
/// learns the structure before the deceptive per-gene gradient collapses.
const TRAP_RATIO: f32 = 0.3;
/// Generation budget for the trap (short by design — the structure, if any, is
/// learned early or never).
const TRAP_GENS: usize = 60;
/// Verbose single-seed trace seed for Demo 3.
const TRAP_TRACE_SEED: u64 = 11;
/// Median seeds for Demo 3 (the ADR 0018 convergence-gate set).
const TRAP_SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
/// Symmetric prior over the binary box `[0, 1]` shared by the continuous models.
const TRAP_INIT_MEAN: f32 = 0.5;
const TRAP_INIT_STD: f32 = 0.5;

/// `ConcatenatedTrap` under the EDA minimization convention. The host genome row
/// is `f32`; widen to `f64` for the landscape, then narrow the scalar back.
#[allow(clippy::cast_possible_truncation)]
fn trap_fitness(trap: ConcatenatedTrap, row: &[f32]) -> f32 {
    let point: Vec<f64> = row.iter().map(|&g| f64::from(g)).collect();
    trap.evaluate(&point) as f32
}

/// UMDA prior for the trap: symmetric over the binary box.
fn trap_umda_params() -> UnivariateGaussianParams {
    let mut p = UnivariateGaussianParams::default_for(TRAP_DIM);
    p.init_mean = TRAP_INIT_MEAN;
    p.init_std = TRAP_INIT_STD;
    p
}

/// MIMIC prior for the trap: same symmetric box prior as UMDA.
fn trap_mimic_params() -> DependencyChainParams {
    let mut p = DependencyChainParams::default_for(TRAP_DIM);
    p.init_mean = TRAP_INIT_MEAN;
    p.init_std = TRAP_INIT_STD;
    p
}

/// `EdaParams` for the continuous trap models (UMDA, MIMIC): clamp to `[0, 1]`.
fn trap_continuous_params<MP>(model: MP) -> EdaParams<MP> {
    EdaParams {
        pop_size: TRAP_POP,
        selection_ratio: TRAP_RATIO,
        bounds: Some(Bounds::new(0.0, 1.0)),
        model,
    }
}

/// `EdaParams` for BOA: emits raw `{0, 1}` genes, so bounds are a no-op.
fn trap_boa_params() -> EdaParams<BayesianNetworkParams> {
    EdaParams {
        pop_size: TRAP_POP,
        selection_ratio: TRAP_RATIO,
        bounds: None,
        model: BayesianNetworkParams::default_for(TRAP_DIM),
    }
}

/// Is the directed edge `parent → child` inside a single trap block? Both
/// endpoints share a block iff they map to the same `TRAP_K`-wide bucket.
fn intra_block(parent: usize, child: usize) -> bool {
    parent / TRAP_K == child / TRAP_K
}

/// Count total edges and intra-block edges in a fitted Bayesian network, plus a
/// per-block tally of intra-block edges (`blocks[b]` = intra edges in block `b`).
fn edge_stats(state: &BayesianNetworkState) -> (usize, usize, [usize; TRAP_BLOCKS]) {
    let mut total = 0usize;
    let mut intra = 0usize;
    let mut blocks = [0usize; TRAP_BLOCKS];
    for (child, parents) in state.parents.iter().enumerate() {
        for &parent in parents {
            total += 1;
            if intra_block(parent, child) {
                intra += 1;
                blocks[parent / TRAP_K] += 1;
            }
        }
    }
    (total, intra, blocks)
}

/// Render the per-block intra-edge tally as `[b0: 3 intra-edges] [b1: 4] …`,
/// spelling "intra-edges" only on the first block to keep the line compact.
fn block_map(blocks: &[usize; TRAP_BLOCKS]) -> String {
    blocks
        .iter()
        .enumerate()
        .map(|(b, &count)| {
            if b == 0 {
                format!("[b0: {count} intra-edges]")
            } else {
                format!("[b{b}: {count}]")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn trap_demo() {
    println!(
        "══ Demo 3: UMDA vs MIMIC vs BOA on ConcatenatedTrap trap-{TRAP_K} × {TRAP_BLOCKS} \
         (dim {TRAP_DIM}) ══"
    );
    println!(
        "Each block costs cost(u) = 0 if u == k else u + 1, with k = {TRAP_K} bits per\n\
         block and u the block's 1-count. The all-ones optimum costs 0; the all-zeros\n\
         basin costs {TRAP_BLOCKS}. In a random population, blocks with FEWER ones are\n\
         cheaper on average, so per-gene statistics march everything to all-zeros\n\
         (cost {TRAP_BLOCKS}). The all-ones optimum (cost 0) is reachable only by a model\n\
         that captures the order-{TRAP_K} intra-block linkage. Note: the incremental\n\
         binary models (PBIL/cGA) partially escape this trap at pop {TRAP_POP} — their\n\
         damped updates resist the deceptive gradient — so this demo compares the\n\
         full-refit models, where the structural story is clean.\n"
    );

    let trap = ConcatenatedTrap::new(TRAP_BLOCKS, TRAP_K);

    // UMDA trace: per-gene Gaussian means collapse toward 0 (deceived).
    println!("UMDA (seed {TRAP_TRACE_SEED}) — independent per-dimension Gaussian:");
    println!("  {:>4} {:>10}", "gen", "mean μ");
    let umda_best = run_eda(
        UnivariateGaussian,
        &trap_continuous_params(trap_umda_params()),
        TRAP_TRACE_SEED,
        TRAP_GENS,
        |row| trap_fitness(trap, row),
        |g, st: &UnivariateGaussianState| {
            if logged(g) {
                println!("  {:>4} {:>10.5}", g + 1, mean(&st.mean));
            }
        },
    );
    println!("  → best fitness (0 = all-ones found): {umda_best:.1}\n");

    // MIMIC trace: means + strongest captured link. A first-order chain ties at
    // most one neighbour per gene, so it cannot represent order-5 linkage and is
    // deceived just like UMDA.
    println!("MIMIC (seed {TRAP_TRACE_SEED}) — first-order dependency chain:");
    println!("  {:>4} {:>10} {:>10}", "gen", "mean μ", "max|r|");
    let mimic_best = run_eda(
        DependencyChain,
        &trap_continuous_params(trap_mimic_params()),
        TRAP_TRACE_SEED,
        TRAP_GENS,
        |row| trap_fitness(trap, row),
        |g, st: &DependencyChainState| {
            if logged(g) {
                let max_corr = st
                    .link_corr
                    .iter()
                    .map(|c| c.abs())
                    .fold(0.0_f32, f32::max);
                println!("  {:>4} {:>10.5} {max_corr:>10.4}", g + 1, mean(&st.mean));
            }
        },
    );
    println!("  → best fitness (0 = all-ones found): {mimic_best:.1}\n");

    // BOA trace: total/intra-block edge counts + a compact per-block parent map.
    // Track the peak intra-block edge count across the WHOLE run (every
    // generation, not just logged ones) — the structure is transient, so the
    // logged snapshots alone understate the linkage BOA exploited.
    println!("BOA (seed {TRAP_TRACE_SEED}) — bounded-in-degree Bayesian network:");
    println!("  per-gen edge structure (an edge u→v is intra-block when u/{TRAP_K} == v/{TRAP_K}):");
    let mut peak_intra = 0usize;
    let boa_best = run_eda(
        BayesianNetwork,
        &trap_boa_params(),
        TRAP_TRACE_SEED,
        TRAP_GENS,
        |row| trap_fitness(trap, row),
        |g, st: &BayesianNetworkState| {
            let (total, intra, blocks) = edge_stats(st);
            // Peak tracking runs every generation, not just on logged ones.
            peak_intra = peak_intra.max(intra);
            if logged(g) {
                println!(
                    "  gen {:>2}: edges {total} (intra-block {intra}) | blocks: {}",
                    g + 1,
                    block_map(&blocks)
                );
            }
        },
    );
    println!("  peak intra-block edges over all {TRAP_GENS} generations: {peak_intra}");
    println!("  → best fitness (0 = all-ones found): {boa_best:.1}\n");

    trap_medians(trap);
}

/// Robustness check: median final cost over the five ADR 0018 seeds for all
/// three models, so BOA's win reads as structural rather than one lucky seed.
fn trap_medians(trap: ConcatenatedTrap) {
    let mut umda: Vec<f32> = TRAP_SEEDS
        .iter()
        .map(|&seed| {
            run_eda(
                UnivariateGaussian,
                &trap_continuous_params(trap_umda_params()),
                seed,
                TRAP_GENS,
                |row| trap_fitness(trap, row),
                |_g, _st: &UnivariateGaussianState| {},
            )
        })
        .collect();
    let mut mimic: Vec<f32> = TRAP_SEEDS
        .iter()
        .map(|&seed| {
            run_eda(
                DependencyChain,
                &trap_continuous_params(trap_mimic_params()),
                seed,
                TRAP_GENS,
                |row| trap_fitness(trap, row),
                |_g, _st: &DependencyChainState| {},
            )
        })
        .collect();
    let mut boa: Vec<f32> = TRAP_SEEDS
        .iter()
        .map(|&seed| {
            run_eda(
                BayesianNetwork,
                &trap_boa_params(),
                seed,
                TRAP_GENS,
                |row| trap_fitness(trap, row),
                |_g, _st: &BayesianNetworkState| {},
            )
        })
        .collect();
    let umda_med = median(&mut umda);
    let mimic_med = median(&mut mimic);
    let boa_med = median(&mut boa);
    println!("Median final cost over {} seeds:", TRAP_SEEDS.len());
    println!("  UMDA  : {umda_med:.1}  (deceived — stalls near the all-zeros basin)");
    println!("  MIMIC : {mimic_med:.1}  (deceived — first-order chain misses order-5 linkage)");
    println!("  BOA   : {boa_med:.1}  (solves — captures the intra-block linkage)");
    let verdict = if boa_med < umda_med && boa_med < mimic_med {
        "BOA wins — only the multivariate model escapes the deceptive trap"
    } else {
        "BOA did not separate from UMDA/MIMIC this run"
    };
    println!("  → {verdict}\n");
}

fn main() {
    println!(
        "Estimation-of-Distribution Algorithms: replace crossover/mutation with a\n\
         fit → sample loop over an explicit probability model.\n"
    );
    rosenbrock_demo();
    onemax_demo();
    trap_demo();
}
