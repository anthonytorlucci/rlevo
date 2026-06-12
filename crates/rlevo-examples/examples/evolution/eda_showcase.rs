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
//! Two demos, each paired with the problem that actually exercises its models:
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
//! multivariate binary model (BOA), which is deferred to issue #37.
//!
//! # Running
//!
//! ```text
//! cargo run --release -p rlevo-examples --example eda_showcase
//! ```
//!
//! No feature flags are required. Release is recommended — the run drives a few
//! thousand generations of Burn tensor ops.

use burn::backend::Flex;
use burn::tensor::backend::BackendTypes;
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::landscapes::rosenbrock::Rosenbrock;
use rlevo_evolution::algorithms::eda::{
    CompactGeneticState, DependencyChainState, UnivariateBernoulliState, UnivariateGaussianState,
};
use rlevo_evolution::{
    CompactGenetic, CompactGeneticParams, DependencyChain, DependencyChainParams, EdaParams,
    EdaStrategy, ProbabilityModel, Strategy, UnivariateBernoulli, UnivariateBernoulliParams,
    UnivariateGaussian, UnivariateGaussianParams,
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
const ROSEN_BOUNDS: Option<(f32, f32)> = Some((-2.048, 2.048));
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

fn main() {
    println!(
        "Estimation-of-Distribution Algorithms: replace crossover/mutation with a\n\
         fit → sample loop over an explicit probability model.\n"
    );
    rosenbrock_demo();
    onemax_demo();
}
