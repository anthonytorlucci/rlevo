//! Shared support for the Santa Fe Trail neuroevolution examples (issue #69).
//!
//! This module is the flagship illustration for `rlevo-evolution`: it evolves a
//! small **recurrent** policy on the artificial-ant POMDP **with no new genome
//! representation**. The classic ant is a genetic-programming benchmark whose
//! solution is a program tree; we do not add one. Every phase-1 strategy in
//! `rlevo-evolution` already evolves a flat `f32` vector (`Tensor<B, 2>`), so we
//! keep it: define a recurrent net, flatten its weights into that same vector
//! ([`ModuleReshaper`] / [`WeightOnly`]), evolve with any unchanged strategy, and
//! score by un-flattening back into a net and running it as the ant's policy.
//!
//! The net supplies the **memory** the POMDP demands (the one-bit `food_ahead`
//! percept makes a reactive policy provably stall at the trail's gaps); the flat
//! genome supplies the **search space**, unchanged.
//!
//! ## Why score through [`ModuleEvalFn`] and not `RolloutFitness`
//!
//! `rlevo-hybrid::RolloutFitness`'s policy closure is stateless
//! (`Fn(&M, &Observation, &Device) -> Action`), so it can only express a reactive
//! map — exactly the class that fails here. [`ModuleEvalFn`]'s scorer is
//! `Fn(&M) -> f32` and owns the *entire* rollout, so the recurrent hidden state
//! lives as a local tensor threaded across the 600 steps. Generalizing
//! `RolloutFitness` to stateful policies is the right library fix, but it is a
//! production change tracked separately (issues #91/#92); this module is the
//! self-contained reference such a generalization would factor out.
//!
//! This file is `#[path]`-included by the two example binaries and by the
//! integration test, so it carries `#![allow(dead_code)]` (each consumer uses a
//! subset).

#![allow(dead_code)]

use burn::module::Module;
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::Backend;

use rand::RngExt;
use rand::rngs::StdRng;

use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::classic::santa_fe_ant::{
    DEFAULT_MAX_STEPS, SantaFeAnt, SantaFeAntAction, SantaFeAntConfig, TOTAL_PELLETS,
};

use rlevo_evolution::algorithms::cma_es::{CmaEs, CmaEsConfig};
use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo_evolution::algorithms::memetic::MemeticParams;
use rlevo_evolution::local_search::{HillClimbing, HillClimbingParams};
use rlevo_evolution::rng::{SeedPurpose, seed_stream};
use rlevo_evolution::{
    CoveragePolicy, EdaParams, EdaStrategy, EvolutionaryHarness, MemeticWrapper, ModuleEvalFn,
    ModuleReshaper, Strategy, UnivariateGaussian, UnivariateGaussianParams, WeightOnly,
    WritebackPolicy,
};

// ---------------------------------------------------------------------------
// Tunable constants (OQ-2 / OQ-4 knobs — revisit empirically).
// ---------------------------------------------------------------------------

/// GRU hidden width. Wider = more memory capacity but a wider genome (more
/// evaluations to solve). `16` is the maintainer-chosen default.
pub const GRU_HIDDEN: usize = 16;

/// Elman hidden width — the smaller, size-comparison architecture.
pub const ELMAN_HIDDEN: usize = 12;

/// Fixed seeds the stochastic example averages over per fitness evaluation.
pub const STOCHASTIC_SEEDS: usize = 16;

/// Steps per rollout (the canonical Santa Fe budget).
pub const MAX_STEPS: usize = DEFAULT_MAX_STEPS;

/// Individuals per generation for the population-based strategies.
pub const POP: usize = 32;

/// Generation budget for the **deterministic** example (one rollout per eval).
/// Illustrative, not exhaustive — total evals ≈ `POP * DET_GENERATIONS`, an order
/// of magnitude below the ~20,696-evaluation GP baseline (Christensen & Oppacher
/// 2007). Scale up to clear the full trail.
pub const DET_GENERATIONS: usize = 40;

/// Generation budget for the **stochastic** example. Lower than `DET_GENERATIONS`
/// because each fitness evaluation runs [`STOCHASTIC_SEEDS`] rollouts, so a
/// generation costs ~16× more.
pub const STO_GENERATIONS: usize = 12;

/// Initial-sample / clamp bounds for genome weights. Narrower than the GA default
/// `(-5.12, 5.12)` because recurrent nets are init-sensitive (OQ-2).
pub const GENOME_BOUNDS: (f32, f32) = (-3.0, 3.0);

/// The full-trail target, as `f32` (89 pellets; exact in `f32`).
#[allow(clippy::cast_precision_loss)]
const SOLVED: f32 = TOTAL_PELLETS as f32;

// ---------------------------------------------------------------------------
// Recurrent policy architectures.
// ---------------------------------------------------------------------------

/// GRU policy: `Gru(1 -> H) -> Linear(H -> 3)`. The primary architecture; reuses
/// the maintainer's proven `CustomGruRnn` shape. Burn's [`Gru`] keeps the hidden
/// state as a runtime tensor passed in/out (not module-stored), so its weights
/// flatten/round-trip through [`ModuleReshaper`] exactly like an MLP's.
#[derive(Module, Debug)]
pub struct GruAntPolicy<B: Backend> {
    gru: Gru<B>,
    head: Linear<B>,
}

impl<B: Backend> GruAntPolicy<B> {
    /// Build a fresh template. Only the *shape* matters — evolution overwrites the
    /// weights at every `unflatten`, so the initializer is irrelevant to search.
    #[must_use]
    pub fn new(device: &B::Device) -> Self {
        Self {
            gru: GruConfig::new(1, GRU_HIDDEN, true).init(device),
            head: LinearConfig::new(GRU_HIDDEN, 3).with_bias(true).init(device),
        }
    }
}

impl<B: Backend> AntPolicy<B> for GruAntPolicy<B> {
    fn hidden_size(&self) -> usize {
        self.gru.d_hidden
    }

    fn step(&self, food_ahead: bool, h: &mut Tensor<B, 2>, device: &B::Device) -> [f32; 3] {
        let x = encode_percept(food_ahead, device).reshape([1, 1, 1]);
        let out = self.gru.forward(x, Some(h.clone())); // [1, 1, H]
        let new_h = out.reshape([1, self.gru.d_hidden]); // [1, H]
        let logits = self.head.forward(new_h.clone()); // [1, 3]
        *h = new_h;
        to_logits3(logits)
    }
}

/// Elman policy: a minimal hand-rolled recurrent cell. With the standard notation
/// `W_x` = [`input`], `W_h` = [`recurrent`], `W_o` = [`output`], one step is
/// `h' = tanh(W_x·x + W_h·h)`, `logits = W_o·h'`. The smallest genome, used as the
/// size-comparison arm and the fairest baseline comparison.
///
/// [`input`]: ElmanAntPolicy::input
/// [`recurrent`]: ElmanAntPolicy::recurrent
/// [`output`]: ElmanAntPolicy::output
#[derive(Module, Debug)]
pub struct ElmanAntPolicy<B: Backend> {
    /// `W_x`: input → hidden.
    input: Linear<B>,
    /// `W_h`: hidden → hidden (recurrent).
    recurrent: Linear<B>,
    /// `W_o`: hidden → 3 action logits.
    output: Linear<B>,
}

impl<B: Backend> ElmanAntPolicy<B> {
    /// Build a fresh template (shape only; weights are evolved).
    #[must_use]
    pub fn new(device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(1, ELMAN_HIDDEN).with_bias(true).init(device),
            recurrent: LinearConfig::new(ELMAN_HIDDEN, ELMAN_HIDDEN)
                .with_bias(false)
                .init(device),
            output: LinearConfig::new(ELMAN_HIDDEN, 3).with_bias(true).init(device),
        }
    }
}

impl<B: Backend> AntPolicy<B> for ElmanAntPolicy<B> {
    fn hidden_size(&self) -> usize {
        // `output` weight has shape [H, 3]; its leading dim is the hidden width.
        self.output.weight.val().dims()[0]
    }

    fn step(&self, food_ahead: bool, h: &mut Tensor<B, 2>, device: &B::Device) -> [f32; 3] {
        let x = encode_percept(food_ahead, device).reshape([1, 1]); // [1, 1]
        let pre = self.input.forward(x) + self.recurrent.forward(h.clone()); // [1, H]
        let new_h = tanh(pre);
        let logits = self.output.forward(new_h.clone()); // [1, 3]
        *h = new_h;
        to_logits3(logits)
    }
}

/// A recurrent ant policy: one env step advances internal state and emits three
/// action logits (`Move`, `TurnLeft`, `TurnRight`).
pub trait AntPolicy<B: Backend> {
    /// Hidden-state width `H`.
    fn hidden_size(&self) -> usize;

    /// Advance the recurrence by one step (mutating `h` in place) and return the
    /// action logits for the current `food_ahead` percept.
    fn step(&self, food_ahead: bool, h: &mut Tensor<B, 2>, device: &B::Device) -> [f32; 3];

    /// Zero-initialised hidden state `[1, H]`.
    fn init_hidden(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::zeros([1, self.hidden_size()], device)
    }
}

/// Encode the one-bit percept as a length-1 float tensor.
fn encode_percept<B: Backend>(food_ahead: bool, device: &B::Device) -> Tensor<B, 1> {
    let v = if food_ahead { 1.0_f32 } else { 0.0_f32 };
    Tensor::<B, 1>::from_floats([v], device)
}

/// Extract three logits from a `[1, 3]` tensor.
fn to_logits3<B: Backend>(logits: Tensor<B, 2>) -> [f32; 3] {
    let data = logits.into_data();
    let v = data.to_vec::<f32>().expect("policy head must yield 3 f32 logits");
    [v[0], v[1], v[2]]
}

// ---------------------------------------------------------------------------
// Action selection (the single knob that distinguishes the two examples).
// ---------------------------------------------------------------------------

/// Greedy argmax over the three action logits.
pub fn argmax3(logits: &[f32; 3]) -> usize {
    let mut best = 0;
    for i in 1..3 {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best
}

/// Numerically stable softmax sample over the three action logits.
pub fn softmax_sample(logits: &[f32; 3], rng: &mut StdRng) -> usize {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = [
        (logits[0] - max).exp(),
        (logits[1] - max).exp(),
        (logits[2] - max).exp(),
    ];
    let sum: f32 = exps.iter().sum();
    let threshold = rng.random::<f32>() * sum;
    let mut acc = 0.0;
    for (i, e) in exps.iter().enumerate() {
        acc += *e;
        if threshold < acc {
            return i;
        }
    }
    2
}

// ---------------------------------------------------------------------------
// Rollout + fitness.
// ---------------------------------------------------------------------------

/// Run one full rollout of `policy` on a fresh [`SantaFeAnt`], threading the
/// recurrent hidden state, and return the number of pellets eaten (`0..=89`).
///
/// `select` maps the three action logits to an action index; it receives `rng`
/// so a stochastic selector can sample. Deterministic callers pass an argmax
/// selector that ignores `rng`.
pub fn rollout_pellets<B, P>(
    policy: &P,
    select: impl Fn(&[f32; 3], &mut StdRng) -> usize,
    rng: &mut StdRng,
    max_steps: usize,
    device: &B::Device,
) -> f32
where
    B: Backend,
    P: AntPolicy<B>,
{
    let mut env = SantaFeAnt::with_config(SantaFeAntConfig {
        max_steps,
        render: false,
    }).expect("valid config");
    let mut snap = env.reset().expect("reset");
    let mut h = policy.init_hidden(device);
    let mut eaten = 0.0_f32;
    loop {
        let food_ahead = snap.observation().food_ahead;
        let logits = policy.step(food_ahead, &mut h, device);
        let action = SantaFeAntAction::from_index(select(&logits, rng));
        snap = env.step(action).expect("step");
        eaten += f32::from(*snap.reward());
        if snap.is_done() {
            break;
        }
    }
    eaten
}

/// How a genome's fitness is measured.
#[derive(Clone, Copy, Debug)]
pub enum ScoreMode {
    /// Argmax action selection, a single deterministic rollout. Bit-reproducible.
    Deterministic,
    /// Seeded softmax sampling, mean pellets over a fixed seed set.
    Stochastic { seeds: usize },
}

/// Build a scorer closure for [`ModuleEvalFn`]. The closure is `Fn` (not `FnMut`):
/// every capture is immutable and any RNG is constructed *inside* per call, so the
/// same genome always scores identically.
#[allow(clippy::cast_precision_loss)] // seed counts are tiny; exact in f32
pub fn make_scorer<B, P>(
    mode: ScoreMode,
    base_seed: u64,
    max_steps: usize,
    device: B::Device,
) -> impl Fn(&P) -> f32 + Send
where
    B: Backend,
    P: AntPolicy<B>,
{
    move |policy: &P| -> f32 {
        match mode {
            ScoreMode::Deterministic => {
                let mut rng = seed_stream(base_seed, 0, SeedPurpose::Trial);
                rollout_pellets::<B, P>(policy, |l, _r| argmax3(l), &mut rng, max_steps, &device)
            }
            ScoreMode::Stochastic { seeds } => {
                let mut total = 0.0_f32;
                for s in 0..seeds {
                    let mut rng = seed_stream(base_seed, s as u64, SeedPurpose::Trial);
                    total +=
                        rollout_pellets::<B, P>(policy, softmax_sample, &mut rng, max_steps, &device);
                }
                total / seeds as f32
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy runner + reporting.
// ---------------------------------------------------------------------------

/// One strategy×architecture result.
#[derive(Clone, Debug)]
pub struct RunSummary {
    pub label: String,
    /// Best pellets found (declared sense — maximise, so higher is better).
    pub best_pellets: f32,
    /// Nominal evaluations to first reach 89 pellets (`pop * generations`), or
    /// `None` if the trail was never cleared within budget. Memetic local-search
    /// probes are *not* counted here (population evaluations only).
    pub evals_to_solved: Option<usize>,
    pub generations: usize,
}

/// Drive an [`EvolutionaryHarness`] to its generation budget and summarise.
#[allow(clippy::too_many_arguments)] // a thin example driver; bundling these into a struct adds noise
pub fn run_strategy<B, S, M, F>(
    label: &str,
    strategy: WeightOnly<B, S, M>,
    params: S::Params,
    fitness: ModuleEvalFn<B, ModuleReshaper<B, M>, F>,
    seed: u64,
    device: B::Device,
    max_gens: usize,
    pop: usize,
) -> RunSummary
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    S::Params: rlevo_core::config::Validate,
    M: Module<B> + Sync,
    F: Fn(&M) -> f32 + Send,
{
    let mut harness = EvolutionaryHarness::new(strategy, params, fitness, seed, device, max_gens).expect("valid params");
    harness.reset();
    let mut evals_to_solved = None;
    let mut generations = 0;
    loop {
        let step = harness.step(());
        generations += 1;
        if evals_to_solved.is_none()
            && let Some(m) = harness.latest_metrics()
            && m.best_fitness_ever >= SOLVED
        {
            evals_to_solved = Some(generations * pop);
        }
        if step.done {
            break;
        }
    }
    let best_pellets = harness.best().map_or(0.0, |(_, f)| f);
    RunSummary {
        label: label.to_string(),
        best_pellets,
        evals_to_solved,
        generations,
    }
}

/// Run all four strategies (GA, EDA/UMDA, CMA-ES, memetic) over one architecture
/// `template`, scored by `mode`. Returns one [`RunSummary`] per strategy.
///
/// `M` is the recurrent policy module: it must be `Clone` (templates feed two
/// constructors), `Sync` (the reshaper bound), and an [`AntPolicy`] (the scorer).
pub fn evolve_all<B, M>(
    arch: &str,
    template: &M,
    mode: ScoreMode,
    base_seed: u64,
    pop: usize,
    gens: usize,
    device: &B::Device,
) -> Vec<RunSummary>
where
    B: Backend,
    M: Module<B> + Clone + Sync + AntPolicy<B>,
{
    let dim = ModuleReshaper::new(template.clone()).num_params();
    let scorer = |seed: u64| make_scorer::<B, M>(mode, seed, MAX_STEPS, device.clone());
    let fitness = |seed: u64| ModuleEvalFn::new(ModuleReshaper::new(template.clone()), scorer(seed));
    let mut out = Vec::new();

    // --- Genetic algorithm -------------------------------------------------
    {
        let wo = WeightOnly::new(GeneticAlgorithm::<B>::new(), template.clone());
        debug_assert_eq!(wo.num_params(), dim);
        let mut cfg = GaConfig::default_for(pop, dim);
        cfg.bounds = GENOME_BOUNDS;
        out.push(run_strategy(
            &format!("{arch}/GA"),
            wo,
            cfg,
            fitness(base_seed),
            base_seed,
            device.clone(),
            gens,
            pop,
        ));
    }

    // --- Estimation of Distribution (UMDA) --------------------------------
    {
        let wo = WeightOnly::new(EdaStrategy::<B, _>::new(UnivariateGaussian), template.clone());
        let params = EdaParams {
            pop_size: pop,
            selection_ratio: 0.5,
            bounds: Some(GENOME_BOUNDS),
            model: UnivariateGaussianParams::default_for(dim),
        };
        out.push(run_strategy(
            &format!("{arch}/EDA"),
            wo,
            params,
            fitness(base_seed + 1),
            base_seed + 1,
            device.clone(),
            gens,
            pop,
        ));
    }

    // --- CMA-ES ------------------------------------------------------------
    {
        let cfg = CmaEsConfig::default_for(dim);
        let cma_pop = cfg.pop_size;
        let wo = WeightOnly::new(CmaEs::<B>::new(), template.clone());
        out.push(run_strategy(
            &format!("{arch}/CMA-ES"),
            wo,
            cfg,
            fitness(base_seed + 2),
            base_seed + 2,
            device.clone(),
            gens,
            cma_pop,
        ));
    }

    // --- Memetic (GA + hill-climbing local search) ------------------------
    {
        let inner_fitness = fitness(base_seed + 3);
        let memetic = MemeticWrapper::new(GeneticAlgorithm::<B>::new(), HillClimbing, inner_fitness);
        let wo = WeightOnly::new(memetic, template.clone());
        let mut ga = GaConfig::default_for(pop, dim);
        ga.bounds = GENOME_BOUNDS;
        let params = MemeticParams {
            inner: ga,
            local: HillClimbingParams::default_for(GENOME_BOUNDS),
            writeback: WritebackPolicy::default(),
            coverage: CoveragePolicy::default(),
        };
        out.push(run_strategy(
            &format!("{arch}/Memetic"),
            wo,
            params,
            fitness(base_seed + 3),
            base_seed + 3,
            device.clone(),
            gens,
            pop,
        ));
    }

    out
}

/// Run a single short GA on the GRU policy. Used by the cross-crate agent test as
/// a fast, self-contained "does evolution find a memory policy" probe.
pub fn quick_evolve_gru<B>(
    mode: ScoreMode,
    seed: u64,
    pop: usize,
    gens: usize,
    device: &B::Device,
) -> RunSummary
where
    B: Backend,
{
    let template = GruAntPolicy::<B>::new(device);
    let dim = ModuleReshaper::new(template.clone()).num_params();
    let wo = WeightOnly::new(GeneticAlgorithm::<B>::new(), template.clone());
    let mut cfg = GaConfig::default_for(pop, dim);
    cfg.bounds = GENOME_BOUNDS;
    let fitness = ModuleEvalFn::new(
        ModuleReshaper::new(template),
        make_scorer::<B, GruAntPolicy<B>>(mode, seed, MAX_STEPS, device.clone()),
    );
    run_strategy("GRU/GA", wo, cfg, fitness, seed, device.clone(), gens, pop)
}

/// Print a per-strategy results table with the GP baseline as reference.
pub fn print_results_table(rows: &[RunSummary]) {
    println!();
    println!("{:<20} {:>14} {:>14} {:>8}", "strategy/arch", "best pellets", "evals→89", "gens");
    println!("{}", "-".repeat(58));
    for r in rows {
        let evals = r
            .evals_to_solved
            .map_or_else(|| "—".to_string(), |e| e.to_string());
        println!(
            "{:<20} {:>14.1} {:>14} {:>8}",
            r.label, r.best_pellets, evals, r.generations
        );
    }
    println!("{}", "-".repeat(58));
    println!(
        "reference: tree-GP solves the ant in ~20,696 evaluations \
         (Christensen & Oppacher 2007); a neuroevolution genome is not directly \
         eval-comparable, so read this as an order-of-magnitude target."
    );
    println!("(89 pellets = full trail cleared; '—' = not solved within budget)");
}

// ---------------------------------------------------------------------------
// Unit tests (run via the integration target that `#[path]`-includes this file).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::backend::BackendTypes;
    use rlevo_evolution::ParamReshaper;

    type B = Flex;

    fn device() -> <B as BackendTypes>::Device {
        Default::default()
    }

    /// Weights survive a flatten → unflatten → flatten round-trip for both archs.
    #[test]
    fn weights_round_trip_gru() {
        let dev = device();
        let template = GruAntPolicy::<B>::new(&dev);
        let reshaper = ModuleReshaper::new(template.clone());
        let flat = reshaper.flatten(&template, &dev);
        let rebuilt = reshaper.unflatten(flat.clone());
        let reflat = reshaper.flatten(&rebuilt, &dev);
        let a = flat.into_data().to_vec::<f32>().unwrap();
        let b = reflat.into_data().to_vec::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "weight drift on round-trip: {x} vs {y}");
        }
    }

    #[test]
    fn weights_round_trip_elman() {
        let dev = device();
        let template = ElmanAntPolicy::<B>::new(&dev);
        let reshaper = ModuleReshaper::new(template.clone());
        let flat = reshaper.flatten(&template, &dev);
        let rebuilt = reshaper.unflatten(flat.clone());
        let reflat = reshaper.flatten(&rebuilt, &dev);
        let a = flat.into_data().to_vec::<f32>().unwrap();
        let b = reflat.into_data().to_vec::<f32>().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    /// A rollout returns a pellet count in the valid range `0..=89`.
    #[test]
    fn rollout_in_range() {
        let dev = device();
        let policy = GruAntPolicy::<B>::new(&dev);
        let mut rng = seed_stream(7, 0, SeedPurpose::Trial);
        let pellets = rollout_pellets::<B, _>(&policy, |l, _r| argmax3(l), &mut rng, MAX_STEPS, &dev);
        assert!((0.0..=SOLVED).contains(&pellets), "pellets out of range: {pellets}");
    }

    /// Recurrence is load-bearing: identical percepts with different prior hidden
    /// states produce different action logits (a memoryless map could not).
    #[test]
    fn recurrence_carries_state_gru() {
        let dev = device();
        let policy = GruAntPolicy::<B>::new(&dev);
        let mut h_zero = policy.init_hidden(&dev);
        let mut h_warm = Tensor::<B, 2>::ones([1, GRU_HIDDEN], &dev) * 0.5;
        let from_zero = policy.step(false, &mut h_zero, &dev);
        let from_warm = policy.step(false, &mut h_warm, &dev);
        let differs = (0..3).any(|i| (from_zero[i] - from_warm[i]).abs() > 1e-6);
        assert!(differs, "hidden state did not influence the output");
    }

    #[test]
    fn recurrence_carries_state_elman() {
        let dev = device();
        let policy = ElmanAntPolicy::<B>::new(&dev);
        let mut h_zero = policy.init_hidden(&dev);
        let mut h_warm = Tensor::<B, 2>::ones([1, ELMAN_HIDDEN], &dev) * 0.5;
        let from_zero = policy.step(false, &mut h_zero, &dev);
        let from_warm = policy.step(false, &mut h_warm, &dev);
        let differs = (0..3).any(|i| (from_zero[i] - from_warm[i]).abs() > 1e-6);
        assert!(differs, "hidden state did not influence the output");
    }
}
