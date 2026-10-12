//! A hand-written ask/tell loop hands `tell` what `EvolutionaryLoop` does.
//!
//! `docs/ask-tell-contract.md` writes the loop out by hand and says that
//! `EvolutionaryLoop` is that loop with bookkeeping. Two public functions are
//! what make the claim one a reader can act on (ADR 0086):
//!
//! - `sanitize_fitness_tensor`, which a hand-written loop calls on the
//!   negated cost before `tell`;
//! - `sanitize_fitness`, which a `Strategy` written outside the crate calls
//!   on the fitness it pulls to host.
//!
//! This file is that reader. It defines a strategy outside the crate, drives
//! it once with `reset` / `step` / `is_done` and once with `init` / `ask` /
//! `tell`, and compares what every `tell` was given, bit for bit. The fitness
//! function returns a `NaN`, a `-inf` and a `+inf` cost in every generation,
//! so each branch of the rule is on the path, and the remaining rows are
//! finite costs of rows drawn from the RNG, so the comparison also covers how
//! the two loops thread it.

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::objective::ObjectiveSense;

use rlevo_evolution::fitness::{BatchFitnessFn, sanitize_fitness, sanitize_fitness_tensor};
use rlevo_evolution::strategy::{EvolutionaryLoop, Strategy, StrategyMetrics};

type B = Flex;
type Device = <B as burn::tensor::backend::BackendTypes>::Device;

const POP: usize = 6;
const DIM: usize = 3;
const GENERATIONS: usize = 5;
const SEED: u64 = 11;

/// Random search, written the way a strategy outside the crate is written.
///
/// Its state keeps what every `tell` received, as bits, which is what the
/// test compares. Nothing else about it matters.
#[derive(Debug, Clone, Copy)]
struct RandomSearch;

#[derive(Debug, Clone, Copy)]
struct Params;

impl Validate for Params {
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct State {
    generation: usize,
    best: f32,
    /// The fitness each `tell` so far was given, one row per generation.
    received: Vec<Vec<u32>>,
}

impl Strategy<B> for RandomSearch {
    type Params = Params;
    type State = State;
    type Genome = Tensor<B, 2>;

    fn init(&self, _: &Params, _: &mut dyn Rng, _: &Device) -> State {
        State {
            generation: 0,
            best: f32::NEG_INFINITY,
            received: Vec::new(),
        }
    }

    fn ask(
        &self,
        _: &Params,
        state: &State,
        rng: &mut dyn Rng,
        device: &Device,
    ) -> (Tensor<B, 2>, State) {
        // Sixteen bits of each draw, mapped onto [-1, 1].
        let genes: Vec<f32> = (0..POP * DIM)
            .map(|_| {
                let bits = u16::try_from(rng.next_u32() >> 16).expect("sixteen bits fit");
                f32::from(bits) / f32::from(u16::MAX) * 2.0 - 1.0
            })
            .collect();
        let population = Tensor::from_data(TensorData::new(genes, [POP, DIM]), device);
        (population, state.clone())
    }

    fn tell(
        &self,
        _: &Params,
        _: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: State,
        _: &mut dyn Rng,
    ) -> (State, StrategyMetrics) {
        let raw = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("the fitness tensor holds f32");
        state
            .received
            .push(raw.iter().map(|f| f.to_bits()).collect());

        // The pull is where a strategy sanitizes: `tell` can be reached
        // without the loop (ADR 0034, Decision 3).
        let sane: Vec<f32> = raw.iter().map(|&f| sanitize_fitness(f)).collect();
        state.generation += 1;
        let metrics = StrategyMetrics::from_host_fitness(state.generation, &sane, state.best);
        state.best = metrics.best_fitness_ever();
        (state, metrics)
    }

    fn best(&self, _: &State) -> Option<(Tensor<B, 2>, f32)> {
        None
    }
}

/// The Sphere cost, with rows 0, 1 and 2 replaced by what an evaluation that
/// diverged returns: `NaN`, `-inf` and `+inf`.
struct BrokenSphere;

impl BatchFitnessFn<B, Tensor<B, 2>> for BrokenSphere {
    fn evaluate_batch(&mut self, population: &Tensor<B, 2>, device: &Device) -> Tensor<B, 1> {
        let genes = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("the population tensor holds f32");
        let mut cost: Vec<f32> = genes
            .chunks(DIM)
            .map(|row| row.iter().map(|x| x * x).sum())
            .collect();
        cost[0] = f32::NAN;
        cost[1] = f32::NEG_INFINITY;
        cost[2] = f32::INFINITY;
        Tensor::from_data(TensorData::new(cost, [POP]), device)
    }

    fn sense(&self) -> ObjectiveSense {
        ObjectiveSense::Minimize
    }
}

/// What each `tell` receives when `EvolutionaryLoop` drives the run.
fn received_under_the_loop() -> Vec<Vec<u32>> {
    let device = Device::default();
    let mut evo_loop = EvolutionaryLoop::<B, _, _>::new(
        RandomSearch,
        Params,
        BrokenSphere,
        SEED,
        device,
        GENERATIONS,
    )
    .expect("`Params` validates");

    evo_loop.reset();
    while !evo_loop.is_done() {
        evo_loop.step();
    }

    evo_loop
        .state()
        .expect("a loop that has stepped has a state")
        .received
        .clone()
}

/// What each `tell` receives from the loop the guide writes out by hand.
fn received_under_a_hand_written_loop() -> Vec<Vec<u32>> {
    let device = Device::default();
    let strategy = RandomSearch;
    let params = Params;
    let mut fitness_fn = BrokenSphere;
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut state = strategy.init(&params, &mut rng, &device);
    for _ in 0..GENERATIONS {
        let (population, next_state) = strategy.ask(&params, &state, &mut rng, &device);
        let cost = fitness_fn.evaluate_batch(&population, &device);
        // Negate, then sanitize. Never the other way round.
        let fitness = sanitize_fitness_tensor(cost.neg());
        let (new_state, _metrics) =
            strategy.tell(&params, population, fitness, next_state, &mut rng);
        state = new_state;
    }

    state.received
}

#[test]
fn a_hand_written_loop_hands_tell_what_the_loop_does() {
    let by_loop = received_under_the_loop();
    let by_hand = received_under_a_hand_written_loop();

    assert_eq!(
        by_loop.len(),
        GENERATIONS,
        "the loop told the strategy once per generation"
    );
    assert_eq!(
        by_hand, by_loop,
        "negating and then calling `sanitize_fitness_tensor` gives `tell` the \
         fitness `EvolutionaryLoop::step` gives it, bit for bit, in every \
         generation"
    );

    for (generation, row) in by_loop.iter().enumerate() {
        assert_eq!(
            row[0],
            f32::NEG_INFINITY.to_bits(),
            "generation {generation}: a `NaN` cost reaches `tell` as the worst value"
        );
        assert_eq!(
            row[1],
            f32::MAX.to_bits(),
            "generation {generation}: a `-inf` cost is the best member and \
             reaches `tell` finite"
        );
        assert_eq!(
            row[2],
            f32::NEG_INFINITY.to_bits(),
            "generation {generation}: a `+inf` cost passes through as the worst value"
        );
        assert!(
            row[3..]
                .iter()
                .all(|&bits| f32::from_bits(bits).is_finite()),
            "generation {generation}: the other rows are finite costs, negated"
        );
    }
}
