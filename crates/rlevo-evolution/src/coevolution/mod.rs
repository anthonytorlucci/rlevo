//! Co-evolutionary algorithms.
//!
//! Two populations evolve **simultaneously**, and each individual's fitness
//! depends on the *other* population. Two regimes ship in v1:
//!
//! - **Competitive** ([`CompetitiveCoEA`]) — populations are adversaries
//!   (predator vs. prey, Hillis 1990). Each is scored by how well it does
//!   against the other; the dynamic is an arms race.
//! - **Cooperative** ([`CooperativeCoEA`], CCGA — Potter & De Jong 1994) — a
//!   high-dimensional problem is decomposed across populations whose
//!   individuals combine (via *representatives*) into a full candidate.
//!
//! # Why not [`Strategy`](crate::strategy::Strategy)?
//!
//! [`Strategy::tell`](crate::strategy::Strategy::tell) accepts a single
//! fitness vector, but co-evolutionary fitness is inherently paired — each
//! population receives a vector computed relative to the other. Rather than
//! distort that contract, co-evolution gets its own [`CoEvolutionaryAlgorithm`]
//! trait and a dedicated [`CoEvolutionaryHarness`] that adapts to the existing
//! `rlevo-core::evaluation::BenchEnv` surface, exactly as
//! [`EvolutionaryHarness`](crate::strategy::EvolutionaryHarness) does. Both
//! co-evolutionary algorithms are built from ordinary inner
//! [`Strategy`](crate::strategy::Strategy) instances, so every phase-1/2/3a/3b
//! algorithm composes in unchanged.
//!
//! # Pathology mitigation
//!
//! Competitive co-evolution is prone to **cycling**; [`HallOfFameFitness`]
//! wraps any [`CoupledFitness`] to anchor scores against an archive of past
//! champions (Rosin & Belew 1997). It composes at construction — algorithms
//! are hall-of-fame-agnostic.
//!
//! # Coupling shape
//!
//! [`CoupledFitness`] takes a slice of populations and is N-population-ready;
//! v1 algorithms always pass exactly two. The harness exposes
//! `min(best_a, best_b)` (canonical maximise) as the benchmark reward (the
//! weaker population — lower canonical fitness — is the binding constraint).
//!
//! # References
//!
//! - Hillis (1990), *Co-evolving parasites improve simulated evolution as an
//!   optimization procedure.*
//! - Potter & De Jong (1994), *A cooperative coevolutionary approach to
//!   function optimization* (CCGA).
//! - Rosin & Belew (1997), *New methods for competitive coevolution* (hall of
//!   fame).
//! - Ficici (2004), *Solution concepts in coevolutionary algorithms* (cycling
//!   / intransitive dominance).

pub mod competitive;
pub mod cooperative;
pub mod fitness;
pub mod harness;
pub mod hof;

pub use competitive::{CompetitiveCoEA, CompetitiveCoEAParams};
pub use cooperative::{
    CooperativeCoEA, CooperativeCoEAParams, CooperativeState, RepresentativePolicy,
};
pub use fitness::CoupledFitness;
pub use harness::{CoEAMetrics, CoEvolutionaryHarness};
pub use hof::{HallOfFame, HallOfFameFitness};

use std::fmt::Debug;

use burn::tensor::backend::Backend;
use rand::Rng;

/// A co-evolutionary algorithm driving two populations under simultaneous
/// updates.
///
/// The analogue of [`Strategy`](crate::strategy::Strategy) for the coupled
/// case: [`init`](Self::init) builds the joint state and [`step`](Self::step)
/// advances one simultaneous-update generation (both populations `ask` → one
/// [`CoupledFitness`] evaluation → both `tell`). Implementors hold their inner
/// strategies and a [`CoupledFitness`] by value and carry no PRNG state — all
/// randomness flows through the explicit `rng` argument, per the crate's
/// host-RNG convention.
///
/// v1 ships [`CompetitiveCoEA`] and [`CooperativeCoEA`]; Stackelberg
/// (alternating) turn order is deferred.
pub trait CoEvolutionaryAlgorithm<B: Backend>: Send + Sync {
    /// Static parameters for a run (inner strategy params plus any coupling
    /// configuration).
    type Params: Clone + Debug + Send + Sync;

    /// Generation-to-generation joint state.
    type State: Clone + Debug + Send;

    /// Build the initial joint state (initializes both inner strategies).
    fn init(
        &self,
        params: &Self::Params,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State;

    /// Advance one simultaneous-update generation, returning the next state
    /// and this generation's [`CoEAMetrics`].
    fn step(
        &self,
        params: &Self::Params,
        state: Self::State,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Self::State, CoEAMetrics);

    /// Snapshot the current metrics without advancing a generation.
    fn metrics(&self, state: &Self::State) -> CoEAMetrics;
}

/// Joined state carrying both sub-strategy states plus per-population
/// best/mean trackers.
///
/// Generic over the two inner strategy *state* types (`StA`, `StB`) rather
/// than the strategies themselves, so it derives [`Clone`]/[`Debug`] without
/// spurious bounds on the strategy types. Used by [`CompetitiveCoEA`];
/// [`CooperativeCoEA`] wraps it in [`CooperativeState`] to add
/// representative archives.
#[derive(Debug, Clone)]
pub struct CoEAState<StA, StB> {
    /// Inner state of population A's strategy.
    pub state_a: StA,
    /// Inner state of population B's strategy.
    pub state_b: StB,
    /// Number of completed simultaneous-update generations.
    pub generation: u64,
    /// Best (highest, canonical maximise) fitness population A has seen across
    /// all generations.
    pub best_a: f32,
    /// Best (highest, canonical maximise) fitness population B has seen across
    /// all generations.
    pub best_b: f32,
    /// Mean fitness of population A in the most recent generation.
    pub mean_a: f32,
    /// Mean fitness of population B in the most recent generation.
    pub mean_b: f32,
}

impl<StA, StB> CoEAState<StA, StB> {
    /// Build the initial joint state with empty best/mean trackers.
    pub(crate) fn new(state_a: StA, state_b: StB) -> Self {
        Self {
            state_a,
            state_b,
            generation: 0,
            best_a: f32::NEG_INFINITY,
            best_b: f32::NEG_INFINITY,
            mean_a: f32::NAN,
            mean_b: f32::NAN,
        }
    }
}
