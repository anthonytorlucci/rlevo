//! A shipped optimiser on the landscape: a `(μ+λ)` evolution strategy in an
//! `EvolutionaryLoop`. The loop owns the run's RNG, its generation budget
//! and the objective's direction, so nothing here draws a random number or
//! negates a cost.

use rlevo::burn::tensor::Device;
use rlevo::evo::algorithms::es_classical::{EsConfig, EsKind, EvolutionStrategy};
use rlevo::evo::fitness::FromLandscape;
use rlevo::evo::strategy::EvolutionaryLoop;

use crate::engine;
use crate::well_siting::{DIM, WellSiting};

/// Parents kept from one generation to the next.
const MU: usize = 4;
/// Offspring proposed each generation.
const LAMBDA: usize = 12;

/// The loop the harness drives, one per trial.
pub type WellLoop = EvolutionaryLoop<EvolutionStrategy, FromLandscape<WellSiting>>;

/// A loop seeded with `seed`, which stops after `generations`.
pub fn evo_loop(seed: u64, generations: usize) -> Result<WellLoop, String> {
    let params = EsConfig::default_for(
        EsKind::MuPlusLambda {
            mu: MU,
            lambda: LAMBDA,
        },
        DIM,
    );
    EvolutionaryLoop::new(
        EvolutionStrategy::new(),
        params,
        engine::fitness(),
        seed,
        Device::flex(),
        generations,
    )
    .map_err(|e| format!("EvolutionaryLoop::new refused the ES's parameters: {e}"))
}

/// The loop must accept the shipped strategy's parameters.
pub fn check() -> Result<String, String> {
    evo_loop(2026, 1)?;
    Ok(format!(
        "EvolutionaryLoop::new accepts a ({MU}+{LAMBDA})-ES on the landscape; \
         the device is named through rlevo::burn"
    ))
}
