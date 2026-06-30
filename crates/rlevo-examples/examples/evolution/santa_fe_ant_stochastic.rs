//! Neuroevolution on the Santa Fe Trail — **stochastic** variant (issue #69).
//!
//! Identical pipeline to the deterministic example but for a single knob: action
//! selection is a seeded softmax sample and fitness is the **mean** pellets over a
//! fixed seed set ([`STOCHASTIC_SEEDS`]). Env dynamics stay deterministic;
//! stochasticity enters only at action selection. This shows the same support
//! code is fitness-regime-agnostic — the maintainer's reusability showcase.
//!
//! ```bash
//! cargo run -p rlevo-examples --example santa_fe_ant_stochastic
//! ```

#[path = "santa_fe_ant_support.rs"]
mod support;

use burn::backend::Flex;
use support::{
    ElmanAntPolicy, GruAntPolicy, POP, RunSummary, STOCHASTIC_SEEDS, STO_GENERATIONS, ScoreMode,
    evolve_all, print_results_table,
};

type B = Flex;

fn main() {
    // Pin rayon to one thread for reproducibility (see the deterministic example).
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();

    let device = Default::default();
    let seed = 0x5A_u64;
    let mode = ScoreMode::Stochastic {
        seeds: STOCHASTIC_SEEDS,
    };

    let mut rows: Vec<RunSummary> = Vec::new();

    let gru = GruAntPolicy::<B>::new(&device);
    rows.extend(evolve_all::<B, _>("GRU", &gru, mode, seed, POP, STO_GENERATIONS, &device));

    let elman = ElmanAntPolicy::<B>::new(&device);
    rows.extend(evolve_all::<B, _>("Elman", &elman, mode, seed, POP, STO_GENERATIONS, &device));

    print_results_table(&rows);
}
