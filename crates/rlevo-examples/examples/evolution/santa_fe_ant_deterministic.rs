//! Neuroevolution on the Santa Fe Trail — **deterministic** variant (issue #69).
//!
//! Evolves a recurrent policy (GRU and Elman) whose flattened weights are the
//! genome, scored by a single argmax rollout (fitness = pellets eaten). Runs four
//! strategies — GA, EDA/UMDA, CMA-ES, memetic — over each architecture and prints
//! a results table against the ~20,696-evaluation GP baseline.
//!
//! The shared policy/scorer/runner code lives in `santa_fe_ant_support.rs`.
//!
//! ```bash
//! cargo run -p rlevo-examples --example santa_fe_ant_deterministic
//! ```

#[path = "santa_fe_ant_support.rs"]
mod support;

use burn::backend::Flex;
use support::{
    DET_GENERATIONS, ElmanAntPolicy, GruAntPolicy, POP, RunSummary, ScoreMode, evolve_all,
    print_results_table,
};

type B = Flex;

fn main() {
    // Pin rayon to one thread: Burn seeds tensor RNG through process-global state,
    // so parallel evaluation would make the run non-reproducible.
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();

    let device = Default::default();
    let seed = 0x5A_u64;
    let mode = ScoreMode::Deterministic;

    let mut rows: Vec<RunSummary> = Vec::new();

    let gru = GruAntPolicy::<B>::new(&device);
    rows.extend(evolve_all::<B, _>("GRU", &gru, mode, seed, POP, DET_GENERATIONS, &device));

    let elman = ElmanAntPolicy::<B>::new(&device);
    rows.extend(evolve_all::<B, _>("Elman", &elman, mode, seed, POP, DET_GENERATIONS, &device));

    print_results_table(&rows);
}
