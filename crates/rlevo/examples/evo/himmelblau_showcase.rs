//! Himmelblau landscape — Four identical global minima at f=0 — a multimodality and niching test.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example himmelblau_showcase`.
//!
//! # Interpreting the output
//!
//! Four equal global minima of value `f* = 0` arranged around the origin,
//! searched over `[-6, 6]²`. A `best` near `0` means a run found one of the
//! four basins — `best` alone cannot tell you *which*. Because the optima are
//! symmetric and equally deep, `mean` reflects how the population spread across
//! them; this is a niching test, and the single scalar summary deliberately
//! does not distinguish multi-basin coverage. See the `common` module's
//! "Reading the output" section for the full column legend.

mod common;

use rlevo_environments::landscapes::himmelblau::Himmelblau;

fn main() {
    let dim = 2;
    let landscape = Himmelblau::new();
    common::showcase("Himmelblau", dim, landscape.bounds(), 0.3, landscape);
}
