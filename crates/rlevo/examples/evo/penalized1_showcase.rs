//! Penalized1 landscape — Yao 1999 f14 — a coupled sine-penalty landscape with boundary penalties outside [-10, 10].
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example penalized1_showcase`.
//!
//! # Interpreting the output
//!
//! Global optimum `f* = 0` at `x_i = −1` (which zeroes every sinusoidal term),
//! searched over `[-50, 50]^10`. A `best` near `0` is full convergence. The
//! coupled sine terms plus the boundary penalty outside `[-10, 10]` punish
//! out-of-range individuals heavily, so an inflated `mean` versus a small
//! `best` usually means part of the population is still drifting in the
//! penalized region. See the `common` module's "Reading the output" section
//! for the full column legend.

mod common;

use rlevo_environments::landscapes::penalized1::Penalized1;

fn main() {
    let dim = 10;
    let landscape = Penalized1::new(dim);
    common::showcase("Penalized1", dim, landscape.bounds(), 2.0, landscape);
}
