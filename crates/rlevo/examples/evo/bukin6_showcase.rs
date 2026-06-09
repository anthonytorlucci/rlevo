//! Bukin6 landscape — A knife-edge ridge: the optimum lies along a near-singular curved seam.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example bukin6_showcase`.
//!
//! # Interpreting the output
//!
//! Global optimum `f* = 0` at `(−10, 1)`, searched over the square box
//! `[-15, 3]²` (the bounding box of the true asymmetric domain). A `best` near
//! `0` is full convergence. The optimum sits on a near-singular parabolic ridge
//! where the transverse gradient diverges, so even strong runs typically
//! plateau around `0.04`–`0.2` rather than reaching `0` — this is the expected
//! difficulty, not a failure. See the `common` module's "Reading the output"
//! section for the full column legend.

mod common;

use rlevo_environments::landscapes::bukin6::Bukin6;

fn main() {
    let dim = 2;
    let landscape = Bukin6::new();
    common::showcase("Bukin6", dim, landscape.bounds(), 0.4, landscape);
}
