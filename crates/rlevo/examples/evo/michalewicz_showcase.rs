//! Michalewicz landscape — Steep ridges and flat valleys (m=10); near-zero gradient away from the narrow optimal channels.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example michalewicz_showcase`.
//!
//! # Interpreting the output
//!
//! Certified global optimum `f* ≈ −9.66015` for `n = 10` (steepness `m = 10`),
//! searched over `[0, π]^10`. Cost is negative, so a more-negative `best` is
//! better; `best ≈ −9.66` is full convergence. The surface is near-flat away
//! from a few narrow optimal channels, giving almost no gradient to follow, so
//! a `best` only reaching `−4` to `−6` is the expected outcome for a strategy
//! that explored poorly. See the `common` module's "Reading the output"
//! section for the full column legend.

mod common;

use rlevo_environments::landscapes::michalewicz::Michalewicz;

fn main() {
    let dim = 10;
    let landscape = Michalewicz::new(dim);
    common::showcase("Michalewicz", dim, landscape.bounds(), 0.1, landscape);
}
