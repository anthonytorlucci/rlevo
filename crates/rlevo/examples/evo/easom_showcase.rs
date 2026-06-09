//! Easom landscape — A near-flat plane with a single sharp needle at (pi, pi) — pure exploration difficulty.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example easom_showcase`.
//!
//! # Interpreting the output
//!
//! Global optimum `f* = −1` at `(π, π)`, searched over `[-10, 10]²`. Cost is
//! negative, so `best = −1` is full convergence and `best ≈ 0` means the search
//! never found the needle. The surface is essentially flat at `0` everywhere
//! except a single sharp dip, so there is almost no gradient to guide search —
//! this is a pure-exploration test, and a `best` stuck at `0` is the common
//! failure mode rather than a bug. See the `common` module's "Reading the
//! output" section for the full column legend.

mod common;

use rlevo_environments::landscapes::easom::Easom;

fn main() {
    let dim = 2;
    let landscape = Easom::new();
    common::showcase("Easom", dim, landscape.bounds(), 0.4, landscape);
}
