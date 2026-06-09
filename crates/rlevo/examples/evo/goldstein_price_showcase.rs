//! Goldstein-Price landscape — Steep multiplicative polynomial valleys; global minimum f=3.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example goldstein_price_showcase`.
//!
//! # Interpreting the output
//!
//! Global optimum `f* = 3` (not `0`) at `(0, −1)`, searched over `[-2, 2]²`. A
//! `best` settling near `3` is full convergence. The value ranges up to
//! `~1.3×10⁶`, so early `mean` readings can be enormous; a `mean` still in the
//! thousands while `best ≈ 3` means most of the population sits on the steep
//! outer walls and has not yet descended. See the `common` module's "Reading
//! the output" section for the full column legend.

mod common;

use rlevo_environments::landscapes::goldstein_price::GoldsteinPrice;

fn main() {
    let dim = 2;
    let landscape = GoldsteinPrice::new();
    common::showcase("GoldsteinPrice", dim, landscape.bounds(), 0.1, landscape);
}
