//! Eggholder landscape — Highly rugged with the optimum pinned to the domain boundary; deep wells separated by tall barriers.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example eggholder_showcase`.
//!
//! # Interpreting the output
//!
//! Searched over `[-512, 512]^10`. Cost is negative, so a more-negative `best`
//! is better. The certified optimum `f* ≈ −959.64` is the `n = 2` value with
//! the optimum pinned to the boundary at `(512, 404.2318)`; the separable sum
//! reaches still-lower totals at higher `n`, so treat `best` here as
//! "how deep" rather than as a fixed target. Deep wells are separated by tall
//! barriers, so a `best`–`mean` gap that stays wide means the population is
//! scattered across competing wells. See the `common` module's "Reading the
//! output" section for the full column legend.

mod common;

use rlevo_environments::landscapes::eggholder::Eggholder;

fn main() {
    let dim = 10;
    let landscape = Eggholder::new(dim);
    common::showcase("Eggholder", dim, landscape.bounds(), 20.0, landscape);
}
