//! Branin landscape — Three equal global minima over an asymmetric domain; a gentle multimodal 2-D test.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example branin_showcase`.
//!
//! # Interpreting the output
//!
//! Global optimum `f* ≈ 0.397887` shared by three equal minima (one at
//! `(π, 2.275)`), searched over the square box `[-5, 10]²`. Note the optimum is
//! *not* `0` — a `best` settling near `0.3979` is full convergence, and any of
//! the three basins counts. A gentle, well-behaved 2-D test, so most strategies
//! should reach `≈ 0.398`. See the `common` module's "Reading the output"
//! section for the full column legend.

mod common;

use rlevo_environments::landscapes::branin::Branin;

fn main() {
    let dim = 2;
    let landscape = Branin::new();
    common::showcase("Branin", dim, landscape.bounds(), 0.5, landscape);
}
