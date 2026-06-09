//! Deb1 landscape — A regular grid of 10^n equally deep minima from a power-five sine; no global basin gradient to exploit.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example deb1_showcase`.
//!
//! # Interpreting the output
//!
//! Global optimum `f* = −1`, attained at any of `10^n` equally deep optima
//! (each `x_i ∈ {±0.1, ±0.3, …, ±0.9}`), searched over `[-1, 1]^10`. Cost is
//! negative, so `best = −1` is full convergence. There is no global basin
//! gradient — every optimum is equivalent and the rest of the surface is
//! near-flat — so reaching `−1` is about landing on the regular grid, and a
//! `best` near `0` means the search never did. See the `common` module's
//! "Reading the output" section for the full column legend.

mod common;

use rlevo_environments::landscapes::deb1::Deb1;

fn main() {
    let dim = 10;
    let landscape = Deb1::new(dim);
    common::showcase("Deb1", dim, landscape.bounds(), 0.05, landscape);
}
