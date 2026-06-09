//! Cross-in-Tray landscape — A non-smooth absolute-value surface with four symmetric global minima.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example cross_in_tray_showcase`.
//!
//! # Interpreting the output
//!
//! Four equal global minima of value `f* ≈ −2.062612` at the sign combinations
//! of `(±1.3494, ±1.3494)`, searched over `[-15, 15]²`. Cost is negative, so a
//! more-negative `best` is better and `best ≈ −2.0626` is full convergence.
//! The non-smooth absolute-value surface is otherwise near-flat, so a `best`
//! close to `0` means the search never reached the central cross. See the
//! `common` module's "Reading the output" section for the full column legend.

mod common;

use rlevo_environments::landscapes::cross_in_tray::CrossInTray;

fn main() {
    let dim = 2;
    let landscape = CrossInTray::new();
    common::showcase("CrossInTray", dim, landscape.bounds(), 0.6, landscape);
}
