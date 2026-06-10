//! Modified (flat) Rosenbrock landscape — Modified Rosenbrock No.01 — the classic valley flattened to stress gradient-free exploration.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example rosenbrock_flat_showcase`.

mod common;

use rlevo_environments::landscapes::rosenbrock_flat::RosenbrockFlat;

fn main() {
    println!(
        "
Interpreting the output

Global optimum `f* = 0` at `x = (1, …, 1)`. Replacing the smooth square in
standard Rosenbrock with an absolute value flattens the valley floor, so
there is even less gradient to exploit and `best` typically descends more
slowly than on the smooth variant. A `best` near `0` is full convergence; a
lingering positive plateau is the expected signature of the flattened
valley.\n\
{:-<80}",
        "",
    );
    let dim = 10;
    let landscape = RosenbrockFlat::new(dim);
    common::showcase("RosenbrockFlat", dim, landscape.bounds(), 1.0, landscape);
}
