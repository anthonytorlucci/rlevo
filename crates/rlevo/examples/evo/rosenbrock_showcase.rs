//! Rosenbrock landscape — A narrow, curved parabolic valley; the global optimum sits at all-ones, hard to track because the valley floor is nearly flat.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example rosenbrock_showcase`.

mod common;

use rlevo_environments::landscapes::rosenbrock::Rosenbrock;

fn main() {
    println!(
        "
Interpreting the output

Global optimum `f* = 0` at `x = (1, …, 1)`, searched over `[-30, 30]^10`.
A `best` column near `0` means the run reached the optimal basin; expect
slow, grinding progress because the valley floor is nearly flat, and watch
for stalls at the secondary local minimum near `(-1, 1, …, 1)` that appears
for `n ≥ 4`.\n\
{:-<80}",
        "",
    );
    let dim = 10;
    let landscape = Rosenbrock::new(dim).expect("dim >= 2");
    common::showcase("Rosenbrock", dim, landscape.bounds(), 1.0, landscape);
}
