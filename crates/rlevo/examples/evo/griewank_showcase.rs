//! Griewank landscape — A wide quadratic envelope dimpled by a product of cosines; many regular local minima over a large [-600, 600] domain.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example griewank_showcase`.

mod common;

use rlevo_environments::landscapes::griewank::Griewank;

fn main() {
    println!(
        "
Interpreting the output

Global optimum `f* = 0` at the origin, searched over `[-600, 600]^10`. A
`best` near `0` is full convergence. The dense cosine lattice creates many
near-equal local minima, so a `best`–`mean` gap that stays open after 500
generations signals the population trapped across several of them rather
than settling in the central bowl.\n\
{:-<80}",
        "",
    );
    let dim = 10;
    let landscape = Griewank::new(dim);
    common::showcase("Griewank", dim, landscape.bounds(), 20.0, landscape);
}
