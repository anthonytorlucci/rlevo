//! Trefethen landscape — Five superimposed frequencies producing a dense rugged landscape (SIAM 100-digit challenge).
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example trefethen_showcase`.

mod common;

use rlevo_environments::landscapes::trefethen::Trefethen;

fn main() {
    println!(
        "
Interpreting the output

Global optimum `f* ≈ −3.3069` at `(−0.0244, 0.2106)`, searched over the
EA-friendly extended box `[-4.5, 4.5]²`. Cost is negative, so a more-negative
`best` is better and `best ≈ −3.31` is full convergence. Five superimposed
frequencies make the surface densely rugged, so a `best` only reaching the
`−2` range means the search settled in one of the many shallower dips.\n\n
{:-<80}",
        "",
    );
    let dim = 2;
    let landscape = Trefethen::new();
    common::showcase("Trefethen", dim, landscape.bounds(), 0.2, landscape);
}
