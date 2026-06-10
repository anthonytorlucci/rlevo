//! Schwefel landscape — Deceptive: the global optimum lies far from the origin near the domain edge, away from the next-best basins.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example schwefel_showcase`.

mod common;

use rlevo_environments::landscapes::schwefel::Schwefel;

fn main() {
    println!(
        "
Interpreting the output

Global optimum `f* = −418.9829… · n ≈ −4189.83` (for `n = 10`) at
`x_i ≈ 420.97`, searched over `[-500, 500]^10`. Because cost is negative
here, `best` is \"good\" when it is *large in magnitude and negative* — i.e.
the most-negative number wins. The optimum sits at ~84% of the domain away
from the deceptive cluster of local minima near the origin, so a `best`
stuck around `0` to a few hundred negative means the search never escaped
the wrong basins.\n\
{:-<80}",
        "",
    );
    let dim = 10;
    let landscape = Schwefel::new(dim);
    common::showcase("Schwefel", dim, landscape.bounds(), 20.0, landscape);
}
