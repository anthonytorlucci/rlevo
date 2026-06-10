//! Six-Hump Camel landscape — Six local minima, two of them global; symmetric about the origin.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example six_hump_camel_showcase`.

mod common;

use rlevo_environments::landscapes::six_hump_camel::SixHumpCamel;

fn main() {
    println!(
        "
Interpreting the output

Global optimum `f* ≈ −1.031628` shared by two equivalent minima at
`(±0.0898, ∓0.7127)`, searched over `[-2, 2]²`. Cost is negative, so a more-
negative `best` is better and `best ≈ −1.0316` is full convergence. Of the
six local minima, four are sub-optimal traps, so a `best` near `−1.0` but
not quite reaching `−1.0316` usually means a run landed in one of those.\n\n
{:-<80}",
        "",
    );
    let dim = 2;
    let landscape = SixHumpCamel::new();
    common::showcase("SixHumpCamel", dim, landscape.bounds(), 0.1, landscape);
}
