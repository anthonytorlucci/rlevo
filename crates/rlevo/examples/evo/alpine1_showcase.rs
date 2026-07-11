//! Alpine1 landscape — Non-smooth |x sin x + 0.1 x| sum; many sharp-edged local minima around the origin.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example alpine1_showcase`.

mod common;

use rlevo_environments::landscapes::alpine1::Alpine1;

fn main() {
    println!(
        "
Interpreting the output

Global optimum `f* = 0` at the origin, searched over `[-10, 10]^10`. A
`best` near `0` is full convergence. The surface is non-smooth with many
sharp-edged local minima (roughly eight kinks per axis), so progress is
jagged and a `best` resting at a small positive value usually means a run
settled in one of the off-origin kinks. See the `common` module's \"Reading
the output\" section for the full column legend.\n\
{:-<80}",
        "",
    );
    let dim = 10;
    let landscape = Alpine1::new(dim).expect("dim >= 1");
    common::showcase("Alpine1", dim, landscape.bounds(), 0.4, landscape);
}
