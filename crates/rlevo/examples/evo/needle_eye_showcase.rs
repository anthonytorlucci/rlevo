//! Needle-Eye landscape — A flat plateau punctured by a single narrow well (the eye) at the origin.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example needle_eye_showcase`.

mod common;

use rlevo_environments::landscapes::needle_eye::Needle;

fn main() {
    println!(
        "
Interpreting the output

Global optimum `f* = 1` inside a tiny hypercube of side `2×10⁻⁴` centred at
the origin; everywhere outside that \"eye\" the value is `≥ 100`. Searched
over `[-10, 10]^10`. The output is effectively binary: `best ≈ 1` means a
run threaded the eye, while `best ≈ 100` (the usual result) means it never
did. The plateau gives no gradient toward the well, so this is a near-pure
luck-of-sampling test.\n\
{:-<80}",
        "",
    );
    let dim = 10;
    let landscape = Needle::new(dim);
    common::showcase("NeedleEye", dim, landscape.bounds(), 0.4, landscape);
}
