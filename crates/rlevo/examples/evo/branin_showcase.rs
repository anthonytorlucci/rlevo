//! Branin landscape — Three equal global minima over an asymmetric domain; a gentle multimodal 2-D test.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example branin_showcase`.

mod common;

use rlevo_environments::landscapes::branin::Branin;

fn main() {
    println!(
        "
Interpreting the output

Global optimum `f* ≈ 0.397887` shared by three equal minima —
`(−π, 12.275)`, `(π, 2.275)`, `(3π, 2.475)` — searched over the square hull
`[-5, 15]²` of Branin's asymmetric published domain (`x₁ ∈ [-5, 10]`,
`x₂ ∈ [0, 15]`). The hull is what makes `(−π, 12.275)` reachable at all: its
`x₂` exceeds `10`, so a `[-5, 10]²` box would leave that basin outside the
search space. Note the optimum is *not* `0` — a `best` settling near `0.3979`
is full convergence, and any of the three basins counts. A gentle,
well-behaved 2-D test, so most strategies should reach `≈ 0.398`.\n\
{:-<80}",
        "",
    );
    let dim = 2;
    let landscape = Branin::new();
    // `mutation_sigma` is an ABSOLUTE step size, not a fraction of the search
    // box, so it must be rescaled whenever `bounds()` changes or the mutation
    // strength silently weakens relative to the box. Widening the box from span
    // 15 (`[-5, 10]`) to span 20 (`[-5, 15]`) rescales the original 0.5 by
    // 20/15 → 0.67, preserving the hand-tuned sigma/span ratio.
    common::showcase("Branin", dim, landscape.bounds(), 0.67, landscape);
}
