//! Lunacek landscape — Two offset Rastrigin funnels: a deep global funnel competing with a broad deceptive one.
//!
//! Runs every real-valued strategy in `rlevo-evolution` and prints a
//! convergence summary. Run with:
//! `cargo run --release -p rlevo --example lunacek_showcase`.

mod common;

use rlevo_environments::landscapes::lunacek_bi_rastrigin::LunacekBiRastrigin;

fn main() {
    println!(
        "
Interpreting the output

Global optimum `f* = 0` at `x_i = μ₁ = 2.5`, searched over `[-5.12, 5.12]^10`.
A `best` near `0` is full convergence. The landscape pairs a narrow global
funnel against a broad deceptive one, so a `best` stuck at a positive plateau
signals the search committed to the wrong (deceptive) funnel — the central
difficulty this benchmark is designed to expose.\n\
{:-<80}",
        "",
    );
    let dim = 10;
    let landscape = LunacekBiRastrigin::new(dim).expect("dim >= 2");
    common::showcase("Lunacek", dim, landscape.bounds(), 0.2, landscape);
}
