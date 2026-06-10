//! Runs every real-valued strategy in the crate on the Ackley-D10
//! landscape and prints a convergence summary.
//!
//! Ackley is multimodal with many local minima around a single global
//! basin at the origin — a step harder than Sphere.
//!
//! Run with `cargo run --release -p rlevo --example ackley_showcase`.

mod common;

use rlevo::envs::landscapes::ackley::Ackley;

fn main() {
    println!(
        "
Interpreting the output

Each strategy prints one row: `label | gens | best | mean`. Fitness is the
raw landscape cost under the minimization convention, so **lower is better**.
`best` is `best_fitness_ever` — the lowest cost seen by any individual across
all generations (a rolling minimum) — and `mean` is the average cost of the
final generation. A small `best`–`mean` gap means the whole population
converged; a large gap suggests premature convergence, with a few good
individuals leading a still spread-out population. Values print in scientific
notation, so `e-6` is near-converged and `e+2` is far off.

Ackley has global optimum `f* = 0` at the origin, searched over
`[-32.768, 32.768]^10`. The many shallow local minima ring a single global
basin, so the difficulty is escaping the outer ripples to reach that basin —
a `best` near `0` is full convergence, while a `best` stuck around `2`–`5`
means a run never crossed into the central funnel.\n\n\
{:-<80}",
        "",
    );
    let dim = 10;
    let landscape = Ackley::new(dim);
    common::showcase("Ackley", dim, landscape.bounds(), 0.4, landscape);
}
