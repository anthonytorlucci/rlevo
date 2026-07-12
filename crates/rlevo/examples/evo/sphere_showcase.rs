//! Runs every real-valued strategy in the crate on the Sphere-D10
//! landscape and prints a convergence summary.
//!
//! Run with `cargo run --release -p rlevo --example sphere_showcase`.

mod common;

use rlevo::envs::landscapes::sphere::Sphere;

fn main() {
    println!(
        "
Interpreting the output

Each strategy prints one row: `label | gens | best | mean`. Fitness is the
raw landscape cost under the minimization convention, so **lower is better**.
`best` is `best_fitness_ever` — the lowest cost seen by any individual across
all generations (a rolling minimum) — and `mean` is the average cost of the
final generation. A small `best`–`mean` gap means the whole population
converged; a large gap means a few good individuals lead a still spread-out
population. Values print in scientific notation, so `e-6` is near-converged
and `e+2` is far off.

Sphere is a smooth, unimodal, convex bowl with global optimum `f* = 0` at the
origin, searched over `[-5.12, 5.12]^10`. It is the easy baseline: with no
local minima to trap search, every strategy should drive `best` close to `0`,
so this run mainly shows their relative *convergence speed* rather than
whether they find the optimum at all.\n\n\
{:-<80}",
        "",
    );
    let dim = 10;
    let landscape = Sphere::new(dim).expect("dim >= 1");
    common::showcase("Sphere", dim, landscape.bounds(), 0.4, landscape);
}
