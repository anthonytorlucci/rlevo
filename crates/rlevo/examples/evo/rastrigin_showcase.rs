//! Runs every real-valued strategy in the crate on the Rastrigin-D10
//! landscape and prints a convergence summary.
//!
//! Rastrigin is highly multimodal with a regular grid of local minima
//! superimposed on a Sphere-like envelope — a harder convergence test
//! than either Sphere or Ackley.
//!
//! Run with `cargo run --release -p rlevo --example rastrigin_showcase`.

mod common;

use rlevo::envs::landscapes::rastrigin::Rastrigin;

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

Rastrigin has global optimum `f* = 0` at the origin, searched over
`[-5.12, 5.12]^10`. A regular grid of deep local minima sits on a Sphere-like
envelope, so each minimum is a genuine trap rather than a shallow ripple.
A `best` near `0` is full convergence; a `best` resting at an integer-ish
plateau (each trapped coordinate adds roughly `A = 10` to the cost) means a
run settled into one of the off-origin minima.\n\n\
{:-<80}",
        "",
    );
    let dim = 10;
    let landscape = Rastrigin::new(dim);
    common::showcase("Rastrigin", dim, landscape.bounds(), 0.4, landscape);
}
