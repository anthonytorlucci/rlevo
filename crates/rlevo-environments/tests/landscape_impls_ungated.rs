//! Guards that the [`Landscape`] impls are reachable with **no cargo feature
//! enabled** — in particular without `bench`.
//!
//! The impls used to live in `bench::landscape`, inside a module gated on
//! `bench = ["dep:rlevo-benchmarks"]`. Nothing in them names the harness: they
//! reference only [`Landscape`] and the landscape structs themselves. The
//! effect was that `FromLandscape::new(Rastrigin::new(10)?)` — a pure
//! evolutionary-search call with no benchmarking in it anywhere — did not
//! compile until the caller enabled a feature named after the harness and
//! pulled two extra crates in behind it.
//!
//! `#![cfg(not(feature = "bench"))]` is the whole point of this file: it makes
//! the test body compile *only* in the configuration the defect described, so
//! re-gating the impls is a compile error here rather than a silent
//! regression. It runs under `cargo xtask test fast`, whose default feature
//! set for this crate leaves `bench` off.
#![cfg(not(feature = "bench"))]

use rlevo_core::fitness::Landscape;
use rlevo_environments::landscapes::{
    branin::Branin, rastrigin::Rastrigin, rosenbrock::Rosenbrock, sphere::Sphere,
};

/// Calls go through the trait, not the inherent method — an inherent
/// `Sphere::evaluate` would satisfy the same syntax, so the binding is made
/// explicit to keep this test honest about what it pins.
fn via_trait<L: Landscape>(landscape: &L, x: &[f64]) -> f64 {
    Landscape::evaluate(landscape, x)
}

/// Asserts a landscape reads its known optimum. The tolerance is not
/// scepticism about these three values — each is exactly representable and
/// each evaluates to a literal zero — it is that `clippy::float_cmp` rejects
/// the exact form crate-wide, and an epsilon that would hide a real regression
/// here would have to be enormous.
fn assert_optimum<L: Landscape>(landscape: &L, x: &[f64], expected: f64, name: &str) {
    let f = via_trait(landscape, x);
    assert!(
        (f - expected).abs() < 1e-12,
        "{name} at its known optimum should read {expected}, got {f}",
    );
}

#[test]
fn n_dimensional_landscapes_implement_landscape_without_bench() {
    assert_optimum(
        &Sphere::new(3).expect("dim >= 1"),
        &[0.0, 0.0, 0.0],
        0.0,
        "Sphere",
    );
    assert_optimum(
        &Rastrigin::new(2).expect("dim >= 1"),
        &[0.0, 0.0],
        0.0,
        "Rastrigin",
    );
    assert_optimum(
        &Rosenbrock::new(2).expect("dim >= 2"),
        &[1.0, 1.0],
        0.0,
        "Rosenbrock",
    );
}

/// The 2-D landscapes adapt a `(x1, x2)` evaluator to the slice signature, so
/// they are a separate impl shape and worth pinning separately.
#[test]
fn two_dimensional_landscapes_implement_landscape_without_bench() {
    let branin = Branin::new();
    // One of Branin's three equal global minima; f* = 0.397887.
    let f = via_trait(&branin, &[std::f64::consts::PI, 2.275]);
    assert!(
        (f - 0.397_887).abs() < 1e-4,
        "Branin at a known global minimum should read f* = 0.397887, got {f}",
    );
}
