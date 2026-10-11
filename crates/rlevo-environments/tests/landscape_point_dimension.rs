//! Every landscape evaluates a slice, and rejects one of the wrong length.
//!
//! All 23 landscapes have one evaluation method with one signature,
//! [`Landscape::evaluate`] over `&[f64]`, and each checks the slice's length
//! before reading it: an n-D landscape against its `dim`, a fixed 2-D one
//! against 2. Without the check a point of another length returns a number:
//! a sum runs over whatever coordinates it was given, and a 2-D formula reads
//! `x[0]` and `x[1]` and ignores the rest.
//!
//! This file is also what pins that the impls need no cargo feature. They
//! once sat behind `bench`; it is built with the crate's default features,
//! where `bench` is off, so gating one again stops it compiling.

use std::panic::{RefUnwindSafe, catch_unwind};

use rlevo_core::fitness::Landscape;
use rlevo_core::objective::ObjectiveSense;

use rlevo_environments::landscapes::{
    ackley::Ackley, alpine1::Alpine1, branin::Branin, bukin6::Bukin6,
    concatenated_trap::ConcatenatedTrap, cross_in_tray::CrossInTray, deb1::Deb1, easom::Easom,
    eggholder::Eggholder, goldstein_price::GoldsteinPrice, griewank::Griewank,
    himmelblau::Himmelblau, lunacek_bi_rastrigin::LunacekBiRastrigin, michalewicz::Michalewicz,
    needle_eye::Needle, penalized1::Penalized1, rastrigin::Rastrigin, rosenbrock::Rosenbrock,
    rosenbrock_flat::RosenbrockFlat, schwefel::Schwefel, six_hump_camel::SixHumpCamel,
    sphere::Sphere, trefethen::Trefethen,
};

/// The message every landscape's length check panics with.
const MISMATCH: &str = "input dimension mismatch";

/// Asserts `landscape`, defined on `dim` coordinates, accepts a point of that
/// length and panics with [`MISMATCH`] on one coordinate fewer and one more.
///
/// Generic over the trait, so the call below is `Landscape::evaluate` whatever
/// inherent methods the type has.
fn assert_checks_point_length<L: Landscape + RefUnwindSafe>(name: &str, landscape: &L, dim: usize) {
    assert!(
        dim >= 2,
        "{name}: the table needs a dimension with a shorter point"
    );
    assert_eq!(
        landscape.sense(),
        ObjectiveSense::Minimize,
        "{name}: a landscape is a cost surface"
    );
    let value = landscape.evaluate(&vec![0.5; dim]);
    assert!(
        !value.is_nan(),
        "{name}: a point of the right length must evaluate"
    );

    for len in [dim - 1, dim + 1] {
        let point = vec![0.5; len];
        let Err(payload) = catch_unwind(|| landscape.evaluate(&point)) else {
            panic!(
                "{name}: a point of {len} coordinates must be rejected, the landscape has {dim}"
            );
        };
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("<a panic payload that is not a string>");
        assert!(
            message.contains(MISMATCH),
            "{name}: a point of {len} coordinates must fail the length check, got: {message}"
        );
    }
}

/// Runs [`assert_checks_point_length`] over `landscape => dim` rows.
macro_rules! assert_each_checks_point_length {
    ($($landscape:expr => $dim:expr),+ $(,)?) => {$(
        assert_checks_point_length(stringify!($landscape), &$landscape, $dim);
    )+};
}

#[test]
fn every_nd_landscape_checks_the_length_of_its_point() {
    assert_each_checks_point_length![
        Sphere::new(3).expect("valid dim") => 3,
        Ackley::new(3).expect("valid dim") => 3,
        Rastrigin::new(3).expect("valid dim") => 3,
        Rosenbrock::new(3).expect("valid dim") => 3,
        Griewank::new(3).expect("valid dim") => 3,
        Schwefel::new(3).expect("valid dim") => 3,
        Michalewicz::new(3).expect("valid dim") => 3,
        Penalized1::new(3).expect("valid dim") => 3,
        LunacekBiRastrigin::new(3).expect("valid dim") => 3,
        Needle::new(3).expect("valid dim") => 3,
        Eggholder::new(3).expect("valid dim") => 3,
        Alpine1::new(3).expect("valid dim") => 3,
        RosenbrockFlat::new(3).expect("valid dim") => 3,
        Deb1::new(2).expect("valid dim") => 2,
        ConcatenatedTrap::new(2, 2).expect("valid block shape") => 4,
    ];
}

#[test]
fn every_2d_landscape_checks_the_length_of_its_point() {
    assert_each_checks_point_length![
        Branin::new() => 2,
        Himmelblau::new() => 2,
        SixHumpCamel::new() => 2,
        Easom::new() => 2,
        GoldsteinPrice::new() => 2,
        CrossInTray::new() => 2,
        Bukin6::new() => 2,
        Trefethen::new() => 2,
    ];
}
