//! [`Landscape`] impls for the
//! numerical landscapes in [`crate::landscapes`].
//!
//! Feature-gated under `bench` so the trait impls compile only when the
//! harness adapter surface is in use.
//!
//! Each impl delegates to the landscape struct's own `evaluate` method, which
//! returns the raw scalar cost value (lower is better, following the
//! minimization convention used throughout `rlevo-evolution`).  These impls
//! make the landscapes usable directly with `rlevo-evolution`'s
//! `FromLandscape` adapter for evolutionary search benchmarking.

use rlevo_core::fitness::Landscape;

use crate::landscapes::{
    ackley::Ackley, alpine1::Alpine1, branin::Branin, bukin6::Bukin6, cross_in_tray::CrossInTray,
    deb1::Deb1, easom::Easom, eggholder::Eggholder, goldstein_price::GoldsteinPrice,
    griewank::Griewank, himmelblau::Himmelblau, lunacek_bi_rastrigin::LunacekBiRastrigin,
    michalewicz::Michalewicz, needle_eye::Needle, penalized1::Penalized1, rastrigin::Rastrigin,
    rosenbrock::Rosenbrock, rosenbrock_flat::RosenbrockFlat, schwefel::Schwefel,
    six_hump_camel::SixHumpCamel, sphere::Sphere, trefethen::Trefethen,
};

// ---------------------------------------------------------------------------
// n-dimensional landscapes — evaluate a full coordinate slice directly.
// ---------------------------------------------------------------------------

impl Landscape for Sphere {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Sphere::evaluate(self, x)
    }
}

impl Landscape for Ackley {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Ackley::evaluate(self, x)
    }
}

impl Landscape for Rastrigin {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Rastrigin::evaluate(self, x)
    }
}

impl Landscape for Rosenbrock {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Rosenbrock::evaluate(self, x)
    }
}

impl Landscape for Griewank {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Griewank::evaluate(self, x)
    }
}

impl Landscape for Schwefel {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Schwefel::evaluate(self, x)
    }
}

impl Landscape for Michalewicz {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Michalewicz::evaluate(self, x)
    }
}

impl Landscape for Penalized1 {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Penalized1::evaluate(self, x)
    }
}

impl Landscape for LunacekBiRastrigin {
    fn evaluate(&self, x: &[f64]) -> f64 {
        LunacekBiRastrigin::evaluate(self, x)
    }
}

impl Landscape for Deb1 {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Deb1::evaluate(self, x)
    }
}

impl Landscape for Needle {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Needle::evaluate(self, x)
    }
}

impl Landscape for Eggholder {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Eggholder::evaluate(self, x)
    }
}

impl Landscape for Alpine1 {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Alpine1::evaluate(self, x)
    }
}

impl Landscape for RosenbrockFlat {
    fn evaluate(&self, x: &[f64]) -> f64 {
        RosenbrockFlat::evaluate(self, x)
    }
}

// ---------------------------------------------------------------------------
// Strictly 2-D landscapes — adapt the `(x1, x2)` evaluator to a slice.
//
// `FromLandscape` evaluates row by row, so a genome of dimension 2 arrives as a
// two-element slice. The assert guards against accidentally driving these with
// a higher genome dimension.
// ---------------------------------------------------------------------------

impl Landscape for Branin {
    fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), 2, "Branin is a 2-D landscape");
        Branin::evaluate(self, x[0], x[1])
    }
}

impl Landscape for Himmelblau {
    fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), 2, "Himmelblau is a 2-D landscape");
        Himmelblau::evaluate(self, x[0], x[1])
    }
}

impl Landscape for SixHumpCamel {
    fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), 2, "Six-Hump Camel is a 2-D landscape");
        SixHumpCamel::evaluate(self, x[0], x[1])
    }
}

impl Landscape for Easom {
    fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), 2, "Easom is a 2-D landscape");
        Easom::evaluate(self, x[0], x[1])
    }
}

impl Landscape for GoldsteinPrice {
    fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), 2, "Goldstein-Price is a 2-D landscape");
        GoldsteinPrice::evaluate(self, x[0], x[1])
    }
}

impl Landscape for CrossInTray {
    fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), 2, "Cross-in-Tray is a 2-D landscape");
        CrossInTray::evaluate(self, x[0], x[1])
    }
}

impl Landscape for Bukin6 {
    fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), 2, "Bukin No.06 is a 2-D landscape");
        Bukin6::evaluate(self, x[0], x[1])
    }
}

impl Landscape for Trefethen {
    fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), 2, "Trefethen is a 2-D landscape");
        Trefethen::evaluate(self, x[0], x[1])
    }
}
