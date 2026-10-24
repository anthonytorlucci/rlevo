//! The landscape: where to dig one well for five houses.
//!
//! Every household walks to the well, so the cost of a site is the mean
//! squared distance from it to the houses. The cheapest site is their
//! centroid, which makes the surface's minimum known in closed form.
//!
//! The file implements `evaluate` and nothing else: a landscape is a cost
//! surface, and the trait's default direction is to minimise it. Every
//! `rlevo` path here is written in full as `rlevo::core::...`, because a
//! landscape needs the contract crate alone, and the walk checks that the
//! file names nothing more.

use rlevo::core::fitness::Landscape;

/// Where the houses stand, in kilometres east and north of the village
/// square.
pub const HOUSES: [[f64; 2]; 5] = [
    [-3.0, 1.5],
    [-1.0, -2.5],
    [0.5, 3.0],
    [2.0, -1.0],
    [4.0, 2.0],
];

/// A site's coordinates: east, then north.
pub const DIM: usize = 2;

/// The siting problem. It holds nothing: the houses do not move.
#[derive(Debug, Clone, Copy, Default)]
pub struct WellSiting;

impl WellSiting {
    /// The cost at the cheapest site, the houses' centroid.
    pub fn minimum_cost() -> f64 {
        cost_at(&centroid())
    }
}

impl Landscape for WellSiting {
    fn evaluate(&self, x: &[f64]) -> f64 {
        cost_at(x)
    }
}

#[allow(clippy::cast_precision_loss)]
fn centroid() -> [f64; DIM] {
    let n = HOUSES.len() as f64;
    let east = HOUSES.iter().map(|house| house[0]).sum::<f64>() / n;
    let north = HOUSES.iter().map(|house| house[1]).sum::<f64>() / n;
    [east, north]
}

/// The mean squared distance from `site` to the houses.
#[allow(clippy::cast_precision_loss)]
fn cost_at(site: &[f64]) -> f64 {
    let total: f64 = HOUSES
        .iter()
        .map(|house| (site[0] - house[0]).powi(2) + (site[1] - house[1]).powi(2))
        .sum();
    total / HOUSES.len() as f64
}
