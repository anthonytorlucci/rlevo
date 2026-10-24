//! The landscape, wrapped for the engine. The evolutionary loop scores a
//! whole population at once, and `FromLandscape` scores each row of it with
//! the landscape's own `evaluate`, in the landscape's own direction.

use rlevo::burn::tensor::{Device, Tensor, TensorData};
use rlevo::core::fitness::Landscape;
use rlevo::core::objective::ObjectiveSense;
use rlevo::evo::fitness::{BatchFitnessFn, FromLandscape};

use crate::well_siting::{DIM, WellSiting};

/// The landscape as the loop takes it.
pub fn fitness() -> FromLandscape<WellSiting> {
    FromLandscape::new(WellSiting)
}

/// Sites the check scores: the village square, the cheapest site, and a
/// far corner.
const SITES: [[f32; DIM]; 3] = [[0.0, 0.0], [0.5, 0.6], [-4.0, 4.0]];

/// The landscape file, as compiled, so the check can say what it does not
/// contain.
const LANDSCAPE_SOURCE: &str = include_str!("well_siting.rs");

/// The adapter must minimise without being told to, and must score each
/// site at the landscape's own cost.
pub fn check() -> Result<String, String> {
    if LANDSCAPE_SOURCE.contains("ObjectiveSense") || LANDSCAPE_SOURCE.contains("fn sense") {
        return Err(
            "well_siting.rs names its direction; the check is that it need not".to_string(),
        );
    }
    let mut fitness = fitness();
    if fitness.sense() != ObjectiveSense::Minimize {
        return Err(format!(
            "FromLandscape reports {:?} for a landscape that declares no direction",
            fitness.sense()
        ));
    }

    let device = Device::flex();
    let batch =
        Tensor::<2>::from_data(TensorData::new(SITES.concat(), [SITES.len(), DIM]), &device);
    let costs = fitness
        .evaluate_batch(&batch, &device)
        .into_data()
        .try_into_vec::<f32>()
        .map_err(|e| format!("the batch's costs are not readable as f32: {e:?}"))?;
    if costs.len() != SITES.len() {
        return Err(format!(
            "{} sites scored as {} costs",
            SITES.len(),
            costs.len()
        ));
    }
    for (site, &cost) in SITES.iter().zip(&costs) {
        let own = WellSiting.evaluate(&[f64::from(site[0]), f64::from(site[1])]);
        if (f64::from(cost) - own).abs() > 1e-5 * own.max(1.0) {
            return Err(format!(
                "the site {site:?} scores {cost} through FromLandscape and {own} on its own"
            ));
        }
    }
    Ok(format!(
        "FromLandscape scores {} sites at the landscape's own cost and minimises; \
         well_siting.rs names no direction",
        SITES.len()
    ))
}
