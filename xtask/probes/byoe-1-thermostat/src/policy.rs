//! The policy: a bang-bang controller. It is frozen, so it implements one
//! method of the agent seam and nothing else.

use rlevo::core::fitness::BenchableAgent;
use rlevo::prelude::Rng;

use crate::thermostat::{Boiler, RoomObs, TARGET_C};

/// Outdoor temperature below which the room loses heat fast enough that
/// cutting off exactly at the target undershoots.
const COLD_OUTSIDE_C: f32 = 5.0;

/// How far above the target the boiler stays on when it is cold outside.
const COLD_DEADBAND_C: f32 = 0.5;

/// Boiler on below the target, off at or above it.
#[derive(Debug, Clone, Copy, Default)]
pub struct BangBang;

impl BenchableAgent<RoomObs, Boiler> for BangBang {
    fn act(&mut self, obs: &RoomObs, _rng: &mut dyn Rng) -> Boiler {
        let deadband = if obs.outdoor_c < COLD_OUTSIDE_C {
            COLD_DEADBAND_C
        } else {
            0.0
        };
        if obs.indoor_c < TARGET_C + deadband {
            Boiler::On
        } else {
            Boiler::Off
        }
    }
}
