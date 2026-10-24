//! The policy: pump A at a constant rate, pump B proportional to tank 2's
//! distance from the target. It is frozen, so it implements one method of
//! the agent seam and nothing else.

use rlevo::prelude::{Agent, Rng};

use crate::tank_farm::{PumpCommand, TARGET_LEVEL, TankObs};

/// Pump A's constant rate. Tank 1 settles where its outflow matches it.
const PUMP_A: f32 = 0.3;

/// Pump B's rate with tank 2 at the target: about what holds it there
/// against the flow from tank 1.
const FEEDFORWARD: f32 = 0.12;

/// Pump B's change in rate per unit of tank 2's distance from the target.
const GAIN: f32 = 2.0;

/// Holds tank 2 near the target, reading only the latest levels.
#[derive(Debug, Clone, Copy, Default)]
pub struct Proportional;

impl Agent<TankObs, PumpCommand> for Proportional {
    fn act(&mut self, obs: &TankObs, _rng: &mut dyn Rng) -> PumpCommand {
        let [_, tank_2] = obs.latest();
        let pump_b = (FEEDFORWARD + GAIN * (TARGET_LEVEL - tank_2)).clamp(0.0, 1.0);
        PumpCommand {
            rates: [PUMP_A, pump_b],
        }
    }
}
