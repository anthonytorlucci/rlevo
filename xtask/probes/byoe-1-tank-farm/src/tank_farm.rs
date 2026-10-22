//! Two tanks in series, written the way a third party would write an
//! environment: from the contract crate's documentation and nothing else.
//!
//! Pump A fills tank 1, which drains into tank 2; pump B fills tank 2
//! directly, and tank 2 drains out. A tank drains in proportion to the square
//! root of its level (Torricelli's law). The task is to hold tank 2 at half
//! full without overflowing either tank. It has no horizon of its own: an
//! overflow ends it, and otherwise it runs until a time limit cuts it off.
//!
//! The observation is each tank's level over the last four steps. It is a
//! rank-2 value, and it depends on history the state does not hold.
//!
//! This file names nothing outside `rlevo::core`, so every path here is
//! written in full as `rlevo::core::...`.

use rlevo::core::action::{BoundedAction, ContinuousAction};
use rlevo::core::base::{Action, Observation};
use rlevo::core::environment::{Environment, EnvironmentError, Snapshot};
use rlevo::core::episode::EpisodeGuard;
use rlevo::core::reward::ScalarReward;

/// The name this environment is registered under.
pub const ENV_ID: &str = "tank-farm";

/// Steps of history in an observation.
pub const HISTORY: usize = 4;

/// The level tank 2 should hold, as a fraction of its capacity.
pub const TARGET_LEVEL: f32 = 0.5;

/// Level gained in a step by a pump at full rate.
const PUMP_GAIN: f32 = 0.1;

/// Outflow coefficient from tank 1 into tank 2.
const TRANSFER: f32 = 0.08;

/// Outflow coefficient from tank 2 out of the farm.
const OUTFLOW: f32 = 0.06;

/// Reward charged per unit of pumping.
const PUMP_COST: f32 = 0.05;

/// Reward charged on the step a tank overflows.
const OVERFLOW_PENALTY: f32 = 1.0;

/// What the controller sees: each tank's level over the last four steps,
/// oldest first, as fractions of capacity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TankObs {
    pub levels: [[f32; HISTORY]; 2],
}

impl TankObs {
    /// Each tank's level after the latest step.
    pub fn latest(&self) -> [f32; 2] {
        [self.levels[0][HISTORY - 1], self.levels[1][HISTORY - 1]]
    }
}

impl Observation<2> for TankObs {
    fn shape() -> [usize; 2] {
        [2, HISTORY]
    }
}

/// The environment's own state: the current level of each tank.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TankState {
    pub levels: [f32; 2],
}

/// The two pump rates, pump A then pump B, each in `[0, 1]`.
///
/// FINDING: `ContinuousAction::as_slice` returns `&[f32]`, so the rates are
/// stored as an array. A struct with two named fields, the natural way to
/// write a command, cannot return a slice without `#[repr(C)]` and `unsafe`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PumpCommand {
    pub rates: [f32; 2],
}

impl Action<1> for PumpCommand {
    fn shape() -> [usize; 1] {
        [2]
    }

    fn is_valid(&self) -> bool {
        self.rates
            .iter()
            .all(|rate| rate.is_finite() && (0.0..=1.0).contains(rate))
    }
}

impl ContinuousAction<1> for PumpCommand {
    const COMPONENTS: usize = 2;

    fn as_slice(&self) -> &[f32] {
        &self.rates
    }

    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), Self::COMPONENTS, "a command has two rates");
        Self {
            rates: [values[0], values[1]],
        }
    }
}

impl BoundedAction<1> for PumpCommand {
    fn low() -> &'static [f32] {
        &[0.0, 0.0]
    }

    fn high() -> &'static [f32] {
        &[1.0, 1.0]
    }
}

/// The two tanks, their level history and the episode guard.
#[derive(Debug)]
pub struct TankFarm {
    initial: [f32; 2],
    state: TankState,
    history: [[f32; HISTORY]; 2],
    guard: EpisodeGuard,
}

impl TankFarm {
    /// A farm whose starting levels are fixed by `seed`, for the life of the
    /// environment: tank 1 between 0.2 and 0.6, tank 2 between 0.1 and 0.5.
    pub fn with_seed(seed: u64) -> Self {
        let initial = [
            0.2 + 0.1 * (seed % 5) as f32,
            0.1 + 0.1 * ((seed / 5) % 5) as f32,
        ];
        Self {
            initial,
            state: TankState { levels: initial },
            history: initial.map(|level| [level; HISTORY]),
            guard: EpisodeGuard::new(),
        }
    }

    /// The state as it stands, for a caller that wants to check it.
    pub fn state(&self) -> TankState {
        self.state
    }

    fn observe(&self) -> TankObs {
        TankObs {
            levels: self.history,
        }
    }
}

impl Environment<2, 1> for TankFarm {
    type ObservationType = TankObs;
    type ActionType = PumpCommand;

    fn reset(&mut self) -> Result<Snapshot<TankObs>, EnvironmentError> {
        self.state = TankState {
            levels: self.initial,
        };
        self.history = self.initial.map(|level| [level; HISTORY]);
        self.guard.reset();
        Ok(Snapshot::running(self.observe(), ScalarReward::new(0.0)))
    }

    fn step(&mut self, action: PumpCommand) -> Result<Snapshot<TankObs>, EnvironmentError> {
        // A step after the episode ended is an error and changes nothing.
        self.guard.check()?;
        // So is a command outside the pumps' range: clamping it would hide a
        // controller's bug as a control problem.
        if !action.is_valid() {
            return Err(EnvironmentError::InvalidAction(format!(
                "pump rates must be finite and in [0, 1], got {:?}",
                action.rates
            )));
        }

        let [tank_1, tank_2] = self.state.levels;
        let [pump_a, pump_b] = action.rates;
        let transfer = TRANSFER * tank_1.sqrt();
        let outflow = OUTFLOW * tank_2.sqrt();
        let next = [
            tank_1 + PUMP_GAIN * pump_a - transfer,
            tank_2 + PUMP_GAIN * pump_b + transfer - outflow,
        ];
        let overflow = next.iter().any(|level| *level > 1.0);
        self.state.levels = next.map(|level| level.clamp(0.0, 1.0));
        for (history, level) in self.history.iter_mut().zip(self.state.levels) {
            history.rotate_left(1);
            history[HISTORY - 1] = level;
        }

        let mut reward =
            -(self.state.levels[1] - TARGET_LEVEL).abs() - PUMP_COST * (pump_a + pump_b);
        if overflow {
            reward -= OVERFLOW_PENALTY;
        }
        let reward = ScalarReward::new(reward);

        // An overflow is the task's own end, so the episode terminates. Nothing
        // else ends it here: a time limit is the caller's to impose.
        let snapshot = if overflow {
            Snapshot::terminated(self.observe(), reward)
        } else {
            Snapshot::running(self.observe(), reward)
        };
        self.guard.record(snapshot.status());
        Ok(snapshot)
    }
}
