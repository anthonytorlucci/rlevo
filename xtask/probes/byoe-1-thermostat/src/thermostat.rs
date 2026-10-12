//! A day of heating control, written the way a third party would write an
//! environment: from the contract crate's documentation and nothing else.
//!
//! One episode is 24 hourly steps. The boiler is on or off; the room loses
//! heat to the outside in proportion to the temperature difference (Newton
//! cooling); the reward is the distance from the target, with a charge for
//! running the boiler.
//!
//! This file names nothing outside `rlevo::core`. The `consumer-core` probe
//! builds a copy of it against `rlevo-core` alone, so every path here is
//! written in full as `rlevo::core::...`.

use rlevo::core::base::{Action, Observation, State};
use rlevo::core::environment::{Environment, EnvironmentError, Snapshot, SnapshotBase};
use rlevo::core::episode::EpisodeGuard;
use rlevo::core::reward::ScalarReward;

/// The name this environment is registered under.
pub const ENV_ID: &str = "thermostat";

/// Steps in one episode: the hours of a day.
pub const HORIZON: usize = 24;

/// The temperature the room should hold, in degrees Celsius.
pub const TARGET_C: f32 = 21.0;

/// Degrees gained in an hour with the boiler on.
const HEAT_PER_HOUR: f32 = 3.0;

/// Fraction of the indoor-outdoor difference lost in an hour.
const LOSS_RATE: f32 = 0.25;

/// Reward charged for an hour with the boiler on.
const BOILER_COST: f32 = 0.4;

/// What the controller sees: both temperatures, in degrees Celsius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoomObs {
    pub indoor_c: f32,
    pub outdoor_c: f32,
}

impl Observation<1> for RoomObs {
    fn shape() -> [usize; 1] {
        [2]
    }
}

/// The environment's own state: the indoor temperature and the hour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoomState {
    pub indoor_c: f32,
    pub hour: usize,
}

impl State<1> for RoomState {
    fn shape() -> [usize; 1] {
        [2]
    }

    fn is_valid(&self) -> bool {
        self.indoor_c.is_finite() && self.hour <= HORIZON
    }
}

/// The boiler switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boiler {
    Off,
    On,
}

impl Action<1> for Boiler {
    fn shape() -> [usize; 1] {
        [2]
    }

    fn is_valid(&self) -> bool {
        true
    }
}

/// The room, its weather and the episode guard.
#[derive(Debug)]
pub struct Thermostat {
    outdoor_c: f32,
    state: RoomState,
    guard: EpisodeGuard,
}

impl Thermostat {
    /// A room whose outdoor temperature is fixed by `seed`, between 2 and 12
    /// degrees, for the life of the environment.
    pub fn with_seed(seed: u64) -> Self {
        let outdoor_c = 2.0 + (seed % 11) as f32;
        Self {
            outdoor_c,
            state: RoomState {
                indoor_c: outdoor_c,
                hour: 0,
            },
            guard: EpisodeGuard::new(),
        }
    }

    /// The state as it stands, for a caller that wants to check it.
    pub fn state(&self) -> RoomState {
        self.state
    }

    fn observe(&self) -> RoomObs {
        RoomObs {
            indoor_c: self.state.indoor_c,
            outdoor_c: self.outdoor_c,
        }
    }
}

impl Environment<1, 1, 1> for Thermostat {
    type StateType = RoomState;
    type ObservationType = RoomObs;
    type ActionType = Boiler;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, RoomObs, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = RoomState {
            indoor_c: self.outdoor_c,
            hour: 0,
        };
        self.guard.reset();
        Ok(SnapshotBase::running(
            self.observe(),
            ScalarReward::new(0.0),
        ))
    }

    fn step(&mut self, action: Boiler) -> Result<Self::SnapshotType, EnvironmentError> {
        // A step after the episode ended is an error and changes nothing.
        self.guard.check()?;

        let heat = match action {
            Boiler::On => HEAT_PER_HOUR,
            Boiler::Off => 0.0,
        };
        let loss = LOSS_RATE * (self.state.indoor_c - self.outdoor_c);
        self.state.indoor_c += heat - loss;
        self.state.hour += 1;

        let cost = match action {
            Boiler::On => BOILER_COST,
            Boiler::Off => 0.0,
        };
        let reward = ScalarReward::new(-(self.state.indoor_c - TARGET_C).abs() - cost);

        // The horizon is the task's own, so the episode terminates; it is
        // never truncated.
        let snapshot = if self.state.hour >= HORIZON {
            SnapshotBase::terminated(self.observe(), reward)
        } else {
            SnapshotBase::running(self.observe(), reward)
        };
        self.guard.record(snapshot.status());
        Ok(snapshot)
    }
}
