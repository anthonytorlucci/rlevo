//! The shipped environment the learner trains on, reached through the
//! umbrella: FrozenLake's 4×4 map, not slippery, its episodes cut by the
//! shipped `TimeLimit`.

use rlevo::envs::toy_text::frozen_lake::{
    FrozenLake, FrozenLakeAction, FrozenLakeConfig, FrozenLakeObservation, FrozenMapSpec,
    FrozenPreset,
};
use rlevo::envs::wrappers::TimeLimit;
use rlevo::prelude::Observation;

use crate::q_learner::QLearner;

/// The name the suite registers the environment under.
pub const ENV_ID: &str = "frozen-lake";
/// Steps after which `TimeLimit` ends an episode `Truncated`: short enough
/// that some training episodes reach it, so step 6 sees both endings.
pub const CAP: usize = 20;
/// Training episodes over which the learner's ε reaches its floor.
pub const SCHEDULE: usize = 1_500;

/// The environment as every trial runs it.
pub type Lake = TimeLimit<FrozenLake>;
/// The submission, on FrozenLake's observations and actions.
pub type Learner = QLearner<FrozenLakeObservation, FrozenLakeAction>;

/// The one task: the learner and the baseline both face it.
pub fn config(seed: u64) -> FrozenLakeConfig {
    FrozenLakeConfig {
        map: FrozenMapSpec::Preset(FrozenPreset::Four4x4),
        is_slippery: false,
        seed,
        ..FrozenLakeConfig::default()
    }
}

/// The environment factory the suite calls with each trial's seed.
pub fn lake(seed: u64) -> Lake {
    // A preset map always builds; the error is for a custom or random one.
    let lake = FrozenLake::with_config(config(seed)).expect("the 4×4 preset builds");
    TimeLimit::new(lake, CAP)
}

/// A state's index in the learner's table: FrozenLake's own `row × ncol +
/// col`.
fn state_of(obs: &FrozenLakeObservation) -> usize {
    usize::from(obs.state_id)
}

/// A fresh learner, sized to every state the observation can name.
pub fn learner() -> Learner {
    QLearner::new(FrozenLakeObservation::shape()[0], state_of, SCHEDULE)
}
