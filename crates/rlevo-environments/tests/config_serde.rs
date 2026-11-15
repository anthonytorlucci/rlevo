//! Every shipped environment config serializes, and reads back unchanged. A
//! config is the input to a task fingerprint
//! (`rlevo_harness::task::task_fingerprint`), which walks it with serde, so
//! one that does not implement `Serialize` cannot be registered with
//! `Suite::with_task`; reading one back is for configs stored with a run.
//! JSON is checked for serializing; the round trip goes through bincode,
//! since JSON has no infinity and `InvertedDoublePendulumConfig`'s default
//! height range is open above (`serde_json` writes `f32::INFINITY` as `null`
//! and cannot read it back).

use serde::Serialize;
use serde::de::DeserializeOwned;

use rlevo_environments::classic::acrobot::AcrobotConfig;
use rlevo_environments::classic::cartpole::CartPoleConfig;
use rlevo_environments::classic::mountain_car::MountainCarConfig;
use rlevo_environments::classic::mountain_car_continuous::MountainCarContinuousConfig;
use rlevo_environments::classic::pendulum::PendulumConfig;
use rlevo_environments::toy_text::blackjack::BlackjackConfig;
use rlevo_environments::toy_text::cliff_walking::CliffWalkingConfig;
use rlevo_environments::toy_text::frozen_lake::{FrozenLakeConfig, FrozenMapSpec};
use rlevo_environments::toy_text::taxi::TaxiConfig;

/// Serializes `config` to JSON, then round-trips it through bincode: the
/// bytes written before and after must be equal.
fn round_trips<C: Serialize + DeserializeOwned>(name: &str, config: &C) {
    serde_json::to_string(config).unwrap_or_else(|e| panic!("{name} to JSON: {e}"));
    let cfg = bincode::config::standard();
    let first = bincode::serde::encode_to_vec(config, cfg).unwrap();
    let (back, _): (C, usize) = bincode::serde::decode_from_slice(&first, cfg)
        .unwrap_or_else(|e| panic!("{name} back: {e}"));
    let second = bincode::serde::encode_to_vec(&back, cfg).unwrap();
    assert_eq!(first, second, "{name} changed on the way back");
}

#[test]
fn classic_and_toy_text_configs_round_trip() {
    round_trips("Acrobot", &AcrobotConfig::default());
    round_trips("CartPole", &CartPoleConfig::default());
    round_trips("MountainCar", &MountainCarConfig::default());
    round_trips(
        "MountainCarContinuous",
        &MountainCarContinuousConfig::default(),
    );
    round_trips("Pendulum", &PendulumConfig::default());
    round_trips("Blackjack", &BlackjackConfig::default());
    round_trips("CliffWalking", &CliffWalkingConfig::default());
    round_trips("FrozenLake", &FrozenLakeConfig::default());
    round_trips(
        "FrozenLake custom map",
        &FrozenLakeConfig {
            map: FrozenMapSpec::Custom(vec!["SF".into(), "HG".into()]),
            ..FrozenLakeConfig::default()
        },
    );
    round_trips("Taxi", &TaxiConfig::default());
}

#[cfg(feature = "locomotion")]
#[test]
fn locomotion_configs_round_trip() {
    use rlevo_environments::locomotion::inverted_double_pendulum::InvertedDoublePendulumConfig;
    use rlevo_environments::locomotion::inverted_pendulum::InvertedPendulumConfig;
    use rlevo_environments::locomotion::reacher::ReacherConfig;
    use rlevo_environments::locomotion::swimmer::SwimmerConfig;

    round_trips(
        "InvertedDoublePendulum",
        &InvertedDoublePendulumConfig::default(),
    );
    round_trips("InvertedPendulum", &InvertedPendulumConfig::default());
    round_trips("Reacher", &ReacherConfig::default());
    round_trips("Swimmer", &SwimmerConfig::default());
}

/// A gear vector serializes as its ratios, and one of the wrong length is
/// refused rather than truncated or padded.
#[cfg(feature = "locomotion")]
#[test]
fn a_gear_is_a_list_of_its_ratios() {
    use rlevo_environments::locomotion::common::Gear;

    let gear = Gear::new([150.0_f32, 200.0]);
    assert_eq!(serde_json::to_string(&gear).unwrap(), "[150.0,200.0]");
    assert_eq!(
        serde_json::from_str::<Gear<2>>("[150.0,200.0]").unwrap(),
        gear
    );
    assert!(serde_json::from_str::<Gear<2>>("[1.0]").is_err());
    assert!(serde_json::from_str::<Gear<2>>("[1.0,2.0,3.0]").is_err());
}
