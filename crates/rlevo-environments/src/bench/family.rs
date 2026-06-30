//! [`RecordedEnvFamily`](rlevo_benchmarks::record::RecordedEnvFamily) impls for the built-in environments.
//!
//! These tie each concrete env type to the [`EnvFamily`] its recordings
//! belong to, so a driver can derive the family once from the env type
//! (`RecordingConfig::for_env::<E>` / `E::FAMILY`) instead of restating the
//! literal at every recording / TUI call site.
//!
//! Gated by the `record` feature — it is the env-side half of the harness's
//! `record` tier and, like the rest of [`crate::bench`], an opt-in coupling
//! to `rlevo-benchmarks` rather than a property of [`Environment`] itself
//! (ADR 0007).
//!
//! [`Environment`]: rlevo_core::environment::Environment
//! [`EnvFamily`]: rlevo_benchmarks::record::EnvFamily

use rlevo_benchmarks::record::{EnvFamily, RecordedEnvFamily};

use crate::classic::cartpole::CartPole;
use crate::classic::santa_fe_ant::SantaFeAnt;
use crate::grids::EmptyEnv;
use crate::toy_text::frozen_lake::FrozenLake;

impl RecordedEnvFamily for EmptyEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

// The Santa Fe ant projects its 32×32 trail onto a `FamilyPayload::Grid`
// (it implements `GridPayloadSource`), so it records as the `Grids` family.
impl RecordedEnvFamily for SantaFeAnt {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for FrozenLake {
    const FAMILY: EnvFamily = EnvFamily::ToyText;
}

impl RecordedEnvFamily for CartPole {
    const FAMILY: EnvFamily = EnvFamily::Classic;
}

// `LunarLanderDiscrete` lives behind the `box2d` feature (rapier2d).
#[cfg(feature = "box2d")]
impl RecordedEnvFamily for crate::box2d::lunar_lander::LunarLanderDiscrete {
    const FAMILY: EnvFamily = EnvFamily::Box2d;
}

// `InvertedPendulum` and its backend trait live behind `locomotion` (rapier3d).
#[cfg(feature = "locomotion")]
impl<B: crate::locomotion::backend::LocomotionBackend> RecordedEnvFamily
    for crate::locomotion::inverted_pendulum::InvertedPendulum<B>
{
    const FAMILY: EnvFamily = EnvFamily::Locomotion;
}
