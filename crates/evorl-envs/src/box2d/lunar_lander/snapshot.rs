//! Custom snapshot type for LunarLander with shaped-reward metadata (D6).

use evorl_core::environment::{EpisodeStatus, SnapshotMetadata, Snapshot};
use evorl_core::reward::ScalarReward;

use super::observation::LunarLanderObservation;

/// Metadata key for the potential-based shaping reward component.
///
/// Consumers can read `snap.metadata().unwrap().components[METADATA_KEY_SHAPING]`
/// to obtain the raw shaping signal separate from the step reward.
pub const METADATA_KEY_SHAPING: &str = "shaping";

/// Snapshot returned by [`super::LunarLanderDiscrete`] and
/// [`super::LunarLanderContinuous`] (design decision D6).
///
/// Unlike [`evorl_core::environment::SnapshotBase`], this type overrides
/// `metadata()` to expose the potential-based shaping component for logging
/// and debugging.
///
/// **Note**: Because this is a custom `SnapshotType`, `TimeLimit<LunarLander*>`
/// does not compile (TimeLimit requires `SnapshotBase`). The step limit is
/// enforced internally via `config.max_steps`.
#[derive(Debug, Clone)]
pub struct LunarLanderSnapshot {
    /// Observation at this step.
    pub observation: LunarLanderObservation,
    /// Net reward at this step.
    pub reward: ScalarReward,
    /// Episode lifecycle status.
    pub status: EpisodeStatus,
    metadata: SnapshotMetadata,
}

impl LunarLanderSnapshot {
    /// Create a running-episode snapshot with shaping metadata.
    pub fn running(
        obs: LunarLanderObservation,
        reward: ScalarReward,
        shaping: f32,
    ) -> Self {
        Self::make(obs, reward, EpisodeStatus::Running, shaping)
    }

    /// Create a terminated-episode snapshot.
    pub fn terminated(
        obs: LunarLanderObservation,
        reward: ScalarReward,
        shaping: f32,
    ) -> Self {
        Self::make(obs, reward, EpisodeStatus::Terminated, shaping)
    }

    /// Create a truncated-episode snapshot (step limit reached).
    pub fn truncated(
        obs: LunarLanderObservation,
        reward: ScalarReward,
        shaping: f32,
    ) -> Self {
        Self::make(obs, reward, EpisodeStatus::Truncated, shaping)
    }

    fn make(
        obs: LunarLanderObservation,
        reward: ScalarReward,
        status: EpisodeStatus,
        shaping: f32,
    ) -> Self {
        Self {
            observation: obs,
            reward,
            status,
            metadata: SnapshotMetadata::new().with(METADATA_KEY_SHAPING, shaping),
        }
    }
}

impl Snapshot<1> for LunarLanderSnapshot {
    type ObservationType = LunarLanderObservation;
    type RewardType = ScalarReward;

    fn observation(&self) -> &LunarLanderObservation {
        &self.observation
    }

    fn reward(&self) -> &ScalarReward {
        &self.reward
    }

    fn status(&self) -> EpisodeStatus {
        self.status
    }

    fn metadata(&self) -> Option<&SnapshotMetadata> {
        Some(&self.metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_shaping_key_present() {
        let obs = LunarLanderObservation::default();
        let snap = LunarLanderSnapshot::running(obs, ScalarReward(1.0), 42.5);
        let meta = snap.metadata().expect("metadata must be Some");
        assert!(
            meta.components.contains_key(METADATA_KEY_SHAPING),
            "metadata must contain the shaping key"
        );
        assert!((meta.components[METADATA_KEY_SHAPING] - 42.5).abs() < 1e-6);
    }

    #[test]
    fn test_status_variants() {
        let obs = LunarLanderObservation::default();
        assert!(!LunarLanderSnapshot::running(obs.clone(), ScalarReward(0.0), 0.0)
            .status()
            .is_done());
        assert!(LunarLanderSnapshot::terminated(obs.clone(), ScalarReward(0.0), 0.0)
            .status()
            .is_terminated());
        assert!(LunarLanderSnapshot::truncated(obs, ScalarReward(0.0), 0.0)
            .status()
            .is_truncated());
    }
}
