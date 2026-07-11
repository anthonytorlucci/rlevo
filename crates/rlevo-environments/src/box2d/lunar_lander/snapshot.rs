//! Snapshot type and shaping-metadata helper for LunarLander (D6).

use rlevo_core::environment::{SnapshotBase, SnapshotMetadata};
use rlevo_core::reward::ScalarReward;

use super::observation::LunarLanderObservation;

/// Metadata key for the **absolute potential-based shaping potential** Φ(t).
///
/// # What the value *is*
///
/// `snap.metadata().unwrap().components[METADATA_KEY_SHAPING]` is Φ(t) — the
/// potential of the state observed at this step, as computed by
/// `LunarLanderCore::shaping`:
///
/// ```text
/// Φ(obs) = -100·dist_to_helipad - 100·speed - 100·|angle|
///          + 10·leg1_contact + 10·leg2_contact
/// ```
///
/// It is **not** the shaping *reward*. In the potential-based reward-shaping
/// (PBRS) framework of Ng, Harada & Russell (1999), *"Policy Invariance Under
/// Reward Transformations"* (ICML 1999, pp. 278–287), Φ is a scalar field over
/// states and the shaping reward is the *difference*
/// `F(s, a, s′) = γ·Φ(s′) − Φ(s)`. Φ is a potential; F is a reward. This key
/// carries Φ.
///
/// # Reconstructing the shaping reward
///
/// The quantity that actually enters `reward` is the potential difference
///
/// ```text
/// F(t) = Φ(t) − Φ(t−1)
/// ```
///
/// (implicit γ = 1 — the standard episodic/undiscounted instantiation; cf.
/// Grześ 2017, where the γ factor "simply disappears" in the undiscounted
/// case). Consumers reconstruct F by subtracting the previous step's value of
/// this key from the current one; a single snapshot in isolation cannot give
/// you F.
///
/// # Caveat: terminal steps
///
/// On a **terminated** step the reward is *replaced* wholesale by ±100
/// (+100 soft landing, −100 crash / out-of-bounds), discarding that step's
/// shaping delta and control cost — this mirrors Gymnasium's
/// `reward = ±100` assignment. So on a terminal snapshot, Φ(t) contributed
/// **nothing** to that step's reward, and the naive reconstruction
/// `Φ(t) − Φ(t−1)` is wrong *precisely there*. Analysis code that decomposes
/// episode returns must special-case the terminal step.
///
/// # Deliberate deviation from strict policy invariance
///
/// Because Φ(s_terminal) ≠ 0 here (a crashed lander still has a large negative
/// potential), this is not a strictly policy-invariant PBRS instantiation:
/// Grześ (2017) shows episodic policy invariance requires Φ(s_terminal) = 0.
/// This is a **deliberate deviation kept for Gymnasium reward parity**, not a
/// bug — rlevo reproduces `gymnasium.envs.box2d.lunar_lander` returns exactly.
/// Do not "fix" the reward function to restore invariance.
// The elided lifetime in a `const` is `'static` (const lifetime elision), so
// this is exactly `&'static str` — the type `SnapshotMetadata::with` requires.
// Written `&str` to match the sibling `METADATA_KEY_*` consts and to satisfy
// `clippy::redundant_static_lifetimes`.
pub const METADATA_KEY_SHAPING: &str = "shaping";

/// Snapshot returned by [`super::LunarLanderDiscrete`] and
/// [`super::LunarLanderContinuous`] (design decision D6).
///
/// A plain [`SnapshotBase`] whose optional [`SnapshotMetadata`] carries the
/// shaping potential under [`METADATA_KEY_SHAPING`] — build the metadata with
/// [`shaping_metadata`]. Because this is `SnapshotBase` and not a bespoke type,
/// these environments compose with `TimeLimit` and any other wrapper bound to
/// `SnapshotBase`. The step limit is *also* enforced internally via
/// `config.max_steps`.
pub type LunarLanderSnapshot = SnapshotBase<1, LunarLanderObservation, ScalarReward>;

/// Packages the shaping potential Φ under this environment's metadata-key
/// contract.
///
/// `shaping` is the **absolute potential** Φ(t), not the shaping reward — see
/// [`METADATA_KEY_SHAPING`] for the full contract.
#[must_use]
pub fn shaping_metadata(shaping: f32) -> SnapshotMetadata {
    SnapshotMetadata::new().with(METADATA_KEY_SHAPING, shaping)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rlevo_core::environment::Snapshot;

    #[test]
    fn test_metadata_shaping_key_present() {
        let obs = LunarLanderObservation::default();
        let snap = LunarLanderSnapshot::running(obs, ScalarReward(1.0))
            .with_metadata(shaping_metadata(42.5));
        let meta = snap.metadata().expect("metadata must be Some");
        assert!(
            meta.components.contains_key(METADATA_KEY_SHAPING),
            "metadata must contain the shaping key"
        );
        assert_relative_eq!(meta.components[METADATA_KEY_SHAPING], 42.5, epsilon = 1e-6);
    }

    /// The shaping potential must be readable on *every* status, not just
    /// `Running`: the terminal snapshots are exactly the ones post-episode
    /// analysis inspects.
    #[test]
    fn test_shaping_metadata_present_on_all_status_variants() {
        let obs = LunarLanderObservation::default();
        let snapshots = [
            LunarLanderSnapshot::running(obs.clone(), ScalarReward(0.0))
                .with_metadata(shaping_metadata(-1.5)),
            LunarLanderSnapshot::terminated(obs.clone(), ScalarReward(-100.0))
                .with_metadata(shaping_metadata(-1.5)),
            LunarLanderSnapshot::truncated(obs, ScalarReward(0.0))
                .with_metadata(shaping_metadata(-1.5)),
        ];
        for snap in &snapshots {
            let meta = snap.metadata().expect("metadata must be Some");
            assert_relative_eq!(meta.components[METADATA_KEY_SHAPING], -1.5, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_status_variants() {
        let obs = LunarLanderObservation::default();
        let meta = shaping_metadata(0.0);
        assert!(
            !LunarLanderSnapshot::running(obs.clone(), ScalarReward(0.0))
                .with_metadata(meta.clone())
                .status()
                .is_done()
        );
        assert!(
            LunarLanderSnapshot::terminated(obs.clone(), ScalarReward(0.0))
                .with_metadata(meta.clone())
                .status()
                .is_terminated()
        );
        assert!(
            LunarLanderSnapshot::truncated(obs, ScalarReward(0.0))
                .with_metadata(meta)
                .status()
                .is_truncated()
        );
    }
}
