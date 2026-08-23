//! Record types, re-exported from [`rlevo_scene`], plus the producer-side
//! pieces that stay here.
//!
//! The schema itself — [`EnvFamily`], [`FamilyPayload`], [`EpisodeRecord`],
//! [`FORMAT_VERSION`], the codec — moved to `rlevo-scene` in ADR 0081 so the
//! recorder and the WASM report client share **one** definition instead of
//! three. This module re-exports it so existing
//! `rlevo_benchmarks::record::schema::…` paths keep resolving, exactly as
//! [`crate::metrics_registry`] does over `rlevo-metrics-registry` (ADR 0015).
//!
//! Six `*Payload` structs that used to live here — each documented as a
//! *"bincode-stable mirror"* of a snapshot type, each with a hand-written `From`
//! impl — are **gone**. [`FamilyPayload`] carries the snapshot types directly.
//!
//! Three things stay, because they are not wire data:
//!
//! | Item | Why |
//! |---|---|
//! | [`new_run_id`](crate::record::schema::new_run_id) | needs a clock and an RNG; the leaf has neither |
//! | [`RecordedEnvFamily`] | a producer-side opt-in on an environment type |
//! | [`default_frame_stride`] | recording policy, not record content |

pub use rlevo_scene::codec::{DecodeError, RecordChunk, decode_episode_record};
pub use rlevo_scene::schema::{
    CheckpointFormat, CheckpointKind, CheckpointRef, EnvFamily, EpisodeKind, EpisodeRecord,
    EpisodeRecordHeader, FORMAT_VERSION, FamilyPayload, FrameRecord, Hyperparameters,
    MIN_SUPPORTED_VERSION, MetricSample, PopulationSample, RunId, TrialRef, bincode_config,
};

/// Constructs a [`RunId`] from the current wall-clock time and a random 24-bit
/// suffix. Collisions are vanishingly unlikely at per-second granularity, but
/// the suffix protects against concurrent processes starting in the same second.
///
/// A free function rather than `RunId::new_now`, because `RunId` is defined in
/// `rlevo-scene` and an inherent impl cannot cross crates. That split is the
/// right one anyway: the *type* is wire data, while *minting* one needs a clock
/// and an RNG, and the leaf deliberately has neither — `rand` is the dependency
/// its whole existence is arranged around avoiding.
#[must_use]
pub fn new_run_id() -> RunId {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let date = time::OffsetDateTime::from_unix_timestamp(i64::try_from(secs).unwrap_or(0))
        .unwrap_or(time::OffsetDateTime::UNIX_EPOCH);
    let suffix: u32 = rand::random::<u32>() & 0x00FF_FFFF;
    RunId(format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}-{:06x}",
        date.year(),
        u8::from(date.month()),
        date.day(),
        date.hour(),
        date.minute(),
        date.second(),
        suffix
    ))
}

/// Opt-in association between a concrete environment type and its
/// [`EnvFamily`].
///
/// Recording and visualisation drivers otherwise restate the family as a
/// literal at every call site — once for [`RecordingConfig`] and again for
/// the TUI config — with nothing tying either back to the environment being
/// run. The two can silently disagree (recording a locomotion env as
/// [`EnvFamily::Classic`] compiles fine and just produces the wrong
/// report-tier adapter). Implementing this trait lets a driver derive the
/// family from the env type *once* via [`RecordingConfig::for_env`] /
/// [`Self::FAMILY`], collapsing the two literals to a single source of truth.
///
/// This is deliberately **not** a supertrait of
/// [`Environment`](rlevo_core::environment::Environment): family/render
/// knowledge stays an opt-in concern off the behavioural trait, per ADR 0013,
/// which carries forward the production-crate isolation rules first stated in
/// the superseded ADR 0007.
///
/// Impls for the built-in environments live in this crate, in
/// `fixtures::family`, behind its `fixtures` feature — unlinked because
/// `fixtures` and `record` are independent, so the target is absent in a
/// `record`-only build. They moved off the environments side in ADR 0080 so that
/// `rlevo-environments` no longer carries an optional dependency on the
/// harness; the orphan rule permits them there because this crate owns the
/// trait.
///
/// [`RecordingConfig`]: crate::record::writer::RecordingConfig
/// [`RecordingConfig::for_env`]: crate::record::writer::RecordingConfig::for_env
pub trait RecordedEnvFamily {
    /// The report adapter that decodes this environment's recorded frames.
    ///
    /// It must agree with the [`FamilyPayload`] variant the environment
    /// actually emits, via whichever producer-side payload trait it implements
    /// — [`GridPayloadSource`](rlevo_scene::GridPayloadSource) means
    /// [`FamilyPayload::Grid`] frames, which only the grids adapter can decode,
    /// hence [`EnvFamily::Grids`]. It is *not* the module the environment lives
    /// in, and not a statement about what kind of task it is.
    ///
    /// `SantaFeAnt` is the worked example: it lives in `rlevo-environments`'
    /// `classic` module, but it implements `GridPayloadSource` and so records as
    /// [`EnvFamily::Grids`]. "Correcting" it to [`EnvFamily::Classic`] to match
    /// its module would route `Grid` payloads through the classic adapter, which
    /// cannot decode them and would quietly degrade every frame to the ASCII
    /// fallback.
    ///
    /// An environment with no rich payload source emits
    /// [`FamilyPayload::Ascii`], which each adapter renders through its fallback
    /// path. For those the family is picked for the closest report tier rather
    /// than for a decodable payload — the bandits live in `classic::bandit`,
    /// emit ASCII, and take [`EnvFamily::Classic`].
    const FAMILY: EnvFamily;
}

/// Per-family default `frame_stride` — locomotion + `Box2D` environments
/// emit denser frame streams so we sub-sample by default. Overridden
/// per-run via `RecordingConfig::frame_stride`.
#[must_use]
pub const fn default_frame_stride(family: EnvFamily) -> u16 {
    match family {
        EnvFamily::Locomotion => 6,
        EnvFamily::Box2d => 4,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_id_format_shape() {
        let id = new_run_id();
        let s = &id.0;
        assert_eq!(s.len(), 22, "expected YYYYMMDD-HHMMSS-XXXXXX, got {s}");
        assert_eq!(&s[8..9], "-");
        assert_eq!(&s[15..16], "-");
    }

    #[test]
    fn default_frame_stride_classic_is_one() {
        assert_eq!(default_frame_stride(EnvFamily::Classic), 1);
        assert_eq!(default_frame_stride(EnvFamily::Grids), 1);
        assert_eq!(default_frame_stride(EnvFamily::ToyText), 1);
        assert_eq!(default_frame_stride(EnvFamily::Landscapes), 1);
    }

    #[test]
    fn default_frame_stride_dense_families_decimate() {
        assert_eq!(default_frame_stride(EnvFamily::Locomotion), 6);
        assert_eq!(default_frame_stride(EnvFamily::Box2d), 4);
    }
}
