//! On-disk record types written by the recording surface.
//!
//! Encoding: bincode 2.x with the explicit
//! [`bincode::config::standard()`] configuration locked at write time
//! and recorded in the [`EpisodeRecordHeader::format_version`] field so
//! the report-tier loader can refuse mismatched files cleanly.
//!
//! The format-version byte lives at the very start of every
//! `episode_*.rec` file so a loader can peek the version with a
//! 16-byte read before committing to a decode pass — the writer's
//! preamble is handled in [`super::writer`].

use std::collections::BTreeMap;

use rlevo_core::render::{
    Box2dSnapshot, Landscape2DSnapshot, Locomotion2DSnapshot, Point2, RigidBody2D, StyledFrame,
};
use serde::{Deserialize, Serialize};

/// Current on-disk schema version. The writer stamps this into every
/// [`EpisodeRecordHeader`]; the loader rejects any file that does not
/// carry exactly this value.
// Mirror: rlevo-benchmarks-report-client/src/wire.rs must declare the
// same value.  The const assertions in tests/wire_format_compat.rs
// enforce this at compile time when tests are built.
pub const FORMAT_VERSION: u16 = 3;

/// Oldest on-disk version this loader accepts. Equal to
/// [`FORMAT_VERSION`] — no backward compatibility is maintained
/// before the first production release.
// Mirror: rlevo-benchmarks-report-client/src/wire.rs must declare the same value.
pub const MIN_SUPPORTED_VERSION: u16 = 3;

/// Locked bincode configuration shared by writer and loader. Kept as a
/// helper rather than a constant because `bincode::config::Configuration`
/// is non-const.
#[must_use]
pub fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

/// Identifies a recording run. Format: `YYYYMMDD-HHMMSS-<6 hex>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunId(pub String);

impl RunId {
    /// Construct a run id from the current wall-clock time and a
    /// random 24-bit suffix. Collisions are vanishingly unlikely for
    /// the per-second granularity but the suffix protects against
    /// concurrent processes starting in the same second.
    #[must_use]
    pub fn new_now() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let date = time::OffsetDateTime::from_unix_timestamp(i64::try_from(secs).unwrap_or(0))
            .unwrap_or(time::OffsetDateTime::UNIX_EPOCH);
        let suffix: u32 = rand::random::<u32>() & 0x00FF_FFFF;
        Self(format!(
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
}

/// Environment family classification used both to choose a
/// [`default_frame_stride`] and to pick the report-tier panel mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EnvFamily {
    Classic,
    Grids,
    ToyText,
    Box2d,
    Locomotion,
    Landscapes,
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

/// Per-family rich payload.
///
/// **Variant ordering is wire-format-stable** — new variants append at
/// the end so existing bincode tags keep decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FamilyPayload {
    /// Plain-text / styled-text rendering — the default for ASCII-able
    /// envs.
    Ascii,
    /// Landscape (search-space) projection for `landscapes` envs.
    Landscape2D(Landscape2DPayload),
    /// Rigid-body world for `box2d` envs.
    Box2dBodies(Box2dPayload),
    /// Sagittal-plane stick figure for `locomotion` envs — their
    /// canonical view, since locomotion has no ASCII path.
    Locomotion2D(Locomotion2DPayload),
}

// ---------------------------------------------------------------------------
// Per-family rich payload structs.
//
// These are bincode-stable mirrors of the snapshot types in
// `rlevo_core::render::payload` plus serde derives. They live here so
// the wire layer stays owned by `rlevo-benchmarks`; the core crate has
// no bincode dependency.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Landscape2DPayload {
    pub bounds_x: (f32, f32),
    pub bounds_y: (f32, f32),
    pub current: Point2,
    pub best: Option<Point2>,
    pub trail: Vec<Point2>,
    pub label: String,
}

impl From<Landscape2DSnapshot> for Landscape2DPayload {
    fn from(s: Landscape2DSnapshot) -> Self {
        Self {
            bounds_x: s.bounds_x,
            bounds_y: s.bounds_y,
            current: s.current,
            best: s.best,
            trail: s.trail,
            label: s.label,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Box2dPayload {
    pub world_bounds: (Point2, Point2),
    pub bodies: Vec<RigidBody2D>,
    pub contacts: Vec<Point2>,
}

impl From<Box2dSnapshot> for Box2dPayload {
    fn from(s: Box2dSnapshot) -> Self {
        Self {
            world_bounds: s.world_bounds,
            bodies: s.bodies,
            contacts: s.contacts,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Locomotion2DPayload {
    pub joints: Vec<Point2>,
    pub bones: Vec<(u32, u32)>,
    pub ground_y: f32,
    pub com: Option<Point2>,
    pub contacts: Vec<Point2>,
}

impl From<Locomotion2DSnapshot> for Locomotion2DPayload {
    fn from(s: Locomotion2DSnapshot) -> Self {
        Self {
            joints: s.joints,
            bones: s.bones,
            ground_y: s.ground_y,
            com: s.com,
            contacts: s.contacts,
        }
    }
}


/// Header written at the start of every `episode_*.rec` file. Carries
/// run-level identification + the format-version stamp the loader uses
/// to refuse mismatched files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeRecordHeader {
    pub format_version: u16,
    pub run_id: RunId,
    pub seed: u64,
    pub env_family: EnvFamily,
    pub created_at: i64,
}

/// One captured frame, written length-prefixed to the per-episode
/// file. `action` is bincode-encoded so heterogeneous action types
/// (Discrete / Continuous / multi-dim) share one carrier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRecord {
    pub step: u32,
    pub action: Vec<u8>,
    pub reward: f32,
    pub ascii: Option<String>,
    pub styled: Option<StyledFrame>,
    pub family_payload: FamilyPayload,
}

/// One scalar metric sample with the global training step (PPO) or
/// generation index (EA) it was emitted at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSample {
    pub step: u32,
    pub name: String,
    pub value: f64,
}

/// One per-generation snapshot of an evolutionary-algorithm population.
///
/// Carried by `RecordChunk::Population` in the on-disk stream
/// (`FORMAT_VERSION = 3` and later). RL-only runs never emit this
/// chunk; EA runs emit one per call to `RecordSink::on_population_sample`,
/// typically once per generation.
///
/// **Wire fields not yet fully populated**:
///
/// - `parents_of_best` is emitted empty until a Strategy-side lineage
///   trait extension lands. The field is present so the schema does not
///   need a further bump when the lineage DAG renders.
/// - `inner_rl_returns` is `None` from pure-EA producers; populated by
///   a future hybrid driver to feed an `(inner_rl_return, fitness)`
///   scatter panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopulationSample {
    pub generation: u32,
    pub fitnesses: Vec<f32>,
    pub diversity: Option<f32>,
    pub best_index: u32,
    pub best_genome_digest: Option<[u8; 16]>,
    pub parents_of_best: Vec<[u8; 16]>,
    pub inner_rl_returns: Option<Vec<f32>>,
}

/// In-memory aggregate of one on-disk episode file. Used by tests and
/// the slim record-file decoder helper; the streaming writer never materialises
/// this whole struct in memory during normal recording.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub header: EpisodeRecordHeader,
    pub frames: Vec<FrameRecord>,
    pub metrics: Vec<MetricSample>,
    #[serde(default)]
    pub population_samples: Vec<PopulationSample>,
}

/// Free-form hyperparameter map captured in the run manifest. Per-algorithm
/// structured fields land in a follow-on milestone once an actual report-tier
/// consumer needs them.
pub type Hyperparameters = BTreeMap<String, String>;

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::render::{StyledLine, StyledSpan};

    #[test]
    fn format_version_is_three_and_min_supported_is_three() {
        assert_eq!(FORMAT_VERSION, 3);
        assert_eq!(MIN_SUPPORTED_VERSION, 3);
    }

    #[test]
    fn family_payload_round_trips_each_rich_variant() {
        let cases = [
            FamilyPayload::Ascii,
            FamilyPayload::Landscape2D(Landscape2DPayload {
                bounds_x: (-1.0, 1.0),
                bounds_y: (-1.0, 1.0),
                current: Point2::new(0.1, 0.2),
                best: None,
                trail: vec![],
                label: "sphere".into(),
            }),
            FamilyPayload::Box2dBodies(Box2dPayload {
                world_bounds: (Point2::new(-1.0, -1.0), Point2::new(1.0, 1.0)),
                bodies: vec![],
                contacts: vec![],
            }),
            FamilyPayload::Locomotion2D(Locomotion2DPayload {
                joints: vec![],
                bones: vec![],
                ground_y: 0.0,
                com: None,
                contacts: vec![],
            }),
        ];
        for c in &cases {
            let bytes = bincode::serde::encode_to_vec(c, bincode_config()).unwrap();
            let (decoded, _): (FamilyPayload, usize) =
                bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
            assert_eq!(c, &decoded);
        }
    }

    #[test]
    fn run_id_format_shape() {
        let id = RunId::new_now();
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

    fn sample_header() -> EpisodeRecordHeader {
        EpisodeRecordHeader {
            format_version: FORMAT_VERSION,
            run_id: RunId("20260527-120000-abc123".into()),
            seed: 42,
            env_family: EnvFamily::Classic,
            created_at: 1_700_000_000,
        }
    }

    fn sample_frame() -> FrameRecord {
        FrameRecord {
            step: 7,
            action: vec![1, 2, 3],
            reward: -0.5,
            ascii: Some("pole\n|".into()),
            styled: Some(StyledFrame {
                lines: vec![StyledLine {
                    spans: vec![StyledSpan::raw("pole")],
                }],
            }),
            family_payload: FamilyPayload::Ascii,
        }
    }

    fn sample_metric() -> MetricSample {
        MetricSample {
            step: 1024,
            name: "policy_loss".into(),
            value: 0.0123,
        }
    }

    #[test]
    fn header_round_trip() {
        let h = sample_header();
        let bytes = bincode::serde::encode_to_vec(&h, bincode_config()).unwrap();
        let (decoded, _): (EpisodeRecordHeader, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn frame_round_trip_with_styled() {
        let f = sample_frame();
        let bytes = bincode::serde::encode_to_vec(&f, bincode_config()).unwrap();
        let (decoded, _): (FrameRecord, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(f, decoded);
    }

    #[test]
    fn frame_round_trip_without_styled() {
        let f = FrameRecord {
            styled: None,
            ascii: None,
            ..sample_frame()
        };
        let bytes = bincode::serde::encode_to_vec(&f, bincode_config()).unwrap();
        let (decoded, _): (FrameRecord, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(f, decoded);
    }

    #[test]
    fn metric_sample_round_trip() {
        let m = sample_metric();
        let bytes = bincode::serde::encode_to_vec(&m, bincode_config()).unwrap();
        let (decoded, _): (MetricSample, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(m, decoded);
    }

    #[test]
    fn full_record_round_trip() {
        let rec = EpisodeRecord {
            header: sample_header(),
            frames: vec![sample_frame(), sample_frame()],
            metrics: vec![sample_metric()],
            population_samples: vec![sample_population()],
        };
        let bytes = bincode::serde::encode_to_vec(&rec, bincode_config()).unwrap();
        let (decoded, _): (EpisodeRecord, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(rec, decoded);
    }

    fn sample_population() -> PopulationSample {
        PopulationSample {
            generation: 4,
            fitnesses: vec![0.5, 0.4, 0.3, 0.6, 0.8],
            diversity: Some(0.12),
            best_index: 2,
            best_genome_digest: Some([7u8; 16]),
            parents_of_best: vec![[3u8; 16], [11u8; 16]],
            inner_rl_returns: None,
        }
    }

    #[test]
    fn population_sample_round_trip() {
        let p = sample_population();
        let bytes = bincode::serde::encode_to_vec(&p, bincode_config()).unwrap();
        let (decoded, _): (PopulationSample, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(p, decoded);
    }
}
