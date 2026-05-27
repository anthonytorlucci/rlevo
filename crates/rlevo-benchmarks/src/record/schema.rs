//! On-disk record types written by the M4 recording surface.
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

use rlevo_core::render::StyledFrame;
use serde::{Deserialize, Serialize};

/// Bumps on any breaking change to the on-disk schema. The writer
/// stamps this into every [`EpisodeRecordHeader`]; the loader refuses
/// values it does not know.
pub const FORMAT_VERSION: u16 = 1;

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

/// Per-family rich payload. M4 ships ASCII only; richer variants
/// (locomotion joint angles, `Box2D` body transforms) land in M6+.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FamilyPayload {
    Ascii,
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

/// In-memory aggregate of one on-disk episode file. Used by tests and
/// the slim M4 decoder helper; the streaming writer never materialises
/// this whole struct in memory during normal recording.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub header: EpisodeRecordHeader,
    pub frames: Vec<FrameRecord>,
    pub metrics: Vec<MetricSample>,
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
    fn format_version_is_one() {
        assert_eq!(FORMAT_VERSION, 1);
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
        };
        let bytes = bincode::serde::encode_to_vec(&rec, bincode_config()).unwrap();
        let (decoded, _): (EpisodeRecord, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(rec, decoded);
    }
}
