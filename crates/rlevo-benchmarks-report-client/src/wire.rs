//! Bincode-compatible mirror of [`rlevo_benchmarks::record::schema`].
//!
//! The native record schema is gated behind `feature = "record"` in
//! `rlevo-benchmarks`, which itself doesn't compile to `wasm32` because
//! `rand`/`getrandom` need the `js` feature there. Rather than restructure
//! the workspace, the client redeclares the wire types here. The
//! cross-crate compatibility test in `rlevo-benchmarks` round-trips a
//! populated `EpisodeRecord` between the two representations on every
//! `cargo test` to guarantee they stay byte-identical.
//!
//! **Sync contract:** these struct/enum definitions MUST stay in
//! lockstep with `rlevo-benchmarks/src/record/schema.rs`. Field order,
//! field types, and `#[non_exhaustive]` markers all participate in the
//! bincode wire format. Bumping `FORMAT_VERSION` is the canonical
//! escape hatch when the contract changes.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---- Mirror of rlevo_core::render::styled (kept thin; the cross-crate
// compat test in rlevo-benchmarks guards every field). -----------------

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyledFrame {
    pub lines: Vec<StyledLine>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyledLine {
    pub spans: Vec<StyledSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyledSpan {
    pub text: String,
    pub style: SpanStyle,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanStyle {
    pub fg: Option<Color>,
    pub bg: Option<Color>,
    pub modifier: Modifier,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Color {
    Reset,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    Gray,
    DarkGray,
    LightRed,
    LightGreen,
    LightYellow,
    LightBlue,
    LightMagenta,
    LightCyan,
    White,
    Indexed(u8),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Modifier(pub u8);

// ---- /styled mirror ---------------------------------------------------

pub const FORMAT_VERSION: u16 = 1;

#[must_use]
pub fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunId(pub String);

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FamilyPayload {
    Ascii,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeRecordHeader {
    pub format_version: u16,
    pub run_id: RunId,
    pub seed: u64,
    pub env_family: EnvFamily,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRecord {
    pub step: u32,
    pub action: Vec<u8>,
    pub reward: f32,
    pub ascii: Option<String>,
    pub styled: Option<StyledFrame>,
    pub family_payload: FamilyPayload,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSample {
    pub step: u32,
    pub name: String,
    pub value: f64,
}

/// Length-prefixed wire-format chunk written by the M4 record writer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecordChunk {
    Frame(FrameRecord),
    Metrics(Vec<MetricSample>),
}

/// In-memory aggregate of one decoded `.rec` file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub header: EpisodeRecordHeader,
    pub frames: Vec<FrameRecord>,
    pub metrics: Vec<MetricSample>,
}

pub type Hyperparameters = BTreeMap<String, String>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: RunId,
    pub seed: u64,
    pub env_family: EnvFamily,
    pub created_at: i64,
    pub finished_at: i64,
    pub episode_count: u32,
    pub frame_stride: u16,
    pub format_version: u16,
    #[serde(default)]
    pub hyperparameters: Hyperparameters,
}

/// Decode the raw bytes of a single `episode_*.rec` file produced by
/// the M4 writer. Tolerates truncated tails by stopping cleanly at
/// the last whole chunk.
///
/// # Errors
///
/// Returns [`DecodeError::Truncated`] if the preamble is shorter than
/// 16 bytes, [`DecodeError::VersionMismatch`] if the format version in
/// the preamble does not match [`FORMAT_VERSION`], or
/// [`DecodeError::Bincode`] if a length-prefixed chunk fails to
/// deserialise.
pub fn decode_episode_record(bytes: &[u8]) -> Result<EpisodeRecord, DecodeError> {
    if bytes.len() < 16 {
        return Err(DecodeError::Truncated("preamble"));
    }
    let version = u16::from_le_bytes([bytes[0], bytes[1]]);
    if version != FORMAT_VERSION {
        return Err(DecodeError::VersionMismatch {
            file: version,
            client: FORMAT_VERSION,
        });
    }
    let mut cursor = 16;
    let header: EpisodeRecordHeader = read_chunk(bytes, &mut cursor)?.ok_or(DecodeError::Truncated("header"))?;

    let mut frames = Vec::new();
    let mut metrics = Vec::new();
    while let Some(chunk) = read_chunk::<RecordChunk>(bytes, &mut cursor)? {
        match chunk {
            RecordChunk::Frame(fr) => frames.push(fr),
            RecordChunk::Metrics(ms) => metrics.extend(ms),
        }
    }
    Ok(EpisodeRecord {
        header,
        frames,
        metrics,
    })
}

fn read_chunk<T: for<'de> Deserialize<'de>>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<Option<T>, DecodeError> {
    if *cursor >= bytes.len() {
        return Ok(None);
    }
    if bytes.len() - *cursor < 4 {
        // Partial length prefix — truncated tail.
        return Ok(None);
    }
    let len = u32::from_le_bytes([
        bytes[*cursor],
        bytes[*cursor + 1],
        bytes[*cursor + 2],
        bytes[*cursor + 3],
    ]) as usize;
    *cursor += 4;
    let available = bytes.len() - *cursor;
    if available < len {
        // Partial payload — truncated tail.
        return Ok(None);
    }
    let payload = &bytes[*cursor..*cursor + len];
    *cursor += len;
    let (value, _): (T, usize) = bincode::serde::decode_from_slice(payload, bincode_config())
        .map_err(|e| DecodeError::Bincode(e.to_string()))?;
    Ok(Some(value))
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("record file truncated at {0}")]
    Truncated(&'static str),
    #[error("format version mismatch: file={file} client={client}")]
    VersionMismatch { file: u16, client: u16 },
    #[error("bincode decode failed: {0}")]
    Bincode(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_header() {
        let h = EpisodeRecordHeader {
            format_version: FORMAT_VERSION,
            run_id: RunId("xyz".into()),
            seed: 42,
            env_family: EnvFamily::Classic,
            created_at: 1700,
        };
        let bytes = bincode::serde::encode_to_vec(&h, bincode_config()).unwrap();
        let (decoded, _): (EpisodeRecordHeader, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn decode_rejects_version_mismatch() {
        let mut bytes = vec![0u8; 16];
        bytes[0..2].copy_from_slice(&999u16.to_le_bytes());
        let err = decode_episode_record(&bytes).unwrap_err();
        match err {
            DecodeError::VersionMismatch { file: 999, client } if client == FORMAT_VERSION => {}
            other => panic!("expected version mismatch, got {other:?}"),
        }
    }
}
