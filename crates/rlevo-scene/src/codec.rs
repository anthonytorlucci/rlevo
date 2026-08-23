//! Reading and writing the framed `.rec` chunk stream.
//!
//! An `episode_*.rec` file is a 16-byte preamble carrying
//! [`FORMAT_VERSION`](crate::FORMAT_VERSION), then a length-prefixed bincode
//! [`EpisodeRecordHeader`](crate::EpisodeRecordHeader), then length-prefixed
//! [`RecordChunk`]s until EOF.
//!
//! [`decode_episode_record`] is the whole-file reader the report client uses.
//! The writer side streams chunks one at a time and lives in
//! `rlevo-benchmarks`, because it needs the filesystem; both sides share
//! [`RecordChunk`], which is the point. It was previously declared twice — a
//! private copy in the writer and a public one in the client's `wire.rs` — so
//! the framing itself was forked, not just the payloads.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::schema::{
    EpisodeRecord, EpisodeRecordHeader, FORMAT_VERSION, FrameRecord, MetricSample,
    PopulationSample, bincode_config,
};

/// Length-prefixed wire-format chunk written by the on-disk record writer.
///
/// **Variant ordering is wire-format-stable** — new variants append at
/// the end so existing bincode tags keep decoding. `Population` is at tag 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecordChunk {
    /// One simulation step (bincode tag 0).
    Frame(FrameRecord),
    /// Batch of metric samples emitted together (bincode tag 1).
    Metrics(Vec<MetricSample>),
    /// One EA population snapshot, present in v2+ records (bincode tag 2).
    Population(PopulationSample),
}

/// Decode the raw bytes of a single `episode_*.rec` file produced by
/// the on-disk record writer. Tolerates truncated tails by stopping
/// cleanly at the last whole chunk.
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
    let header: EpisodeRecordHeader =
        read_chunk(bytes, &mut cursor)?.ok_or(DecodeError::Truncated("header"))?;

    let mut frames = Vec::new();
    let mut metrics = Vec::new();
    let mut population_samples = Vec::new();
    while let Some(chunk) = read_chunk::<RecordChunk>(bytes, &mut cursor)? {
        match chunk {
            RecordChunk::Frame(fr) => frames.push(fr),
            RecordChunk::Metrics(ms) => metrics.extend(ms),
            RecordChunk::Population(ps) => population_samples.push(ps),
        }
    }
    Ok(EpisodeRecord {
        header,
        frames,
        metrics,
        population_samples,
    })
}

/// Reads one length-prefixed bincode chunk from `bytes` at `*cursor`.
///
/// Advances `*cursor` past the 4-byte length prefix and the payload.
/// Returns `Ok(None)` on a truncated length prefix or partial payload so
/// the caller can stop cleanly without treating a truncated tail as an error.
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

/// Errors that can occur while decoding a `.rec` binary file.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// The file ended before the named section was fully read.
    #[error("record file truncated at {0}")]
    Truncated(&'static str),
    /// The 16-bit version tag in the file does not equal [`FORMAT_VERSION`].
    #[error("format version mismatch: file={file} client={client}")]
    VersionMismatch {
        /// Version stamped in the file's preamble.
        file: u16,
        /// Version this build of the reader supports, i.e. [`FORMAT_VERSION`].
        client: u16,
    },
    /// A length-prefixed chunk failed bincode deserialization; carries the error message.
    #[error("bincode decode failed: {0}")]
    Bincode(String),
}
