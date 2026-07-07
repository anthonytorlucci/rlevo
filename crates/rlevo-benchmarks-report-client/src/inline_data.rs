//! Readers for the data blocks the report emitter inlines into `index.html`.
//!
//! The emitter writes four `<script type="application/json">` (or base64)
//! blocks, each with a stable `id`:
//!
//! | id | content | decoded as |
//! |----|---------|------------|
//! | `rlevo-manifest` | JSON | [`RunManifest`] |
//! | `rlevo-warnings` | JSON | `Vec<`[`WarningEntry`]`>` |
//! | `rlevo-episode-index` | JSON | `Vec<`[`EpisodeMeta`]`>` |
//! | `rlevo-ep-<script_id>` | base64 bincode | [`EpisodeRecord`] |
//!
//! [`read_manifest`], [`read_warnings`], [`read_episode_index`], and
//! [`read_episode_record`] each read one block.  [`read_all_episode_records`]
//! and [`read_all_population_samples`] add `OnceLock` caching so reactive
//! panels never re-decode the same bincode payload twice.
//!
//! [`RunManifest`]: crate::wire::RunManifest
//! [`EpisodeRecord`]: crate::wire::EpisodeRecord

use std::sync::OnceLock;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde::Deserialize;
use thiserror::Error;
use web_sys::wasm_bindgen::JsCast;

use crate::wire::{
    DecodeError, EpisodeRecord, PopulationSample, RunManifest, decode_episode_record,
};

/// Summary metadata produced by the emitter alongside the raw `.rec` payloads.
///
/// Matches the `EpisodeMeta` struct in `rlevo-benchmarks/src/report/html.rs`
/// and is decoded from the `rlevo-episode-index` JSON block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EpisodeMeta {
    /// Zero-based episode number within the run.
    pub episode: u32,
    /// Total number of frames recorded in this episode.
    pub frame_count: u32,
    /// Cumulative reward accumulated over the episode.
    pub episode_reward: f64,
    /// Episode length in simulation steps (may differ from `frame_count` when
    /// the frame stride is greater than one).
    pub length: u32,
    /// Stable identifier used as the `<script>` element id for this episode's
    /// base64 bincode payload.
    pub script_id: String,
    /// `"training"` or `"evaluation"` (v6). Defaults to `"training"` for
    /// pre-v6 indices that omit the field.
    #[serde(default = "default_kind")]
    pub kind: String,
}

/// Default episode kind for index entries written before the field existed.
fn default_kind() -> String {
    "training".to_string()
}

/// JSON wire form for an `OpenWarning` emitted into `rlevo-warnings`.
///
/// Field names match the emitter side
/// (`rlevo-benchmarks::report::html::warnings_to_json`).
#[derive(Debug, Clone, Deserialize)]
pub struct WarningEntry {
    /// Human-readable warning category (e.g. `"episode_count_mismatch"`).
    pub kind: String,
    /// Expected count from the manifest, when applicable.
    pub manifest_count: Option<u32>,
    /// Actual count discovered at emit time, when applicable.
    pub found_count: Option<u32>,
}

/// Errors that can occur while reading or decoding an inlined data block.
#[derive(Debug, Error)]
pub enum InlineError {
    #[error("missing script element with id `{0}`")]
    MissingScript(String),
    #[error("script element `{id}` could not be downcast to HtmlElement")]
    WrongElementType { id: String },
    #[error("json decode of `{id}` failed: {source}")]
    Json {
        id: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("base64 decode failed: {0}")]
    Base64(String),
    #[error("bincode decode of `{id}` failed: {source}")]
    Wire {
        id: String,
        #[source]
        source: DecodeError,
    },
}

/// Returns the `innerText` of the `<script>` element with the given `id`.
///
/// # Errors
///
/// Returns [`InlineError::MissingScript`] if `window`, `document`, or the
/// element cannot be obtained, or [`InlineError::WrongElementType`] if the
/// element cannot be downcast to `HtmlElement`.
fn script_text(id: &str) -> Result<String, InlineError> {
    let window = web_sys::window().ok_or_else(|| InlineError::MissingScript("window".into()))?;
    let doc = window
        .document()
        .ok_or_else(|| InlineError::MissingScript("document".into()))?;
    let el = doc
        .get_element_by_id(id)
        .ok_or_else(|| InlineError::MissingScript(id.into()))?;
    let html: web_sys::HtmlElement = el
        .dyn_into()
        .map_err(|_| InlineError::WrongElementType { id: id.into() })?;
    Ok(html.inner_text())
}

/// Read and JSON-decode `<script id="rlevo-manifest">`.
///
/// # Errors
///
/// Returns [`InlineError::MissingScript`] if the script tag is not in
/// the document, or [`InlineError::Json`] if its content is not a
/// valid `RunManifest`.
pub fn read_manifest() -> Result<RunManifest, InlineError> {
    let id = "rlevo-manifest";
    let text = script_text(id)?;
    serde_json::from_str(text.trim()).map_err(|e| InlineError::Json {
        id: id.into(),
        source: e,
    })
}

/// Read `<script id="rlevo-warnings">`. Returns an empty vec when the
/// script tag is missing (graceful degradation for legacy emitters).
///
/// # Errors
///
/// Returns [`InlineError::Json`] if the script content is malformed.
pub fn read_warnings() -> Result<Vec<WarningEntry>, InlineError> {
    let id = "rlevo-warnings";
    let text = match script_text(id) {
        Ok(t) => t,
        Err(InlineError::MissingScript(_)) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    serde_json::from_str(text.trim()).map_err(|e| InlineError::Json {
        id: id.into(),
        source: e,
    })
}

/// Read `<script id="rlevo-episode-index">` — the summary metadata
/// every emitted episode carries.
///
/// # Errors
///
/// Returns [`InlineError::MissingScript`] if the index is absent or
/// [`InlineError::Json`] if it is malformed.
pub fn read_episode_index() -> Result<Vec<EpisodeMeta>, InlineError> {
    let id = "rlevo-episode-index";
    let text = script_text(id)?;
    serde_json::from_str(text.trim()).map_err(|e| InlineError::Json {
        id: id.into(),
        source: e,
    })
}

/// Read a single base64 episode block, decode it, and run the bincode
/// pass via [`decode_episode_record`].
///
/// # Errors
///
/// Returns [`InlineError::MissingScript`] for missing script tags,
/// [`InlineError::Base64`] for malformed base64 payloads, or
/// [`InlineError::Wire`] for bincode failures.
pub fn read_episode_record(script_id: &str) -> Result<EpisodeRecord, InlineError> {
    let text = script_text(script_id)?;
    let trimmed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = B64
        .decode(&trimmed)
        .map_err(|e| InlineError::Base64(format!("{script_id}: {e}")))?;
    decode_episode_record(&bytes).map_err(|e| InlineError::Wire {
        id: script_id.into(),
        source: e,
    })
}

/// Cached batch decode of every episode block referenced in the index.
///
/// The RL convergence panel needs the full per-episode reward / length
/// trajectory plus the concatenated metric stream. The cache means the
/// reactive panel re-render does not re-decode the underlying bincode
/// payloads — `OnceLock` initialises on the first call and every
/// subsequent call returns the same `&'static Vec`. Per-record decode
/// failures are skipped so a single malformed episode does not blank
/// the whole panel.
#[must_use]
pub fn read_all_episode_records() -> &'static Vec<EpisodeRecord> {
    static CACHE: OnceLock<Vec<EpisodeRecord>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let index = match read_episode_index() {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        index
            .iter()
            .filter_map(|m| read_episode_record(&m.script_id).ok())
            .collect()
    })
}

/// Cached flat list of every population sample across every episode in
/// the run.
///
/// Concatenated in episode-then-emission order — for a typical EA run
/// there is one episode with one sample per generation, so the result
/// is effectively the per-generation snapshot stream. Used by the EA
/// Population section; RL-only runs return an empty slice and the
/// section suppresses cleanly.
#[must_use]
pub fn read_all_population_samples() -> &'static Vec<PopulationSample> {
    static CACHE: OnceLock<Vec<PopulationSample>> = OnceLock::new();
    CACHE.get_or_init(|| {
        read_all_episode_records()
            .iter()
            .flat_map(|r| r.population_samples.iter().cloned())
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::EpisodeMeta;

    /// Guards the JSON contract against the host emitter's `EpisodeMeta` in
    /// `rlevo-benchmarks/src/report/html.rs`, which has no compile-time link to
    /// this struct (architecture review finding R6). The field names and the
    /// `kind` string values ("training"/"evaluation") must match.
    #[test]
    fn episode_meta_decodes_emitter_json_with_kind() {
        let json = r#"{
            "episode": 3,
            "frame_count": 120,
            "episode_reward": 42.5,
            "length": 120,
            "script_id": "rlevo-episode-000003",
            "kind": "evaluation"
        }"#;
        let m: EpisodeMeta = serde_json::from_str(json).expect("decodes");
        assert_eq!(m.episode, 3);
        assert_eq!(m.kind, "evaluation");
    }

    /// A pre-v6 index omitting `kind` must still decode, defaulting to training.
    #[test]
    fn episode_meta_defaults_kind_to_training() {
        let json = r#"{
            "episode": 0,
            "frame_count": 10,
            "episode_reward": 1.0,
            "length": 10,
            "script_id": "rlevo-episode-000000"
        }"#;
        let m: EpisodeMeta = serde_json::from_str(json).expect("decodes without kind");
        assert_eq!(m.kind, "training");
    }
}
