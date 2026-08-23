//! The run manifest, and the one thing the client still mirrors on purpose.
//!
//! Everything that reaches the client as **bincode** — [`EpisodeRecord`],
//! [`FrameRecord`], [`FamilyPayload`], the payload shapes, the styled-text
//! types, [`FORMAT_VERSION`], and [`decode_episode_record`] — now comes from
//! [`rlevo_scene`], defined once and shared with the recorder. This module
//! re-exports it so `crate::wire::…` paths inside the client keep working.
//!
//! # Why `RunManifest` is still a copy
//!
//! **This is deliberate, not leftover.** Do not "finish the job" by deleting it.
//!
//! The manifest does not travel as bincode. `rlevo-benchmarks` emits it with
//! `serde_json::to_string` into a `<script type="application/json">` block, and
//! that difference decides everything:
//!
//! | Crosses as | Keyed by | Drift behaviour |
//! |---|---|---|
//! | bincode | position / variant tag | **silent corruption** — hence one definition |
//! | JSON | field name | graceful: unknown ignored, missing become `None` |
//!
//! *Mirror what is self-describing; share what is positional.*
//!
//! Sharing it instead would mean moving [`ObjectiveSense`] — which the manifest
//! carries — out of `rlevo-core`, where an optimisation concept belongs, and
//! into a record crate. That is a worse trade than a 70-line struct whose worst
//! failure mode is a field silently reading `None`. See ADR 0081.

pub use rlevo_scene::codec::{DecodeError, RecordChunk, decode_episode_record};
pub use rlevo_scene::payload::{
    BodyKind, Box2dSnapshot, CardTable, Classic2DBody, Classic2DRole, Classic2DSnapshot,
    GridAgentMarker, GridColor, GridDir, GridDoorState, GridSnapshot, GridTile,
    Landscape2DSnapshot, Locomotion2DSnapshot, Point2, RigidBody2D, TabularCell, TabularGrid,
    TabularLayout, TabularMarker, TabularMarkerKind, TabularSnapshot,
};
pub use rlevo_scene::schema::{
    CheckpointFormat, CheckpointKind, CheckpointRef, EnvFamily, EpisodeKind, EpisodeRecord,
    EpisodeRecordHeader, FORMAT_VERSION, FamilyPayload, FrameRecord, Hyperparameters,
    MIN_SUPPORTED_VERSION, MetricSample, PopulationSample, RunId, TrialRef, bincode_config,
};
pub use rlevo_scene::styled::{Color, Modifier, SpanStyle, StyledFrame, StyledLine, StyledSpan};

use serde::{Deserialize, Serialize};

/// JSON manifest embedded in `index.html` describing the overall run.
///
/// `Eq` is intentionally not derived: `success_threshold` and the metrics
/// carried by [`CheckpointRef`] are `f64`, which is not `Eq`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    /// Identifier of the training run.
    pub run_id: RunId,
    /// RNG seed used for the run.
    pub seed: u64,
    /// Environment family for all episodes in the run.
    pub env_family: EnvFamily,
    /// Unix timestamp (seconds) when the run started.
    pub created_at: i64,
    /// Unix timestamp (seconds) when the run finished.
    pub finished_at: i64,
    /// Number of episodes recorded (may differ from `index.html` episode count on partial runs).
    pub episode_count: u32,
    /// Only every `frame_stride`-th simulation step was written to disk.
    pub frame_stride: u16,
    /// Wire-format version the emitter used; should match [`FORMAT_VERSION`].
    pub format_version: u16,
    /// Algorithm hyperparameters logged at run start; empty when not provided.
    #[serde(default)]
    pub hyperparameters: Hyperparameters,
    /// Algorithm identity (e.g. `"ppo"`). Added in `FORMAT_VERSION = 6`.
    #[serde(default)]
    pub algorithm: Option<String>,
    /// `rlevo` crate version. Added in v6.
    #[serde(default)]
    pub rlevo_version: Option<String>,
    /// Rust toolchain version string. Added in v6.
    #[serde(default)]
    pub rustc_version: Option<String>,
    /// Resolved `burn` dependency version. Added in v6.
    #[serde(default)]
    pub burn_version: Option<String>,
    /// `OS`-`ARCH` platform string. Added in v6.
    #[serde(default)]
    pub platform: Option<String>,
    /// Git commit hash of the build, if known. Added in v6.
    #[serde(default)]
    pub git_commit: Option<String>,
    /// Whether the working tree was dirty at build time. Added in v6.
    #[serde(default)]
    pub git_dirty: Option<bool>,
    /// Backend device descriptor. Added in v6.
    #[serde(default)]
    pub device: Option<String>,
    /// Distinct seed count across the trial suite. Added in v6.
    #[serde(default)]
    pub num_seeds: Option<u32>,
    /// Success threshold that produced `success_rate`. Added in v6.
    #[serde(default)]
    pub success_threshold: Option<f64>,
    /// Deep-RL learner checkpoints (Burn-`Recorder` files referenced, never
    /// embedded). Empty for EA and un-wired RL. Added in v6.
    #[serde(default)]
    pub checkpoints: Vec<CheckpointRef>,
    /// Objective direction for the run. `None` ⇒ `Maximize` (the canonical
    /// engine sense), so RL and unspecified runs render "best/worst"
    /// correctly. Added in `FORMAT_VERSION = 7`.
    #[serde(default)]
    pub objective_sense: Option<ObjectiveSense>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveSense {
    /// Lower is better (cost, loss, error, distance-to-target).
    Minimize,
    /// Higher is better (reward, fitness, accuracy, score).
    Maximize,
}
