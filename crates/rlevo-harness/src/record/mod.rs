//! On-disk per-episode recording surface (feature `record`).
//!
//! Four parallel producers all push into the same [`RecordSink`], each
//! owning a single concern:
//!
//! | Producer | Role |
//! |---|---|
//! | [`RecordingTap`] | Captures every `reset`/`step` frame from a raw env. |
//! | [`RecordingReporter`] | Routes harness lifecycle events (episode boundaries, manifest). |
//! | [`PopulationReporter`] | Forwards EA population snapshots (generation, fitness stats). |
//! | [`RecordingLayer`] | Extracts canonical metric fields from `tracing` events. |
//!
//! The on-disk implementation is [`RecordWriter`], which creates one
//! `episode_<N>.rec` file per episode in a run directory, then writes
//! `run.toml` at suite end via [`RunManifest::write_atomic`]. For testing
//! without touching the filesystem, use [`InMemoryRecordSink`].
//!
//! The [`writer`] module documents the binary layout of an episode file.
//!
//! [`RecordWriter`]: crate::record::writer::RecordWriter
//! [`InMemoryRecordSink`]: crate::record::writer::InMemoryRecordSink

pub mod env_tap;
pub mod error;
pub mod manifest;
pub mod population_reporter;
pub mod reporter;
pub mod schema;
pub mod tracing_layer;
pub mod writer;

pub use env_tap::RecordingTap;
pub use error::RecordError;
pub use population_reporter::PopulationReporter;
pub use reporter::{RecordingReporter, empty_hyperparameters};
pub use tracing_layer::RecordingLayer;

pub use manifest::RunManifest;
pub use schema::{
    Box2dPayload, CheckpointFormat, CheckpointKind, CheckpointRef, Classic2DPayload, EnvFamily,
    EpisodeKind, EpisodeRecord, EpisodeRecordHeader, FORMAT_VERSION, FamilyPayload, FrameRecord,
    GridPayload, Hyperparameters, Landscape2DPayload, Locomotion2DPayload, MIN_SUPPORTED_VERSION,
    MetricSample, PopulationSample, RecordedEnvFamily, RunId, TabularPayload, TrialRef,
    bincode_config, default_frame_stride,
};
pub use writer::{
    InMemoryRecordSink, RecordSink, RecordWriter, RecordingConfig, default_runs_dir,
    read_episode_record,
};
