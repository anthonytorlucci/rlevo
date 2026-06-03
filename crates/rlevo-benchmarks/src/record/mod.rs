//! On-disk per-episode recording surface (feature `record`).
//!
//! Three parallel producers all push into the same [`RecordSink`], each
//! owning a single concern:
//!
//! | Producer | Role |
//! |---|---|
//! | [`RecordingTap`] | Captures every `reset`/`step` frame from a raw env. |
//! | [`RecordingReporter`] | Routes harness lifecycle events (episode boundaries, manifest). |
//! | [`RecordingLayer`] | Extracts canonical metric fields from `tracing` events. |
//!
//! The on-disk implementation is [`RecordWriter`], which creates one
//! `episode_<N>.rec` file per episode in a run directory, then writes
//! `run.toml` at suite end via [`RunManifest::write_atomic`]. For testing
//! without touching the filesystem, use [`InMemoryRecordSink`].
//!
//! See the project spec (§8 wire format, §10 writer state machine) for the
//! full binary layout.
//!
//! [`RecordWriter`]: crate::record::writer::RecordWriter
//! [`InMemoryRecordSink`]: crate::record::writer::InMemoryRecordSink

/// [`RecordingTap`] — env wrapper that captures every reset/step frame.
pub mod env_tap;
/// [`RecordError`] — non-fatal write failures retained for post-run query.
pub mod error;
/// [`RunManifest`] — atomic `run.toml` writer for run-level metadata.
pub mod manifest;
/// [`PopulationReporter`] — EA population-snapshot sink adapter.
pub mod population_reporter;
/// [`RecordingReporter`] — suite-lifecycle producer (episode start/end, manifest).
pub mod reporter;
/// On-disk type definitions and wire-format constants.
pub mod schema;
/// [`RecordingLayer`] — `tracing` subscriber that forwards metric events to the sink.
pub mod tracing_layer;
/// [`RecordSink`] trait, [`RecordWriter`], and [`InMemoryRecordSink`].
pub mod writer;

pub use env_tap::RecordingTap;
pub use error::RecordError;
pub use population_reporter::PopulationReporter;
pub use reporter::{RecordingReporter, empty_hyperparameters};
pub use tracing_layer::RecordingLayer;

pub use manifest::RunManifest;
pub use schema::{
    Box2dPayload, EnvFamily, EpisodeRecord, EpisodeRecordHeader, FORMAT_VERSION, FamilyPayload,
    FrameRecord, GridPayload, Hyperparameters, Landscape2DPayload, Locomotion2DPayload,
    MIN_SUPPORTED_VERSION, MetricSample, PopulationSample, RecordedEnvFamily, RunId, TabularPayload,
    TrialRef, bincode_config, default_frame_stride,
};
pub use writer::{
    InMemoryRecordSink, RecordSink, RecordWriter, RecordingConfig, default_runs_dir,
    read_episode_record,
};
