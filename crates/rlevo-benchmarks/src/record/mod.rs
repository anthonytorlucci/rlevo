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
//! # What recording asks of an environment
//!
//! One thing, and it is not on the [`Environment`] trait: the action type
//! must implement [`serde::Serialize`], because each action is written to
//! the episode file. See [`RecordableAction`] — every `Serialize` type
//! satisfies it automatically, so this is a `#[derive(Serialize)]` on the
//! action enum and nothing more.
//!
//! It is declared here rather than as a supertrait of
//! [`Action`](rlevo_core::base::Action) on purpose: environments that are
//! never recorded should not pay for serde (`docs/rules.md`, ADR 0064).
//! Nothing else is required — an environment needs no
//! [`RecordedEnvFamily`] impl and no [`AsciiRenderable`] impl to be
//! recorded.
//!
//! See the project spec (the wire-format and writer-state-machine sections)
//! for the full binary layout.
//!
//! [`Environment`]: rlevo_core::environment::Environment
//! [`AsciiRenderable`]: rlevo_core::render::AsciiRenderable
//!
//! [`RecordWriter`]: crate::record::writer::RecordWriter
//! [`InMemoryRecordSink`]: crate::record::writer::InMemoryRecordSink

/// [`RecordableAction`] — the `Serialize` requirement recording places on
/// action types, and the diagnostic that explains it.
pub mod action;
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

pub use action::RecordableAction;
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
