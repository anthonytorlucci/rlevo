//! On-disk per-episode recording.
//!
//! The recording surface ships a sink trait ([`RecordSink`]) plus three
//! parallel producers (env wrapper, harness reporter, tracing layer)
//! that all push into the same sink. This mirrors the live-TUI
//! architecture: one transport, several producers, each owning a single
//! concern (env frames, suite-lifecycle events, canonical training
//! metrics).
//!
//! See `projects/rlevo/specs/2026-05-26-env-vis/rlevo-viz-overview.md`
//! §8 for the wire format and §10 for the writer state machine.

pub mod env_tap;
pub mod manifest;
pub mod population_reporter;
pub mod reporter;
pub mod schema;
pub mod tracing_layer;
pub mod writer;

pub use env_tap::RecordingTap;
pub use population_reporter::PopulationReporter;
pub use reporter::{RecordingReporter, empty_hyperparameters};
pub use tracing_layer::RecordingLayer;

pub use manifest::RunManifest;
pub use schema::{
    Box2dPayload, EnvFamily, EpisodeRecord, EpisodeRecordHeader, FORMAT_VERSION, FamilyPayload,
    FrameRecord, Hyperparameters, Landscape2DPayload, Locomotion2DPayload, MIN_SUPPORTED_VERSION,
    MetricSample, PopulationSample, RunId, bincode_config, default_frame_stride,
};
pub use writer::{InMemoryRecordSink, RecordSink, RecordWriter, RecordingConfig, read_episode_record};
