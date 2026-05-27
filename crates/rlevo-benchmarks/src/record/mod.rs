//! Milestone 4 — on-disk per-episode recording.
//!
//! The recording surface ships a sink trait ([`RecordSink`]) plus three
//! parallel producers (env wrapper, harness reporter, tracing layer)
//! that all push into the same sink. The architecture mirrors the M2 /
//! M3 / M3.5 unification pattern: one transport, several producers,
//! each owning a single concern (env frames, suite-lifecycle events,
//! canonical training metrics).
//!
//! See `projects/rlevo/specs/2026-05-26-env-vis/rlevo-viz-overview.md`
//! §8 for the wire format and §10 for the writer state machine.

pub mod manifest;
pub mod schema;
pub mod writer;

pub use manifest::RunManifest;
pub use schema::{
    EnvFamily, EpisodeRecord, EpisodeRecordHeader, FORMAT_VERSION, FamilyPayload, FrameRecord,
    Hyperparameters, MetricSample, RunId, bincode_config, default_frame_stride,
};
pub use writer::{InMemoryRecordSink, RecordSink, RecordWriter, RecordingConfig, read_episode_record};
