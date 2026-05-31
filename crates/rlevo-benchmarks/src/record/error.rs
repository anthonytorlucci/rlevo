//! Errors surfaced by the recording sink.
//!
//! Recording is best-effort *during* a run: the [`RecordWriter`] logs a
//! failure via `tracing` and keeps going rather than aborting the rollout.
//! But the **first** failure it sees is retained, so a driver can query it
//! after the run via [`RecordSink::take_error`] and fail loudly instead of
//! silently shipping a truncated recording.
//!
//! [`RecordWriter`]: super::writer::RecordWriter
//! [`RecordSink::take_error`]: super::writer::RecordSink::take_error

use std::io;

/// A non-fatal failure encountered while writing a recording.
///
/// Returned by [`RecordSink::take_error`](super::writer::RecordSink::take_error)
/// after a run. `None` means every write succeeded.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RecordError {
    /// An IO failure while opening, writing, fsyncing, or finalising a
    /// recording file. `context` names the operation that failed.
    #[error("recording io failed ({context}): {source}")]
    Io {
        /// Short label for the operation that failed (e.g. `"write frame chunk"`).
        context: &'static str,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
    /// A recording sink was driven from more than one thread concurrently.
    ///
    /// A single [`RecordWriter`](super::writer::RecordWriter) is
    /// single-stream — it holds one open episode file. Recording harness
    /// runs therefore require `EvaluatorConfig.num_threads = Some(1)`. When
    /// concurrent use is detected the interloping write is **dropped**
    /// rather than allowed to truncate the in-flight episode, and this
    /// error is recorded.
    #[error(
        "recording sink used concurrently from multiple threads; \
         recording requires EvaluatorConfig.num_threads = Some(1)"
    )]
    ConcurrentUse,
    /// An action value could not be bincode-encoded for a recorded frame.
    ///
    /// The frame is still emitted (with an empty `action`, matching the
    /// reset-frame convention) so the trajectory stays decodable, but the
    /// run is flagged so a driver can fail loudly via
    /// [`RecordSink::take_error`](super::writer::RecordSink::take_error)
    /// instead of silently shipping frames whose action collapsed to the
    /// same empty-bytes sentinel a reset frame uses.
    #[error("recording action encode failed (step {step}): {message}")]
    ActionEncode {
        /// Episode-local step index of the frame whose action failed to encode.
        step: u32,
        /// Rendered encoder error.
        message: String,
    },
}
