//! `tracing` event capture for tests that assert on *logging behaviour*.
//!
//! Most assertions in this workspace are about values an algorithm returns.
//! A few are about what it **emits**: a training loop's periodic progress line
//! is a documented part of its contract (`log_every`), and a loop can satisfy
//! every numeric assertion while silently logging nothing at all. That is not
//! hypothetical — issue #321 was exactly this failure: PPO/PPG gated progress
//! on `global_step % log_every == 0` at a rollout boundary, so any `log_every`
//! that did not divide `num_steps` fired on `lcm(num_steps, log_every)` and a
//! short run emitted zero lines. Nothing panicked, no metric moved, and every
//! existing test passed, because every call site passed `log_every = 0`.
//!
//! [`FieldCapture`] closes that class of gap: install it around a run, then
//! assert on the field values the run actually emitted.
//!
//! # Scope
//!
//! This is a **test assertion tool**, not a logging framework. It collects one
//! named integer field and nothing else — no span tracking, no formatting, no
//! filtering DSL. If a test needs more than "which values of field `X` did this
//! code emit, in order", it wants a real subscriber, not this.

use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

/// Collects the values of one named integer field across every `tracing` event
/// emitted while installed.
///
/// The captured values arrive in emission order, so a test can assert on both
/// *whether* something was logged and the *cadence* it was logged at (the gap
/// between consecutive values).
///
/// # Which fields are captured
///
/// Any event carrying a field with the configured name, recorded as an integer.
/// `usize` and `u64` both arrive through `tracing`'s `record_u64`; every other
/// field on the event is ignored, as are events without the named field.
///
/// Capture is deliberately **not** filtered by target or level. A test installs
/// this around a specific call — the scope is the closure, not the process — so
/// a target filter would add configuration without excluding anything the test
/// did not already choose to run. Add one only if a test appears that genuinely
/// needs to disambiguate two emitters.
///
/// # Examples
///
/// ```
/// use rlevo_test_support::capture::FieldCapture;
///
/// let capture = FieldCapture::new("step");
/// capture.record(|| {
///     tracing::info!(step = 128_usize, "progress");
///     tracing::info!(step = 256_usize, "progress");
/// });
/// assert_eq!(capture.values(), vec![128, 256]);
/// ```
///
/// Asserting that a training loop logs at all, and that its final line reports
/// the terminal step:
///
/// ```ignore
/// let capture = FieldCapture::new("step");
/// capture.record(|| {
///     train_discrete(&mut agent, &mut env, &mut rng, TOTAL, LOG_EVERY).expect("training");
/// });
/// let steps = capture.values();
/// assert!(!steps.is_empty(), "the run must emit at least one progress line");
/// assert_eq!(steps.last().copied(), Some(TOTAL as u64));
/// ```
#[derive(Debug, Clone)]
pub struct FieldCapture {
    /// Name of the event field to collect.
    field: &'static str,
    /// Values seen so far, in emission order.
    ///
    /// Shared with the installed [`Layer`], which `tracing` requires to be
    /// `Send + Sync`, hence the `Arc<Mutex<_>>` rather than a plain `Vec`.
    values: Arc<Mutex<Vec<u64>>>,
}

impl FieldCapture {
    /// Creates a capture for the event field named `field`.
    ///
    /// The capture starts empty and collects nothing until
    /// [`record`](Self::record) installs it.
    #[must_use]
    pub fn new(field: &'static str) -> Self {
        Self {
            field,
            values: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Runs `closure` with this capture installed as the default `tracing`
    /// subscriber, returning the closure's value.
    ///
    /// Installation is scoped to the closure (`tracing::subscriber::with_default`)
    /// and thread-local, so it neither leaks into later tests nor races other
    /// tests in the same binary. Values accumulate across repeated calls; build
    /// a fresh [`FieldCapture`] per assertion rather than clearing.
    ///
    /// Note that a `tracing` subscriber only observes events emitted on **this**
    /// thread. Code under test that logs from a spawned thread will not be
    /// captured.
    pub fn record<T>(&self, closure: impl FnOnce() -> T) -> T {
        let layer = CaptureLayer {
            field: self.field,
            values: Arc::clone(&self.values),
        };
        tracing::subscriber::with_default(tracing_subscriber::registry().with(layer), closure)
    }

    /// Returns the captured values, in emission order.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock was poisoned by a panic inside a previous
    /// [`record`](Self::record) call while the lock was held.
    #[must_use]
    pub fn values(&self) -> Vec<u64> {
        self.values.lock().expect("capture lock poisoned").clone()
    }
}

/// The [`Layer`] installed by [`FieldCapture::record`].
///
/// Separate from [`FieldCapture`] because `tracing` consumes the layer by value
/// on install, while the test keeps the handle it reads results from.
struct CaptureLayer {
    field: &'static str,
    values: Arc<Mutex<Vec<u64>>>,
}

impl<S: Subscriber> Layer<S> for CaptureLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor {
            field: self.field,
            value: None,
        };
        event.record(&mut visitor);
        if let Some(value) = visitor.value {
            self.values
                .lock()
                .expect("capture lock poisoned")
                .push(value);
        }
    }
}

/// Extracts a single named integer field from one event.
///
/// [`Visit`] requires `record_debug`, and every other `record_*` method
/// defaults to delegating to it; overriding only `record_u64` therefore
/// discards all non-integer fields, which is exactly the intent.
struct FieldVisitor {
    field: &'static str,
    value: Option<u64>,
}

impl Visit for FieldVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == self.field {
            self.value = Some(value);
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_capture_collects_named_field_in_emission_order() {
        let capture = FieldCapture::new("step");
        capture.record(|| {
            tracing::info!(step = 128_usize, "first");
            tracing::info!(step = 256_usize, "second");
        });
        assert_eq!(
            capture.values(),
            vec![128, 256],
            "values must be collected in emission order"
        );
    }

    #[test]
    fn test_field_capture_ignores_events_without_the_field() {
        let capture = FieldCapture::new("step");
        capture.record(|| {
            tracing::info!(step = 42_usize, "matching");
            tracing::info!(other = 99_usize, "non-matching field");
            tracing::info!("no fields at all");
        });
        assert_eq!(
            capture.values(),
            vec![42],
            "only events carrying the named field may be captured"
        );
    }

    #[test]
    fn test_field_capture_ignores_non_integer_values() {
        let capture = FieldCapture::new("step");
        capture.record(|| {
            tracing::info!(step = "not an integer", "string-valued");
            tracing::info!(step = 7_usize, "integer-valued");
        });
        assert_eq!(
            capture.values(),
            vec![7],
            "a same-named non-integer field must not be captured"
        );
    }

    #[test]
    fn test_field_capture_is_empty_when_nothing_is_emitted() {
        let capture = FieldCapture::new("step");
        capture.record(|| {});
        assert!(
            capture.values().is_empty(),
            "a run that emits nothing must capture nothing"
        );
    }

    #[test]
    fn test_field_capture_does_not_observe_events_outside_record() {
        let capture = FieldCapture::new("step");
        tracing::info!(step = 1_usize, "before install");
        capture.record(|| tracing::info!(step = 2_usize, "during"));
        tracing::info!(step = 3_usize, "after install");
        assert_eq!(
            capture.values(),
            vec![2],
            "installation must be scoped to the `record` closure"
        );
    }
}
