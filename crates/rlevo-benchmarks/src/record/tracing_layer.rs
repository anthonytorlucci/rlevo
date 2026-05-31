//! [`RecordingLayer`] — a `tracing` Layer that pushes
//! [`MetricSample`]s into a [`RecordSink`].
//!
//! Counterpart to [`TuiCaptureLayer`](crate::tui::log_layer::TuiCaptureLayer)
//! but writes to the record-side sink instead of the TUI channel.
//! Both layers share the [`crate::metrics_registry::CANONICAL_METRICS`]
//! registry so a single algorithm `tracing::info!(...)` call lights up
//! both the live sparklines and the on-disk metrics stream.
//!
//! Step coordinate: an `AtomicU64` increments on every captured event.
//! When the emitting algorithm includes a numeric `step` field on the
//! event, that value wins; otherwise the layer's own counter is used.
//! Multiple metrics on the same event share the same step value.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer};

use super::schema::MetricSample;
use super::writer::RecordSink;

pub use crate::metrics_registry::{CANONICAL_METRICS, is_canonical_metric};

/// `tracing_subscriber::Layer` that captures canonical metric fields
/// into the on-disk record stream.
#[derive(Clone)]
pub struct RecordingLayer {
    sink: Arc<Mutex<dyn RecordSink>>,
    step: Arc<AtomicU64>,
}

impl fmt::Debug for RecordingLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordingLayer")
            .field("sink", &"Arc<Mutex<dyn RecordSink>>")
            .field("step", &self.step.load(Ordering::Relaxed))
            .finish()
    }
}

impl RecordingLayer {
    /// Constructs the layer wrapped around a shared sink. The same
    /// `sink` should also be held by the env-tap / reporter producers
    /// in the same run so all three write to the same files.
    #[must_use]
    pub fn new(sink: Arc<Mutex<dyn RecordSink>>) -> Self {
        Self {
            sink,
            step: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl<S> Layer<S> for RecordingLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = CaptureVisitor::default();
        event.record(&mut visitor);
        if visitor.metrics.is_empty() {
            return;
        }
        let event_step = visitor.step.unwrap_or_else(|| {
            let prev = self.step.fetch_add(1, Ordering::Relaxed);
            u32::try_from(prev).unwrap_or(u32::MAX)
        });
        let mut sink = self.sink.lock();
        for (name, value) in visitor.metrics {
            sink.on_metric(MetricSample {
                step: event_step,
                name,
                value,
            });
        }
    }
}

#[derive(Default)]
struct CaptureVisitor {
    metrics: Vec<(String, f64)>,
    step: Option<u32>,
}

impl CaptureVisitor {
    fn record_canonical(&mut self, field: &Field, value: f64) {
        if is_canonical_metric(field.name()) {
            self.metrics.push((field.name().to_string(), value));
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "metrics need f64; integer counters lose precision only at extreme magnitudes"
)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "step coordinate is a u32 by schema; saturating cast is fine for diagnostic step indices"
)]
#[allow(
    clippy::cast_sign_loss,
    reason = "tracing step fields are non-negative by construction; clamp negatives to 0"
)]
impl Visit for CaptureVisitor {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.record_canonical(field, value);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if field.name() == "step" {
            self.step = Some(value.max(0) as u32);
            return;
        }
        self.record_canonical(field, value as f64);
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "step" {
            self.step = Some(value.min(u64::from(u32::MAX)) as u32);
            return;
        }
        self.record_canonical(field, value as f64);
    }

    fn record_str(&mut self, _field: &Field, _value: &str) {}
    fn record_debug(&mut self, _field: &Field, _value: &dyn fmt::Debug) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::writer::InMemoryRecordSink;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    fn with_layer<F: FnOnce()>(f: F) -> Arc<Mutex<InMemoryRecordSink>> {
        let probe: Arc<Mutex<InMemoryRecordSink>> = Arc::new(Mutex::new(InMemoryRecordSink::new()));
        let dyn_sink: Arc<Mutex<dyn RecordSink>> = probe.clone();
        // Open an episode so MetricSamples land somewhere.
        probe.lock().on_episode_start(0);
        let subscriber = tracing_subscriber::registry().with(RecordingLayer::new(dyn_sink));
        let _guard = subscriber.set_default();
        f();
        probe
    }

    #[test]
    fn is_canonical_metric_recognises_registry() {
        assert!(is_canonical_metric("policy_loss"));
        assert!(is_canonical_metric("best_fitness"));
        assert!(!is_canonical_metric("batch_size"));
    }

    #[test]
    fn canonical_field_lands_as_metric_sample() {
        let probe = with_layer(|| {
            tracing::info!(policy_loss = 0.5_f64, "ppo update");
        });
        let probe = probe.lock();
        let ep = &probe.episodes[&0];
        assert_eq!(ep.metrics.len(), 1);
        assert_eq!(ep.metrics[0].name, "policy_loss");
        assert!((ep.metrics[0].value - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn multiple_canonical_fields_emit_multiple_samples() {
        let probe = with_layer(|| {
            tracing::info!(
                policy_loss = 0.25_f64,
                entropy = 1.1_f64,
                approx_kl = 0.02_f64,
                "ppo update",
            );
        });
        let probe = probe.lock();
        let ep = &probe.episodes[&0];
        assert_eq!(ep.metrics.len(), 3);
        let names: Vec<_> = ep.metrics.iter().map(|m| m.name.clone()).collect();
        assert!(names.contains(&"policy_loss".to_string()));
        assert!(names.contains(&"entropy".to_string()));
        assert!(names.contains(&"approx_kl".to_string()));
    }

    #[test]
    fn unknown_field_does_not_emit() {
        let probe = with_layer(|| {
            tracing::info!(batch_size = 64_u64, "training step");
        });
        let probe = probe.lock();
        let ep = &probe.episodes[&0];
        assert!(ep.metrics.is_empty());
    }

    #[test]
    fn explicit_step_field_wins() {
        let probe = with_layer(|| {
            tracing::info!(step = 1024_u64, entropy = 0.7_f64, "labelled");
        });
        let probe = probe.lock();
        let ep = &probe.episodes[&0];
        assert_eq!(ep.metrics.len(), 1);
        assert_eq!(ep.metrics[0].step, 1024);
    }

    #[test]
    fn implicit_step_increments_across_events() {
        let probe = with_layer(|| {
            tracing::info!(policy_loss = 0.1_f64, "e1");
            tracing::info!(policy_loss = 0.2_f64, "e2");
            tracing::info!(policy_loss = 0.3_f64, "e3");
        });
        let probe = probe.lock();
        let ep = &probe.episodes[&0];
        assert_eq!(ep.metrics.len(), 3);
        let steps: Vec<u32> = ep.metrics.iter().map(|m| m.step).collect();
        assert_eq!(steps, vec![0, 1, 2]);
    }

    #[test]
    fn integer_canonical_field_coerces() {
        let probe = with_layer(|| {
            tracing::info!(entropy = 3_i64, "edge");
        });
        let probe = probe.lock();
        let ep = &probe.episodes[&0];
        assert_eq!(ep.metrics.len(), 1);
        assert!((ep.metrics[0].value - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn events_without_canonical_fields_are_ignored() {
        let probe = with_layer(|| {
            tracing::info!("just a message");
            tracing::warn!("with a level");
        });
        let probe = probe.lock();
        let ep = &probe.episodes[&0];
        assert!(
            ep.metrics.is_empty(),
            "messages without canonical fields should not produce metrics"
        );
    }
}
