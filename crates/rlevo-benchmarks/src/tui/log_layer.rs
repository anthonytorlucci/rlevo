//! [`TuiCaptureLayer`] — a `tracing` Layer that fans every event into the
//! TUI channel as both a [`TuiEvent::LogLine`] (for the scrolling log
//! panel) and, when the event carries numeric fields matching the
//! canonical metric registry, one or more [`TuiEvent::MetricUpdate`]
//! samples (for the sparkline panels).
//!
//! # Wiring contract
//!
//! Production algorithms emit progress via structured `tracing::info!`
//! events with named fields:
//!
//! ```ignore
//! tracing::info!(
//!     policy_loss = stats.policy_loss,
//!     entropy = stats.entropy,
//!     approx_kl = stats.approx_kl,
//!     "ppo training progress",
//! );
//! ```
//!
//! Field names matching the [`CANONICAL_METRICS`] registry become metric
//! samples. Everything else is captured only as a log line. Adding a new
//! algorithm that emits its own metric names means extending the
//! constant — the registry is the contract.
//!
//! # Threading
//!
//! `tracing` invokes `on_event` from whatever thread emitted the event.
//! Rayon worker threads, the render thread, and the main thread can all
//! drive the Layer concurrently. `TuiHandle` is `Send + Sync`, so cloning
//! it across the boundary is safe; the underlying `mpsc::Sender` handles
//! the serialisation.
//!
//! Backpressure stays lossy: if the receiver has been dropped (render
//! thread gone), `try_push_*` returns `false` and the event is silently
//! discarded. The hot path never blocks on render state.

use std::fmt::{self, Write as _};

use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

use crate::reporter::tui::TuiHandle;

/// Field names the Layer extracts as metric samples. Anything outside
/// this list still lands in the log panel via the message capture, but
/// does not produce a [`TuiEvent::MetricUpdate`].
///
/// Update this constant when a new algorithm starts emitting a metric
/// the TUI should chart. Adding a name here is the *only* change needed
/// — the panel side reads the same registry to decide which sparklines
/// to draw.
///
/// [`TuiEvent::MetricUpdate`]: crate::reporter::tui::TuiEvent::MetricUpdate
pub const CANONICAL_METRICS: &[&str] = &[
    // RL training stats
    "policy_loss",
    "value_loss",
    "loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    // Evolution training stats emitted by `EvolutionaryHarness`.
    "best_fitness",
    "mean_fitness",
    "worst_fitness",
    "best_fitness_ever",
];

/// `true` if `name` is one of the recognised metric field names.
#[must_use]
pub fn is_canonical_metric(name: &str) -> bool {
    CANONICAL_METRICS.contains(&name)
}

/// `tracing_subscriber::Layer` that captures events for the live TUI.
///
/// Cloneable so the same Layer can be installed alongside other layers
/// (e.g. a file-backed log layer) via the standard `Registry::with`
/// pattern.
#[derive(Debug, Clone)]
pub struct TuiCaptureLayer {
    handle: TuiHandle,
}

impl TuiCaptureLayer {
    /// Construct the Layer wrapped around an outgoing [`TuiHandle`].
    #[must_use]
    pub const fn new(handle: TuiHandle) -> Self {
        Self { handle }
    }
}

impl<S> Layer<S> for TuiCaptureLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = CaptureVisitor::default();
        event.record(&mut visitor);

        for (name, value) in visitor.metrics {
            let _ = self.handle.try_push_metric(name, value);
        }

        let meta = event.metadata();
        let _ = self.handle.try_push_log(
            *meta.level(),
            meta.target().to_string(),
            visitor.message,
        );
    }
}

/// Per-event visitor that collects (a) any canonical numeric fields and
/// (b) the formatted message body.
///
/// The visitor allocates on every event — acceptable because the Layer
/// runs at `tracing::info!` cadence (a handful per second), not on the
/// rollout hot path.
#[derive(Default)]
struct CaptureVisitor {
    metrics: Vec<(String, f64)>,
    message: String,
}

impl CaptureVisitor {
    fn record_canonical(&mut self, field: &Field, value: f64) {
        let name = field.name();
        if is_canonical_metric(name) {
            self.metrics.push((name.to_string(), value));
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "metric panels need f64; integer counters lose precision only at extreme magnitudes the sparkline already truncates"
)]
impl Visit for CaptureVisitor {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.record_canonical(field, value);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_canonical(field, value as f64);
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record_canonical(field, value as f64);
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        // String fields never become metrics, but the literal message
        // field (when callers pass a `String` instead of `format_args!`)
        // still drives the log panel.
        if field.name() == "message" {
            self.message.push_str(value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            // `tracing`'s message is typically a `format_args!` Arguments;
            // Debug-formatting it strips the surrounding quotes that a
            // string Debug would add.
            let _ = write!(&mut self.message, "{value:?}");
        }
        // All other non-numeric fields are ignored. They neither become
        // metrics nor enrich the log line; future work can append them
        // if the log panel grows a structured field display.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::Receiver;
    use std::time::Duration;

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    use crate::reporter::tui::TuiEvent;

    /// Build a Layer + receiver pair, install the subscriber for the
    /// duration of `f`, and return the receiver so the caller can drain
    /// it. `set_default` is scoped — the previous global subscriber is
    /// restored when the guard drops, so tests don't bleed into each
    /// other.
    fn with_layer<F: FnOnce()>(f: F) -> Receiver<TuiEvent> {
        let (handle, rx) = TuiHandle::channel();
        let subscriber =
            tracing_subscriber::registry().with(TuiCaptureLayer::new(handle));
        let _guard = subscriber.set_default();
        f();
        // Give the runtime a beat to flush — usually unnecessary since
        // tracing is synchronous, but cheap insurance against future
        // buffered layers added higher in the stack.
        std::thread::sleep(Duration::from_millis(1));
        rx
    }

    fn drain(rx: &Receiver<TuiEvent>) -> Vec<TuiEvent> {
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        out
    }

    /// A canonical numeric field on an event lands as `MetricUpdate`.
    #[test]
    fn numeric_field_emits_metric_update() {
        let rx = with_layer(|| {
            tracing::info!(policy_loss = 0.5_f64, "ppo update");
        });
        let events = drain(&rx);

        let metrics: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                TuiEvent::MetricUpdate { name, value } => Some((name.clone(), *value)),
                _ => None,
            })
            .collect();
        assert_eq!(metrics, vec![("policy_loss".to_string(), 0.5)]);
    }

    /// Multiple canonical fields on one event produce one `MetricUpdate`
    /// each. PPO emits up to five fields per training-progress event.
    #[test]
    fn multiple_canonical_fields_emit_multiple_updates() {
        let rx = with_layer(|| {
            tracing::info!(
                policy_loss = 0.25_f64,
                entropy = 1.1_f64,
                approx_kl = 0.02_f64,
                "ppo update",
            );
        });
        let events = drain(&rx);

        let mut names: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                TuiEvent::MetricUpdate { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "approx_kl".to_string(),
                "entropy".to_string(),
                "policy_loss".to_string(),
            ]
        );
    }

    /// A non-canonical numeric field (e.g. `batch_size`) lands in the
    /// log but does NOT produce a `MetricUpdate`.
    #[test]
    fn unknown_numeric_field_ignored_for_metrics() {
        let rx = with_layer(|| {
            tracing::info!(batch_size = 64_u64, "training step");
        });
        let events = drain(&rx);

        assert!(
            !events
                .iter()
                .any(|e| matches!(e, TuiEvent::MetricUpdate { .. })),
            "batch_size should not produce a MetricUpdate"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TuiEvent::LogLine { .. })),
            "log line should still land"
        );
    }

    /// Integer canonical fields coerce to `f64` (the panel ingests f64
    /// uniformly).
    #[test]
    fn integer_canonical_field_coerces_to_f64() {
        let rx = with_layer(|| {
            // `generation` isn't currently canonical, so use a stable
            // canonical name. `entropy` as an integer makes no physical
            // sense but exercises the coercion path.
            tracing::info!(entropy = 3_i64, "edge case");
        });
        let events = drain(&rx);
        let metric = events.iter().find_map(|e| match e {
            TuiEvent::MetricUpdate { name, value } if name == "entropy" => Some(*value),
            _ => None,
        });
        assert_eq!(metric, Some(3.0));
    }

    /// The log capture carries level, target, and the formatted message.
    #[test]
    fn log_line_carries_level_target_and_message() {
        let rx = with_layer(|| {
            tracing::warn!("careful now");
        });
        let events = drain(&rx);
        let line = events
            .iter()
            .find_map(|e| match e {
                TuiEvent::LogLine {
                    level,
                    target,
                    message,
                } => Some((*level, target.clone(), message.clone())),
                _ => None,
            })
            .expect("expected at least one LogLine");
        assert_eq!(line.0, tracing::Level::WARN);
        // tracing module path lands in `target` — sufficient to assert
        // it's non-empty and contains the crate name.
        assert!(line.1.contains("rlevo_benchmarks"), "target was {:?}", line.1);
        assert!(
            line.2.contains("careful now"),
            "message body was {:?}",
            line.2
        );
    }

    /// Every event yields exactly one `LogLine`, regardless of whether it
    /// also produced metric updates.
    #[test]
    fn one_log_line_per_event_even_with_metrics() {
        let rx = with_layer(|| {
            tracing::info!(policy_loss = 0.1_f64, entropy = 0.5_f64, "two metrics");
        });
        let events = drain(&rx);
        let log_count = events
            .iter()
            .filter(|e| matches!(e, TuiEvent::LogLine { .. }))
            .count();
        assert_eq!(log_count, 1);
    }

    /// Dropping the receiver before events fire must not panic.
    /// `try_push_*` swallows the `SendError` internally.
    #[test]
    fn lossy_when_receiver_dropped() {
        let (handle, rx) = TuiHandle::channel();
        drop(rx);
        let subscriber =
            tracing_subscriber::registry().with(TuiCaptureLayer::new(handle));
        let _guard = subscriber.set_default();

        // Must not panic.
        tracing::info!(policy_loss = 0.5_f64, "lost in space");
        tracing::error!("also lost");
    }

    #[test]
    fn is_canonical_metric_accepts_registry_entries() {
        assert!(is_canonical_metric("policy_loss"));
        assert!(is_canonical_metric("best_fitness"));
        assert!(!is_canonical_metric("not_a_metric"));
        assert!(!is_canonical_metric(""));
    }
}
