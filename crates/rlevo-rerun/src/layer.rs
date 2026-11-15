//! [`UpdateLayer`]: update-rate `tracing` events into trial recordings.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use rlevo_core::fitness::UPDATE_TARGET;
use rlevo_harness::suite::TrialKey;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::{Event, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

use crate::sink::Shared;

/// A `tracing` layer that records update-rate events into the trial files of
/// the [`RerunSink`](crate::RerunSink) it came from
/// ([`RerunSink::layer`](crate::RerunSink::layer)).
///
/// It reads events at [`UPDATE_TARGET`] and no others. Of such an event,
/// every numeric field is a metric, logged at
/// `/metrics/<field name>/<run label>/<environment>/<trial>`; `step`, when
/// present, is the trial's step count and not a metric; anything that is not
/// a number is left out. No list of names is consulted, so a statistic an
/// outside learner logs is recorded as the workspace's own are. Each value
/// is on two timelines: `update`, the event's index among its trial's, and
/// `step`.
///
/// # How an event finds its trial
///
/// The harness runs each trial inside a `trial` span that names the run and
/// the trial. An event emitted in that span, on the trial's thread, is filed
/// under that trial, whichever worker thread runs it and whatever else runs
/// beside it. An event of another run is ignored. An event outside any trial
/// span (emitted by work the trial handed to another thread) cannot be
/// filed: it is counted, and
/// [`RerunSink::unattributed_updates`](crate::RerunSink::unattributed_updates)
/// reports the count, so nothing is dropped unseen.
#[derive(Debug, Clone)]
pub struct UpdateLayer {
    shared: Arc<Shared>,
}

impl UpdateLayer {
    pub(crate) fn new(shared: Arc<Shared>) -> Self {
        Self { shared }
    }
}

/// The trial a `trial` span belongs to, kept in the span's extensions.
#[derive(Debug, Clone)]
struct SpanTrial {
    run: String,
    key: TrialKey,
}

/// Reads a `trial` span's `run`, `env_idx` and `trial` fields.
#[derive(Default)]
struct SpanFields {
    run: Option<String>,
    env_idx: Option<u64>,
    trial: Option<u64>,
}

impl Visit for SpanFields {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "run" {
            self.run = Some(value.to_string());
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "env_idx" => self.env_idx = Some(value),
            "trial" => self.trial = Some(value),
            _ => {}
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

/// Reads an update event: its `step`, and every other number as a metric.
#[derive(Default)]
struct UpdateFields {
    step: Option<u64>,
    metrics: Vec<(&'static str, f64)>,
}

impl Visit for UpdateFields {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.metrics.push((field.name(), value));
    }

    // A count in an update event is far below 2^53, so the cast is exact.
    #[allow(clippy::cast_precision_loss)]
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "step" {
            self.step = Some(value);
        } else {
            self.metrics.push((field.name(), value as f64));
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn record_i64(&mut self, field: &Field, value: i64) {
        match (field.name(), u64::try_from(value)) {
            ("step", Ok(step)) => self.step = Some(step),
            ("step", Err(_)) => {}
            (name, _) => self.metrics.push((name, value as f64)),
        }
    }

    // The message, a string, a `bool`: not a number, so not a metric.
    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

impl<S> Layer<S> for UpdateLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let metadata = attrs.metadata();
        if metadata.name() != "trial" || metadata.target() != "rlevo_harness" {
            return;
        }
        let mut fields = SpanFields::default();
        attrs.record(&mut fields);
        let (Some(run), Some(env_idx), Some(trial)) = (fields.run, fields.env_idx, fields.trial)
        else {
            return;
        };
        let (Ok(env_idx), Ok(trial_idx)) = (usize::try_from(env_idx), usize::try_from(trial))
        else {
            return;
        };
        if let Some(span) = ctx.span(id) {
            // `replace`, not `insert`: another sink's layer in the same
            // subscriber writes the same value, and `insert` panics on a
            // second one.
            span.extensions_mut().replace(SpanTrial {
                run,
                key: TrialKey { env_idx, trial_idx },
            });
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        if event.metadata().target() != UPDATE_TARGET {
            return;
        }
        // The innermost enclosing `trial` span; the event may sit in spans
        // of the learner's own inside it.
        let trial = ctx.event_scope(event).and_then(|mut scope| {
            scope.find_map(|span| span.extensions().get::<SpanTrial>().cloned())
        });
        let Some(trial) = trial else {
            self.shared.unattributed.fetch_add(1, Ordering::Relaxed);
            return;
        };
        // Another run's trial, with a sink and a layer of its own.
        if trial.run != *self.shared.run_id.read() {
            return;
        }
        let entry = self.shared.trials.lock().get(&trial.key).cloned();
        let Some(entry) = entry.filter(|entry| entry.run_id() == trial.run) else {
            self.shared.unattributed.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let mut fields = UpdateFields::default();
        event.record(&mut fields);
        entry.log_update(fields.step, fields.metrics);
    }
}
