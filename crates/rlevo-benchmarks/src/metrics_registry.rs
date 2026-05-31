//! Canonical metric field-name registry shared by all tracing layers.
//!
//! Both the live-TUI layer (`TuiCaptureLayer`) and
//! the on-disk recording layer (`RecordingLayer`)
//! extract the same named fields from `tracing` events. Keeping the registry
//! here means a single edit covers both surfaces — the two layers just
//! import [`CANONICAL_METRICS`] instead of duplicating it.

/// Field names recognised as chartable metrics across all tracing layers.
///
/// Any `tracing` event field whose name appears in this list is treated as
/// a numeric metric sample. Everything else is captured only as a log line
/// (TUI) or ignored (record layer).
///
/// # Extending the registry
///
/// Add the field name here when a new algorithm starts emitting a metric.
/// That single edit makes the field visible in both the live TUI sparklines
/// and the on-disk recording stream.
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

/// `true` if `name` is a recognised metric field name.
#[must_use]
pub fn is_canonical_metric(name: &str) -> bool {
    CANONICAL_METRICS.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_names_are_recognised() {
        for name in CANONICAL_METRICS {
            assert!(is_canonical_metric(name), "{name} should be canonical");
        }
    }

    #[test]
    fn unknown_names_are_rejected() {
        assert!(!is_canonical_metric("batch_size"));
        assert!(!is_canonical_metric("not_a_metric"));
        assert!(!is_canonical_metric(""));
    }
}
