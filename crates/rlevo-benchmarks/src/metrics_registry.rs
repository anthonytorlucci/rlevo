//! Canonical metric registry — re-exported from the shared
//! [`rlevo-metrics-registry`] leaf crate (ADR-0015).
//!
//! Both tracing layers in this crate (the live-TUI [`TuiCaptureLayer`] and the
//! on-disk [`RecordingLayer`]) classify `tracing` field names against the same
//! table that the WASM report client uses. Before ADR-0015 the list was a flat
//! `&[&str]` here that was hand-copied into the report client with no guard;
//! the table now lives in one `#![no_std]` crate that both sides depend on, so
//! there is a single source of truth and nothing to keep in sync.
//!
//! Add a new metric by editing the table in `rlevo-metrics-registry`, not here.
//!
//! [`TuiCaptureLayer`]: crate::tui::log_layer::TuiCaptureLayer
//! [`RecordingLayer`]: crate::record::tracing_layer::RecordingLayer

pub use rlevo_metrics_registry::{
    Cadence, CANONICAL_METRICS, MetricDescriptor, MetricKind, descriptor, is_canonical_metric,
    is_per_generation, title_for,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_reachable_through_benchmarks() {
        // Smoke test: the re-export resolves and the well-known names the
        // recorder/TUI depend on are present. Exhaustive coverage lives in
        // the `rlevo-metrics-registry` crate's own test module.
        assert!(is_canonical_metric("policy_loss"));
        assert!(is_canonical_metric("best_fitness_ever"));
        assert!(is_canonical_metric("explained_variance"));
        assert!(!is_canonical_metric("not_a_metric"));
        assert!(!CANONICAL_METRICS.is_empty());
    }
}
