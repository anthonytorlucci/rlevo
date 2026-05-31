//! Per-generation observer surface for [`EvolutionaryHarness`].
//!
//! The harness emits a scalar tracing event per generation
//! (`best_fitness`, `mean_fitness`, …) which feeds the on-disk metric
//! stream. Population-level reporting — the full fitness vector, the
//! best-individual digest, parent lineage — does not fit through
//! `tracing::info!` cleanly: events are string-keyed and scalar-valued.
//! [`PopulationObserver`] is the structured callback the EA-population
//! recorder (`rlevo_benchmarks::record::PopulationReporter`) attaches
//! to to capture that shape.
//!
//! The trait is intentionally narrow — the snapshot fields mirror the
//! report-tier `PopulationSample` schema 1:1 minus the `inner_rl_returns`
//! field, which is the hybrid driver's responsibility.
//!
//! [`EvolutionaryHarness`]: crate::strategy::EvolutionaryHarness

use std::sync::Arc;

use parking_lot::Mutex;

/// Per-generation population snapshot delivered to a
/// [`PopulationObserver`].
///
/// Fields mirror the on-disk `PopulationSample` schema; the harness
/// emits one of these after every successful
/// [`Strategy::tell`](crate::Strategy::tell) call when an observer is
/// attached.
///
/// **Field semantics**:
///
/// - `fitnesses` is the full per-individual fitness vector (lower is
///   better, per the project minimization convention).
/// - `diversity` is currently `None` — the harness has no
///   strategy-agnostic geometry over the population tensor. A future
///   `Strategy::diversity` extension fills it in.
/// - `best_genome_digest` and `parents_of_best` are emitted empty /
///   `None` until per-strategy digest + parent-tracking lands (will
///   feed a lineage DAG panel).
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationSnapshot {
    pub generation: u32,
    pub fitnesses: Vec<f32>,
    pub diversity: Option<f32>,
    pub best_index: u32,
    pub best_genome_digest: Option<[u8; 16]>,
    pub parents_of_best: Vec<[u8; 16]>,
}

/// Callback the harness invokes once per generation, after
/// [`Strategy::tell`](crate::Strategy::tell) has returned and the
/// canonical `tracing::info!` aggregate event has been emitted.
///
/// `Send + 'static` so observers can sit behind
/// [`Arc<Mutex<dyn PopulationObserver>>`](SharedPopulationObserver) and
/// be shared across rayon worker threads — same shape as the
/// `RecordSink` trait in `rlevo-benchmarks`.
pub trait PopulationObserver: Send + 'static {
    fn on_population(&mut self, snapshot: PopulationSnapshot);
}

/// Shared handle to a [`PopulationObserver`]. Backed by
/// [`parking_lot::Mutex`] so the observer handle and the recording-tier
/// sinks in `rlevo-benchmarks` share one lock type (ADR 0010); aliased so
/// call sites do not have to spell out the `Arc<Mutex<…>>` shape every time.
pub type SharedPopulationObserver = Arc<Mutex<dyn PopulationObserver>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct CollectingObserver {
        snapshots: Vec<PopulationSnapshot>,
    }

    impl PopulationObserver for CollectingObserver {
        fn on_population(&mut self, snapshot: PopulationSnapshot) {
            self.snapshots.push(snapshot);
        }
    }

    #[test]
    fn collecting_observer_records_each_call() {
        let obs = Arc::new(Mutex::new(CollectingObserver::default()));
        // Type-check that the concrete handle can be coerced to the
        // dyn-shape the harness builder accepts.
        let _shared: SharedPopulationObserver = obs.clone();
        for g in 0..3 {
            let snapshot = PopulationSnapshot {
                generation: g,
                fitnesses: vec![1.0, 2.0, 3.0],
                diversity: None,
                best_index: 0,
                best_genome_digest: None,
                parents_of_best: Vec::new(),
            };
            obs.lock().on_population(snapshot);
        }
        let guard = obs.lock();
        assert_eq!(guard.snapshots.len(), 3);
        assert_eq!(guard.snapshots[2].generation, 2);
    }
}
