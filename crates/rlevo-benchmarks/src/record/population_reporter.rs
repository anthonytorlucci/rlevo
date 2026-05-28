//! EA-population producer: adapts a [`PopulationObserver`] to push
//! [`PopulationSample`] chunks into a shared [`RecordSink`].
//!
//! Wiring:
//!
//! ```text
//! EvolutionaryHarness::with_observer(reporter.clone())
//!     ↓ on_population(snapshot)         (per generation, post-tell)
//! PopulationReporter
//!     ↓ sink.lock().on_population_sample(sample)
//! Arc<Mutex<dyn RecordSink>>            (RecordWriter → .rec stream)
//! ```
//!
//! The producer is intentionally thin — the trait-to-chunk
//! translation is a 1:1 field copy plus an `inner_rl_returns = None`
//! default. A future hybrid driver attaches a separate observer that
//! fills `inner_rl_returns`.
//!
//! [`PopulationObserver`]: rlevo_evolution::PopulationObserver

use std::sync::{Arc, Mutex};

use rlevo_evolution::{PopulationObserver, PopulationSnapshot};

use super::schema::PopulationSample;
use super::writer::RecordSink;

/// Forwards EA population snapshots into the on-disk recording sink.
///
/// Construct one per recording run and pass via
/// [`EvolutionaryHarness::with_observer`](rlevo_evolution::EvolutionaryHarness::with_observer).
/// The reporter clones cheaply through the inner [`Arc`].
#[derive(Clone)]
pub struct PopulationReporter {
    sink: Arc<Mutex<dyn RecordSink>>,
}

impl std::fmt::Debug for PopulationReporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PopulationReporter")
            .field("sink", &"Arc<Mutex<dyn RecordSink>>")
            .finish()
    }
}

impl PopulationReporter {
    #[must_use]
    pub fn new(sink: Arc<Mutex<dyn RecordSink>>) -> Self {
        Self { sink }
    }
}

impl PopulationObserver for PopulationReporter {
    fn on_population(&mut self, snapshot: PopulationSnapshot) {
        let sample = PopulationSample {
            generation: snapshot.generation,
            fitnesses: snapshot.fitnesses,
            diversity: snapshot.diversity,
            best_index: snapshot.best_index,
            best_genome_digest: snapshot.best_genome_digest,
            parents_of_best: snapshot.parents_of_best,
            inner_rl_returns: None,
        };
        if let Ok(mut guard) = self.sink.lock() {
            guard.on_population_sample(sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::writer::InMemoryRecordSink;

    fn snapshot(generation: u32) -> PopulationSnapshot {
        PopulationSnapshot {
            generation,
            fitnesses: vec![0.5, 0.4, 0.3, 0.2, 0.1],
            diversity: Some(0.21),
            best_index: 4,
            best_genome_digest: Some([1u8; 16]),
            parents_of_best: vec![[2u8; 16]],
        }
    }

    #[test]
    fn reporter_forwards_snapshot_to_sink() {
        let probe: Arc<Mutex<InMemoryRecordSink>> =
            Arc::new(Mutex::new(InMemoryRecordSink::new()));
        let sink: Arc<Mutex<dyn RecordSink>> = probe.clone();
        {
            let mut s = sink.lock().unwrap();
            s.on_episode_start(0);
        }
        let mut reporter = PopulationReporter::new(sink);
        reporter.on_population(snapshot(0));
        reporter.on_population(snapshot(1));
        let guard = probe.lock().unwrap();
        let ep = guard.episodes.get(&0).expect("episode 0 created");
        assert_eq!(ep.population_samples.len(), 2);
        assert_eq!(ep.population_samples[0].generation, 0);
        assert_eq!(ep.population_samples[1].best_index, 4);
        // Producer fills None on the schema field the hybrid driver
        // owns — see M8.1 plan deferral.
        assert!(ep.population_samples[0].inner_rl_returns.is_none());
    }
}
