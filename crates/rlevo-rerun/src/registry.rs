//! The runs being recorded in this process, where the tracing layer looks an
//! event's trial up.
//!
//! A process has one global `tracing` subscriber, set once, and may record
//! any number of runs, one after another or at once. So the layer is not a
//! sink's: it is the process's, and a sink registers its run here while the
//! run is under way.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use rlevo_harness::suite::TrialKey;

use crate::trial::TrialStream;

/// One sink's view of the run it is recording.
#[derive(Debug, Default)]
pub(crate) struct RunState {
    /// The open trials, by key.
    pub(crate) trials: Mutex<HashMap<TrialKey, Arc<TrialStream>>>,
    /// Update events of this run that named a trial not open in it.
    pub(crate) late: AtomicU64,
}

/// Runs by ID. More than one sink may record a run (two in a
/// `MultiReporter`), so an ID maps to each sink's state.
static RUNS: LazyLock<Mutex<HashMap<String, Vec<Arc<RunState>>>>> = LazyLock::new(Mutex::default);

/// Update events emitted outside any trial, in this process, ever.
static STRAY: AtomicU64 = AtomicU64::new(0);

/// Starts recording `run_id`, and returns the state to register trials in.
pub(crate) fn register(run_id: &str) -> Arc<RunState> {
    let state = Arc::new(RunState::default());
    RUNS.lock()
        .entry(run_id.to_string())
        .or_default()
        .push(Arc::clone(&state));
    state
}

/// Stops recording: `state` is no longer found under `run_id`.
pub(crate) fn unregister(run_id: &str, state: &Arc<RunState>) {
    let mut runs = RUNS.lock();
    if let Some(states) = runs.get_mut(run_id) {
        states.retain(|other| !Arc::ptr_eq(other, state));
        if states.is_empty() {
            runs.remove(run_id);
        }
    }
}

/// Every sink's state for `run_id`: empty for a run no sink is recording.
pub(crate) fn recording(run_id: &str) -> Vec<Arc<RunState>> {
    RUNS.lock().get(run_id).cloned().unwrap_or_default()
}

/// Counts an update event that was in no trial.
pub(crate) fn note_stray() {
    STRAY.fetch_add(1, Ordering::Relaxed);
}

/// How many update events have been in no trial, so far.
pub(crate) fn strays() -> u64 {
    STRAY.load(Ordering::Relaxed)
}
