//! A [rerun](https://rerun.io) recording sink for the `rlevo` evaluation
//! harness.
//!
//! This crate is the one place in the workspace that links rerun. The
//! harness ([`rlevo_harness`]) reports a run through its
//! [`Reporter`](rlevo_harness::reporter::Reporter) trait and knows nothing
//! about rerun; [`RerunSink`] is a `Reporter` that turns those reports into
//! `.rrd` recordings the rerun viewer opens. No rerun type appears in this
//! crate's API, so a rerun upgrade is not a breaking change of it.
//!
//! # What a run writes
//!
//! One file per trial, all under one recording ID (the run's, which the
//! harness draws and a resumed run keeps), so the viewer shows the files of a
//! run as one recording and overlays its seeds. A trial's file is
//!
//! ```text
//! <output dir>/<run ID>/<run label>.<environment index>-<environment>.<trial index>.rrd
//! ```
//!
//! with free text cleaned to letters, digits, `_` and `-`. The run ID is in
//! the harness's report ([`BenchmarkReport::run_id`](rlevo_harness::report::BenchmarkReport::run_id)), which is what joins a
//! report's JSON to its recordings. Each file holds, for its trial:
//!
//! - every episode's return and length, training and evaluation apart;
//! - every metric the trial's agent or generation loop reported, at each
//!   point the harness read it;
//! - every update's statistics that a learner logs as a `tracing` event at
//!   [`UPDATE_TARGET`](rlevo_core::fitness::UPDATE_TARGET), when
//!   [`UpdateLayer`] is installed;
//! - the trial's identity, task and build, as properties of the recording;
//! - a status entry, when the trial ended in an error.
//!
//! Values sit at `/metrics/<metric>/<run label>/<environment>/<trial>`, the
//! metric first, so one view rooted at a metric overlays every trial. The
//! environment is its ID; in a suite that has one ID more than once, each of
//! those is `<environment index>-<ID>`, so the two do not share a path.
//!
//! # Examples
//!
//! ```no_run
//! use rlevo_rerun::{RerunConfig, RerunSink};
//!
//! let mut sink = RerunSink::new(RerunConfig::new("runs", "ppo"))?;
//! // Hand `&mut sink` to `Evaluator::run_suite` or `run_trials`, then:
//! if let Some(how) = sink.viewer_instructions() {
//!     println!("{how}");
//! }
//! # Ok::<(), rlevo_rerun::RerunError>(())
//! ```

mod layer;
mod registry;
mod sink;
mod trial;

pub use layer::UpdateLayer;
pub use sink::{RerunConfig, RerunError, RerunSink};
