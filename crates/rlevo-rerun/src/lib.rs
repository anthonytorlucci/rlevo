//! A [rerun](https://rerun.io) recording sink for the `rlevo` evaluation
//! harness.
//!
//! This crate is the one place in the workspace that links rerun. The
//! harness ([`rlevo_harness`]) reports a run through its
//! [`Reporter`](rlevo_harness::reporter::Reporter) trait and knows nothing
//! about rerun; a sink here turns those reports into `.rrd` recordings that
//! the rerun viewer opens. No other published crate depends on this one by
//! default: the `rlevo` umbrella carries it only behind an off-by-default
//! feature.
//!
//! The crate exports nothing yet.
