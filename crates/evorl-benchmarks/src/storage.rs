//! Reserved for durable storage of benchmark reports.
//!
//! This module is a placeholder for a future queryable backing store that
//! will persist `BenchmarkReport`s beyond the single-file output produced
//! by `JsonReporter` (e.g. an append-only log, an embedded database, or a
//! cloud object-store adapter).
//!
//! No contract exists in 0.1.0 and no public items should be added until
//! one is specified. Consumers should use `reporter::json::JsonReporter`
//! and `checkpoint` for the report-persistence paths that ship today.
