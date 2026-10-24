//! Visualisation, recording, and reporting examples for `rlevo`.
//!
//! This crate contains application-tier examples that compose the full
//! viz/record/report stack. Lightweight single-subcrate demos live in
//! `crates/rlevo/examples/` instead.
//!
//! One viz product (ADRs 0013, 0096): the post-run record + report pipeline
//! (`viz-report`, which pulls recording in transitively).
//!
//! # Running examples
//!
//! ```bash
//! # Record a run and emit a static-HTML report
//! cargo run -p rlevo-examples --example report_ppo_cartpole_with_client \
//!     --features viz-report
//! ```
