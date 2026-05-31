//! Visualisation, recording, and reporting examples for `rlevo`.
//!
//! This crate contains application-tier examples that compose the full
//! viz/record/report stack. Lightweight single-subcrate demos live in
//! `crates/rlevo/examples/` instead.
//!
//! # Running examples
//!
//! ```bash
//! # Live TUI dashboard
//! cargo run -p rlevo-examples --example tui_ppo_cartpole --features viz-tui
//!
//! # Record a run to disk
//! cargo run -p rlevo-examples --example record_ppo_cartpole --features viz-tui,viz-record
//!
//! # Emit a static-HTML report
//! cargo run -p rlevo-examples --example report_ppo_cartpole_with_client \
//!     --features viz-record,viz-report
//! ```
