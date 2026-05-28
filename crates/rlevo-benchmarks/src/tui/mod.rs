//! Live `ratatui` TUI for benchmark runs (feature `tui`).
//!
//! This module wires the styled environment output defined in
//! [`rlevo_core::render`] up to a terminal dashboard with an env panel,
//! reward sparkline, metric sparklines (loss / entropy / `approx_kl`),
//! and a scrolling log panel fed by `tracing`.
//!
//! Production crates (`rlevo-core`, `rlevo-environments`,
//! `rlevo-reinforcement-learning`, `rlevo-evolution`, `rlevo-hybrid`)
//! stay ratatui-free; every dependency on `ratatui` and `crossterm` is
//! gated behind this feature on the `rlevo-benchmarks` crate.

pub mod convert;
pub mod log_layer;
pub mod panels;
pub mod runner;
pub mod state;
pub mod theme;

pub use log_layer::{TuiCaptureLayer, CANONICAL_METRICS};
pub use runner::{TuiConfig, TuiError, TuiRunner, DEFAULT_TICK_MS};
