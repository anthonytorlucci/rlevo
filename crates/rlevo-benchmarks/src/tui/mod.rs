//! Live `ratatui` TUI for benchmark runs (feature `tui`).
//!
//! This module wires the styled environment output defined in
//! [`rlevo_core::render`] up to a terminal dashboard. Milestone 2 of the
//! visualisation roadmap (see `projects/rlevo/specs/2026-05-26-env-vis/`
//! in the project vault) lands the skeleton — frame conversion, runner
//! lifecycle, env panel + reward sparkline. Later milestones add the
//! loss / fitness / log panels and the on-disk record format.
//!
//! Production crates (`rlevo-core`, `rlevo-environments`,
//! `rlevo-reinforcement-learning`, `rlevo-evolution`, `rlevo-hybrid`)
//! stay ratatui-free; every dependency on `ratatui` and `crossterm` is
//! gated behind this feature on the `rlevo-benchmarks` crate.

pub mod convert;
pub mod panels;
pub mod runner;
pub mod state;

pub use runner::{TuiConfig, TuiError, TuiRunner, DEFAULT_TICK_MS};
