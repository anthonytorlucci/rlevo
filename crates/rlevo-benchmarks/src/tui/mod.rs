//! Live `ratatui` terminal dashboard for benchmark runs (feature `tui`).
//!
//! Wires the styled environment output defined in [`rlevo_core::render`] into
//! a four-panel terminal UI: an ASCII env canvas, a reward-return sparkline,
//! per-metric sparklines (loss, entropy, `approx_kl`), and a scrolling log
//! panel driven by a `tracing` subscriber layer.
//!
//! All `ratatui` and `crossterm` dependencies are confined to this module;
//! production crates (`rlevo-core`, `rlevo-environments`,
//! `rlevo-reinforcement-learning`, `rlevo-evolution`, `rlevo-hybrid`) remain
//! ratatui-free.
//!
//! ## Sub-modules
//!
//! | Module       | Purpose                                                          |
//! |--------------|------------------------------------------------------------------|
//! | `convert`    | Maps [`StyledFrame`] / [`SpanStyle`] to `ratatui` `Span`s       |
//! | `log_layer`  | `tracing` subscriber that captures events into the TUI channel  |
//! | `panels`     | Individual panel widgets (env canvas, sparklines, log)          |
//! | `runner`     | Top-level [`TuiRunner`] event loop and [`TuiConfig`]            |
//! | `state`      | [`AppState`] — in-memory model the render thread writes to      |
//! | `theme`      | Colour palette and `ratatui` style constants                    |
//!
//! ## Entry point
//!
//! Most callers need only [`TuiRunner`], [`TuiConfig`], and the
//! [`TuiCaptureLayer`] tracing subscriber, all of which are re-exported at
//! this level:
//!
//! ```no_run
//! # use rlevo_benchmarks::tui::{TuiConfig, TuiRunner};
//! let (runner, handle) = TuiRunner::new(TuiConfig::default());
//! // pass `handle.as_reporter()` to the Evaluator and spawn `runner.run(rx)`.
//! ```
//!
//! [`StyledFrame`]: rlevo_core::render::StyledFrame
//! [`SpanStyle`]: rlevo_core::render::SpanStyle
//! [`AppState`]: crate::tui::state::AppState

pub mod convert;
pub mod log_layer;
pub mod panels;
pub mod runner;
pub mod state;
pub mod theme;

pub use log_layer::{TuiCaptureLayer, CANONICAL_METRICS};
pub use runner::{TuiConfig, TuiError, TuiRunner, DEFAULT_TICK_MS};
