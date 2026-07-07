//! Live `ratatui` terminal dashboard for benchmark runs (feature `tui`).
//!
//! A metrics-only terminal UI (ADR-0013): a reward-return sparkline,
//! per-metric sparklines (loss, entropy, `approx_kl`, `best_fitness`), and a
//! scrolling log panel driven by a `tracing` subscriber layer. The live TUI
//! answers *"is it learning?"*; env playback lives in the post-run report,
//! not here.
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
//! | `convert`    | Maps [`Color`] / [`SpanStyle`] to `ratatui` styles for the palette |
//! | `log_layer`  | `tracing` subscriber that captures events into the TUI channel  |
//! | `panels`     | Individual panel widgets (reward/metric sparklines, log)        |
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
//! let runner = TuiRunner::start(TuiConfig::default()).unwrap();
//! let handle = runner.handle();
//! // pass `handle.as_reporter()` to the Evaluator and `handle.clone()` to
//! // the TuiCaptureLayer; the render thread started by `start` drains the
//! // channel until `runner.shutdown()`.
//! # let _ = handle;
//! ```
//!
//! [`Color`]: rlevo_core::render::Color
//! [`SpanStyle`]: rlevo_core::render::SpanStyle
//! [`AppState`]: crate::tui::state::AppState

pub mod convert;
pub mod log_layer;
pub mod panels;
pub mod runner;
pub mod state;
pub mod theme;

#[doc(inline)]
pub use log_layer::{CANONICAL_METRICS, TuiCaptureLayer};
#[doc(inline)]
pub use runner::{DEFAULT_TICK_MS, TuiConfig, TuiError, TuiRunner};
#[doc(inline)]
pub use state::MetricsLayout;
