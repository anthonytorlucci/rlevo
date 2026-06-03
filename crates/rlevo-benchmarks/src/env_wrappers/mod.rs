//! Composable wrappers around environment traits.
//!
//! Wrappers in this module add cross-cutting behaviour (episode-return
//! tallying, tracing hooks, …) without touching the rollout loop in
//! [`Evaluator`].
//!
//! - [`TuiEnvTap`] wraps a raw [`Environment`] for training loops that
//!   bypass the harness (PPO's `train_discrete`, future evolutionary
//!   loops). It owns episode-return accumulation since no `Reporter` is
//!   involved, feeding the live TUI's reward sparkline. The live TUI is
//!   metrics-only (ADR-0013), so the tap no longer captures env frames.
//!
//! [`Environment`]: rlevo_core::environment::Environment
//! [`Evaluator`]: crate::evaluator::Evaluator

pub mod tui_env_tap;

pub use tui_env_tap::TuiEnvTap;
