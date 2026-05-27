//! Composable wrappers around environment traits.
//!
//! Wrappers in this module add cross-cutting behaviour (frame capture,
//! tracing hooks, …) without touching the rollout loop in
//! [`Evaluator`]. Two trait flavours are wrapped:
//!
//! - [`RenderTap`] wraps a [`BenchEnv`] for the benchmarks harness flow
//!   ([`Suite`](crate::suite::Suite) + [`Evaluator`]).
//! - [`TuiEnvTap`] wraps a raw [`Environment`] for training loops that
//!   bypass the harness (PPO's `train_discrete`, future evolutionary
//!   loops). It owns episode-return accumulation since no `Reporter` is
//!   involved.
//!
//! [`BenchEnv`]: rlevo_core::evaluation::BenchEnv
//! [`Environment`]: rlevo_core::environment::Environment
//! [`Evaluator`]: crate::evaluator::Evaluator

pub mod render_tap;
pub mod tui_env_tap;

pub use render_tap::RenderTap;
pub use tui_env_tap::TuiEnvTap;
