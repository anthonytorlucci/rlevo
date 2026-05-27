//! Composable wrappers around the [`BenchEnv`] trait.
//!
//! Wrappers in this module add cross-cutting behaviour (frame capture,
//! tracing hooks, …) without touching the rollout loop in
//! [`Evaluator`]. Each wrapper newtypes a `BenchEnv` implementor and
//! delegates the core methods through, so the wrapper composes with any
//! agent that already knows how to drive the underlying env.
//!
//! [`BenchEnv`]: rlevo_core::evaluation::BenchEnv
//! [`Evaluator`]: crate::evaluator::Evaluator

pub mod render_tap;

pub use render_tap::RenderTap;
