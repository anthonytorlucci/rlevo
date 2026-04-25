//! Renderers for `rlevo-envs` environments.
//!
//! The primary renderer is [`ascii::AsciiRenderer`], which produces
//! `String` frames for classic-control and toy-text environments.
//! Use [`rlevo_core::render::NullRenderer`] when rendering is not needed.
pub mod ascii;

pub use ascii::{AsciiRenderable, AsciiRenderer};
