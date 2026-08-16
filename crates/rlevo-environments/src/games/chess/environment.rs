//! Chess `Environment` trait implementation (work in progress).
//!
//! This module is a placeholder. The surrounding `games` module already has
//! board representation, move generation, and observation types
//! ([`super::observations`], e.g. `ChessState`, `CastlingRights`, repetition
//! history), but nothing here implements
//! [`rlevo_core::environment::Environment`] for chess yet — this file, and
//! its `ConnectFour` counterpart in `games::connect_four`, are stubs.
//! That is why `games` is still `#[doc(hidden)]` with a module-level
//! `#[allow(dead_code, ...)]`: nothing downstream consumes it. Completing
//! this impl (wrapping [`super::board`] state, [`super::moves`] encoding,
//! and episode lifecycle into `Environment`) is what lets `games` drop
//! `#[doc(hidden)]`, drop the dead-code allow, and be listed as a real
//! environment family. There is no near-term timeline: it is deliberately
//! deferred, not part of the current roadmap (which is oriented toward
//! model-based RL/MPC, POMDPs, and robotics), and is left here in source
//! rather than as undocumented folklore.
