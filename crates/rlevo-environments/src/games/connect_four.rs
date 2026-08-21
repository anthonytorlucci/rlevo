//! Connect Four environment (work in progress).
//!
//! This module is a placeholder: it compiles but does not yet implement
//! [`rlevo_core::environment::Environment`] for `ConnectFour`. Along with its
//! sibling `games::chess::environment` stub, it is why the parent `games`
//! module is `#[doc(hidden)]` and carries a module-level
//! `#[allow(dead_code, ...)]` — nothing downstream consumes either
//! environment yet. Once both `Environment` impls land, `games` can drop the
//! `#[doc(hidden)]` and the dead-code allow and be listed as a real
//! environment family. There is no fixed timeline for this work; it falls
//! outside the current roadmap's focus on model-based RL/MPC, POMDPs, and
//! robotics.
