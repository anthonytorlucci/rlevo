//! Shared utilities used across `rlevo-core` consumers.
//!
//! Currently houses the [`seed`] module's deterministic seed-derivation
//! primitives (see [`seed::SeedStream`]).
//!
//! [`seed`]: crate::util::seed
//! [`seed::SeedStream`]: crate::util::seed::SeedStream

pub mod seed;
