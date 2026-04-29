//! Agent traits — reserved for a future release.
//!
//! This module is intentionally empty in v0.1.0. The planned trait hierarchy
//! will distinguish gradient-based reinforcement-learning agents from
//! population-based evolutionary agents, with a shared surface for
//! `act` / `learn` / checkpoint operations.
//!
//! Concrete algorithm implementations currently live in the `rlevo-reinforcement-learning` and
//! `rlevo-evolution` crates. They will migrate behind the traits defined
//! here once the API stabilizes.
