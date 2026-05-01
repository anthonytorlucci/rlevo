//! Concrete evolutionary algorithms.
//!
//! # Classical families
//!
//! - [`ga`] / [`ga_binary`] — Genetic Algorithm (real- and binary-coded).
//! - [`es_classical`] — (1+1), (1+λ), (μ,λ), (μ+λ) Evolution Strategies.
//! - [`de`] — Differential Evolution (rand/best/current-to-best × bin/exp).
//! - [`ep`] — Evolutionary Programming (Fogel-style).
//! - [`gp_cgp`] — Cartesian Genetic Programming.
//!
//! # Swarm / nature-inspired metaheuristics
//!
//! - [`metaheuristic`] — PSO, `ACO_R`, ABC, GWO, WOA, CS, FA, BA, SSA.

pub mod de;
pub mod ep;
pub mod es_classical;
pub mod ga;
pub mod ga_binary;
pub mod gp_cgp;
pub mod metaheuristic;
