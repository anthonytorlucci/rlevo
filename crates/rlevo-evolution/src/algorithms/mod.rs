//! Concrete evolutionary algorithms.
//!
//! # Classical families
//!
//! - [`ga`] / [`ga_binary`] — Genetic Algorithm (real- and binary-coded).
//! - [`es_classical`] — (1+1), (1+λ), (μ,λ), (μ+λ) Evolution Strategies.
//! - [`cma_es`] — Covariance Matrix Adaptation ES (CSA + evolution paths +
//!   rank-1/rank-μ covariance updates).
//! - [`cmsa_es`] — Covariance Matrix Self-Adaptation ES (path-free; per-individual
//!   log-normal σ + rank-μ ML covariance blend).
//! - [`de`] — Differential Evolution (rand/best/current-to-best × bin/exp).
//! - [`ep`] — Evolutionary Programming (Fogel-style).
//! - [`gp_cgp`] — Cartesian Genetic Programming.
//! - [`gep`] — Gene Expression Programming (linear head/tail genome decoded to
//!   an expression tree).
//!
//! # Swarm / nature-inspired metaheuristics
//!
//! - [`metaheuristic`] — PSO, `ACO_R`, ABC, GWO, WOA, CS, FA, BA, SSA.
//!
//! # Estimation-of-distribution algorithms
//!
//! - [`eda`] — [`EdaStrategy`](eda::EdaStrategy) over a
//!   [`ProbabilityModel`](crate::ProbabilityModel): UMDA, PBIL, compact-GA,
//!   a dependency-chain MIMIC model, and a BIC-scored Bayesian network (BOA).
//!
//! # Hybrid / composite strategies
//!
//! - [`memetic`] — [`MemeticWrapper`](memetic::MemeticWrapper): wraps any
//!   real-valued strategy with per-individual local-search refinement
//!   (Lamarckian / Baldwinian / Partial writeback).
//! - [`neuroevolution`] — [`WeightOnly`](neuroevolution::WeightOnly): wraps any
//!   real-valued strategy to evolve the flattened weights of a Burn `Module`.

pub mod cma_es;
pub mod cmsa_es;
pub mod de;
pub mod eda;
pub mod ep;
pub mod es_classical;
pub mod ga;
pub mod ga_binary;
pub mod gep;
pub mod gp_cgp;
pub mod memetic;
pub mod metaheuristic;
pub mod neuroevolution;
