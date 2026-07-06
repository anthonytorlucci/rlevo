//! Swarm-intelligence and nature-inspired metaheuristics.
//!
//! Each algorithm implements the [`Strategy`](crate::Strategy) trait over
//! tensor-backed populations so it plugs into the same
//! [`EvolutionaryHarness`](crate::EvolutionaryHarness) pipeline as the
//! classical families (GA, ES, DE, EP, CGP).
//!
//! # Shipping algorithms
//!
//! | Module | Algorithm | Genome kind | Status |
//! |---|---|---|---|
//! | [`pso`] | Particle Swarm Optimization | Real | Solid baseline |
//! | [`aco_r`] | Ant Colony Optimization (continuous) | Real | Niche but principled |
//! | [`abc`] | Artificial Bee Colony | Real | Competitive on simple multimodal |
//! | [`gwo`] | Grey Wolf Optimizer | Real | Legacy comparator |
//! | [`woa`] | Whale Optimization Algorithm | Real | Legacy comparator |
//! | [`cuckoo`] | Cuckoo Search | Real | Lévy flights + random walk |
//! | [`firefly`] | Firefly Algorithm | Real | Useful for multimodal |
//! | [`bat`] | Bat Algorithm | Real | Legacy comparator |
//! | [`salp`] | Salp Swarm Algorithm | Real | Legacy comparator |
//! | [`aco_perm`] | Ant Colony (permutation) | Permutation | **Stub** — deferred to a future release |
//!
//! # Calibration
//!
//! Not every algorithm in this module is competitive on serious
//! benchmarks. Several (GWO, WOA, BA, SSA) are flagged as
//! "legacy comparator" in their module docs, per Camacho Villalón et al.
//! (2020) and Sörensen (2015). The library ships them because they are
//! widely cited; users asking "which one should I pick?" should start
//! with [`pso`] and move to CMA-ES or LSHADE once those land.
//!
//! See `README.md` in this module's source directory for the full
//! calibration table and prose guidance.
//!
//! # Custom kernels
//!
//! The [`kernels`] submodule ships fused `CubeCL` kernels for two
//! operator paths where the pure-tensor decomposition is measurably
//! wasteful: the pairwise-attract inner loop used by
//! [`firefly`] and the Lévy-flight sampling used by
//! [`cuckoo`] (and optionally [`bat`]). The kernels are gated behind
//! the crate-level `custom-kernels` feature; pure-tensor fallbacks
//! always compile.

pub mod abc;
pub mod aco_perm;
pub mod aco_r;
pub mod bat;
pub mod cuckoo;
pub mod firefly;
pub mod gwo;
pub mod kernels;
pub mod pso;
pub mod salp;
pub mod woa;

use rlevo_core::config::{ConfigError, ConstraintKind};

/// Rejects a host-side per-individual vector whose length is neither `pop`
/// nor `0`.
///
/// Swarm `*State` structs cache one host-side scalar per individual (fitness,
/// trial counters, loudness, …). Every such vector must have length `pop`,
/// except at bootstrap where `init` leaves the caches empty until the first
/// `tell`. This is the shared length invariant checked by each state's
/// `try_new`, so a mismatched non-empty length is rejected at construction
/// rather than surfacing as an out-of-bounds panic generations later.
pub(crate) fn len_matches_pop(
    config: &'static str,
    field: &'static str,
    pop: usize,
    len: usize,
) -> Result<(), ConfigError> {
    if len == 0 || len == pop {
        Ok(())
    } else {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::Custom("per-individual vector length must equal pop_size"),
        })
    }
}
