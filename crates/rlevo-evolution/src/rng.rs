//! Deterministic seed derivation for strategies.
//!
//! Callers derive sub-seeds by mixing a `base`, a `generation` index, and
//! a [`SeedPurpose`] so parallel streams (selection, mutation, crossover)
//! do not alias. The mixer is [`rlevo_core::util::seed::splitmix64`], the
//! single frozen mixer shared with [`SeedStream`]'s trial-level seed fan-out
//! (ADR 0033). The two seed-derivation *schemes* remain independent: they
//! share the mixer, not a derivation contract.
//!
//! [`SeedStream`]: rlevo_core::util::seed::SeedStream
//!
//! # Host-RNG convention
//!
//! All randomness in `rlevo-evolution` **must** go through [`seed_stream`].
//! Do **not** use `B::seed(…) + Tensor::random(…)` for stochastic EA
//! operators. Burn's backend-level RNG is a process-wide mutex; seeding it
//! from inside an operator races with parallel test threads and with other
//! concurrent strategy calls, making results non-reproducible and causing
//! intermittent test failures. [`seed_stream`] returns a fully-isolated
//! [`rand::rngs::StdRng`] whose state is private to the call site.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rlevo_core::util::seed::splitmix64;

/// Tag identifying which evolutionary operation a sub-stream is for.
///
/// Mixing the purpose into the seed means that, within a single
/// generation, selection and mutation draw from non-overlapping PRNG
/// streams — critical for reproducing trial-level determinism across
/// refactors that reorder operator calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SeedPurpose {
    /// Initial population sampling.
    Init = 0,
    /// Parent selection.
    Selection = 1,
    /// Recombination / crossover.
    Crossover = 2,
    /// Mutation.
    Mutation = 3,
    /// Replacement / survivor selection.
    Replacement = 4,
    /// Differential-evolution trial-vector construction.
    Trial = 5,
    /// Catch-all for strategy-specific stochastic steps.
    ///
    /// # Isolation caveat
    ///
    /// Unlike the named purposes (each owning a distinct operator role),
    /// `Other` is a **shared bucket** used by many unrelated strategies
    /// (evolutionary programming, ES, NEAT, NAS, and the swarm family:
    /// SALP/WOA/GWO/ABC/PSO/BAT). Two different strategies both passing
    /// `Other` get the **same** domain constant, so their cross-strategy
    /// isolation relies *entirely* on each call site passing a distinct
    /// `base` (and/or `generation`) — typically a fresh `rng.next_u64()`.
    /// If two `Other` call sites ever share the same `(base, generation)`,
    /// their streams alias. Prefer a dedicated named variant for any operator
    /// that needs guaranteed isolation within a fixed `(base, generation)`.
    ///
    /// Note: this variant's constant `0x9E37_79B9_7F4A_7C15` coincides with the
    /// φ64 golden-ratio multiplier applied to `generation` in [`seed_stream`].
    /// No concrete collision exists today (no purpose uses constant `0`, and
    /// the generation term is multiplied), but it is a latent footgun — do not
    /// assume the `Other` domain is independent of the generation axis.
    Other = 6,
    /// Local-search refinement (memetic algorithms).
    ///
    /// Used by [`crate::local_search`] searchers so a memetic wrapper's
    /// per-individual refinement draws from a stream independent of the
    /// inner strategy's selection/mutation/crossover/replacement streams.
    LocalSearch = 7,
    /// Model sampling in estimation-of-distribution (EDA) strategies.
    ///
    /// Used by [`crate::algorithms::eda`] so each generation draws its new
    /// population from an independent per-generation stream, isolated from
    /// every other operator purpose.
    EdaSampling = 8,
    /// Representative selection in cooperative co-evolution (CCGA).
    ///
    /// Used by [`crate::coevolution`] so the `Random` and `Archive`
    /// representative-selection policies draw their opposing-population
    /// representatives from a stream independent of either sub-strategy's
    /// selection/mutation/crossover/replacement streams.
    Representative = 9,
    /// Transposition operators (gene expression programming).
    ///
    /// Used by [`crate::algorithms::gep`] so IS/RIS transposition draws from a
    /// stream independent of the point-mutation ([`Mutation`](Self::Mutation))
    /// and crossover ([`Crossover`](Self::Crossover)) streams within the same
    /// generation.
    Transposition = 10,
    /// Multivariate-Gaussian sampling in covariance-matrix strategies.
    ///
    /// Used by [`crate::algorithms::cma_es`] and
    /// [`crate::algorithms::cmsa_es`] so each generation draws its `N(m, σ²C)`
    /// offspring (and, for CMSA-ES, the per-individual log-normal σ mutations)
    /// from a stream independent of every other operator purpose.
    CmaSampling = 11,
}

impl SeedPurpose {
    const fn constant(self) -> u64 {
        match self {
            SeedPurpose::Init => 0xA5A5_A5A5_A5A5_A5A5,
            SeedPurpose::Selection => 0x1234_5678_9ABC_DEF0,
            SeedPurpose::Crossover => 0xDEAD_BEEF_CAFE_F00D,
            SeedPurpose::Mutation => 0xFEED_FACE_0BAD_F00D,
            SeedPurpose::Replacement => 0x0123_4567_89AB_CDEF,
            SeedPurpose::Trial => 0xBAAD_F00D_DEAD_C0DE,
            SeedPurpose::Other => 0x9E37_79B9_7F4A_7C15,
            SeedPurpose::LocalSearch => 0xC0FF_EE15_600D_F00D,
            SeedPurpose::EdaSampling => 0xEDA0_5EED_BEEF_CAFE,
            SeedPurpose::Representative => 0xC0EA_5E1E_C7ED_0009,
            SeedPurpose::Transposition => 0x7A05_9051_70F0_000A,
            SeedPurpose::CmaSampling => 0xC3A0_5EED_1AC0_B11B,
        }
    }
}

/// Derives a seeded PRNG from a base seed, generation counter, and purpose.
///
/// Each combination of `(base, generation, purpose)` produces an
/// independent [`rand::rngs::StdRng`]; repeated calls with the same
/// arguments return bit-identical sequences.
///
/// The mixer is two rounds of splitmix64 applied to
/// `base + generation * φ64 + purpose_constant`, where φ64 is the
/// 64-bit golden-ratio constant `0x9E3779B97F4A7C15`. This produces
/// well-distributed seeds even for small or sequential inputs.
///
/// # Examples
///
/// ```
/// use rand::Rng;
/// use rlevo_evolution::rng::{seed_stream, SeedPurpose};
///
/// // Same arguments → identical stream.
/// let mut a = seed_stream(42, 0, SeedPurpose::Mutation);
/// let mut b = seed_stream(42, 0, SeedPurpose::Mutation);
/// assert_eq!(a.next_u64(), b.next_u64());
///
/// // Different purposes → independent streams from the same base/generation.
/// let first_mutation  = seed_stream(42, 0, SeedPurpose::Mutation).next_u64();
/// let first_selection = seed_stream(42, 0, SeedPurpose::Selection).next_u64();
/// assert_ne!(first_mutation, first_selection);
/// ```
#[must_use]
pub fn seed_stream(base: u64, generation: u64, purpose: SeedPurpose) -> StdRng {
    let mut x = base
        .wrapping_add(generation.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(purpose.constant());
    x = splitmix64(x);
    x = splitmix64(x);
    StdRng::seed_from_u64(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, RngExt};

    #[test]
    fn seed_stream_is_deterministic() {
        let mut a = seed_stream(42, 0, SeedPurpose::Init);
        let mut b = seed_stream(42, 0, SeedPurpose::Init);
        for _ in 0..8 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    // All `SeedPurpose` variants, kept exhaustive by `_exhaustiveness_guard`.
    const ALL_PURPOSES: [SeedPurpose; 12] = [
        SeedPurpose::Init,
        SeedPurpose::Selection,
        SeedPurpose::Crossover,
        SeedPurpose::Mutation,
        SeedPurpose::Replacement,
        SeedPurpose::Trial,
        SeedPurpose::Other,
        SeedPurpose::LocalSearch,
        SeedPurpose::EdaSampling,
        SeedPurpose::Representative,
        SeedPurpose::Transposition,
        SeedPurpose::CmaSampling,
    ];

    // Compile-time guard: adding a variant makes this match non-exhaustive,
    // forcing `ALL_PURPOSES` (and its length) to be updated in lock-step.
    #[allow(dead_code)]
    fn _exhaustiveness_guard(p: SeedPurpose) {
        match p {
            SeedPurpose::Init
            | SeedPurpose::Selection
            | SeedPurpose::Crossover
            | SeedPurpose::Mutation
            | SeedPurpose::Replacement
            | SeedPurpose::Trial
            | SeedPurpose::Other
            | SeedPurpose::LocalSearch
            | SeedPurpose::EdaSampling
            | SeedPurpose::Representative
            | SeedPurpose::Transposition
            | SeedPurpose::CmaSampling => {}
        }
    }

    #[test]
    fn all_purpose_domain_constants_are_pairwise_distinct() {
        // Stronger than stream distinctness: catches the root cause (a
        // duplicated or zero domain constant) directly, independent of the
        // mixer.
        for (i, &p) in ALL_PURPOSES.iter().enumerate() {
            for &q in &ALL_PURPOSES[i + 1..] {
                assert_ne!(
                    p.constant(),
                    q.constant(),
                    "domain constants collide for {p:?} and {q:?}"
                );
            }
        }
    }

    #[test]
    fn all_purposes_produce_distinct_first_draws() {
        // Exhaustive all-pairs over every `SeedPurpose` variant at a fixed
        // base/generation.
        for (i, &p) in ALL_PURPOSES.iter().enumerate() {
            let first_p = seed_stream(42, 0, p).next_u64();
            for &q in &ALL_PURPOSES[i + 1..] {
                let first_q = seed_stream(42, 0, q).next_u64();
                assert_ne!(first_p, first_q, "first draws collide for {p:?} and {q:?}");
            }
        }
    }

    #[test]
    fn different_generations_produce_different_streams() {
        let a = seed_stream(42, 0, SeedPurpose::Mutation).next_u64();
        let b = seed_stream(42, 1, SeedPurpose::Mutation).next_u64();
        assert_ne!(a, b);
    }

    #[test]
    fn different_bases_produce_different_streams() {
        let a = seed_stream(1, 0, SeedPurpose::Init).next_u64();
        let b = seed_stream(2, 0, SeedPurpose::Init).next_u64();
        assert_ne!(a, b);
    }

    #[test]
    fn rng_generates_bounded_values() {
        let mut rng = seed_stream(7, 0, SeedPurpose::Init);
        let x: u32 = rng.random_range(0..100);
        assert!(x < 100);
    }
}
