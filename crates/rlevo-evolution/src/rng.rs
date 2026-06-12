//! Deterministic seed derivation for strategies.
//!
//! Callers derive sub-seeds by mixing a `base`, a `generation` index, and
//! a [`SeedPurpose`] so parallel streams (selection, mutation, crossover)
//! do not alias. The mixer is splitmix64, matching the algorithm used by
//! [`rlevo_core::util::seed::SeedStream`] for trial-level seed fan-out.
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

const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
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

    // One single-letter binding per purpose under test; the names index the
    // purpose list rather than carrying independent meaning.
    #[allow(clippy::many_single_char_names)]
    #[test]
    fn different_purposes_produce_different_streams() {
        let a = seed_stream(42, 0, SeedPurpose::Init).next_u64();
        let b = seed_stream(42, 0, SeedPurpose::Selection).next_u64();
        let c = seed_stream(42, 0, SeedPurpose::Mutation).next_u64();
        let d = seed_stream(42, 0, SeedPurpose::LocalSearch).next_u64();
        let e = seed_stream(42, 0, SeedPurpose::EdaSampling).next_u64();
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
        assert_ne!(a, d);
        assert_ne!(b, d);
        assert_ne!(c, d);
        assert_ne!(a, e);
        assert_ne!(b, e);
        assert_ne!(c, e);
        assert_ne!(d, e);
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
