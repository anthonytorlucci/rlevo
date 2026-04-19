//! Deterministic seed derivation for strategies.
//!
//! The workspace's canonical `SeedStream` lives in `evorl-benchmarks` and
//! fans a base seed out to per-trial sub-seeds via splitmix64. This module
//! re-implements the same splitmix64 mixer locally so `evorl-evolution`
//! does not depend on `evorl-benchmarks` transitively — keeping the
//! dependency graph a strict DAG.
//!
//! The two implementations must stay in lock-step: a base seed fed to this
//! module's [`seed_stream`] must produce the same bytes as the benchmark
//! harness's [`SeedStream`](evorl_benchmarks::seed::SeedStream).
//!
//! Callers derive sub-seeds by mixing a `base`, a `generation` index, and
//! a [`SeedPurpose`] so parallel streams (selection, mutation, crossover)
//! do not alias.

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
        }
    }
}

/// Derives a seeded PRNG from a base seed, generation counter, and purpose.
///
/// Each combination produces an independent stream; repeated calls with
/// the same arguments return bit-identical sequences.
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

    #[test]
    fn different_purposes_produce_different_streams() {
        let a = seed_stream(42, 0, SeedPurpose::Init).next_u64();
        let b = seed_stream(42, 0, SeedPurpose::Selection).next_u64();
        let c = seed_stream(42, 0, SeedPurpose::Mutation).next_u64();
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
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
