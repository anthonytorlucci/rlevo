//! Deterministic seed derivation for benchmark trials.
//!
//! A single `base_seed` fans out to per-trial, per-env, and per-agent seeds
//! via splitmix64 mixing. Outputs are independent of thread scheduling,
//! ensuring that two runs with the same base seed produce identical seed
//! sequences regardless of how work is distributed across threads.
//!
//! # Reproducibility contract
//!
//! All `rlevo` algorithms must draw randomness by calling into a
//! [`SeedStream`] rather than reaching for the backend's process-wide RNG
//! (e.g. `B::seed` + `Tensor::random`). The process-wide RNG is a global
//! mutex; parallel test workers racing on it produce non-deterministic
//! initialisation order even when the top-level seed is fixed.
//!
//! The recommended pattern:
//!
//! ```rust
//! use rlevo_core::util::seed::SeedStream;
//!
//! let stream = SeedStream::new(42);
//! let t = stream.trial_seed(/*env_idx=*/0, /*trial_idx=*/0);
//! let env_rng_seed = stream.env_seed(t);
//! let agent_rng_seed = stream.agent_seed(t);
//! // Pass env_rng_seed / agent_rng_seed to the respective constructors.
//! ```

/// A deterministic, fan-out seed generator for benchmark and training trials.
///
/// `SeedStream` derives independent 64-bit seeds for each (environment index,
/// trial index) pair, and further splits each trial seed into separate env and
/// agent seeds. Every derivation is a pure function of `base` and the index
/// arguments, so the same `SeedStream` always produces the same sequence.
///
/// # Design
///
/// Derivation uses splitmix64 — a bijective mixing function — XOR'd with
/// index-dependent multipliers. The env and agent branches are separated by
/// distinct domain constants so `env_seed(t) != agent_seed(t)` for all `t`.
///
/// # Thread safety
///
/// `SeedStream` holds no mutable state and is `Copy`. All methods are
/// `const fn`. It is safe to share across threads without synchronisation.
#[derive(Debug, Clone, Copy)]
pub struct SeedStream {
    base: u64,
}

impl SeedStream {
    /// Creates a new `SeedStream` from the given base seed.
    ///
    /// All seeds derived from this stream are fully determined by `base`.
    /// Two streams constructed with the same `base` produce identical output.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_core::util::seed::SeedStream;
    ///
    /// let stream = SeedStream::new(0xDEAD_BEEF);
    /// assert_eq!(stream.base(), 0xDEAD_BEEF);
    /// ```
    #[must_use]
    pub const fn new(base: u64) -> Self {
        Self { base }
    }

    /// Returns the base seed this stream was constructed with.
    ///
    /// Useful for serialising or logging the root of a reproducible run.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_core::util::seed::SeedStream;
    ///
    /// let stream = SeedStream::new(42);
    /// assert_eq!(stream.base(), 42);
    /// ```
    #[must_use]
    pub const fn base(&self) -> u64 {
        self.base
    }

    /// Derives a deterministic seed for a specific (env index, trial index) pair.
    ///
    /// The returned value is the root from which [`env_seed`](Self::env_seed)
    /// and [`agent_seed`](Self::agent_seed) are derived for that trial.
    /// Different `(env_idx, trial_idx)` pairs always produce different seeds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_core::util::seed::SeedStream;
    ///
    /// let stream = SeedStream::new(42);
    ///
    /// // Repeated calls with identical arguments are stable.
    /// assert_eq!(stream.trial_seed(0, 0), stream.trial_seed(0, 0));
    ///
    /// // Different index pairs yield different seeds.
    /// assert_ne!(stream.trial_seed(0, 0), stream.trial_seed(0, 1));
    /// assert_ne!(stream.trial_seed(0, 0), stream.trial_seed(1, 0));
    /// ```
    #[must_use]
    pub const fn trial_seed(&self, env_idx: usize, trial_idx: usize) -> u64 {
        let mixed = splitmix64(self.base ^ (env_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        splitmix64(mixed ^ (trial_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
    }

    /// Derives the environment seed for a given trial seed.
    ///
    /// Pass the result of [`trial_seed`](Self::trial_seed) as `trial_seed`.
    /// The returned value is guaranteed to differ from [`agent_seed`](Self::agent_seed)
    /// for the same input, preventing env and agent RNGs from correlating.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_core::util::seed::SeedStream;
    ///
    /// let stream = SeedStream::new(42);
    /// let t = stream.trial_seed(0, 0);
    /// let env_seed = stream.env_seed(t);
    /// let agent_seed = stream.agent_seed(t);
    /// assert_ne!(env_seed, agent_seed);
    /// ```
    #[must_use]
    pub const fn env_seed(&self, trial_seed: u64) -> u64 {
        splitmix64(trial_seed ^ 0xD1B5_4A32_D192_ED03)
    }

    /// Derives the agent seed for a given trial seed.
    ///
    /// Pass the result of [`trial_seed`](Self::trial_seed) as `trial_seed`.
    /// The returned value is guaranteed to differ from [`env_seed`](Self::env_seed)
    /// for the same input, preventing agent and env RNGs from correlating.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_core::util::seed::SeedStream;
    ///
    /// let stream = SeedStream::new(42);
    /// let t = stream.trial_seed(0, 0);
    ///
    /// // agent_seed and env_seed are derived with different domain constants,
    /// // so they are always distinct for the same trial seed.
    /// assert_ne!(stream.agent_seed(t), stream.env_seed(t));
    ///
    /// // Repeated calls are stable.
    /// assert_eq!(stream.agent_seed(t), stream.agent_seed(t));
    /// ```
    #[must_use]
    pub const fn agent_seed(&self, trial_seed: u64) -> u64 {
        splitmix64(trial_seed ^ 0x94D0_49BB_1331_11EB)
    }
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

    #[test]
    fn trial_seeds_are_deterministic() {
        let s = SeedStream::new(42);
        assert_eq!(s.trial_seed(0, 0), s.trial_seed(0, 0));
        assert_eq!(s.trial_seed(3, 7), s.trial_seed(3, 7));
    }

    #[test]
    fn trial_seeds_are_distinct() {
        let s = SeedStream::new(42);
        let a = s.trial_seed(0, 0);
        let b = s.trial_seed(0, 1);
        let c = s.trial_seed(1, 0);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn env_and_agent_seeds_differ() {
        let s = SeedStream::new(42);
        let t = s.trial_seed(0, 0);
        assert_ne!(s.env_seed(t), s.agent_seed(t));
    }

    #[test]
    fn base_seed_changes_output() {
        let a = SeedStream::new(1).trial_seed(0, 0);
        let b = SeedStream::new(2).trial_seed(0, 0);
        assert_ne!(a, b);
    }
}
