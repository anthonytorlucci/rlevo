//! Deterministic seed derivation for benchmark trials.
//!
//! A single `base_seed` fans out to per-trial, per-env, and per-agent seeds
//! via splitmix64 mixing. Outputs are independent of thread scheduling.

#[derive(Debug, Clone, Copy)]
pub struct SeedStream {
    base: u64,
}

impl SeedStream {
    #[must_use]
    pub const fn new(base: u64) -> Self {
        Self { base }
    }

    #[must_use]
    pub const fn base(&self) -> u64 {
        self.base
    }

    #[must_use]
    pub const fn trial_seed(&self, env_idx: usize, trial_idx: usize) -> u64 {
        let mixed = splitmix64(self.base ^ (env_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        splitmix64(mixed ^ (trial_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
    }

    #[must_use]
    pub const fn env_seed(&self, trial_seed: u64) -> u64 {
        splitmix64(trial_seed ^ 0xD1B5_4A32_D192_ED03)
    }

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
