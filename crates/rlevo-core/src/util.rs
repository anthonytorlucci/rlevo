//! Shared math helpers used across `rlevo-core` consumers.
//!
//! Lifted from the (now-removed) `rlevo-utils` crate per ADR 0003 — the
//! crate held a single function with zero workspace consumers and was
//! folded here to remove an empty workspace member.

/// Returns the binomial coefficient `n choose k` (number of combinations of
/// `n` items taken `k` at a time).
///
/// Returns `0` when `k > n`.
///
/// # Examples
///
/// ```
/// use rlevo_core::util::combinations;
/// assert_eq!(combinations(54, 6), 25_827_165);
/// assert_eq!(combinations(5, 0), 1);
/// assert_eq!(combinations(3, 5), 0);
/// ```
pub fn combinations(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let mut result = 1u64;
    for i in 1..=k {
        result = result * (n - i + 1) / i;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combinations_known_values() {
        assert_eq!(combinations(0, 0), 1);
        assert_eq!(combinations(5, 0), 1);
        assert_eq!(combinations(5, 5), 1);
        assert_eq!(combinations(5, 2), 10);
        assert_eq!(combinations(54, 6), 25_827_165);
    }

    #[test]
    fn test_combinations_k_greater_than_n_returns_zero() {
        assert_eq!(combinations(3, 5), 0);
    }
}
