//! Shared utilities used across `rlevo-core` consumers.
//!
//! Currently houses small helpers ([`combinations`] and its fallible
//! counterpart [`checked_combinations`]) and the
//! [`seed`] module's deterministic seed-derivation primitives
//! (see [`seed::SeedStream`]).
//!
//! [`combinations`] and [`checked_combinations`] have **zero consumers** in the
//! workspace, as expected: they are public API for downstream users, retained
//! when the `rlevo-utils` crate was folded in (ADR 0003) — not dead code
//! awaiting a caller, nor anticipatory API for a planned internal feature.
//! Removing them would break `rlevo-core`'s published surface with no
//! offsetting benefit at alpha.
//!
//! [`combinations`]: crate::util::combinations
//! [`checked_combinations`]: crate::util::checked_combinations
//! [`seed`]: crate::util::seed
//! [`seed::SeedStream`]: crate::util::seed::SeedStream

pub mod seed;

/// Returns the binomial coefficient `n choose k` (number of combinations of
/// `n` items taken `k` at a time).
///
/// Returns `0` when `k > n`.
///
/// # Panics
///
/// Panics when the **exact** value of `C(n, k)` exceeds [`u64::MAX`] — for
/// example `C(68, 34)`. The condition is the exact result, *not* an
/// intermediate product: every `C(n, k)` representable in a `u64` is computed
/// and returned. Use [`checked_combinations`] to cover the full domain without
/// panicking.
///
/// # Examples
///
/// ```
/// use rlevo_core::util::combinations;
/// assert_eq!(combinations(54, 6), 25_827_165);
/// assert_eq!(combinations(5, 0), 1);
/// assert_eq!(combinations(3, 5), 0);
/// ```
#[must_use]
pub fn combinations(n: u64, k: u64) -> u64 {
    checked_combinations(n, k).expect("combinations: C(n, k) exceeds u64::MAX")
}

/// Returns the binomial coefficient `n choose k`, or [`None`] when the exact
/// value does not fit in a `u64`.
///
/// Returns `Some(0)` when `k > n`, matching [`combinations`].
///
/// # Examples
///
/// ```
/// use rlevo_core::util::checked_combinations;
/// assert_eq!(checked_combinations(66, 33), Some(7_219_428_434_016_265_740));
/// assert_eq!(checked_combinations(3, 5), Some(0));
/// // C(68, 34) ≈ 2.85e19 exceeds u64::MAX ≈ 1.84e19.
/// assert_eq!(checked_combinations(68, 34), None);
/// ```
#[must_use]
pub fn checked_combinations(n: u64, k: u64) -> Option<u64> {
    if k > n {
        return Some(0);
    }
    // Symmetry reduction: C(n, k) == C(n, n - k), so iterate the smaller side.
    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 1..=u128::from(k) {
        // Partial results are exactly C(n - k + i, i), so the division is
        // always exact and never truncates. Those partials increase
        // monotonically in `i`, so bailing out the moment one exceeds
        // `u64::MAX` cannot discard a final value that would have fit.
        result = result * (u128::from(n - k) + i) / i;
        if result > u128::from(u64::MAX) {
            return None;
        }
    }
    // `result` is bounded by `u64::MAX` on entry to each step and the
    // multiplier by `n`, so the `u128` accumulator cannot itself overflow, and
    // the loop guard leaves this conversion infallible.
    u64::try_from(result).ok()
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

    /// Values that fit in a `u64` but which the former naive loop corrupted by
    /// wrapping its intermediate product (silently, in release).
    #[test]
    fn test_combinations_large_values_are_exact() {
        assert_eq!(combinations(66, 33), 7_219_428_434_016_265_740);
        assert_eq!(combinations(67, 33), 14_226_520_737_620_288_370);
    }

    /// The representability frontier: `C(67, 33)` fits, `C(68, 34)` (≈2.85e19)
    /// does not.
    #[test]
    fn test_checked_combinations_frontier() {
        assert_eq!(
            checked_combinations(67, 33),
            Some(14_226_520_737_620_288_370)
        );
        assert_eq!(checked_combinations(68, 34), None);
    }

    #[test]
    #[should_panic(expected = "exceeds u64::MAX")]
    fn test_combinations_panics_when_result_exceeds_u64() {
        let _ = combinations(68, 34);
    }

    /// Checks the `C(n, k) == C(n, n - k)` symmetry invariant only — it is
    /// *not* a regression test for the u64-overflow fix below, since wrapping
    /// corrupts both sides identically and this test passes against the
    /// pre-fix implementation just as well as the fixed one.
    #[test]
    fn test_combinations_is_symmetric() {
        for n in 0..40u64 {
            for k in 0..=n {
                assert_eq!(
                    combinations(n, k),
                    combinations(n, n - k),
                    "asymmetry at C({n}, {k})"
                );
            }
        }
    }

    /// Independent oracle: Pascal's triangle built by addition shares no code
    /// with the multiplicative implementation, so it catches arithmetic errors
    /// that comparing `combinations` to `checked_combinations` cannot — the
    /// former delegates to the latter, making that comparison self-referential.
    ///
    /// `n <= 60` keeps every entry representable (`C(60, 30)` ≈ 1.18e17), so
    /// the oracle itself never overflows and every cell is a live assertion.
    #[test]
    fn test_checked_combinations_matches_pascals_triangle() {
        const N: usize = 60;
        let mut row: Vec<u64> = vec![1];
        for n in 0..=N {
            for (k, &expected) in row.iter().enumerate() {
                assert_eq!(
                    checked_combinations(n as u64, k as u64),
                    Some(expected),
                    "mismatch at C({n}, {k})"
                );
            }
            // Next row by addition: c[n+1][k] = c[n][k - 1] + c[n][k].
            let mut next = Vec::with_capacity(row.len() + 1);
            next.push(1);
            for w in row.windows(2) {
                next.push(w[0] + w[1]);
            }
            next.push(1);
            row = next;
        }
    }

    /// Boundary cases that pin the *placement* and *strictness* of the
    /// `result > u64::MAX` guard, which no other test constrains.
    #[test]
    fn test_checked_combinations_guard_boundaries() {
        // Guards the guard being INSIDE the loop. `C(9975, 4987)` is ~1e3001;
        // partial results cross `u64::MAX` at i = 6, so the in-loop check bails
        // immediately. "Simplify" the fix by moving that check after the loop
        // and the `u128` accumulator overflows on its own — the same silent-
        // wrap-in-release, panic-in-debug failure mode `combinations` itself
        // had, one layer up. No other test uses an `n` large enough
        // to reach that. (Verified: the after-the-loop variant panics here in a
        // debug build. In release its wrapped value still lands above
        // `u64::MAX`, so this assertion alone does not catch it there — the
        // debug run is what kills that mutant.)
        assert_eq!(checked_combinations(9975, 4987), None);

        // Exact fit with zero headroom: `C(u64::MAX, 1) == u64::MAX`. A `>=`
        // guard would reject this; the correct `>` guard accepts it.
        assert_eq!(checked_combinations(u64::MAX, 1), Some(u64::MAX));
        assert_eq!(checked_combinations(u64::MAX, 2), None);

        // `k > n` returns before `n - k` is evaluated, so no underflow.
        assert_eq!(checked_combinations(1, u64::MAX), Some(0));

        // Degenerate rows, mirroring the `combinations` known-value test.
        assert_eq!(checked_combinations(0, 0), Some(1));
        assert_eq!(checked_combinations(5, 5), Some(1));
    }

    #[test]
    fn test_checked_combinations_k_greater_than_n_returns_zero() {
        assert_eq!(checked_combinations(3, 5), Some(0));
        assert_eq!(checked_combinations(0, 1), Some(0));
    }
}
