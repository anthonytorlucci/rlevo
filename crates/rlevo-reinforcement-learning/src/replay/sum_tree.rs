//! The priority index behind [`PrioritizedReplay`](super::PrioritizedReplay):
//! a [`SumTree`], and the [`PriorityIndex`] contract it shares with an O(n)
//! reference implementation.
//!
//! # Why there are two implementations
//!
//! ADR 0050 names the hazard explicitly: "a sum-tree is ~150 lines of new index
//! arithmetic on a path where an off-by-one **silently biases sampling rather
//! than crashing**". A biased sampler still returns plausible-looking indices,
//! still trains, and still converges — just to the wrong thing, with no failing
//! assertion anywhere.
//!
//! The mitigation is structural rather than inspectional. Both the tree and a
//! trivially-auditable O(n) prefix scan implement [`PriorityIndex`]; the entire
//! draw is expressed once, in [`stratified_draw`], against that trait; and the
//! test suite runs the *same* seeded draw through both and asserts identical
//! slot sequences. An off-by-one in the tree's descent cannot survive that
//! comparison, because the reference has no descent to be off by one in.
//!
//! # The scan is the specification, the tree is the optimization
//!
//! `PrefixScanIndex` is written to be read: `find(v)` returns the first slot
//! whose running total exceeds `v`, which is the inverse CDF exactly as it
//! appears in the definition of `P(i)`. [`SumTree`] computes the same function
//! in O(log N). The one place they could disagree is floating-point
//! associativity — the tree sums pairwise up the levels, the scan sums left to
//! right — so both accumulate in `f64` over `f32`-precision leaves, which makes
//! the two orders agree exactly for every buffer the tests and the shipped
//! configs construct. The cross-implementation equivalence tests are the
//! standing guard on that.
//!
//! `PrefixScanIndex` is `#[cfg(test)]`: it is the oracle, not a shipped code
//! path, and `rules.md` §9 does not permit an `#[allow(dead_code)]` for it.

use rand::{Rng, RngExt};

use crate::MAX_BUFFER_CAPACITY;

/// A mapping from buffer slot to non-negative sampling mass, supporting the two
/// operations Schaul's proportional variant needs: point update, and inverse-CDF
/// lookup.
///
/// Implemented by [`SumTree`] (shipped, O(log N)) and `PrefixScanIndex`
/// (test-only, O(n)). The trait exists so [`stratified_draw`] is written once
/// and both implementations are exercised by literally the same draw code.
///
/// # Invariants
///
/// - Every value passed to [`set`](Self::set) is finite and `>= 0.0`.
/// - [`total`](Self::total) equals the sum of all slot masses.
/// - For `0.0 <= v < total()`, [`find`](Self::find) returns a slot whose mass
///   is strictly positive.
pub(crate) trait PriorityIndex {
    /// Sets slot `slot`'s mass to `value`.
    fn set(&mut self, slot: usize, value: f64);

    /// The mass currently stored at `slot`.
    fn get(&self, slot: usize) -> f64;

    /// The total mass across all slots — Schaul's `p_total`.
    fn total(&self) -> f64;

    /// The inverse CDF: the slot whose half-open mass interval contains `v`.
    ///
    /// Formally, the smallest `i` such that `sum(0..=i) > v`.
    fn find(&self, v: f64) -> usize;
}

/// A fixed-capacity binary sum tree over slot priorities (Schaul et al. 2016,
/// Appendix B.2.1: "parent = sum of children, root = `p_total`").
///
/// Backed by a single flat `Vec<f64>` of length `2 * n`, where `n` is `capacity`
/// rounded up to a power of two. Node `1` is the root, node `k`'s children are
/// `2k` and `2k + 1`, and slot `s` lives at leaf `n + s`. Slots in
/// `capacity..n` are padding, permanently `0.0`, and therefore unreachable from
/// [`find`](PriorityIndex::find) — see that method for the argument.
///
/// Both update and lookup are O(log n).
#[derive(Debug, Clone)]
pub(crate) struct SumTree {
    /// `nodes[1]` is the root; leaves occupy `nodes[leaf_base..leaf_base * 2]`.
    /// `nodes[0]` is unused, which is what keeps the child index arithmetic
    /// free of `+1`/`-1` corrections.
    nodes: Vec<f64>,
    /// `capacity` rounded up to a power of two; also the index of leaf `0`.
    leaf_base: usize,
    /// The number of *addressable* slots, which may be less than `leaf_base`.
    capacity: usize,
}

impl SumTree {
    /// Builds an all-zero tree addressing `capacity` slots.
    ///
    /// # Panics
    ///
    /// Panics when `capacity == 0`, and when `capacity` exceeds
    /// [`MAX_BUFFER_CAPACITY`]. Capacity reaches here only from a
    /// [`PrioritizedReplayConfig`](super::PrioritizedReplayConfig) that has
    /// already passed `validate()`, which rejects both, so either is a
    /// programming error in the buffer rather than user-supplied data.
    ///
    /// The upper bound is nonetheless asserted here rather than left to the
    /// caller, because the failure it prevents is invisible. `leaf_base * 2`
    /// and `capacity.next_power_of_two()` both **wrap** in release builds — the
    /// workspace root `Cargo.toml` has no `[profile]` section, so
    /// `overflow-checks = false` is inherited. Measured on `aarch64` macOS with
    /// `rustc -O`:
    ///
    /// | `capacity` | `leaf_base` | `nodes.len()` |
    /// |---|---|---|
    /// | `2^62 + 1` | `2^63` | `0` |
    /// | `2^63` | `2^63` | `0` |
    /// | `2^63 + 1` | `0` | `0` |
    /// | `usize::MAX` | `0` | `0` |
    ///
    /// Without the assert, debug panics on the overflow while release *returns
    /// a corrupt tree*: an empty `nodes` array whose `total()` indexes out of
    /// bounds on the very first read. A constructor whose contract differs
    /// between profiles is worse than one that always panics, so the bound is
    /// checked explicitly and identically in both.
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SumTree::new: capacity must be non-zero");
        assert!(
            capacity <= MAX_BUFFER_CAPACITY,
            "SumTree::new: capacity {capacity} exceeds MAX_BUFFER_CAPACITY \
             {MAX_BUFFER_CAPACITY}"
        );
        let leaf_base = capacity.next_power_of_two();
        Self {
            nodes: vec![0.0; leaf_base * 2],
            leaf_base,
            capacity,
        }
    }
}

impl PriorityIndex for SumTree {
    fn set(&mut self, slot: usize, value: f64) {
        assert!(
            slot < self.capacity,
            "SumTree::set: slot {slot} out of range for capacity {}",
            self.capacity
        );
        debug_assert!(
            value.is_finite() && value >= 0.0,
            "SumTree::set: mass must be finite and non-negative, got {value}"
        );
        let mut node = self.leaf_base + slot;
        self.nodes[node] = value;
        // Recompute each ancestor from its two children rather than applying a
        // delta. A delta accumulates rounding error over a training run of
        // millions of updates; recomputation keeps every parent exactly the sum
        // of its current children, which is the invariant `find` relies on.
        while node > 1 {
            node /= 2;
            self.nodes[node] = self.nodes[node * 2] + self.nodes[node * 2 + 1];
        }
    }

    fn get(&self, slot: usize) -> f64 {
        self.nodes[self.leaf_base + slot]
    }

    fn total(&self) -> f64 {
        self.nodes[1]
    }

    fn find(&self, v: f64) -> usize {
        let mut remaining = v;
        let mut node = 1;
        // Descend maintaining the invariant `remaining < subtree_sum(node)`,
        // which holds on entry whenever `v < total()`. Going left preserves it
        // directly; going right subtracts the left mass, and
        // `remaining - left < node_sum - left == right_sum`. At the leaf the
        // invariant reads `remaining < leaf_mass`, so the leaf mass is strictly
        // positive — which is precisely why a zero-mass padding slot, or an
        // unoccupied slot, can never be returned.
        while node < self.leaf_base {
            let left = node * 2;
            let left_mass = self.nodes[left];
            if remaining < left_mass {
                node = left;
            } else {
                remaining -= left_mass;
                node = left + 1;
            }
        }
        let slot = node - self.leaf_base;
        // The invariant above is exact in real arithmetic, and `set` keeps every
        // parent equal to the sum of its children, so the only way to land in
        // the padding is a caller passing `v >= total()`. Clamping into the
        // addressable range keeps that a bounded wrong answer rather than an
        // out-of-bounds index; `stratified_draw` never produces such a `v`.
        slot.min(self.capacity - 1)
    }
}

/// Draws `batch_size` slots by Schaul et al. (2016) Appendix B.2.1's
/// **stratified** scheme, verbatim: "To sample a minibatch of size k, the range
/// `[0, p_total]` is divided equally into k ranges. Next, a value is uniformly
/// sampled from each range."
///
/// §3.3 states the purpose: "sample exactly one transition from each segment –
/// this is a form of stratified sampling that has the added advantage of
/// **balancing out the minibatch**."
///
/// This is **not** `batch_size` i.i.d. draws from the categorical distribution
/// `P(i)`. That is a different algorithm: same marginal, different joint. It
/// admits minibatches drawn entirely from the high-priority tail, which is
/// exactly the variance the stratification exists to remove. The pre-ADR-0050
/// `memory.rs` sampled i.i.d.; this does not.
///
/// Exactly one `rng` draw is issued per segment, in segment order, and the
/// returned slots are in that same order — so the draw sequence is a pinnable
/// contract in the same sense as `UniformReplay`'s (ADR 0050 §5).
///
/// # Panics
///
/// Panics when `batch_size == 0`, or when `index.total()` is not strictly
/// positive and finite. Both are guaranteed by
/// [`PrioritizedReplay`](super::PrioritizedReplay)'s
/// [`sample`](super::ReplayStrategy::sample)
/// preconditions: it rejects an under-filled buffer before calling, and every
/// stored priority is finite and `> 0` by [`Priority`](super::Priority)'s
/// construction, so a non-empty buffer has a positive total.
pub(crate) fn stratified_draw<I, R>(index: &I, batch_size: usize, rng: &mut R) -> Vec<usize>
where
    I: PriorityIndex + ?Sized,
    R: Rng + ?Sized,
{
    assert!(
        batch_size > 0,
        "stratified_draw: batch_size must be non-zero"
    );
    let total = index.total();
    assert!(
        total.is_finite() && total > 0.0,
        "stratified_draw: p_total must be finite and positive, got {total}"
    );

    // `batch_size` is a minibatch size (<= 2^13 in every shipped config), so the
    // `usize -> f64` conversion is exact; the lint is about the general case.
    #[allow(clippy::cast_precision_loss)]
    let k = batch_size as f64;
    let segment = total / k;
    // The largest `f64` strictly below `total`, used to keep the top of the
    // final stratum inside `find`'s half-open domain.
    let ceiling = total.next_down();

    (0..batch_size)
        .map(|j| {
            #[allow(clippy::cast_precision_loss)] // as above
            let j = j as f64;
            let lo = segment * j;
            let hi = segment * (j + 1.0);
            // `random_range` requires a non-empty range. `hi <= lo` can only
            // arise when `segment` underflows against a denormal `total`, in
            // which case every draw is the same degenerate point and `lo` is
            // the right answer.
            let v = if hi > lo {
                rng.random_range(lo..hi)
            } else {
                lo
            };
            index.find(v.min(ceiling))
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod reference {
    //! The O(n) prefix-scan oracle the [`SumTree`](super::SumTree) is validated
    //! against.

    use super::PriorityIndex;

    /// A flat `Vec<f64>` of slot masses, scanned linearly.
    ///
    /// The inverse CDF is written here in its definitional form — accumulate
    /// left to right, stop at the first running total that exceeds `v` — with
    /// no index arithmetic to get wrong. That is the entire point: it is the
    /// oracle, so it must be obviously correct rather than fast.
    #[derive(Debug, Clone)]
    pub(crate) struct PrefixScanIndex {
        masses: Vec<f64>,
    }

    impl PrefixScanIndex {
        /// Builds an all-zero index addressing `capacity` slots.
        pub(crate) fn new(capacity: usize) -> Self {
            assert!(
                capacity > 0,
                "PrefixScanIndex::new: capacity must be non-zero"
            );
            Self {
                masses: vec![0.0; capacity],
            }
        }
    }

    impl PriorityIndex for PrefixScanIndex {
        fn set(&mut self, slot: usize, value: f64) {
            self.masses[slot] = value;
        }

        fn get(&self, slot: usize) -> f64 {
            self.masses[slot]
        }

        fn total(&self) -> f64 {
            self.masses.iter().sum()
        }

        fn find(&self, v: f64) -> usize {
            let mut acc = 0.0;
            for (slot, &mass) in self.masses.iter().enumerate() {
                acc += mass;
                if v < acc {
                    return slot;
                }
            }
            // Only reachable for `v >= total()`, which `stratified_draw` never
            // produces. Mirror `SumTree::find`'s clamp so the two agree even
            // off their shared precondition.
            self.masses.len() - 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reference::PrefixScanIndex;
    use super::{PriorityIndex, SumTree, stratified_draw};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Fills both index implementations with the same masses.
    fn build_pair(masses: &[f64]) -> (SumTree, PrefixScanIndex) {
        let mut tree = SumTree::new(masses.len());
        let mut scan = PrefixScanIndex::new(masses.len());
        for (slot, &m) in masses.iter().enumerate() {
            tree.set(slot, m);
            scan.set(slot, m);
        }
        (tree, scan)
    }

    #[test]
    fn test_sum_tree_total_is_the_sum_of_slots() {
        let (tree, _) = build_pair(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(
            (tree.total() - 15.0).abs() < 1e-12,
            "root must equal p_total, got {}",
            tree.total()
        );
    }

    #[test]
    fn test_sum_tree_total_updates_on_overwrite() {
        let (mut tree, _) = build_pair(&[1.0, 2.0, 3.0]);
        tree.set(1, 10.0);
        assert!(
            (tree.total() - 14.0).abs() < 1e-12,
            "overwriting a slot must re-propagate to the root, got {}",
            tree.total()
        );
        assert!(
            (tree.get(1) - 10.0).abs() < 1e-12,
            "the leaf itself must hold the new mass"
        );
    }

    /// Boundary map for masses `[1, 2, 3, 4]` (total 10): slot 0 owns `[0, 1)`,
    /// slot 1 owns `[1, 3)`, slot 2 owns `[3, 6)`, slot 3 owns `[6, 10)`. Both
    /// implementations must reproduce it exactly — this is the off-by-one
    /// detector.
    #[test]
    fn test_find_boundaries_are_half_open_and_identical() {
        let masses = [1.0, 2.0, 3.0, 4.0];
        let (tree, scan) = build_pair(&masses);
        let cases = [
            (0.0, 0),
            (0.999, 0),
            (1.0, 1), // exactly on a boundary: belongs to the upper slot
            (2.999, 1),
            (3.0, 2),
            (5.999, 2),
            (6.0, 3),
            (9.999, 3),
        ];
        for (v, expected) in cases {
            assert_eq!(tree.find(v), expected, "SumTree::find({v}) boundary");
            assert_eq!(
                scan.find(v),
                expected,
                "PrefixScanIndex::find({v}) boundary"
            );
        }
    }

    /// A zero-mass slot must be unreachable — otherwise an unoccupied slot could
    /// be drawn and dereferenced.
    #[test]
    fn test_find_never_returns_a_zero_mass_slot() {
        let masses = [0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (tree, scan) = build_pair(&masses);
        let total = tree.total();
        for i in 0..1_000 {
            let v = total * f64::from(i) / 1_000.0;
            for (name, slot) in [("tree", tree.find(v)), ("scan", scan.find(v))] {
                assert!(
                    masses[slot] > 0.0,
                    "{name}::find({v}) returned zero-mass slot {slot}"
                );
            }
        }
    }

    /// Padding slots exist whenever capacity is not a power of two. They must
    /// never be returned.
    #[test]
    fn test_sum_tree_non_power_of_two_capacity_excludes_padding() {
        let (tree, _) = build_pair(&[1.0; 5]);
        let total = tree.total();
        for i in 0..500 {
            let v = total * f64::from(i) / 500.0;
            assert!(
                tree.find(v) < 5,
                "find({v}) escaped into the power-of-two padding"
            );
        }
    }

    #[test]
    #[should_panic(expected = "slot 5 out of range")]
    fn test_sum_tree_set_panics_on_out_of_range_slot() {
        let mut tree = SumTree::new(5);
        tree.set(5, 1.0);
    }

    #[test]
    #[should_panic(expected = "capacity must be non-zero")]
    fn test_sum_tree_new_panics_on_zero_capacity() {
        let _ = SumTree::new(0);
    }

    /// Issue #404, the release-mode half. Before the `MAX_BUFFER_CAPACITY`
    /// assert, `SumTree::new` computed `capacity.next_power_of_two()` and then
    /// `leaf_base * 2` with no guard, and the workspace root `Cargo.toml`
    /// declares no `[profile]` section — so release builds inherit
    /// `overflow-checks = false` and **both** operations wrap. Measured on
    /// `aarch64` macOS with `rustc -O`, the four capacities below produced:
    ///
    /// | `capacity` | `leaf_base` | `nodes.len()` |
    /// |---|---|---|
    /// | `2^62 + 1` | `2^63` | `0` — `leaf_base * 2` wrapped |
    /// | `2^63` | `2^63` | `0` — `leaf_base * 2` wrapped |
    /// | `2^63 + 1` | `0` | `0` — `next_power_of_two` wrapped |
    /// | `usize::MAX` | `0` | `0` — `next_power_of_two` wrapped |
    ///
    /// That is: in release the constructor *succeeded*, returning a priority
    /// index with an empty node array; `total()` then reads `nodes[1]` out of
    /// bounds. In debug the same inputs panicked on the overflow. Every one of
    /// the four must now be rejected, identically in both profiles.
    ///
    /// `catch_unwind` rather than `#[should_panic]`: the attribute stops the
    /// test at the first panic, so four `#[should_panic]` cases in a loop would
    /// pin only the first capacity and silently skip the other three — and the
    /// two wrap mechanisms above differ, so each needs its own assertion.
    ///
    /// The **message** is asserted, not merely the fact of a panic, and that is
    /// load-bearing rather than fussy. Against the unguarded constructor this
    /// test is red only in release; in debug the overflow check panics on its
    /// own, so a bare `is_err()` would pass and the test would be pinning the
    /// profile's accident instead of the guard. Requiring the guard's own
    /// message distinguishes "rejected by `MAX_BUFFER_CAPACITY`" from "wrapped,
    /// and happened to be caught by `-C overflow-checks`", and so fails in both
    /// profiles the day the assert is deleted.
    #[test]
    fn test_sum_tree_new_rejects_capacities_that_wrapped_in_release() {
        // Silence the default hook: four deliberate panics would otherwise
        // print four backtraces over a passing test.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcomes: Vec<(usize, Option<String>)> = [
            (1_usize << 62) + 1,
            1_usize << 63,
            (1_usize << 63) + 1,
            usize::MAX,
        ]
        .into_iter()
        .map(|capacity| {
            let message = std::panic::catch_unwind(|| SumTree::new(capacity))
                .err()
                .map(|payload| match payload.downcast::<String>() {
                    Ok(s) => *s,
                    Err(other) => match other.downcast::<&'static str>() {
                        Ok(s) => (*s).to_owned(),
                        Err(_) => "<non-string panic payload>".to_owned(),
                    },
                });
            (capacity, message)
        })
        .collect();
        std::panic::set_hook(previous);

        for (capacity, message) in outcomes {
            let message = message.unwrap_or_else(|| {
                panic!(
                    "SumTree::new({capacity}) must panic; unguarded it returned \
                     a tree with an empty node array in release builds"
                )
            });
            assert!(
                message.contains("exceeds MAX_BUFFER_CAPACITY"),
                "SumTree::new({capacity}) must be rejected by the capacity \
                 guard, not by an overflow check that only exists in debug; \
                 got {message:?}"
            );
        }
    }

    /// The whole justification for keeping the reference around: identical seeds
    /// must yield identical slot sequences from both implementations, across
    /// ragged capacities, skewed masses, sparse occupancy, and every batch size.
    #[test]
    fn test_stratified_draw_matches_reference_across_seeds_and_shapes() {
        let mass_sets: [Vec<f64>; 6] = [
            vec![1.0; 8],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![1e-6, 1.0, 1e-6, 1e-6, 500.0],
            vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0],
            vec![3.5],
            (1..=33).map(f64::from).collect(),
        ];
        for (set_idx, masses) in mass_sets.iter().enumerate() {
            let (tree, scan) = build_pair(masses);
            for seed in 0..40u64 {
                for batch_size in [1usize, 2, 3, 5, 8, 32] {
                    let mut rng_a = StdRng::seed_from_u64(seed);
                    let mut rng_b = StdRng::seed_from_u64(seed);
                    let from_tree = stratified_draw(&tree, batch_size, &mut rng_a);
                    let from_scan = stratified_draw(&scan, batch_size, &mut rng_b);
                    assert_eq!(
                        from_tree, from_scan,
                        "sum-tree diverged from the O(n) reference \
                         (mass set {set_idx}, seed {seed}, batch {batch_size})"
                    );
                }
            }
        }
    }

    /// Stratification, not i.i.d.: with `k` equal-mass slots and `k` draws every
    /// segment boundary coincides with a slot boundary, so the draw is forced to
    /// be `[0, 1, ..., k-1]` regardless of seed. An i.i.d. categorical sampler
    /// reproduces that with probability `k!/k^k` — ≈1.5% at `k = 6`.
    #[test]
    fn test_stratified_draw_puts_exactly_one_draw_per_segment() {
        let (tree, _) = build_pair(&[1.0; 6]);
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let drawn = stratified_draw(&tree, 6, &mut rng);
            assert_eq!(
                drawn,
                vec![0, 1, 2, 3, 4, 5],
                "each equal-mass segment must yield exactly its own slot (seed {seed})"
            );
        }
    }

    /// One `rng` draw per segment, in segment order.
    #[test]
    fn test_stratified_draw_issues_exactly_batch_size_rng_draws() {
        let (tree, _) = build_pair(&[1.0, 2.0, 3.0, 4.0]);
        let mut rng = StdRng::seed_from_u64(7);
        let first = stratified_draw(&tree, 4, &mut rng);
        let second = stratified_draw(&tree, 4, &mut rng);

        // Reproduce with an RNG advanced by exactly four `random_range` calls.
        let mut probe = StdRng::seed_from_u64(7);
        let _ = stratified_draw(&tree, 4, &mut probe);
        let replay = stratified_draw(&tree, 4, &mut probe);
        assert_eq!(first.len(), 4, "one slot per segment");
        assert_eq!(
            second, replay,
            "the second batch must depend only on four consumed draws from the first"
        );
    }

    #[test]
    #[should_panic(expected = "batch_size must be non-zero")]
    fn test_stratified_draw_panics_on_zero_batch_size() {
        let (tree, _) = build_pair(&[1.0, 2.0]);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = stratified_draw(&tree, 0, &mut rng);
    }

    #[test]
    #[should_panic(expected = "p_total must be finite and positive")]
    fn test_stratified_draw_panics_on_empty_index() {
        let tree = SumTree::new(4);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = stratified_draw(&tree, 2, &mut rng);
    }
}
