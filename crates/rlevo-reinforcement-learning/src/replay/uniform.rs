//! Uniform experience replay: a FIFO ring drawn i.i.d. with replacement.
//!
//! This is the strategy every shipped off-policy agent uses. It is a separate
//! type from any future prioritized strategy rather than its `α = 0` special
//! case: Schaul's scheme draws *stratified*, one per equal-mass segment, while
//! uniform draws i.i.d. with replacement, so one implementation cannot honour
//! both semantics (ADR 0050 §4).

use std::collections::VecDeque;

use rand::{Rng, RngExt};

use super::{ImportanceExponent, ReplayBufferError, ReplayStrategy, SampledBatch, TransitionId};

/// A fixed-capacity FIFO replay buffer sampled uniformly with replacement.
///
/// Pushing into a full buffer evicts the oldest transition. Sampling draws
/// `batch_size` independent uniform indices over the live window — the same
/// transition may appear more than once in a batch, which is the standard
/// (and, before ADR 0050, hand-rolled per agent) behaviour.
///
/// # Draw-order contract
///
/// [`sample`](ReplayStrategy::sample) issues **exactly `batch_size` calls to
/// `rng.random_range(0..len)`, in order, and returns the drawn ids in that
/// order**. This is a pinned behavioural contract, not an implementation
/// detail: it is what makes the ADR 0050 seam migration a bit-identical no-op
/// against the six hand-rolled buffers it replaced, and every seeded baseline
/// in this crate depends on it. A later "optimization" — batched draws, a
/// shuffle, sampling without replacement — would silently move every one of
/// those baselines, so the contract is enforced by
/// `test_uniform_replay_draw_order_matches_pinned_contract` below rather than
/// left as a comment.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
/// use rlevo_reinforcement_learning::replay::{
///     ImportanceExponent, ReplayStrategy, UniformReplay,
/// };
///
/// let mut buffer: UniformReplay<u32> = UniformReplay::new(3);
/// for value in 0..5 {
///     buffer.push(value);
/// }
/// // Capacity 3: values 0 and 1 were evicted.
/// assert_eq!(buffer.iter().copied().collect::<Vec<_>>(), vec![2, 3, 4]);
///
/// let mut rng = StdRng::seed_from_u64(1);
/// let batch = buffer
///     .sample(2, ImportanceExponent::ONE, &mut rng)
///     .expect("enough data");
/// assert_eq!(batch.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct UniformReplay<T> {
    /// Live transitions, oldest at the front.
    buffer: VecDeque<T>,
    /// Maximum number of live transitions.
    capacity: usize,
    /// Total number of pushes over the buffer's lifetime.
    ///
    /// Slot `i` of a buffer holding `len` items after `pushes` inserts has
    /// absolute id `pushes - len + i`, so resolving a [`TransitionId`] is one
    /// subtraction and a bounds check.
    pushes: u64,
}

impl<T> UniformReplay<T> {
    /// Creates an empty buffer that holds at most `capacity` transitions.
    ///
    /// The backing storage is pre-allocated to `capacity`, matching the
    /// pre-seam `VecDeque::with_capacity(config.replay_buffer_capacity)` each
    /// agent used.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`. A zero-capacity replay buffer cannot hold the
    /// transition it was just handed, and the pre-seam eviction shape would
    /// have grown without bound instead of refusing. Every in-crate call site
    /// passes a config field that
    /// [`Validate`](rlevo_core::config::Validate) has already rejected zero
    /// for, so this is a programming error, never user data.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "replay buffer capacity must be non-zero");
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            pushes: 0,
        }
    }

    /// Maximum number of transitions this buffer will hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Absolute id of the oldest live transition.
    ///
    /// `pushes >= buffer.len()` always holds — `pushes` only ever increases and
    /// `buffer.len()` never exceeds it — so the subtraction cannot underflow.
    fn oldest_id(&self) -> u64 {
        self.pushes - self.buffer.len() as u64
    }

    /// Maps an absolute id onto a slot in the live window, or `None` if the id
    /// is outside it.
    fn slot_of(&self, id: TransitionId) -> Option<usize> {
        let offset = id.index().checked_sub(self.oldest_id())?;
        let slot = usize::try_from(offset).ok()?;
        (slot < self.buffer.len()).then_some(slot)
    }
}

impl<T> ReplayStrategy<T> for UniformReplay<T> {
    fn push(&mut self, item: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
        self.pushes += 1;
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn get(&self, id: TransitionId) -> Option<&T> {
        self.buffer.get(self.slot_of(id)?)
    }

    fn get_mut(&mut self, id: TransitionId) -> Option<&mut T> {
        let slot = self.slot_of(id)?;
        self.buffer.get_mut(slot)
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        self.buffer.iter()
    }

    /// Draws `batch_size` ids uniformly at random, with replacement.
    ///
    /// `beta` is ignored — uniform replay emits no importance-sampling
    /// weights, so the returned [`SampledBatch`] always has
    /// [`weights()`](SampledBatch::weights) of `None`. It stays on the
    /// signature so that adding a weighted strategy does not break this
    /// implementor.
    ///
    /// See the [type-level draw-order contract](UniformReplay#draw-order-contract):
    /// exactly `batch_size` calls to `rng.random_range(0..len)`, in order.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayBufferError::InsufficientData`] when fewer than
    /// `batch_size` transitions are stored.
    fn sample<R: Rng + ?Sized>(
        &self,
        batch_size: usize,
        _beta: ImportanceExponent,
        rng: &mut R,
    ) -> Result<SampledBatch, ReplayBufferError> {
        let len = self.buffer.len();
        if batch_size > len {
            return Err(ReplayBufferError::InsufficientData {
                requested: batch_size,
                available: len,
            });
        }
        let oldest = self.oldest_id();
        let ids = (0..batch_size)
            .map(|_| TransitionId::new(oldest + rng.random_range(0..len) as u64))
            .collect();
        Ok(SampledBatch::unweighted(ids))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Builds a buffer of `capacity` pre-filled with `0..pushes`.
    fn filled(capacity: usize, pushes: u32) -> UniformReplay<u32> {
        let mut buffer = UniformReplay::new(capacity);
        for value in 0..pushes {
            buffer.push(value);
        }
        buffer
    }

    /// The ADR 0050 §5 contract, spelled as executable code: the *pre-seam*
    /// expression every one of the six agents inlined.
    ///
    /// If this ever disagrees with `UniformReplay::sample`, every seeded
    /// baseline in the crate has moved.
    fn pinned_reference_draw<R: Rng + ?Sized>(
        batch_size: usize,
        len: usize,
        rng: &mut R,
    ) -> Vec<usize> {
        (0..batch_size).map(|_| rng.random_range(0..len)).collect()
    }

    #[test]
    fn test_uniform_replay_new_preallocates_and_starts_empty() {
        let buffer: UniformReplay<u32> = UniformReplay::new(8);
        assert_eq!(buffer.capacity(), 8, "capacity is what new() was given");
        assert_eq!(buffer.len(), 0, "a fresh buffer holds nothing");
        assert!(buffer.is_empty(), "a fresh buffer reports empty");
    }

    #[test]
    #[should_panic(expected = "replay buffer capacity must be non-zero")]
    fn test_uniform_replay_new_rejects_zero_capacity() {
        let _: UniformReplay<u32> = UniformReplay::new(0);
    }

    #[test]
    fn test_uniform_replay_push_evicts_oldest_at_capacity() {
        let buffer = filled(3, 5);
        assert_eq!(buffer.len(), 3, "length is clamped to capacity");
        assert_eq!(
            buffer.iter().copied().collect::<Vec<_>>(),
            vec![2, 3, 4],
            "eviction is FIFO: the two oldest pushes are gone"
        );
    }

    #[test]
    fn test_uniform_replay_get_resolves_live_ids_to_their_transitions() {
        let buffer = filled(3, 5);
        for (slot, expected) in [2_u32, 3, 4].into_iter().enumerate() {
            let id = TransitionId::new(2 + slot as u64);
            assert_eq!(
                buffer.get(id).copied(),
                Some(expected),
                "id {} must resolve to the value pushed at that absolute index",
                2 + slot
            );
        }
    }

    #[test]
    fn test_uniform_replay_get_returns_none_for_evicted_id() {
        let buffer = filled(3, 5);
        assert!(
            buffer.get(TransitionId::new(1)).is_none(),
            "an evicted id must resolve to None, never to a different transition"
        );
        assert!(
            buffer.get(TransitionId::new(5)).is_none(),
            "an id that was never issued must resolve to None"
        );
    }

    #[test]
    fn test_uniform_replay_get_mut_writes_through_to_storage() {
        let mut buffer = filled(3, 5);
        let id = TransitionId::new(3);
        *buffer.get_mut(id).expect("id 3 is live") = 99;
        assert_eq!(
            buffer.get(id).copied(),
            Some(99),
            "get_mut must alias the same storage get() reads"
        );
        assert!(
            buffer.get_mut(TransitionId::new(0)).is_none(),
            "get_mut on an evicted id must return None"
        );
    }

    /// ADR 0050 §5: the pinned draw-order contract.
    ///
    /// `sample` must issue exactly `batch_size` calls to
    /// `rng.random_range(0..len)`, in order, and return the drawn indices in
    /// that order. Asserted against a reference RNG advanced in lockstep, so a
    /// batched draw, a shuffle, or a switch to sampling without replacement
    /// fails here before it can move a single training baseline.
    #[test]
    fn test_uniform_replay_draw_order_matches_pinned_contract() {
        for &(capacity, pushes, batch_size) in &[
            (8_usize, 5_u32, 4_usize),
            (8, 8, 8),
            (4, 10, 3),
            (16, 16, 16),
            (1, 1, 1),
        ] {
            let buffer = filled(capacity, pushes);
            let len = buffer.len();
            let oldest = buffer.oldest_id();

            let mut rng_actual = StdRng::seed_from_u64(0x0050_ADD0);
            let mut rng_reference = StdRng::seed_from_u64(0x0050_ADD0);

            let batch = buffer
                .sample(batch_size, ImportanceExponent::ONE, &mut rng_actual)
                .expect("batch_size never exceeds len in this table");
            let reference = pinned_reference_draw(batch_size, len, &mut rng_reference);

            let actual: Vec<usize> = batch
                .ids()
                .iter()
                .map(|id| usize::try_from(id.index() - oldest).expect("live slot"))
                .collect();
            assert_eq!(
                actual, reference,
                "draw order must match the pinned pre-seam expression \
                 (capacity {capacity}, pushes {pushes}, batch {batch_size})"
            );

            // The RNG must also be left in the same state: same number of
            // draws, of the same type, over the same range.
            assert_eq!(
                rng_actual.random::<u64>(),
                rng_reference.random::<u64>(),
                "sample must consume exactly batch_size draws and no more"
            );
        }
    }

    #[test]
    fn test_uniform_replay_sample_ids_all_resolve() {
        let buffer = filled(4, 9);
        let mut rng = StdRng::seed_from_u64(11);
        let batch = buffer.sample(16, ImportanceExponent::ONE, &mut rng);
        assert!(
            batch.is_err(),
            "requesting more than len must be InsufficientData"
        );

        let batch = buffer
            .sample(4, ImportanceExponent::ONE, &mut rng)
            .expect("exactly len");
        assert!(
            batch.weights().is_none(),
            "uniform replay must not emit importance-sampling weights"
        );
        for &id in batch.ids() {
            assert!(
                buffer.get(id).is_some(),
                "every freshly drawn id must resolve"
            );
        }
    }

    #[test]
    fn test_uniform_replay_sample_draws_with_replacement() {
        // A single-element buffer can only be drawn with replacement.
        let buffer = filled(1, 1);
        let mut rng = StdRng::seed_from_u64(3);
        let batch = buffer
            .sample(1, ImportanceExponent::ONE, &mut rng)
            .expect("one element");
        assert_eq!(batch.len(), 1, "one draw requested, one returned");

        let buffer = filled(2, 2);
        let mut rng = StdRng::seed_from_u64(3);
        let batch = buffer
            .sample(2, ImportanceExponent::ONE, &mut rng)
            .expect("two elements");
        assert_eq!(
            batch.len(),
            2,
            "with replacement, batch size is independent of duplicate draws"
        );
    }

    #[test]
    fn test_uniform_replay_sample_reports_insufficient_data() {
        let buffer = filled(8, 3);
        let mut rng = StdRng::seed_from_u64(5);
        let err = buffer
            .sample(4, ImportanceExponent::ONE, &mut rng)
            .expect_err("4 > 3 stored transitions");
        match err {
            ReplayBufferError::InsufficientData {
                requested,
                available,
            } => {
                assert_eq!(requested, 4, "error reports the requested batch size");
                assert_eq!(available, 3, "error reports the stored count");
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn test_uniform_replay_sample_on_empty_buffer_is_insufficient_data() {
        let buffer: UniformReplay<u32> = UniformReplay::new(4);
        let mut rng = StdRng::seed_from_u64(5);
        assert!(
            buffer.sample(1, ImportanceExponent::ONE, &mut rng).is_err(),
            "an empty buffer cannot serve a draw"
        );
        let batch = buffer
            .sample(0, ImportanceExponent::ONE, &mut rng)
            .expect("a zero-size batch needs no data");
        assert!(batch.is_empty(), "zero draws requested, zero returned");
    }

    #[test]
    fn test_uniform_replay_ids_stay_stable_across_eviction() {
        let mut buffer = filled(3, 3);
        let id = TransitionId::new(2);
        assert_eq!(
            buffer.get(id).copied(),
            Some(2),
            "precondition: id 2 is live"
        );
        buffer.push(3);
        assert_eq!(
            buffer.get(id).copied(),
            Some(2),
            "an id must keep naming the same transition after an eviction \
             shifts every raw slot index"
        );
    }
}
