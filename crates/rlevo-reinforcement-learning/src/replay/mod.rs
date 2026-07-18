//! The replay-strategy seam: how stored transitions are held, addressed, and
//! drawn (ADR 0050).
//!
//! # What this module owns
//!
//! - [`Transition<O, P>`] — the one stored item type, with the action payload
//!   erased into a type parameter ([`DiscreteTransition`] /
//!   [`ContinuousTransition`]).
//! - [`ReplayStrategy<T>`] — the seam. It decides **which items come back**;
//!   it never sees a `Tensor`, a `Backend`, or a device.
//! - [`TransitionId`] / [`SampledBatch`] — the sample-identity handles that
//!   travel from `sample` to the staging code and (in a later step) back to a
//!   priority writeback.
//! - [`UniformReplay<T>`] — the FIFO, uniform-with-replacement implementation
//!   every shipped agent uses today.
//! - [`PrioritizedReplay<T>`] — Schaul et al. (2016)'s proportional prioritized
//!   replay: `P(i) ∝ p_i^α` over a sum-tree, Schaul's *stratified* draw (one
//!   value per equal-mass segment, not i.i.d.), importance weights
//!   max-normalized over the sampled minibatch, and the `update_priorities`
//!   feedback edge. Its priorities are [`Priority`] values, finite and strictly
//!   positive by construction, configured by [`PrioritizedReplayConfig`].
//! - [`ReplayBufferError`] — the error domain for all of the above.
//!
//! # Choosing a strategy
//!
//! [`UniformReplay`] is the default and is what every agent uses today.
//! [`PrioritizedReplay`] is opt-in and is intended for the **value-based**
//! agents (DQN, C51, QR-DQN), where Rainbow's ablation puts prioritized replay
//! among the two most crucial of seven components. It is deliberately *not* the
//! default for DDPG/TD3/SAC: the continuous-control literature is
//! contested-to-negative on vanilla PER, so enabling it there would be a
//! paper-fidelity defect dressed as a feature (ADR 0050 §Context).
//!
//! The two are separate types rather than one α-parameterised implementation
//! because their **draw semantics differ**: uniform is i.i.d. with replacement,
//! Schaul's is stratified without it. One implementation cannot honour both, and
//! either choice would silently change the other's behaviour (ADR 0050 §4).
//!
//! # Where the line is drawn
//!
//! Staging — turning drawn transitions into tensors — stays in the agent,
//! because only the agent knows whether an action becomes a rank-2 `Int` index
//! tensor (DQN's `gather`) or a rank-2 float vector tensor (a continuous
//! critic's input). A seam that returned tensors could not serve both
//! families with one type, which is precisely why the pre-ADR-0050
//! `TrainingBatch` had no consumers.
//!
//! # Seeding
//!
//! [`ReplayStrategy::sample`] takes `rng: &mut R` from the caller and holds no
//! RNG of its own, matching every agent's
//! `learn_step<R: Rng + ?Sized>(&mut self, rng: &mut R)` signature and the
//! host-RNG convention of ADR 0029. A buffer that reached for `rand::rng()`
//! internally could never appear in a reproducibility test.
//!
//! # Examples
//!
//! ```
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//! use rlevo_reinforcement_learning::replay::{
//!     DiscreteTransition, ReplayStrategy, UniformReplay,
//! };
//!
//! let mut buffer: UniformReplay<DiscreteTransition<f32>> = UniformReplay::new(4);
//! for step in 0..6 {
//!     buffer.push(DiscreteTransition {
//!         obs: step as f32,
//!         action: 0,
//!         reward: 1.0,
//!         next_obs: step as f32 + 1.0,
//!         terminated: false,
//!     });
//! }
//! // Capacity 4, six pushes: the two oldest transitions were evicted.
//! assert_eq!(buffer.len(), 4);
//!
//! let mut rng = StdRng::seed_from_u64(7);
//! let batch = buffer.sample(4, 1.0, &mut rng).expect("four transitions stored");
//! assert_eq!(batch.ids().len(), 4, "one id per requested draw");
//! assert!(batch.weights().is_none(), "uniform emits no IS weights");
//! for &id in batch.ids() {
//!     assert!(buffer.get(id).is_some(), "a freshly drawn id always resolves");
//! }
//! ```

mod config;
mod error;
mod prioritized;
mod priority;
mod sum_tree;
mod transition;
mod uniform;

pub use config::{DEFAULT_PRIORITY_EPSILON, DEFAULT_PRIORITY_EXPONENT, PrioritizedReplayConfig};
pub use error::ReplayBufferError;
pub use prioritized::PrioritizedReplay;
pub use priority::{Priority, PriorityError};
pub use transition::{ContinuousTransition, DiscreteTransition, Transition};
pub use uniform::UniformReplay;

/// Absolute, monotone identifier of a stored transition.
///
/// The id is the transition's index in the buffer's *lifetime* insertion
/// sequence, not its slot in the current ring. It therefore survives FIFO
/// eviction in the only sense that matters: an evicted id resolves to `None`
/// instead of silently aliasing whichever transition now occupies that slot.
///
/// # Why not a bare `usize` slot
///
/// Today every agent samples and stages inside a single `learn_step` with no
/// interleaved `push`, so raw slot indices happen to be safe. Under the
/// distributed-replay architectures on the roadmap a learner's ids outlive the
/// collector's inserts and every raw index silently shifts by the eviction
/// count — a bug that corrupts training without ever raising an error. A `u64`
/// counter makes that class of bug unrepresentable for the cost of one
/// increment per insert.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TransitionId(u64);

impl TransitionId {
    /// Constructs an id from an absolute insertion index.
    ///
    /// Only a [`ReplayStrategy`] implementation should need this; it is public
    /// so out-of-crate strategies can be written against the same handle.
    #[must_use]
    pub const fn new(index: u64) -> Self {
        Self(index)
    }

    /// The absolute insertion index this id refers to.
    #[must_use]
    pub const fn index(self) -> u64 {
        self.0
    }
}

/// The result of one [`ReplayStrategy::sample`] call: which transitions were
/// drawn, and with what weight.
///
/// Carries *sample identity* across the staging → loss → (eventual) writeback
/// sequence. It is deliberately **not** an ADR 0046 [`Slot`], despite the
/// superficially similar "a value that must survive a learn step" description:
/// `Slot` exists to bound the window in which a Burn module is out of its field
/// across a by-value `Optimizer::step`, and carries poisoning semantics and an
/// ownership hazard that `SampledBatch` — plain host data — has none of.
///
/// [`Slot`]: crate::algorithms::shared::Slot
#[derive(Clone, Debug, PartialEq)]
pub struct SampledBatch {
    /// Drawn ids, in draw order.
    ids: Vec<TransitionId>,
    /// Per-id importance-sampling weights, or `None` for strategies that emit
    /// none. [`UniformReplay`] always yields `None`.
    weights: Option<Vec<f32>>,
}

impl SampledBatch {
    /// Constructs an unweighted batch — the uniform case.
    #[must_use]
    pub const fn unweighted(ids: Vec<TransitionId>) -> Self {
        Self { ids, weights: None }
    }

    /// Constructs a weighted batch.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != ids.len()`. A weight vector that does not
    /// line up with its ids is a programming error in the strategy, and one
    /// that would otherwise surface as a silently mis-scaled loss.
    #[must_use]
    pub fn weighted(ids: Vec<TransitionId>, weights: Vec<f32>) -> Self {
        assert_eq!(
            ids.len(),
            weights.len(),
            "SampledBatch weights must be one-per-id"
        );
        Self {
            ids,
            weights: Some(weights),
        }
    }

    /// The drawn ids, in draw order.
    #[must_use]
    pub fn ids(&self) -> &[TransitionId] {
        &self.ids
    }

    /// The per-id importance-sampling weights, if the strategy emits any.
    #[must_use]
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// Number of transitions drawn.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// `true` when no transitions were drawn.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// How a replay buffer stores transitions and chooses which to hand back.
///
/// Implement this to add a new replay scheme (prioritized, hindsight-relabeled,
/// reservoir). The trait is generic over the stored item `T` and knows nothing
/// about tensors, so an implementation is pure host-side bookkeeping and is
/// testable without a Burn backend.
///
/// # Invariants
///
/// Implementations must uphold all of:
///
/// - `len()` never exceeds the configured capacity. Eviction is explicit; do
///   not rely on a container's internal capacity.
/// - Every id in a freshly returned [`SampledBatch`] resolves through
///   [`get`](Self::get) at the moment `sample` returns. Ids may later become
///   unresolvable through eviction — that is the point of [`TransitionId`].
/// - `sample` never mutates the buffer, and draws all of its randomness from
///   the supplied `rng` (ADR 0029).
///
/// # Examples
///
/// See the [module documentation](self) for an end-to-end
/// push / sample / resolve cycle against [`UniformReplay`].
pub trait ReplayStrategy<T> {
    /// Stores `item`, evicting as needed to stay within capacity.
    fn push(&mut self, item: T);

    /// Number of transitions currently stored.
    fn len(&self) -> usize;

    /// `true` when no transitions are stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolves an id to the transition it names.
    ///
    /// Returns `None` if the id has been evicted or was never issued by this
    /// buffer.
    fn get(&self, id: TransitionId) -> Option<&T>;

    /// Mutable counterpart of [`get`](Self::get), for post-insertion mutation.
    ///
    /// Prioritized replay's priority writeback and any future in-place
    /// relabeling are instances of this one capability, which is why the seam
    /// carries it from day one rather than growing it later as a breaking
    /// change.
    fn get_mut(&mut self, id: TransitionId) -> Option<&mut T>;

    /// Iterates the stored transitions, oldest first.
    ///
    /// The explicit `'a` and `T: 'a` are what let the returned opaque iterator
    /// borrow from `self`; RPITIT cannot infer the relationship between the
    /// receiver's lifetime and the item type on its own.
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a;

    /// Draws `batch_size` transition ids.
    ///
    /// # Arguments
    ///
    /// - `batch_size` — number of ids to draw. Whether draws are independent,
    ///   with replacement, or stratified is the strategy's choice; see the
    ///   implementation's own docs.
    /// - `beta` — the current importance-sampling exponent. Ignored entirely by
    ///   strategies that emit no weights, including [`UniformReplay`]. It is on
    ///   the signature so that adding a weighted strategy does not break this
    ///   trait's other implementors.
    /// - `rng` — the caller's RNG. All randomness comes from here.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayBufferError::InsufficientData`] when the buffer holds
    /// fewer than `batch_size` transitions.
    fn sample<R: rand::Rng + ?Sized>(
        &self,
        batch_size: usize,
        beta: f32,
        rng: &mut R,
    ) -> Result<SampledBatch, ReplayBufferError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_id_roundtrips_its_index() {
        let id = TransitionId::new(1_234);
        assert_eq!(
            id.index(),
            1_234,
            "index() must return what new() was given"
        );
    }

    #[test]
    fn test_sampled_batch_unweighted_has_no_weights() {
        let batch = SampledBatch::unweighted(vec![TransitionId::new(0), TransitionId::new(1)]);
        assert_eq!(batch.len(), 2, "batch length is its id count");
        assert!(!batch.is_empty(), "a two-id batch is not empty");
        assert!(
            batch.weights().is_none(),
            "an unweighted batch must expose no weights"
        );
    }

    #[test]
    fn test_sampled_batch_weighted_exposes_weights() {
        let batch = SampledBatch::weighted(
            vec![TransitionId::new(3), TransitionId::new(4)],
            vec![0.5, 1.0],
        );
        assert_eq!(
            batch.weights(),
            Some([0.5_f32, 1.0].as_slice()),
            "weights must be returned one-per-id, in order"
        );
    }

    #[test]
    #[should_panic(expected = "SampledBatch weights must be one-per-id")]
    fn test_sampled_batch_weighted_rejects_length_mismatch() {
        let _ = SampledBatch::weighted(vec![TransitionId::new(0)], vec![0.5, 1.0]);
    }

    #[test]
    fn test_sampled_batch_empty_is_empty() {
        let batch = SampledBatch::unweighted(Vec::new());
        assert!(batch.is_empty(), "a zero-id batch reports empty");
        assert_eq!(batch.len(), 0, "a zero-id batch has length zero");
    }
}
