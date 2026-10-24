//! The replay buffers the off-policy agents store transitions in: how stored
//! transitions are held, addressed, and drawn (ADR 0050).
//!
//! Only [`PrioritizedReplaySettings`], the config opt-in, is public; the
//! buffers are crate-private (ADR 0095). What follows is their design, for
//! contributors.
//!
//! # What this module owns
//!
//! - `Transition<O, P>` — the one stored item type, with the action payload
//!   erased into a type parameter (`DiscreteTransition` /
//!   `ContinuousTransition`).
//! - `ReplayStrategy<T>` — the seam. It decides **which items come back**;
//!   it never sees a `Tensor`, a `Backend`, or a device.
//! - `TransitionId` / `SampledBatch` — the sample-identity handles that
//!   travel from `sample` to the staging code and (in a later step) back to a
//!   priority writeback.
//! - `UniformReplay<T>` — the FIFO, uniform-with-replacement implementation
//!   every shipped agent uses today. Built from a literal capacity with
//!   `UniformReplay::new`, or from a `UniformReplayConfig` — via
//!   `UniformReplay::from_config` — whenever the capacity is runtime-derived
//!   or deserialized.
//! - `PrioritizedReplay<T>` — Schaul et al. (2016)'s proportional prioritized
//!   replay: `$P(i) \propto p_i^\alpha$` over a sum-tree, Schaul's *stratified* draw (one
//!   value per equal-mass segment, not i.i.d.), importance weights
//!   max-normalized over the sampled minibatch, and the `update_priorities`
//!   feedback edge. Its priorities are `Priority` values, finite and strictly
//!   positive by construction, configured by `PrioritizedReplayConfig`.
//! - `ImportanceExponent` — the `$\beta$` passed to `ReplayStrategy::sample`,
//!   `finite && [0, 1]` by construction so a non-finite exponent cannot reach
//!   `powf` and turn the importance weights — and then the loss, and then every
//!   gradient — into `NaN` (ADR 0051 §3).
//! - `ReplayBufferError` — the error domain for all of the above.
//!
//! # Choosing a strategy
//!
//! `UniformReplay` is the default and is what every agent uses today.
//! `PrioritizedReplay` is opt-in and is intended for the **value-based**
//! agents (DQN, C51, QR-DQN), where Rainbow's ablation puts prioritized replay
//! among the two most crucial of seven components. It is deliberately *not* the
//! default for DDPG/TD3/SAC: the continuous-control literature is
//! contested-to-negative on vanilla PER, so enabling it there would be a
//! paper-fidelity defect dressed as a feature (ADR 0050 §Context).
//!
//! The two are separate types rather than one `$\alpha$`-parameterised implementation because their
//! **draw semantics differ**: uniform is i.i.d. with replacement, Schaul's is stratified without
//! it. One implementation cannot honour both, and either choice would silently change the other's
//! behaviour (ADR 0050 §4).
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
//! `ReplayStrategy::sample` takes `rng: &mut R` from the caller and holds no
//! RNG of its own, matching every agent's
//! `learn_step<R: Rng + ?Sized>(&mut self, rng: &mut R)` signature and the
//! host-RNG convention of ADR 0029. A buffer that reached for `rand::rng()`
//! internally could never appear in a reproducibility test.
//!

mod config;
mod error;
mod importance_exponent;
mod kind;
mod prioritized;
mod priority;
mod settings;
mod sum_tree;
mod transition;
mod uniform;

#[cfg(test)]
pub(crate) use config::PrioritizedReplayConfig;
pub(crate) use config::UniformReplayConfig;
pub(crate) use error::ReplayBufferError;
pub(crate) use importance_exponent::ImportanceExponent;
pub(crate) use kind::ReplayKind;
pub use settings::PrioritizedReplaySettings;
pub(crate) use transition::{ContinuousTransition, DiscreteTransition};
pub(crate) use uniform::UniformReplay;

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
pub(crate) struct TransitionId(u64);

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
/// [`Slot`]: crate::algorithms::Slot
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SampledBatch {
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

    #[cfg(test)]
    /// Number of transitions drawn.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[cfg(test)]
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
pub(crate) trait ReplayStrategy<T> {
    /// Stores `item`, evicting as needed to stay within capacity.
    fn push(&mut self, item: T);

    /// Number of transitions currently stored.
    fn len(&self) -> usize;

    /// Resolves an id to the transition it names.
    ///
    /// Returns `None` if the id has been evicted or was never issued by this
    /// buffer.
    fn get(&self, id: TransitionId) -> Option<&T>;

    /// Draws `batch_size` transition ids.
    ///
    /// # Arguments
    ///
    /// - `batch_size` — number of ids to draw. Whether draws are independent,
    ///   with replacement, or stratified is the strategy's choice; see the
    ///   implementation's own docs.
    /// - `beta` — the current importance-sampling exponent, already evaluated
    ///   for this step. Its `finite && [0, 1]` invariant is carried by
    ///   [`ImportanceExponent`], so no implementation needs to re-check it.
    ///   Ignored entirely by strategies that emit no weights, including
    ///   [`UniformReplay`]. It is on the signature so that adding a weighted
    ///   strategy does not break this trait's other implementors.
    /// - `rng` — the caller's RNG. All randomness comes from here.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayBufferError::InsufficientData`] when the buffer holds fewer than
    /// `batch_size` transitions. Note that this is *not* the place a bad `$\beta$` is reported:
    /// `$\beta$` cannot be bad here, so [`UniformReplay`] is not made to carry an error variant it
    /// could never produce (ADR 0051 §2).
    fn sample<R: rand::Rng + ?Sized>(
        &self,
        batch_size: usize,
        beta: ImportanceExponent,
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
