//! [`ReplayKind`] — the agent-side dispatch over replay strategies (ADR 0051 §1).
//!
//! An agent that can be configured either way holds a `ReplayKind<T>` rather than
//! a generic strategy parameter, so prioritization stays a **config-level**
//! opt-in: `PrioritizedReplay` need not appear in the agent's type, and a
//! deserialized config field can turn it on. `ReplayKind` forwards every
//! [`ReplayStrategy`] method to the active variant, and dispatches the priority
//! writeback through an **exhaustive `match`**.
//!
//! # Why the writeback is an exhaustive match, not a defaulted trait method
//!
//! ADR 0050 §3 sketched `update_priorities` as a *defaulted* trait method
//! (no-op for uniform). ADR 0051 §1 supersedes that: a defaulted no-op lets a
//! future third strategy silently swallow a priority writeback it ought to
//! honour — the exact "priorities write-once, TD error never fed back" defect
//! that made the pre-ADR-0050 `PrioritizedExperienceReplay` not-PER. An
//! exhaustive `match` turns the same case into a **compile error**: when HER
//! lands as a third variant, [`update_priorities_from_td_errors`] fails to
//! compile until someone decides, explicitly, what a hindsight buffer does with
//! a writeback. That is the right place for the question to be asked.
//!
//! [`update_priorities_from_td_errors`]: ReplayKind::update_priorities_from_td_errors

use rand::Rng;
use rlevo_core::config::ConfigError;

use super::config::PrioritizedReplayConfig;
use super::error::ReplayBufferError;
use super::importance_exponent::ImportanceExponent;
use super::prioritized::PrioritizedReplay;
use super::priority::PriorityError;
use super::uniform::UniformReplay;
use super::{ReplayStrategy, SampledBatch, TransitionId};

/// A replay buffer that is either [`UniformReplay`] or [`PrioritizedReplay`],
/// chosen at agent-construction time from a config field.
///
/// Implements [`ReplayStrategy<T>`] by forwarding each method to the active
/// variant, so an agent's `learn_step` calls `push`/`len`/`get`/`sample`/`iter`
/// without knowing which strategy backs them. The one operation that is *not* on
/// the trait — the priority writeback — is an inherent method
/// ([`update_priorities_from_td_errors`](Self::update_priorities_from_td_errors))
/// dispatched through an exhaustive `match`.
///
/// # A deliberately closed set
///
/// `ReplayKind` is closed to out-of-crate strategies (ADR 0051 §1): no such
/// consumer exists, `Slot`'s `pub(crate)` visibility is in-repo precedent, and
/// it is a two-way door — the enum can grow a variant, or the dispatch can move
/// to a generic parameter later, without breaking a stored format.
/// [`ReplayStrategy`] itself stays public and implementable; only the agent's
/// dispatch is closed.
pub enum ReplayKind<T> {
    /// FIFO, uniform-with-replacement — the default for every shipped agent.
    Uniform(UniformReplay<T>),
    /// Schaul et al. (2016) proportional prioritized replay — opt-in for the
    /// value-based agents (DQN, C51, QR-DQN).
    Prioritized(PrioritizedReplay<T>),
}

// `Debug` reports the variant and live length only, and is written by hand so it
// does **not** require `T: Debug` — agents print `buffer_len`, never the stored
// transitions themselves.
impl<T> std::fmt::Debug for ReplayKind<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, len) = match self {
            Self::Uniform(u) => ("Uniform", ReplayStrategy::len(u)),
            Self::Prioritized(p) => ("Prioritized", ReplayStrategy::len(p)),
        };
        f.debug_struct("ReplayKind")
            .field("kind", &kind)
            .field("len", &len)
            .finish()
    }
}

impl<T> ReplayKind<T> {
    /// Builds a uniform buffer holding at most `capacity` transitions.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0` (see [`UniformReplay::new`]). Every in-crate
    /// call site passes a `replay_buffer_capacity` a [`Validate`] impl has
    /// already rejected zero for.
    ///
    /// [`Validate`]: rlevo_core::config::Validate
    #[must_use]
    pub fn uniform(capacity: usize) -> Self {
        Self::Uniform(UniformReplay::new(capacity))
    }

    /// Builds a prioritized buffer from a validated [`PrioritizedReplayConfig`].
    ///
    /// # Errors
    ///
    /// Returns the first [`ConfigError`] from the config's `validate` — a zero
    /// capacity, a `priority_exponent` outside `[0, 1]`, or a non-positive
    /// `priority_epsilon`.
    pub fn prioritized(config: PrioritizedReplayConfig) -> Result<Self, ConfigError> {
        Ok(Self::Prioritized(PrioritizedReplay::new(config)?))
    }

    /// Whether this buffer prioritizes draws (and therefore emits importance
    /// weights and honours a priority writeback).
    ///
    /// An agent's `learn_step` gates its per-sample priority-signal computation
    /// on this: uniform replay pays for none of it.
    #[must_use]
    pub const fn is_prioritized(&self) -> bool {
        matches!(self, Self::Prioritized(_))
    }

    /// Feeds per-sample TD-error-like priority signals back for previously
    /// sampled ids — Schaul Algorithm 1 lines 11-12, the feedback edge that
    /// makes prioritized replay PER rather than a weighted-random buffer.
    ///
    /// Dispatched through an **exhaustive `match`**, not a defaulted trait
    /// method: a third strategy cannot silently swallow the writeback (ADR 0051
    /// §1, and the module docs).
    ///
    /// - **Uniform** has no priorities, so this is a deliberate no-op — written
    ///   out explicitly so the compiler forces the same decision for any future
    ///   variant.
    /// - **Prioritized** applies `p_i = |signal_i| + ε` to each signal and
    ///   writes it back, dropping any id evicted since it was sampled.
    ///
    /// # Errors
    ///
    /// Returns [`PriorityError`] for the first non-finite signal — the reachable
    /// production case being a diverging network — having written nothing, so a
    /// single `NaN` leaves the buffer unmodified rather than half-updated.
    ///
    /// Every in-crate caller — each agent's `learn_step` — logs this `Err` with
    /// `tracing::warn!` and skips the writeback rather than propagating it, for
    /// three reasons. First, `learn_step` returns `Option<LearnOutcome>` with no
    /// error channel; threading one through for this event would be a breaking
    /// signature change out of all proportion to it. Second, a diverging network
    /// that poisons the TD errors with `NaN` poisons the loss the same step —
    /// both are functions of the same prediction/target tensors — so
    /// `LearnOutcome.loss` already surfaces the divergence to the caller; the
    /// skip declines to report it twice, it hides nothing. Third, the all-or-
    /// nothing validation above leaves the buffer untouched, so a skipped
    /// writeback costs only one stale priority per sampled id — never a
    /// corrupted total.
    ///
    /// # Panics
    ///
    /// Panics when `ids.len() != signals.len()` on the prioritized path; both
    /// come from one [`SampledBatch`] in every intended call.
    pub fn update_priorities_from_td_errors(
        &mut self,
        ids: &[TransitionId],
        signals: &[f32],
    ) -> Result<(), PriorityError> {
        match self {
            Self::Uniform(_) => Ok(()),
            Self::Prioritized(p) => p.update_priorities_from_td_errors(ids, signals),
        }
    }

    /// Test-only view of the inner prioritized buffer, for inspecting priorities
    /// and totals after a writeback.
    #[cfg(test)]
    pub(crate) fn as_prioritized(&self) -> Option<&PrioritizedReplay<T>> {
        match self {
            Self::Prioritized(p) => Some(p),
            Self::Uniform(_) => None,
        }
    }
}

/// Iterator over the stored transitions of either [`ReplayKind`] variant.
///
/// [`ReplayStrategy::iter`] is RPITIT, so a `match` cannot delegate it directly —
/// the two arms return different opaque iterator types. This two-variant adapter
/// unifies them without a `Box<dyn Iterator>` allocation, which would defeat the
/// point of a per-`learn_step` hot-path call.
#[derive(Debug)]
pub enum KindIter<A, B> {
    /// Iterating a [`UniformReplay`].
    Uniform(A),
    /// Iterating a [`PrioritizedReplay`].
    Prioritized(B),
}

impl<A, B, I> Iterator for KindIter<A, B>
where
    A: Iterator<Item = I>,
    B: Iterator<Item = I>,
{
    type Item = I;

    fn next(&mut self) -> Option<I> {
        match self {
            Self::Uniform(a) => a.next(),
            Self::Prioritized(b) => b.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Uniform(a) => a.size_hint(),
            Self::Prioritized(b) => b.size_hint(),
        }
    }
}

impl<T> ReplayStrategy<T> for ReplayKind<T> {
    fn push(&mut self, item: T) {
        match self {
            Self::Uniform(u) => u.push(item),
            Self::Prioritized(p) => p.push(item),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Uniform(u) => u.len(),
            Self::Prioritized(p) => p.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Uniform(u) => u.is_empty(),
            Self::Prioritized(p) => p.is_empty(),
        }
    }

    fn get(&self, id: TransitionId) -> Option<&T> {
        match self {
            Self::Uniform(u) => u.get(id),
            Self::Prioritized(p) => p.get(id),
        }
    }

    fn get_mut(&mut self, id: TransitionId) -> Option<&mut T> {
        match self {
            Self::Uniform(u) => u.get_mut(id),
            Self::Prioritized(p) => p.get_mut(id),
        }
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        match self {
            Self::Uniform(u) => KindIter::Uniform(u.iter()),
            Self::Prioritized(p) => KindIter::Prioritized(p.iter()),
        }
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        batch_size: usize,
        beta: ImportanceExponent,
        rng: &mut R,
    ) -> Result<SampledBatch, ReplayBufferError> {
        match self {
            Self::Uniform(u) => u.sample(batch_size, beta, rng),
            Self::Prioritized(p) => p.sample(batch_size, beta, rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReplayKind;
    use crate::replay::{
        DiscreteTransition, ImportanceExponent, PrioritizedReplayConfig, ReplayStrategy,
        TransitionId,
    };
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn transition(label: f32) -> DiscreteTransition<f32> {
        DiscreteTransition {
            obs: label,
            action: 0,
            reward: 1.0,
            next_obs: label + 1.0,
            terminated: false,
        }
    }

    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
    fn uniform() -> ReplayKind<DiscreteTransition<f32>> {
        let mut b = ReplayKind::uniform(8);
        for i in 0..4 {
            b.push(transition(i as f32));
        }
        b
    }

    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
    fn prioritized() -> ReplayKind<DiscreteTransition<f32>> {
        let mut b = ReplayKind::prioritized(PrioritizedReplayConfig {
            capacity: 8,
            ..PrioritizedReplayConfig::default()
        })
        .expect("valid config");
        for i in 0..4 {
            b.push(transition(i as f32));
        }
        b
    }

    #[test]
    fn test_uniform_kind_reports_not_prioritized_and_emits_no_weights() {
        let b = uniform();
        assert!(
            !b.is_prioritized(),
            "the uniform variant is not prioritized"
        );
        assert_eq!(b.len(), 4, "forwarded len must reflect the pushes");
        let mut rng = StdRng::seed_from_u64(1);
        let batch = b
            .sample(3, ImportanceExponent::ONE, &mut rng)
            .expect("enough data");
        assert!(
            batch.weights().is_none(),
            "uniform replay emits no importance weights"
        );
    }

    #[test]
    fn test_prioritized_kind_reports_prioritized_and_emits_weights() {
        let b = prioritized();
        assert!(b.is_prioritized(), "the prioritized variant is prioritized");
        let mut rng = StdRng::seed_from_u64(1);
        let batch = b
            .sample(3, ImportanceExponent::new(0.4), &mut rng)
            .expect("enough data");
        assert!(
            batch.weights().is_some(),
            "prioritized replay emits max-normalized importance weights"
        );
    }

    #[test]
    fn test_update_priorities_is_a_noop_on_uniform() {
        let mut b = uniform();
        let ids: Vec<TransitionId> = (0..4).map(TransitionId::new).collect();
        b.update_priorities_from_td_errors(&ids, &[0.5, 1.0, 2.0, 3.0])
            .expect("uniform writeback is a no-op, never an error");
        assert_eq!(b.len(), 4, "the no-op must not disturb the buffer");
    }

    #[test]
    fn test_update_priorities_shifts_the_prioritized_distribution() {
        let mut b = prioritized();
        let ids: Vec<TransitionId> = (0..4).map(TransitionId::new).collect();
        let before = b.as_prioritized().expect("prioritized").total_priority();
        b.update_priorities_from_td_errors(&ids, &[0.0, 0.0, 5.0, 0.0])
            .expect("finite signals");
        let after = b.as_prioritized().expect("prioritized").total_priority();
        assert!(
            (after - before).abs() > 1e-9,
            "a real writeback must move the total priority mass: {before} -> {after}"
        );
    }

    #[test]
    fn test_iter_forwards_through_both_variants_oldest_first() {
        for b in [uniform(), prioritized()] {
            let obs: Vec<f32> = b.iter().map(|t| t.obs).collect();
            assert_eq!(
                obs,
                vec![0.0, 1.0, 2.0, 3.0],
                "KindIter must yield the live window oldest-first for both variants"
            );
        }
    }

    /// The `Debug` impl must not require `T: Debug`: this transition payload
    /// deliberately does not derive `Debug`, so the test compiling at all proves
    /// the bound is absent.
    #[derive(Clone)]
    struct NoDebug(#[allow(dead_code)] f32);

    #[test]
    fn test_debug_does_not_require_t_debug() {
        let mut b: ReplayKind<NoDebug> = ReplayKind::uniform(4);
        b.push(NoDebug(1.0));
        let rendered = format!("{b:?}");
        assert!(
            rendered.contains("Uniform") && rendered.contains("len"),
            "Debug must summarise the variant and length without dumping items, got: {rendered}"
        );
    }
}
