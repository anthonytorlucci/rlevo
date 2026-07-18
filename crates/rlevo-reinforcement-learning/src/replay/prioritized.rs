//! [`PrioritizedReplay`] — Schaul et al. (2016), proportional variant.

use rand::Rng;
use rlevo_core::config::{ConfigError, Validate};

use super::config::PrioritizedReplayConfig;
use super::error::ReplayBufferError;
use super::importance_exponent::ImportanceExponent;
use super::priority::{Priority, PriorityError};
use super::sum_tree::{PriorityIndex, SumTree, stratified_draw};
use super::{ReplayStrategy, SampledBatch, TransitionId};

/// A replay buffer that samples transitions in proportion to `p_i^α`, per
/// Schaul, Quan, Antonoglou & Silver, *Prioritized Experience Replay*
/// (ICLR 2016, arXiv:1511.05952v4), **proportional variant**.
///
/// Generic over the stored item `T`, so it composes with
/// [`Transition<O, P>`](super::Transition) and its
/// [`DiscreteTransition`](super::DiscreteTransition) /
/// [`ContinuousTransition`](super::ContinuousTransition) aliases without knowing
/// anything about observations, actions, or tensors.
///
/// # Scope: the value-based agents only
///
/// Wire this into DQN, C51, and QR-DQN, where Rainbow's ablation puts
/// prioritized replay among the two most crucial of seven components (full
/// Rainbow beat the no-priority ablation in 53 of 57 games). Do **not** make it
/// the default for DDPG/TD3/SAC: Panahi et al. (RLJ 2024) find that "in control
/// tasks, none of the prioritized variants consistently outperform uniform
/// replay", and Saglam et al. (JAIR 2022) give the mechanism — an actor cannot
/// be effectively trained on high-TD-error transitions, because the policy
/// gradient under a noisy Q diverges from the gradient under the optimal Q.
/// Every positive continuous-control PER result in the literature *modifies*
/// PER first (ADR 0050 §Context).
///
/// # Fidelity to the paper
///
/// Every equation is implemented as written, and every place the paper leaves a
/// choice open is named as a choice.
///
/// | Paper | Here |
/// |---|---|
/// | §3.3 `p_i = \|δ_i\| + ε` | [`Priority::from_td_error`] / [`priority_from_td_error`](Self::priority_from_td_error) |
/// | §3.3 ε "a small positive constant" — **no value given** | [`DEFAULT_PRIORITY_EPSILON`] = `1e-6`, **our** choice, justified there |
/// | Eq. 1 `P(i) = p_i^α / Σ_k p_k^α` | [`sampling_probability`](Self::sampling_probability), via a sum-tree |
/// | Eq. 1 "α = 0 corresponding to the uniform case" | exact: `p^0 == 1` for every stored `p`, which is `> 0` by construction |
/// | Appendix B.2.1 stratified draw | one draw per equal-mass segment — **not** i.i.d. |
/// | Alg. 1 line 6 `p_t = max_{i<t} p_i` | [`max_priority`](Self::max_priority), a running max tracked incrementally |
/// | §3.4 `w_i = (1/N · 1/P(i))^β` | [`sample`](Self::sample) — see *Importance weights* below |
/// | Alg. 1 line 10 max-normalization | over the **sampled minibatch** |
/// | Alg. 1 lines 11-12 priority writeback | [`update_priorities`](Self::update_priorities) |
/// | Appendix B.2.1 sum-tree | O(log N) update and draw |
/// | Table 3 proportional row `α = 0.6`, `β₀ = 0.4 → 1` | α defaults here; β and its schedule live on the agent config (ADR 0050 §11) |
///
/// # Importance weights: which maximum, and why the `1/N` vanishes
///
/// Schaul §3.4 gives `w_i = (1/N · 1/P(i))^β` and then: "For stability reasons,
/// we **always** normalize weights by `1/max_i w_i` so that they only scale the
/// update downwards." Algorithm 1 line 10 takes that maximum **over the sampled
/// minibatch**. That is what this implementation does.
///
/// The alternative — normalizing by a whole-buffer bound derived from the
/// minimum stored priority — is common in reimplementations and is defensible
/// (it makes a transition's weight independent of which other transitions
/// happened to be drawn alongside it), but it is **not Algorithm 1**, and it is
/// not what ships here.
///
/// Appendix B.2.1 warns that "this normalization interacts with annealing on β",
/// so the two are not independent knobs. Accordingly the normalization is folded
/// into the same expression as `w_i`, and there is **no way to obtain
/// unnormalized weights** from this type. Because `w` is monotonically
/// decreasing in `P`, the minibatch maximum is attained at the minimum sampled
/// probability, and the ratio collapses:
///
/// ```text
///     w_i        (1/N · 1/P_i)^β       ( P_min )^β       ( m_min )^β
///  ---------  =  ----------------  =   ( ----- )     =   ( ----- )
///  max_j w_j     (1/N · 1/P_min)^β     (  P_i  )         (  m_i  )
/// ```
///
/// where `m = p^α` is the unnormalized mass. Both `N` and `p_total` cancel
/// exactly. The shipped code evaluates the right-hand form, which is *the same
/// number* as the left with strictly fewer rounding steps and no dependence on
/// `N` — a simplification of the arithmetic, not of the algorithm. One
/// consequence is worth pinning: the largest weight in every batch is exactly
/// `1.0`, bit for bit, because `1.0f64.powf(β) == 1.0`.
///
/// **`w_i` scales the per-sample loss only.** It must never enter the target
/// computation and must never alter `δ` itself (ADR 0050 §10). This type cannot
/// enforce that — it is a property of the agent's loss site — but it is the
/// implementer bug class the literature review names, so it is stated here too.
///
/// # The `NaN` defect this closes
///
/// Priorities are [`Priority`] values: finite and strictly positive **by
/// construction**. The pre-ADR-0050 `memory.rs` stored bare `f32` unvalidated,
/// and a single `NaN` silently pinned its sampler on the oldest transition
/// forever. See the [`priority`](super::priority) module docs for the full
/// chain.
///
/// # Reproducibility
///
/// [`sample`](Self::sample) takes `rng: &mut R` and the buffer owns no RNG,
/// matching every agent's `learn_step<R: Rng + ?Sized>(&mut self, rng: &mut R)`
/// (ADR 0029), so a seeded run is reproducible end to end. The retired type
/// called `rand::rng()` internally and was therefore untestable.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
/// use rlevo_reinforcement_learning::replay::{
///     ImportanceExponent, PrioritizedReplay, PrioritizedReplayConfig, ReplayStrategy,
/// };
///
/// let config = PrioritizedReplayConfig { capacity: 8, ..Default::default() };
/// let mut buffer: PrioritizedReplay<&str> = PrioritizedReplay::new(config)?;
///
/// for label in ["a", "b", "c", "d"] {
///     buffer.push(label); // enters at the running max priority
/// }
///
/// let mut rng = StdRng::seed_from_u64(7);
/// let batch = buffer.sample(2, ImportanceExponent::new(0.4), &mut rng)?;
/// assert_eq!(batch.len(), 2);
///
/// // The largest importance weight in a batch is always exactly 1.0.
/// let weights = batch.weights().expect("prioritized replay emits weights");
/// assert!(weights.iter().any(|&w| w == 1.0));
///
/// // Feed the TD errors back: this is the edge that makes it PER.
/// let ids = batch.ids().to_vec();
/// buffer.update_priorities_from_td_errors(&ids, &[0.5, -2.0])?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// [`DEFAULT_PRIORITY_EPSILON`]: super::DEFAULT_PRIORITY_EPSILON
#[derive(Debug, Clone)]
pub struct PrioritizedReplay<T> {
    /// Ring-buffered items, indexed by slot. Grows to `capacity`, then
    /// overwrites in place — a `VecDeque` with `pop_front` would shift every
    /// slot on eviction and force a full sum-tree rebuild per insert.
    items: Vec<T>,
    /// Raw priorities `p_i`, parallel to `items`. Kept alongside the tree's
    /// `p_i^α` masses so `α` never has to be inverted to recover `p_i`.
    priorities: Vec<Priority>,
    /// `p_i^α` by slot, plus `p_total` and the inverse CDF.
    index: SumTree,
    capacity: usize,
    /// Total inserts ever. The transition in the live window at offset `s` from
    /// the oldest carries absolute id `pushes - len + s` (ADR 0050 §6); this is
    /// what makes an evicted [`TransitionId`] resolve to `None` rather than
    /// alias a live one.
    pushes: u64,
    priority_exponent: f32,
    priority_epsilon: f32,
    /// Schaul Algorithm 1 line 6's `max_{i<t} p_i`: the running maximum over
    /// every priority this buffer has ever held. Never decreases.
    max_priority: Priority,
}

impl<T> PrioritizedReplay<T> {
    /// The priority assigned to the very first insert, before any TD error has
    /// been observed.
    ///
    /// Schaul Algorithm 1 line 6 sets `p_t = max_{i<t} p_i`, which is undefined
    /// at `t = 0` (the maximum of an empty set). `1.0` is the conventional seed
    /// and matches the paper's intent — the first transition must be sampled,
    /// and the running maximum is scale-free, so any positive constant gives
    /// identical behaviour once the first writeback lands.
    const INITIAL_MAX_PRIORITY: f32 = 1.0;

    /// Builds an empty buffer from a validated config.
    ///
    /// # Errors
    ///
    /// Returns the first [`ConfigError`] from
    /// [`PrioritizedReplayConfig::validate`] — a zero `capacity`, a
    /// `priority_exponent` outside `[0, 1]`, or a non-positive
    /// `priority_epsilon`.
    pub fn new(config: PrioritizedReplayConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self {
            items: Vec::with_capacity(config.capacity),
            priorities: Vec::with_capacity(config.capacity),
            index: SumTree::new(config.capacity),
            capacity: config.capacity,
            pushes: 0,
            priority_exponent: config.priority_exponent,
            priority_epsilon: config.priority_epsilon,
            max_priority: Priority::new(Self::INITIAL_MAX_PRIORITY),
        })
    }

    /// The maximum number of transitions held before eviction begins.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Schaul Eq. 1's α.
    #[must_use]
    pub const fn priority_exponent(&self) -> f32 {
        self.priority_exponent
    }

    /// Schaul §3.3's ε.
    #[must_use]
    pub const fn priority_epsilon(&self) -> f32 {
        self.priority_epsilon
    }

    /// Schaul Algorithm 1 line 6's `max_{i<t} p_i` — the priority the next
    /// [`push`](ReplayStrategy::push) will assign.
    ///
    /// This is a running maximum over every priority the buffer has ever held,
    /// **not** a maximum over its current contents: it never decreases, even
    /// when the transition that set it is evicted. That is what guarantees the
    /// paper's stated property, that "all experience is seen at least once" — a
    /// max over live contents would let the bar drift down, and would make a
    /// fresh transition's priority depend on eviction order.
    #[must_use]
    pub const fn max_priority(&self) -> Priority {
        self.max_priority
    }

    /// Total inserts over the buffer's lifetime. The next pushed transition
    /// receives `TransitionId::new(pushes())`.
    #[must_use]
    pub const fn pushes(&self) -> u64 {
        self.pushes
    }

    /// Schaul §3.3's `p_i = |δ_i| + ε`, applying **this buffer's** configured ε.
    ///
    /// Prefer this over [`Priority::from_td_error`] at agent call sites, so
    /// there is one ε rather than two copies that can drift apart.
    ///
    /// # Errors
    ///
    /// Returns [`PriorityError`] when `td_error` is `NaN` or infinite — the
    /// reachable production case being a diverging network. Surfacing it here is
    /// deliberate: it turns silent buffer degeneracy into a reported error at
    /// the writeback site.
    pub fn priority_from_td_error(&self, td_error: f32) -> Result<Priority, PriorityError> {
        Priority::from_td_error(td_error, self.priority_epsilon)
    }

    /// The stored priority `p_i` for `id`, or `None` if it has been evicted.
    #[must_use]
    pub fn priority_of(&self, id: TransitionId) -> Option<Priority> {
        self.slot_of(id).map(|slot| self.priorities[slot])
    }

    /// Schaul Eq. 1's denominator `Σ_k p_k^α` — the sum-tree root, `p_total`.
    #[must_use]
    pub fn total_priority(&self) -> f64 {
        self.index.total()
    }

    /// Schaul Eq. 1's `P(i) = p_i^α / Σ_k p_k^α`, or `None` if `id` has been
    /// evicted.
    ///
    /// Exposed because it is the quantity the paper's correctness is stated in:
    /// it lets a caller — or a test — check realized draw frequencies against
    /// the intended distribution without reaching into the sum-tree.
    #[must_use]
    pub fn sampling_probability(&self, id: TransitionId) -> Option<f64> {
        let slot = self.slot_of(id)?;
        let total = self.index.total();
        (total > 0.0).then(|| self.index.get(slot) / total)
    }

    /// Writes new priorities back for previously sampled ids — Schaul
    /// Algorithm 1 lines 11-12, the feedback edge that makes this PER rather
    /// than a weighted-random buffer. Its absence was the first of the four
    /// defects that made the pre-ADR-0050 type not-PER.
    ///
    /// Ids evicted since they were sampled are **silently dropped** (ADR 0050
    /// §6), which is what makes the absolute-id design pay: with raw slot
    /// indices the same stale writeback would land on whichever transition now
    /// occupies that slot.
    ///
    /// The running maximum of Algorithm 1 line 6 advances only for updates that
    /// were actually applied — a dropped writeback is treated as never having
    /// happened, consistently for both the stored priority and the maximum.
    ///
    /// # An inherent method, not a `ReplayStrategy` one
    ///
    /// ADR 0050 §3 sketches `update_priorities` as a defaulted trait method. The
    /// [`ReplayStrategy`] trait as landed does not carry it, and this type does
    /// not add it there, for two reasons: agents hold a concrete
    /// `PrioritizedReplay` rather than a `dyn ReplayStrategy`, so nothing needs
    /// the dynamic dispatch; and a defaulted no-op on the trait would let a
    /// future weighted strategy silently swallow a writeback it ought to honour.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `ids.len() != priorities.len()`; in release
    /// the shorter slice wins. Both slices come from one [`SampledBatch`] in
    /// every intended call, so a mismatch is a programming error at the agent's
    /// loss site rather than user-supplied data.
    pub fn update_priorities(&mut self, ids: &[TransitionId], priorities: &[Priority]) {
        debug_assert_eq!(
            ids.len(),
            priorities.len(),
            "update_priorities: one priority per id"
        );
        for (&id, &p) in ids.iter().zip(priorities) {
            let Some(slot) = self.slot_of(id) else {
                continue;
            };
            self.priorities[slot] = p;
            let mass = self.mass(p);
            self.index.set(slot, mass);
            self.max_priority = self.max_priority.max(p);
        }
    }

    /// Convenience writeback: applies `p_i = |δ_i| + ε` to each TD error and
    /// forwards to [`update_priorities`](Self::update_priorities).
    ///
    /// This is the call agents make. Every residual is validated **before** any
    /// priority is written, so a single `NaN` anywhere in the batch leaves the
    /// buffer completely unmodified rather than half-updated.
    ///
    /// # Errors
    ///
    /// Returns [`PriorityError`] for the first non-finite `td_error`, having
    /// written nothing.
    ///
    /// # Panics
    ///
    /// Panics when `ids.len() != td_errors.len()`. Both come from the same
    /// [`SampledBatch`] in every intended call.
    pub fn update_priorities_from_td_errors(
        &mut self,
        ids: &[TransitionId],
        td_errors: &[f32],
    ) -> Result<(), PriorityError> {
        assert_eq!(
            ids.len(),
            td_errors.len(),
            "update_priorities_from_td_errors: one TD error per sampled id"
        );
        let priorities = td_errors
            .iter()
            .map(|&d| self.priority_from_td_error(d))
            .collect::<Result<Vec<_>, _>>()?;
        self.update_priorities(ids, &priorities);
        Ok(())
    }

    /// Schaul Eq. 1's `p_i^α`, computed in `f64`.
    ///
    /// `f64` is not for accuracy of the power itself but for the *summation*:
    /// the sum-tree adds these pairwise up the levels while the O(n) reference
    /// oracle adds them left to right, and `f64` accumulation over
    /// `f32`-precision inputs is what makes those two orders agree exactly.
    ///
    /// At `α = 0` this is `1.0` for every stored priority — `p` is strictly
    /// positive by construction, so there is no `0^0` case — which is how
    /// Eq. 1's "α = 0 corresponding to the uniform case" is recovered *exactly*,
    /// not approximately.
    fn mass(&self, p: Priority) -> f64 {
        f64::from(p.get()).powf(f64::from(self.priority_exponent))
    }

    /// `capacity` as a `u64`, for the ring arithmetic.
    // `usize -> u64` is lossless on every target this workspace supports; the
    // lint fires only because `usize` could in principle exceed 64 bits.
    #[allow(clippy::cast_possible_truncation)]
    const fn capacity_u64(&self) -> u64 {
        self.capacity as u64
    }

    /// `min(pushes, capacity)` — the number of live transitions, as a `u64`.
    const fn live_len(&self) -> u64 {
        if self.pushes < self.capacity_u64() {
            self.pushes
        } else {
            self.capacity_u64()
        }
    }

    /// The absolute id of the oldest live transition.
    const fn first_id(&self) -> u64 {
        self.pushes - self.live_len()
    }

    /// Resolves an absolute id to its ring slot, or `None` if it is outside the
    /// live window — evicted, or not yet pushed.
    fn slot_of(&self, id: TransitionId) -> Option<usize> {
        let raw = id.index();
        if raw < self.first_id() || raw >= self.pushes {
            return None;
        }
        // Cannot fail: the remainder is below `capacity`, itself a `usize`.
        usize::try_from(raw % self.capacity_u64()).ok()
    }

    /// The inverse of [`slot_of`](Self::slot_of) for an **occupied** slot.
    ///
    /// The live window is the `live_len` consecutive ids `[first_id, pushes)`,
    /// and `live_len <= capacity`, so at most one id in that window maps to any
    /// given slot: the map is a bijection onto the occupied slots, and this
    /// inverse is well defined.
    fn id_of_slot(&self, slot: usize) -> TransitionId {
        let cap = self.capacity_u64();
        let first = self.first_id();
        // `slot < capacity`, so this conversion cannot fail.
        let slot = u64::try_from(slot).unwrap_or(0);
        let offset = (slot + cap - first % cap) % cap;
        TransitionId::new(first + offset)
    }
}

impl<T> ReplayStrategy<T> for PrioritizedReplay<T> {
    /// Stores `item` at Schaul Algorithm 1 line 6's `p_t = max_{i<t} p_i`, "to
    /// guarantee that all experience is seen at least once".
    ///
    /// The priority is the buffer's own running maximum
    /// ([`max_priority`](PrioritizedReplay::max_priority)) — neither a constant
    /// nor caller-supplied. Making it caller-supplied and unvalidated was one of
    /// the four defects that made the pre-ADR-0050 type not-PER.
    ///
    /// At capacity the oldest transition is evicted in place: its slot is
    /// overwritten, and its [`TransitionId`] stops resolving.
    fn push(&mut self, item: T) {
        let p = self.max_priority;
        let mass = self.mass(p);
        let slot = usize::try_from(self.pushes % self.capacity_u64()).unwrap_or(0);
        if self.items.len() < self.capacity {
            debug_assert_eq!(slot, self.items.len(), "the fill phase writes sequentially");
            self.items.push(item);
            self.priorities.push(p);
        } else {
            self.items[slot] = item;
            self.priorities[slot] = p;
        }
        self.index.set(slot, mass);
        self.pushes += 1;
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn get(&self, id: TransitionId) -> Option<&T> {
        self.slot_of(id).map(|slot| &self.items[slot])
    }

    fn get_mut(&mut self, id: TransitionId) -> Option<&mut T> {
        let slot = self.slot_of(id)?;
        Some(&mut self.items[slot])
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        // FIFO over the live window, oldest first — which is *not* slot order
        // once the ring has wrapped.
        let first = self.first_id();
        let cap = self.capacity_u64();
        (0..self.live_len()).map(move |offset| {
            let slot = usize::try_from((first + offset) % cap).unwrap_or(0);
            &self.items[slot]
        })
    }

    /// Draws `batch_size` ids by Schaul's **stratified** scheme (Appendix
    /// B.2.1) and returns them with max-normalized importance weights.
    ///
    /// The range `[0, p_total]` is split into `batch_size` equal ranges and one
    /// value is drawn uniformly from each, so exactly one `rng` call is issued
    /// per segment, in segment order. This is deliberately **not** `batch_size`
    /// i.i.d. categorical draws — see the [type documentation](Self) for the
    /// weight derivation and for which maximum the normalization uses.
    ///
    /// # β
    ///
    /// `beta` is the *already-evaluated* importance exponent for this step; the
    /// annealing schedule lives on the agent config (ADR 0050 §11). Its
    /// `finite && [0, 1]` invariant is **carried by the
    /// [`ImportanceExponent`] type**, so this method neither re-checks nor can
    /// be handed a value that would make `powf` produce `NaN`.
    ///
    /// That is a correction to ADR 0050 §11, recorded in ADR 0051 §3: the
    /// caller's schedule cannot be the validation site, because a config holds
    /// schedule *endpoints* while what reaches `powf` is the *evaluated*
    /// interpolation — which is `NaN` for a zero-length anneal even when every
    /// endpoint validates.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayBufferError::InsufficientData`] when the buffer holds
    /// fewer than `batch_size` transitions — matching
    /// [`UniformReplay`](super::UniformReplay), so the two strategies are
    /// interchangeable at an agent's call site.
    fn sample<R: Rng + ?Sized>(
        &self,
        batch_size: usize,
        beta: ImportanceExponent,
        rng: &mut R,
    ) -> Result<SampledBatch, ReplayBufferError> {
        if batch_size > self.items.len() {
            return Err(ReplayBufferError::InsufficientData {
                requested: batch_size,
                available: self.items.len(),
            });
        }
        if batch_size == 0 {
            return Ok(SampledBatch::weighted(Vec::new(), Vec::new()));
        }

        let slots = stratified_draw(&self.index, batch_size, rng);

        // Every stored priority is finite and `> 0`, and `stratified_draw`
        // cannot return an unoccupied slot, so every drawn mass is `> 0` and the
        // minimum is a safe divisor.
        let masses: Vec<f64> = slots.iter().map(|&s| self.index.get(s)).collect();
        let min_mass = masses
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .max(f64::MIN_POSITIVE);

        let beta = f64::from(beta.get());
        let weights = masses
            .iter()
            .map(|&m| {
                // `(m_min / m_i)^beta` — Algorithm 1 line 10 with `1/N` and
                // `p_total` cancelled analytically. The ratio lies in `(0, 1]`,
                // so the result does too and the `f64 -> f32` narrowing cannot
                // overflow; the largest weight is exactly `1.0`.
                #[allow(clippy::cast_possible_truncation)]
                let w = (min_mass / m).powf(beta) as f32;
                w
            })
            .collect();

        let ids = slots.into_iter().map(|s| self.id_of_slot(s)).collect();
        Ok(SampledBatch::weighted(ids, weights))
    }
}

#[cfg(test)]
mod tests {
    use super::{PrioritizedReplay, PrioritizedReplayConfig};
    use crate::replay::priority::Priority;
    use crate::replay::sum_tree::reference::PrefixScanIndex;
    use crate::replay::sum_tree::{PriorityIndex, stratified_draw};
    use crate::replay::{ImportanceExponent, ReplayBufferError, ReplayStrategy, TransitionId};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// An empty buffer of `usize` labels with `capacity` slots and exponent
    /// `alpha`.
    fn buffer(capacity: usize, alpha: f32) -> PrioritizedReplay<usize> {
        PrioritizedReplay::new(PrioritizedReplayConfig {
            capacity,
            priority_exponent: alpha,
            ..PrioritizedReplayConfig::default()
        })
        .expect("test config is valid")
    }

    /// A buffer holding `0..ps.len()`, where the transition with absolute id `i`
    /// carries priority `ps[i]`.
    fn filled(capacity: usize, alpha: f32, ps: &[f32]) -> PrioritizedReplay<usize> {
        let mut b = buffer(capacity, alpha);
        for i in 0..ps.len() {
            b.push(i);
        }
        let ids: Vec<TransitionId> = (0..ps.len()).map(|i| TransitionId::new(i as u64)).collect();
        let priorities: Vec<Priority> = ps.iter().map(|&p| Priority::new(p)).collect();
        b.update_priorities(&ids, &priorities);
        b
    }

    // ---- construction ----------------------------------------------------

    #[test]
    fn test_prioritized_replay_new_rejects_invalid_config() {
        let bad = PrioritizedReplayConfig {
            capacity: 0,
            ..PrioritizedReplayConfig::default()
        };
        assert!(
            PrioritizedReplay::<usize>::new(bad).is_err(),
            "construction must propagate ConfigError rather than panic"
        );
    }

    #[test]
    fn test_prioritized_replay_starts_empty() {
        let b = buffer(4, 0.6);
        assert!(b.is_empty(), "a fresh buffer holds nothing");
        assert_eq!(b.len(), 0, "len must agree with is_empty");
        assert_eq!(b.pushes(), 0, "no inserts have happened");
        assert_eq!(b.capacity(), 4, "capacity is taken from the config");
    }

    // ---- Algorithm 1 line 6: new transitions enter at the running max ----

    #[test]
    fn test_push_assigns_the_running_max_priority() {
        let mut b = buffer(8, 1.0);
        b.push(0);
        assert_eq!(
            b.priority_of(TransitionId::new(0)).map(Priority::get),
            Some(1.0),
            "the first insert seeds the running max at 1.0"
        );

        b.update_priorities(&[TransitionId::new(0)], &[Priority::new(7.5)]);
        assert_eq!(b.max_priority().get(), 7.5, "the running max must advance");
        b.push(1);
        assert_eq!(
            b.priority_of(TransitionId::new(1)).map(Priority::get),
            Some(7.5),
            "a new transition enters at max_{{i<t}} p_i, not at a constant"
        );
    }

    /// The maximum is over priorities *seen so far*, not over live contents: a
    /// writeback that lowers a stored priority must not lower the bar for the
    /// next insert.
    #[test]
    fn test_running_max_never_decreases() {
        let mut b = buffer(4, 1.0);
        b.push(0);
        b.update_priorities(&[TransitionId::new(0)], &[Priority::new(9.0)]);
        b.update_priorities(&[TransitionId::new(0)], &[Priority::new(0.01)]);
        assert_eq!(
            b.max_priority().get(),
            9.0,
            "the running max is monotone; lowering a stored priority must not lower it"
        );
        b.push(1);
        assert_eq!(
            b.priority_of(TransitionId::new(1)).map(Priority::get),
            Some(9.0),
            "the next insert still enters at the historical maximum"
        );
    }

    /// The maximum must survive eviction of the transition that set it —
    /// otherwise a fresh transition's priority would depend on eviction order.
    #[test]
    fn test_running_max_survives_eviction_of_its_source() {
        let mut b = buffer(2, 1.0);
        b.push(0);
        b.update_priorities(&[TransitionId::new(0)], &[Priority::new(50.0)]);
        b.push(1);
        b.push(2); // evicts id 0
        assert!(
            b.get(TransitionId::new(0)).is_none(),
            "id 0 must be evicted by this point"
        );
        assert_eq!(
            b.max_priority().get(),
            50.0,
            "the running max is over history, not over live contents"
        );
    }

    // ---- eviction and TransitionId ---------------------------------------

    #[test]
    fn test_evicted_transition_id_resolves_to_none() {
        let mut b = buffer(3, 0.6);
        for i in 0..5 {
            b.push(i);
        }
        assert_eq!(b.len(), 3, "the buffer is capped at its capacity");
        for evicted in [0u64, 1] {
            let id = TransitionId::new(evicted);
            assert!(
                b.get(id).is_none(),
                "evicted id {evicted} must resolve to None, not alias a live slot"
            );
            assert!(b.priority_of(id).is_none(), "an evicted id has no priority");
            assert!(
                b.sampling_probability(id).is_none(),
                "an evicted id has no sampling probability"
            );
        }
        for live in [2usize, 3, 4] {
            assert_eq!(
                b.get(TransitionId::new(live as u64)).copied(),
                Some(live),
                "live id {live} must resolve to the item pushed under it"
            );
        }
        assert!(
            b.get(TransitionId::new(5)).is_none(),
            "an id that has not been pushed yet must resolve to None"
        );
    }

    #[test]
    fn test_update_priorities_drops_evicted_ids_silently() {
        let mut b = filled(3, 1.0, &[1.0, 1.0, 1.0]);
        for i in 3..6 {
            b.push(i);
        }
        let before = b.total_priority();
        b.update_priorities(&[TransitionId::new(0)], &[Priority::new(100.0)]);
        assert!(
            (b.total_priority() - before).abs() < 1e-12,
            "a writeback for an evicted id must change nothing"
        );
        assert_eq!(
            b.max_priority().get(),
            1.0,
            "a dropped writeback must not advance the running max either"
        );
    }

    #[test]
    fn test_get_mut_reaches_the_stored_item() {
        let mut b = buffer(4, 0.6);
        b.push(10);
        let id = TransitionId::new(0);
        *b.get_mut(id).expect("id 0 is live") = 99;
        assert_eq!(b.get(id).copied(), Some(99), "get_mut must alias storage");
    }

    #[test]
    fn test_iter_yields_the_live_window_oldest_first_after_wrap() {
        let mut b = buffer(3, 0.6);
        for i in 0..5 {
            b.push(i);
        }
        let seen: Vec<usize> = b.iter().copied().collect();
        assert_eq!(
            seen,
            vec![2, 3, 4],
            "iter must be FIFO over the live window, not raw slot order"
        );
    }

    // ---- Eq. 1: P(i) = p^alpha / sum p^alpha -----------------------------

    #[test]
    fn test_sampling_probability_matches_equation_1() {
        let b = filled(4, 1.0, &[1.0, 2.0, 3.0, 4.0]);
        for (i, expected) in [0.1, 0.2, 0.3, 0.4].into_iter().enumerate() {
            let got = b
                .sampling_probability(TransitionId::new(i as u64))
                .expect("live id");
            assert!(
                (got - expected).abs() < 1e-12,
                "P({i}) must be p_i / sum p_k = {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_sampling_probability_applies_the_exponent() {
        // p^0.5 = [1, 2], total 3.
        let b = filled(2, 0.5, &[1.0, 4.0]);
        let p0 = b
            .sampling_probability(TransitionId::new(0))
            .expect("live id");
        assert!(
            (p0 - 1.0 / 3.0).abs() < 1e-12,
            "alpha must be applied before normalization, got {p0}"
        );
    }

    /// Schaul Eq. 1: "α = 0 corresponding to the uniform case". This must be
    /// **exact**, not approximate, across a spread of priority magnitudes.
    #[test]
    fn test_alpha_zero_reduces_exactly_to_uniform() {
        let b = filled(5, 0.0, &[1e-6, 0.5, 7.0, 1234.0, 1e6]);
        for i in 0..5u64 {
            let p = b
                .sampling_probability(TransitionId::new(i))
                .expect("live id");
            assert!(
                (p - 0.2).abs() < 1e-15,
                "at alpha = 0 every transition must carry probability 1/N exactly, \
                 got {p} for id {i}"
            );
        }
        assert!(
            (b.total_priority() - 5.0).abs() < 1e-15,
            "at alpha = 0 the total mass is exactly N"
        );
    }

    // ---- Appendix B.2.1: stratified sampling -----------------------------

    /// With `k` equal-priority transitions and `k` draws, each equal-mass
    /// segment coincides with exactly one transition, so the batch is forced to
    /// contain every transition exactly once — for **every** seed. An i.i.d.
    /// categorical sampler produces that with probability `k!/k^k` (≈1.5% at
    /// `k = 6`), so this test separates the two algorithms outright.
    #[test]
    fn test_sampling_is_stratified_not_iid() {
        let b = filled(6, 0.6, &[1.0; 6]);
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let batch = b
                .sample(6, ImportanceExponent::new(0.4), &mut rng)
                .expect("full buffer");
            let mut ids: Vec<u64> = batch.ids().iter().map(|id| id.index()).collect();
            ids.sort_unstable();
            assert_eq!(
                ids,
                vec![0, 1, 2, 3, 4, 5],
                "one draw per equal-mass segment must cover the buffer exactly (seed {seed})"
            );
        }
    }

    /// Stratification confines each draw to its own segment. With priorities
    /// `[1, 2, 3, 4]` (total 10) and `k = 2`, segment 0 is `[0, 5)` and segment
    /// 1 is `[5, 10)`, while the transitions own `[0,1) [1,3) [3,6) [6,10)`. So
    /// draw 0 can only be id 0, 1 or 2, and draw 1 only id 2 or 3 — for every
    /// seed. Hand-derivable from the paper; no pinned constants needed.
    #[test]
    fn test_stratification_confines_each_draw_to_its_own_segment() {
        let b = filled(4, 1.0, &[1.0, 2.0, 3.0, 4.0]);
        for seed in 0..200u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let batch = b
                .sample(2, ImportanceExponent::new(0.4), &mut rng)
                .expect("full buffer");
            let ids: Vec<u64> = batch.ids().iter().map(|id| id.index()).collect();
            assert!(
                (0..=2).contains(&ids[0]),
                "segment [0, 5) can only reach ids 0..=2, got {} (seed {seed})",
                ids[0]
            );
            assert!(
                (2..=3).contains(&ids[1]),
                "segment [5, 10) can only reach ids 2..=3, got {} (seed {seed})",
                ids[1]
            );
        }
    }

    /// The buffer's shipped sum-tree path must agree, draw for draw, with the
    /// O(n) prefix-scan oracle over the same masses and the same seed. If the
    /// tree's descent ever gains an off-by-one this fails loudly, rather than
    /// the sampler biasing silently.
    #[test]
    fn test_draws_match_the_on_reference_implementation() {
        let ps = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let alpha = 0.6_f32;
        let b = filled(8, alpha, &ps);

        let mut oracle = PrefixScanIndex::new(ps.len());
        for (slot, &p) in ps.iter().enumerate() {
            oracle.set(slot, f64::from(p).powf(f64::from(alpha)));
        }

        for seed in 0..100u64 {
            for batch_size in [1usize, 2, 4, 8] {
                let mut rng_buf = StdRng::seed_from_u64(seed);
                let mut rng_ref = StdRng::seed_from_u64(seed);
                let from_buffer: Vec<u64> = b
                    .sample(batch_size, ImportanceExponent::new(0.4), &mut rng_buf)
                    .expect("full buffer")
                    .ids()
                    .iter()
                    .map(|id| id.index())
                    .collect();
                let from_oracle: Vec<u64> = stratified_draw(&oracle, batch_size, &mut rng_ref)
                    .into_iter()
                    .map(|slot| slot as u64)
                    .collect();
                assert_eq!(
                    from_buffer, from_oracle,
                    "sum-tree draw diverged from the O(n) reference \
                     (seed {seed}, batch {batch_size})"
                );
            }
        }
    }

    /// Seeded reproducibility: the buffer owns no RNG, so the same seed must
    /// reproduce the same batch. This is the property the retired type's
    /// internal `rand::rng()` call made untestable.
    #[test]
    fn test_sample_is_reproducible_from_a_seed() {
        let b = filled(6, 0.6, &[1.0, 5.0, 2.0, 9.0, 0.5, 3.0]);
        let mut a = StdRng::seed_from_u64(1234);
        let mut c = StdRng::seed_from_u64(1234);
        let first = b
            .sample(4, ImportanceExponent::new(0.7), &mut a)
            .expect("full buffer");
        let second = b
            .sample(4, ImportanceExponent::new(0.7), &mut c)
            .expect("full buffer");
        assert_eq!(
            first, second,
            "identical seeds must yield identical batches"
        );
    }

    // ---- Algorithm 1 line 10: max-normalized IS weights ------------------

    #[test]
    fn test_weights_are_max_normalized_over_the_sampled_minibatch() {
        let b = filled(8, 0.6, &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]);
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let batch = b
                .sample(4, ImportanceExponent::new(1.0), &mut rng)
                .expect("full buffer");
            let w = batch.weights().expect("prioritized replay emits weights");
            let max = w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (max - 1.0).abs() < f32::EPSILON,
                "the largest weight in the minibatch must be exactly 1 (seed {seed}), got {max}"
            );
            assert!(
                w.iter().all(|&x| x > 0.0 && x <= 1.0),
                "weights only scale the update downwards (seed {seed}): {w:?}"
            );
        }
    }

    /// The normalization is over the sampled minibatch, not the whole buffer. A
    /// single-sample minibatch normalizes to its own only weight, even though
    /// the buffer holds a transition 1000x heavier that was not drawn.
    #[test]
    fn test_weight_normalization_is_minibatch_scoped_not_buffer_scoped() {
        let b = filled(5, 1.0, &[1.0, 1.0, 1.0, 1.0, 1000.0]);
        let mut rng = StdRng::seed_from_u64(3);
        let batch = b
            .sample(1, ImportanceExponent::new(1.0), &mut rng)
            .expect("full buffer");
        assert_eq!(
            batch.weights().expect("weights"),
            [1.0],
            "a one-sample minibatch normalizes to its own maximum, i.e. exactly 1"
        );
    }

    #[test]
    fn test_beta_zero_yields_unit_weights() {
        let b = filled(8, 0.6, &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]);
        let mut rng = StdRng::seed_from_u64(11);
        let batch = b
            .sample(4, ImportanceExponent::new(0.0), &mut rng)
            .expect("full buffer");
        let w = batch.weights().expect("weights");
        assert!(
            w.iter().all(|&x| (x - 1.0).abs() < f32::EPSILON),
            "beta = 0 disables the importance correction entirely: {w:?}"
        );
    }

    /// Equal priorities carry no bias to correct, so every weight is 1
    /// regardless of β.
    #[test]
    fn test_uniform_priorities_yield_unit_weights_for_any_beta() {
        let b = filled(4, 0.6, &[3.0; 4]);
        for beta in [0.0_f32, 0.4, 0.75, 1.0] {
            let mut rng = StdRng::seed_from_u64(5);
            let batch = b
                .sample(4, ImportanceExponent::new(beta), &mut rng)
                .expect("full buffer");
            let w = batch.weights().expect("weights");
            assert!(
                w.iter().all(|&x| (x - 1.0).abs() < f32::EPSILON),
                "equal priorities imply no importance correction at beta = {beta}: {w:?}"
            );
        }
    }

    /// The closed form, checked against independently recomputed masses:
    /// `w_i = (p_min / p_i)^beta` at `alpha = 1`.
    #[test]
    fn test_weights_match_the_closed_form() {
        let b = filled(2, 1.0, &[1.0, 4.0]);
        let beta = ImportanceExponent::new(0.5);
        let mut rng = StdRng::seed_from_u64(0);
        let batch = b.sample(2, beta, &mut rng).expect("full buffer");
        let ids: Vec<usize> = batch
            .ids()
            .iter()
            .map(|id| usize::try_from(id.index()).expect("small id"))
            .collect();
        let w = batch.weights().expect("weights");
        let masses = [1.0_f64, 4.0];
        let min = ids.iter().map(|&i| masses[i]).fold(f64::INFINITY, f64::min);
        for (k, &id) in ids.iter().enumerate() {
            let expected = (min / masses[id]).powf(f64::from(beta.get()));
            assert!(
                (f64::from(w[k]) - expected).abs() < 1e-6,
                "w[{k}] must equal (p_min / p_i)^beta = {expected}, got {}",
                w[k]
            );
        }
    }

    // ---- error paths -----------------------------------------------------

    #[test]
    fn test_sample_reports_insufficient_data() {
        let mut b = buffer(8, 0.6);
        b.push(0);
        b.push(1);
        let mut rng = StdRng::seed_from_u64(0);
        let err = b
            .sample(4, ImportanceExponent::new(0.4), &mut rng)
            .expect_err("only two transitions");
        assert!(
            matches!(
                err,
                ReplayBufferError::InsufficientData {
                    requested: 4,
                    available: 2
                }
            ),
            "an under-filled buffer must report the shortfall, got {err:?}"
        );
    }

    /// The end-to-end form of the regression: a `NaN` TD error off a diverging
    /// network must surface as an error and leave the buffer untouched, rather
    /// than silently pinning the sampler at slot 0 forever.
    #[test]
    fn test_nan_td_error_is_rejected_and_leaves_the_buffer_unmodified() {
        let mut b = filled(4, 0.6, &[1.0, 2.0, 3.0, 4.0]);
        let before = b.total_priority();
        let ids: Vec<TransitionId> = (0..4).map(TransitionId::new).collect();

        let err = b
            .update_priorities_from_td_errors(&ids, &[0.5, f32::NAN, 0.25, 0.75])
            .expect_err("a NaN residual must be rejected");
        assert!(err.got.is_nan(), "the error must carry the offending value");
        assert!(
            (b.total_priority() - before).abs() < 1e-12,
            "validation happens before any write: the buffer must be untouched"
        );

        // And the sampler still works — no NaN reached the tree.
        let mut rng = StdRng::seed_from_u64(0);
        let batch = b
            .sample(4, ImportanceExponent::new(0.4), &mut rng)
            .expect("buffer is still healthy");
        assert!(
            batch
                .weights()
                .expect("weights")
                .iter()
                .all(|w| w.is_finite()),
            "no NaN may have reached the sampling distribution"
        );
    }

    #[test]
    fn test_update_priorities_from_td_errors_applies_the_epsilon_floor() {
        let mut b = filled(2, 1.0, &[5.0, 5.0]);
        let ids: Vec<TransitionId> = (0..2).map(TransitionId::new).collect();
        b.update_priorities_from_td_errors(&ids, &[0.0, 1.0])
            .expect("finite residuals");
        let p0 = b.priority_of(ids[0]).expect("live").get();
        assert!(
            (p0 - b.priority_epsilon()).abs() < 1e-12,
            "a zero residual must floor at epsilon, got {p0}"
        );
        assert!(
            b.sampling_probability(ids[0]).expect("live") > 0.0,
            "a converged transition must retain a strictly positive draw probability"
        );
    }

    // ---- the feedback edge -----------------------------------------------

    /// `update_priorities` is the edge the retired type lacked entirely. Raising
    /// one transition's priority must actually shift the draw distribution.
    #[test]
    fn test_priority_writeback_shifts_the_sampling_distribution() {
        let mut b = filled(4, 1.0, &[1.0; 4]);
        let target = TransitionId::new(2);
        let before = b.sampling_probability(target).expect("live");
        b.update_priorities(&[target], &[Priority::new(97.0)]);
        let after = b.sampling_probability(target).expect("live");
        assert!(
            (before - 0.25).abs() < 1e-12,
            "equal priorities start uniform, got {before}"
        );
        assert!(
            after > 0.9,
            "raising p_i must raise P(i): {before} -> {after}"
        );

        let mut rng = StdRng::seed_from_u64(2);
        let hits = (0..200)
            .map(|_| {
                b.sample(1, ImportanceExponent::new(0.4), &mut rng)
                    .expect("full buffer")
            })
            .filter(|batch| batch.ids()[0] == target)
            .count();
        assert!(
            hits > 180,
            "the raised transition must dominate the draws, got {hits}/200"
        );
    }

    /// Realized draw frequencies must track Eq. 1's `P(i)`. Stratification makes
    /// this tight: over `n` batches of size `k`, transition `i` is drawn about
    /// `n * k * P(i)` times, with far less spread than i.i.d. sampling gives.
    #[test]
    fn test_draw_frequencies_track_equation_1() {
        let b = filled(4, 1.0, &[1.0, 2.0, 3.0, 4.0]);
        let mut rng = StdRng::seed_from_u64(99);
        let (batches, batch_size) = (2_000usize, 4usize);
        let mut counts = [0usize; 4];
        for _ in 0..batches {
            let batch = b
                .sample(batch_size, ImportanceExponent::new(0.4), &mut rng)
                .expect("full buffer");
            for id in batch.ids() {
                counts[usize::try_from(id.index()).expect("small id")] += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)] // exact for these magnitudes
        let draws = (batches * batch_size) as f64;
        for (i, expected) in [0.1, 0.2, 0.3, 0.4].into_iter().enumerate() {
            #[allow(clippy::cast_precision_loss)] // exact for these magnitudes
            let observed = counts[i] as f64 / draws;
            assert!(
                (observed - expected).abs() < 0.02,
                "empirical P({i}) = {observed} must track Eq. 1's {expected}"
            );
        }
    }
}
