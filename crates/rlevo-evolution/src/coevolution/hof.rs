//! Hall-of-fame pathology mitigation for competitive co-evolution.
//!
//! Naive competitive co-evolution suffers from **cycling** (Ficici 2004): a
//! population evolves a best response to the opponent's current composition,
//! the opponent shifts in turn, and neither makes lasting progress (the
//! rock-paper-scissors trap). Rosin & Belew (1997) mitigate this with a
//! *hall of fame* — an archive of past champions that the current population
//! must also perform well against, anchoring the fitness landscape so it can
//! no longer be chased in a circle.
//!
//! [`HallOfFame`] is the archive; [`HallOfFameFitness`] is the
//! [`CoupledFitness`] wrapper that blends each individual's score against the
//! current opponents with its score against the archived champions. Passing a
//! `HallOfFameFitness` to a co-evolutionary algorithm enables the mitigation;
//! passing the raw fitness disables it — no flag inside the algorithm.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use parking_lot::Mutex;
use rlevo_core::objective::ObjectiveSense;

use super::fitness::CoupledFitness;
use crate::fitness::sanitize_fitness;

/// Per-population archive of past champions, capped at a fixed capacity.
///
/// Each [`update`](Self::update) appends the current generation's best
/// individual (highest fitness, canonical maximise convention) to each
/// population's archive. When an archive would exceed `capacity` the single
/// worst-fitness (lowest) member is dropped, so the archive always retains the
/// `capacity` best champions seen across the whole run.
///
/// The capacity is computed by [`capacity_for`](Self::capacity_for) as
/// `max(10, pop_size / 5)` (Rosin & Belew sizing).
///
/// # Invariants
///
/// - All archives share a single `genome_dim`; v1 co-evolution is
///   bi-population over equal-width genomes. Asymmetric genome widths are out
///   of scope.
/// - `archives()[p].dims()[0] <= capacity` after every `update`.
#[derive(Debug, Clone)]
pub struct HallOfFame<B: Backend> {
    /// Top-k champions retained per population, each `(size_p, genome_dim)`.
    archives: Vec<Tensor<B, 2>>,
    /// Host-side fitness of each archived champion (as inserted), parallel to
    /// `archives`. Used to prune the worst member on overflow.
    archive_fitness: Vec<Vec<f32>>,
    /// Maximum number of champions retained per population.
    capacity: usize,
}

impl<B: Backend> HallOfFame<B> {
    /// Build an empty hall of fame for `num_populations` populations.
    ///
    /// Each archive starts as a `(0, genome_dim)` tensor and grows by one row
    /// per [`update`](Self::update) until `capacity` is reached.
    #[must_use]
    pub fn new(
        num_populations: usize,
        capacity: usize,
        genome_dim: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        let archives = (0..num_populations)
            .map(|_| Tensor::<B, 2>::empty([0, genome_dim], device))
            .collect();
        let archive_fitness = vec![Vec::new(); num_populations];
        Self {
            archives,
            archive_fitness,
            capacity,
        }
    }

    /// Recommended capacity for a population of `pop_size`: `max(10, pop_size / 5)`.
    #[must_use]
    pub fn capacity_for(pop_size: usize) -> usize {
        (pop_size / 5).max(10)
    }

    /// Insert the best individual of each population into its archive.
    ///
    /// For population `p`, the highest-fitness row of `populations[p]` is
    /// appended to `archives[p]` (canonical maximise: higher is better). If that
    /// pushes the archive past `capacity`, the single lowest-fitness (worst)
    /// archived member is removed. Empty populations are skipped.
    ///
    /// This method is **sense-blind**: it argmaxes highest = best and evicts
    /// lowest = worst, which is correct only in canonical (maximise) space. The
    /// **caller is responsible for passing canonical fitness** — for a
    /// `Minimize` objective the natural cost must be negated first, or the
    /// highest-*cost* (worst) individual would be crowned champion.
    ///
    /// # Panics
    ///
    /// Panics if a population's fitness tensor cannot be read back to host as
    /// `f32` (a device→host transfer failure). A legitimately empty population
    /// is a valid non-error host-read and is skipped, not a panic.
    pub fn update(&mut self, populations: &[Tensor<B, 2>], fitnesses: &[Tensor<B, 1>]) {
        let n = self
            .archives
            .len()
            .min(populations.len())
            .min(fitnesses.len());
        for p in 0..n {
            let fit_host = fitnesses[p]
                .clone()
                .into_data()
                .into_vec::<f32>()
                .expect("fitness tensor must be readable as f32");
            if fit_host.is_empty() {
                continue;
            }
            // Sanitize NaN → −inf (worst) so a NaN-fitness member can never be
            // crowned champion over a finite one; this also keeps `archive_fitness`
            // NaN-free, which the eviction `min_by` below relies on.
            let sane: Vec<f32> = fit_host.iter().map(|&f| sanitize_fitness(f)).collect();
            // Argmax (best, highest fitness — canonical maximise) — ties
            // resolve to the lowest index. Hand-rolled with a strict
            // `total_cmp == Greater` so equal-fitness ties keep the earliest
            // index (`Iterator::max_by` would instead keep the last).
            let mut best_idx = 0_usize;
            for i in 1..sane.len() {
                if sane[i].total_cmp(&sane[best_idx]) == std::cmp::Ordering::Greater {
                    best_idx = i;
                }
            }
            let best_f = sane[best_idx];
            let device = populations[p].device();
            // usize → i64 index tensor; population indices never approach i64::MAX.
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            let champion = populations[p].clone().select(0, idx);

            self.archives[p] = if self.archives[p].dims()[0] == 0 {
                champion
            } else {
                Tensor::cat(vec![self.archives[p].clone(), champion], 0)
            };
            self.archive_fitness[p].push(best_f);

            if self.archive_fitness[p].len() > self.capacity {
                // Worst = lowest fitness under the maximise convention.
                // `archive_fitness` is sanitised at push (no NaN), so a plain
                // `total_cmp` correctly evicts the worst here.
                // Over capacity => non-empty, so `min_by` is always `Some`.
                let Some(worst_idx) = self.archive_fitness[p]
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i)
                else {
                    continue;
                };
                let len = self.archive_fitness[p].len();
                #[allow(clippy::cast_possible_wrap)]
                let keep: Vec<i64> = (0..len)
                    .filter(|&i| i != worst_idx)
                    .map(|i| i as i64)
                    .collect();
                let keep_len = keep.len();
                let keep_idx =
                    Tensor::<B, 1, Int>::from_data(TensorData::new(keep, [keep_len]), &device);
                self.archives[p] = self.archives[p].clone().select(0, keep_idx);
                self.archive_fitness[p].remove(worst_idx);
            }
        }
    }

    /// Borrow the per-population champion archives.
    #[must_use]
    pub fn archives(&self) -> &[Tensor<B, 2>] {
        &self.archives
    }

    /// The per-population capacity cap.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// A [`CoupledFitness`] wrapper that anchors fitness against a hall of fame.
///
/// Wraps any concrete `F: CoupledFitness<B>` and, each evaluation, blends an
/// individual's score against the *current* opponents with its score against
/// the *archived* champions:
///
/// ```text
/// fitness_blended = (1 - w) * fitness_current + w * fitness_hof
/// ```
///
/// where `w` is [`hof_blend_weight`](Self::with_blend_weight) (default `0.3`).
/// Setting `w = 0.0` disables the mitigation without removing the wrapper;
/// the archive is still maintained so the wrapper can be re-enabled or its
/// archive inspected. The archive is updated after each evaluation with the
/// current generation's champions.
///
/// Because [`CoupledFitness::evaluate_coupled`] takes `&self` but the archive
/// must mutate per generation, the [`HallOfFame`] is held behind a
/// `parking_lot::Mutex` (the project-standard lock, ADR-0010). Each harness
/// runs its own wrapper instance, so the lock is effectively uncontended.
///
/// # Invariants
///
/// - v1 is bi-population: `evaluate_coupled` debug-asserts
///   `populations.len() == 2`.
pub struct HallOfFameFitness<B: Backend, F: CoupledFitness<B>> {
    inner: F,
    hall: Mutex<HallOfFame<B>>,
    hof_blend_weight: f32,
}

impl<B: Backend, F: CoupledFitness<B>> std::fmt::Debug for HallOfFameFitness<B, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HallOfFameFitness")
            .field("hof_blend_weight", &self.hof_blend_weight)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, F: CoupledFitness<B>> HallOfFameFitness<B, F> {
    /// Default blend weight (`0.3`).
    pub const DEFAULT_BLEND_WEIGHT: f32 = 0.3;

    /// Wrap `inner` with a hall of fame sized `max(10, pop_size / 5)`.
    ///
    /// `num_populations` and `genome_dim` size the archives; the blend weight
    /// starts at [`DEFAULT_BLEND_WEIGHT`](Self::DEFAULT_BLEND_WEIGHT). Use
    /// [`with_blend_weight`](Self::with_blend_weight) to override it.
    #[must_use]
    pub fn new(
        inner: F,
        num_populations: usize,
        pop_size: usize,
        genome_dim: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        let capacity = HallOfFame::<B>::capacity_for(pop_size);
        let hall = HallOfFame::new(num_populations, capacity, genome_dim, device);
        Self {
            inner,
            hall: Mutex::new(hall),
            hof_blend_weight: Self::DEFAULT_BLEND_WEIGHT,
        }
    }

    /// Override the hall-of-fame blend weight (clamped to `[0.0, 1.0]`).
    ///
    /// `0.0` disables the mitigation (pure current-generation fitness); `1.0`
    /// evaluates purely against the archive.
    #[must_use]
    pub fn with_blend_weight(mut self, weight: f32) -> Self {
        self.hof_blend_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// The current blend weight.
    #[must_use]
    pub fn blend_weight(&self) -> f32 {
        self.hof_blend_weight
    }
}

/// `cur * (1 - w) + hof * w`, element-wise.
fn blend<B: Backend>(cur: &Tensor<B, 1>, hof: &Tensor<B, 1>, w: f32) -> Tensor<B, 1> {
    cur.clone()
        .mul_scalar(1.0 - w)
        .add(hof.clone().mul_scalar(w))
}

impl<B: Backend, F: CoupledFitness<B>> CoupledFitness<B> for HallOfFameFitness<B, F> {
    /// Blend the inner fitness against the hall-of-fame archive, in NATURAL
    /// space.
    ///
    /// The returned `blended` is left in the inner objective's **natural** sense
    /// — the co-evolutionary algorithm canonicalises it, exactly as for the raw
    /// inner fitness. This is correct because the blend is affine and
    /// `to_canonical` is negation: `neg((1−w)·cur + w·res) == (1−w)·neg(cur) +
    /// w·neg(res)`, so canonicalising the blend equals blending the
    /// canonicalised terms. The internal archive champion-selection is a
    /// *separate* concern and **is** canonicalised here (see `current_canon`),
    /// because [`HallOfFame::update`] argmaxes highest = best in maximise space.
    ///
    /// This method is logically **serial per instance**: the archive snapshot
    /// and the later `update` are two separate lock acquisitions, so a single
    /// instance's generations must not be evaluated concurrently (each harness
    /// owns its own wrapper instance, so this holds by construction).
    fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>> {
        debug_assert_eq!(populations.len(), 2, "v1 hall-of-fame is bi-population");
        let sense = self.inner.sense();
        let current = self.inner.evaluate_coupled(populations); // natural
        let w = self.hof_blend_weight;

        // Canonicalise the current-gen fitness for archive champion-selection
        // and eviction (both run in maximise-native space in `HallOfFame::update`).
        // Done UNCONDITIONALLY — even at w=0 the archive is still updated and its
        // champion selection must be canonical.
        let current_canon: Vec<Tensor<B, 1>> = current
            .iter()
            .map(|t| match sense {
                ObjectiveSense::Maximize => t.clone(),
                ObjectiveSense::Minimize => t.clone().neg(),
            })
            .collect();

        // Snapshot the archives under the lock, then RELEASE it before the heavy
        // inner `evaluate_coupled` calls. Burn tensors are Arc-backed, so these
        // clones are cheap handle bumps.
        let (archive_a, archive_b) = {
            let hall = self.hall.lock();
            (hall.archives()[0].clone(), hall.archives()[1].clone())
        };

        let blended = if w <= 0.0 {
            current.clone()
        } else {
            // Population A scored against the archived B champions.
            let blended_a = if archive_b.dims()[0] > 0 {
                let res = self
                    .inner
                    .evaluate_coupled(&[populations[0].clone(), archive_b]);
                blend(&current[0], &res[0], w)
            } else {
                current[0].clone()
            };
            // Population B scored against the archived A champions (index 1).
            let blended_b = if archive_a.dims()[0] > 0 {
                let res = self
                    .inner
                    .evaluate_coupled(&[archive_a, populations[1].clone()]);
                blend(&current[1], &res[1], w)
            } else {
                current[1].clone()
            };
            vec![blended_a, blended_b]
        };

        // Re-acquire only for the cheap archive mutation. Champions are selected
        // from the CANONICAL current-gen fitness (`HallOfFame::update` is
        // sense-blind and argmaxes highest = best in maximise space).
        self.hall.lock().update(populations, &current_canon);
        blended
    }

    fn sense(&self) -> ObjectiveSense {
        self.inner.sense()
    }

    fn archive_sizes(&self) -> Vec<usize> {
        self.hall
            .lock()
            .archives()
            .iter()
            .map(|a| a.dims()[0])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    type B = Flex;

    fn pop(rows: &[f32], n: usize, d: usize) -> Tensor<B, 2> {
        let device = Default::default();
        Tensor::<B, 2>::from_data(TensorData::new(rows.to_vec(), [n, d]), &device)
    }

    fn fit(values: &[f32]) -> Tensor<B, 1> {
        let device = Default::default();
        Tensor::<B, 1>::from_data(TensorData::new(values.to_vec(), [values.len()]), &device)
    }

    #[test]
    fn capacity_formula() {
        assert_eq!(HallOfFame::<B>::capacity_for(10), 10);
        assert_eq!(HallOfFame::<B>::capacity_for(50), 10);
        assert_eq!(HallOfFame::<B>::capacity_for(100), 20);
        assert_eq!(HallOfFame::<B>::capacity_for(0), 10);
    }

    #[test]
    fn archive_grows_to_capacity_then_prunes_worst() {
        let device = Default::default();
        let mut hof = HallOfFame::<B>::new(2, 3, 1, &device);
        // Each generation's champion (index 0, highest fitness) is 5,4,3,2,1;
        // the index-1 value of −100 is always the worst, so it is never the
        // champion under the maximise convention.
        for g in 0..5_usize {
            #[allow(clippy::cast_precision_loss)]
            let p = pop(&[g as f32, g as f32 + 0.5], 2, 1);
            #[allow(clippy::cast_precision_loss)]
            let f = fit(&[(5 - g) as f32, -100.0]);
            hof.update(&[p.clone(), p], &[f.clone(), f]);
            assert!(
                hof.archives()[0].dims()[0] <= 3,
                "archive exceeded capacity at gen {g}"
            );
        }
        // After 5 generations at capacity 3, the three best (highest) champions
        // survive: 5, 4, 3.
        assert_eq!(hof.archives()[0].dims()[0], 3);
        let mut surviving = hof.archive_fitness[0].clone();
        surviving.sort_by(f32::total_cmp);
        assert_eq!(surviving, vec![3.0, 4.0, 5.0]);
    }

    /// Inner fitness = row sum; used to exercise the wrapper plumbing.
    struct RowSum;
    impl CoupledFitness<B> for RowSum {
        fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>> {
            populations
                .iter()
                .map(|p| p.clone().sum_dim(1).squeeze_dim::<1>(1))
                .collect()
        }
        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Maximize
        }
    }

    #[test]
    fn wrapper_reports_archive_sizes_and_grows() {
        let device = Default::default();
        let wrapper = HallOfFameFitness::new(RowSum, 2, 50, 2, &device);
        assert_eq!(wrapper.archive_sizes(), vec![0, 0]);
        let a = pop(&[1.0, 1.0, 2.0, 2.0], 2, 2);
        let b = pop(&[0.0, 0.0, 3.0, 3.0], 2, 2);
        let out = wrapper.evaluate_coupled(&[a.clone(), b.clone()]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].dims(), [2]);
        // One champion archived per population after one evaluation.
        assert_eq!(wrapper.archive_sizes(), vec![1, 1]);
    }

    /// Inner cost fitness: each individual's natural cost is its genome's first
    /// column value (lower is better), declared [`ObjectiveSense::Minimize`].
    struct MinCost;
    impl CoupledFitness<B> for MinCost {
        fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>> {
            populations
                .iter()
                .map(|p| {
                    // Cost = first-column value of each row.
                    p.clone().narrow(1, 0, 1).squeeze_dim::<1>(1)
                })
                .collect()
        }
        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Minimize
        }
    }

    /// Under [`ObjectiveSense::Minimize`], the archived champion must be the
    /// LOWEST-cost (best) individual, not the highest — proving the
    /// canonicalisation in `HallOfFameFitness::evaluate_coupled` reaches
    /// `HallOfFame::update`'s champion selection. Row 1 (cost `1.0`) is the
    /// unique minimum; the archive must hold its genome, not row 2 (cost `5.0`).
    #[test]
    fn minimize_archives_lowest_cost_champion() {
        let device = Default::default();
        let wrapper = HallOfFameFitness::new(MinCost, 2, 50, 1, &device);
        // Rows: costs 3.0, 1.0, 5.0 -> min is row 1.
        let a = pop(&[3.0, 1.0, 5.0], 3, 1);
        let b = pop(&[3.0, 1.0, 5.0], 3, 1);
        let _ = wrapper.evaluate_coupled(&[a, b]);

        let champ = {
            let hall = wrapper.hall.lock();
            hall.archives()[0]
                .clone()
                .into_data()
                .into_vec::<f32>()
                .expect("archived champion host-read")
        };
        assert_eq!(
            champ,
            vec![1.0],
            "Minimize champion must be the min-cost genome (1.0), not the max-cost one"
        );
    }

    /// The highest-risk invariant of the ADR 0035 change: the `current_canon`
    /// canonicalisation must sit OUTSIDE the `if w <= 0.0` branch of
    /// `HallOfFameFitness::evaluate_coupled`. At blend weight `0` the blend is
    /// skipped but the archive is STILL updated, so champion selection must
    /// remain canonical — otherwise a `Minimize` objective would crown the
    /// highest-cost (worst) individual. This pins that: if someone moves
    /// `current_canon` inside the `w <= 0.0` branch, the archived champion flips
    /// to the max-cost row and this test fails.
    #[test]
    fn minimize_archives_lowest_cost_champion_even_at_zero_blend() {
        let device = Default::default();
        let wrapper = HallOfFameFitness::new(MinCost, 2, 50, 1, &device).with_blend_weight(0.0);
        // Rows: costs 3.0, 1.0, 5.0 -> min is row 1.
        let a = pop(&[3.0, 1.0, 5.0], 3, 1);
        let b = pop(&[3.0, 1.0, 5.0], 3, 1);
        let _ = wrapper.evaluate_coupled(&[a, b]);

        let champ = {
            let hall = wrapper.hall.lock();
            hall.archives()[0]
                .clone()
                .into_data()
                .into_vec::<f32>()
                .expect("archived champion host-read")
        };
        assert_eq!(
            champ,
            vec![1.0],
            "at w=0 the Minimize champion must still be the min-cost genome (1.0), \
             proving canonicalisation reaches champion selection with blending disabled"
        );
    }

    #[test]
    fn blend_zero_passes_through_current_fitness() {
        let device = Default::default();
        let wrapper = HallOfFameFitness::new(RowSum, 2, 50, 2, &device).with_blend_weight(0.0);
        let a = pop(&[1.0, 1.0, 2.0, 2.0], 2, 2);
        let b = pop(&[0.0, 0.0, 3.0, 3.0], 2, 2);
        // First eval seeds the archive; second would blend if w > 0.
        let _ = wrapper.evaluate_coupled(&[a.clone(), b.clone()]);
        let out = wrapper.evaluate_coupled(&[a, b]);
        let va = out[0]
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("fitness host-read of a tensor this test just built");
        // Pure row sums regardless of the (non-empty) archive.
        assert_eq!(va, vec![2.0, 4.0]);
    }
}
