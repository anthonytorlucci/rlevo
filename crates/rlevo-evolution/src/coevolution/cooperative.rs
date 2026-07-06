//! Cooperative co-evolution — CCGA (Potter & De Jong 1994).
//!
//! A high-dimensional problem is decomposed by assigning disjoint subsets of
//! dimensions to two populations. Neither population holds a complete solution
//! on its own: to score a member of population A, it is combined with a
//! *representative* drawn from population B (and vice versa) to assemble a
//! full-dimensional candidate, which the [`CoupledFitness`] then evaluates
//! row-wise.
//!
//! # Where the coupling lives
//!
//! Representative selection and full-genome assembly live in
//! [`CooperativeCoEA::step`] — the *algorithm* owns the coupling, alongside
//! [`RepresentativePolicy`] (in [`CooperativeCoEAParams`]). The
//! [`CoupledFitness`] stays a pure, stateless
//! row-wise objective (e.g. Rastrigin): it receives already-assembled
//! full-dimensional populations and never performs selection or holds a lock.

use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::rngs::StdRng;
use rand::{Rng, RngExt};
use rlevo_core::config::{ConfigError, ConstraintKind, Validate};

use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::Strategy;

use super::fitness::CoupledFitness;
use super::harness::CoEAMetrics;
use super::{CoEAState, CoEvolutionaryAlgorithm};

/// Which member of the opposing population completes a partial candidate
/// during fitness evaluation.
///
/// These map onto the standard CCGA coupling strategies: `Best` is the
/// canonical single representative; `Random` is sampled coupling with a
/// fresh draw per generation; `Archive` keeps a bounded ring of past
/// champions and cycles through it.
#[derive(Clone, Copy, Debug, Default)]
pub enum RepresentativePolicy {
    /// Always pair against the best individual seen so far (previous
    /// generation's best). Canonical CCGA.
    #[default]
    Best,
    /// Pair against a uniformly random member of the opposing population,
    /// re-drawn each generation.
    Random,
    /// Maintain a bounded archive of past champions and cycle through it.
    Archive {
        /// Maximum number of past champions retained.
        capacity: usize,
    },
}

/// Static parameters for [`CooperativeCoEA`].
///
/// Population A evolves the genes named by `dims_a`; population B evolves the
/// complement within `0..total_dims`. The split is validated by the
/// [`Validate`] impl (called by [`new`](Self::new), and asserted in
/// [`CooperativeCoEA`]'s `init` in debug builds).
#[derive(Debug, Clone)]
pub struct CooperativeCoEAParams<PA, PB> {
    /// Params for population A's inner strategy (genome width must equal
    /// `dims_a.len()`).
    pub params_a: PA,
    /// Params for population B's inner strategy (genome width must equal
    /// `total_dims - dims_a.len()`).
    pub params_b: PB,
    /// Global dimension indices assigned to population A. Population B
    /// receives the complement within `0..total_dims`.
    pub dims_a: Vec<usize>,
    /// Total problem dimensionality.
    pub total_dims: usize,
    /// Representative-selection policy used to complete partial candidates.
    pub representative_policy: RepresentativePolicy,
    /// Advisory fitness-evaluation budget per generation (informational in
    /// v1; the simultaneous loop evaluates each population once per
    /// generation).
    pub evaluations_per_generation: usize,
}

impl<PA, PB> CooperativeCoEAParams<PA, PB> {
    /// Build and eagerly [`validate`](Validate::validate) the parameters.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if the dimension split is invalid — see the
    /// [`Validate`] impl.
    pub fn new(
        params_a: PA,
        params_b: PB,
        dims_a: Vec<usize>,
        total_dims: usize,
        representative_policy: RepresentativePolicy,
        evaluations_per_generation: usize,
    ) -> Result<Self, ConfigError> {
        let params = Self {
            params_a,
            params_b,
            dims_a,
            total_dims,
            representative_policy,
            evaluations_per_generation,
        };
        params.validate()?;
        Ok(params)
    }
}

/// Validates that `dims_a` is a proper subset of `0..total_dims`, so that
/// `dims_a.len() + dims_b_count == total_dims` with a non-empty B.
///
/// The inner `params_a` / `params_b` are validated when their respective
/// strategies consume them; this impl checks only the dimension split (it has
/// no `PA: Validate` / `PB: Validate` bound so it stays usable with unit-typed
/// params in tests).
impl<PA, PB> Validate for CooperativeCoEAParams<PA, PB> {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "CooperativeCoEAParams";
        if self.total_dims == 0 {
            return Err(ConfigError { config: C, field: "total_dims", kind: ConstraintKind::Zero });
        }
        if self.dims_a.is_empty() {
            return Err(ConfigError {
                config: C,
                field: "dims_a",
                kind: ConstraintKind::Custom("dims_a must be non-empty"),
            });
        }
        for &d in &self.dims_a {
            if d >= self.total_dims {
                return Err(ConfigError {
                    config: C,
                    field: "dims_a",
                    kind: ConstraintKind::Custom("dims_a index is out of range for total_dims"),
                });
            }
        }
        let unique: HashSet<usize> = self.dims_a.iter().copied().collect();
        if unique.len() != self.dims_a.len() {
            return Err(ConfigError {
                config: C,
                field: "dims_a",
                kind: ConstraintKind::Custom("dims_a contains duplicate indices"),
            });
        }
        if self.total_dims - self.dims_a.len() == 0 {
            return Err(ConfigError {
                config: C,
                field: "dims_a",
                kind: ConstraintKind::Custom(
                    "dims_a covers every dimension, leaving population B empty",
                ),
            });
        }
        Ok(())
    }
}

/// Joint state for [`CooperativeCoEA`]: the shared [`CoEAState`] plus the
/// per-population representative archives used by
/// [`RepresentativePolicy::Archive`].
#[derive(Debug, Clone)]
pub struct CooperativeState<StA, StB, B: Backend> {
    /// Shared best/mean/generation trackers and inner strategy states.
    pub base: CoEAState<StA, StB>,
    /// Bounded archive of population A's past champions (`Archive` policy only).
    rep_archive_a: Option<Tensor<B, 2>>,
    /// Bounded archive of population B's past champions (`Archive` policy only).
    rep_archive_b: Option<Tensor<B, 2>>,
}

/// Cooperative (CCGA-style) co-evolutionary algorithm.
///
/// Generic over the backend `B`, the two inner strategies `SA`/`SB` (each
/// producing `Tensor<B, 2>` sub-genomes), and a stateless row-wise
/// [`CoupledFitness`] `F` evaluating assembled full-dimensional candidates.
pub struct CooperativeCoEA<B, SA, SB, F>
where
    B: Backend,
    SA: Strategy<B, Genome = Tensor<B, 2>>,
    SB: Strategy<B, Genome = Tensor<B, 2>>,
    F: CoupledFitness<B>,
{
    strategy_a: SA,
    strategy_b: SB,
    fitness: F,
    _backend: PhantomData<fn() -> B>,
}

impl<B, SA, SB, F> Debug for CooperativeCoEA<B, SA, SB, F>
where
    B: Backend,
    SA: Strategy<B, Genome = Tensor<B, 2>>,
    SB: Strategy<B, Genome = Tensor<B, 2>>,
    F: CoupledFitness<B>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CooperativeCoEA").finish_non_exhaustive()
    }
}

impl<B, SA, SB, F> CooperativeCoEA<B, SA, SB, F>
where
    B: Backend,
    SA: Strategy<B, Genome = Tensor<B, 2>>,
    SB: Strategy<B, Genome = Tensor<B, 2>>,
    F: CoupledFitness<B>,
{
    /// Build a cooperative co-evolution from two inner strategies and a
    /// (stateless, row-wise) coupled fitness.
    pub fn new(strategy_a: SA, strategy_b: SB, fitness: F) -> Self {
        Self {
            strategy_a,
            strategy_b,
            fitness,
            _backend: PhantomData,
        }
    }

    fn snapshot(&self, state: &CooperativeState<SA::State, SB::State, B>) -> CoEAMetrics {
        let sizes = self.fitness.archive_sizes();
        CoEAMetrics {
            generation: state.base.generation,
            best_fitness_a: state.base.best_a,
            best_fitness_b: state.base.best_b,
            mean_fitness_a: state.base.mean_a,
            mean_fitness_b: state.base.mean_b,
            hof_size_a: sizes.first().copied().unwrap_or(0),
            hof_size_b: sizes.get(1).copied().unwrap_or(0),
        }
    }
}

impl<B, SA, SB, F> CoEvolutionaryAlgorithm<B> for CooperativeCoEA<B, SA, SB, F>
where
    B: Backend,
    SA: Strategy<B, Genome = Tensor<B, 2>>,
    SB: Strategy<B, Genome = Tensor<B, 2>>,
    F: CoupledFitness<B>,
{
    type Params = CooperativeCoEAParams<SA::Params, SB::Params>;
    type State = CooperativeState<SA::State, SB::State, B>;

    fn init(
        &self,
        params: &Self::Params,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State {
        debug_assert!(
            params.validate().is_ok(),
            "invalid CooperativeCoEAParams reached init: {:?}",
            params.validate().err()
        );
        let state_a = self.strategy_a.init(&params.params_a, rng, device);
        let state_b = self.strategy_b.init(&params.params_b, rng, device);
        CooperativeState {
            base: CoEAState::new(state_a, state_b),
            rep_archive_a: None,
            rep_archive_b: None,
        }
    }

    fn step(
        &self,
        params: &Self::Params,
        mut state: Self::State,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Self::State, CoEAMetrics) {
        let dims_a = &params.dims_a;
        let dims_b = complement(dims_a, params.total_dims);
        let generation = state.base.generation;

        // Both populations propose sub-genomes simultaneously.
        let (pop_a, asked_a) = self.strategy_a.ask(&params.params_a, &state.base.state_a, rng, device);
        let (pop_b, asked_b) = self.strategy_b.ask(&params.params_b, &state.base.state_b, rng, device);

        // Previous-generation bests (the canonical CCGA representatives).
        let prev_best_a = self.strategy_a.best(&state.base.state_a).map(|(g, _)| g);
        let prev_best_b = self.strategy_b.best(&state.base.state_b).map(|(g, _)| g);

        // One representative stream per generation (host-RNG convention).
        let mut rep_rng = seed_stream(rng.next_u64(), generation, SeedPurpose::Representative);
        let rep_a = select_representative(
            &pop_a,
            prev_best_a.as_ref(),
            &mut state.rep_archive_a,
            params.representative_policy,
            &mut rep_rng,
            generation,
            device,
        );
        let rep_b = select_representative(
            &pop_b,
            prev_best_b.as_ref(),
            &mut state.rep_archive_b,
            params.representative_policy,
            &mut rep_rng,
            generation,
            device,
        );

        // Assemble full-dimensional candidates: each varying sub-population is
        // completed by the opposing fixed representative.
        let full_a = assemble(&pop_a, dims_a, &rep_b, &dims_b, params.total_dims, device);
        let full_b = assemble(&pop_b, &dims_b, &rep_a, dims_a, params.total_dims, device);

        let fits = self.fitness.evaluate_coupled(&[full_a, full_b]);
        debug_assert_eq!(fits.len(), 2, "cooperative co-evolution is bi-population");
        let fit_a = fits[0].clone();
        let fit_b = fits[1].clone();

        // Each strategy consumes its own sub-population with the assembled fitness.
        let (next_a, metrics_a) = self
            .strategy_a
            .tell(&params.params_a, pop_a, fit_a, asked_a, rng);
        let (next_b, metrics_b) = self
            .strategy_b
            .tell(&params.params_b, pop_b, fit_b, asked_b, rng);

        state.base.state_a = next_a;
        state.base.state_b = next_b;
        state.base.generation += 1;
        state.base.best_a = metrics_a.best_fitness_ever();
        state.base.best_b = metrics_b.best_fitness_ever();
        state.base.mean_a = metrics_a.mean_fitness();
        state.base.mean_b = metrics_b.mean_fitness();

        let metrics = self.snapshot(&state);
        (state, metrics)
    }

    fn metrics(&self, state: &Self::State) -> CoEAMetrics {
        self.snapshot(state)
    }
}

/// The complement of `dims_a` within `0..total_dims`, ascending.
fn complement(dims_a: &[usize], total_dims: usize) -> Vec<usize> {
    let set: HashSet<usize> = dims_a.iter().copied().collect();
    (0..total_dims).filter(|d| !set.contains(d)).collect()
}

/// Extract row `idx` of `pop` as a `(1, genome_dim)` tensor.
fn row<B: Backend>(pop: &Tensor<B, 2>, idx: usize) -> Tensor<B, 2> {
    let device = pop.device();
    #[allow(clippy::cast_possible_wrap)]
    let i = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![idx as i64], [1]), &device);
    pop.clone().select(0, i)
}

/// Pick the opposing-population representative under `policy`.
///
/// `prev_best` is the opposing strategy's best-so-far (`None` at generation 0,
/// before the first `tell`). The `archive` is mutated in place for the
/// `Archive` policy. Returns a `(1, genome_dim)` representative.
fn select_representative<B: Backend>(
    pop: &Tensor<B, 2>,
    prev_best: Option<&Tensor<B, 2>>,
    archive: &mut Option<Tensor<B, 2>>,
    policy: RepresentativePolicy,
    rng: &mut StdRng,
    generation: u64,
    _device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let n = pop.dims()[0];
    match policy {
        RepresentativePolicy::Best => match prev_best {
            Some(best) => best.clone(),
            None => row(pop, 0),
        },
        RepresentativePolicy::Random => {
            let idx = rng.random_range(0..n.max(1));
            row(pop, idx)
        }
        RepresentativePolicy::Archive { capacity } => {
            if let Some(best) = prev_best {
                let updated = match archive.take() {
                    None => best.clone(),
                    Some(existing) => {
                        let cat = Tensor::cat(vec![existing, best.clone()], 0);
                        let rows = cat.dims()[0];
                        if capacity > 0 && rows > capacity {
                            // Keep the most recent `capacity` rows (FIFO window).
                            cat.narrow(0, rows - capacity, capacity)
                        } else {
                            cat
                        }
                    }
                };
                *archive = Some(updated);
            }
            match archive.as_ref() {
                Some(a) if a.dims()[0] > 0 => {
                    let rows = a.dims()[0];
                    // `generation % rows < rows`, so the conversion never fails.
                    let pick = usize::try_from(generation % rows as u64).unwrap_or(0);
                    row(a, pick)
                }
                _ => row(pop, 0),
            }
        }
    }
}

/// Scatter a sub-population and a fixed opposing representative into
/// full-dimensional rows.
///
/// `sub_pop` is `(n, my_dims.len())`; its column `j` maps to global dimension
/// `my_dims[j]`. `rep_other` is `(1, other_dims.len())`; its column `j` maps
/// to global dimension `other_dims[j]` and is broadcast across all `n` rows.
/// Returns an `(n, total_dims)` tensor. Assembly is host-side, which is also
/// where the row-wise objective evaluates, so no extra device round-trip is
/// incurred.
fn assemble<B: Backend>(
    sub_pop: &Tensor<B, 2>,
    my_dims: &[usize],
    rep_other: &Tensor<B, 2>,
    other_dims: &[usize],
    total_dims: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let dims = sub_pop.dims();
    let n = dims[0];
    let sub_w = dims[1];
    debug_assert_eq!(sub_w, my_dims.len(), "sub-population width must match my_dims");
    let sub_flat = sub_pop.clone().into_data().into_vec::<f32>().unwrap_or_default();
    let rep_flat = rep_other.clone().into_data().into_vec::<f32>().unwrap_or_default();
    debug_assert_eq!(rep_flat.len(), other_dims.len(), "representative width must match other_dims");

    let mut full = vec![0.0_f32; n * total_dims];
    for i in 0..n {
        let base = i * total_dims;
        for (j, &d) in my_dims.iter().enumerate() {
            full[base + d] = sub_flat[i * sub_w + j];
        }
        for (j, &d) in other_dims.iter().enumerate() {
            full[base + d] = rep_flat[j];
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(full, [n, total_dims]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    type B = Flex;

    fn make(rows: &[f32], n: usize, d: usize) -> Tensor<B, 2> {
        let device = Default::default();
        Tensor::<B, 2>::from_data(TensorData::new(rows.to_vec(), [n, d]), &device)
    }

    #[test]
    fn complement_is_ascending_set_difference() {
        assert_eq!(complement(&[0, 2], 4), vec![1, 3]);
        assert_eq!(complement(&[3, 1], 4), vec![0, 2]);
        assert_eq!(complement(&[0, 1], 2), Vec::<usize>::new());
    }

    #[test]
    fn assemble_scatters_into_global_positions() {
        let device = Default::default();
        // dims_a = [0, 2], dims_b = [1, 3], total = 4.
        let pop_a = make(&[10.0, 20.0, 11.0, 21.0], 2, 2); // rows: (10,20),(11,21)
        let rep_b = make(&[5.0, 7.0], 1, 2); // global dims 1,3 -> 5,7
        let full = assemble(&pop_a, &[0, 2], &rep_b, &[1, 3], 4, &device);
        let v = full.into_data().into_vec::<f32>().unwrap();
        // row0: dim0=10, dim1=5, dim2=20, dim3=7
        assert_eq!(&v[0..4], &[10.0, 5.0, 20.0, 7.0]);
        // row1: dim0=11, dim1=5, dim2=21, dim3=7
        assert_eq!(&v[4..8], &[11.0, 5.0, 21.0, 7.0]);
    }

    #[test]
    fn representative_best_uses_prev_best_else_row_zero() {
        let device = Default::default();
        let pop = make(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let mut rng = seed_stream(0, 0, SeedPurpose::Representative);
        let mut archive = None;
        // No prev best -> row 0.
        let r0 = select_representative(&pop, None, &mut archive, RepresentativePolicy::Best, &mut rng, 0, &device);
        assert_eq!(r0.into_data().into_vec::<f32>().unwrap(), vec![1.0, 2.0]);
        // With prev best -> that genome.
        let best = make(&[9.0, 9.0], 1, 2);
        let r1 = select_representative(&pop, Some(&best), &mut archive, RepresentativePolicy::Best, &mut rng, 1, &device);
        assert_eq!(r1.into_data().into_vec::<f32>().unwrap(), vec![9.0, 9.0]);
    }

    #[test]
    fn archive_policy_bounds_archive_size() {
        let device = Default::default();
        let pop = make(&[0.0, 0.0], 1, 2);
        let mut rng = seed_stream(0, 0, SeedPurpose::Representative);
        let mut archive = None;
        for g in 0..5_u64 {
            #[allow(clippy::cast_precision_loss)]
            let best = make(&[g as f32, g as f32], 1, 2);
            let _ = select_representative(
                &pop,
                Some(&best),
                &mut archive,
                RepresentativePolicy::Archive { capacity: 2 },
                &mut rng,
                g,
                &device,
            );
            if let Some(a) = archive.as_ref() {
                assert!(a.dims()[0] <= 2, "archive exceeded capacity at gen {g}");
            }
        }
        assert_eq!(archive.unwrap().dims()[0], 2);
    }

    #[test]
    fn params_new_rejects_out_of_range_dim() {
        let err = CooperativeCoEAParams::new((), (), vec![0, 1, 4], 4, RepresentativePolicy::Best, 0)
            .unwrap_err();
        assert_eq!(err.field, "dims_a");
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn params_new_rejects_when_a_covers_everything() {
        let err =
            CooperativeCoEAParams::new((), (), vec![0, 1, 2, 3], 4, RepresentativePolicy::Best, 0)
                .unwrap_err();
        assert!(err.to_string().contains("leaving population B empty"));
    }

    #[test]
    fn params_new_rejects_duplicate_dims() {
        let err = CooperativeCoEAParams::new((), (), vec![0, 0, 1], 4, RepresentativePolicy::Best, 0)
            .unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn params_new_accepts_equal_split() {
        let p = CooperativeCoEAParams::new((), (), vec![0, 1], 4, RepresentativePolicy::Best, 16)
            .unwrap();
        assert_eq!(complement(&p.dims_a, p.total_dims), vec![2, 3]);
    }
}
