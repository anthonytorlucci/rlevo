//! Salp Swarm Algorithm.
//!
//! The population splits into a *leader* half and a *follower* chain:
//!
//! - Leaders (indices `0 .. N/2`) update toward the food source `F`
//!   (the best-so-far position), modulated by a time-decayed coefficient
//!   `$c_1 = 2 \exp(-(4t/T)^2)$`:
//!
//!   `$X_i \leftarrow F \pm c_1 \cdot ((\text{ub} - \text{lb}) c_2 + \text{lb})$` with `$c_2, c_3 \in U[0, 1]$`,
//!   where the sign follows `$c_3 \geq 0.5 \to +$` and `$c_3 < 0.5 \to -$`.
//!
//! - Followers (indices `N/2 .. N`) track the chain:
//!
//!   `$X_i \leftarrow (X_i + X_{i-1}) / 2$`.
//!
//! # The schedule's horizon
//!
//! In `$c_1$`, `$t$` counts generations and `$T$` is
//! [`SalpConfig::max_generations`]. `$c_1$` falls from 2 at the start to
//! about `$2 \cdot 10^{-7}$` at `$t = T$`, and stays there. The algorithm is
//! defined over a run of known length, so `$T$` has no default:
//! [`SalpConfig::default_for`] takes it, and the usual value is the
//! generation budget given to
//! [`EvolutionaryLoop`](crate::strategy::EvolutionaryLoop). The strategy
//! cannot read that budget itself, which is why the number is written
//! twice. The field's documentation says what a loop that is shorter or
//! longer than `$T$` gets.
//!
//! # Implementation
//!
//! The follower rule is realized as a parallel stencil — the shifted copy is
//! `$\text{concat}([\text{last\_leader}, \text{followers}[..-1]])$` — rather than a host-side
//! Python-style loop. That keeps the update GPU-friendly and deterministic.
//!
//! # Candor
//!
//! Legacy comparator. The leader update is a thinly-veiled biased
//! random walk toward the best; the follower chain is a stencil average
//! of old positions. Neither mechanism introduces a novel search
//! dynamic not already present in PSO or DE. Ship it for API coverage;
//! prefer [CMA-ES](crate::algorithms::cma_es) for serious work.
//!
//! # References
//!
//! - Mirjalili, Gandomi, Mirjalili, Saremi, Faris & Mirjalili (2017),
//!   *Salp Swarm Algorithm*.

use burn::tensor::{Device, Int, Tensor, TensorData};
use rand::Rng;
use rand::RngExt;

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, Validate};

use crate::ops::selection::argmax_host;
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for [`SalpSwarm`].
#[derive(Debug, Clone)]
pub struct SalpConfig {
    /// Swarm size; must be `$\geq 2$` so there is at least one leader and
    /// one follower.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds. The leader update pulls from a scaled
    /// uniform over this range.
    pub bounds: Bounds,
    /// The number of generations the decay of `$c_1$` is spread over:
    /// `$c_1 = 2 \exp(-(4t/\text{max\_generations})^2)$`, held at its final
    /// value from there on.
    ///
    /// Normally this is the generation budget given to
    /// [`EvolutionaryLoop`](crate::strategy::EvolutionaryLoop). A strategy
    /// never sees the loop's budget, so the two are separate numbers, and
    /// when they differ the run is not the one the algorithm describes:
    ///
    /// - **A loop that stops before this many generations** ends with
    ///   `$c_1$` still large: the leaders are still thrown a good part of
    ///   the domain's width from the food source every generation, and the
    ///   swarm has not started to converge.
    /// - **A loop that runs for more** keeps refining around the food
    ///   source with `$c_1$` at its floor.
    ///
    /// There is no default, because no one value is right for two loops of
    /// different lengths.
    pub max_generations: usize,
}

impl SalpConfig {
    /// Default configuration for a swarm of `pop_size` salps over
    /// `genome_dim` dimensions, with the decay of `$c_1$` spread over
    /// `max_generations` generations.
    ///
    /// Pass the generation budget you give the loop as `max_generations`;
    /// the [field](Self::max_generations) says what any other value does.
    /// The search domain is `[-5.12, 5.12]` in every dimension. Set `bounds`
    /// on the returned config to search another one.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_evolution::algorithms::metaheuristic::salp::SalpConfig;
    ///
    /// // One number, given to the config here and to the loop as its budget.
    /// const GENERATIONS: usize = 300;
    ///
    /// let config = SalpConfig::default_for(32, 10, GENERATIONS);
    /// assert_eq!(config.max_generations, GENERATIONS);
    /// ```
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize, max_generations: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: Bounds::new(-5.12, 5.12),
            max_generations,
        }
    }
}

impl Validate for SalpConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "SalpConfig";
        config::at_least(C, "pop_size", self.pop_size, 2)?;
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        config::at_least(C, "max_generations", self.max_generations, 1)?;
        Ok(())
    }
}

/// Generation state for [`SalpSwarm`].
#[derive(Debug, Clone)]
pub struct SalpState {
    /// Current positions, shape `(pop_size, D)`.
    pub positions: Tensor<2>,
    /// Host-side fitness cache.
    pub fitness: Vec<f32>,
    /// Food-source (best) genome.
    pub best_genome: Option<Tensor<2>>,
    /// Food-source fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Salp Swarm Algorithm strategy.
///
/// # Panics
///
/// [`Strategy::init`] panics if `params.pop_size < 2`, since the
/// leader/follower split requires at least one of each.
///
/// [`Strategy::ask`] panics if called a second time without an intervening
/// [`Strategy::tell`] — `state.best_genome` is `None` until the first
/// `tell` populates it. Normal loop-driven usage prevents this: the
/// first `ask` returns initial positions unevaluated, `tell` sets
/// `best_genome`, and every subsequent `ask` finds it populated.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::metaheuristic::salp::{SalpConfig, SalpSwarm};
///
/// let strategy = SalpSwarm::new();
/// // The decay runs over 300 generations; give the loop the same budget.
/// let params = SalpConfig::default_for(32, 10, 300);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SalpSwarm {}

impl SalpSwarm {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Strategy for SalpSwarm {
    type Params = SalpConfig;
    type State = SalpState;
    type Genome = Tensor<2>;

    /// Samples the initial swarm uniformly within [`SalpConfig::bounds`]
    /// using the host-RNG convention and sets the generation counter to zero.
    ///
    /// The `pop_size >= 2` invariant is enforced by [`Validate::validate`] at
    /// the loop chokepoint.
    fn init(&self, params: &SalpConfig, rng: &mut dyn Rng, device: &Device) -> SalpState {
        debug_assert!(
            params.validate().is_ok(),
            "invalid SalpConfig reached init: {params:?}"
        );
        let (lo, hi): (f32, f32) = params.bounds.into();
        // Host-sample the initial swarm from a deterministic `seed_stream`
        // rather than the process-wide Flex RNG (`B::seed` + `Tensor::random`),
        // whose draws interleave with sibling tests under the parallel
        // runner and are therefore not reproducible across thread schedules.
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mut position_rows = Vec::with_capacity(pop * genome_dim);
        for _ in 0..pop * genome_dim {
            position_rows.push(lo + (hi - lo) * stream.random::<f32>());
        }
        let positions =
            Tensor::<2>::from_data(TensorData::new(position_rows, [pop, genome_dim]), device);
        SalpState {
            positions,
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Proposes the next swarm positions.
    ///
    /// On the first call returns the initial positions unchanged. On
    /// subsequent calls, applies the two-phase update:
    ///
    /// - **Leaders** (`0..pop_size/2`): each leader moves toward the food
    ///   source with a signed, decaying step proportional to `c1` and a
    ///   host-sampled uniform offset over [`SalpConfig::bounds`].
    /// - **Followers** (`pop_size/2..pop_size`): each follower averages its
    ///   current position with the position of the salp directly ahead (a
    ///   parallel stencil over the updated leader block).
    ///
    /// Results are clamped to [`SalpConfig::bounds`].
    fn ask(
        &self,
        params: &SalpConfig,
        state: &SalpState,
        rng: &mut dyn Rng,
        device: &Device,
    ) -> (Tensor<2>, SalpState) {
        if state.fitness.is_empty() {
            return (state.positions.clone(), state.clone());
        }

        let pop_size = params.pop_size;
        let genome_dim = params.genome_dim;
        let n_leaders = pop_size / 2;
        let (lo, hi): (f32, f32) = params.bounds.into();

        // c1 decay: 2 * exp(-(4t/T)^2)
        #[allow(clippy::cast_precision_loss)]
        let t = state.generation as f32;
        #[allow(clippy::cast_precision_loss)]
        let max_t = params.max_generations.max(1) as f32;
        let frac = (4.0 * t / max_t).min(4.0);
        let c1 = 2.0 * (-(frac * frac)).exp();

        // Host-side sampling for c2, c3 so the leader step is
        // backend-agnostic and reproducible under the splitmix contract.
        let mut stream = seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Other);
        let mut leader_delta: Vec<f32> = Vec::with_capacity(n_leaders * genome_dim);
        for _ in 0..n_leaders {
            for _ in 0..genome_dim {
                let c2: f32 = stream.random::<f32>();
                let c3: f32 = stream.random::<f32>();
                let scaled = (hi - lo) * c2 + lo;
                let sign = if c3 >= 0.5 { 1.0 } else { -1.0 };
                leader_delta.push(sign * c1 * scaled);
            }
        }

        let best = state
            .best_genome
            .as_ref()
            .expect("best_genome populated after first tell")
            .clone()
            .expand([n_leaders, genome_dim]);
        let delta = Tensor::<2>::from_data(
            TensorData::new(leader_delta, [n_leaders, genome_dim]),
            device,
        );
        let new_leaders = (best + delta).clamp(lo, hi);

        // Follower stencil: X_i ← (X_i + X_{i-1}) / 2.
        let followers = state
            .positions
            .clone()
            .slice([n_leaders..pop_size, 0..genome_dim]);
        // Shifted copy: follower i reads the position of the salp
        // directly ahead, which is the last leader for the first
        // follower and `followers[i-1]` otherwise. We build the shifted
        // stack by gathering indices `[n_leaders-1, n_leaders, …, pop_size-2]`
        // from the *updated* leader block + the old followers slice.
        let joined = Tensor::cat(vec![new_leaders.clone(), followers.clone()], 0); // (pop_size, D) — leaders are already the new ones.
        #[allow(clippy::cast_possible_wrap)]
        let shift_idx: Vec<i64> = (0..(pop_size - n_leaders))
            .map(|k| (n_leaders + k - 1) as i64)
            .collect();
        let idx =
            Tensor::<1, Int>::from_data(TensorData::new(shift_idx, [pop_size - n_leaders]), device);
        let previous = joined.clone().select(0, idx);
        let new_followers = (followers + previous).mul_scalar(0.5).clamp(lo, hi);

        let new_positions = Tensor::cat(vec![new_leaders, new_followers], 0);
        let mut next = state.clone();
        next.positions.clone_from(&new_positions);
        (new_positions, next)
    }

    /// Records evaluated fitness, updates the food-source (best-so-far), and
    /// increments the generation counter.
    ///
    /// Returns the updated [`SalpState`] and a [`StrategyMetrics`] snapshot
    /// for the completed generation.
    fn tell(
        &self,
        _params: &SalpConfig,
        population: Tensor<2>,
        fitness: Tensor<1>,
        mut state: SalpState,
        _rng: &mut dyn Rng,
    ) -> (SalpState, StrategyMetrics) {
        // Sanitize at the pull (NaN → `$-\infty$`, `$+\infty$` → f32::MAX). This is the per-site
        // correctness floor for a caller driving `ask`/`tell` directly instead of
        // `EvolutionaryLoop::step`, which already sanitizes (the ADR 0034 decision-3 bypass
        // hole): otherwise a raw NaN lands in the public `state.fitness` cache. `sanitize_fitness`
        // is idempotent, so on the loop path this is a provable no-op — not redundant,
        // load-bearing.
        let fitness_host: Vec<f32> = fitness
            .into_data()
            .try_into_vec::<f32>()
            .expect("fitness tensor must be readable as f32")
            .into_iter()
            .map(crate::fitness::sanitize_fitness)
            .collect();
        state.fitness.clone_from(&fitness_host);
        state.positions.clone_from(&population);
        let best_idx = argmax_host(&fitness_host);
        if fitness_host[best_idx] > state.best_fitness {
            state.best_fitness = fitness_host[best_idx];
            let device = population.device();
            #[allow(clippy::cast_possible_wrap)]
            let idx =
                Tensor::<1, Int>::from_data(TensorData::new(vec![best_idx as i64], [1]), &device);
            state.best_genome = Some(population.select(0, idx));
        }
        state.generation += 1;
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever();
        (state, m)
    }

    /// Returns the food-source genome and its fitness, or `None` before the
    /// first [`tell`](Strategy::tell) call.
    fn best(&self, state: &SalpState) -> Option<(Tensor<2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::FromLandscape;
    use crate::strategy::EvolutionaryLoop;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::fitness::Landscape;

    /// A schedule horizon for the tests that drive no loop, or drive one
    /// too briefly for the schedule to matter.
    const HORIZON: usize = 200;

    /// Distinct finite fitness with the maximum at index 0, for direct
    /// (non-loop) `tell` calls. Finite, so it respects ADR 0034.
    #[allow(clippy::trivially_copy_pass_by_ref)] // mirror the by-ref device idiom
    fn finite_fitness(n: usize, device: &Device) -> Tensor<1> {
        #[allow(clippy::cast_precision_loss)]
        let vals: Vec<f32> = (0..n).map(|i| -(i as f32) - 1.0).collect();
        Tensor::<1>::from_data(TensorData::new(vals, [n]), device)
    }

    #[test]
    fn default_config_validates() {
        assert!(SalpConfig::default_for(30, 10, HORIZON).validate().is_ok());
    }

    #[test]
    fn rejects_pop_size_below_two() {
        let mut cfg = SalpConfig::default_for(30, 10, HORIZON);
        cfg.pop_size = 1;
        assert_eq!(cfg.validate().unwrap_err().field, "pop_size");
    }

    struct Sphere;
    impl Landscape for Sphere {
        fn evaluate(&self, x: &[f64]) -> f64 {
            x.iter().map(|v| v * v).sum()
        }
    }

    #[test]
    fn ssa_converges_on_sphere_d10() {
        // SSA reduces strongly but, per module-level candor, is not
        // expected to reach machine precision. A 1e-2 target in 600
        // generations is the strong-reduction guarantee mirroring the
        // DE premature-convergence test pattern.
        let device = Device::flex();
        let strategy = SalpSwarm::new();
        let params = SalpConfig::default_for(40, 10, 600);
        let fitness_fn = FromLandscape::new(Sphere);
        let mut evo_loop = EvolutionaryLoop::new(strategy, params, fitness_fn, 3, device, 600)
            .expect("valid params");
        evo_loop.reset();
        while !evo_loop.is_done() {
            evo_loop.step();
        }
        let best = evo_loop.latest_metrics().unwrap().best_fitness_ever();
        assert!(best < 1e-2, "SSA D10 best={best}");
    }

    #[test]
    fn best_is_none_until_first_tell() {
        let device = Device::flex();
        let strategy = SalpSwarm::new();
        let params = SalpConfig::default_for(4, 3, HORIZON);
        let mut rng = StdRng::seed_from_u64(0);
        let state = strategy.init(&params, &mut rng, &device);
        assert!(strategy.best(&state).is_none());
        let (pop, state) = strategy.ask(&params, &state, &mut rng, &device);
        let fitness = finite_fitness(4, &device);
        let (state, _m) = strategy.tell(&params, pop, fitness, state, &mut rng);
        assert!(strategy.best(&state).is_some());
    }

    #[test]
    fn first_ask_returns_initial_positions_unchanged() {
        // Before any `tell`, `state.fitness` is empty, so `ask` short-circuits
        // and hands back the untouched initial swarm.
        let device = Device::flex();
        let strategy = SalpSwarm::new();
        let params = SalpConfig::default_for(6, 4, HORIZON);
        let mut rng = StdRng::seed_from_u64(3);
        let state = strategy.init(&params, &mut rng, &device);
        let expected = state
            .positions
            .clone()
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();
        let (pop, _state) = strategy.ask(&params, &state, &mut rng, &device);
        let got = pop.into_data().try_into_vec::<f32>().unwrap();
        assert_eq!(expected, got);
    }

    #[test]
    fn minimal_and_odd_pop_sizes_run() {
        // pop_size = 2 is the leader/follower minimum; pop_size = 3 exercises
        // the odd split (1 leader, 2 followers). Both must complete without a
        // panic in the follower stencil.
        for pop in [2usize, 3] {
            let device = Device::flex();
            let strategy = SalpSwarm::new();
            let params = SalpConfig::default_for(pop, 3, 5);
            let fitness_fn = FromLandscape::new(Sphere);
            let mut evo_loop = EvolutionaryLoop::new(strategy, params, fitness_fn, 0, device, 5)
                .expect("valid params");
            evo_loop.reset();
            while !evo_loop.is_done() {
                evo_loop.step();
            }
            assert!(
                evo_loop
                    .latest_metrics()
                    .unwrap()
                    .best_fitness_ever()
                    .is_finite(),
                "pop_size {pop} produced a non-finite best"
            );
        }
    }

    #[test]
    fn ask_keeps_positions_in_bounds() {
        // Every position proposed by the two-phase update (leader step +
        // follower stencil) is clamped to bounds; verify across seeds.
        let device = Device::flex();
        let strategy = SalpSwarm::new();
        let params = SalpConfig::default_for(6, 4, HORIZON);
        let (lo, hi): (f32, f32) = params.bounds.into();
        for seed in 0..32 {
            let mut rng = StdRng::seed_from_u64(seed);
            let state = strategy.init(&params, &mut rng, &device);
            let (pop1, state) = strategy.ask(&params, &state, &mut rng, &device);
            let fitness = finite_fitness(6, &device);
            let (state, _m) = strategy.tell(&params, pop1, fitness, state, &mut rng);
            let (pop2, _state) = strategy.ask(&params, &state, &mut rng, &device);
            let values = pop2.into_data().try_into_vec::<f32>().unwrap();
            for v in values {
                assert!(
                    v >= lo - 1e-4 && v <= hi + 1e-4,
                    "seed {seed}: position {v} out of bounds [{lo}, {hi}]"
                );
            }
        }
    }

    #[test]
    fn zero_budget_loop_produces_no_metrics() {
        // The loop *budget* (distinct from SalpConfig::max_generations) is
        // zero: a well-behaved driver takes no step, so `latest_metrics` stays
        // None after `reset`. The schedule's horizon cannot follow the budget
        // down: a horizon of zero does not validate.
        let device = Device::flex();
        let strategy = SalpSwarm::new();
        let params = SalpConfig::default_for(4, 3, HORIZON);
        let fitness_fn = FromLandscape::new(Sphere);
        let mut evo_loop = EvolutionaryLoop::new(strategy, params, fitness_fn, 0, device, 0)
            .expect("valid params");
        evo_loop.reset();
        assert!(evo_loop.latest_metrics().is_none());
    }

    /// Cache hygiene (not a freeze): `tell` overwrites `state.fitness`
    /// wholesale every generation, so a `NaN` can never latch here. What this
    /// pins is the **store** — driving `init → ask → tell` directly, the
    /// ADR-0034 decision-3 bypass of `EvolutionaryLoop::step`, must not
    /// leave a raw `NaN` in the public fitness cache a caller can read back.
    #[test]
    fn nan_fitness_does_not_latch_without_loop() {
        let device = Device::flex();
        let strategy = SalpSwarm::new();
        let params = SalpConfig::default_for(6, 3, HORIZON);
        let mut rng = StdRng::seed_from_u64(7);
        let state = strategy.init(&params, &mut rng, &device);
        let (pop, state) = strategy.ask(&params, &state, &mut rng, &device);
        // Gen-0 fitness supplied by the test, never by a landscape: [NaN, finite…].
        #[allow(clippy::cast_precision_loss)]
        let mut vals: Vec<f32> = (0..params.pop_size).map(|i| -(i as f32) - 1.0).collect();
        vals[0] = f32::NAN;
        let fitness = Tensor::<1>::from_data(TensorData::new(vals, [params.pop_size]), &device);
        let (state, _m) = strategy.tell(&params, pop, fitness, state, &mut rng);
        assert!(
            state.fitness.iter().all(|f| !f.is_nan()),
            "raw NaN reached the public fitness cache: {:?}",
            state.fitness
        );
        // Pin the *value*, not just "not NaN": under the canonical maximise convention (ADR 0023 /
        // ADR 0034) `$-\infty$` is the worst representable fitness, and that is precisely what
        // makes a sanitized member unable to win a champion scan. Any other finite substitute (e.g.
        // `0.0`) clears `is_nan` yet would rank salp 0 *above* every finite -1/-2/… row and elect
        // the NaN-scoring salp as the food source — the leader poisoning this regression exists to
        // catch.
        assert!(
            state.fitness[0].is_infinite() && state.fitness[0].is_sign_negative(),
            "sanitized NaN must land as -inf in the salp-0 fitness cache: {:?}",
            state.fitness
        );
    }

    #[test]
    fn unit_budget_loop_runs_exactly_one_generation() {
        let device = Device::flex();
        let strategy = SalpSwarm::new();
        let params = SalpConfig::default_for(4, 3, 1);
        let fitness_fn = FromLandscape::new(Sphere);
        let mut evo_loop = EvolutionaryLoop::new(strategy, params, fitness_fn, 0, device, 1)
            .expect("valid params");
        evo_loop.reset();
        evo_loop.step();
        assert!(evo_loop.is_done());
        assert_eq!(evo_loop.generation(), 1);
        assert!(evo_loop.latest_metrics().is_some());
    }
}
