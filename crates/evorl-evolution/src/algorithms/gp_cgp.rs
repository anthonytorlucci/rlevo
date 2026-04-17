//! Cartesian Genetic Programming.
//!
//! CGP encodes a directed acyclic computation graph on a fixed
//! `rows × cols` grid. Each node stores `(function_id, input_0, input_1)`,
//! plus the final output gene picks which node produces the output.
//! The genotype is a fixed-length integer vector, so populations are
//! `Tensor<B, 2, Int>` and fit the tensor abstraction cleanly.
//!
//! # Evolutionary engine
//!
//! Canonical CGP uses a `(1 + λ)` Evolution Strategy with point
//! mutation and no crossover. This module re-implements just that
//! engine directly — not via [`crate::algorithms::es_classical`] — so
//! the mutation logic can be specialized to the CGP genome semantics
//! (constrained feed-forward connections, function_id range, …).
//!
//! # Function set
//!
//! The v1 function set is fixed at construction time:
//!
//! | id | op | arity | formula |
//! |---|---|---|---|
//! | 0 | add | 2 | `a + b` |
//! | 1 | sub | 2 | `a − b` |
//! | 2 | mul | 2 | `a · b` |
//! | 3 | protected_div | 2 | `a / b` (or `a` if `|b| < ε`) |
//! | 4 | sin | 1 | `sin(a)` |
//! | 5 | cos | 1 | `cos(a)` |
//! | 6 | tanh | 1 | `tanh(a)` |
//! | 7 | const 1.0 | 0 | `1.0` |
//!
//! # Phenotype evaluation
//!
//! Evaluation runs on the host because the per-node dispatch is not a
//! good fit for dense tensor ops; node values are computed in
//! topological order (left-to-right across the grid columns).
//! Genotype storage stays on-device to match the other strategies.
//!
//! # Reference
//!
//! - Miller (2011), *Cartesian Genetic Programming* (Natural Computing
//!   Series).

use std::marker::PhantomData;

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use rand::{Rng, RngExt};

use crate::rng::{seed_stream, SeedPurpose};
use crate::strategy::{Strategy, StrategyMetrics};

/// Fixed v1 function set: arity of each opcode.
pub const FUNCTION_ARITIES: [usize; 8] = [2, 2, 2, 2, 1, 1, 1, 0];
/// Number of opcodes in the v1 function set.
pub const NUM_FUNCTIONS: usize = FUNCTION_ARITIES.len();

/// Static configuration for a [`CartesianGeneticProgramming`] run.
#[derive(Debug, Clone)]
pub struct CgpConfig {
    /// Number of offspring per generation (λ in `(1 + λ)`).
    pub lambda: usize,
    /// Number of inputs (independent variables) the program sees.
    pub n_inputs: usize,
    /// Number of grid rows.
    pub rows: usize,
    /// Number of grid columns.
    pub cols: usize,
    /// Mutation rate applied to each gene of the integer genome.
    pub mutation_rate: f32,
    /// Levels-back parameter: how many previous columns a node can
    /// connect to. `usize::MAX` means "any previous column".
    pub levels_back: usize,
}

impl CgpConfig {
    /// Sensible defaults: 1-output, 1-row, 30-column grid, mutation
    /// rate tuned to flip ~3 genes per genome.
    #[must_use]
    pub fn default_for(n_inputs: usize) -> Self {
        let rows = 1;
        let cols = 30;
        let genes_per_node = 3; // (function, input_0, input_1)
        let output_genes = 1;
        let total_genes = rows * cols * genes_per_node + output_genes;
        #[allow(clippy::cast_precision_loss)]
        let mutation_rate = 3.0 / total_genes as f32;
        Self {
            lambda: 4,
            n_inputs,
            rows,
            cols,
            mutation_rate,
            levels_back: usize::MAX,
        }
    }

    /// Genes per node in the genotype layout.
    pub const GENES_PER_NODE: usize = 3;
    /// Number of output genes (one per program output).
    pub const OUTPUT_GENES: usize = 1;

    /// Total genome length (nodes × 3 + outputs).
    #[must_use]
    pub fn genome_len(&self) -> usize {
        self.rows * self.cols * Self::GENES_PER_NODE + Self::OUTPUT_GENES
    }
}

/// Generation state for [`CartesianGeneticProgramming`].
#[derive(Debug, Clone)]
pub struct CgpState<B: Backend> {
    /// Parent genotype, shape `(1, genome_len)`.
    pub parent: Tensor<B, 2, Int>,
    /// Parent fitness (host-side scalar cache).
    pub parent_fitness: f32,
    /// Best-so-far genotype.
    pub best_genome: Option<Tensor<B, 2, Int>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Classical Cartesian GP with `(1 + λ)` ES.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use evorl_evolution::algorithms::gp_cgp::{CartesianGeneticProgramming, CgpConfig};
///
/// let strategy = CartesianGeneticProgramming::<NdArray>::new();
/// let params = CgpConfig::default_for(1);
/// assert!(params.genome_len() > 0);
/// let _ = strategy;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CartesianGeneticProgramming<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> CartesianGeneticProgramming<B> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    fn sample_initial_genome(
        params: &CgpConfig,
        rng: &mut dyn Rng,
    ) -> Vec<i64> {
        let mut genome = Vec::with_capacity(params.genome_len());
        for col in 0..params.cols {
            for _row in 0..params.rows {
                let func =
                    rng.random_range(0..NUM_FUNCTIONS as i64);
                let (inp0, inp1) = sample_input_pair(col, params, rng);
                genome.push(func);
                genome.push(inp0);
                genome.push(inp1);
            }
        }
        // Output gene: any node index or input index.
        let max_node_idx = params.n_inputs + params.rows * params.cols;
        #[allow(clippy::cast_possible_wrap)]
        genome.push(rng.random_range(0..max_node_idx as i64));
        genome
    }

    fn genome_to_host(genome: &Tensor<B, 2, Int>) -> Vec<i64> {
        genome
            .clone()
            .into_data()
            .into_vec::<i64>()
            .unwrap_or_default()
    }
}

fn sample_input_pair(col: usize, params: &CgpConfig, rng: &mut dyn Rng) -> (i64, i64) {
    let min_col = col.saturating_sub(params.levels_back);
    let node_indices_start = params.n_inputs + min_col * params.rows;
    let node_indices_end = params.n_inputs + col * params.rows;
    let max = node_indices_end.max(params.n_inputs);
    // Allowed inputs: 0..n_inputs (graph inputs) ∪ previous nodes.
    let input_count = params.n_inputs + (max - params.n_inputs).saturating_sub(
        node_indices_start.saturating_sub(params.n_inputs),
    );
    let pool: Vec<i64> = (0..params.n_inputs)
        .chain(node_indices_start..node_indices_end)
        .map(|i| {
            #[allow(clippy::cast_possible_wrap)]
            let v = i as i64;
            v
        })
        .collect();
    let pool = if pool.is_empty() {
        #[allow(clippy::cast_possible_wrap)]
        (0..params.n_inputs as i64).collect()
    } else {
        pool
    };
    let _ = input_count;
    let pick = |rng: &mut dyn Rng| -> i64 {
        let idx = rng.random_range(0..pool.len());
        pool[idx]
    };
    (pick(rng), pick(rng))
}

fn mutate_genome(
    genome: &mut [i64],
    params: &CgpConfig,
    rng: &mut dyn Rng,
) {
    let genes_per_node = CgpConfig::GENES_PER_NODE;
    let node_genes = params.rows * params.cols * genes_per_node;
    for gene_idx in 0..genome.len() {
        if rng.random::<f32>() >= params.mutation_rate {
            continue;
        }
        if gene_idx < node_genes {
            let within = gene_idx % genes_per_node;
            let node_idx = gene_idx / genes_per_node;
            let col = node_idx / params.rows;
            if within == 0 {
                // function
                #[allow(clippy::cast_possible_wrap)]
                {
                    genome[gene_idx] = rng.random_range(0..NUM_FUNCTIONS as i64);
                }
            } else {
                let (new0, new1) = sample_input_pair(col, params, rng);
                genome[gene_idx] = if within == 1 { new0 } else { new1 };
            }
        } else {
            // output gene
            let max_node_idx = params.n_inputs + params.rows * params.cols;
            #[allow(clippy::cast_possible_wrap)]
            {
                genome[gene_idx] = rng.random_range(0..max_node_idx as i64);
            }
        }
    }
}

/// Evaluates a CGP genotype at a set of input rows.
///
/// - `genome` is the host-side integer genotype.
/// - `inputs` has shape `(n_samples, n_inputs)`.
/// - Returns `(n_samples,)` predicted outputs as `f32`.
#[must_use]
pub fn evaluate_cgp(
    genome: &[i64],
    params: &CgpConfig,
    inputs: &[Vec<f32>],
) -> Vec<f32> {
    let node_count = params.rows * params.cols;
    let n_inputs = params.n_inputs;
    let output_idx = genome[genome.len() - 1] as usize;

    let mut outputs = Vec::with_capacity(inputs.len());
    let mut buf = vec![0.0_f32; n_inputs + node_count];

    for sample in inputs {
        for (i, v) in sample.iter().enumerate() {
            buf[i] = *v;
        }
        for node in 0..node_count {
            let base = node * 3;
            #[allow(clippy::cast_sign_loss)]
            let func = genome[base] as usize;
            #[allow(clippy::cast_sign_loss)]
            let a_idx = genome[base + 1] as usize;
            #[allow(clippy::cast_sign_loss)]
            let b_idx = genome[base + 2] as usize;
            let a = buf[a_idx.min(buf.len() - 1)];
            let b = buf[b_idx.min(buf.len() - 1)];
            let v = match func {
                0 => a + b,
                1 => a - b,
                2 => a * b,
                3 => {
                    if b.abs() < 1e-6 {
                        a
                    } else {
                        a / b
                    }
                }
                4 => a.sin(),
                5 => a.cos(),
                6 => a.tanh(),
                7 => 1.0,
                _ => 0.0,
            };
            buf[n_inputs + node] = if v.is_finite() { v } else { 0.0 };
        }
        outputs.push(buf[output_idx.min(buf.len() - 1)]);
    }

    outputs
}

impl<B: Backend> Strategy<B> for CartesianGeneticProgramming<B>
where
    B::Device: Clone,
{
    type Params = CgpConfig;
    type State = CgpState<B>;
    type Genome = Tensor<B, 2, Int>;

    fn init(
        &self,
        params: &CgpConfig,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> CgpState<B> {
        let genome_vec = Self::sample_initial_genome(params, rng);
        let parent = Tensor::<B, 2, Int>::from_data(
            TensorData::new(genome_vec, [1, params.genome_len()]),
            device,
        );
        CgpState {
            parent,
            parent_fitness: f32::INFINITY,
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &CgpConfig,
        state: &CgpState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, CgpState<B>) {
        // First call: evaluate the parent as "offspring" of size 1.
        if !state.parent_fitness.is_finite() {
            return (state.parent.clone(), state.clone());
        }

        let mut mut_rng =
            seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Mutation);
        let parent_vec = Self::genome_to_host(&state.parent);
        let mut offspring_genomes: Vec<i64> =
            Vec::with_capacity(params.lambda * params.genome_len());
        for _ in 0..params.lambda {
            let mut child = parent_vec.clone();
            mutate_genome(&mut child, params, &mut mut_rng);
            offspring_genomes.extend(child);
        }
        let offspring = Tensor::<B, 2, Int>::from_data(
            TensorData::new(offspring_genomes, [params.lambda, params.genome_len()]),
            device,
        );
        (offspring, state.clone())
    }

    fn tell(
        &self,
        _params: &CgpConfig,
        offspring: Tensor<B, 2, Int>,
        fitness: Tensor<B, 1>,
        mut state: CgpState<B>,
        _rng: &mut dyn Rng,
    ) -> (CgpState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();

        if !state.parent_fitness.is_finite() {
            // First tell: initial parent fitness.
            state.parent_fitness = fitness_host[0];
            state.generation += 1;
            update_best(&mut state, &offspring, &fitness_host);
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever;
            return (state, m);
        }

        // (1+λ): parent survives only if NO offspring strictly beats it;
        // canonical CGP uses `<=` to break ties in favor of offspring
        // (neutral mutations accumulate).
        let best_off_idx = fitness_host
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        let best_off_fit = fitness_host[best_off_idx];
        if best_off_fit <= state.parent_fitness {
            let device = offspring.device();
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_off_idx as i64], [1]),
                &device,
            );
            state.parent = offspring.clone().select(0, idx);
            state.parent_fitness = best_off_fit;
        }

        state.generation += 1;
        update_best(&mut state, &offspring, &fitness_host);
        let m = StrategyMetrics::from_host_fitness(
            state.generation,
            &fitness_host,
            state.best_fitness,
        );
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &CgpState<B>) -> Option<(Tensor<B, 2, Int>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

fn update_best<B: Backend>(
    state: &mut CgpState<B>,
    pop: &Tensor<B, 2, Int>,
    fitness: &[f32],
) {
    if fitness.is_empty() {
        return;
    }
    let mut best_idx = 0usize;
    let mut best_f = fitness[0];
    for (i, &f) in fitness.iter().enumerate().skip(1) {
        if f < best_f {
            best_f = f;
            best_idx = i;
        }
    }
    if best_f < state.best_fitness {
        let device = pop.device();
        #[allow(clippy::cast_possible_wrap)]
        let idx = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![best_idx as i64], [1]),
            &device,
        );
        state.best_genome = Some(pop.clone().select(0, idx));
        state.best_fitness = best_f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::BatchFitnessFn;
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::NdArray;
    use evorl_benchmarks::env::BenchEnv;
    type TestBackend = NdArray;

    /// Symbolic regression on `x² + 1` over 20 evenly spaced x ∈ [−1, 1].
    struct SymRegression {
        params: CgpConfig,
        xs: Vec<f32>,
        ys: Vec<f32>,
    }

    impl SymRegression {
        fn new(params: CgpConfig) -> Self {
            let xs: Vec<f32> = (0..20).map(|i| -1.0 + 2.0 * (i as f32) / 19.0).collect();
            let ys: Vec<f32> = xs.iter().map(|x| x * x + 1.0).collect();
            Self { params, xs, ys }
        }
    }

    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2, Int>> for SymRegression {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2, Int>,
            device: &B::Device,
        ) -> Tensor<B, 1> {
            let pop_size = population.shape().dims[0];
            let data = population.clone().into_data().into_vec::<i64>().unwrap();
            let gl = self.params.genome_len();
            let inputs: Vec<Vec<f32>> = self.xs.iter().map(|&x| vec![x]).collect();
            let mut fitness = Vec::with_capacity(pop_size);
            for row in 0..pop_size {
                let genome = &data[row * gl..(row + 1) * gl];
                let preds = evaluate_cgp(genome, &self.params, &inputs);
                let mse: f32 = preds
                    .iter()
                    .zip(self.ys.iter())
                    .map(|(p, y)| (p - y).powi(2))
                    .sum::<f32>()
                    / (self.ys.len() as f32);
                fitness.push(mse);
            }
            Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
        }
    }

    #[test]
    fn cgp_reduces_error_on_square_plus_one() {
        let device = Default::default();
        let params = CgpConfig::default_for(1);
        let landscape = SymRegression::new(params.clone());
        let initial_error = {
            // Baseline: random genome MSE on a single seed.
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(123);
            let genome = CartesianGeneticProgramming::<TestBackend>::sample_initial_genome(
                &params, &mut rng,
            );
            let inputs: Vec<Vec<f32>> =
                landscape.xs.iter().map(|&x| vec![x]).collect();
            let preds = evaluate_cgp(&genome, &params, &inputs);
            preds
                .iter()
                .zip(landscape.ys.iter())
                .map(|(p, y)| (p - y).powi(2))
                .sum::<f32>()
                / (landscape.ys.len() as f32)
        };

        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            CartesianGeneticProgramming::<TestBackend>::new(),
            params,
            landscape,
            21,
            device,
            2000,
        );
        harness.reset();
        loop {
            if harness.step(()).done {
                break;
            }
        }
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        // CGP should substantially beat the random-genome baseline.
        assert!(
            best < initial_error,
            "CGP did not improve: best={best} initial={initial_error}"
        );
        // Bias check: ought to beat predicting a constant y=1 (mean ~= 1.33).
        assert!(best < 0.2, "expected MSE < 0.2 but got {best}");
    }
}
