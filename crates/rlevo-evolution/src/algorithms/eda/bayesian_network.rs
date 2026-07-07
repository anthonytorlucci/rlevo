//! Bayesian-network model (BOA — Bayesian Optimization Algorithm) for binary
//! search spaces.
//!
//! Unlike the univariate binary models ([`super::univariate_bernoulli`],
//! [`super::compact_genetic`]) and the first-order chain of
//! [`super::dependency_chain`], this model learns an arbitrary-topology directed
//! acyclic graph (DAG) over the binary genes, bounded to at most
//! [`BayesianNetworkParams::max_parents`] parents per node. [`fit`] greedily
//! constructs the network by adding the single edge with the highest score gain
//! each round; [`sample`] performs ancestral sampling along a topological order,
//! drawing each gene from its conditional probability table (CPT) given the
//! already-sampled parent configuration.
//!
//! The chain is built à la BOA (Pelikan, Goldberg & Cantú-Paz, 1999): starting
//! from an edgeless network, the algorithm repeatedly scores every candidate
//! edge `u → v` and commits the one with the largest strictly-positive gain,
//! subject to the `max_parents` cap and an acyclicity check, until no profitable
//! edge remains.
//!
//! The `fitness` tensor is accepted by the [`ProbabilityModel`] interface but
//! always ignored; the fit is unweighted (the MIMIC precedent).
//!
//! # Non-incremental fit
//!
//! `prev` is consumed only as the *not-bootstrap* signal: when `prev = Some(_)`
//! the whole network — structure and CPTs — is relearned from scratch from the
//! current generation's selected rows. Canonical BOA carries no cross-generation
//! state, so the previous [`BayesianNetworkState`] is never read. The `Some`
//! arm exists purely to distinguish the learning path from the
//! [`params`](BayesianNetworkParams)-only prior path.
//!
//! # Structure score: BIC
//!
//! Edges are scored with the Bayesian Information Criterion (BIC). For a node
//! `v` with sorted parent set `Pa` (`q = |Pa|`), let `N(c, x)` be the number of
//! selected rows in which the parents take packed configuration `c` and gene `v`
//! takes bit `x ∈ {0, 1}`, and `N(c) = N(c, 0) + N(c, 1)`:
//!
//! ```text
//! score(v, Pa) = Σ_c Σ_x  N(c, x) · ln( N(c, x) / N(c) )  −  (ln(n) / 2) · 2^q
//! ```
//!
//! The log-likelihood term rewards parents that make `v` more predictable; the
//! `−½·ln(n)·2^q` complexity term penalises CPT size (`2^q` cells), which grows
//! exponentially in the parent count. This penalty is the structural analogue
//! of [`super::dependency_chain`]'s `|r| < 2/√k` significance filter: both
//! suppress spurious dependencies that a univariate model would never pay for,
//! here by requiring an edge's likelihood improvement to outweigh the cost of
//! doubling the child's CPT. The score is computed on **raw maximum-likelihood
//! counts** — never the Laplace-smoothed counts used for CPT estimation — so the
//! penalty is the sole overfitting guard. Terms with `N(c, x) = 0` contribute
//! exactly `0` (the `p·ln p → 0` limit), and configurations with `N(c) = 0`
//! contribute `0` likelihood while still counting toward the `2^q` penalty. All
//! scoring arithmetic is performed in `f64`.
//!
//! # Parent-configuration bit-packing
//!
//! For a node `v` with `parents[v]` sorted ascending, a row's parent
//! configuration is packed as `config = Σ_j bit(gene[parents[v][j]]) << j`:
//! parent `j` (in sorted order) contributes bit `j`. [`fit`] and [`sample`] use
//! the identical packing, so the CPT index computed at sampling time matches the
//! one used during estimation. See the [`cpt`](BayesianNetworkState::cpt) field.
//!
//! # Complexity
//!
//! [`fit`] is `O(D² · N · κ)` per generation with gain caching: the first sweep
//! scores all `D²` candidate edges (each an `O(N·κ)` counting pass), and after
//! each accepted edge only the `D` entries sharing the affected child are
//! rescored. It is fully host-side and sequential. [`sample`] is `O(D)` per
//! drawn individual: one conditional Bernoulli draw per gene.
//!
//! # Binary gene convention
//!
//! Genes are emitted as raw `{0, 1}` `f32` values, so
//! [`EdaParams::bounds`](crate::algorithms::eda::EdaParams::bounds) clamps are a
//! documented no-op (the PBIL / cGA precedent).
//!
//! # References
//!
//! - Pelikan, Goldberg & Cantú-Paz (1999), *BOA: The Bayesian optimization
//!   algorithm*.
//!
//! [`fit`]: crate::ProbabilityModel::fit
//! [`sample`]: crate::ProbabilityModel::sample

use burn::tensor::{Tensor, TensorData, backend::Backend};
use rand::{Rng, RngExt};

use crate::probability_model::ProbabilityModel;

/// Per-run configuration for the [`BayesianNetwork`] model.
///
/// Held inside [`EdaParams::model`](crate::algorithms::eda::EdaParams::model)
/// for the lifetime of a run. Use [`BayesianNetworkParams::default_for`] for
/// typical binary-optimisation defaults.
#[derive(Debug, Clone)]
pub struct BayesianNetworkParams {
    /// Number of bits per genome; the number of nodes `D` in the network and
    /// the length of [`BayesianNetworkState::order`] and
    /// [`BayesianNetworkState::parents`].
    pub genome_dim: usize,
    /// Maximum number of parents per node (`κ`); bounds each node's CPT to
    /// `2^κ` cells and caps the greedy edge-addition search.
    pub max_parents: usize,
    /// Prior marginal probability of a `1` gene, used to seed every edgeless
    /// CPT on the prior path (`prev = None`).
    pub init_prob: f32,
    /// Laplace pseudo-count `s` added per CPT cell during estimation; `s ≥ 1`
    /// keeps every probability strictly inside `(0, 1)`. Applies only to CPT
    /// estimation for sampling, never to the BIC structure score. The value is
    /// floored to `1` inside [`fit`](ProbabilityModel::fit), so a supplied `0`
    /// is treated as `1` to uphold the strictly-interior guarantee.
    pub smoothing_count: usize,
}

impl BayesianNetworkParams {
    /// Sensible BOA defaults for a `genome_dim`-bit problem.
    #[must_use]
    pub fn default_for(genome_dim: usize) -> Self {
        Self {
            genome_dim,
            max_parents: 3,
            init_prob: 0.5,
            smoothing_count: 1,
        }
    }
}

/// Fitted state for the [`BayesianNetwork`] model after one call to
/// [`ProbabilityModel::fit`].
///
/// On the prior path (`prev = None`) the network is edgeless: `order` is the
/// natural order `[0, 1, …, D-1]`, every `parents[v]` is empty, and every
/// `cpt[v]` is the single-entry vector `[init_prob]`.
#[derive(Debug, Clone)]
pub struct BayesianNetworkState {
    /// Topological sampling order: a permutation of `0..D` such that every
    /// node appears after all of its parents. Ancestral sampling walks this
    /// order so each gene's parents are already drawn.
    pub order: Vec<usize>,
    /// `parents[node]` is the node's parent index set, kept sorted ascending.
    /// The sort defines the bit positions used by the CPT packing (see the
    /// [module docs](self)).
    pub parents: Vec<Vec<usize>>,
    /// Conditional probability tables: `cpt[node][config]` is
    /// `P(node = 1 | parents = config)`, where `config` is the bit-packed
    /// parent configuration `Σ_j bit(parent_j) << j` over `parents[node]` in
    /// sorted order. Each inner vector has length `2^|parents[node]|`.
    pub cpt: Vec<Vec<f32>>,
}

/// Bayesian-network model for binary spaces (BOA).
///
/// Implements [`ProbabilityModel`] by greedily learning a bounded-in-degree DAG
/// over the binary genes with a BIC structure score, then ancestral-sampling
/// from the fitted CPTs (see the [module docs](self) for the algorithm, the BIC
/// rationale, bit-packing, and references). Fitness is accepted but ignored; the
/// fit is always unweighted and non-incremental.
///
/// [`fit`](ProbabilityModel::fit) is `O(D² · N · κ)`;
/// [`sample`](ProbabilityModel::sample) is `O(D)` per individual.
#[derive(Debug, Clone, Copy, Default)]
pub struct BayesianNetwork;

/// Build the edgeless prior state: natural order, no parents, single-cell CPTs
/// initialised to `init_prob`.
///
/// `init_prob` is clamped into the open interior `(0, 1)` before it seeds the
/// CPTs. This is the single chokepoint for every prior return, so a
/// misconfigured or non-finite `init_prob` (e.g. `NaN`, `1.5`, `-0.3`) cannot
/// silently produce a degenerate population during sampling. `NaN` maps to the
/// neutral `0.5` (`f32::clamp` would *propagate* `NaN`); `±inf` clamp to the
/// interior bounds.
fn prior_state(d: usize, init_prob: f32) -> BayesianNetworkState {
    let p = if init_prob.is_nan() {
        0.5
    } else {
        init_prob.clamp(1e-6, 1.0 - 1e-6)
    };
    BayesianNetworkState {
        order: (0..d).collect(),
        parents: vec![Vec::new(); d],
        cpt: vec![vec![p]; d],
    }
}

/// Pack the parent configuration for node `v` from a single row's bits.
///
/// `parents` is `parents[v]` (sorted ascending); parent `j` contributes bit `j`.
fn pack_config(bits: &[u8], row_base: usize, parents: &[usize]) -> usize {
    let mut config = 0usize;
    for (j, &p) in parents.iter().enumerate() {
        if bits[row_base + p] == 1 {
            config |= 1 << j;
        }
    }
    config
}

/// BIC score of node `v` given the (sorted) candidate parent set `parents`.
///
/// Single pass over the `n` rows: pack each row's parent config and increment
/// `counts[config * 2 + bit_v]`. The likelihood term sums
/// `N(c, x) · ln(N(c, x) / N(c))` over occupied cells (zero counts skipped), and
/// the `½·ln(n)·2^q` complexity penalty applies regardless of which configs were
/// observed. All arithmetic is `f64`; scores use raw MLE counts.
//
// Single-char math names (n, d, v, q, c, x) mirror the BIC formula and the
// sibling EDA models; spelling them out would obscure the algebra.
#[allow(clippy::many_single_char_names)]
fn bic_score(bits: &[u8], n: usize, d: usize, v: usize, parents: &[usize]) -> f64 {
    let q = parents.len();
    let num_configs = 1usize << q;
    // counts[config * 2 + x] = N(config, x).
    let mut counts = vec![0u32; num_configs * 2];
    for i in 0..n {
        let base = i * d;
        let x = usize::from(bits[base + v]);
        let config = pack_config(bits, base, parents);
        counts[config * 2 + x] += 1;
    }

    let mut log_likelihood = 0.0_f64;
    for c in 0..num_configs {
        let count_0 = counts[c * 2];
        let count_1 = counts[c * 2 + 1];
        let count_total = count_0 + count_1;
        if count_total == 0 {
            // N(c) == 0: zero likelihood, but the config still counts toward the
            // 2^q penalty below (that is the pressure keeping q small).
            continue;
        }
        // f64::from(u32) is a lossless widening, not a lossy cast.
        let total_f = f64::from(count_total);
        for &count_x in &[count_0, count_1] {
            if count_x == 0 {
                // N(c, x) == 0 contributes exactly 0 (the p·ln p → 0 limit); no
                // ln(0) path is ever reached.
                continue;
            }
            let count_x_f = f64::from(count_x);
            log_likelihood += count_x_f * (count_x_f / total_f).ln();
        }
    }

    // Complexity penalty: ½·ln(n)·2^q over all 2^q configs.
    #[allow(clippy::cast_precision_loss)]
    let nf = n as f64;
    #[allow(clippy::cast_precision_loss)]
    let penalty = 0.5 * nf.ln() * (num_configs as f64);
    log_likelihood - penalty
}

/// Insert `value` into the ascending-sorted vector `parents`, keeping it sorted.
fn insert_sorted(parents: &mut Vec<usize>, value: usize) {
    let pos = parents.partition_point(|&p| p < value);
    parents.insert(pos, value);
}

/// Does adding edge `u → v` create a cycle?
///
/// A cycle appears iff `v` is already an ancestor of `u`. Iterative DFS upward
/// from `u` through `parents[]`, looking for `v`. `O(D·κ)` per check.
fn creates_cycle(parents: &[Vec<usize>], u: usize, v: usize) -> bool {
    let d = parents.len();
    let mut visited = vec![false; d];
    let mut stack = vec![u];
    while let Some(node) = stack.pop() {
        if node == v {
            return true;
        }
        if visited[node] {
            continue;
        }
        visited[node] = true;
        for &p in &parents[node] {
            if !visited[p] {
                stack.push(p);
            }
        }
    }
    false
}

/// Kahn's algorithm with deterministic minimum-index selection.
///
/// Repeatedly emits the smallest-index node whose remaining (unemitted) parent
/// count is zero. The greedy loop guarantees the graph is a DAG.
fn topological_order(parents: &[Vec<usize>]) -> Vec<usize> {
    let d = parents.len();
    let mut indegree: Vec<usize> = parents.iter().map(Vec::len).collect();
    let mut emitted = vec![false; d];
    let mut order = Vec::with_capacity(d);
    while order.len() < d {
        // Smallest-index node with indegree 0 not yet emitted.
        let mut next = None;
        for v in 0..d {
            if !emitted[v] && indegree[v] == 0 {
                next = Some(v);
                break;
            }
        }
        let Some(node) = next else { break };
        emitted[node] = true;
        order.push(node);
        // Decrement indegree of every node that has `node` as a parent.
        for (child, ps) in parents.iter().enumerate() {
            if !emitted[child] && ps.contains(&node) {
                indegree[child] -= 1;
            }
        }
    }
    order
}

impl<B: Backend> ProbabilityModel<B> for BayesianNetwork {
    type Params = BayesianNetworkParams;
    type State = BayesianNetworkState;

    /// Fit the Bayesian network to the selected population.
    ///
    /// When `prev = None` returns the edgeless prior (natural order, empty
    /// parent lists, single-cell CPTs at `init_prob`); `population` and
    /// `fitness` are ignored. Otherwise the whole network is relearned from
    /// scratch (the fit is non-incremental — `prev` is the not-bootstrap signal
    /// only):
    ///
    /// 1. Bitizes the selected rows to `{0, 1}` via `>= 0.5`.
    /// 2. Greedily adds the highest-BIC-gain edge each round (subject to the
    ///    `max_parents` cap and acyclicity) until no strictly-positive gain
    ///    remains, using a `D × D` gain cache.
    /// 3. Estimates Laplace-smoothed CPTs from the final structure.
    /// 4. Computes a deterministic topological order (Kahn, min-index).
    ///
    /// The `fitness` argument is accepted but always ignored.
    ///
    /// # Panics
    ///
    /// Does not panic; the closing `debug_assert_eq!` checks the topological
    /// order covers all `D` nodes (guaranteed by the DAG invariant).
    // The counting passes, gain-cached greedy search, CPT estimation, and
    // topological ordering form one coherent fit; splitting them would scatter
    // the shared `bits`/`parents` buffers without aiding readability.
    // Single-char math names (n, d, v, q, s, u) mirror the BIC/CPT formulae and
    // the sibling EDA models; spelling them out would obscure the algebra.
    #[allow(clippy::too_many_lines, clippy::many_single_char_names)]
    fn fit(
        &self,
        params: &Self::Params,
        prev: Option<&Self::State>,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State {
        let _ = device;
        // Fitness is accepted but ignored: the fit is unweighted.
        let _ = fitness;
        let Some(_prev) = prev else {
            // Prior path: edgeless network in natural order; population and
            // fitness ignored. `prev` is consumed only as the bootstrap signal.
            return prior_state(params.genome_dim, params.init_prob);
        };

        // Host extraction and bitization. The population's column count is the
        // row stride for every counting pass below, so it — not
        // `params.genome_dim` — is the authoritative `d` (mirrors
        // `DependencyChain::fit`); the two must agree.
        let [n, d] = population.dims();
        debug_assert_eq!(
            d, params.genome_dim,
            "population column count must match params.genome_dim"
        );
        // CPT sizes are 2^q with q <= max_parents; a cap at or above the
        // word width would overflow the `1usize << q` table sizing.
        debug_assert!(
            params.max_parents < usize::BITS as usize,
            "max_parents must be below usize::BITS"
        );
        let rows = population.into_data().into_vec::<f32>().expect("population tensor must be readable as f32");
        if n == 0 {
            // Degenerate input: nothing to learn, return the prior-shaped
            // state (params-shaped, since a 0×0 tensor carries no width).
            return prior_state(params.genome_dim, params.init_prob);
        }
        let bits: Vec<u8> = rows.iter().map(|&v| u8::from(v >= 0.5)).collect();

        // Greedy structure learning with a D×D gain cache.
        let mut parents: Vec<Vec<usize>> = vec![Vec::new(); d];
        let mut base_score: Vec<f64> = (0..d).map(|v| bic_score(&bits, n, d, v, &[])).collect();

        // gain_cache[u * d + v] = score(v, parents[v] ∪ {u}) − base_score[v],
        // an exact recomputation. Entries are recomputed for child v* after each
        // accepted edge; eligibility (cycle / cap / already-present) is checked
        // live at selection time, so the cache holds only the score gain.
        // NOTE: this `NEG_INFINITY` is a structure-score (BDeu/likelihood)
        // gain sentinel for greedy edge maximisation — NOT objective fitness.
        // It is independent of the crate's maximise convention; do not flip it.
        let mut gain_cache = vec![f64::NEG_INFINITY; d * d];
        // Helper closure would need to borrow `bits`/`parents`/`base_score`
        // mutably and immutably; an inline recompute keeps borrows simple.
        // Initial full sweep.
        for u in 0..d {
            for v in 0..d {
                if u == v {
                    continue;
                }
                let mut cand = parents[v].clone();
                insert_sorted(&mut cand, u);
                gain_cache[u * d + v] = bic_score(&bits, n, d, v, &cand) - base_score[v];
            }
        }

        loop {
            // Select the eligible (u, v) with the maximal cached gain.
            // Lexicographic (u, v) scan with strict '>' ⇒ first pair wins ties.
            let mut best: Option<(f64, usize, usize)> = None;
            for u in 0..d {
                for v in 0..d {
                    if u == v
                        || parents[v].len() >= params.max_parents
                        || parents[v].contains(&u)
                        || creates_cycle(&parents, u, v)
                    {
                        continue;
                    }
                    let g = gain_cache[u * d + v];
                    if best.is_none_or(|(bg, _, _)| g > bg) {
                        best = Some((g, u, v));
                    }
                }
            }

            let Some((gain, u, v)) = best else { break };
            if gain <= 0.0 {
                // Strictly-positive gain required to add an edge.
                break;
            }
            insert_sorted(&mut parents[v], u);
            base_score[v] += gain;

            // Only entries with child == v are now stale; recompute just those.
            for uu in 0..d {
                if uu == v {
                    continue;
                }
                let mut cand = parents[v].clone();
                if !cand.contains(&uu) {
                    insert_sorted(&mut cand, uu);
                }
                gain_cache[uu * d + v] = bic_score(&bits, n, d, v, &cand) - base_score[v];
            }
        }

        // CPT estimation from the final structure: one counting pass per node.
        // Floor the smoothing at 1 so every probability stays strictly inside
        // `(0, 1)` (the field-doc guarantee): with `s ≥ 1`, `den = N(c) + 2s > 0`
        // always, so the `0/0` case is unreachable and `count_1/count_total`
        // cannot pin a cell to an absorbing `0.0`/`1.0`.
        let s = params.smoothing_count.max(1);
        let mut cpt: Vec<Vec<f32>> = Vec::with_capacity(d);
        // Laplace pseudo-count as f64; `s` is a tiny smoothing constant, far
        // below f64's exact-integer range, so the cast is lossless.
        #[allow(clippy::cast_precision_loss)]
        let s_f = s as f64;
        for v in 0..d {
            let q = parents[v].len();
            let num_configs = 1usize << q;
            let mut counts = vec![0u32; num_configs * 2];
            for i in 0..n {
                let base = i * d;
                let x = usize::from(bits[base + v]);
                let config = pack_config(&bits, base, &parents[v]);
                counts[config * 2 + x] += 1;
            }
            let mut table = Vec::with_capacity(num_configs);
            for c in 0..num_configs {
                let count_1 = counts[c * 2 + 1];
                let count_total = counts[c * 2] + count_1;
                // (N(c,1) + s) / (N(c) + 2s); f64::from(u32) is lossless. With
                // `s ≥ 1` the denominator is always positive, so no `0/0` guard
                // is needed.
                let num = f64::from(count_1) + s_f;
                let den = f64::from(count_total) + 2.0 * s_f;
                // Probability in (0, 1) for s ≥ 1; the f64→f32 narrowing of a
                // value in [0, 1] cannot truncate meaningfully.
                #[allow(clippy::cast_possible_truncation)]
                let prob = (num / den) as f32;
                table.push(prob);
            }
            cpt.push(table);
        }

        let order = topological_order(&parents);
        debug_assert_eq!(order.len(), d, "topological order must cover all nodes");

        BayesianNetworkState {
            order,
            parents,
            cpt,
        }
    }

    /// Draw `n` binary genomes by ancestral sampling along the topological order.
    ///
    /// Each gene `v` is sampled from `P(v = 1 | parents)` read out of its CPT at
    /// the bit-packed parent configuration (parents already sampled, since the
    /// traversal follows the topological order). Exactly one `rng.random::<f32>()`
    /// call is consumed per gene regardless of structure, keeping RNG
    /// consumption stable. Host RNG only (never `Tensor::random` / `B::seed`).
    /// The returned tensor has shape `(n, D)` and contains only `0.0` and `1.0`.
    fn sample(
        &self,
        state: &Self::State,
        n: usize,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 2> {
        let d = state.parents.len();
        let mut rows = vec![0.0_f32; n * d];
        for i in 0..n {
            let base = i * d;
            for &v in &state.order {
                let mut config = 0usize;
                for (j, &p) in state.parents[v].iter().enumerate() {
                    if rows[base + p] >= 0.5 {
                        config |= 1 << j;
                    }
                }
                let p1 = state.cpt[v][config];
                rows[base + v] = if rng.random::<f32>() < p1 { 1.0 } else { 0.0 };
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(rows, [n, d]), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type TestBackend = Flex;

    fn pop(rows: Vec<f32>, n: usize, d: usize) -> Tensor<TestBackend, 2> {
        let device = Default::default();
        Tensor::<TestBackend, 2>::from_data(TensorData::new(rows, [n, d]), &device)
    }

    fn fitness(values: Vec<f32>) -> Tensor<TestBackend, 1> {
        let device = Default::default();
        let n = values.len();
        Tensor::<TestBackend, 1>::from_data(TensorData::new(values, [n]), &device)
    }

    fn fit_prior(p: &BayesianNetworkParams) -> BayesianNetworkState {
        let device = Default::default();
        <BayesianNetwork as ProbabilityModel<TestBackend>>::fit(
            &BayesianNetwork,
            p,
            None,
            pop(vec![], 0, 0),
            fitness(vec![]),
            &device,
        )
    }

    fn refit(p: &BayesianNetworkParams, rows: Vec<f32>, n: usize, d: usize) -> BayesianNetworkState {
        let device = Default::default();
        let prior = fit_prior(p);
        // Test row counts are tiny; the cast is lossless.
        #[allow(clippy::cast_precision_loss)]
        let fit_values: Vec<f32> = (0..n).map(|i| i as f32).collect();
        <BayesianNetwork as ProbabilityModel<TestBackend>>::fit(
            &BayesianNetwork,
            p,
            Some(&prior),
            pop(rows, n, d),
            fitness(fit_values),
            &device,
        )
    }

    #[test]
    fn prior_is_edgeless_with_init_prob() {
        let p = BayesianNetworkParams::default_for(3);
        let state = fit_prior(&p);
        assert_eq!(state.order, vec![0, 1, 2], "prior order is natural");
        for ps in &state.parents {
            assert!(ps.is_empty(), "prior parent lists must be empty");
        }
        for table in &state.cpt {
            assert_eq!(table, &vec![0.5], "prior CPT is single-cell init_prob");
        }
    }

    #[test]
    fn two_fits_same_data_identical_state() {
        let p = BayesianNetworkParams::default_for(3);
        // gene1 = copy of gene0; gene2 a balanced, decorrelated pattern.
        let rows = vec![
            0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, //
            1.0, 1.0, 1.0, //
            1.0, 1.0, 0.0, //
            1.0, 1.0, 1.0, //
            1.0, 1.0, 0.0, //
            1.0, 1.0, 1.0, //
        ];
        let a = refit(&p, rows.clone(), 10, 3);
        let b = refit(&p, rows, 10, 3);
        assert_eq!(a.order, b.order, "order must be bit-deterministic");
        assert_eq!(a.parents, b.parents, "parents must be bit-deterministic");
        assert_eq!(a.cpt, b.cpt, "CPTs must be bit-deterministic");
    }

    #[test]
    fn cpt_probabilities_strictly_interior() {
        let p = BayesianNetworkParams::default_for(3);
        // gene1 is constant 1 (constant column); Laplace smoothing must keep
        // every CPT entry strictly inside (0, 1).
        let rows = vec![
            0.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, //
            1.0, 1.0, 0.0, //
            1.0, 1.0, 1.0, //
            0.0, 1.0, 1.0, //
            1.0, 1.0, 0.0, //
        ];
        let state = refit(&p, rows, 6, 3);
        for (v, table) in state.cpt.iter().enumerate() {
            for (c, &prob) in table.iter().enumerate() {
                assert!(
                    prob > 0.0 && prob < 1.0,
                    "cpt[{v}][{c}] = {prob} not strictly interior"
                );
            }
        }
    }

    #[test]
    fn samples_are_binary_and_finite() {
        let p = BayesianNetworkParams::default_for(4);
        let rows = vec![
            0.0, 0.0, 1.0, 1.0, //
            1.0, 1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, 0.0, //
        ];
        let state = refit(&p, rows, 4, 4);
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(7);
        let samples = <BayesianNetwork as ProbabilityModel<TestBackend>>::sample(
            &BayesianNetwork,
            &state,
            1000,
            &mut rng,
            &device,
        );
        let data = samples.into_data().into_vec::<f32>().expect("samples host-read of a tensor this test just built");
        for v in data {
            assert!(v.is_finite(), "sampled gene must be finite, got {v}");
            // Exact float compare is correct: sample() writes literal 0.0/1.0.
            #[allow(clippy::float_cmp)]
            let is_binary = v == 0.0 || v == 1.0;
            assert!(is_binary, "non-binary gene {v}");
        }
    }

    #[test]
    fn recovers_pairwise_dependency() {
        // d=3, n=20: gene0 balanced (10 zeros then 10 ones), gene1 = copy of
        // gene0, gene2 alternating within each half (zero correlation to gene0).
        // BIC: dependence gain ≈ n·ln2 ≈ 13.9 vs penalty increment ½·ln20 ≈ 1.5
        // ⇒ exactly one edge between 0 and 1; gene2 isolated.
        let p = BayesianNetworkParams::default_for(3);
        let mut rows = Vec::with_capacity(20 * 3);
        for i in 0..20 {
            let g0 = if i < 10 { 0.0 } else { 1.0 };
            let g1 = g0; // exact copy
            let g2 = if i % 2 == 0 { 0.0 } else { 1.0 }; // alternating, decorrelated
            rows.push(g0);
            rows.push(g1);
            rows.push(g2);
        }
        let state = refit(&p, rows, 20, 3);
        // Direction-agnostic single edge between 0 and 1.
        let edge_0_to_1 = state.parents[1] == vec![0];
        let edge_1_to_0 = state.parents[0] == vec![1];
        assert!(
            edge_0_to_1 ^ edge_1_to_0,
            "expected exactly one 0↔1 edge, parents = {:?}",
            state.parents
        );
        // gene2 has no parents and appears in nobody's parent list.
        assert!(state.parents[2].is_empty(), "gene2 must have no parents");
        for ps in &state.parents {
            assert!(!ps.contains(&2), "gene2 must not be a parent: {ps:?}");
        }
        // The child's 2-entry CPT is ≈ [<0.2, >0.8] after smoothing.
        let child = usize::from(edge_0_to_1);
        assert_eq!(state.cpt[child].len(), 2, "child CPT has 2 cells");
        assert!(
            state.cpt[child][0] < 0.2,
            "P(child=1 | parent=0) too high: {}",
            state.cpt[child][0]
        );
        assert!(
            state.cpt[child][1] > 0.8,
            "P(child=1 | parent=1) too low: {}",
            state.cpt[child][1]
        );
    }

    #[test]
    fn recovers_two_parent_dependency() {
        // gene2 = gene0 AND gene1 over the four balanced combos repeated 8×.
        let p = BayesianNetworkParams::default_for(3);
        let mut rows = Vec::with_capacity(32 * 3);
        for _ in 0..8 {
            for &(a, b) in &[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
                let c = if a >= 0.5 && b >= 0.5 { 1.0 } else { 0.0 };
                rows.push(a);
                rows.push(b);
                rows.push(c);
            }
        }
        let state = refit(&p, rows, 32, 3);
        // Assert specifically on gene2's parents (a 0↔1 edge, if added, is fine).
        assert_eq!(
            state.parents[2],
            vec![0, 1],
            "gene2 must depend on both 0 and 1, got {:?}",
            state.parents[2]
        );
    }

    #[test]
    fn independent_data_yields_no_edges() {
        // d=2, all four combinations equally → no detectable dependency.
        let p = BayesianNetworkParams::default_for(2);
        let rows = vec![
            0.0, 0.0, //
            0.0, 1.0, //
            1.0, 0.0, //
            1.0, 1.0, //
        ];
        let state = refit(&p, rows, 4, 2);
        for ps in &state.parents {
            assert!(ps.is_empty(), "independent data must yield no edges: {ps:?}");
        }
    }

    #[test]
    fn max_parents_cap_respected() {
        // AND dataset with max_parents = 1: must complete and respect the cap.
        let mut p = BayesianNetworkParams::default_for(3);
        p.max_parents = 1;
        let mut rows = Vec::with_capacity(32 * 3);
        for _ in 0..8 {
            for &(a, b) in &[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
                let c = if a >= 0.5 && b >= 0.5 { 1.0 } else { 0.0 };
                rows.push(a);
                rows.push(b);
                rows.push(c);
            }
        }
        let state = refit(&p, rows, 32, 3);
        for ps in &state.parents {
            assert!(ps.len() <= 1, "max_parents=1 violated: {ps:?}");
        }
    }

    #[test]
    fn order_is_topological() {
        // After a structure-learning fit, order is a permutation of 0..d with
        // every parent preceding its child.
        let p = BayesianNetworkParams::default_for(3);
        let mut rows = Vec::with_capacity(32 * 3);
        for _ in 0..8 {
            for &(a, b) in &[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
                let c = if a >= 0.5 && b >= 0.5 { 1.0 } else { 0.0 };
                rows.push(a);
                rows.push(b);
                rows.push(c);
            }
        }
        let state = refit(&p, rows, 32, 3);
        // Permutation check.
        let mut seen = state.order.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2], "order must be a permutation of 0..d");
        // Parent precedes child.
        let position: Vec<usize> = {
            let mut pos = vec![0usize; state.order.len()];
            for (idx, &node) in state.order.iter().enumerate() {
                pos[node] = idx;
            }
            pos
        };
        for (child, ps) in state.parents.iter().enumerate() {
            for &parent in ps {
                assert!(
                    position[parent] < position[child],
                    "parent {parent} must precede child {child} in {:?}",
                    state.order
                );
            }
        }
    }

    #[test]
    fn sampling_respects_learned_dependency() {
        // Fit the copy dataset, sample 5000, columns 0 and 1 agree on > 90%.
        let p = BayesianNetworkParams::default_for(2);
        let mut rows = Vec::with_capacity(20 * 2);
        for i in 0..20 {
            let g0 = if i < 10 { 0.0 } else { 1.0 };
            rows.push(g0);
            rows.push(g0); // gene1 = copy of gene0
        }
        let state = refit(&p, rows, 20, 2);
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(123);
        let n = 5000;
        let samples = <BayesianNetwork as ProbabilityModel<TestBackend>>::sample(
            &BayesianNetwork,
            &state,
            n,
            &mut rng,
            &device,
        );
        let data = samples.into_data().into_vec::<f32>().expect("samples host-read of a tensor this test just built");
        let mut agree = 0usize;
        for i in 0..n {
            if (data[i * 2] - data[i * 2 + 1]).abs() < 0.5 {
                agree += 1;
            }
        }
        // Tiny counts vs f64 exact range; lossless.
        #[allow(clippy::cast_precision_loss)]
        let frac = agree as f64 / n as f64;
        assert!(
            frac > 0.9,
            "sampled columns 0 and 1 should agree on > 90% of rows, got {frac}"
        );
    }

    #[test]
    fn nan_init_prob_clamped_on_prior() {
        // A non-finite init_prob must not propagate into the CPTs (#129): the
        // prior clamps it into the open interior (0, 1).
        let mut p = BayesianNetworkParams::default_for(3);
        p.init_prob = f32::NAN;
        let state = fit_prior(&p);
        for table in &state.cpt {
            let v = table[0];
            assert!(v.is_finite(), "clamped init_prob must be finite, got {v}");
            assert!(v > 0.0 && v < 1.0, "clamped init_prob must be interior, got {v}");
        }
    }

    #[test]
    fn out_of_range_init_prob_clamped_on_prior() {
        for bad in [1.5_f32, -0.3, f32::INFINITY] {
            let mut p = BayesianNetworkParams::default_for(2);
            p.init_prob = bad;
            let state = fit_prior(&p);
            for table in &state.cpt {
                let v = table[0];
                assert!(v > 0.0 && v < 1.0, "init_prob {bad} must clamp interior, got {v}");
            }
        }
    }

    #[test]
    fn smoothing_count_zero_keeps_cpt_interior() {
        // s = 0 with a constant-1 column would give count_1/count_total = 1.0
        // (an absorbing gene) without the floor. Flooring s at 1 keeps it in
        // (0, 1). Single gene, all ones ⇒ one CPT cell.
        let mut p = BayesianNetworkParams::default_for(1);
        p.smoothing_count = 0;
        let state = refit(&p, vec![1.0, 1.0, 1.0, 1.0], 4, 1);
        let v = state.cpt[0][0];
        assert!(v > 0.0 && v < 1.0, "s=0 must be floored to keep CPT interior, got {v}");
    }
}
