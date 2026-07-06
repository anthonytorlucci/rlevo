//! Continuous-Gaussian dependency-chain model (MIMIC-style EDA) for continuous
//! search spaces.
//!
//! Unlike [`super::univariate_gaussian`], this model captures *pairwise*
//! dependencies. [`fit`] estimates per-dimension Gaussians **and** builds a
//! dimension ordering (chain `c₀ → c₁ → … → c_{D-1}`) that maximises captured
//! mutual information, then represents the joint as a first-order chain: each
//! dimension is conditionally Gaussian given its predecessor. [`sample`] walks
//! the chain, drawing each gene from the conditional Gaussian of its parent's
//! sampled value.
//!
//! The chain is built greedily à la MIMIC (De Bonet et al., 1997): the root
//! is the dimension with the smallest marginal standard deviation (lowest
//! marginal entropy), and each subsequent link is the unvisited dimension with
//! the highest mutual information to the last chosen one.
//!
//! The `fitness` tensor is accepted by the [`ProbabilityModel`] interface but
//! always ignored; the fit is unweighted.
//!
//! # Estimator regularisation
//!
//! Sample Pearson correlations from `k` selected rows have a standard error
//! of approximately `1/√k` under the null hypothesis of independence. Treating
//! those spurious correlations as real dependency injects noise into every
//! conditional mean — a penalty the univariate model never pays. To suppress
//! this effect, any Pearson `|r| < 2/√k` is zeroed before the chain is built,
//! causing the affected link to degenerate to independent marginal sampling
//! exactly where no statistically detectable dependency exists. Correlations
//! that survive this threshold are clamped to `[−0.9999, 0.9999]` to keep
//! conditional variances positive.
//!
//! # Complexity
//!
//! [`fit`] is `O(k · D²)`: it forms the full `D × D` mutual-information matrix
//! from the `k` selected rows and greedily orders the `D` dimensions.
//! [`sample`] is `O(D)` per individual: one conditional Gaussian draw per
//! chain link.
//!
//! # References
//!
//! - De Bonet, Isbell & Viola (1997), *MIMIC: Finding optima by estimating
//!   probability densities*.
//!
//! [`fit`]: crate::ProbabilityModel::fit
//! [`sample`]: crate::ProbabilityModel::sample

use burn::tensor::{Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand_distr::{Distribution as _, Normal};

use crate::probability_model::ProbabilityModel;

/// Per-run configuration for the [`DependencyChain`] model.
///
/// Held inside [`EdaParams::model`](crate::algorithms::eda::EdaParams::model)
/// for the lifetime of a run. Use [`DependencyChainParams::default_for`] for
/// typical continuous-optimisation defaults.
#[derive(Debug, Clone)]
pub struct DependencyChainParams {
    /// Number of genes per genome; determines the length of all vectors in
    /// [`DependencyChainState`] and the chain dimension `D`.
    pub genome_dim: usize,
    /// Prior mean for every dimension, used when `prev = None`.
    pub init_mean: f32,
    /// Prior standard deviation for every dimension, used when `prev = None`.
    pub init_std: f32,
    /// Minimum per-dimension variance; prevents the model from collapsing to a
    /// point mass and keeps conditional standard deviations positive.
    pub min_variance: f32,
}

impl DependencyChainParams {
    /// Sensible defaults for a `genome_dim`-dimensional continuous problem.
    #[must_use]
    pub fn default_for(genome_dim: usize) -> Self {
        Self {
            genome_dim,
            init_mean: 0.0,
            init_std: 2.0,
            min_variance: 1e-6,
        }
    }
}

/// Fitted state for the [`DependencyChain`] model after one call to
/// [`ProbabilityModel::fit`].
///
/// All four vectors have length `genome_dim` and are indexed by **dimension**
/// (not chain position), except `chain` which is indexed by chain position.
/// On the prior path (`prev = None`) the chain is the natural order
/// `[0, 1, …, D-1]`, all means are `init_mean`, all standard deviations are
/// `init_std`, and all `link_corr` entries are `0.0`.
#[derive(Debug, Clone)]
pub struct DependencyChainState {
    /// Dimension permutation: `chain[t]` is the dimension index sampled at
    /// chain position `t`. `chain[0]` is the root (marginal Gaussian).
    pub chain: Vec<usize>,
    /// Per-dimension MLE mean (indexed by dimension, not chain position).
    pub mean: Vec<f32>,
    /// Per-dimension standard deviation, floored at
    /// [`DependencyChainParams::min_variance`]`.sqrt()` (indexed by
    /// dimension).
    pub std: Vec<f32>,
    /// Pearson correlation of dimension `d` with its chain parent, after the
    /// `|r| < 2/√k` significance filter and `[-0.9999, 0.9999]` clamp.
    /// The root dimension's entry is `0.0` (unused in sampling).
    pub link_corr: Vec<f32>,
}

/// MIMIC-style dependency-chain model for continuous spaces.
///
/// Implements [`ProbabilityModel`] by fitting a greedy first-order dependency
/// chain over dimensions (see [module docs](self) for the algorithm, estimator
/// regularisation, and references). Fitness is accepted but ignored; the fit
/// is always unweighted.
///
/// [`fit`](ProbabilityModel::fit) is `O(k · D²)`; [`sample`](ProbabilityModel::sample)
/// is `O(D)` per individual.
#[derive(Debug, Clone, Copy, Default)]
pub struct DependencyChain;

impl<B: Backend> ProbabilityModel<B> for DependencyChain {
    type Params = DependencyChainParams;
    type State = DependencyChainState;

    /// Fit the dependency-chain model to the selected population.
    ///
    /// When `prev = None` returns the prior (natural-order chain, uniform
    /// `init_mean` / `init_std`, zero correlations). Otherwise:
    ///
    /// 1. Computes per-dimension MLE means and standard deviations
    ///    (`÷k` variance, floored at `min_variance`).
    /// 2. Builds the full `D × D` Pearson correlation matrix.
    /// 3. Applies the `|r| < 2/√k` significance filter (see [module docs](self)
    ///    for the estimator regularisation rationale) and clamps surviving
    ///    correlations to `[−0.9999, 0.9999]`.
    /// 4. Converts to mutual information `MI = −0.5 · ln(1 − r²)`.
    /// 5. Builds the chain greedily: root = minimum-σ dimension, each
    ///    subsequent link = unvisited dimension with the highest MI to the
    ///    last chosen one.
    ///
    /// The `fitness` argument is accepted but always ignored.
    ///
    /// # Panics
    ///
    /// Does not panic. The `unwrap()` on the last greedy-chain iteration is
    /// safe because at least one unvisited dimension remains when `d > 1`.
    // The MI matrix, greedy chain ordering, and per-link conditional
    // extraction form one coherent algorithmic unit; splitting it would
    // scatter the shared intermediate buffers without aiding readability.
    #[allow(clippy::too_many_lines)]
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
            // Prior path: independent dimensions in natural order; population
            // and fitness ignored.
            let d = params.genome_dim;
            return DependencyChainState {
                chain: (0..d).collect(),
                mean: vec![params.init_mean; d],
                std: vec![params.init_std; d],
                link_corr: vec![0.0; d],
            };
        };

        let [k, d] = population.dims();
        if k < 2 {
            // Correlation is unidentifiable from fewer than two rows: `kf` would
            // be `0`/`1`, driving `/= kf` to `NaN`/degenerate stats that then
            // poison `std`/`link_corr` and later panic in `sample`. Return the
            // prior-shaped state (independent dimensions) to keep the run alive.
            // `EdaStrategy::tell` clamps `k ≥ 2`, but `fit` is a public trait
            // method reachable directly with a `0×D`/`1×D` population.
            return DependencyChainState {
                chain: (0..d).collect(),
                mean: vec![params.init_mean; d],
                std: vec![params.init_std; d],
                link_corr: vec![0.0; d],
            };
        }
        let rows = population.into_data().into_vec::<f32>().unwrap_or_default();
        // k is a selected-population count, far below f32's 2^24 exact-integer
        // limit; the cast is lossless in practice.
        #[allow(clippy::cast_precision_loss)]
        let kf = k as f32;

        // Column means.
        let mut mean = vec![0.0_f32; d];
        for i in 0..k {
            for j in 0..d {
                mean[j] += rows[i * d + j];
            }
        }
        for m in &mut mean {
            *m /= kf;
        }

        // Raw MLE variances (unfloored) for the correlation guard, plus the
        // floored std stored in state.
        let mut raw_var = vec![0.0_f32; d];
        for i in 0..k {
            for j in 0..d {
                let diff = rows[i * d + j] - mean[j];
                raw_var[j] += diff * diff;
            }
        }
        for v in &mut raw_var {
            *v /= kf;
        }
        let std: Vec<f32> = raw_var
            .iter()
            .map(|&v| v.max(params.min_variance).sqrt())
            .collect();

        // Pairwise covariances → Pearson correlations.
        // cov[a][b] = Σ (x_a - μ_a)(x_b - μ_b) / k.
        let mut cov = vec![0.0_f32; d * d];
        for i in 0..k {
            for a in 0..d {
                let da = rows[i * d + a] - mean[a];
                for b in 0..d {
                    let db = rows[i * d + b] - mean[b];
                    cov[a * d + b] += da * db;
                }
            }
        }
        for c in &mut cov {
            *c /= kf;
        }

        // r[a][b] = cov / (raw_σ_a · raw_σ_b); guarded and clamped.
        //
        // Sample correlations from k rows are noisy with std ≈ 1/√k under
        // independence; conditioning the chain on spurious correlations
        // injects that noise into every conditional mean — a penalty a
        // univariate model never pays. Estimates below the ~2σ significance
        // threshold are therefore zeroed, so the chain degenerates to
        // independent sampling exactly where no dependency is detectable.
        let significance = 2.0 / kf.sqrt();
        let mut corr = vec![0.0_f32; d * d];
        // Mutual information MI[a][b] = -0.5 ln(1 - r²); computed explicitly
        // for fidelity though it is monotone in r².
        let mut mi = vec![0.0_f32; d * d];
        for a in 0..d {
            for b in 0..d {
                let r = if raw_var[a] < params.min_variance || raw_var[b] < params.min_variance {
                    0.0
                } else {
                    let raw = cov[a * d + b] / (raw_var[a].sqrt() * raw_var[b].sqrt());
                    if raw.abs() < significance {
                        0.0
                    } else {
                        raw.clamp(-0.9999, 0.9999)
                    }
                };
                corr[a * d + b] = r;
                mi[a * d + b] = -0.5 * (1.0 - r * r).ln();
            }
        }

        // NOTE: the sentinels in this structure-learning routine are about
        // marginal entropy (σ) and mutual information, NOT objective fitness.
        // They are independent of the crate's maximise convention — do not
        // "fix" them to match it.
        //
        // Root: smallest floored std (Gaussian entropy is monotone in σ, so the
        // lowest-σ dimension has the smallest marginal entropy); tie → lowest
        // index.
        let mut root = 0_usize;
        let mut root_std = f32::INFINITY;
        for (j, &sj) in std.iter().enumerate() {
            if sj < root_std {
                root_std = sj;
                root = j;
            }
        }

        // Greedy chain: append the unvisited dimension with maximal MI to the
        // last chosen one; tie → lowest index.
        let mut visited = vec![false; d];
        let mut chain = Vec::with_capacity(d);
        chain.push(root);
        visited[root] = true;
        for _ in 1..d {
            let last = *chain.last().unwrap();
            let mut best_j = usize::MAX;
            let mut best_mi = f32::NEG_INFINITY;
            for j in 0..d {
                if visited[j] {
                    continue;
                }
                if mi[last * d + j] > best_mi {
                    best_mi = mi[last * d + j];
                    best_j = j;
                }
            }
            chain.push(best_j);
            visited[best_j] = true;
        }

        // link_corr[chain[t]] = r[chain[t-1]][chain[t]]; root entry stays 0.
        let mut link_corr = vec![0.0_f32; d];
        for t in 1..chain.len() {
            let parent = chain[t - 1];
            let cur = chain[t];
            link_corr[cur] = corr[parent * d + cur];
        }

        DependencyChainState {
            chain,
            mean,
            std,
            link_corr,
        }
    }

    /// Draw `n` genomes by ancestral sampling along the fitted chain.
    ///
    /// The root dimension is sampled from its marginal Gaussian; each
    /// subsequent dimension is sampled from the conditional Gaussian given its
    /// parent's sampled value:
    ///
    /// ```text
    /// μ_cond = μ_c + r · (σ_c / σ_p) · (x_parent − μ_p)
    /// σ_cond = σ_c · √(1 − r²)
    /// ```
    ///
    /// All randomness is drawn from `rng` (host RNG only; never
    /// `Tensor::random` / `B::seed`). The returned tensor has shape `(n, D)`.
    /// This is `O(D)` per individual drawn.
    ///
    /// # Panics
    ///
    /// Does not panic under normal operation. The `Normal::new` calls are
    /// guarded: `σ_c` is floored at `min_variance.sqrt()` during `fit`, and
    /// `r` is clamped to `[-0.9999, 0.9999]` so `1 − r² ≥ 0.0002 > 0`.
    fn sample(
        &self,
        state: &Self::State,
        n: usize,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 2> {
        let d = state.mean.len();
        let mut rows = vec![0.0_f32; n * d];
        for i in 0..n {
            let base = i * d;
            // Root: marginal Gaussian.
            let root = state.chain[0];
            let root_normal = Normal::new(state.mean[root], state.std[root])
                .expect("floored std is positive and finite");
            rows[base + root] = root_normal.sample(rng);
            // Subsequent links: conditional Gaussian given the chain parent.
            for t in 1..state.chain.len() {
                let parent = state.chain[t - 1];
                let cur = state.chain[t];
                let r = state.link_corr[cur];
                let mu_c = state.mean[cur];
                let mu_p = state.mean[parent];
                let sigma_c = state.std[cur];
                let sigma_p = state.std[parent]; // > 0 by floor.
                let cond_mean = mu_c + r * (sigma_c / sigma_p) * (rows[base + parent] - mu_p);
                // 1 - r² ≥ 1 - 0.9999² > 0.
                let cond_std = (sigma_c * sigma_c * (1.0 - r * r)).sqrt();
                // `Normal::new` rejects a non-finite std but accepts any mean, so
                // an overflowed `cond_mean` (large parent value × near-1 `r`)
                // would silently emit `NaN` samples and poison the next
                // generation. If either parameter is non-finite, fall back to the
                // marginal Gaussian of `cur` — the distribution the link
                // degenerates to at `r = 0`.
                rows[base + cur] = if cond_mean.is_finite() && cond_std.is_finite() && cond_std > 0.0
                {
                    Normal::new(cond_mean, cond_std)
                        .expect("guarded: conditional std positive and finite")
                        .sample(rng)
                } else {
                    Normal::new(mu_c, sigma_c)
                        .expect("floored marginal std is positive and finite")
                        .sample(rng)
                };
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

    fn fit_prior(p: &DependencyChainParams) -> DependencyChainState {
        let device = Default::default();
        <DependencyChain as ProbabilityModel<TestBackend>>::fit(
            &DependencyChain,
            p,
            None,
            pop(vec![], 0, 0),
            fitness(vec![]),
            &device,
        )
    }

    fn refit(p: &DependencyChainParams, rows: Vec<f32>, n: usize, d: usize) -> DependencyChainState {
        let device = Default::default();
        let prior = fit_prior(p);
        // Test row counts are tiny; the cast is lossless.
        #[allow(clippy::cast_precision_loss)]
        let fit_values: Vec<f32> = (0..n).map(|i| i as f32).collect();
        <DependencyChain as ProbabilityModel<TestBackend>>::fit(
            &DependencyChain,
            p,
            Some(&prior),
            pop(rows, n, d),
            fitness(fit_values),
            &device,
        )
    }

    #[test]
    fn prior_is_natural_order_independent() {
        let p = DependencyChainParams::default_for(3);
        let state = fit_prior(&p);
        assert_eq!(state.chain, vec![0, 1, 2]);
        assert_eq!(state.mean, vec![0.0, 0.0, 0.0]);
        assert_eq!(state.std, vec![2.0, 2.0, 2.0]);
        assert_eq!(state.link_corr, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn chain_links_correlated_dimensions_adjacently() {
        // x0 spread; x1 = x0 + tiny noise (strongly correlated with x0);
        // x2 independent. The chain should place 0 and 1 adjacently.
        let p = DependencyChainParams::default_for(3);
        let rows = vec![
            -2.0, -2.01, 5.0, //
            -1.0, -0.99, -3.0, //
            0.0, 0.01, 1.0, //
            1.0, 1.02, -4.0, //
            2.0, 1.98, 0.5, //
        ];
        let state = refit(&p, rows, 5, 3);
        // Find positions of dims 0 and 1 in the chain; they must be adjacent.
        let pos0 = state.chain.iter().position(|&x| x == 0).unwrap();
        let pos1 = state.chain.iter().position(|&x| x == 1).unwrap();
        assert_eq!(
            pos0.abs_diff(pos1),
            1,
            "dims 0 and 1 should be adjacent in chain {:?}",
            state.chain
        );
        // Whichever of 0/1 is the child (later in the chain) carries a high
        // link correlation.
        let child = usize::from(pos0 <= pos1);
        assert!(
            state.link_corr[child].abs() > 0.99,
            "expected strong link corr, got {}",
            state.link_corr[child]
        );
    }

    #[test]
    fn zero_variance_column_yields_zero_correlation() {
        let p = DependencyChainParams::default_for(2);
        // Column 1 is constant → raw variance 0 → r guarded to 0.
        let rows = vec![0.0, 5.0, 1.0, 5.0, 2.0, 5.0, 3.0, 5.0];
        let state = refit(&p, rows, 4, 2);
        for &r in &state.link_corr {
            approx::assert_relative_eq!(r, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn perfect_correlation_is_clamped() {
        let p = DependencyChainParams::default_for(2);
        // Column 1 is an exact copy of column 0 → r would be 1, clamped to
        // 0.9999.
        let rows = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let state = refit(&p, rows, 4, 2);
        let child = *state.chain.last().unwrap();
        assert!(state.link_corr[child].abs() <= 0.9999 + 1e-6);
        assert!(state.link_corr[child].abs() > 0.99);
    }

    #[test]
    fn link_corr_matches_expected_pearson() {
        let p = DependencyChainParams::default_for(2);
        // Column 1 = 2 * column 0 → Pearson r = 1, clamped to 0.9999.
        let rows = vec![-2.0, -4.0, -1.0, -2.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0];
        let state = refit(&p, rows, 5, 2);
        let child = *state.chain.last().unwrap();
        approx::assert_relative_eq!(state.link_corr[child], 0.9999, epsilon = 1e-3);
    }

    #[test]
    fn sampling_respects_chain_correlation() {
        // Two strongly-correlated dimensions → sampled values must track.
        let p = DependencyChainParams::default_for(2);
        let rows = vec![-2.0, -2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let state = refit(&p, rows, 5, 2);
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(99);
        let n = 5_000;
        let samples = <DependencyChain as ProbabilityModel<TestBackend>>::sample(
            &DependencyChain,
            &state,
            n,
            &mut rng,
            &device,
        );
        let data = samples.into_data().into_vec::<f32>().unwrap();
        // Pearson correlation of sampled columns 0 and 1.
        let mut s0 = 0.0_f64;
        let mut s1 = 0.0_f64;
        for i in 0..n {
            s0 += f64::from(data[i * 2]);
            s1 += f64::from(data[i * 2 + 1]);
        }
        // n = 5_000 fits f64 exactly; the cast is lossless here.
        #[allow(clippy::cast_precision_loss)]
        let nf = n as f64;
        let m0 = s0 / nf;
        let m1 = s1 / nf;
        let (mut cov, mut v0, mut v1) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..n {
            let a = f64::from(data[i * 2]) - m0;
            let b = f64::from(data[i * 2 + 1]) - m1;
            cov += a * b;
            v0 += a * a;
            v1 += b * b;
        }
        let corr = cov / (v0.sqrt() * v1.sqrt());
        assert!(corr > 0.9, "sampled correlation too low: {corr}");
    }

    #[test]
    fn two_fits_same_data_identical_state() {
        let p = DependencyChainParams::default_for(3);
        let rows = vec![
            -2.0, 1.0, 0.5, //
            -1.0, 2.0, -0.5, //
            0.0, 0.0, 1.0, //
            1.0, -1.0, -1.0, //
        ];
        let a = refit(&p, rows.clone(), 4, 3);
        let b = refit(&p, rows, 4, 3);
        assert_eq!(a.chain, b.chain);
        assert_eq!(a.mean, b.mean);
        assert_eq!(a.std, b.std);
        assert_eq!(a.link_corr, b.link_corr);
    }

    #[test]
    fn fit_k_less_than_two_returns_prior() {
        // n = 1 (k = 1): correlation is unidentifiable and `/= kf` would poison
        // the state with NaN. The guard (#129) returns the prior-shaped state.
        let p = DependencyChainParams::default_for(2);
        let state = refit(&p, vec![1.0, 2.0], 1, 2);
        assert_eq!(state.chain, vec![0, 1]);
        assert_eq!(state.mean, vec![p.init_mean, p.init_mean]);
        assert_eq!(state.std, vec![p.init_std, p.init_std]);
        assert_eq!(state.link_corr, vec![0.0, 0.0]);
    }

    #[test]
    fn sample_with_degenerate_link_stays_finite() {
        // A pathological state whose conditional link overflows (σ_c/σ_p → inf):
        // the sample() guard (#129) must fall back to the marginal Gaussian
        // rather than emit NaN/inf into the population.
        let device = Default::default();
        let state = DependencyChainState {
            chain: vec![0, 1],
            mean: vec![0.0, 0.0],
            std: vec![1e-30, 1e30],
            link_corr: vec![0.0, 0.9999],
        };
        let mut rng = StdRng::seed_from_u64(7);
        let samples = <DependencyChain as ProbabilityModel<TestBackend>>::sample(
            &DependencyChain,
            &state,
            16,
            &mut rng,
            &device,
        );
        for v in samples.into_data().into_vec::<f32>().unwrap() {
            assert!(v.is_finite(), "degenerate link must yield finite samples, got {v}");
        }
    }
}
