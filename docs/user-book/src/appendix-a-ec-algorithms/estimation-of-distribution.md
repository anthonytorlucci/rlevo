# Estimation-of-Distribution Algorithms

<!-- source: crates/rlevo-evolution/src/algorithms/eda/ -->
<!-- source: crates/rlevo-evolution/src/probability_model.rs -->

An **estimation-of-distribution algorithm** (EDA) replaces the crossover and
mutation operators of a classical genetic algorithm with an *explicit
probabilistic model* of the promising region of search space. Where a GA
recombines parents directly, an EDA fits a model to the selected parents and
then draws the next generation from that model. The model — not a pair of
operators — is what carries information forward.

`rlevo` factors this into two pieces. A single generic driver,
`EdaStrategy`, runs the generation loop; the model is supplied as a
`ProbabilityModel`. Five reference models ship under
`rlevo::evo::algorithms::eda`, spanning continuous and binary spaces and
ranging from fully independent marginals to a learned Bayesian network. Only
the model changes — the driver is model-agnostic.

## The shared `fit → sample` driver

Every EDA in `rlevo` is `EdaStrategy<B, M>` parameterised by a model
`M: ProbabilityModel<B>`. The strategy is stateless: the model is held by
value and all per-generation state lives in the returned `EdaState`, so it
slots straight into `EvolutionaryHarness`. Each generation runs four steps:

1. **Evaluate** the current population (the harness does this externally).
2. **Truncation-select** the best fraction of the population.
3. **`fit`** the model to those survivors.
4. **`sample`** a fresh population from the fitted model.

The driver owns steps 2–4; the model owns only `fit` and `sample`. The
[`ProbabilityModel`](https://docs.rs/rlevo-evolution) trait is the seam:

```rust,no_run
pub trait ProbabilityModel<B: Backend>: Send + Sync {
    type Params: Clone + Debug + Send + Sync;
    type State:  Clone + Debug + Send + Sync;

    fn fit(
        &self,
        params: &Self::Params,
        prev: Option<&Self::State>,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        device: &<B as BackendTypes>::Device,
    ) -> Self::State;

    fn sample(
        &self,
        state: &Self::State,
        n: usize,
        rng: &mut dyn Rng,
        device: &<B as BackendTypes>::Device,
    ) -> Tensor<B, 2>;
}
```

`Params` is static per-run configuration (priors, learning rates); `State` is
the evolving fitted statistics. Three invariants from the trait shape every
model and every page below:

- **Prior path.** When `prev = None`, `fit` builds its prior *purely* from
  `params`. `EdaStrategy::init` passes a `0 × 0` population and a length-`0`
  fitness tensor on this path, so a model must never read their contents when
  `prev` is `None`.
- **Host RNG only.** All randomness in `sample` comes from the supplied
  `rng`. Models never call `Tensor::random` or `B::seed` — Burn's GPU PRNG is
  seeded through process-global state, which interleaves across parallel
  strategy calls and breaks the per-stream determinism the crate guarantees.
  Models sample on the host and upload with `Tensor::from_data`
  (the [evolution host-RNG convention](../part-3-evolution/index.md)).
- **Selection order is a convenience, not a contract.** The rows handed to
  `fit` arrive in descending-fitness order (best = highest first), deterministically.
  Models that need the best or worst row (PBIL, cGA) still compute
  argmax/argmin themselves rather than hard-code an index.

### Truncation selection and the best-so-far tracker

`tell` does the bookkeeping every model would otherwise repeat. It pulls the
fitness vector to host, sanitises \\(\mathrm{NaN} \to -\infty\\), and keeps the
best \\(k\\) rows in descending-fitness order (highest first), where

```math
k = \min\!\bigl(\max(2,\ \lceil \rho \cdot \texttt{pop\_size} \rceil),\ \texttt{pop\_size}\bigr)
```

and \\(\rho = \texttt{selection\_ratio}\\). The clamp to \\([2,\ \texttt{pop\_size}]\\)
guarantees at least two parents (a single parent collapses variance estimates)
and never more than the population. `EdaStrategy::init` debug-asserts that
\\(\rho\\) lies strictly in \\((0, 1)\\) — a ratio of \\(0\\) selects no parents and
a ratio of \\(1\\) defeats truncation entirely.

The selected rows are forwarded to `ProbabilityModel::fit` together with their
fitness. All five built-in models perform an **unweighted** fit and ignore the
fitness tensor; it is part of the interface so a future rank- or
weight-sensitive model (a weighted MLE, a rank-μ update) can use it without a
trait change. The driver also tracks the best genome ever seen
(argmax, ties → lowest index) independently of the model, so `best` returns a
genome even for models that never store one.

### Sampling determinism

`ask` draws exactly one `u64` from the host RNG and uses it to seed a
per-generation `SeedPurpose::EdaSampling` stream keyed by the generation
counter, then samples `pop_size` individuals through that stream. Two runs with
the same seed and the same model produce identical trajectories. The optional
`EdaParams::bounds` clamp is applied after sampling.

## The five models

| Model | Wrapper type | Space | Dependencies captured | Classical name |
|---|---|---|---|---|
| Per-dimension Gaussian | `UnivariateGaussian` | continuous | none (independent marginals) | UMDA |
| Per-bit probability vector | `UnivariateBernoulli` | binary | none | PBIL |
| Virtual-population probability vector | `CompactGenetic` | binary | none | cGA |
| Gaussian dependency chain | `DependencyChain` | continuous | pairwise (first-order chain) | MIMIC |
| Bayesian network | `BayesianNetwork` | binary | bounded-in-degree DAG | BOA |

The three binary models emit raw \\(\{0, 1\}\\) genes, so the `bounds` clamp is a
no-op for them. The two continuous models honour it.

### UMDA — `UnivariateGaussian`

The Univariate Marginal Distribution Algorithm models each dimension as an
independent Gaussian and captures no cross-dimension structure. `fit` is an
unweighted maximum-likelihood estimate over the \\(k\\) selected rows, dividing by
\\(k\\) (not \\(k-1\\)):

```math
\mu_j = \frac{1}{k}\sum_{i=1}^{k} x_{ij},
\qquad
\sigma_j^2 = \max\!\left(\texttt{min\_variance},\ \frac{1}{k}\sum_{i=1}^{k}(x_{ij} - \mu_j)^2\right)
```

The `min_variance` floor (default \\(10^{-6}\\)) stops the variance collapsing to a
point mass on a converged or constant column, which would otherwise freeze that
dimension. `sample` draws each gene from its dimension's fitted Gaussian
\\(x_j \sim \mathcal{N}(\mu_j, \sigma_j^2)\\) on the host. This is the EDA analogue
of the [Evolution Strategies](evolution-strategies.md) Gaussian mutation, but
the mean and per-axis scale are *estimated from the population* each generation
rather than carried as self-adapted step sizes.

### PBIL — `UnivariateBernoulli`

Population-Based Incremental Learning maintains a per-bit probability vector
\\(p_j\\), initialised to \\(0.5\\). Each generation it nudges \\(p_j\\) towards the
gene values of the **best** selected individual, with an extra pull on the genes
where the best and worst individuals disagree. With learning rate \\(\alpha\\)
(`learning_rate`, default \\(0.1\\)) and negative learning rate \\(\beta\\)
(`negative_learning_rate`, default \\(0.075\\)):

```math
p_j \leftarrow p_j\,(1 - \alpha) + \alpha\, b_j,
\qquad\text{then if } b_j \neq w_j:\quad
p_j \leftarrow p_j\,(1 - \beta) + \beta\, b_j
```

where \\(b_j\\) and \\(w_j\\) are the best and worst individuals' bit \\(j\\). For
binary genes the second step — interpolating again towards the best — is
identical to moving away from the worst. `sample` draws each gene as a
Bernoulli trial: \\(1\\) with probability \\(p_j\\), else \\(0\\).

Two deviations from Baluja's original formulation, both consequences of
fitting inside `EdaStrategy`: the classic probability-mutation step is **not**
applied, and the best/worst individuals are the argmax/argmin over the
*truncation-selected subset* rather than a freshly drawn sample.

### cGA — `CompactGenetic`

The compact GA emulates a population of `virtual_pop_size` individuals
(default \\(50\\)) with the same per-bit probability vector \\(p_j\\), again starting
at \\(0.5\\). It competes a winner against a loser and shifts each disagreeing bit
by one virtual-population quantum \\(1 / N_v\\) towards the winner's value:

```math
\text{for each } j \text{ with } b_j \neq w_j:\quad
p_j \leftarrow \operatorname{clamp}_{[0,1]}\!\left(p_j \pm \frac{1}{N_v}\right)
```

with the sign chosen so the step moves towards the winner's bit. Larger \\(N_v\\)
takes smaller steps, slowing convergence and preserving diversity. `sample` is
Bernoulli per bit, as in PBIL.

The textbook cGA (Harik, Lobo & Goldberg, 1999) draws two individuals
uniformly from the virtual population and competes them. Here the winner is the
**argmax** and the loser the **argmin** of the truncation-selected subset handed to
`fit`, so the update is biased by the selection pressure `EdaStrategy` has
already applied — a deliberate consequence of the shared driver.

### MIMIC — `DependencyChain`

`DependencyChain` is a continuous-Gaussian extension of MIMIC. Unlike UMDA it
captures *pairwise* dependencies, representing the joint as a first-order chain
\\(c_0 \to c_1 \to \dots \to c_{D-1}\\) in which each dimension is conditionally
Gaussian given its predecessor. `fit` estimates per-dimension MLE means and
floored standard deviations as in UMDA, then forms the full Pearson correlation
matrix and converts each entry to a mutual information

```math
\mathrm{MI}_{ab} = -\tfrac{1}{2}\ln\!\left(1 - r_{ab}^2\right).
```

The chain is built greedily: the root \\(c_0\\) is the dimension of smallest
floored \\(\sigma\\) (lowest marginal entropy), and each subsequent link appends
the unvisited dimension of maximal mutual information with the last one.

A sample Pearson correlation over \\(k\\) rows has standard error \\(\approx 1/\sqrt{k}\\)
under independence, so spurious correlations would inject noise into every
conditional mean. To suppress this, any \\(|r| < 2/\sqrt{k}\\) is zeroed before the
chain is built — that link degenerates to independent marginal sampling exactly
where no dependency is statistically detectable. Surviving correlations are
clamped to \\([-0.9999,\ 0.9999]\\) to keep conditional variances positive.

`sample` is ancestral along the chain: the root is drawn from its marginal
\\(\mathcal{N}(\mu_{c_0}, \sigma_{c_0}^2)\\), and each subsequent gene from the
conditional Gaussian given its parent's already-sampled value:

```math
\mu_{\text{cond}} = \mu_c + r\,\frac{\sigma_c}{\sigma_p}\,(x_p - \mu_p),
\qquad
\sigma_{\text{cond}} = \sigma_c\sqrt{1 - r^2}.
```

`fit` is \\(O(k\,D^2)\\) (it forms the full \\(D \times D\\) correlation matrix);
`sample` is \\(O(D)\\) per individual.

### BOA — `BayesianNetwork`

The Bayesian Optimization Algorithm learns a bounded-in-degree DAG over the
binary genes and a conditional probability table (CPT) per node. Structure
learning is greedy edge addition scored by the Bayesian Information Criterion.
For a node \\(v\\) with parent set \\(\mathrm{Pa}\\) of size \\(q\\), let \\(N(c, x)\\)
count the selected rows whose parents take configuration \\(c\\) and whose gene
\\(v\\) takes bit \\(x\\), and \\(N(c) = N(c,0) + N(c,1)\\):

```math
\text{score}(v, \mathrm{Pa})
= \sum_{c} \sum_{x \in \{0,1\}} N(c, x)\,\ln\frac{N(c, x)}{N(c)}
\;-\; \frac{\ln n}{2}\,2^{\,q}.
```

The first term is the maximum-likelihood fit; the \\(\tfrac{1}{2}\ln(n)\,2^q\\)
term is the BIC penalty and the sole overfitting guard, growing exponentially
in the number of parents. The greedy loop repeatedly adds the single
highest-gain edge that keeps the graph acyclic and respects `max_parents`
(\\(\kappa\\), default \\(3\\)), stopping when no edge yields a strictly positive
gain. CPTs are then estimated with Laplace smoothing of pseudo-count
\\(s\\) (`smoothing_count`, default \\(1\\)):

```math
P(v = 1 \mid c) = \frac{N(c, 1) + s}{N(c) + 2s}.
```

For \\(s \ge 1\\) every probability lies strictly in \\((0, 1)\\), so no
configuration is ever impossible to sample. `sample` walks the nodes in
topological order, reading each gene's parents from the already-sampled genes
to index its CPT, and drawing it as a Bernoulli trial. The fit is
non-incremental: each generation relearns the structure and CPTs from scratch
(the `prev` argument serves only to distinguish the prior path), matching
canonical BOA.

## Configuration

`EdaParams<MP>` carries the driver-level settings plus the model-specific
parameters `MP`:

| Field | Type | Notes |
|---|---|---|
| `pop_size` | `usize` | Individuals sampled per generation |
| `selection_ratio` | `f32` | Truncation fraction; strictly in \\((0, 1)\\). Effective \\(k = \lceil \rho \cdot \texttt{pop\_size}\rceil\\) clamped to \\([2,\ \texttt{pop\_size}]\\) |
| `bounds` | `Option<(f32, f32)>` | Inclusive clamp applied after sampling; a no-op for the binary models |
| `model` | `MP` | Model parameters (includes `genome_dim`) |

Each model supplies a `default_for(genome_dim)` constructor:

| Model params | Key fields (defaults) |
|---|---|
| `UnivariateGaussianParams` | `init_mean` (0.0), `init_std` (2.0), `min_variance` (1e-6) |
| `UnivariateBernoulliParams` | `learning_rate` (0.1), `negative_learning_rate` (0.075) |
| `CompactGeneticParams` | `virtual_pop_size` (50) |
| `DependencyChainParams` | `init_mean` (0.0), `init_std` (2.0), `min_variance` (1e-6) |
| `BayesianNetworkParams` | `max_parents` (3), `init_prob` (0.5), `smoothing_count` (1) |

## Fitness convention

All strategies in `rlevo::evo` maximise a **canonical** fitness — higher is better. You declare a cost objective's direction with [`ObjectiveSense::Minimize`](https://docs.rs/rlevo-core) and the harness reconciles it at one chokepoint, so you never hand-negate. Truncation selection keeps the \\(k\\) highest-fitness rows in descending-fitness order (best = highest first); the best-so-far tracker is an argmax.

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::eda::{
    EdaParams, EdaStrategy, UnivariateGaussian, UnivariateGaussianParams,
};
use rlevo::evo::fitness::BatchFitnessFn;
use rlevo::evo::strategy::EvolutionaryHarness;

type B = Flex;

/// Sphere function: f(x) = Σ xᵢ², minimum 0 at the origin.
struct SphereCost;

impl BatchFitnessFn<B, Tensor<B, 2>> for SphereCost {
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let [pop_size, _] = population.dims();
        let data = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .unwrap_or_default();
        let dim = data.len() / pop_size;
        let fitness: Vec<f32> = (0..pop_size)
            .map(|row| (0..dim).map(|col| data[row * dim + col].powi(2)).sum::<f32>())
            .collect();
        Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
    }
}

fn main() {
    let device = Default::default();

    // UMDA: one independent Gaussian per dimension.
    let strategy = EdaStrategy::<B, _>::new(UnivariateGaussian);
    let params = EdaParams {
        pop_size: 64,
        selection_ratio: 0.5,
        bounds: Some((-5.12, 5.12)),
        model: UnivariateGaussianParams::default_for(10),
    };

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        strategy,
        params,
        SphereCost,
        /* seed */ 42,
        device,
        /* max_generations */ 500,
    ).expect("valid config");

    harness.reset();
    loop {
        if harness.step(()).done {
            break;
        }
    }

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best cost = {best:.4e}"); // converges near 0
}
```

Swapping the model is the only change needed to switch algorithm — for a binary
problem, construct `EdaStrategy::new(BayesianNetwork)` with
`BayesianNetworkParams::default_for(D)` and drop the `bounds` clamp.

## Implementation notes

**Full refit, not incremental.** The continuous models (`UnivariateGaussian`,
`DependencyChain`) and `BayesianNetwork` discard `prev` on the fitting path and
re-estimate from the current survivors. The two binary probability-vector
models (`UnivariateBernoulli`, `CompactGenetic`) are the exception: they read
`prev` to nudge the carried-over probability vector, so their state genuinely
accumulates across generations.

**Prior path is deterministic.** `EdaStrategy::init` calls `fit` with
`prev = None` and ignores the `rng`; every model's prior is a fixed function of
its params. The first `ask` therefore samples from a known prior before any
fitness has been seen.

**Binary `bounds` are a no-op, not an error.** Passing `bounds: Some(..)` with
a binary model is harmless — the clamp simply never changes a \\(\{0, 1\}\\) gene.
Leave it `None` for clarity.

**State is `Sync`.** `ProbabilityModel::State` requires `Sync` (a stronger
bound than the `Send`-only `Strategy::State`), deliberately, to leave room for
a future thread-shared covariance-carrying model under the same `fit`/`sample`
shape.

## When to use

| Situation | Recommendation |
|---|---|
| Continuous, separable landscape | `UnivariateGaussian` (UMDA) — cheapest model, no cross-axis cost |
| Continuous with pairwise structure | `DependencyChain` (MIMIC) — captures first-order dependencies for \\(O(k\,D^2)\\) fit |
| Binary, independent bits | `UnivariateBernoulli` (PBIL) or `CompactGenetic` (cGA) — start with PBIL |
| Binary with epistasis / building blocks | `BayesianNetwork` (BOA) — learns the dependency DAG; costliest fit |
| Ill-conditioned / rotated continuous landscape | [CMA-ES](cma-es.md) — learns a full covariance the univariate EDA cannot |
| Need crossover / schema recombination | [Real-Valued GA](real-valued-genetic-algorithm.md) or [Binary GA](binary-encoded-genetic-algorithm.md) |
| Self-adapted step sizes rather than re-estimated scale | [Evolution Strategies](evolution-strategies.md) |

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
