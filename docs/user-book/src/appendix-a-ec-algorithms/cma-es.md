# CMA-ES and CMSA-ES

<!-- source: crates/rlevo-evolution/src/algorithms/cma_es.rs -->
<!-- source: crates/rlevo-evolution/src/algorithms/cmsa_es.rs -->

The **Covariance Matrix Adaptation** Evolution Strategy (CMA-ES) and its
self-adaptive cousin **CMSA-ES** sample each generation from a multivariate
normal \\(\mathcal{N}(m, \sigma^2 C)\\) and adapt the mean \\(m\\), the global
step size \\(\sigma\\), and the full covariance matrix \\(C\\) from the ranked
offspring. Unlike the [classical Evolution
Strategies](evolution-strategies.md), they learn the **shape** of the search
distribution — the contour ellipses of \\(C\\) rotate and stretch to match the
local landscape, which is what makes them the strongest general-purpose
black-box optimisers for low-to-medium dimensionality.

<!--Simon, 2013, p.135
... The idea of CMSA-ES is to learn the shape of the search space during 
evolution, and adapt the mutation variance.
-->

Both ship as self-contained `Strategy<B>` implementations and slot into the
`EvolutionaryHarness` unchanged.

## CMA-ES vs CMSA-ES at a glance

| Feature | CMA-ES | CMSA-ES |
|---|---|---|
| Step-size control | Cumulative Step-size Adaptation (CSA) via the conjugate path \\(p_\sigma\\) | Per-individual log-normal self-adaptation |
| Evolution paths | Two (\\(p_\sigma\\) for σ, \\(p_c\\) for \\(C\\)) | None |
| Covariance update | Rank-1 (path) + rank-μ (population) | Rank-μ maximum-likelihood blend |
| Linear algebra per generation | Symmetric eigendecomposition (needs \\(C^{-1/2}\\)) | Cholesky factor only |
| Tuning | Fully derandomised; few knobs | Simpler; one time constant \\(\tau_c\\) |
| Origin | Hansen & Ostermeier (2001) | Beyer & Sendhoff (2008) |

CMSA-ES trades CSA's faster step-size control for a markedly simpler update —
no paths, no \\(C^{-1/2}\\) — at a small cost in convergence speed.

## Default parameters

For dimensionality \\(D\\), `default_for(D)` follows Hansen (2016):

```math
\lambda = 4 + \lfloor 3 \ln D \rfloor, \qquad \mu = \left\lfloor \tfrac{\lambda}{2} \right\rfloor
```

with positive recombination weights, normalised to sum to one:

```math
w_i = \frac{w_i'}{\sum_{j=1}^{\mu} w_j'}, \qquad w_i' = \ln\!\left(\mu + \tfrac{1}{2}\right) - \ln i, \qquad \mu_\text{eff} = \frac{1}{\sum_i w_i^2}
```

CMA-ES derives its learning rates (\\(c_\sigma, d_\sigma, c_c, c_1, c_\mu\\)) and
the expected step length \\(\chi_n \approx \sqrt{D}\,(1 - \tfrac{1}{4D} + \tfrac{1}{21D^2})\\)
from \\((\lambda, D, \mu_\text{eff})\\). CMSA-ES needs only two:

```math
\tau = \frac{1}{\sqrt{2D}}, \qquad \tau_c = 1 + \frac{D(D+1)}{2\mu}
```

> **A note on \\(\tau\\).** CMSA-ES uses the canonical \\(1/\sqrt{2D}\\) (Beyer &
> Sendhoff, 2008). This differs from the classical ES learning rate
> \\(1/\sqrt{2\sqrt{D}}\\): the two strategies share the log-normal
> σ-self-adaptation *mechanism*, but each keeps its own algorithm-faithful
> constant.

## CMA-ES update equations

Each generation samples \\(\lambda\\) offspring, ranks them by fitness
(ascending — lower is better), and applies the following updates. Let
\\(y_{(i)} = (x_{(i)} - m)/\sigma\\) be the step of the \\(i\\)-th best offspring
and \\(y_w = \sum_{i=1}^{\mu} w_i\, y_{(i)}\\) the weighted recombination.

**Mean** (weighted recombination, \\(c_m = 1\\)):

```math
m \leftarrow m + \sigma\, y_w
```

**Conjugate path and step size** (CSA):

```math
p_\sigma \leftarrow (1 - c_\sigma)\, p_\sigma + \sqrt{c_\sigma(2 - c_\sigma)\,\mu_\text{eff}}\; C^{-1/2} y_w
```

```math
\sigma \leftarrow \sigma \cdot \exp\!\left( \frac{c_\sigma}{d_\sigma} \left( \frac{\lVert p_\sigma \rVert}{\chi_n} - 1 \right) \right)
```

When recent steps line up (\\(\lVert p_\sigma \rVert > \chi_n\\)), σ grows; when
they cancel, σ shrinks.

**Anisotropic path and covariance** (rank-1 + rank-μ):

```math
p_c \leftarrow (1 - c_c)\, p_c + h_\sigma \sqrt{c_c(2 - c_c)\,\mu_\text{eff}}\; y_w
```

```math
C \leftarrow (1 - c_1 - c_\mu)\, C + c_1 \underbrace{\left(p_c p_c^\top + \delta(h_\sigma)\, C\right)}_{\text{rank-1}} + c_\mu \underbrace{\sum_{i=1}^{\mu} w_i\, y_{(i)} y_{(i)}^\top}_{\text{rank-}\mu}
```

The Heaviside factor \\(h_\sigma\\) stalls the rank-1 update when the conjugate
path is over-long (a sign of overshoot), and \\(\delta(h_\sigma) = (1 - h_\sigma)\, c_c (2 - c_c)\\)
keeps \\(\mathbb{E}[C]\\) unbiased when it does. The conjugate path needs
\\(C^{-1/2}\\), obtained from a symmetric eigendecomposition
\\(C = B\,\mathrm{diag}(\Lambda)\,B^\top\\): \\(C^{-1/2} = B\,\mathrm{diag}(\Lambda^{-1/2})\,B^\top\\).
Offspring are drawn as \\(x_i = m + \sigma\, B\,\mathrm{diag}(\sqrt{\Lambda})\, z_i\\)
with \\(z_i \sim \mathcal{N}(0, I)\\).

## CMSA-ES update equations

CMSA-ES drops the paths entirely. Each offspring gets its **own** step size
before the covariance mutation:

```math
\sigma_i = \bar{\sigma} \cdot \exp(\tau\, \mathcal{N}(0, 1)), \qquad s_i = A z_i, \qquad x_i = m + \sigma_i\, s_i
```

where \\(A\\) is the Cholesky factor of \\(C\\) (\\(A A^\top = C\\)) and
\\(z_i \sim \mathcal{N}(0, I)\\). After ranking, the \\(\mu\\) best recombine by
equal weights:

```math
m \leftarrow \frac{1}{\mu} \sum_{i=1}^{\mu} x_{(i)}, \qquad \bar{\sigma} \leftarrow \frac{1}{\mu} \sum_{i=1}^{\mu} \sigma_{(i)}
```

and the covariance relaxes toward the rank-μ maximum-likelihood estimate of the
selected mutation directions, with time constant \\(\tau_c\\):

```math
C \leftarrow \left(1 - \frac{1}{\tau_c}\right) C + \frac{1}{\tau_c} \cdot \frac{1}{\mu} \sum_{i=1}^{\mu} s_{(i)} s_{(i)}^\top
```

## Pseudocode

This follows Simon (2013, Fig. 6.17), the CMSA-ES outline. Here \\(\mu\\) is the
number of parents recombined and \\(\lambda\\) the number of offspring sampled
per generation.

---

Initialize constants \\(\tau\\) and \\(\tau_c\\)\
\\(C \leftarrow I = n \times n\\) identity matrix\
\\(\\{ (x_k, \sigma_k) \\} \leftarrow\\) randomly generated individuals, \\(k \in [1, \mu]\\)\
Each \\(x_k\\) is a candidate solution, and each \\(\sigma_k\\) is a standard deviation.\
Note that \\(x_k \in \mathbb{R}^n\\) and \\(\sigma_k \in \mathbb{R}\\).\
While not(termination criterion):\
\\(\qquad \bar{\sigma} \leftarrow \sum_{k=1}^{\mu} \sigma_k / \mu\\)\
\\(\qquad \bar{x} \leftarrow \sum_{k=1}^{\mu} x_k / \mu\\)\
\\(\qquad \text{For} k = 1, \dots, \lambda\\):\
\\(\qquad \qquad r \leftarrow \mathcal{N}(0, 1)\\) — Gaussian random scalar\
\\(\qquad \qquad \sigma_k \leftarrow \bar{\sigma}\, \exp(r\tau)\\)\
\\(\qquad \qquad R \leftarrow \mathcal{N}(0, I)\\) — \\(n\\)-dimensional Gaussian random vector\
\\(\qquad \qquad s_k \leftarrow \sqrt{C}\, R\\)\
\\(\qquad \qquad z_k \leftarrow \sigma_k s_k\\)\
\\(\qquad \qquad x_k \leftarrow \bar{x} + z_k\\)\
\\(\qquad \text{Next} k\\)\
\\(\qquad \hat{S} \leftarrow \sum_{k=1}^{\lambda} s_k s_k^\top / \lambda\\)\
\\(\qquad C \leftarrow (1 - 1/\tau_c)\, C + \hat{S}/\tau_c\\)\
Next generation

---

## Configuration

```rust,no_run
use rlevo::evo::algorithms::cma_es::CmaEsConfig;
use rlevo::evo::algorithms::cmsa_es::CmsaEsConfig;

// Hansen-2016 defaults: λ = 4 + ⌊3 ln D⌋, μ = ⌊λ/2⌋, derived learning rates.
let cma = CmaEsConfig::default_for(10);
let cmsa = CmsaEsConfig::default_for(10);

// Explicit population — a larger λ improves basin-finding on multimodal
// landscapes (e.g. Rastrigin). All weights and learning rates re-derive
// from (λ, D).
let mut cma = CmaEsConfig::with_pop_size(200, 10);
cma.initial_sigma = 2.0; // wider initial step covers more basins
```

| Field | Type | Typical | Notes |
|---|---|---|---|
| `pop_size` | `usize` | `4 + ⌊3 ln D⌋` | Offspring count λ; raise it for multimodal landscapes |
| `genome_dim` | `usize` | problem-defined | Dimensionality D |
| `bounds` | `(f32, f32)` | problem-defined | Used **only** to sample the initial mean; offspring are not clamped |
| `initial_sigma` | `f32` | 0.3–3.0 × range | Initial global step size |
| derived | — | — | `mu`, `weights`, `mu_eff`, learning rates, `chi_n` (CMA); `tau`, `tau_c` (CMSA) |

## Fitness convention

Like all strategies in `rlevo::evo`, fitness is **cost** — lower is better.
Maximization problems must be negated. The Sphere function (\\(\sum x_i^2\\),
minimum 0) needs no transformation.

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::cma_es::{CmaEs, CmaEsConfig};
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
    let config = CmaEsConfig::default_for(10);

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        CmaEs::<B>::new(),
        config,
        SphereCost,
        /* seed */ 42,
        device,
        /* max_generations */ 1000,
    );

    harness.reset();
    loop {
        let result = harness.step(());
        if result.done {
            break;
        }
    }

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best cost = {best:.4e}"); // converges below 1e-6
}
```

Swapping `CmaEs`/`CmaEsConfig` for `CmsaEs`/`CmsaEsConfig` is the only change
needed to run CMSA-ES instead.

## Implementation notes

**Host-side linear algebra.** Burn 0.21 ships no Cholesky or eigendecomposition
primitive, and `rlevo` deliberately avoids a `nalgebra` dependency. Both
routines run on host `Vec<f32>` buffers in
[`ops::linalg`](https://docs.rs/rlevo-evolution): a cyclic **Jacobi**
eigensolver (the same algorithm `pycma` uses — slower than tridiagonal QR but
more accurate on the small eigenvalues of an ill-conditioned covariance) and a
plain Cholesky with diagonal-jitter fallback. Covariance matrices are tiny
(\\(D \le 30\\)), so a device round-trip would dominate any on-device kernel; a
tensorised GPU path is a deferred optimisation.

**No internal PRNG; reproducible by seed.** Every \\(\mathcal{N}(0, C)\\) draw —
and CMSA-ES's per-individual log-normal σ mutation — goes through
`seed_stream(base, generation, SeedPurpose::CmaSampling)` and `rand_distr`,
sampled host-side and loaded with `Tensor::from_data`. The backend's
process-global RNG is never touched, so two runs with the same seed produce
bit-identical trajectories regardless of thread scheduling.

**Unbounded offspring.** CMA-ES samples in unbounded \\(\mathbb{R}^D\\); the
`bounds` field seeds only the initial mean. The offspring are **not** clamped,
which preserves the \\(y_i = (x_i - m)/\sigma\\) relationship the covariance
update depends on. On bounded analytic landscapes (Sphere, Rastrigin) the
quadratic envelope pulls the distribution back on its own.

**Not an EDA.** A full-covariance multivariate-Gaussian EDA (EMNA) is CMA-ES
minus the paths and CSA. Because the path and step-size machinery does not fit a
`fit → sample` model, these strategies are self-contained and do **not** use the
`ProbabilityModel` trait (architecture decision ADR 0021).

## When to use

| Situation | Recommendation |
|---|---|
| Continuous optimisation, low-to-medium D (≤ 30) | **CMA-ES** — the strongest general-purpose default |
| Ill-conditioned / rotated landscapes | CMA-ES — covariance adaptation learns the rotation |
| Want simpler tuning, fewer moving parts | **CMSA-ES** — no paths, only \\(\tau_c\\) to reason about |
| Multimodal landscape (e.g. Rastrigin) | Raise `pop_size` (larger λ) and widen `initial_sigma`; both help basin-finding |
| Very cheap evaluations, tiny budget | [Classical ES](evolution-strategies.md) `(1+1)` — lower per-generation overhead |
| High dimensionality (D ≫ 30) | A tensorised / separable variant (deferred); host-side covariance cost grows as \\(O(D^3)\\) |
| Discrete / binary search space | [Binary GA](binary-encoded-genetic-algorithm.md) or an [EDA](index.md#estimation-of-distribution-algorithms-edas) |

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
