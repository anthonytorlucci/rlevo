# Fitness Landscapes

<!-- see also https://en.wikipedia.org/wiki/Fitness_landscape -->

A **fitness landscape** is the mental model that makes population-based search
intuitive: picture every candidate solution as a point on a map, with elevation
standing in for fitness. Optimisation becomes navigation — find the highest peak
(or, under `rlevo`'s minimisation convention, the lowest valley) without being
able to see the whole terrain at once.

**Benchmark functions** are standardised landscapes with known optima, designed
to stress one navigational difficulty at a time. They are the test tracks we run
algorithms on before trusting them with a real problem.

## The map: a function from solutions to scores

Formally a landscape is just a function $f(x)$:

- the **domain** $x$ is the search space — every weight vector, every gene
  string, every configuration you could propose;
- the **codomain** $f(x)$ is a scalar score.

For a two-variable problem this is a 3D surface where height is fitness. The
abstraction scales to any dimension; we just lose the ability to draw it.

## The terrain: what makes a landscape hard

Three properties decide how much trouble an algorithm is in:

- **Modality.** A **unimodal** landscape has a single optimum; a **multimodal**
  one has many. Rastrigin is the textbook multimodal case — a regular lattice of
  local minima laid over a gentle bowl — and it punishes any method that commits
  to the first basin it finds.
- **Ruggedness.** How sharply fitness changes over small distances. Rugged
  landscapes (high noise, strong variable interactions) defeat methods that
  assume a smooth gradient.
- **Reachability.** Whether an algorithm can actually move between basins.
  Ridges, plateaus, and narrow valleys can leave a searcher stranded with
  nowhere uphill to go.

## Epistasis: how much the variables interact

**Epistasis** measures how strongly genes depend on one another.

On a **separable** landscape, the contribution of variable $x_1$ doesn't depend
on $x_2$, so you can optimise one axis at a time — easy. On a **non-separable**
landscape the variables are entangled: $x_1$'s ideal value shifts depending on
$x_2$, carving long curved valleys that axis-aligned moves crawl along. The
Rosenbrock function is the canonical example, and it's why algorithms that adapt
a full covariance (CMA-ES) tend to beat ones that mutate each gene
independently.

## Why benchmark functions exist

Each benchmark isolates one trait so you can diagnose an algorithm's strengths
and weaknesses in isolation:

| Benchmark | Landscape character | What it probes |
| :--- | :--- | :--- |
| **Sphere** | Smooth, unimodal, convex | Raw convergence speed |
| **Rosenbrock** | Long, narrow, curved valley | Following a low-gradient, non-separable ridge |
| **Rastrigin** | Regular multimodal lattice | Escaping local optima / global exploration |
| **Ackley** | Near-flat plateau, one steep central basin | Finding a needle in a haystack |
| **Griewank** | Multimodal with a product coupling term | Handling variable interactions at scale |

## Available in `rlevo`

`rlevo::envs::landscapes` ships these benchmarks (all under the minimisation
convention, most known-optimum-at-or-near zero):

- **n-dimensional:** Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel,
  Michalewicz, Alpine1, Penalized1, Lunacek bi-Rastrigin, Deb1, Needle-in-a-
  haystack, Eggholder, a flat-region Rosenbrock variant, and the deceptive
  Concatenated Trap (binary).
- **2-D only:** Branin, Himmelblau, Six-Hump Camel, Easom, Goldstein–Price,
  Cross-in-Tray, Bukin N.6, and Trefethen.

The continuous ones implement the `Landscape` trait from
[Optimising a Function](../part-2-guided-tour/01-optimizing-a-function.md), so
you can drop any of them into the harness in place of `Sphere`.

## The hiker analogy

Think of the landscape as a mountain range and the algorithm as a hiker in fog,
able to see only the ground at their feet. On one smooth hill (unimodal), every
upward step eventually reaches the top. In a maze of thousands of small peaks
(multimodal, rugged), the hiker can summit a minor hill, declare victory, and
never learn that a far taller mountain sat behind the next ridge. Benchmark
functions are the standardised maps we hand different hikers to see whose
navigation holds up.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
