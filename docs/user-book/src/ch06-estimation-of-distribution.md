# Beating a deceptive problem: EDAs

> **Status:** stub — prose and tested include coming in a follow-up PR.

**The problem.** Some landscapes actively mislead a hill-climber: every local
signal points *away* from the global optimum (a "trap"). A plain GA stalls.
You need a searcher that models **structure**, not just fitness.

**Learning goal.** Estimation-of-Distribution Algorithms (EDAs): strategies that
*learn a probability model* of the good solutions and sample from it, rather than
recombining parents.

## The new seams

- `ProbabilityModel<B>` trait — `fit(population, fitness, prev)` / `sample(n)`.
- `EdaStrategy<B, M>` — generic EDA wrapper over any `ProbabilityModel<B>`
  (ADR-0017).
- Concrete models:
  - `UnivariateGaussian` (UMDA) — independent Gaussians per gene.
  - `UnivariateBernoulli` (PBIL) — incremental Bernoulli update.
  - `CompactGenetic` (cGA) — probability-vector GA for binary problems.
  - `DependencyChain` (MIMIC) — first-order chain of pairwise dependencies.
  - `BayesianNetwork` (BOA) — DAG structure learned by greedy BIC scoring
    (ADR-0018).
- `ConcatenatedTrap` — the deceptive landscape that exercises all of the above.

## The headline result

On `ConcatenatedTrap` at pop 2000 / ratio 0.3 / 60 gens:

- **BOA reaches cost 0** — it learned the block structure.
- **UMDA and MIMIC stall around cost 3** — they missed the inter-block dependencies.
- **cGA partially escapes** — an interesting surprise noted in ADR-0018.

Population is the load-bearing knob: BIC gain scales with N, penalty scales with
ln N. Drop to pop 200 and watch BOA lose its edge.

## Outline

1. The `ConcatenatedTrap` landscape — why block structure defeats independent models.
2. UMDA and PBIL — the probability-vector mechanic on `OneMax`.
3. MIMIC on Rosenbrock — first-order chain discovers the ridge coupling.
4. BOA on `ConcatenatedTrap` — BIC-scored DAG, ancestral sampling.
5. Comparing convergence curves: UMDA vs MIMIC vs BOA on the trap.
6. Make it yours — drop population to 200 and watch BOA lose its edge.

## Example

```bash
cargo run -p rlevo-examples --example ch06_eda_deceptive
```

<!-- TODO: {{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch06_eda_deceptive.rs}} -->
