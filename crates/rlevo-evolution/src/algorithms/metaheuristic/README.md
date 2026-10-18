# `rlevo-evolution::algorithms::metaheuristic`

Nine swarm-intelligence and nature-inspired metaheuristics plugged into
the existing `Strategy<B>` trait.

## Which one should I pick?

**Start with [pso](pso.rs).** It is the best-studied, best-behaved
member of this family. If PSO isn't good enough, the next honest step
is CMA-ES ([cma_es.rs](../cma_es.rs)), or contributing LSHADE — not to
try every algorithm in this directory hoping one of them wins.

## Calibration

The metaheuristics literature has matured enough to say this plainly.
Independent benchmarking (IEEE CEC competitions, Piotrowski et al.
2014, Camacho Villalón et al. 2020) shows:

| Algorithm                              | Status   | Recommendation |
|----------------------------------------|----------|----------------|
| **PSO** ([pso.rs](pso.rs))             | Ships    | Solid baseline; well-studied; good comparator. Inertia and constriction variants both exposed. |
| **ACO_R** ([aco_r.rs](aco_r.rs))       | Ships    | Niche but principled — useful where the Gaussian-kernel-mixture structure fits. |
| **ABC** ([abc.rs](abc.rs))             | Ships    | Competitive on simple multimodal; weaker in high dimensions. |
| **GWO** ([gwo.rs](gwo.rs))             | Ships    | *Legacy comparator.* No novel mechanism over weighted PSO; prefer CMA-ES / LSHADE. |
| **WOA** ([woa.rs](woa.rs))             | Ships    | *Legacy comparator.* Same caveat. |
| **CS** ([cuckoo.rs](cuckoo.rs))        | Ships    | Lévy flights are the interesting part; otherwise a thin wrapper around random walk + abandonment. The fractional-power step is FMA-reorder-sensitive — see the backend-parity caveat in the module doc. |
| **FA** ([firefly.rs](firefly.rs))      | Ships    | *Legacy comparator.* Camacho Villalón et al. (2020/2023) show the pairwise attraction reduces to a PSO-style attractor under a new metaphor — no novel mechanism. Occasionally informative on multimodal landscapes where the $O(N^2)$ multi-pull is. The swarm is capped at $\text{pop\_size} \leq 128$, because the attraction step allocates an $(N, N, D)$ tensor. |
| **BA** ([bat.rs](bat.rs))              | Ships    | *Legacy comparator.* Same caveat. |
| **SSA** ([salp.rs](salp.rs))           | Ships    | *Legacy comparator.* Same caveat. |

Module-level doc comments restate the caveats at the point of use so
they show up in `cargo doc`.

## Why ship the "legacy comparator" algorithms at all?

Three reasons:

1. The library is a reference implementation, not an opinion piece.
2. These algorithms are widely cited in applied literature; users will
   reach for them expecting to find them.
3. They exercise the `Strategy<B>` trait across a wider design space,
   surfacing abstraction leaks.
