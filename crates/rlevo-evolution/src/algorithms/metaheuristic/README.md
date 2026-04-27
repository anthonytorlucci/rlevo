# `rlevo-evolution::algorithms::metaheuristic`

Nine swarm-intelligence and nature-inspired metaheuristics plugged into
the existing `Strategy<B>` trait, plus one stubbed permutation-domain
module reserved for future work.

## Which one should I pick?

**Start with [pso](pso.rs).** It is the best-studied, best-behaved
member of this family. If PSO isn't good enough, the next honest step
is to wait for (or contribute) CMA-ES or LSHADE — not to try every
algorithm in this directory hoping one of them wins.

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
| **FA** ([firefly.rs](firefly.rs))      | Ships    | Useful on multimodal landscapes where `O(N²)` attraction is informative. Pure-tensor path is capped at `pop_size ≤ 128`; the fused CubeCL kernel that lifts the cap is designed but not yet implemented. |
| **BA** ([bat.rs](bat.rs))              | Ships    | *Legacy comparator.* Same caveat. |
| **SSA** ([salp.rs](salp.rs))           | Ships    | *Legacy comparator.* Same caveat. |
| **ACO (permutation)** ([aco_perm.rs](aco_perm.rs)) | Stubbed | All `Strategy` methods invoke `todo!()`. The struct is constructible so downstream crates can pin the future API surface; not drivable through a harness. |

Module-level doc comments restate the caveats at the point of use so
they show up in `cargo doc`.

## Custom kernels

Two CubeCL kernels are designed under [`kernels/`](kernels/) and gated
behind the crate-level `custom-kernels` feature:

- `pairwise_attract_cube` — the planned `O(ND)` replacement for
  Firefly's `O(N²D)` attraction sum.
- `levy_flight_cube` — a fused Mantegna sampler shared by Cuckoo and
  optionally Bat. Lower priority than the pairwise-attract kernel.

Both are stubs; the strategies fall back to pure-tensor paths until the
kernels land. See the module docs in `kernels/` for the rationale.

## Why ship the "legacy comparator" algorithms at all?

Three reasons:

1. The library is a reference implementation, not an opinion piece.
2. These algorithms are widely cited in applied literature; users will
   reach for them expecting to find them.
3. They exercise the `Strategy<B>` trait across a wider design space,
   surfacing abstraction leaks before more serious algorithms (CMA-ES,
   LSHADE) land.
