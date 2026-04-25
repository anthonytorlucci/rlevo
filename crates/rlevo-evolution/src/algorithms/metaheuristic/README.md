# `evorl-evolution::algorithms::swarm`

Nine swarm-intelligence and nature-inspired metaheuristics plugged into
the existing `Strategy<B>` trait.

## Which one should I pick?

**Start with [`pso`].** It is the best-studied, best-behaved member of
this family. If PSO isn't good enough, the next honest step is to wait
for (or contribute) CMA-ES or LSHADE — not to try every algorithm in
this directory hoping one of them wins.

## Calibration

The metaheuristics literature has matured enough to say this plainly.
Independent benchmarking (IEEE CEC competitions, Piotrowski et al.
2014, Camacho Villalón et al. 2020) shows:

| Algorithm  | Status | Recommendation |
|------------|--------|---------------|
| **PSO**    | Ships  | Solid baseline; well-studied; good comparator |
| **ACO_R**  | Ships  | Niche but principled — useful where the Gaussian-kernel-mixture structure fits |
| **ABC**    | Ships  | Competitive on simple multimodal; weaker in high dimensions |
| **GWO**    | Ships  | *Legacy comparator.* No novel mechanism over weighted PSO; prefer CMA-ES / LSHADE |
| **WOA**    | Ships  | *Legacy comparator.* Same caveat |
| **CS**     | Ships  | Lévy flights are the interesting part; otherwise a thin wrapper around random walk + abandonment |
| **FA**     | Ships  | Useful on multimodal landscapes where `O(N²)` attraction is informative |
| **BA**     | Ships  | *Legacy comparator.* Same caveat |
| **SSA**    | Ships  | *Legacy comparator.* Same caveat |
| **ACO (permutation)** | Stubbed | `todo!()` — deferred to a future release |

Module-level doc comments restate the caveats at the point of use so
they show up in `cargo doc`.

## Why ship the "legacy comparator" algorithms at all?

Three reasons:

1. The library is a reference implementation, not an opinion piece.
2. These algorithms are widely cited in applied literature; users will
   reach for them expecting to find them.
3. They exercise the `Strategy<B>` trait across a wider design space,
   surfacing abstraction leaks before more serious algorithms (CMA-ES,
   LSHADE) land.
