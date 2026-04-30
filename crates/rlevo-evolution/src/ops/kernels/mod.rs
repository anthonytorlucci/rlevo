//! Custom `CubeCL` kernels for hot-path operators.
//!
//! This module is a **design placeholder**. The current release ships
//! only the pure-tensor operator baselines in [`crate::ops::selection`],
//! [`crate::ops::crossover`], and [`crate::ops::mutation`]. Those
//! compose from Burn tensor primitives and run on every backend Burn
//! supports (ndarray, wgpu, …) with no extra work. The `custom-kernels`
//! Cargo feature exists so downstream crates can pin the future ABI
//! when kernels land.
//!
//! # Why kernels aren't in the current release
//!
//! Three operator paths were identified where a fused `CubeCL` kernel
//! would eliminate multi-launch overhead. Landing real kernels requires
//! non-trivial `CubeCL` integration (`cubecl 0.9` ships with Burn 0.20.1)
//! and device-specific validation on wgpu. None of that work blocks the
//! core strategy machinery, so it was deferred to keep the release
//! shippable.
//!
//! The three designs below document the intended interfaces so a
//! future contributor can write them without re-deriving the
//! motivation.
//!
//! # Tournament selection
//!
//! Today the pure-tensor path
//! ([`super::selection::tournament_select`]) samples tournament
//! indices on the host, packs them into a 1-D `Int` tensor, and does a
//! single `tensor.select(0, indices)` gather. Cost at `pop_size = N`:
//! `N` host-side index draws and one device kernel launch.
//!
//! A fused kernel would take the form
//!
//! ```ignore
//! fn tournament_select_cube<F: Float, I: Int>(
//!     fitness: &Tensor<F>,          // (N,)
//!     rng_state: &Tensor<I>,         // (N, k)  pre-sampled index pairs
//!     winners: &mut Tensor<I>,       // (N,)    output
//! )
//! ```
//!
//! performing the index sampling and comparison in a single launch,
//! eliminating the host trip entirely. Expected speedup at `N ≥ 256`
//! on wgpu: order-of-magnitude.
//!
//! # DE trial-vector construction
//!
//! Classical DE computes `v_i = x_{r1} + F · (x_{r2} − x_{r3})` plus
//! a binomial-crossover mask per gene. In
//! [`crate::algorithms::de`] this is composed from three `select`s,
//! one subtract, one `mul_scalar`, one mask-build, and one
//! `mask_where` — seven kernel launches per generation.
//!
//! A fused kernel that takes the whole population plus pre-sampled
//! indices and emits the trial vector in one pass:
//!
//! ```ignore
//! fn de_trial_cube<F: Float, I: Int>(
//!     pop: &Tensor<F>,               // (N, D)
//!     indices: &Tensor<I>,           // (N, k)  sampled parent indices
//!     f: F, cr: F,                   // scalars
//!     rng_bits: &Tensor<I>,          // crossover mask seeds
//!     variant: u32,                  // const-generic DeVariant discriminant
//!     trial: &mut Tensor<F>,         // (N, D)  output
//! )
//! ```
//!
//! Expected impact: DE's inner loop is dominated by these 7 launches;
//! collapsing to 1 would likely double throughput at `pop_size ≥ 256`.
//!
//! # Fitness-proportionate (roulette) selection
//!
//! Roulette selection is a prefix-sum + inverse-CDF lookup. Burn's
//! `cumsum` + `searchsorted` would work but materializes two
//! intermediate tensors. This kernel is lower priority than the two
//! above — the pure-tensor path is fine for typical population sizes —
//! but worth writing if profiling later shows roulette on a hot path.
//!
//! # Kernel infrastructure
//!
//! When implementing these:
//!
//! 1. Add `cubecl` to `rlevo-evolution`'s dependencies (gated on the
//!    `custom-kernels` feature).
//! 2. Use `#[cube(launch_unchecked)]` with Burn's `backend::custom` API
//!    to plug into the `Backend` trait.
//! 3. Provide a pure-tensor fallback (which is the current
//!    implementation) for backends that don't support `CubeCL`.
//! 4. Expose a toggle at the operator level so benchmarks can A/B
//!    the two paths.

#![allow(dead_code)]
