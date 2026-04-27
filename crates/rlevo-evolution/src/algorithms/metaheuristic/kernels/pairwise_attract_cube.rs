//! Fused pairwise-attract kernel for [`super::super::firefly`].
//!
//! # Status: designed, not yet implemented
//!
//! The current release ships the **pure-tensor** `O(N²D)` Firefly path
//! (the `pure_tensor_attract` helper inside
//! [`super::super::firefly::FireflyAlgorithm`]) which works on every
//! backend Burn supports and is capped at
//! `pop_size ≤ 128` because the `(N, N, D)` difference tensor dominates
//! device memory beyond that. This module reserves the CubeCL-native
//! replacement path behind the `custom-kernels` feature.
//!
//! Landing the real kernel requires first-time CubeCL integration in
//! `rlevo-evolution` and per-backend validation on wgpu and the cpu-jit
//! path. None of that blocks the strategy machinery, so the kernel is
//! scheduled as follow-up work alongside the operator-level kernels
//! sketched in [`crate::ops::kernels`].
//!
//! # Target interface
//!
//! ```ignore
//! use cubecl::prelude::*;
//!
//! #[cube(launch_unchecked)]
//! fn pairwise_attract_cube<F: Float>(
//!     positions: &Tensor<F>,       // (N, D)
//!     fitness: &Tensor<F>,         // (N,)
//!     beta0: F,
//!     gamma: F,
//!     alpha: F,
//!     rng_bits: &Tensor<u32>,      // (N, D) noise seeds
//!     out: &mut Tensor<F>,         // (N, D)
//! ) { /* … */ }
//! ```
//!
//! The launch shape is `(N, cube_workgroup_size)`; each workgroup
//! computes a single row's attraction sum by streaming over the
//! neighbour axis, keeping memory `O(ND)` instead of the pure-tensor
//! `O(N²D)`. The noise term `α · (rand − 0.5)` is derived from
//! `rng_bits` via xorshift inside the cube.
//!
//! # Expected impact
//!
//! At `N ≥ 128` on wgpu, the pure-tensor path allocates a `(N, N, D)`
//! tensor — ~10 MB at `N = 512, D = 10` in `f32`. The fused kernel
//! eliminates that allocation and collapses three launches (pairwise
//! diff, attractiveness `β`, weighted sum) into one, with a target of
//! measurable speedup at `N ≥ 128` on both wgpu and ndarray.
//!
//! # Fallback
//!
//! When the `custom-kernels` feature is *on* but this kernel is
//! unimplemented, the strategy continues to use the pure-tensor
//! fallback exported from [`super::super::firefly`] — an honest
//! "kernel feature-flag is declared, implementation is future work"
//! state that mirrors the [`crate::ops::kernels`] treatment.

#![allow(dead_code)]
