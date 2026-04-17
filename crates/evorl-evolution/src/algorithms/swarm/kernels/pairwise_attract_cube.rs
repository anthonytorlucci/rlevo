//! Fused pairwise-attract kernel for [`super::super::firefly`] (§9.1).
//!
//! # Status: designed, not yet implemented
//!
//! The v1 milestone ships the **pure-tensor** `O(N²D)` Firefly path
//! (see [`super::super::firefly::FireflyAlgorithm::pure_tensor_attract`])
//! which works on every backend Burn supports and is capped at
//! `pop_size ≤ 128` because the `(N, N, D)` difference tensor dominates
//! device memory beyond that. This module reserves the CubeCL-native
//! replacement path behind the `custom-kernels` feature.
//!
//! Landing the real kernel requires first-time integration with
//! `cubecl 0.9` in `evorl-evolution` (phase-2 adds the optional
//! dependency; phase-1 designs documented in `crate::ops::kernels` are
//! still unwritten) and per-backend validation on wgpu and the cpu-jit
//! path. None of that blocks the strategy machinery, so the kernel
//! itself is scheduled as follow-up work tracked by the plan's task
//! C1 in `.claude/plans/in-this-session-let-s-immutable-lamport.md`.
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
//! tensor — 10 MB at `N = 512, D = 10`. The fused kernel eliminates
//! that allocation and collapses three launches (diff, β, attraction
//! sum) into one. Spec §13.6 targets a measurable speedup at
//! `N ≥ 128` on both backends.
//!
//! # Fallback
//!
//! When the `custom-kernels` feature is *on* but this kernel is
//! unimplemented, the strategy continues to use the pure-tensor
//! fallback exported from [`super::super::firefly`] — an honest
//! "kernel feature-flag is declared, implementation is future work"
//! state that mirrors the phase-1 `ops/kernels` treatment.

#![allow(dead_code)]
