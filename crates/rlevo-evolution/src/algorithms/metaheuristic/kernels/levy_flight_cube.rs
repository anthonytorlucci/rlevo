//! Fused Lévy-flight (Mantegna) sampling kernel for
//! [`super::super::cuckoo`] and optionally [`super::super::bat`].
//!
//! # Status: designed, not yet implemented
//!
//! This kernel is **lower priority** than the pairwise-attract kernel
//! and should only be wired into the strategies when profiling shows it
//! on a hot path. The pure-tensor Mantegna path currently in
//! [`super::super::cuckoo::CuckooSearch`] and
//! [`super::super::bat::BatAlgorithm`] samples `u, v ∼ N(0, σ²)`
//! host-side with `rand_distr::Normal` and bulk-uploads the resulting
//! tensor — one host↔device transfer per generation, which profiles
//! as a non-issue at the population sizes the integration tests exercise.
//!
//! # Target interface
//!
//! ```ignore
//! use cubecl::prelude::*;
//!
//! #[cube(launch_unchecked)]
//! fn levy_flight_cube<F: Float>(
//!     beta: F,
//!     sigma_u: F,                   // precomputed Mantegna σ_u
//!     rng_u: &Tensor<u32>,          // (N, D) seeds for u ∼ N(0, σ_u²)
//!     rng_v: &Tensor<u32>,          // (N, D) seeds for v ∼ N(0, 1)
//!     out: &mut Tensor<F>,          // (N, D)  step = u / |v|^(1/β)
//! ) { /* … */ }
//! ```
//!
//! # Why this kernel isn't in the current release
//!
//! The fractional-power step `|v|^(1/β)` is FMA-reorder-sensitive
//! enough that wgpu reductions drift ~`1e-3` relative from the
//! ndarray path on identical seeds (`tests/backend_parity.rs` pins
//! the cross-backend tolerance). The fusion win here is only "save
//! two launches per generation"; unless profiling shows those launches
//! dominate, the host-sampled path is both faster to ship and easier
//! to keep numerically aligned across backends — the host copy already
//! defeats most of the fusion argument.
//!
//! If a future profiling pass reveals the Lévy-draw path is a
//! bottleneck, this module is the place to wire the kernel in.

#![allow(dead_code)]
