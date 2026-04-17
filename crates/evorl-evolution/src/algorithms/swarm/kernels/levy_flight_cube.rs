//! Fused Lévy-flight (Mantegna) sampling kernel for
//! [`super::super::cuckoo`] and optionally [`super::super::bat`] (§9.2).
//!
//! # Status: designed, not yet implemented
//!
//! Per the phase-2 spec (§9.2, §14.5), this kernel is **lower
//! priority** than the pairwise-attract kernel and should only be
//! wired into the strategies when profiling shows it on the hot path.
//! The pure-tensor Mantegna path currently in
//! [`super::super::cuckoo::CuckooSearch`] and
//! [`super::super::bat::BatAlgorithm`] samples host-side with
//! `rand_distr::Normal` and bulk-uploads the resulting tensor — a
//! single host↔device transfer per generation, which profiles as a
//! non-issue at the population sizes the spec's acceptance tests
//! exercise.
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
//! # Why we are not shipping the kernel in v1
//!
//! The fractional-power step `|v|^(1/β)` is FMA-reorder-sensitive
//! enough that wgpu reductions drift ~`1e-3` relative from the
//! ndarray path on identical seeds (see
//! `tests/backend_parity.rs` and spec §8.1). The kernel fusion
//! argument collapses to "save two launches per generation"; unless
//! profiling shows that those two launches dominate, the
//! host-sampled path is both faster to ship and easier to keep
//! bit-identical on ndarray. The spec explicitly accepts this
//! trade-off:
//!
//! > Recommend: accept the `1e-3` tolerance and document the cause;
//! > the host copy defeats the kernel-fusion argument of §9.2.
//!
//! If a future profiling pass reveals the Lévy-draw path is a
//! bottleneck, this module is the place to wire the kernel in.

#![allow(dead_code)]
