//! Custom `CubeCL` kernels for swarm-algorithm hot paths.
//!
//! Two kernels live here, both gated behind the crate-level
//! `custom-kernels` feature:
//!
//! - [`pairwise_attract_cube`] — fuses Firefly's `O(N²D)` attractiveness
//!   update into a single launch that streams over neighbours to keep
//!   memory `O(ND)`.
//! - [`levy_flight_cube`] — fuses Mantegna's three-op Lévy-stable
//!   sampler into one launch. Wired into [`super::cuckoo`] and optionally
//!   [`super::bat`] only when profiling shows it on a hot path.
//!
//! When `custom-kernels` is off, the [`super::firefly`] module caps
//! population at `N ≤ 128` and emits the pure-tensor `O(N²D)`
//! decomposition; [`super::cuckoo`] and [`super::bat`] fall back to
//! pure-tensor Mantegna sampling built from `Tensor::random_normal` +
//! element-wise ops.

#[cfg(feature = "custom-kernels")]
pub mod levy_flight_cube;
#[cfg(feature = "custom-kernels")]
pub mod pairwise_attract_cube;
