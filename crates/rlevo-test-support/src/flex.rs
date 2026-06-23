//! Burn `Flex` backend determinism helpers.
//!
//! Burn's `Flex` backend exposes a **process-global** RNG that governs weight
//! initialisation, and dispatches matrix operations through `rayon` whose
//! floating-point reduction order is non-deterministic under multi-threading.
//! Reproducible training therefore requires three things, every time:
//!
//! 1. **Pin `rayon` to one thread** so reduction order is fixed.
//! 2. **Serialise tests within the binary** so concurrent `seed` calls cannot
//!    interleave and corrupt the global RNG sequence.
//! 3. **Seed the backend** before constructing any network.
//!
//! [`flex_guard`] covers (1) and (2); [`seeded_device`] covers (3). Together
//! they replace the four-line preamble that every algorithm test used to
//! duplicate.

use std::sync::{Mutex, MutexGuard};

use burn::backend::{Autodiff, Flex};
use burn::tensor::backend::{Backend, BackendTypes};

/// The autodiff-wrapped `Flex` backend used by every algorithm integration
/// test. Aliased here so test files don't each redeclare `type Be = ...`.
pub type FlexAutodiff = Autodiff<Flex>;

/// Process-wide lock serialising access to Burn's global `Flex` RNG.
///
/// Each integration-test binary links its own copy of this crate, so this
/// static is per-binary — exactly the granularity needed to serialise the
/// `#[test]` functions within one file while leaving separate test binaries
/// free to run in parallel.
static BACKEND_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard returned by [`flex_guard`]. Holding it keeps the backend lock
/// claimed for the duration of a test; dropping it releases the lock.
#[derive(Debug)]
pub struct FlexGuard(#[allow(dead_code)] MutexGuard<'static, ()>);

/// Prepares the process for a deterministic `Flex` run and claims the backend
/// lock.
///
/// Pins the global `rayon` pool to a single thread (idempotent — the first
/// caller in the process wins, later calls are no-ops) and returns a
/// [`FlexGuard`] that serialises this test against others in the same binary.
/// Bind it for the whole test: `let _guard = flex_guard();`.
///
/// # Panics
///
/// Panics if the backend lock has been poisoned by a previous test panicking
/// while holding it.
#[must_use]
pub fn flex_guard() -> FlexGuard {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    FlexGuard(BACKEND_LOCK.lock().expect("backend lock poisoned"))
}

/// Returns a default device for backend `B` after seeding its global RNG.
///
/// Two calls with the same `seed` produce bit-for-bit identical subsequent
/// weight initialisation, provided the caller is holding the [`flex_guard`]
/// lock so no other test reseeds in between.
#[must_use]
pub fn seeded_device<B>(seed: u64) -> <B as BackendTypes>::Device
where
    B: Backend,
    <B as BackendTypes>::Device: Default,
{
    let device = <B as BackendTypes>::Device::default();
    <B as Backend>::seed(&device, seed);
    device
}
