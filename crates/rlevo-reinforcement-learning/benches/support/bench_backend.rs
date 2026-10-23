//! Device selection shared by benches that sweep the same workload across
//! more than one Burn backend.
//!
//! Every `[[bench]]` is its own compilation unit, so this module is pulled in
//! per-file via `#[path = "support/bench_backend.rs"] mod bench_backend;`; it
//! sits in a subdirectory so Cargo's bench autodiscovery does not treat it as
//! a standalone target (mirrors `crates/rlevo/benches/support/`).
//!
//! [`BenchDevice`] exists so a bench body is written once, taking the device
//! to run on as a value, and called per device at the call site — adding a
//! backend later (CUDA, `ROCm`, ...) is one new variant here, not a rewrite of
//! every bench that sweeps backends. See `staging_bench.rs` for the reference
//! usage.

#![allow(dead_code)]

use burn::tensor::Device;

/// A device benches can select.
///
/// This is deliberately a closed list — only backends a bench author has
/// actually validated on the target hardware (see [`device`](Self::device)'s
/// doc for the wgpu-adapter smoke test) get a variant here, so a typo'd or
/// unavailable backend fails at the call site, not at bench-report time.
/// Report text must pair a backend with the concrete hardware/adapter it ran
/// on (a `wgpu` number on this machine is Metal on an Apple M2 Pro; it says
/// nothing about CUDA).
#[derive(Clone, Copy, Debug)]
pub enum BenchDevice {
    /// The `Flex` CPU device.
    Flex,
    /// The default wgpu adapter.
    Wgpu,
}

impl BenchDevice {
    /// Constructs the device this variant benches against.
    ///
    /// `Flex` has exactly one (CPU) device, and wgpu's default device selects
    /// the highest-priority GPU adapter present. On this machine (Apple M2
    /// Pro) that resolves to Metal — confirmed independently by forcing
    /// `cubecl::wgpu::init_setup::<Metal>` in the step-0 smoke test, which
    /// reported `backend: Metal` in the resulting `WgpuSetup`.
    pub fn device(self) -> Device {
        match self {
            Self::Flex => Device::flex(),
            Self::Wgpu => Device::wgpu(Default::default()),
        }
    }
}

// Adding CUDA later is exactly this shape, gated behind the `cuda` feature:
//
// #[cfg(feature = "cuda")]
// Cuda,
//
// with its `device()` arm. No existing bench body changes — callers add one
// more call (`bench_stage(.., BenchDevice::Cuda)`) alongside the Flex/Wgpu
// ones.
